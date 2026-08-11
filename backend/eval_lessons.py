#!/usr/bin/env python
"""Offline evaluation for classic vs quality-diverse lesson retrieval.

Examples:
  python eval_lessons.py \
    --lessons fixtures/lesson_eval_lessons.jsonl \
    --queries fixtures/lesson_eval_queries.jsonl

  python eval_lessons.py --lessons /tmp/skill_lessons.db --pipeline quality_diverse

The evaluator never reads or accepts an API key. It replaces embeddings with a
deterministic lexical hash vector so the comparison is reproducible and cannot
leak credentials. Production embedding/rerank calls are covered by separate API
smoke tests where the key is sent only in the request body.
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen_optimizer import SkillOptimizer


def mock_embed(text, dimensions=128):
    """Deterministic signed feature hashing over the production sparse tokens."""
    vector = [0.0] * dimensions
    for token in SkillOptimizer._lesson_tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        vector[index] += 1.0 if digest[4] % 2 else -1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def lesson_id(lesson, index=None):
    value = lesson.get("eval_id", lesson.get("_id"))
    return str(value if value is not None else index)


def build_queries(lessons):
    """Self-query fallback with exact relevance labels (smoke, not benchmark)."""
    queries = []
    for index, lesson in enumerate(lessons):
        query = SkillOptimizer._lesson_index_text(lesson)
        if not query:
            query = f"skill: {lesson.get('skill_name') or ''}".strip()
        target = lesson.get("target_dimension")
        queries.append({
            "query": query,
            "skill_name": lesson.get("skill_name"),
            "domain": lesson.get("domain"),
            "target_dimension": target,
            "relevant_ids": [lesson_id(lesson, index)],
        })
    return queries


def relevant(lesson, query, index=None):
    """Prefer explicit judgments; keep tag matching only as a fallback."""
    explicit_ids = {str(value) for value in query.get("relevant_ids", [])}
    if explicit_ids:
        return lesson_id(lesson, index) in explicit_ids
    strategies = {str(value) for value in query.get("relevant_strategies", [])}
    if strategies:
        return str(lesson.get("strategy")) in strategies
    target = query.get("target_dimension")
    if target:
        dimensions = set((lesson.get("dimension_gains") or {}).keys())
        if lesson.get("target_dimension"):
            dimensions.add(lesson["target_dimension"])
        return target in dimensions
    if query.get("skill_name") and lesson.get("skill_name") == query["skill_name"]:
        return True
    return bool(
        query.get("domain")
        and lesson.get("domain") == query["domain"]
        and lesson.get("domain")
    )


def _dcg(relevance):
    return sum(value / math.log2(rank + 2) for rank, value in enumerate(relevance))


def ranking_metrics(results, all_lessons, query, top_k):
    judgments = [1 if relevant(lesson, query) else 0 for lesson in results]
    relevant_total = sum(1 for lesson in all_lessons if relevant(lesson, query))
    hits = sum(judgments)
    reciprocal_rank = next(
        (1.0 / (rank + 1) for rank, value in enumerate(judgments) if value), 0.0
    )
    ideal = [1] * min(relevant_total, top_k)
    ndcg = _dcg(judgments) / _dcg(ideal) if ideal else 0.0
    strategies = {lesson.get("strategy") for lesson in results if lesson.get("strategy")}
    pair_scores = []
    for left in range(len(results)):
        left_tokens = SkillOptimizer._lesson_tokens(
            SkillOptimizer._lesson_index_text(results[left])
        )
        for right in range(left + 1, len(results)):
            right_tokens = SkillOptimizer._lesson_tokens(
                SkillOptimizer._lesson_index_text(results[right])
            )
            pair_scores.append(1.0 - SkillOptimizer._token_jaccard(left_tokens, right_tokens))
    return {
        "top1": float(bool(judgments and judgments[0])),
        "precision": hits / len(results) if results else 0.0,
        "recall": hits / relevant_total if relevant_total else 0.0,
        "mrr": reciprocal_rank,
        "ndcg": ndcg,
        "strategy_diversity": len(strategies) / len(results) if results else 0.0,
        "pairwise_diversity": sum(pair_scores) / len(pair_scores) if pair_scores else 1.0,
    }


async def evaluate(lessons, queries, pipeline, retrieval_mode, top_k, threshold):
    opt = SkillOptimizer(
        api_key="offline-eval-placeholder",
        lesson_retrieval=retrieval_mode,
        lesson_threshold=threshold,
        lesson_top_k=top_k,
        lesson_rag_pipeline=pipeline,
    )

    async def fake_embed(text):
        return mock_embed(text)

    opt._embed = fake_embed
    eval_lessons = [dict(lesson, embedding=None) for lesson in lessons]
    totals = {
        "top1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "mrr": 0.0,
        "ndcg": 0.0,
        "strategy_diversity": 0.0,
        "pairwise_diversity": 0.0,
    }
    latencies = []
    completed = 0
    for query in queries:
        weak_dimensions = []
        if query.get("target_dimension"):
            weak_dimensions.append({
                "dimension": query["target_dimension"],
                "pct": query.get("dimension_pct", 0.0),
            })
        context = {
            "skill_name": query.get("skill_name") or "",
            "domain": query.get("domain") or "",
            "weak_dimensions": weak_dimensions,
        }
        started = time.perf_counter()
        results = await opt._retrieve_lessons(
            eval_lessons,
            query.get("query", ""),
            retrieval_mode,
            top_k,
            context=context,
        )
        latencies.append((time.perf_counter() - started) * 1000)
        metrics = ranking_metrics(results, eval_lessons, query, top_k)
        for name in totals:
            totals[name] += metrics[name]
        completed += 1
    if not completed:
        return None
    report = {name: value / completed for name, value in totals.items()}
    report["misinjection"] = 1.0 - report["precision"]
    report["latency_ms"] = sum(latencies) / len(latencies)
    report["queries"] = completed
    return report


def print_report(name, report, top_k):
    print(f"\n[{name}]")
    print(f"  Top-1 accuracy:       {report['top1']:.3f}")
    print(f"  MRR:                  {report['mrr']:.3f}")
    print(f"  nDCG@{top_k}:             {report['ndcg']:.3f}")
    print(f"  Recall@{top_k}:           {report['recall']:.3f}")
    print(f"  Precision@{top_k}:        {report['precision']:.3f}")
    print(f"  Misinjection@{top_k}:     {report['misinjection']:.3f}")
    print(f"  Strategy diversity:   {report['strategy_diversity']:.3f}")
    print(f"  Pairwise diversity:   {report['pairwise_diversity']:.3f}")
    print(f"  Local latency/query:  {report['latency_ms']:.2f} ms")


async def main():
    parser = argparse.ArgumentParser(description="离线评估经验 RAG 排序质量")
    parser.add_argument("--lessons", required=True, help="经验库路径（jsonl 或 SQLite）")
    parser.add_argument("--queries", help="带 relevant_ids 等显式标注的查询集 jsonl")
    parser.add_argument(
        "--pipeline", choices=("classic", "quality_diverse", "compare"), default="compare"
    )
    parser.add_argument("--retrieval", choices=("semantic", "hybrid"), default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    lessons = SkillOptimizer._load_lessons(args.lessons, limit=1000)
    if not lessons:
        print(f"[eval] 经验库为空: {args.lessons}")
        return 1
    for index, lesson in enumerate(lessons):
        lesson.setdefault("eval_id", lesson_id(lesson, index))
    queries = load_jsonl(args.queries) if args.queries else build_queries(lessons)
    if not queries:
        print("[eval] 查询集为空")
        return 1

    print(f"[eval] lessons={len(lessons)} queries={len(queries)} mode={args.retrieval}")
    pipelines = (
        ("classic", "quality_diverse") if args.pipeline == "compare" else (args.pipeline,)
    )
    reports = {}
    for pipeline in pipelines:
        reports[pipeline] = await evaluate(
            lessons, queries, pipeline, args.retrieval, max(1, args.top_k), args.threshold
        )
        print_report(pipeline, reports[pipeline], max(1, args.top_k))

    if len(reports) == 2:
        classic = reports["classic"]
        enhanced = reports["quality_diverse"]
        print("\n[delta quality_diverse - classic]")
        for metric in ("mrr", "ndcg", "recall", "precision", "strategy_diversity"):
            print(f"  {metric:20s} {enhanced[metric] - classic[metric]:+.3f}")
        # The benchmark fails only on a material relevance regression. Diversity
        # is reported separately because some query sets have only one valid fix.
        ok = enhanced["mrr"] + 1e-9 >= classic["mrr"] and enhanced["ndcg"] + 1e-9 >= classic["ndcg"]
        print(f"\n[eval] {'PASS' if ok else 'FAIL'}: enhanced relevance must not regress")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
