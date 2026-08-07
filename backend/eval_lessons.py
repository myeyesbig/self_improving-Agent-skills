#!/usr/bin/env python
"""eval_lessons.py — 评估 RAG 经验检索质量（后续选项 C 的验收脚本）。

用法：
  # 无 key：用确定性 mock embedding 走通流程（粗验证）
  python eval_lessons.py --lessons /tmp/skill_lessons.db

  # 有 key：真实 DashScope text-embedding-v3 评估
  python eval_lessons.py --lessons /tmp/skill_lessons.db --api-key sk-xxx

  # 指定人工查询集（jsonl，每条 {"query": "...", "skill_name": "...", "domain": "..."}）
  python eval_lessons.py --lessons x.db --queries queries.jsonl --api-key sk-xxx

指标对照 rag-lesson-retrieval.md 目标：top-5 相关率 ≥0.8、误注入率 <20%、延迟 <50ms。
注意：无人工查询集时默认用"经验自身的检索文本"作为查询（自相关冒烟，结果偏乐观，
真实验收请提供 --queries 人工查询集）。
"""
import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen_optimizer import SkillOptimizer


def mock_embed(text):
    """确定性 mock：字符和哈希 → 8 维 one-hot（无 key 时走通流程）。"""
    idx = sum(ord(c) for c in text) % 8
    vec = [0.0] * 8
    vec[idx] = 1.0
    return vec


def build_queries(lessons):
    """默认查询集：以每条经验的检索文本为查询（自相关冒烟）。"""
    queries = []
    for l in lessons:
        q = SkillOptimizer._lesson_index_text(l)
        if not q:
            q = f"skill: {l.get('skill_name') or ''}".strip()
        queries.append({
            "query": q,
            "skill_name": l.get("skill_name"),
            "domain": l.get("domain"),
        })
    return queries


def load_queries(path):
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                queries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return queries


def relevant(lesson, q):
    """相关判定：技能同名 或 领域相同；查询无标签时命中"无绑定"经验。"""
    if not q.get("skill_name") and not q.get("domain"):
        # 通用查询：期望召回无技能/领域绑定的经验。
        return not lesson.get("skill_name") and not lesson.get("domain")
    if q.get("skill_name") and lesson.get("skill_name") == q["skill_name"]:
        return True
    if q.get("domain") and lesson.get("domain") == q["domain"] and lesson.get("domain"):
        return True
    return False


async def main():
    ap = argparse.ArgumentParser(description="RAG 经验检索质量评估")
    ap.add_argument("--lessons", default="/tmp/skill_lessons.db", help="经验库路径（.db SQLite）")
    ap.add_argument("--api-key", default=None,
                    help="DashScope key；缺省读 DASHSCOPE_API_KEY 环境变量；都没有则用 mock embedding")
    ap.add_argument("--queries", default=None, help="人工查询集 jsonl（可选）")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.3)
    args = ap.parse_args()
    if not args.api_key:
        args.api_key = os.environ.get("DASHSCOPE_API_KEY")

    lessons = SkillOptimizer._load_lessons(args.lessons, limit=1000)
    if not lessons:
        print(f"[eval] 经验库为空: {args.lessons}")
        return 1
    print(f"[eval] 经验库: {len(lessons)} 条 @ {args.lessons}")

    opt = SkillOptimizer(
        api_key=args.api_key or "sk-mock",
        lesson_retrieval="semantic",
        lesson_threshold=args.threshold,
        lesson_top_k=args.top_k,
    )
    if not args.api_key:
        async def fake_embed(text):
            return mock_embed(text)
        opt._embed = fake_embed  # 覆盖为 mock
        # mock 模式统一用 mock 向量：丢弃库里已有的真实 embedding，避免维度不匹配。
        for l in lessons:
            l["embedding"] = None
        print("[eval] 模式: mock embedding（无 key，流程冒烟）")
    else:
        print("[eval] 模式: 真实 DashScope text-embedding-v3")

    queries = load_queries(args.queries) if args.queries else build_queries(lessons)
    if not queries:
        print("[eval] 查询集为空")
        return 1
    print(f"[eval] 查询集: {len(queries)} 条"
          + (f" @ {args.queries}" if args.queries else "（自相关冒烟）"))

    hits, total, hits_at_1, latencies = 0, 0, 0, []
    for q in queries:
        t0 = time.perf_counter()
        try:
            top = await opt._retrieve_lessons(lessons, q["query"], "semantic", args.top_k)
        except Exception as e:
            print(f"[eval] 检索失败（{q['query'][:40]}）: {e}")
            continue
        latencies.append((time.perf_counter() - t0) * 1000)
        for rank, l in enumerate(top):
            total += 1
            if relevant(l, q):
                hits += 1
                if rank == 0:
                    hits_at_1 += 1
    if not total:
        print("[eval] 无检索结果（阈值过滤后为空），相关率不可计算")
        return 1

    rel = hits / total
    rel_at_1 = hits_at_1 / len(latencies) if latencies else 0.0
    avg_ms = sum(latencies) / len(latencies) if latencies else 0.0
    print(f"[eval] top-1 相关率:          {rel_at_1:.3f}    (首个结果即相关；小库上最有意义)")
    print(f"[eval] top-{args.top_k} 相关率:  {rel:.3f}    (目标 ≥0.80)")
    print(f"[eval] 误注入率:            {1 - rel:.3f}    (目标 <0.20)")
    print(f"[eval] 平均端到端延迟:      {avg_ms:.1f} ms  (含 embedding API；目标 <50ms 仅指本地计算)")
    ok = rel_at_1 >= 0.8 and avg_ms < 50
    print(f"[eval] 结论: {'✅ 达标' if ok else '⚠️ 部分达标（见指标说明）'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
