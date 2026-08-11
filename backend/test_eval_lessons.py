"""Deterministic tests for the lesson-retrieval quality benchmark."""
import os
import unittest

from eval_lessons import evaluate, load_jsonl, mock_embed, ranking_metrics
from qwen_optimizer import SkillOptimizer


FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


class TestLessonRetrievalEvaluation(unittest.IsolatedAsyncioTestCase):
    def test_mock_embedding_is_deterministic(self):
        left = mock_embed("修复 YAML frontmatter format")
        right = mock_embed("修复 YAML frontmatter format")
        other = mock_embed("unrelated incident recovery")
        self.assertEqual(left, right)
        self.assertGreater(SkillOptimizer._cosine(left, right), SkillOptimizer._cosine(left, other))

    def test_explicit_relevance_drives_metrics(self):
        results = [
            {"eval_id": "wrong", "strategy": "a", "summary": "x"},
            {"eval_id": "right", "strategy": "b", "summary": "y"},
        ]
        query = {"relevant_ids": ["right"]}
        metrics = ranking_metrics(results, results, query, top_k=2)
        self.assertEqual(metrics["top1"], 0.0)
        self.assertEqual(metrics["mrr"], 0.5)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertLess(metrics["ndcg"], 1.0)

    async def test_quality_diverse_fixture_does_not_regress_relevance(self):
        lessons = load_jsonl(os.path.join(FIXTURES, "lesson_eval_lessons.jsonl"))
        queries = load_jsonl(os.path.join(FIXTURES, "lesson_eval_queries.jsonl"))
        classic = await evaluate(lessons, queries, "classic", "hybrid", 3, 0.3)
        enhanced = await evaluate(lessons, queries, "quality_diverse", "hybrid", 3, 0.3)
        self.assertGreaterEqual(enhanced["mrr"], classic["mrr"])
        self.assertGreaterEqual(enhanced["ndcg"], classic["ndcg"])
        self.assertGreaterEqual(enhanced["recall"], classic["recall"])
        self.assertGreater(enhanced["pairwise_diversity"], classic["pairwise_diversity"])


if __name__ == "__main__":
    unittest.main()
