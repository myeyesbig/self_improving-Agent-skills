"""Unit tests for the Qwen-Agent migration.

Uses only the standard library (unittest) plus mock/fake agents, so no real
DashScope API key or network access is required.
"""

import json
import os
import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from qwen_optimizer import SkillOptimizer, FailureAnalysis, SkillMutation
from app import AnalyzeRequest, RegenerateRequest, StartRequest


class FakeAgent:
    """Fake Qwen-Agent assistant whose run() is a synchronous generator."""

    def __init__(self, batches):
        self.batches = batches

    def run(self, messages, **kwargs):
        for batch in self.batches:
            yield batch


def make_optimizer(api_key="sk-test-1234"):
    return SkillOptimizer(api_key=api_key)


class TestSyncExtraction(unittest.TestCase):
    """Verify the final batch of a sync Qwen response generator is extracted."""

    def test_final_batch_text_is_extracted(self):
        agent = FakeAgent([
            [{"role": "assistant", "content": "partial stream"}],
            [{"role": "assistant", "content": ""}],
            [{"role": "assistant", "content": "final answer"}],
        ])
        opt = make_optimizer()
        self.assertEqual(opt._run_agent_sync(agent, "hi"), "final answer")

    def test_list_content_is_joined(self):
        agent = FakeAgent([[{"role": "assistant", "content": [{"text": "a"}, {"text": "b"}]}]])
        opt = make_optimizer()
        self.assertEqual(opt._run_agent_sync(agent, "hi"), "ab")

    def test_reasoning_content_is_ignored(self):
        agent = FakeAgent([
            [{"role": "assistant", "content": "only this", "reasoning_content": "hidden"}],
        ])
        opt = make_optimizer()
        self.assertEqual(opt._run_agent_sync(agent, "hi"), "only this")

    def test_no_text_raises_runtime_error(self):
        agent = FakeAgent([[{"role": "assistant", "content": ""}]])
        opt = make_optimizer()
        with self.assertRaises(RuntimeError):
            opt._run_agent_sync(agent, "hi")


class TestAskThreadBridge(unittest.IsolatedAsyncioTestCase):
    """Verify _ask() returns text through the asyncio.to_thread bridge."""

    async def test_ask_returns_text_via_thread(self):
        opt = make_optimizer()
        agent = FakeAgent([[{"role": "assistant", "content": "hello"}]])

        with patch(
            "qwen_optimizer.asyncio.to_thread",
            new=AsyncMock(return_value="hello"),
        ) as mock_to_thread:
            text = await opt._ask(agent, "hi")

        self.assertEqual(text, "hello")
        mock_to_thread.assert_called_once()
        # The thread bridge must invoke the sync runner with (agent, prompt).
        self.assertEqual(mock_to_thread.call_args[0][0], opt._run_agent_sync)
        self.assertEqual(mock_to_thread.call_args[0][1], agent)
        self.assertEqual(mock_to_thread.call_args[0][2], "hi")


class TestJsonParsing(unittest.IsolatedAsyncioTestCase):
    """Verify tolerant JSON extraction (code fences / extra text)."""

    async def test_code_fenced_json_is_parsed(self):
        payload = json.dumps({"scenarios": [{"id": 1}], "evals": []})
        text = f"Here you go:\n```json\n{payload}\n```\nDone."
        opt = make_optimizer()
        agent = FakeAgent([[{"role": "assistant", "content": text}]])
        result = await opt._ask_json(agent, "prompt")
        self.assertEqual(result["scenarios"], [{"id": 1}])

    async def test_extra_text_before_json_is_handled(self):
        text = 'Sure!\n{"a": 1}\nmore text'
        opt = make_optimizer()
        agent = FakeAgent([[{"role": "assistant", "content": text}]])
        result = await opt._ask_json(agent, "prompt")
        self.assertEqual(result, {"a": 1})

    async def test_non_json_uses_fallback(self):
        opt = make_optimizer()
        agent = FakeAgent([[{"role": "assistant", "content": "I cannot do that."}]])
        result = await opt._ask_json(agent, "prompt", fallback={"results": []})
        self.assertEqual(result, {"results": []})


class TestSchemaValidation(unittest.IsolatedAsyncioTestCase):
    """Verify Analyst/Mutator Pydantic schema validation and fallback."""

    async def test_analyst_schema_validation_passes(self):
        valid = {
            "diagnosis": "missing examples",
            "mutation_strategy": "add_example",
            "target_section": "body",
            "suggested_change": "add an example",
        }
        opt = make_optimizer()
        agent = FakeAgent([[{"role": "assistant", "content": json.dumps(valid)}]])
        result = await opt._ask_json(
            agent, "prompt",
            fallback={"diagnosis": "fb", "mutation_strategy": "add_constraint",
                      "target_section": "x", "suggested_change": "y"},
            schema=FailureAnalysis,
        )
        self.assertEqual(result["diagnosis"], "missing examples")

    async def test_analyst_schema_validation_falls_back(self):
        opt = make_optimizer()
        # Missing required fields -> schema validation must fail -> fallback
        agent = FakeAgent([[{"role": "assistant", "content": json.dumps({"diagnosis": "only"})}]])
        fallback = {"diagnosis": "fb", "mutation_strategy": "add_constraint",
                    "target_section": "x", "suggested_change": "y"}
        result = await opt._ask_json(agent, "prompt", fallback=fallback, schema=FailureAnalysis)
        self.assertEqual(result, fallback)

    async def test_mutator_schema_validation_passes(self):
        valid = {
            "description": "added example",
            "reasoning": "examples help",
            "new_skill_md": "# Updated",
        }
        opt = make_optimizer()
        agent = FakeAgent([[{"role": "assistant", "content": json.dumps(valid)}]])
        result = await opt._ask_json(agent, "prompt", fallback={}, schema=SkillMutation)
        self.assertEqual(result["new_skill_md"], "# Updated")

    async def test_mutator_schema_non_json_falls_back(self):
        opt = make_optimizer()
        agent = FakeAgent([[{"role": "assistant", "content": "not json at all"}]])
        fallback = {"description": "fb", "reasoning": "", "new_skill_md": "orig"}
        result = await opt._ask_json(agent, "prompt", fallback=fallback, schema=SkillMutation)
        self.assertEqual(result, fallback)


class TestApiKeyHandling(unittest.TestCase):
    """Verify the API key is never written to os.environ."""

    def test_api_key_not_written_to_os_environ(self):
        before = set(os.environ)
        key = "sk-test-secret-abc"
        make_optimizer(api_key=key)
        after = set(os.environ)
        self.assertEqual(after, before)
        self.assertNotIn(key, list(os.environ.values()))
        self.assertNotIn("GOOGLE_" + "API_KEY", os.environ)

    def test_model_defaults_to_qwen_plus(self):
        opt = make_optimizer()
        self.assertEqual(opt.model, "qwen-plus")

    def test_qwen_model_env_override(self):
        with patch.dict(os.environ, {"QWEN_MODEL": "qwen-max"}, clear=False):
            opt = make_optimizer()
            self.assertEqual(opt.model, "qwen-max")


class TestFastApiRequestModels(unittest.TestCase):
    """Verify request models only accept qwen_api_key."""

    def test_analyze_accepts_qwen_api_key(self):
        req = AnalyzeRequest(session_id="s1", qwen_api_key="k")
        self.assertEqual(req.qwen_api_key, "k")

    def test_analyze_rejects_legacy_key_field(self):
        with self.assertRaises(ValidationError):
            AnalyzeRequest(session_id="s1", **{"gemini_" + "api_key": "k"})

    def test_regenerate_accepts_qwen_api_key(self):
        req = RegenerateRequest(session_id="s1", qwen_api_key="k")
        self.assertEqual(req.qwen_api_key, "k")

    def test_regenerate_rejects_legacy_key_field(self):
        with self.assertRaises(ValidationError):
            RegenerateRequest(session_id="s1", **{"gemini_" + "api_key": "k"})

    def test_start_accepts_qwen_api_key(self):
        req = StartRequest(qwen_api_key="k")
        self.assertEqual(req.qwen_api_key, "k")
        self.assertEqual(req.max_rounds, 20)

    def test_start_rejects_legacy_key_field(self):
        with self.assertRaises(ValidationError):
            StartRequest(**{"gemini_" + "api_key": "k"})


if __name__ == "__main__":
    unittest.main()
