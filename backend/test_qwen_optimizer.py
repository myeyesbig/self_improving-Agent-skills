# =============================================================================
# 【文件头】test_qwen_optimizer.py —— 优化器的单元测试
# 职责：验证 Qwen-Agent 迁移后的核心行为：同步生成器取文本、asyncio.to_thread
#       线程桥接、宽容 JSON 提取、Pydantic schema 校验与 fallback、API Key
#       安全（不写环境变量）、请求模型只接受 qwen_api_key。
# 接收：无外部输入；使用 unittest 标准库。
# 输出：unittest 测试结果（不需要真实 DashScope Key 或网络）。
# 建议先看：FakeAgent（假模型）→ TestSyncExtraction → TestAskThreadBridge。
# 【初学者提示】测试不调用真实模型：FakeAgent 用"同步生成器"模拟 Assistant.run()
#       的输出；mock 用来替换 asyncio.to_thread，让测试无需真线程；API Key
#       安全测试刻意使用假密钥（sk-test-*），从不上真网络。
# =============================================================================

"""Unit tests for the Qwen-Agent migration.

Uses only the standard library (unittest) plus mock/fake agents, so no real
DashScope API key or network access is required.
"""

import asyncio
import io
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import ValidationError

from llm_client import DEFAULT_MODEL, GLM_REQUEST_TIMEOUT, LLMClient, _is_codex_model
from qwen_optimizer import (
    ANALYST_SYSTEM_PROMPT,
    EXECUTOR_SYSTEM_PROMPT,
    MUTATOR_SYSTEM_PROMPT,
    STRATEGY_TEMPLATES,
    FailureAnalysis,
    SkillMutation,
    SkillOptimizer,
    StopOptimizationError,
    _active_strategy_pool,
)
from optimize_graph import build_optimize_graph
from app import (
    AnalyzeRequest,
    RegenerateRequest,
    StartRequest,
    list_examples,
    load_example,
    parse_skill_frontmatter,
)


class FakeAgent:
    """Fake Qwen-Agent assistant whose run() is a synchronous generator."""

    # 【初学者提示】FakeAgent 是"替身模型"：run() 按批次依次 yield 预先写好的
    # 消息列表，行为与真实 Assistant.run() 的同步生成器一致，但完全不联网。
    def __init__(self, batches):
        self.batches = batches

    def run(self, messages, **kwargs):
        for batch in self.batches:
            yield batch


def make_optimizer(api_key="sk-test-1234", **kwargs):
    # 构造测试用的 SkillOptimizer；默认使用假密钥，绝不触碰真实凭据。
    return SkillOptimizer(api_key=api_key, **kwargs)


class TestSyncExtraction(unittest.TestCase):
    """Verify the sync LLM call returns the assistant text via the thin shell.

    阶段 1 起 _run_agent_sync 不再消费 Qwen-Agent 的同步生成器，而是把
    (system message, prompt, json_mode) 转发给 llm_client.sync_call 并返回文本。
    agent 参数仅为兼容保留：有 system_message 就用，没有则回退空串。
    """

    def _opt_with_llm(self, return_value="final answer"):
        opt = make_optimizer()
        mock_llm = MagicMock()
        mock_llm.sync_call.return_value = return_value
        opt._llm = mock_llm
        return opt, mock_llm

    def test_final_text_returned(self):
        opt, mock_llm = self._opt_with_llm()
        agent = FakeAgent([])
        self.assertEqual(opt._run_agent_sync(agent, "hi"), "final answer")
        mock_llm.sync_call.assert_called_once_with(system="", user="hi", json_mode=False)

    def test_system_message_taken_from_agent(self):
        opt, mock_llm = self._opt_with_llm()
        agent = FakeAgent([])
        agent.system_message = "sys prompt"
        opt._run_agent_sync(agent, "hi")
        mock_llm.sync_call.assert_called_once_with(system="sys prompt", user="hi", json_mode=False)

    def test_agent_without_system_message_falls_back_empty(self):
        opt, mock_llm = self._opt_with_llm()
        agent = FakeAgent([])
        opt._run_agent_sync(agent, "hi")
        mock_llm.sync_call.assert_called_once_with(system="", user="hi", json_mode=False)

    def test_json_mode_passthrough(self):
        opt, mock_llm = self._opt_with_llm()
        agent = FakeAgent([])
        opt._run_agent_sync(agent, "hi", json_mode=True)
        mock_llm.sync_call.assert_called_once_with(system="", user="hi", json_mode=True)


class TestRoleSpecificModels(unittest.TestCase):
    """目标 1：角色模型可独立覆盖，未配置时保持单模型旧路径。"""

    def test_roles_inherit_base_model_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            opt = make_optimizer(model="qwen-plus")
        self.assertEqual(opt.executor_model, "qwen-plus")
        self.assertEqual(opt.analyst_model, "qwen-plus")
        self.assertEqual(opt.mutator_model, "qwen-plus")
        self.assertEqual(opt._role_llms, {})

    def test_role_environment_overrides_create_only_needed_clients(self):
        env = {
            "EXECUTOR_MODEL": "qwen-turbo",
            "ANALYST_MODEL": "deepseek-reasoner",
            "MUTATOR_MODEL": "qwen-plus",
        }
        with patch.dict(os.environ, env, clear=True):
            opt = make_optimizer(model="qwen-plus")
        self.assertEqual(opt.executor_model, "qwen-turbo")
        self.assertEqual(opt.analyst_model, "deepseek-reasoner")
        self.assertEqual(opt.mutator_model, "qwen-plus")
        self.assertEqual(opt._role_llms["executor"].model, "qwen-turbo")
        self.assertEqual(opt._role_llms["analyst"].model, "deepseek-reasoner")
        self.assertNotIn("mutator", opt._role_llms)

    def test_calls_route_by_stable_role_including_parallel_slots(self):
        with patch.dict(os.environ, {"MUTATOR_MODEL": "deepseek-chat"}, clear=True):
            opt = make_optimizer(model="qwen-plus", parallelism=2)
        base = MagicMock()
        role_client = MagicMock()
        base.sync_call.return_value = "executor"
        role_client.sync_call.return_value = "mutator"
        opt._llm = base
        opt._role_llms["mutator"] = role_client

        self.assertEqual(opt._run_agent_sync(opt.executor, "run"), "executor")
        self.assertEqual(opt._run_agent_sync(opt.mutator_pool[1], "edit"), "mutator")
        base.sync_call.assert_called_once_with(system=EXECUTOR_SYSTEM_PROMPT, user="run", json_mode=False)
        role_client.sync_call.assert_called_once_with(system=MUTATOR_SYSTEM_PROMPT, user="edit", json_mode=False)


class TestAskThreadBridge(unittest.IsolatedAsyncioTestCase):
    """Verify _ask() returns text through the asyncio.to_thread bridge."""

    # 用 mock 替换 asyncio.to_thread，验证：_ask 一定走线程桥接，并且把
    # (同步执行器, agent, prompt) 三个参数正确传递过去。
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
    """Verify tolerant JSON extraction (code fences / extra text).

    _ask 已 mock（LLM 层在 llm_client 测试中覆盖），这里专注 _ask_json 的
    宽容解析链：代码块包裹、前后杂文、完全非 JSON 三种情况。
    """

    # 宽容解析 1：模型把 JSON 包在 ```json 代码块里也能提取。
    async def test_code_fenced_json_is_parsed(self):
        payload = json.dumps({"scenarios": [{"id": 1}], "evals": []})
        text = f"Here you go:\n```json\n{payload}\n```\nDone."
        opt = make_optimizer()
        with patch.object(opt, "_ask", new=AsyncMock(return_value=text)):
            result = await opt._ask_json(None, "prompt")
        self.assertEqual(result["scenarios"], [{"id": 1}])

    # 宽容解析 2：JSON 前后有额外文字也能定位并解析。
    async def test_extra_text_before_json_is_handled(self):
        text = 'Sure!\n{"a": 1}\nmore text'
        opt = make_optimizer()
        with patch.object(opt, "_ask", new=AsyncMock(return_value=text)):
            result = await opt._ask_json(None, "prompt")
        self.assertEqual(result, {"a": 1})

    # 完全不是 JSON 时返回 fallback（兜底值），而不是抛异常。
    async def test_non_json_uses_fallback(self):
        opt = make_optimizer()
        with patch.object(opt, "_ask", new=AsyncMock(return_value="I cannot do that.")):
            result = await opt._ask_json(None, "prompt", fallback={"results": []})
        self.assertEqual(result, {"results": []})


class TestSchemaValidation(unittest.IsolatedAsyncioTestCase):
    """Verify Analyst/Mutator Pydantic schema validation and fallback."""

    # Analyst 返回符合 FailureAnalysis schema 时正常通过校验。
    async def test_analyst_schema_validation_passes(self):
        valid = {
            "diagnosis": "missing examples",
            "mutation_strategy": "add_example",
            "target_section": "body",
            "suggested_change": "add an example",
        }
        opt = make_optimizer()
        with patch.object(opt, "_ask", new=AsyncMock(return_value=json.dumps(valid))):
            result = await opt._ask_json(
                None, "prompt",
                fallback={"diagnosis": "fb", "mutation_strategy": "add_constraint",
                          "target_section": "x", "suggested_change": "y"},
                schema=FailureAnalysis,
            )
        self.assertEqual(result["diagnosis"], "missing examples")

    # 缺少必填字段 → schema 校验失败 → 使用 fallback（结构化输出兜底）。
    async def test_analyst_schema_validation_falls_back(self):
        opt = make_optimizer()
        # Missing required fields -> schema validation must fail -> fallback
        with patch.object(opt, "_ask", new=AsyncMock(return_value=json.dumps({"diagnosis": "only"}))):
            fallback = {"diagnosis": "fb", "mutation_strategy": "add_constraint",
                        "target_section": "x", "suggested_change": "y"}
            result = await opt._ask_json(None, "prompt", fallback=fallback, schema=FailureAnalysis)
        self.assertEqual(result, fallback)

    # Mutator 返回符合 SkillMutation schema 时正常通过。
    async def test_mutator_schema_validation_passes(self):
        valid = {
            "description": "added example",
            "reasoning": "examples help",
            "new_skill_md": "# Updated",
        }
        opt = make_optimizer()
        with patch.object(opt, "_ask", new=AsyncMock(return_value=json.dumps(valid))):
            result = await opt._ask_json(None, "prompt", fallback={}, schema=SkillMutation)
        self.assertEqual(result["new_skill_md"], "# Updated")

    # Mutator 返回非 JSON → fallback（此时 new_skill_md 兜底为原内容）。
    async def test_mutator_schema_non_json_falls_back(self):
        opt = make_optimizer()
        with patch.object(opt, "_ask", new=AsyncMock(return_value="not json at all")):
            fallback = {"description": "fb", "reasoning": "", "new_skill_md": "orig"}
            result = await opt._ask_json(None, "prompt", fallback=fallback, schema=SkillMutation)
        self.assertEqual(result, fallback)


class TestApiKeyHandling(unittest.TestCase):
    """Verify the API key is never written to os.environ."""

    # 【安全】构造优化器前后对比 os.environ：密钥既不新增环境变量，也不
    # 出现在任何已有环境变量的值里；同时确认不存在旧版提供商的密钥变量。
    def test_api_key_not_written_to_os_environ(self):
        before = set(os.environ)
        key = "sk-test-secret-abc"
        make_optimizer(api_key=key)
        after = set(os.environ)
        self.assertEqual(after, before)
        self.assertNotIn(key, list(os.environ.values()))
        self.assertNotIn("GOOGLE_" + "API_KEY", os.environ)

    # 临时默认模型使用当前 ChatGPT 账户 model/list 标记的 Codex 默认项。
    def test_model_defaults_to_codex_sol(self):
        opt = make_optimizer()
        self.assertEqual(opt.model, DEFAULT_MODEL)

    # QWEN_MODEL 环境变量可覆盖默认模型。
    def test_qwen_model_env_override(self):
        with patch.dict(os.environ, {"QWEN_MODEL": "qwen-max"}, clear=False):
            opt = make_optimizer()
            self.assertEqual(opt.model, "qwen-max")


class TestEarlyStopAt100(unittest.IsolatedAsyncioTestCase):
    """评分达到 100% 时优化必须立即停止，不再执行后续轮次。

    通过 mock 掉 _score_skill / _analyze_failures / _mutate_skill，统计
    调用次数：若提前终止生效，达到 100% 之后不应再有 Analyst/Mutator/复评调用。
    """

    async def test_baseline_at_100_skips_all_rounds(self):
        opt = make_optimizer()
        # 基线评分直接全通过（4/4 = 100%）
        perfect = {"passed": 4, "total": 4, "per_eval": [], "details": []}
        events = []

        async def collect(event):
            events.append(event)

        with (
            patch.object(opt, "_score_skill", new=AsyncMock(return_value=perfect)) as mock_score,
            patch.object(opt, "_analyze_failures", new=AsyncMock()) as mock_analyze,
            patch.object(opt, "_mutate_skill", new=AsyncMock()) as mock_mutate,
        ):
            result = await opt.optimize(
                {"SKILL.md": "# S"}, [], [], max_rounds=5, callback=collect
            )

        # 基线 100% → 一轮都不跑：分析/修改零调用，评分只有基线那一次
        mock_analyze.assert_not_called()
        mock_mutate.assert_not_called()
        self.assertEqual(mock_score.await_count, 1)
        self.assertEqual(result["final_score"], 100.0)
        self.assertEqual(result["score_history"], [100.0])
        self.assertFalse(any(e["type"] == "experiment_start" for e in events))
        self.assertTrue(any(e["type"] == "complete" for e in events))

    async def test_stops_after_round_reaches_100(self):
        opt = make_optimizer()
        # 基线 50%（2/4）→ 第 1 轮复评 100%（4/4），之后不应再有第 2 轮
        half = {"passed": 2, "total": 4, "per_eval": [], "details": []}
        perfect = {"passed": 4, "total": 4, "per_eval": [], "details": []}
        analysis = {
            "diagnosis": "missing examples",
            "mutation_strategy": "add_example",
            "target_section": "body",
            "suggested_change": "add examples",
        }
        mutation = {"description": "added examples", "reasoning": "r", "new_skill_md": "# New"}
        events = []

        async def collect(event):
            events.append(event)

        with (
            patch.object(opt, "_score_skill", new=AsyncMock(side_effect=[half, perfect])) as mock_score,
            patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=analysis)) as mock_analyze,
            patch.object(opt, "_mutate_skill", new=AsyncMock(return_value=mutation)) as mock_mutate,
        ):
            result = await opt.optimize(
                {"SKILL.md": "# S"}, [], [], max_rounds=5, callback=collect
            )

        # 只跑第 1 轮：分析/修改各一次，评分两次（基线 + 一轮复评）
        self.assertEqual(mock_analyze.await_count, 1)
        self.assertEqual(mock_mutate.await_count, 1)
        self.assertEqual(mock_score.await_count, 2)
        self.assertEqual(result["final_score"], 100.0)
        self.assertEqual(result["score_history"], [50.0, 100.0])
        self.assertEqual(len(result["mutation_log"]), 1)
        # 只有 1 个 experiment_start 事件，绝无第 2 轮
        starts = [e for e in events if e["type"] == "experiment_start"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0]["data"]["round"], 1)
        self.assertTrue(any(e["type"] == "complete" for e in events))


class TestFastApiRequestModels(unittest.TestCase):
    """Verify request models only accept qwen_api_key."""

    # 【安全】请求模型只接受 qwen_api_key；旧版提供商的凭据字段会被
    # Pydantic 拒绝（ValidationError），保证迁移后不兼容旧凭据。
    def test_analyze_accepts_qwen_api_key(self):
        req = AnalyzeRequest(session_id="s1", qwen_api_key="k")
        self.assertEqual(req.qwen_api_key, "k")

    def test_codex_default_accepts_empty_compatibility_key(self):
        analyze = AnalyzeRequest(session_id="s1", qwen_api_key="")
        regenerate = RegenerateRequest(session_id="s1", qwen_api_key="")
        start = StartRequest(qwen_api_key="")
        self.assertEqual(analyze.qwen_api_key, "")
        self.assertEqual(regenerate.qwen_api_key, "")
        self.assertEqual(start.qwen_api_key, "")

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

    # 【双 key】请求模型可选接受 deepseek_api_key（前端双输入框场景）。
    def test_analyze_accepts_deepseek_key(self):
        req = AnalyzeRequest(session_id="s1", qwen_api_key="qk", deepseek_api_key="dk")
        self.assertEqual(req.deepseek_api_key, "dk")

    def test_analyze_deepseek_key_optional(self):
        req = AnalyzeRequest(session_id="s1", qwen_api_key="qk")
        self.assertIsNone(req.deepseek_api_key)

    def test_regenerate_accepts_deepseek_key(self):
        req = RegenerateRequest(session_id="s1", qwen_api_key="qk", deepseek_api_key="dk")
        self.assertEqual(req.deepseek_api_key, "dk")

    def test_start_accepts_deepseek_key(self):
        req = StartRequest(qwen_api_key="qk", deepseek_api_key="dk")
        self.assertEqual(req.deepseek_api_key, "dk")

    def test_start_rejects_legacy_key_field(self):
        with self.assertRaises(ValidationError):
            StartRequest(**{"gemini_" + "api_key": "k"})

    # 【C1】StartRequest 支持策略白名单 / 并行数 / 提升阈值等新旋钮。
    def test_start_accepts_new_knobs(self):
        req = StartRequest(
            qwen_api_key="k",
            parallel_mutations=2,
            strategy_pool=["add_example", "add_constraint"],
            improvement_threshold=0.5,
        )
        self.assertEqual(req.parallel_mutations, 2)
        self.assertEqual(req.strategy_pool, ["add_example", "add_constraint"])
        self.assertEqual(req.improvement_threshold, 0.5)


class TestStrategyPool(unittest.TestCase):
    """C1: strategy pool resolution + pool filtering in analyst prompt."""

    def test_all_templates_available_by_default(self):
        self.assertIn("add_example", STRATEGY_TEMPLATES)
        self.assertIn("add_reference", STRATEGY_TEMPLATES)
        self.assertIn("rewrite_section", STRATEGY_TEMPLATES)
        self.assertIn("fix_format", STRATEGY_TEMPLATES)

    def test_active_pool_prefers_explicit_whitelist(self):
        pool = _active_strategy_pool(["add_example", "bogus_strategy"])
        self.assertEqual(pool, ["add_example"])

    def test_active_pool_falls_back_to_all_when_empty(self):
        pool = _active_strategy_pool([])
        self.assertEqual(pool, list(STRATEGY_TEMPLATES.keys()))

    def test_active_pool_reads_env(self):
        with patch.dict(os.environ, {"MUTATION_STRATEGIES": "restructure,fix_format"}, clear=False):
            pool = _active_strategy_pool(None)
            self.assertEqual(pool, ["restructure", "fix_format"])


class TestAdaptiveSearchPolicy(unittest.IsolatedAsyncioTestCase):
    """目标 2：可选自适应策略组合；classic 默认路径不变。"""

    def test_default_and_invalid_policy_are_classic(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(make_optimizer().search_policy, "classic")
        with patch.dict(os.environ, {"OPTIMIZATION_SEARCH": "unknown"}, clear=True):
            self.assertEqual(make_optimizer().search_policy, "classic")

    def test_cold_start_honors_analyst_then_diversifies(self):
        opt = make_optimizer(search_policy="adaptive")
        picked = opt._select_candidate_strategies(
            ["add_example", "add_constraint", "restructure"],
            [], 3, "add_constraint",
        )
        self.assertEqual(picked, ["add_constraint", "add_example", "restructure"])

    def test_observed_gain_drives_exploitation_when_exploration_zero(self):
        opt = make_optimizer(search_policy="adaptive", search_exploration=0.0)
        history = [
            {"strategy_type": "add_example", "score_before": 50, "score_after": 80},
            {"strategy_type": "add_constraint", "score_before": 50, "score_after": 55},
        ]
        picked = opt._select_candidate_strategies(
            ["add_example", "add_constraint"], history, 1, "add_constraint",
        )
        self.assertEqual(picked, ["add_example"])

    async def test_parallel_candidates_receive_distinct_consistent_strategies(self):
        opt = make_optimizer(search_policy="adaptive")
        baseline = {"passed": 1, "total": 2, "per_eval": [], "details": []}
        same = {"passed": 1, "total": 2, "per_eval": [], "details": []}
        analysis = {"diagnosis": "d", "mutation_strategy": "add_constraint",
                    "target_section": "s", "suggested_change": "c"}
        seen = []

        async def mutate(_md, slot_analysis, **_kwargs):
            seen.append(slot_analysis["mutation_strategy"])
            return {"description": slot_analysis["mutation_strategy"], "reasoning": "r",
                    "new_skill_md": "# " + slot_analysis["mutation_strategy"]}

        with (
            patch.object(opt, "_score_skill", new=AsyncMock(side_effect=[baseline, same, same])),
            patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=analysis)),
            patch.object(opt, "_mutate_skill", new=mutate),
        ):
            result = await opt.optimize(
                {"SKILL.md": "# S"}, [], [], max_rounds=1, parallel_mutations=2,
                strategy_pool=["add_example", "add_constraint"],
            )

        self.assertEqual(seen, ["add_constraint", "add_example"])
        self.assertEqual(
            [m["strategy_type"] for m in result["mutation_log"]],
            ["add_constraint", "add_example"],
        )
        self.assertFalse(any(m["kept"] for m in result["mutation_log"]))


class TestTieDimensionLessons(unittest.IsolatedAsyncioTestCase):
    """目标 3：平局候选可贡献维度经验，但绝不替换严格提升的 winner。"""

    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(make_optimizer().tie_dimension_lessons)

    def test_dimension_advantages_apply_minimum_gain(self):
        opt = make_optimizer(tie_dimension_min_gain=5.0)
        gains = opt._dimension_advantages(
            {"clarity": {"pct": 80}, "quality": {"pct": 62}},
            {"clarity": {"pct": 60}, "quality": {"pct": 60}},
        )
        self.assertEqual(gains["clarity"]["gain"], 20.0)
        self.assertNotIn("quality", gains)

    async def test_tied_nonwinner_persists_only_dimension_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            lesson_path = os.path.join(td, "lessons.jsonl")
            opt = make_optimizer(
                lesson_file=lesson_path,
                search_policy="adaptive",
                tie_dimension_lessons=True,
            )
            baseline = {"passed": 2, "total": 4, "per_eval": [], "details": [],
                        "dimension_scores": {}}
            winner = {"passed": 3, "total": 4, "per_eval": [], "details": [],
                      "dimension_scores": {
                          "correctness": {"pct": 100}, "clarity": {"pct": 50},
                      }}
            tied = {"passed": 3, "total": 4, "per_eval": [], "details": [],
                    "dimension_scores": {
                        "correctness": {"pct": 75}, "clarity": {"pct": 100},
                    }}
            analysis = {"diagnosis": "d", "mutation_strategy": "add_constraint",
                        "target_section": "s", "suggested_change": "c"}

            async def mutate(_md, slot_analysis, **_kwargs):
                strategy = slot_analysis["mutation_strategy"]
                new_md = "# Winner" if strategy == "add_constraint" else "# Tie"
                return {"description": strategy, "reasoning": "r", "new_skill_md": new_md}

            async def score(skill_md, *_args, **_kwargs):
                return {"# S": baseline, "# Winner": winner, "# Tie": tied}[skill_md]

            with (
                patch.object(opt, "_score_skill", new=score),
                patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=analysis)),
                patch.object(opt, "_mutate_skill", new=mutate),
            ):
                result = await opt.optimize(
                    {"SKILL.md": "# S"}, [], [], max_rounds=1, parallel_mutations=2,
                    strategy_pool=["add_example", "add_constraint"],
                )

            kept = [m for m in result["mutation_log"] if m["kept"]]
            self.assertEqual(len(kept), 1)
            self.assertEqual(kept[0]["description"], "add_constraint")
            self.assertEqual(result["improved_skill_md"], "# Winner")
            lessons = SkillOptimizer._load_lessons(lesson_path, limit=10)
            tie_lessons = [l for l in lessons if l.get("lesson_type") == "tie_dimension"]
            self.assertEqual(len(tie_lessons), 1)
            self.assertEqual(tie_lessons[0]["comparison_scope"], "same_round")
            self.assertEqual(tie_lessons[0]["target_dimension"], "clarity")
            self.assertEqual(tie_lessons[0]["dimension_gains"]["clarity"]["gain"], 50.0)
            self.assertNotIn("skill_md", tie_lessons[0])
            self.assertNotIn("new_skill_md", tie_lessons[0])

    def test_sqlite_round_trip_preserves_tie_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "lessons.db")
            lesson = {
                "strategy": "add_example", "diagnosis": "d", "summary": "s",
                "score_before": 50.0, "score_after": 75.0,
                "lesson_type": "tie_dimension",
                "dimension_gains": {"clarity": {"gain": 25.0}},
            }
            SkillOptimizer._append_lesson(path, lesson)
            loaded = SkillOptimizer._load_lessons(path, limit=1)[0]
            self.assertEqual(loaded["lesson_type"], "tie_dimension")
            self.assertEqual(loaded["comparison_scope"], "same_round")
            self.assertEqual(loaded["dimension_gains"]["clarity"]["gain"], 25.0)


class TestCrossRoundTieDimensionLessons(unittest.IsolatedAsyncioTestCase):
    """跨轮同分候选只贡献元数据，并始终以最近一次 kept incumbent 为参照。"""

    @staticmethod
    def _dimensions(**scores):
        return {
            name: {"passed": 1, "total": 1, "pct": float(pct)}
            for name, pct in scores.items()
        }

    @classmethod
    def _score(cls, passed, total=100, *, dimensions=None, details=None):
        return {
            "passed": passed,
            "total": total,
            "per_eval": [],
            "details": list(details or []),
            "dimension_scores": dimensions or {},
        }

    @staticmethod
    def _analysis(**overrides):
        analysis = {
            "diagnosis": "model diagnosis",
            "mutation_strategy": "add_example",
            "target_section": "Workflow",
            "suggested_change": "add one focused example",
        }
        analysis.update(overrides)
        return analysis

    @staticmethod
    def _candidate(candidate_id, score, dimensions, *, new_md=None, strategy=None,
                   rejected=None, description=None):
        candidate = {
            "candidate_id": candidate_id,
            "new_md": new_md if new_md is not None else f"# Candidate {candidate_id}",
            "description": description or f"candidate {candidate_id} summary",
            "reasoning": "model reasoning",
            "strategy_type": strategy or f"strategy-{candidate_id}",
            "target_dimension": None,
            "rejected": rejected,
            "score_after": score,
            "per_eval": [],
            "details": [],
            "dimension_scores": dimensions,
        }
        return candidate

    @staticmethod
    async def _decide(opt, *, baseline=80.0, incumbent_md="# Incumbent",
                      incumbent_dimensions=None, candidates=None, rescored=None,
                      confirmation=None):
        graph = build_optimize_graph(
            opt,
            max_rounds=1,
            n_candidates=max(1, len(candidates or [])),
            strategy_pool=["add_example"],
            effective_patience=0,
            scenarios=[],
            evals=[],
        )
        state = {
            "skill_md": "---\nname: safe-skill\n---\n# Original",
            "domain": "writing",
            "current_md": incumbent_md,
            "baseline_pct": baseline,
            "current_details": [{"eval_id": 1, "passed": False}],
            "current_dimension_scores": incumbent_dimensions or {},
            "weak_dimensions": [],
            "score_history": [baseline],
            "mutation_log": [],
            "round_memory": [],
            "round_idx": 0,
            "no_improve": 0,
            "analysis": TestCrossRoundTieDimensionLessons._analysis(),
            "candidates": candidates or [],
            "rescored": rescored or [],
            "confirmation": confirmation or {},
        }
        return await graph.nodes["decide"].ainvoke(state)

    @staticmethod
    def _tie_lessons(path, scope=None):
        lessons = [
            lesson for lesson in SkillOptimizer._load_lessons(path, limit=100)
            if lesson.get("lesson_type") == "tie_dimension"
        ]
        if scope is not None:
            lessons = [
                lesson for lesson in lessons
                if lesson.get("comparison_scope") == scope
            ]
        return lessons

    def test_default_off_and_independent_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            opt = make_optimizer()
        self.assertFalse(opt.cross_round_tie_dimension_lessons)
        self.assertEqual(opt.cross_round_tie_dimension_min_gain, 0.0)

        # 开启旧的同轮功能不得静默开启跨轮功能。
        with patch.dict(os.environ, {"TIE_DIMENSION_LESSONS": "1"}, clear=True):
            opt = make_optimizer()
        self.assertTrue(opt.tie_dimension_lessons)
        self.assertFalse(opt.cross_round_tie_dimension_lessons)

        with patch.dict(os.environ, {
            "CROSS_ROUND_TIE_DIMENSION_LESSONS": "1",
            "CROSS_ROUND_TIE_DIMENSION_MIN_GAIN": "6.5",
        }, clear=True):
            opt = make_optimizer()
        self.assertTrue(opt.cross_round_tie_dimension_lessons)
        self.assertEqual(opt.cross_round_tie_dimension_min_gain, 6.5)

        # 显式构造参数优先于环境变量，且跨轮阈值不复用同轮阈值。
        with patch.dict(os.environ, {
            "CROSS_ROUND_TIE_DIMENSION_LESSONS": "1",
            "CROSS_ROUND_TIE_DIMENSION_MIN_GAIN": "9",
            "TIE_DIMENSION_MIN_GAIN": "99",
        }, clear=True):
            opt = make_optimizer(
                cross_round_tie_dimension_lessons=False,
                cross_round_tie_dimension_min_gain=2.0,
            )
        self.assertFalse(opt.cross_round_tie_dimension_lessons)
        self.assertEqual(opt.cross_round_tie_dimension_min_gain, 2.0)
        self.assertEqual(opt.tie_dimension_min_gain, 99.0)

    def test_invalid_and_negative_minimum_gain_fall_back_to_zero(self):
        with patch.dict(os.environ, {
            "CROSS_ROUND_TIE_DIMENSION_MIN_GAIN": "not-a-number",
        }, clear=True):
            self.assertEqual(make_optimizer().cross_round_tie_dimension_min_gain, 0.0)
        self.assertEqual(
            make_optimizer(cross_round_tie_dimension_min_gain=-3.0)
            .cross_round_tie_dimension_min_gain,
            0.0,
        )

    def test_dimension_gain_boundary_is_strict_and_requires_reference_score(self):
        opt = make_optimizer(cross_round_tie_dimension_min_gain=5.0)
        gains = opt._dimension_advantages(
            {
                "equal-boundary": {"pct": 85.0},
                "over-boundary": {"pct": 85.001},
                "missing-reference": {"pct": 100.0},
                "non-finite": {"pct": float("nan")},
            },
            {
                "equal-boundary": {"pct": 80.0},
                "over-boundary": {"pct": 80.0},
                "non-finite": {"pct": 0.0},
            },
            min_gain=opt.cross_round_tie_dimension_min_gain,
        )
        self.assertNotIn("equal-boundary", gains)
        self.assertEqual(gains["over-boundary"]["gain"], 5.0)
        self.assertNotIn("missing-reference", gains)
        self.assertNotIn("non-finite", gains)

    def test_target_dimension_uses_largest_gain_then_dimension_name(self):
        opt = make_optimizer()
        self.assertEqual(opt._tie_dimension_target({
            "quality": {"gain": 10.0},
            "clarity": {"gain": 20.0},
            "correctness": {"gain": 20.0},
        }), "clarity")

    async def test_disabled_feature_preserves_tied_candidate_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lessons.jsonl")
            opt = make_optimizer(lesson_file=path)
            dims = self._dimensions(clarity=90, correctness=70)
            candidate = self._candidate(0, 80.0, dims)
            result = await self._decide(
                opt,
                incumbent_dimensions=self._dimensions(clarity=70, correctness=90),
                candidates=[candidate],
                rescored=[candidate],
            )
            self.assertEqual(self._tie_lessons(path), [])
        self.assertEqual(result["current_md"], "# Incumbent")
        self.assertEqual(result["baseline_pct"], 80.0)
        self.assertFalse(result["kept"])
        self.assertNotIn("tie_dimension_lessons", result["round_memory"][0])

    async def test_lower_higher_and_tied_without_advantage_do_not_learn(self):
        cases = (
            ("lower", 79.0, self._dimensions(clarity=100), False),
            ("higher", 81.0, self._dimensions(clarity=100), True),
            ("tie-no-advantage", 80.0, self._dimensions(clarity=70), False),
        )
        for name, score, dimensions, expected_kept in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "lessons.jsonl")
                opt = make_optimizer(
                    lesson_file=path,
                    cross_round_tie_dimension_lessons=True,
                )
                candidate = self._candidate(0, score, dimensions)
                result = await self._decide(
                    opt,
                    incumbent_dimensions=self._dimensions(clarity=70),
                    candidates=[candidate],
                    rescored=[candidate],
                )
                self.assertEqual(self._tie_lessons(path, "cross_round"), [])
                self.assertEqual(result["kept"], expected_kept)

    async def test_isclose_total_learns_but_normalizes_total_scores_to_incumbent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lessons.jsonl")
            opt = make_optimizer(
                lesson_file=path,
                cross_round_tie_dimension_lessons=True,
            )
            candidate = self._candidate(
                0, 80.0 - 5e-10, self._dimensions(clarity=90)
            )
            result = await self._decide(
                opt,
                incumbent_dimensions=self._dimensions(clarity=70),
                candidates=[candidate],
                rescored=[candidate],
            )
            lessons = self._tie_lessons(path, "cross_round")
        self.assertEqual(len(lessons), 1)
        self.assertEqual(lessons[0]["score_before"], 80.0)
        self.assertEqual(lessons[0]["score_after"], 80.0)
        self.assertFalse(result["kept"])
        self.assertEqual(result["baseline_pct"], 80.0)

    async def test_latest_kept_incumbent_survives_an_intervening_discarded_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lessons.jsonl")
            opt = make_optimizer(
                lesson_file=path,
                regression_check=False,
                cross_round_tie_dimension_lessons=True,
            )
            scores = {
                "# Initial": self._score(
                    60, dimensions=self._dimensions(clarity=40, correctness=80)
                ),
                "# Latest incumbent": self._score(
                    80, dimensions=self._dimensions(clarity=60, correctness=100)
                ),
                "# Discarded lower": self._score(
                    70, dimensions=self._dimensions(clarity=95, correctness=45)
                ),
                "# Cross-round tie": self._score(
                    80, dimensions=self._dimensions(clarity=90, correctness=70)
                ),
            }
            mutations = [
                {"description": "kept incumbent", "reasoning": "r",
                 "new_skill_md": "# Latest incumbent"},
                {"description": "lower candidate", "reasoning": "r",
                 "new_skill_md": "# Discarded lower"},
                {"description": "tie candidate", "reasoning": "r",
                 "new_skill_md": "# Cross-round tie"},
            ]

            async def score(skill_md, *_args, **_kwargs):
                return scores[skill_md]

            with (
                patch.object(opt, "_score_skill", new=score),
                patch.object(
                    opt, "_analyze_failures",
                    new=AsyncMock(return_value=self._analysis()),
                ),
                patch.object(opt, "_mutate_skill", new=AsyncMock(side_effect=mutations)),
            ):
                result = await opt.optimize(
                    {"SKILL.md": "# Initial"}, [], [], max_rounds=3,
                )

            lessons = self._tie_lessons(path, "cross_round")
        self.assertEqual(result["improved_skill_md"], "# Latest incumbent")
        self.assertEqual(result["final_score"], 80.0)
        self.assertEqual(result["score_history"], [60.0, 80.0, 80.0, 80.0])
        self.assertEqual([entry["kept"] for entry in result["mutation_log"]], [True, False, False])
        self.assertEqual(len(lessons), 1)
        clarity = lessons[0]["dimension_gains"]["clarity"]
        # 参照 60 分 clarity 的最近 kept incumbent，而非初始版本的 40 分。
        self.assertEqual(clarity, {
            "candidate_pct": 90.0,
            "winner_pct": 60.0,
            "gain": 30.0,
        })

    async def test_all_scored_candidates_learn_and_each_candidate_aggregates_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lessons.jsonl")
            opt = make_optimizer(
                lesson_file=path,
                cross_round_tie_dimension_lessons=True,
            )
            incumbent = self._dimensions(clarity=60, correctness=60, quality=60)
            first = self._candidate(
                0, 80.0,
                self._dimensions(clarity=90, correctness=75, quality=60),
                strategy="rewrite_section",
            )
            second = self._candidate(
                1, 80.0,
                self._dimensions(clarity=60, correctness=65, quality=95),
                strategy="add_example",
            )
            result = await self._decide(
                opt,
                incumbent_dimensions=incumbent,
                candidates=[first, second],
                rescored=[first, second],
            )
            lessons = self._tie_lessons(path, "cross_round")
        self.assertEqual(len(lessons), 2)
        by_strategy = {lesson["strategy"]: lesson for lesson in lessons}
        self.assertEqual(
            set(by_strategy["rewrite_section"]["dimension_gains"]),
            {"clarity", "correctness"},
        )
        self.assertEqual(by_strategy["rewrite_section"]["target_dimension"], "clarity")
        self.assertEqual(
            set(by_strategy["add_example"]["dimension_gains"]),
            {"correctness", "quality"},
        )
        self.assertEqual(by_strategy["add_example"]["target_dimension"], "quality")
        self.assertFalse(result["kept"])
        self.assertFalse(any(entry["kept"] for entry in result["mutation_log"]))
        for record in result["round_memory"][0]["tie_dimension_lessons"]:
            self.assertNotIn("candidate_id", record)
        for lesson in lessons:
            self.assertNotIn("candidate_id", lesson)

    async def test_same_round_and_cross_round_paths_deduplicate_candidate_dimension(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lessons.jsonl")
            opt = make_optimizer(
                lesson_file=path,
                tie_dimension_lessons=True,
                cross_round_tie_dimension_lessons=True,
            )
            incumbent = self._dimensions(clarity=70)
            microscopic_winner = self._candidate(
                0, 80.0 + 5e-10, incumbent, strategy="add_example"
            )
            tied_nonwinner = self._candidate(
                1, 80.0, self._dimensions(clarity=90), strategy="rewrite_section"
            )
            await self._decide(
                opt,
                incumbent_dimensions=incumbent,
                candidates=[microscopic_winner, tied_nonwinner],
                rescored=[microscopic_winner, tied_nonwinner],
            )
            lessons = self._tie_lessons(path)
        self.assertEqual(len(lessons), 1)
        self.assertEqual(lessons[0]["comparison_scope"], "same_round")
        self.assertEqual(set(lessons[0]["dimension_gains"]), {"clarity"})

    async def test_guard_edit_limit_and_unscored_candidates_cannot_learn(self):
        # 回归守卫与编辑幅度拒绝都必须发生在评分前。
        rejection_cases = (
            (
                "regression",
                make_optimizer(
                    cross_round_tie_dimension_lessons=True,
                    regression_check=True,
                ),
                "---\nname: safe\n---\n# Broken without frontmatter",
                "# Broken without frontmatter",
            ),
            (
                "edit-limit",
                make_optimizer(
                    cross_round_tie_dimension_lessons=True,
                    regression_check=False,
                    edit_limit=0.1,
                ),
                "# S\n" + "a" * 100,
                "# S\n" + "a" * 100 + "b" * 50,
            ),
        )
        for name, opt, original_md, candidate_md in rejection_cases:
            baseline = self._score(
                80, dimensions=self._dimensions(clarity=70)
            )
            mutation = {
                "description": "must not learn",
                "reasoning": "r",
                "new_skill_md": candidate_md,
            }
            with self.subTest(name=name):
                with (
                    patch.object(
                        opt, "_score_skill", new=AsyncMock(return_value=baseline)
                    ) as scorer,
                    patch.object(
                        opt, "_analyze_failures",
                        new=AsyncMock(return_value=self._analysis()),
                    ),
                    patch.object(
                        opt, "_mutate_skill", new=AsyncMock(return_value=mutation)
                    ),
                ):
                    result = await opt.optimize(
                        {"SKILL.md": original_md}, [], [], max_rounds=1
                    )
                    self.assertEqual(scorer.await_count, 1)
                    self.assertFalse(result["mutation_log"][0]["kept"])

        # guard 的 score_after 只是占位基线；没有出现在 rescored 的候选不算完成评分。
        opt = make_optimizer(cross_round_tie_dimension_lessons=True)
        unscored = self._candidate(0, 80.0, self._dimensions(clarity=100))
        result = await self._decide(
            opt,
            incumbent_dimensions=self._dimensions(clarity=70),
            candidates=[unscored],
            rescored=[],
        )
        self.assertNotIn("tie_dimension_lessons", result["round_memory"][0])
        self.assertFalse(result["kept"])

    async def test_confirmation_failed_candidate_cannot_learn(self):
        opt = make_optimizer(cross_round_tie_dimension_lessons=True)
        candidate = self._candidate(0, 80.0, self._dimensions(clarity=95))
        result = await self._decide(
            opt,
            incumbent_dimensions=self._dimensions(clarity=70),
            candidates=[candidate],
            rescored=[candidate],
            confirmation={
                "candidate_id": 0,
                "passed": False,
                "runs": 1,
                "incumbent_scores": [80.0],
                "challenger_scores": [80.0],
                "confirmed_score": 80.0,
            },
        )
        self.assertNotIn("tie_dimension_lessons", result["round_memory"][0])
        self.assertFalse(result["kept"])
        self.assertEqual(result["mutation_log"][0]["reason"], "confirmation_failed")

    async def test_tied_candidate_does_not_trigger_confirmation_but_still_learns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lessons.jsonl")
            opt = make_optimizer(
                lesson_file=path,
                regression_check=False,
                candidate_confirm_runs=2,
                cross_round_tie_dimension_lessons=True,
            )
            baseline = self._score(
                80, dimensions=self._dimensions(clarity=70)
            )
            tied = self._score(
                80, dimensions=self._dimensions(clarity=90)
            )
            with (
                patch.object(
                    opt, "_score_skill", new=AsyncMock(side_effect=[baseline, tied])
                ) as scorer,
                patch.object(
                    opt, "_analyze_failures",
                    new=AsyncMock(return_value=self._analysis()),
                ),
                patch.object(opt, "_mutate_skill", new=AsyncMock(return_value={
                    "description": "clarity metadata",
                    "reasoning": "r",
                    "new_skill_md": "# Tie",
                })),
            ):
                result = await opt.optimize({"SKILL.md": "# Original"}, [], [], max_rounds=1)
            lessons = self._tie_lessons(path, "cross_round")
        self.assertEqual(scorer.await_count, 2)
        self.assertEqual(len(lessons), 1)
        self.assertFalse(result["mutation_log"][0]["kept"])
        self.assertNotEqual(result["mutation_log"][0]["reason"], "confirmation_failed")

    async def test_tie_lesson_does_not_mutate_incumbent_state(self):
        opt = make_optimizer(cross_round_tie_dimension_lessons=True)
        incumbent_dimensions = self._dimensions(clarity=70, correctness=90)
        candidate = self._candidate(
            0, 80.0, self._dimensions(clarity=90, correctness=70)
        )
        result = await self._decide(
            opt,
            incumbent_dimensions=incumbent_dimensions,
            candidates=[candidate],
            rescored=[candidate],
        )
        self.assertEqual(result["current_md"], "# Incumbent")
        self.assertEqual(result["baseline_pct"], 80.0)
        self.assertEqual(result["current_dimension_scores"], incumbent_dimensions)
        self.assertEqual(result["score_history"], [80.0])
        self.assertFalse(result["mutation_log"][0]["kept"])

    async def test_cross_round_quality_gate_ignores_gain_but_honors_final_waterline(self):
        with tempfile.TemporaryDirectory() as tmp:
            pass_path = os.path.join(tmp, "pass.jsonl")
            blocked_path = os.path.join(tmp, "blocked.jsonl")
            candidate = self._candidate(0, 80.0, self._dimensions(clarity=90))
            incumbent = self._dimensions(clarity=70)

            ignores_gain = make_optimizer(
                lesson_file=pass_path,
                lesson_min_gain=999.0,
                lesson_min_final=0.0,
                cross_round_tie_dimension_lessons=True,
            )
            pass_result = await self._decide(
                ignores_gain,
                incumbent_dimensions=incumbent,
                candidates=[candidate],
                rescored=[candidate],
            )
            self.assertEqual(len(self._tie_lessons(pass_path, "cross_round")), 1)

            at_waterline_path = os.path.join(tmp, "at-waterline.jsonl")
            at_waterline = make_optimizer(
                lesson_file=at_waterline_path,
                lesson_min_gain=999.0,
                lesson_min_final=80.0,
                cross_round_tie_dimension_lessons=True,
            )
            await self._decide(
                at_waterline,
                incumbent_dimensions=incumbent,
                candidates=[candidate],
                rescored=[candidate],
            )
            self.assertEqual(
                len(self._tie_lessons(at_waterline_path, "cross_round")), 1
            )

            below_waterline = make_optimizer(
                lesson_file=blocked_path,
                lesson_min_gain=0.0,
                lesson_min_final=80.001,
                cross_round_tie_dimension_lessons=True,
            )
            blocked_result = await self._decide(
                below_waterline,
                incumbent_dimensions=incumbent,
                candidates=[candidate],
                rescored=[candidate],
            )
            self.assertEqual(self._tie_lessons(blocked_path, "cross_round"), [])

        # 持久化水位不影响当轮 advisory memory。
        self.assertIn("tie_dimension_lessons", pass_result["round_memory"][0])
        self.assertIn("tie_dimension_lessons", blocked_result["round_memory"][0])

    def test_jsonl_round_trip_and_legacy_scope_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lessons.jsonl")
            SkillOptimizer._append_lesson(path, {
                "strategy": "legacy",
                "lesson_type": "tie_dimension",
                "dimension_gains": {"clarity": {"gain": 10.0}},
            })
            SkillOptimizer._append_lesson(path, {
                "strategy": "cross",
                "lesson_type": "tie_dimension",
                "comparison_scope": "cross_round",
                "target_dimension": "quality",
                "dimension_gains": {"quality": {"gain": 20.0}},
            })
            legacy, cross = SkillOptimizer._load_lessons(path, limit=10)
        self.assertEqual(legacy["comparison_scope"], "same_round")
        self.assertEqual(cross["comparison_scope"], "cross_round")
        self.assertEqual(cross["target_dimension"], "quality")

    def test_sqlite_legacy_schema_migrates_and_round_trips_scope(self):
        import sqlite3

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lessons.db")
            conn = sqlite3.connect(path)
            try:
                conn.execute(
                    "CREATE TABLE lessons ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "skill_name TEXT, domain TEXT, skill_description TEXT,"
                    "strategy TEXT, diagnosis TEXT, summary TEXT,"
                    "score_before REAL, score_after REAL, created_at TEXT,"
                    "embedding BLOB, lesson_type TEXT, dimension_gains TEXT,"
                    "target_dimension TEXT)"
                )
                conn.execute(
                    "INSERT INTO lessons (strategy, lesson_type, dimension_gains, "
                    "target_dimension) VALUES (?, ?, ?, ?)",
                    (
                        "legacy", "tie_dimension",
                        json.dumps({"clarity": {"gain": 10.0}}), "clarity",
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            SkillOptimizer._append_lesson(path, {
                "strategy": "cross",
                "lesson_type": "tie_dimension",
                "comparison_scope": "cross_round",
                "target_dimension": "quality",
                "dimension_gains": {"quality": {"gain": 20.0}},
            })
            loaded = SkillOptimizer._load_lessons(path, limit=10)
            conn = sqlite3.connect(path)
            try:
                columns = {
                    row[1] for row in conn.execute("PRAGMA table_info(lessons)")
                }
            finally:
                conn.close()
        self.assertIn("comparison_scope", columns)
        self.assertEqual(loaded[0]["comparison_scope"], "same_round")
        self.assertEqual(loaded[1]["comparison_scope"], "cross_round")
        self.assertEqual(loaded[1]["dimension_gains"]["quality"]["gain"], 20.0)

    async def test_cross_round_dimension_metadata_is_a_weak_dimension_signal(self):
        opt = make_optimizer(
            lesson_rag_pipeline="quality_diverse",
            lesson_signal_weights={"semantic": 1, "dimension": 5},
        )
        lessons = [
            {
                "strategy": "cross-clarity",
                "summary": "rewrite guidance",
                "lesson_type": "tie_dimension",
                "comparison_scope": "cross_round",
                "target_dimension": "clarity",
                "dimension_gains": {"clarity": {"gain": 20.0}},
                "embedding": [1.0, 0.0],
            },
            {
                "strategy": "same-correctness",
                "summary": "rewrite guidance",
                "lesson_type": "tie_dimension",
                "comparison_scope": "same_round",
                "target_dimension": "correctness",
                "dimension_gains": {"correctness": {"gain": 20.0}},
                "embedding": [1.0, 0.0],
            },
        ]
        context = {"weak_dimensions": [{"dimension": "clarity", "pct": 25.0}]}
        with patch.object(opt, "_embed", new=AsyncMock(return_value=[1.0, 0.0])):
            top = await opt._retrieve_lessons(
                lessons, "rewrite guidance", "hybrid", 2, context=context,
            )
        self.assertEqual(top[0]["strategy"], "cross-clarity")
        self.assertIn("weak_dimension", top[0]["_retrieval"]["matched"])

    def test_prompt_whitelist_keeps_scope_but_drops_internal_and_raw_fields(self):
        opt = make_optimizer(lesson_rag_pipeline="quality_diverse")
        compact = opt._compact_lessons_for_prompt([{
            "strategy": "rewrite_section",
            "summary": "clarify workflow",
            "lesson_type": "tie_dimension",
            "comparison_scope": "cross_round",
            "target_dimension": "clarity",
            "dimension_gains": {"clarity": {"gain": 20.0}},
            "_id": 42,
            "embedding": [1.0, 0.0],
            "candidate_id": 7,
            "skill_md": "PRIVATE SKILL BODY",
            "scenarios": ["PRIVATE SCENARIO"],
            "output": "PRIVATE OUTPUT",
            "api_key": "PRIVATE KEY",
        }])[0]
        self.assertEqual(compact["comparison_scope"], "cross_round")
        for forbidden in (
            "_id", "embedding", "candidate_id", "skill_md", "scenarios", "output", "api_key"
        ):
            self.assertNotIn(forbidden, compact)

    async def test_cross_round_mode_refreshes_rag_each_round_with_latest_incumbent(self):
        opt = make_optimizer(
            regression_check=False,
            parallelism=2,
            cross_round_tie_dimension_lessons=True,
        )
        scores = {
            "# Original": self._score(
                80, dimensions=self._dimensions(clarity=70, correctness=90),
                details=[{"eval_id": 1, "passed": False, "reason": "original"}],
            ),
            "# Winner": self._score(
                90, dimensions=self._dimensions(clarity=80, correctness=100),
                details=[{"eval_id": 1, "passed": False, "reason": "winner"}],
            ),
            "# Historical tie": self._score(
                80, dimensions=self._dimensions(clarity=95, correctness=65)
            ),
            "# Low A": self._score(70, dimensions=self._dimensions(clarity=70)),
            "# Low B": self._score(60, dimensions=self._dimensions(clarity=60)),
        }
        mutations = [
            {"description": "winner", "reasoning": "r", "new_skill_md": "# Winner"},
            {"description": "cross tie", "reasoning": "r", "new_skill_md": "# Historical tie"},
            {"description": "low a", "reasoning": "r", "new_skill_md": "# Low A"},
            {"description": "low b", "reasoning": "r", "new_skill_md": "# Low B"},
        ]
        recalled = {
            "strategy": "rewrite_section",
            "summary": "round-one cross tie",
            "lesson_type": "tie_dimension",
            "comparison_scope": "cross_round",
            "target_dimension": "clarity",
            "dimension_gains": {"clarity": {"gain": 25.0}},
        }
        prepared = AsyncMock(side_effect=[[], [recalled]])
        analyst_lessons = []

        async def score(skill_md, *_args, **_kwargs):
            return scores[skill_md]

        async def analyze(*_args, **kwargs):
            analyst_lessons.append(kwargs.get("lessons", []))
            return self._analysis()

        with (
            patch.object(opt, "_score_skill", new=score),
            patch.object(opt, "_prepare_lessons", new=prepared),
            patch.object(opt, "_analyze_failures", new=analyze),
            patch.object(opt, "_mutate_skill", new=AsyncMock(side_effect=mutations)),
        ):
            result = await opt.optimize(
                {"SKILL.md": "# Original"}, [], [], max_rounds=2,
                parallel_mutations=2,
            )

        self.assertEqual(prepared.await_count, 2)
        self.assertEqual(prepared.await_args_list[0].args[0], "# Original")
        self.assertEqual(prepared.await_args_list[1].args[0], "# Winner")
        self.assertEqual(analyst_lessons, [[], [recalled]])
        self.assertEqual(result["improved_skill_md"], "# Winner")

    async def test_disabled_mode_does_not_add_per_round_rag_refresh(self):
        opt = make_optimizer(regression_check=False)
        baseline = self._score(80, dimensions=self._dimensions(clarity=70))
        same = self._score(80, dimensions=self._dimensions(clarity=90))
        prepare = AsyncMock(return_value=[])
        with (
            patch.object(
                opt, "_score_skill", new=AsyncMock(side_effect=[baseline, same, same])
            ),
            patch.object(opt, "_prepare_lessons", new=prepare),
            patch.object(
                opt, "_analyze_failures", new=AsyncMock(return_value=self._analysis())
            ),
            patch.object(opt, "_mutate_skill", new=AsyncMock(return_value={
                "description": "same", "reasoning": "r", "new_skill_md": "# Same",
            })),
        ):
            await opt.optimize({"SKILL.md": "# Original"}, [], [], max_rounds=2)
        self.assertEqual(prepare.await_count, 1)

    async def test_cooperative_stop_runs_before_next_round_rag_refresh(self):
        stop_values = iter([False, False, True])
        opt = make_optimizer(
            regression_check=False,
            cross_round_tie_dimension_lessons=True,
            stop_provider=lambda: next(stop_values),
        )
        baseline = self._score(80, dimensions=self._dimensions(clarity=70))
        tied = self._score(80, dimensions=self._dimensions(clarity=90))
        prepare = AsyncMock(return_value=[])
        with (
            patch.object(opt, "_score_skill", new=AsyncMock(side_effect=[baseline, tied])),
            patch.object(opt, "_prepare_lessons", new=prepare),
            patch.object(
                opt, "_analyze_failures", new=AsyncMock(return_value=self._analysis())
            ),
            patch.object(opt, "_mutate_skill", new=AsyncMock(return_value={
                "description": "tie", "reasoning": "r", "new_skill_md": "# Tie",
            })),
        ):
            with self.assertRaises(StopOptimizationError):
                await opt.optimize({"SKILL.md": "# Original"}, [], [], max_rounds=2)
        self.assertEqual(prepare.await_count, 1)

    async def test_persistence_excludes_skill_scenarios_outputs_and_credentials(self):
        raw_skill = "PRIVATE_SKILL_BODY"
        raw_scenario = "PRIVATE_SCENARIO_INPUT"
        raw_output = "PRIVATE_EXECUTOR_OUTPUT"
        secret = "sk-" + "A" * 32
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lessons.jsonl")
            original_md = f"---\nname: safe-skill\n---\n# Skill\n{raw_skill}"
            candidate_md = original_md + "\nPRIVATE_CANDIDATE_BODY"
            opt = make_optimizer(
                api_key=secret,
                lesson_file=path,
                regression_check=False,
                cross_round_tie_dimension_lessons=True,
            )
            baseline = self._score(
                80,
                dimensions=self._dimensions(clarity=70),
                details=[{"output": raw_output, "reason": "baseline detail"}],
            )
            tied = self._score(
                80,
                dimensions=self._dimensions(clarity=90),
                details=[{"output": raw_output, "reason": "candidate detail"}],
            )
            with (
                patch.object(opt, "_score_skill", new=AsyncMock(side_effect=[baseline, tied])),
                patch.object(
                    opt, "_analyze_failures", new=AsyncMock(return_value=self._analysis())
                ),
                patch.object(opt, "_mutate_skill", new=AsyncMock(return_value={
                    "description": "clarity improved",
                    "reasoning": "model metadata",
                    "new_skill_md": candidate_md,
                })),
            ):
                result = await opt.optimize(
                    {"SKILL.md": original_md},
                    [{"id": 1, "input": raw_scenario}],
                    [{"id": 1, "question": "PRIVATE_EVAL_TEXT"}],
                    max_rounds=1,
                )
            with open(path, encoding="utf-8") as lesson_file:
                persisted = lesson_file.read()

        self.assertFalse(result["mutation_log"][0]["kept"])
        for forbidden in (
            raw_skill,
            "PRIVATE_CANDIDATE_BODY",
            raw_scenario,
            "PRIVATE_EVAL_TEXT",
            raw_output,
            secret,
        ):
            self.assertNotIn(forbidden, persisted)


class TestWeakDimensionFocus(unittest.IsolatedAsyncioTestCase):
    """目标 4：识别低分维度，并把它贯穿诊断、策略与单点变异。"""

    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            opt = make_optimizer()
        self.assertFalse(opt.weak_dimension_focus)
        self.assertEqual(opt._identify_weak_dimensions({
            "clarity": {"passed": 0, "total": 2, "pct": 0.0},
        }), [])

    def test_identifies_lowest_measured_dimensions_only(self):
        opt = make_optimizer(
            weak_dimension_focus=True, weak_dimension_threshold=50, weak_dimension_max=2,
        )
        weak = opt._identify_weak_dimensions({
            "clarity": {"passed": 1, "total": 4, "pct": 25.0},
            "correctness": {"passed": 2, "total": 4, "pct": 50.0},
            "quality": {"passed": 0, "total": 0, "pct": 0.0},
            "executability": {"passed": 3, "total": 4, "pct": 75.0},
        })
        self.assertEqual([d["dimension"] for d in weak], ["clarity", "correctness"])

    def test_focused_strategy_pool_respects_allowed_whitelist(self):
        focused = SkillOptimizer._focused_strategy_pool(
            ["add_example", "rewrite_section", "add_constraint"],
            [{"dimension": "clarity", "pct": 20}],
        )
        self.assertEqual(focused, ["rewrite_section", "add_example", "add_constraint"])

    async def test_analyst_and_mutator_receive_explicit_dimension_target(self):
        opt = make_optimizer(weak_dimension_focus=True)
        weak = [{"dimension": "clarity", "pct": 25.0, "passed": 1, "total": 4}]
        evals = [
            {"id": 1, "dimension": "clarity", "question": "Is it clear?"},
            {"id": 2, "dimension": "correctness", "question": "Is it correct?"},
        ]
        details = [
            {"eval_id": 1, "passed": False, "reason": "ambiguous"},
            {"eval_id": 2, "passed": False, "reason": "wrong"},
        ]
        prompts = []
        analysis = {
            "diagnosis": "ambiguous steps", "mutation_strategy": "rewrite_section",
            "target_section": "Workflow", "suggested_change": "clarify one step",
            "target_dimension": "clarity",
        }

        async def fake_ask_json(_agent, prompt, **_kwargs):
            prompts.append(prompt)
            return analysis if "Diagnose these failures" in prompt else {
                "description": "clarified one step", "reasoning": "r", "new_skill_md": "# S clarified",
            }

        with patch.object(opt, "_ask_json", new=fake_ask_json):
            result = await opt._analyze_failures(
                "# S", [], evals, details, weak_dimensions=weak,
            )
            await opt._mutate_skill("# S", result)

        self.assertIn("Priority weak dimensions", prompts[0])
        self.assertIn("clarity", prompts[0])
        self.assertIn("ambiguous", prompts[0])
        self.assertIn("Target evaluation dimension: clarity", prompts[1])
        self.assertIn("ONE edit", prompts[1])

    async def test_graph_passes_baseline_weakness_to_analyst(self):
        opt = make_optimizer(weak_dimension_focus=True, weak_dimension_threshold=50)
        baseline = {
            "passed": 1, "total": 2, "per_eval": [],
            "details": [{"eval_id": 1, "passed": False}],
            "dimension_scores": {
                "clarity": {"passed": 0, "total": 1, "pct": 0.0},
                "correctness": {"passed": 1, "total": 1, "pct": 100.0},
            },
        }
        same = {**baseline, "passed": 1, "total": 2}
        captured = {}

        async def analyze(*_args, **kwargs):
            captured["weak"] = kwargs.get("weak_dimensions")
            return {"diagnosis": "d", "mutation_strategy": "rewrite_section",
                    "target_section": "s", "suggested_change": "c",
                    "target_dimension": "clarity"}

        mutation = {"description": "m", "reasoning": "r", "new_skill_md": "# New"}
        with (
            patch.object(opt, "_score_skill", new=AsyncMock(side_effect=[baseline, same])),
            patch.object(opt, "_analyze_failures", new=analyze),
            patch.object(opt, "_mutate_skill", new=AsyncMock(return_value=mutation)),
        ):
            await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=1)

        self.assertEqual(captured["weak"][0]["dimension"], "clarity")


class TestWeightedScoring(unittest.IsolatedAsyncioTestCase):
    """C2: dimension-weighted scoring; no weights == old pass-rate math."""

    async def test_no_weights_equals_pass_rate(self):
        opt = make_optimizer()
        evals = [
            {"id": 1, "question": "q1", "dimension": "correctness"},
            {"id": 2, "question": "q2", "dimension": "clarity"},
        ]
        scenarios = [{"id": 1, "input": "x"}]
        skill_md = "# S"
        # Fake the two executor calls: execute output + scoring JSON.
        with (
            patch.object(opt, "_ask", new=AsyncMock(return_value="some output")),
            patch.object(opt, "_ask_json", new=AsyncMock(return_value={"results": [
                {"eval_id": 1, "passed": True},
                {"eval_id": 2, "passed": False},
            ]})),
        ):
            result = await opt._score_skill(skill_md, scenarios, evals)
        self.assertEqual(result["passed"], 1)
        self.assertEqual(result["total"], 2)
        # 无权重：weighted_pct 与通过率完全一致。
        self.assertEqual(result["weighted_pct"], 50.0)
        self.assertEqual(result["dimension_scores"]["correctness"]["pct"], 100.0)
        self.assertEqual(result["dimension_scores"]["clarity"]["pct"], 0.0)

    async def test_weights_change_weighted_pct(self):
        opt = make_optimizer()
        opt.dimension_weights = {"correctness": 0.8, "clarity": 0.2}
        evals = [
            {"id": 1, "question": "q1", "dimension": "correctness"},
            {"id": 2, "question": "q2", "dimension": "clarity"},
        ]
        scenarios = [{"id": 1, "input": "x"}]
        with (
            patch.object(opt, "_ask", new=AsyncMock(return_value="some output")),
            patch.object(opt, "_ask_json", new=AsyncMock(return_value={"results": [
                {"eval_id": 1, "passed": True},
                {"eval_id": 2, "passed": False},
            ]})),
        ):
            result = await opt._score_skill("# S", scenarios, evals)
        # weighted = (0.8*1.0 + 0.2*0.0) / 1.0 = 0.8 -> 80%
        self.assertEqual(result["weighted_pct"], 80.0)

    async def test_optimize_uses_weighted_decision_when_configured(self):
        opt = make_optimizer()
        opt.dimension_weights = {"correctness": 1.0}
        baseline = {"passed": 2, "total": 4, "per_eval": [], "details": [], "weighted_pct": 50.0}
        improved = {"passed": 3, "total": 4, "per_eval": [], "details": [], "weighted_pct": 75.0}
        analysis = {"diagnosis": "d", "mutation_strategy": "add_example",
                    "target_section": "s", "suggested_change": "c"}
        mutation = {"description": "m", "reasoning": "r", "new_skill_md": "# New"}
        with (
            patch.object(opt, "_score_skill", new=AsyncMock(side_effect=[baseline, improved])) as mock_score,
            patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=analysis)),
            patch.object(opt, "_mutate_skill", new=AsyncMock(return_value=mutation)),
        ):
            result = await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=1)
        self.assertEqual(result["final_score"], 75.0)
        self.assertEqual(mock_score.await_count, 2)


class TestParallelMutations(unittest.IsolatedAsyncioTestCase):
    """C3: parallel mutations per round; only best strictly-improving kept."""

    async def test_parallel_keeps_only_best_improving(self):
        opt = make_optimizer()
        # 强制并行 2。
        opt.mutator_pool = [opt.mutator, opt.mutator]
        opt.executor_pool = [opt.executor, opt.executor]
        baseline = {"passed": 2, "total": 4, "per_eval": [], "details": []}
        # 两个候选：候选 0（# A）得 75%（提升但不最优），候选 1（# B）得 100%（最优）。
        res_bad = {"passed": 3, "total": 4, "per_eval": [], "details": []}
        res_good = {"passed": 4, "total": 4, "per_eval": [], "details": []}
        # 复评顺序：score_skill 第 1 次 = 基线；随后并行复评 2 个候选。
        # 通过 side_effect 按顺序返回，其中第 3 次是 50%，第 4 次是 75%。
        analysis = {"diagnosis": "d", "mutation_strategy": "add_example",
                    "target_section": "s", "suggested_change": "c"}
        mutation_50 = {"description": "m1", "reasoning": "r1", "new_skill_md": "# A"}
        mutation_75 = {"description": "m2", "reasoning": "r2", "new_skill_md": "# B"}

        # 75% 候选必须排在 50% 候选后面，验证"取最优"。
        def score_side_effect(skill_md, scenarios, evals, **kwargs):
            if skill_md == "# A":
                return res_bad
            if skill_md == "# B":
                return res_good
            return baseline

        async def score_side_effect_async(*args, **kwargs):
            return score_side_effect(args[0], args[1], args[2])

        with (
            patch.object(opt, "_score_skill", new=AsyncMock(side_effect=score_side_effect_async)),
            patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=analysis)),
            patch.object(opt, "_mutate_skill", new=AsyncMock(side_effect=[mutation_50, mutation_75])),
        ):
            result = await opt.optimize(
                {"SKILL.md": "# S"}, [], [], max_rounds=1, parallel_mutations=2
            )

        # 两个候选都记入 mutation_log；只有最优者 kept。
        self.assertEqual(len(result["mutation_log"]), 2)
        kept_entries = [m for m in result["mutation_log"] if m.get("kept")]
        self.assertEqual(len(kept_entries), 1)
        self.assertEqual(kept_entries[0]["description"], "m2")
        self.assertEqual(kept_entries[0]["reason"], "best")
        self.assertEqual(result["final_score"], 100.0)

    async def test_parallel_none_improve_keeps_nothing(self):
        opt = make_optimizer()
        opt.mutator_pool = [opt.mutator, opt.mutator]
        opt.executor_pool = [opt.executor, opt.executor]
        baseline = {"passed": 3, "total": 4, "per_eval": [], "details": []}
        same = {"passed": 3, "total": 4, "per_eval": [], "details": []}
        analysis = {"diagnosis": "d", "mutation_strategy": "add_constraint",
                    "target_section": "s", "suggested_change": "c"}
        mutation = {"description": "m", "reasoning": "r", "new_skill_md": "# Same"}

        async def score_side_effect(*args, **kwargs):
            return same

        with (
            patch.object(opt, "_score_skill", new=AsyncMock(side_effect=score_side_effect)),
            patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=analysis)),
            patch.object(opt, "_mutate_skill", new=AsyncMock(return_value=mutation)),
        ):
            result = await opt.optimize(
                {"SKILL.md": "# S"}, [], [], max_rounds=1, parallel_mutations=2
            )

        # 分数未严格提升 → 全部丢弃。
        self.assertEqual(result["final_score"], 75.0)
        self.assertFalse(any(m.get("kept") for m in result["mutation_log"]))


class TestRegressionGuard(unittest.IsolatedAsyncioTestCase):
    """C4: regression checks reject destructive mutations before re-scoring."""

    def test_missing_frontmatter_rejected(self):
        ok, reason = SkillOptimizer._regression_check("---\nname: x\n---\n# S", "# No frontmatter")
        self.assertFalse(ok)
        self.assertIn("regression:", reason)

    def test_name_field_missing_rejected(self):
        ok, reason = SkillOptimizer._regression_check(
            "---\nname: My Skill\nversion: 1.0\n---\n# S", "---\nversion: 1.0\n---\n# S"
        )
        self.assertFalse(ok)
        self.assertIn("regression:", reason)

    def test_heading_removal_rejected(self):
        orig = "---\nname: x\n---\n# A\n## B\n## C\n## D"
        new = "---\nname: x\n---\n# A"
        ok, reason = SkillOptimizer._regression_check(orig, new)
        self.assertFalse(ok)
        self.assertIn("regression:", reason)

    def test_shrink_rejected(self):
        orig = "---\nname: x\n---\n" + "x" * 100
        new = "---\nname: x\n---\nshort"
        ok, reason = SkillOptimizer._regression_check(orig, new)
        self.assertFalse(ok)
        self.assertIn("regression:", reason)

    def test_clean_mutation_passes(self):
        orig = "---\nname: x\n---\n# A\n## B\nsome body text here"
        new = "---\nname: x\n---\n# A\n## B\nsome body text here and more"
        ok, reason = SkillOptimizer._regression_check(orig, new)
        self.assertTrue(ok)

    async def test_regression_rejection_skips_rescore(self):
        opt = make_optimizer()
        baseline = {"passed": 2, "total": 4, "per_eval": [], "details": []}
        analysis = {"diagnosis": "d", "mutation_strategy": "restructure",
                    "target_section": "s", "suggested_change": "c"}
        # 变异丢掉了 frontmatter —— 回归守卫应拒绝且不触发复评。
        mutation = {"description": "m", "reasoning": "r",
                    "new_skill_md": "# Broken (no frontmatter)"}
        with (
            patch.object(opt, "_score_skill", new=AsyncMock(return_value=baseline)) as mock_score,
            patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=analysis)),
            patch.object(opt, "_mutate_skill", new=AsyncMock(return_value=mutation)),
        ):
            result = await opt.optimize({"SKILL.md": "---\nname: x\n---\n# S"}, [], [], max_rounds=1)
        # 基线一次评分 + 无复评（被回归拒绝）→ 共 1 次。
        self.assertEqual(mock_score.await_count, 1)
        self.assertFalse(any(m.get("kept") for m in result["mutation_log"]))
        self.assertIn("regression", result["mutation_log"][0]["reason"])


class TestImprovementThreshold(unittest.IsolatedAsyncioTestCase):
    """C4: improvement_threshold gates keeping mutations."""

    async def test_threshold_requires_strictly_higher_gain(self):
        opt = make_optimizer()
        opt.improvement_threshold = 10.0
        baseline = {"passed": 2, "total": 4, "per_eval": [], "details": []}
        # 75% vs 基线 50%：提升 25% > 阈值 10% → 保留。
        improved = {"passed": 3, "total": 4, "per_eval": [], "details": []}
        analysis = {"diagnosis": "d", "mutation_strategy": "add_example",
                    "target_section": "s", "suggested_change": "c"}
        mutation = {"description": "m", "reasoning": "r", "new_skill_md": "# New"}
        with (
            patch.object(opt, "_score_skill", new=AsyncMock(side_effect=[baseline, improved])),
            patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=analysis)),
            patch.object(opt, "_mutate_skill", new=AsyncMock(return_value=mutation)),
        ):
            result = await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=1)
        self.assertTrue(any(m.get("kept") for m in result["mutation_log"]))
        self.assertEqual(result["final_score"], 75.0)

    async def test_threshold_blocks_small_gains(self):
        opt = make_optimizer()
        opt.improvement_threshold = 10.0
        baseline = {"passed": 3, "total": 4, "per_eval": [], "details": []}
        # 75% vs 基线 75%：无提升，未过阈值 → 丢弃。
        same = {"passed": 3, "total": 4, "per_eval": [], "details": []}
        analysis = {"diagnosis": "d", "mutation_strategy": "add_example",
                    "target_section": "s", "suggested_change": "c"}
        mutation = {"description": "m", "reasoning": "r", "new_skill_md": "# Same"}
        with (
            patch.object(opt, "_score_skill", new=AsyncMock(side_effect=[baseline, same])),
            patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=analysis)),
            patch.object(opt, "_mutate_skill", new=AsyncMock(return_value=mutation)),
        ):
            result = await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=1)
        self.assertFalse(any(m.get("kept") for m in result["mutation_log"]))


class TestImprovementThresholdWiring(unittest.IsolatedAsyncioTestCase):
    """P0: /api/start passes improvement_threshold through to SkillOptimizer.

    修复断链：此前构造 SkillOptimizer 只传 api_key，前端配置的阈值从未生效。
    """

    async def _run_start(self, threshold):
        import app as app_module

        sid = "wiring-session"
        app_module.sessions[sid] = {
            "skill_files": {"SKILL.md": "# S"},
            "scenarios": [{"id": 1, "input": "x"}],
            "evals": [{"id": 1, "criterion": "c"}],
            "original_skill_md": "# S",
        }
        fake = AsyncMock()
        fake.optimize = AsyncMock(return_value={
            "baseline_score": 50.0,
            "final_score": 50.0,
            "improved_skill_md": "# S",
            "score_history": [50.0],
            "mutation_log": [],
        })
        try:
            with patch.object(app_module, "SkillOptimizer", return_value=fake) as cls_mock:
                req = StartRequest(qwen_api_key="sk-test-1", improvement_threshold=threshold)
                await app_module.start_optimization(sid, req)
                # 后台任务在 create_task 中异步运行，让出事件循环使其跑到构造处。
                for _ in range(5):
                    await asyncio.sleep(0)
            return cls_mock
        finally:
            del app_module.sessions[sid]

    async def test_threshold_passed_to_optimizer_constructor(self):
        cls_mock = await self._run_start(5.0)
        self.assertEqual(cls_mock.call_args.kwargs["api_key"], "sk-test-1")
        self.assertEqual(cls_mock.call_args.kwargs["improvement_threshold"], 5.0)

    async def test_threshold_none_is_preserved(self):
        cls_mock = await self._run_start(None)
        self.assertIsNone(cls_mock.call_args.kwargs["improvement_threshold"])


class TestAnalyzeSkillDimensions(unittest.IsolatedAsyncioTestCase):
    """P0: analyze_skill fills missing eval dimension so weighted scoring works."""

    async def test_missing_dimension_filled_with_correctness(self):
        opt = make_optimizer()
        raw = {
            "scenarios": [{"id": 1, "name": "s", "description": "s", "input": "u"}],
            "evals": [{"id": 1, "name": "n", "criterion": "c"}],
        }
        with patch.object(opt, "_ask_json", new=AsyncMock(return_value=raw)):
            result = await opt.analyze_skill({"SKILL.md": "# S"})
        self.assertEqual(result["evals"][0]["dimension"], "correctness")

    async def test_existing_dimension_kept(self):
        opt = make_optimizer()
        raw = {"scenarios": [], "evals": [{"id": 1, "dimension": "clarity"}]}
        with patch.object(opt, "_ask_json", new=AsyncMock(return_value=raw)):
            result = await opt.analyze_skill({"SKILL.md": "# S"})
        self.assertEqual(result["evals"][0]["dimension"], "clarity")

    async def test_prompt_instructs_dimension_field(self):
        opt = make_optimizer()
        captured = {}

        async def fake_ask(agent, prompt):
            captured["prompt"] = prompt
            return {"scenarios": [], "evals": []}

        with patch.object(opt, "_ask_json", new=fake_ask):
            await opt.analyze_skill({"SKILL.md": "# S"})
        self.assertIn("dimension", captured["prompt"])


class TestRuleBasedScoring(unittest.IsolatedAsyncioTestCase):
    """P1: check_type evals are judged by Python rules, not the LLM."""

    def test_keyword_rule(self):
        self.assertEqual(
            SkillOptimizer._rule_check({"check_type": "keyword", "keywords": ["TODO", "abc"]}, "has TODO and abc"),
            (True, "keyword:ok"),
        )
        self.assertEqual(
            SkillOptimizer._rule_check({"check_type": "keyword", "keywords": ["xyz"]}, "has TODO"),
            (False, "keyword:missing"),
        )

    def test_regex_rule(self):
        self.assertEqual(
            SkillOptimizer._rule_check(
                {"check_type": "regex", "pattern": r"[Vv]ersion\s+\d+\.\d+", "flags": "i"},
                "Version 1.2 here",
            ),
            (True, "regex:match"),
        )
        self.assertEqual(
            SkillOptimizer._rule_check({"check_type": "regex", "pattern": r"^\d+$"}, "abc"),
            (False, "regex:no_match"),
        )
        # 非法正则保守判失败，不抛异常。
        self.assertEqual(
            SkillOptimizer._rule_check({"check_type": "regex", "pattern": "("}, "x"),
            (False, "regex:no_match"),
        )

    def test_yaml_or_json_rule(self):
        self.assertEqual(
            SkillOptimizer._rule_check({"check_type": "yaml_or_json", "required_key": "name"}, '{"name": "x"}'),
            (True, "format:valid"),
        )
        self.assertEqual(
            SkillOptimizer._rule_check({"check_type": "yaml_or_json", "required_key": "name"}, "not json"),
            (False, "format:invalid"),
        )

    def test_length_rule(self):
        self.assertEqual(
            SkillOptimizer._rule_check({"check_type": "length", "min_length": 3, "max_length": 5}, "abcd"),
            (True, "length:4"),
        )
        self.assertEqual(
            SkillOptimizer._rule_check({"check_type": "length", "min_length": 10}, "abcd"),
            (False, "length:4"),
        )

    def test_unknown_check_type_is_conservative(self):
        self.assertEqual(
            SkillOptimizer._rule_check({"check_type": "bogus"}, "x"),
            (False, "unknown_check_type:bogus"),
        )

    async def test_score_skill_mixes_rule_and_llm_evals(self):
        opt = make_optimizer()
        evals = [
            {"id": 1, "check_type": "keyword", "keywords": ["TODO"], "dimension": "correctness"},
            {"id": 2, "question": "q2", "dimension": "clarity"},
        ]
        scenarios = [{"id": 1, "input": "x"}]
        # 分别 mock 执行与打分：规则型 eval 不进 LLM 打分调用。
        with (
            patch.object(opt, "_ask", new=AsyncMock(return_value="output has TODO")),
            patch.object(opt, "_ask_json", new=AsyncMock(return_value={
                "results": [{"eval_id": 2, "passed": True}],
            })) as mock_json,
        ):
            result = await opt._score_skill("# S", scenarios, evals)
        self.assertEqual(result["passed"], 2)  # 规则型 1 + LLM 型 1
        self.assertEqual(result["total"], 2)
        # 规则型的 reason 是机器文本，Analyst 也能读到。
        rule_detail = [d for d in result["details"] if d["eval_id"] == 1][0]
        self.assertEqual(rule_detail["reason"], "keyword:ok")
        # LLM 打分调用只收到 LLM 型 eval（规则型不占调用）。
        prompt = mock_json.call_args[0][1]
        self.assertIn('"id": 2', prompt)
        self.assertNotIn("check_type", prompt)  # 规则型 eval 不出现在 Criteria 里

    async def test_no_check_type_keeps_old_call_sequence(self):
        opt = make_optimizer()
        evals = [
            {"id": 1, "question": "q1", "dimension": "correctness"},
            {"id": 2, "question": "q2", "dimension": "clarity"},
        ]
        scenarios = [{"id": 1, "input": "x"}]
        with (
            patch.object(opt, "_ask", new=AsyncMock(return_value="some output")),
            patch.object(opt, "_ask_json", new=AsyncMock(return_value={"results": [
                {"eval_id": 1, "passed": True},
                {"eval_id": 2, "passed": False},
            ]})),
        ):
            result = await opt._score_skill("# S", scenarios, evals)
        # 与旧行为一致：全走 LLM 打分，passed 来自 mock 返回的完整结果。
        self.assertEqual(result["passed"], 1)
        self.assertEqual(result["total"], 2)


class TestNoiseFloor(unittest.IsolatedAsyncioTestCase):
    """P1: noise_floor raises the bar beyond plain strict improvement."""

    async def test_noise_floor_blocks_small_gain(self):
        opt = make_optimizer()
        opt.noise_floor = 20.0
        baseline = {"passed": 4, "total": 8, "per_eval": [], "details": []}       # 50%
        improved = {"passed": 5, "total": 8, "per_eval": [], "details": []}       # 62.5%，提升 12.5 < 20
        analysis = {"diagnosis": "d", "mutation_strategy": "add_example",
                    "target_section": "s", "suggested_change": "c"}
        mutation = {"description": "m", "reasoning": "r", "new_skill_md": "# New"}
        with (
            patch.object(opt, "_score_skill", new=AsyncMock(side_effect=[baseline, improved])),
            patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=analysis)),
            patch.object(opt, "_mutate_skill", new=AsyncMock(return_value=mutation)),
        ):
            result = await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=1)
        self.assertFalse(any(m.get("kept") for m in result["mutation_log"]))

    async def test_noise_floor_allows_big_gain(self):
        opt = make_optimizer()
        opt.noise_floor = 10.0
        baseline = {"passed": 4, "total": 8, "per_eval": [], "details": []}       # 50%
        improved = {"passed": 6, "total": 8, "per_eval": [], "details": []}       # 75%，提升 25 > 10
        analysis = {"diagnosis": "d", "mutation_strategy": "add_example",
                    "target_section": "s", "suggested_change": "c"}
        mutation = {"description": "m", "reasoning": "r", "new_skill_md": "# New"}
        with (
            patch.object(opt, "_score_skill", new=AsyncMock(side_effect=[baseline, improved])),
            patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=analysis)),
            patch.object(opt, "_mutate_skill", new=AsyncMock(return_value=mutation)),
        ):
            result = await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=1)
        self.assertTrue(any(m.get("kept") for m in result["mutation_log"]))


class TestEditLimit(unittest.IsolatedAsyncioTestCase):
    """P1: edit_limit rejects oversized single mutations before re-scoring."""

    async def test_oversized_edit_rejected_before_rescore(self):
        opt = make_optimizer()
        opt.edit_limit = 0.5
        current_md = "# S\n" + "a" * 100
        big_md = current_md + "b" * 100  # 变化 ~99% > 50%
        baseline = {"passed": 2, "total": 4, "per_eval": [], "details": []}
        analysis = {"diagnosis": "d", "mutation_strategy": "add_example",
                    "target_section": "s", "suggested_change": "c"}
        mutation = {"description": "m", "reasoning": "r", "new_skill_md": big_md}
        with (
            patch.object(opt, "_score_skill", new=AsyncMock(side_effect=[baseline])) as mock_score,
            patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=analysis)),
            patch.object(opt, "_mutate_skill", new=AsyncMock(return_value=mutation)),
        ):
            result = await opt.optimize({"SKILL.md": current_md}, [], [], max_rounds=1)
        log = result["mutation_log"]
        self.assertEqual(log[0]["reason"], "edit_limit_exceeded")
        self.assertFalse(log[0]["kept"])
        # 只评了基线，候选被拒后没有复评（省 token）。
        self.assertEqual(mock_score.await_count, 1)

    async def test_modest_edit_passes_when_limit_roomy(self):
        opt = make_optimizer()
        opt.edit_limit = 2.0
        current_md = "# S\n" + "a" * 100
        modest_md = current_md + "b" * 50  # 变化 ~50% < 200%
        baseline = {"passed": 2, "total": 4, "per_eval": [], "details": []}
        improved = {"passed": 3, "total": 4, "per_eval": [], "details": []}
        analysis = {"diagnosis": "d", "mutation_strategy": "add_example",
                    "target_section": "s", "suggested_change": "c"}
        mutation = {"description": "m", "reasoning": "r", "new_skill_md": modest_md}
        with (
            patch.object(opt, "_score_skill", new=AsyncMock(side_effect=[baseline, improved])),
            patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=analysis)),
            patch.object(opt, "_mutate_skill", new=AsyncMock(return_value=mutation)),
        ):
            result = await opt.optimize({"SKILL.md": current_md}, [], [], max_rounds=1)
        self.assertTrue(any(m.get("kept") for m in result["mutation_log"]))


class TestRoundMemoryAndBlacklist(unittest.IsolatedAsyncioTestCase):
    """P2: round memory + strategy blacklist + rejected-edit buffer injection."""

    async def test_analyze_injects_memory_blacklist_and_rejected_edits(self):
        opt = make_optimizer()
        captured = {}

        async def fake_ask(agent, prompt, **kwargs):
            captured["prompt"] = prompt
            return {"diagnosis": "d", "mutation_strategy": "add_example",
                    "target_section": "s", "suggested_change": "c"}

        details = [{"eval_id": 1, "passed": False, "scenario_id": 1}]
        round_memory = [{
            "round": 1, "strategy": "add_example", "diagnosis": "x", "target": "s",
            "outcome": "discarded", "score_before": 50.0, "score_after": 50.0, "reason": "not_best",
        }]
        mutation_log = [
            {"strategy_type": "add_example", "kept": False, "reason": "not_best",
             "description": "add an example about dates"},
            {"strategy_type": "add_example", "kept": False, "reason": "regression:content_shrunk",
             "description": "shrink"},
        ]
        with patch.object(opt, "_ask_json", new=fake_ask):
            await opt._analyze_failures(
                "# S", [], [], details,
                round_memory=round_memory, mutation_log=mutation_log,
            )
        p = captured["prompt"]
        self.assertIn("Recent attempts (previous rounds)", p)
        self.assertIn("(blocked: tried without improvement)", p)
        self.assertIn("Avoid repeating these rejected edits", p)
        self.assertIn("original purpose", p)

    async def test_optimize_injects_round_memory_across_rounds(self):
        opt = make_optimizer()
        baseline = {"passed": 2, "total": 4, "per_eval": [], "details": []}
        improved = {"passed": 3, "total": 4, "per_eval": [], "details": []}
        analysis = {"diagnosis": "d", "mutation_strategy": "add_example",
                    "target_section": "s", "suggested_change": "c"}
        mutation = {"description": "m", "reasoning": "r", "new_skill_md": "# New"}
        captured = []

        async def fake_analyze(skill_md, scenarios, evals, details, strategy_pool=None,
                               round_memory=None, mutation_log=None, lessons=None):
            # 快照拷贝：避免后续轮次 append 改动影响已捕获的引用。
            captured.append({
                "round_memory": list(round_memory or []),
                "mutation_log": list(mutation_log or []),
            })
            return analysis

        with (
            patch.object(opt, "_score_skill", new=AsyncMock(side_effect=[baseline, improved, improved])),
            patch.object(opt, "_analyze_failures", new=fake_analyze),
            patch.object(opt, "_mutate_skill", new=AsyncMock(return_value=mutation)),
        ):
            await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=2)
        # 第 1 轮无记忆；第 2 轮携带第 1 轮的记录。
        self.assertEqual(len(captured[0]["round_memory"]), 0)
        self.assertEqual(len(captured[1]["round_memory"]), 1)
        self.assertEqual(captured[1]["round_memory"][0]["round"], 1)
        self.assertIn("add_example", captured[1]["round_memory"][0]["strategy"])


class TestRegressionTooLarge(unittest.TestCase):
    """P2: oversized SKILL.md is rejected by the regression guard."""

    def test_oversized_md_rejected(self):
        ok, reason = SkillOptimizer._regression_check("# S", "# S\n" + "x" * 16000)
        self.assertFalse(ok)
        self.assertEqual(reason, "regression:too_large")

    def test_normal_sized_md_passes(self):
        ok, reason = SkillOptimizer._regression_check("# S", "# S\n" + "x" * 100)
        self.assertTrue(ok)


class TestPatienceEarlyStop(unittest.IsolatedAsyncioTestCase):
    """P3: patience stops the loop after N consecutive non-improving rounds."""

    async def test_patience_stops_after_stagnant_rounds(self):
        opt = make_optimizer()
        opt.patience = 2
        baseline = {"passed": 2, "total": 4, "per_eval": [], "details": []}   # 50%
        same = {"passed": 2, "total": 4, "per_eval": [], "details": []}       # 50%
        analysis = {"diagnosis": "d", "mutation_strategy": "add_example",
                    "target_section": "s", "suggested_change": "c"}
        mutation = {"description": "m", "reasoning": "r", "new_skill_md": "# Same"}
        with (
            patch.object(opt, "_score_skill", new=AsyncMock(side_effect=[baseline, same, same])) as mock_score,
            patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=analysis)),
            patch.object(opt, "_mutate_skill", new=AsyncMock(return_value=mutation)),
        ):
            result = await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=5)
        # 连续 2 轮未提升后停：基线 + 2 轮复评。
        self.assertEqual(mock_score.await_count, 3)
        self.assertEqual(len(result["score_history"]), 3)

    async def test_patience_zero_keeps_old_behavior(self):
        opt = make_optimizer()
        opt.patience = 0  # 默认关闭
        baseline = {"passed": 2, "total": 4, "per_eval": [], "details": []}
        same = {"passed": 2, "total": 4, "per_eval": [], "details": []}
        analysis = {"diagnosis": "d", "mutation_strategy": "add_example",
                    "target_section": "s", "suggested_change": "c"}
        mutation = {"description": "m", "reasoning": "r", "new_skill_md": "# Same"}
        with (
            patch.object(opt, "_score_skill", new=AsyncMock(side_effect=[baseline, same, same, same])) as mock_score,
            patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=analysis)),
            patch.object(opt, "_mutate_skill", new=AsyncMock(return_value=mutation)),
        ):
            result = await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=3)
        # 3 轮全跑：基线 + 3 轮复评。
        self.assertEqual(mock_score.await_count, 4)


class TestSaturationExit(unittest.IsolatedAsyncioTestCase):
    """P3: saturation pre-flight skips all rounds with no headroom."""

    async def test_zero_score_all_failed_skips_rounds(self):
        opt = make_optimizer()
        opt.saturation_exit = True
        baseline = {"passed": 0, "total": 4, "per_eval": [], "details": []}   # 0%，全失败
        with (
            patch.object(opt, "_score_skill", new=AsyncMock(return_value=baseline)) as mock_score,
        ):
            result = await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=5)
        self.assertEqual(mock_score.await_count, 1)  # 只评基线，零轮
        self.assertEqual(result["final_score"], 0.0)

    async def test_saturation_off_runs_rounds(self):
        opt = make_optimizer()
        baseline = {"passed": 0, "total": 4, "per_eval": [], "details": []}
        same = {"passed": 0, "total": 4, "per_eval": [], "details": []}
        analysis = {"diagnosis": "d", "mutation_strategy": "add_example",
                    "target_section": "s", "suggested_change": "c"}
        mutation = {"description": "m", "reasoning": "r", "new_skill_md": "# Same"}
        with (
            patch.object(opt, "_score_skill", new=AsyncMock(side_effect=[baseline, same])) as mock_score,
            patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=analysis)),
            patch.object(opt, "_mutate_skill", new=AsyncMock(return_value=mutation)),
        ):
            result = await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=1)
        self.assertEqual(mock_score.await_count, 2)  # 默认关闭：正常跑 1 轮


class TestFinalConfirm(unittest.IsolatedAsyncioTestCase):
    """P3: final confirm re-scores the winner and rolls back on luck."""

    async def test_confirm_low_rolls_back_to_original(self):
        opt = make_optimizer()
        opt.final_confirm = True
        baseline = {"passed": 2, "total": 4, "per_eval": [], "details": []}   # 50%
        improved = {"passed": 3, "total": 4, "per_eval": [], "details": []}   # 75%
        confirm_low = {"passed": 2, "total": 4, "per_eval": [], "details": []}  # 50% 复核不达标
        analysis = {"diagnosis": "d", "mutation_strategy": "add_example",
                    "target_section": "s", "suggested_change": "c"}
        mutation = {"description": "m", "reasoning": "r", "new_skill_md": "# New"}
        with (
            patch.object(opt, "_score_skill", new=AsyncMock(side_effect=[baseline, improved, confirm_low])),
            patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=analysis)),
            patch.object(opt, "_mutate_skill", new=AsyncMock(return_value=mutation)),
        ):
            result = await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=1)
        self.assertEqual(result["improved_skill_md"], "# S")  # 回退为原始
        self.assertEqual(result["final_score"], 50.0)

    async def test_confirm_ok_keeps_improvement(self):
        opt = make_optimizer()
        opt.final_confirm = True
        baseline = {"passed": 2, "total": 4, "per_eval": [], "details": []}
        improved = {"passed": 3, "total": 4, "per_eval": [], "details": []}
        confirm_ok = {"passed": 3, "total": 4, "per_eval": [], "details": []}  # 75% 复核达标
        analysis = {"diagnosis": "d", "mutation_strategy": "add_example",
                    "target_section": "s", "suggested_change": "c"}
        mutation = {"description": "m", "reasoning": "r", "new_skill_md": "# New"}
        with (
            patch.object(opt, "_score_skill", new=AsyncMock(side_effect=[baseline, improved, confirm_ok])),
            patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=analysis)),
            patch.object(opt, "_mutate_skill", new=AsyncMock(return_value=mutation)),
        ):
            result = await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=1)
        self.assertEqual(result["improved_skill_md"], "# New")
        self.assertEqual(result["final_score"], 75.0)


class TestCandidateConfirmGate(unittest.IsolatedAsyncioTestCase):
    """候选配对复核：只确认初评胜者，每一对都须严格胜过当轮 incumbent。"""

    @staticmethod
    def _score(passed, total=4):
        return {"passed": passed, "total": total, "per_eval": [], "details": []}

    @staticmethod
    def _analysis():
        return {
            "diagnosis": "d", "mutation_strategy": "add_example",
            "target_section": "s", "suggested_change": "c",
        }

    @staticmethod
    def _mutation():
        return {"description": "m", "reasoning": "r", "new_skill_md": "# New"}

    def test_default_disabled_and_constructor_clamped(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(make_optimizer().candidate_confirm_runs, 0)
        with patch.dict(os.environ, {"CANDIDATE_CONFIRM_RUNS": "2"}, clear=True):
            self.assertEqual(make_optimizer().candidate_confirm_runs, 2)
        self.assertEqual(make_optimizer(candidate_confirm_runs=99).candidate_confirm_runs, 3)
        self.assertEqual(make_optimizer(candidate_confirm_runs=-2).candidate_confirm_runs, 0)

    async def test_stable_paired_win_is_kept(self):
        opt = make_optimizer(candidate_confirm_runs=1)
        scores = [
            self._score(2),  # 初始 incumbent 50
            self._score(3),  # 候选初评 75
            self._score(2),  # incumbent 确认 50
            self._score(3),  # challenger 确认 75
        ]
        with (
            patch.object(opt, "_score_skill", new=AsyncMock(side_effect=scores)) as mock_score,
            patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=self._analysis())),
            patch.object(opt, "_mutate_skill", new=AsyncMock(return_value=self._mutation())),
        ):
            result = await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=1)
        self.assertEqual(mock_score.await_count, 4)
        self.assertEqual(result["improved_skill_md"], "# New")
        self.assertEqual(result["final_score"], 75.0)
        self.assertTrue(result["mutation_log"][0]["kept"])

    async def test_confirmation_tie_rejects_lucky_initial_winner(self):
        opt = make_optimizer(candidate_confirm_runs=1)
        scores = [
            self._score(2),  # baseline 50
            self._score(3),  # lucky initial challenger 75
            self._score(2),  # fresh incumbent 50
            self._score(2),  # fresh challenger ties 50 -> reject
        ]
        with (
            patch.object(opt, "_score_skill", new=AsyncMock(side_effect=scores)),
            patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=self._analysis())),
            patch.object(opt, "_mutate_skill", new=AsyncMock(return_value=self._mutation())),
        ):
            result = await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=1)
        self.assertEqual(result["improved_skill_md"], "# S")
        self.assertEqual(result["final_score"], 50.0)
        self.assertFalse(result["mutation_log"][0]["kept"])
        self.assertEqual(result["mutation_log"][0]["reason"], "confirmation_failed")

    async def test_every_confirmation_pair_must_win(self):
        opt = make_optimizer(candidate_confirm_runs=2)
        scores = [
            self._score(2), self._score(3),  # baseline / initial challenger
            self._score(2), self._score(3),  # pair 1: 75 > 50
            self._score(2), self._score(2),  # pair 2: tie -> reject
        ]
        with (
            patch.object(opt, "_score_skill", new=AsyncMock(side_effect=scores)) as mock_score,
            patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=self._analysis())),
            patch.object(opt, "_mutate_skill", new=AsyncMock(return_value=self._mutation())),
        ):
            result = await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=1)
        self.assertEqual(mock_score.await_count, 6)
        self.assertEqual(result["mutation_log"][0]["reason"], "confirmation_failed")

    async def test_non_improving_candidate_skips_confirmation_cost(self):
        opt = make_optimizer(candidate_confirm_runs=3)
        scores = [self._score(2), self._score(2)]
        with (
            patch.object(opt, "_score_skill", new=AsyncMock(side_effect=scores)) as mock_score,
            patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=self._analysis())),
            patch.object(opt, "_mutate_skill", new=AsyncMock(return_value=self._mutation())),
        ):
            result = await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=1)
        self.assertEqual(mock_score.await_count, 2)
        self.assertEqual(result["final_score"], 50.0)

    async def test_accepted_score_uses_most_conservative_observation(self):
        opt = make_optimizer(candidate_confirm_runs=2)
        scores = [
            self._score(2), self._score(4),  # 50 -> initial 100
            self._score(2), self._score(3),  # pair 1 confirms at 75
            self._score(2), self._score(4),  # pair 2 confirms at 100
        ]
        with (
            patch.object(opt, "_score_skill", new=AsyncMock(side_effect=scores)),
            patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=self._analysis())),
            patch.object(opt, "_mutate_skill", new=AsyncMock(return_value=self._mutation())),
        ):
            result = await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=1)
        self.assertEqual(result["final_score"], 75.0)
        self.assertTrue(result["mutation_log"][0]["kept"])


class TestLessonStore(unittest.IsolatedAsyncioTestCase):
    """P4: cross-session lesson store persists kept fixes as jsonl."""

    def test_append_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lessons.jsonl")
            SkillOptimizer._append_lesson(path, {"strategy": "add_example", "summary": "s1"})
            SkillOptimizer._append_lesson(path, {"strategy": "add_constraint", "summary": "s2"})
            lessons = SkillOptimizer._load_lessons(path)
        self.assertEqual(len(lessons), 2)
        self.assertEqual(lessons[0]["strategy"], "add_example")
        self.assertEqual(lessons[1]["summary"], "s2")

    def test_load_limited_to_recent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lessons.jsonl")
            for i in range(7):
                SkillOptimizer._append_lesson(path, {"i": i})
            lessons = SkillOptimizer._load_lessons(path, limit=5)
        self.assertEqual([l["i"] for l in lessons], [2, 3, 4, 5, 6])

    def test_truncate_at_1000_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lessons.jsonl")
            for i in range(1005):
                SkillOptimizer._append_lesson(path, {"i": i})
            with open(path, encoding="utf-8") as f:
                lines = [l for l in f.read().splitlines() if l.strip()]
            lessons = SkillOptimizer._load_lessons(path, limit=1000)
        self.assertEqual(len(lines), 1000)
        self.assertEqual(lessons[0]["i"], 5)  # 保留最近的 1000 条

    def test_missing_file_returns_empty(self):
        self.assertEqual(SkillOptimizer._load_lessons("/nonexistent/lessons.jsonl"), [])

    def test_disabled_when_no_path(self):
        self.assertEqual(SkillOptimizer._load_lessons(None), [])
        SkillOptimizer._append_lesson(None, {"strategy": "x"})  # 不抛异常

    async def test_analyze_injects_lessons(self):
        opt = make_optimizer()
        captured = {}

        async def fake_ask(agent, prompt, **kwargs):
            captured["prompt"] = prompt
            return {"diagnosis": "d", "mutation_strategy": "add_example",
                    "target_section": "s", "suggested_change": "c"}

        details = [{"eval_id": 1, "passed": False, "scenario_id": 1}]
        lessons = [{"strategy": "add_example", "summary": "worked before"}]
        with patch.object(opt, "_ask_json", new=fake_ask):
            await opt._analyze_failures("# S", [], [], details, lessons=lessons)
        self.assertIn("Lessons from past successful fixes", captured["prompt"])

    async def test_optimize_persists_kept_fix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lessons.jsonl")
            opt = make_optimizer()
            opt.lesson_file = path
            baseline = {"passed": 2, "total": 4, "per_eval": [], "details": []}
            improved = {"passed": 3, "total": 4, "per_eval": [], "details": []}
            analysis = {"diagnosis": "d", "mutation_strategy": "add_example",
                        "target_section": "s", "suggested_change": "c"}
            mutation = {"description": "m", "reasoning": "r", "new_skill_md": "# New"}
            with (
                patch.object(opt, "_score_skill", new=AsyncMock(side_effect=[baseline, improved])),
                patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=analysis)),
                patch.object(opt, "_mutate_skill", new=AsyncMock(return_value=mutation)),
            ):
                await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=1)
            lessons = SkillOptimizer._load_lessons(path)
        self.assertEqual(len(lessons), 1)
        self.assertEqual(lessons[0]["strategy"], "add_example")
        self.assertEqual(lessons[0]["score_after"], 75.0)
        self.assertNotIn("skill_md", lessons[0])  # 只记模型生成内容


class TestLessonStoreSqlite(unittest.IsolatedAsyncioTestCase):
    """RAG: SQLite-backed lesson store (embedding BLOB, 1000 cap)."""

    def test_sqlite_append_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lessons.db")
            SkillOptimizer._append_lesson(path, {
                "skill_name": "writer", "strategy": "add_example",
                "summary": "s1", "score_before": 50.0, "score_after": 75.0,
                "created_at": "2026-08-07T14:00:00",
            })
            SkillOptimizer._append_lesson(path, {
                "skill_name": "coder", "strategy": "add_constraint", "summary": "s2",
            })
            lessons = SkillOptimizer._load_lessons(path)
        self.assertEqual(len(lessons), 2)
        self.assertEqual(lessons[0]["skill_name"], "writer")
        self.assertEqual(lessons[0]["score_after"], 75.0)
        self.assertEqual(lessons[1]["skill_name"], "coder")

    def test_sqlite_embedding_blob_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lessons.db")
            SkillOptimizer._append_lesson(path, {
                "strategy": "a", "summary": "x",
                "embedding": [1.0, 0.0, 0.0],
            })
            lessons = SkillOptimizer._load_lessons(path)
        import numpy as np
        emb = lessons[0]["embedding"]
        self.assertIsNotNone(emb)
        self.assertEqual(np.asarray(emb, dtype=np.float32).shape, (3,))

    def test_sqlite_truncate_at_1000(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lessons.db")
            for i in range(1005):
                SkillOptimizer._append_lesson(path, {"strategy": str(i)})
            lessons = SkillOptimizer._load_lessons(path, limit=1000)
        self.assertEqual(len(lessons), 1000)
        self.assertEqual(lessons[0]["strategy"], "5")  # 保留最近的 1000 条

    def test_jsonl_extension_still_used(self):
        # 无 .db 后缀 → 走 jsonl 兼容路径。
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lessons.jsonl")
            SkillOptimizer._append_lesson(path, {"strategy": "a"})
            lessons = SkillOptimizer._load_lessons(path)
        self.assertEqual(len(lessons), 1)
        self.assertEqual(lessons[0]["strategy"], "a")

    async def test_lazy_embedding_persisted_back_to_sqlite(self):
        # semantic 首次检索惰性算 embedding 后应写回 DB，跨会话复用。
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lessons.db")
            SkillOptimizer._append_lesson(path, {"strategy": "a", "summary": "add example"})
            opt = make_optimizer()
            opt.lesson_file = path
            opt.lesson_retrieval = "semantic"
            with patch.object(opt, "_embed", new=AsyncMock(return_value=[1.0, 0.0, 0.0])):
                lessons = await opt._prepare_lessons(
                    "---\nname: writer\n---", [], [], [],
                )
            self.assertEqual(lessons[0]["strategy"], "a")
            self.assertNotIn("_id", lessons[0])  # 内部 id 不进入注入内容
            self.assertNotIn("embedding", lessons[0])  # 向量不污染模型上下文
            # 重新加载：embedding 已持久化（不再需要惰性计算）。
            reloaded = SkillOptimizer._load_lessons(path)
        self.assertIsNotNone(reloaded[0].get("embedding"))


class TestRetrievalModes(unittest.IsolatedAsyncioTestCase):
    """RAG: LESSON_RETRIEVAL off/tag/semantic/hybrid + degradation chain."""

    def test_invalid_mode_falls_back_to_off(self):
        opt = make_optimizer(lesson_retrieval="bogus")
        self.assertEqual(opt.lesson_retrieval, "off")

    def test_skill_name_extracted_from_frontmatter(self):
        md = "---\nname: my-skill\ndescription: x\n---\n# Title"
        self.assertEqual(SkillOptimizer._skill_name_from_md(md), "my-skill")
        self.assertEqual(SkillOptimizer._skill_name_from_md("# no frontmatter"), "")

    async def test_off_mode_returns_recent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lessons.jsonl")
            for i in range(3):
                SkillOptimizer._append_lesson(path, {"i": i})
            opt = make_optimizer()
            opt.lesson_file = path
            opt.lesson_retrieval = "off"
            with patch.dict(os.environ, {"LESSON_N": "2"}, clear=False):
                lessons = await opt._prepare_lessons("# S", [], [], [])
        self.assertEqual([l["i"] for l in lessons], [1, 2])  # 最近 2 条（旧行为）

    async def test_tag_mode_prefers_same_skill_then_generic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lessons.db")
            SkillOptimizer._append_lesson(path, {"skill_name": "writer", "summary": "w1"})
            SkillOptimizer._append_lesson(path, {"skill_name": "coder", "summary": "c2"})
            SkillOptimizer._append_lesson(path, {"summary": "g3"})  # 通用（无绑定）
            opt = make_optimizer()
            opt.lesson_file = path
            opt.lesson_retrieval = "tag"
            lessons = await opt._prepare_lessons("---\nname: writer\n---", [], [], [])
        self.assertEqual([l["summary"] for l in lessons], ["w1", "g3"])  # 同技能优先 + 通用兜底

    async def test_semantic_topk_with_threshold(self):
        opt = make_optimizer()
        opt.lesson_retrieval = "semantic"
        lessons = [
            {"strategy": "a", "summary": "x", "embedding": [1.0, 0.0]},
            {"strategy": "b", "summary": "y", "embedding": [0.0, 1.0]},
        ]
        with patch.object(opt, "_embed", new=AsyncMock(return_value=[1.0, 0.0])):
            top = await opt._retrieve_lessons(lessons, "q", "semantic", 5)
        # 与 query 余弦=1 的保留；余弦=0 的被 0.3 阈值滤掉。
        self.assertEqual([l["strategy"] for l in top], ["a"])

    async def test_semantic_degrades_to_tag_on_embed_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lessons.db")
            SkillOptimizer._append_lesson(path, {"skill_name": "writer", "summary": "w1"})
            SkillOptimizer._append_lesson(path, {"summary": "g2"})
            opt = make_optimizer()
            opt.lesson_file = path
            opt.lesson_retrieval = "semantic"
            with patch.object(opt, "_embed", new=AsyncMock(side_effect=RuntimeError("no key"))):
                lessons = await opt._prepare_lessons("---\nname: writer\n---", [], [], [])
        self.assertEqual([l["summary"] for l in lessons], ["w1", "g2"])  # 降级到 tag

    async def test_hybrid_uses_rrf_fusion(self):
        opt = make_optimizer()
        opt.lesson_retrieval = "hybrid"
        lessons = [
            {"strategy": "a", "summary": "add example", "embedding": [1.0, 0.0]},
            {"strategy": "b", "summary": "other thing", "embedding": [0.0, 1.0]},
        ]
        # query 与 a 语义相近（dense 排 a 第一），但词面更贴近 b（sparse 排 b 第一）。
        with patch.object(opt, "_embed", new=AsyncMock(side_effect=[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])):
            top = await opt._retrieve_lessons(lessons, "other thing", "hybrid", 5)
        self.assertEqual(len(top), 2)  # RRF 融合两者都在


class TestLessonKnobs(unittest.IsolatedAsyncioTestCase):
    """Option A: LESSON_THRESHOLD / LESSON_TOP_K configurable."""

    def test_defaults(self):
        opt = make_optimizer()
        self.assertEqual(opt.lesson_threshold, 0.3)
        self.assertEqual(opt.lesson_top_k, 5)

    def test_env_override(self):
        with patch.dict(os.environ, {"LESSON_THRESHOLD": "0.5", "LESSON_TOP_K": "3"}, clear=False):
            opt = make_optimizer()
        self.assertEqual(opt.lesson_threshold, 0.5)
        self.assertEqual(opt.lesson_top_k, 3)

    def test_constructor_override_beats_env(self):
        with patch.dict(os.environ, {"LESSON_THRESHOLD": "0.5"}, clear=False):
            opt = make_optimizer(lesson_threshold=0.9)
        self.assertEqual(opt.lesson_threshold, 0.9)

    def test_invalid_threshold_clamped(self):
        opt = make_optimizer(lesson_threshold=5.0)  # 超范围 → 钳到 1.0
        self.assertEqual(opt.lesson_threshold, 1.0)

    async def test_semantic_uses_configured_threshold(self):
        opt = make_optimizer()
        opt.lesson_retrieval = "semantic"
        opt.lesson_threshold = 0.9  # 只有余弦 1.0 的能过
        lessons = [
            {"strategy": "a", "summary": "x", "embedding": [1.0, 0.0]},
            {"strategy": "b", "summary": "y", "embedding": [0.5, 0.5]},
        ]
        with patch.object(opt, "_embed", new=AsyncMock(return_value=[1.0, 0.0])):
            top = await opt._retrieve_lessons(lessons, "q", "semantic", 5)
        self.assertEqual([l["strategy"] for l in top], ["a"])


class TestDomainTagging(unittest.IsolatedAsyncioTestCase):
    """Option B: analyze_skill domain label + domain-aware tag filter."""

    async def test_analyze_fills_missing_domain(self):
        opt = make_optimizer()
        raw = {"scenarios": [], "evals": []}  # 无 domain
        with patch.object(opt, "_ask_json", new=AsyncMock(return_value=raw)):
            result = await opt.analyze_skill({"SKILL.md": "# S"})
        self.assertEqual(result["domain"], "other")

    async def test_analyze_keeps_domain(self):
        opt = make_optimizer()
        raw = {"scenarios": [], "evals": [], "domain": "coding"}
        with patch.object(opt, "_ask_json", new=AsyncMock(return_value=raw)):
            result = await opt.analyze_skill({"SKILL.md": "# S"})
        self.assertEqual(result["domain"], "coding")

    def test_tag_filter_prefers_same_skill_then_domain_then_generic(self):
        lessons = [
            {"skill_name": "writer", "domain": "writing", "summary": "w1"},
            {"skill_name": "coder", "domain": "coding", "summary": "c2"},
            {"skill_name": "report-bot", "domain": "writing", "summary": "r3"},  # 同领域不同技能
            {"summary": "g4"},  # 通用
        ]
        pool = SkillOptimizer._tag_filter(lessons, "writer", "writing", 5)
        self.assertEqual([l["summary"] for l in pool], ["w1", "r3", "g4"])

    def test_build_query_includes_domain(self):
        q = SkillOptimizer._build_query("writer", "writing", [], [], [])
        self.assertIn("domain: writing", q)
        self.assertIn("skill: writer", q)


class TestLessonRerank(unittest.IsolatedAsyncioTestCase):
    """Option D: LESSON_RERANK gte-rerank reorders candidates, degrades on failure."""

    def test_default_off(self):
        opt = make_optimizer()
        self.assertFalse(opt.lesson_rerank)
        self.assertEqual(opt.lesson_rerank_pool, 20)

    def test_env_override(self):
        with patch.dict(os.environ, {"LESSON_RERANK": "1", "LESSON_RERANK_POOL": "10"}, clear=False):
            opt = make_optimizer()
        self.assertTrue(opt.lesson_rerank)
        self.assertEqual(opt.lesson_rerank_pool, 10)

    async def test_rerank_reorders_candidates(self):
        opt = make_optimizer()
        opt.lesson_retrieval = "semantic"
        opt.lesson_rerank = True
        lessons = [
            {"strategy": "a", "summary": "alpha fix", "embedding": [1.0, 0.0]},
            {"strategy": "b", "summary": "beta fix", "embedding": [0.9, 0.1]},
        ]
        # dense 排序 a 第一；rerank 认为 b 更相关 → 结果应按 b, a。
        rerank_result = [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.8},
        ]
        with (
            patch.object(opt, "_embed", new=AsyncMock(return_value=[1.0, 0.0])),
            patch.object(opt, "_rerank", new=AsyncMock(return_value=rerank_result)) as mock_rerank,
        ):
            top = await opt._retrieve_lessons(lessons, "query", "semantic", 5)
        self.assertEqual([l["strategy"] for l in top], ["b", "a"])
        # rerank 收到的是候选的检索文本列表。
        self.assertEqual(len(mock_rerank.await_args[0][1]), 2)

    async def test_rerank_off_does_not_call(self):
        opt = make_optimizer()
        opt.lesson_retrieval = "semantic"
        lessons = [
            {"strategy": "a", "summary": "alpha fix", "embedding": [1.0, 0.0]},
            {"strategy": "b", "summary": "beta fix", "embedding": [0.9, 0.1]},
        ]
        with (
            patch.object(opt, "_embed", new=AsyncMock(return_value=[1.0, 0.0])),
            patch.object(opt, "_rerank", new=AsyncMock()) as mock_rerank,
        ):
            top = await opt._retrieve_lessons(lessons, "query", "semantic", 5)
        self.assertEqual([l["strategy"] for l in top], ["a", "b"])
        mock_rerank.assert_not_awaited()

    async def test_rerank_failure_degrades_to_original_order(self):
        opt = make_optimizer()
        opt.lesson_retrieval = "semantic"
        opt.lesson_rerank = True
        lessons = [
            {"strategy": "a", "summary": "alpha fix", "embedding": [1.0, 0.0]},
            {"strategy": "b", "summary": "beta fix", "embedding": [0.9, 0.1]},
        ]
        with (
            patch.object(opt, "_embed", new=AsyncMock(return_value=[1.0, 0.0])),
            patch.object(opt, "_rerank", new=AsyncMock(side_effect=RuntimeError("rerank down"))),
        ):
            top = await opt._retrieve_lessons(lessons, "query", "semantic", 5)
        self.assertEqual([l["strategy"] for l in top], ["a", "b"])  # 降级保持原始排序


class TestQualityDiverseLessonRag(unittest.IsolatedAsyncioTestCase):
    """目标 5：增强经验检索默认关闭，开启后质量/维度/去重/多样性均生效。"""

    def test_defaults_preserve_classic_pipeline(self):
        opt = make_optimizer()
        self.assertEqual(opt.lesson_rag_pipeline, "classic")
        self.assertEqual(opt.lesson_candidate_pool, 20)
        self.assertEqual(opt.lesson_diversity, 0.2)
        self.assertEqual(opt.lesson_dedup_threshold, 0.92)
        self.assertEqual(opt.lesson_context_chars, 6000)

    def test_env_knobs_and_weights(self):
        env = {
            "LESSON_RAG_PIPELINE": "quality_diverse",
            "LESSON_CANDIDATE_POOL": "12",
            "LESSON_DIVERSITY": "0.35",
            "LESSON_DEDUP_THRESHOLD": "0.8",
            "LESSON_CONTEXT_CHARS": "900",
            "LESSON_SIGNAL_WEIGHTS": json.dumps({"quality": 3, "semantic": 1}),
        }
        with patch.dict(os.environ, env, clear=False):
            opt = make_optimizer()
        self.assertEqual(opt.lesson_rag_pipeline, "quality_diverse")
        self.assertEqual(opt.lesson_candidate_pool, 12)
        self.assertEqual(opt.lesson_diversity, 0.35)
        self.assertEqual(opt.lesson_dedup_threshold, 0.8)
        self.assertEqual(opt.lesson_context_chars, 900)
        self.assertAlmostEqual(opt.lesson_signal_weights["quality"], 0.75)
        self.assertAlmostEqual(opt.lesson_signal_weights["semantic"], 0.25)

    def test_invalid_pipeline_falls_back_to_classic(self):
        opt = make_optimizer(lesson_rag_pipeline="unknown")
        self.assertEqual(opt.lesson_rag_pipeline, "classic")

    def test_sparse_jaccard_supports_chinese_bigrams(self):
        score = SkillOptimizer._sparse_jaccard("修复格式错误", "增加一条格式错误处理规则")
        self.assertGreater(score, 0.0)

    async def test_quality_prior_breaks_equal_relevance_tie(self):
        opt = make_optimizer(
            lesson_rag_pipeline="quality_diverse",
            lesson_signal_weights={"semantic": 1, "quality": 3},
        )
        lessons = [
            {
                "strategy": "weak",
                "summary": "same advice",
                "score_before": 70,
                "score_after": 71,
                "embedding": [1.0, 0.0],
            },
            {
                "strategy": "strong",
                "summary": "same advice",
                "score_before": 40,
                "score_after": 90,
                "embedding": [1.0, 0.0],
            },
        ]
        with patch.object(opt, "_embed", new=AsyncMock(return_value=[1.0, 0.0])):
            top = await opt._retrieve_lessons(lessons, "advice", "hybrid", 1)
        self.assertEqual(top[0]["strategy"], "strong")
        self.assertEqual(top[0]["_retrieval"]["pipeline"], "quality_diverse")

    async def test_weak_dimension_metadata_is_a_retrieval_signal(self):
        opt = make_optimizer(
            lesson_rag_pipeline="quality_diverse",
            lesson_signal_weights={"semantic": 1, "dimension": 4},
        )
        lessons = [
            {
                "strategy": "clarity-fix",
                "summary": "rewrite guidance",
                "target_dimension": "clarity",
                "embedding": [1.0, 0.0],
            },
            {
                "strategy": "correctness-fix",
                "summary": "rewrite guidance",
                "target_dimension": "correctness",
                "embedding": [1.0, 0.0],
            },
        ]
        context = {"weak_dimensions": [{"dimension": "clarity", "pct": 25.0}]}
        with patch.object(opt, "_embed", new=AsyncMock(return_value=[1.0, 0.0])):
            top = await opt._retrieve_lessons(
                lessons, "rewrite guidance", "hybrid", 2, context=context
            )
        self.assertEqual(top[0]["strategy"], "clarity-fix")
        self.assertIn("weak_dimension", top[0]["_retrieval"]["matched"])

    async def test_near_duplicates_are_removed_before_diverse_selection(self):
        opt = make_optimizer(
            lesson_rag_pipeline="quality_diverse",
            lesson_dedup_threshold=0.9,
            lesson_diversity=0.5,
        )
        lessons = [
            {
                "strategy": "add_example",
                "summary": "add exact output example",
                "score_before": 30,
                "score_after": 90,
                "embedding": [1.0, 0.0],
            },
            {
                "strategy": "add_example",
                "summary": "add exact output example",
                "score_before": 50,
                "score_after": 60,
                "embedding": [1.0, 0.0],
            },
            {
                "strategy": "add_constraint",
                "summary": "reject unsafe paths",
                "score_before": 40,
                "score_after": 80,
                "embedding": [0.9, 0.1],
            },
        ]
        with patch.object(opt, "_embed", new=AsyncMock(return_value=[1.0, 0.0])):
            top = await opt._retrieve_lessons(lessons, "improve output", "hybrid", 3)
        self.assertEqual(len(top), 2)
        self.assertEqual({l["strategy"] for l in top}, {"add_example", "add_constraint"})

    async def test_prepare_compacts_prompt_and_persists_target_dimension(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lessons.db")
            SkillOptimizer._append_lesson(path, {
                "skill_name": "writer",
                "domain": "writing",
                "strategy": "rewrite_section",
                "diagnosis": "x" * 1000,
                "summary": "make it clear",
                "target_dimension": "clarity",
                "score_before": 40,
                "score_after": 90,
                "embedding": [1.0, 0.0],
            })
            opt = make_optimizer(
                lesson_file=path,
                lesson_retrieval="semantic",
                lesson_rag_pipeline="quality_diverse",
                lesson_context_chars=500,
            )
            with patch.object(opt, "_embed", new=AsyncMock(return_value=[1.0, 0.0])):
                lessons = await opt._prepare_lessons(
                    "---\nname: writer\n---", [], [], [], domain="writing",
                    weak_dimensions=[{"dimension": "clarity", "pct": 20}],
                )
            reloaded = SkillOptimizer._load_lessons(path)
        self.assertEqual(reloaded[0]["target_dimension"], "clarity")
        self.assertEqual(lessons[0]["target_dimension"], "clarity")
        if "diagnosis" in lessons[0]:
            self.assertLessEqual(len(lessons[0]["diagnosis"]), 300)
        self.assertLessEqual(len(json.dumps(lessons, ensure_ascii=False)), 500)
        self.assertNotIn("embedding", lessons[0])
        self.assertNotIn("_id", lessons[0])


class TestLessonFailureContextAndBatching(unittest.IsolatedAsyncioTestCase):
    """后续改进：失败 eval 进入查询，冷 lesson embedding 可按官方上限批处理。"""

    def test_defaults_preserve_old_query_and_single_embedding_calls(self):
        opt = make_optimizer()
        self.assertFalse(opt.lesson_failure_context)
        self.assertEqual(opt.lesson_embed_batch_size, 1)

    def test_env_enables_failure_context_and_clamps_batch_to_ten(self):
        with patch.dict(os.environ, {
            "LESSON_FAILURE_CONTEXT": "1",
            "LESSON_EMBED_BATCH_SIZE": "99",
        }, clear=False):
            opt = make_optimizer()
        self.assertTrue(opt.lesson_failure_context)
        self.assertEqual(opt.lesson_embed_batch_size, 10)

    def test_failure_context_uses_failed_eval_and_matching_scenario(self):
        scenarios = [
            {"id": 1, "input": "unrelated first scenario"},
            {"id": 2, "input": "validate required CSV columns"},
        ]
        evals = [{
            "id": 7,
            "name": "required columns",
            "criterion": "Reject CSV input when required columns are missing.",
            "question": "Does it reject a missing schema column?",
            "pass_condition": "The missing column is named and rejected.",
            "dimension": "correctness",
        }]
        details = [{
            "eval_id": 7,
            "scenario_id": 2,
            "passed": False,
            "reason": "required field was accepted",
        }]
        classic = SkillOptimizer._build_query(
            "csv-validator", "data", scenarios, evals, details
        )
        enhanced = SkillOptimizer._build_query(
            "csv-validator", "data", scenarios, evals, details,
            include_failure_context=True,
        )
        self.assertNotIn("Reject CSV input", classic)
        self.assertIn("unrelated first scenario", classic)
        self.assertIn("Reject CSV input", enhanced)
        self.assertIn("correctness", enhanced)
        self.assertIn("validate required CSV columns", enhanced)
        self.assertNotIn("unrelated first scenario", enhanced)

    async def test_embed_many_deduplicates_and_batches_in_input_order(self):
        opt = make_optimizer(lesson_embed_batch_size=2)
        calls = []

        def fake_batch(texts):
            calls.append(list(texts))
            return [[float(len(text)), 1.0] for text in texts]

        with patch.object(opt, "_embed_many_sync", side_effect=fake_batch):
            vectors = await opt._embed_many(["a", "bb", "a", "ccc"])
        self.assertEqual(calls, [["a", "bb"], ["ccc"]])
        self.assertEqual(vectors, [[1.0, 1.0], [2.0, 1.0], [1.0, 1.0], [3.0, 1.0]])

    def test_embed_many_sync_restores_dashscope_text_index_order(self):
        response = SimpleNamespace(
            status_code=200,
            output={"embeddings": [
                {"text_index": 1, "embedding": [0.0, 1.0]},
                {"text_index": 0, "embedding": [1.0, 0.0]},
            ]},
        )
        fake_dashscope = SimpleNamespace(
            TextEmbedding=SimpleNamespace(call=MagicMock(return_value=response))
        )
        opt = make_optimizer(lesson_embed_batch_size=2)
        with patch.dict("sys.modules", {"dashscope": fake_dashscope}):
            vectors = opt._embed_many_sync(["first", "second"])
        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])
        call_kwargs = fake_dashscope.TextEmbedding.call.call_args.kwargs
        self.assertEqual(call_kwargs["input"], ["first", "second"])
        self.assertNotIn("api_key", str(call_kwargs["input"]))

    async def test_batch_failure_still_degrades_to_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lessons.jsonl")
            SkillOptimizer._append_lesson(path, {
                "skill_name": "writer", "strategy": "a", "summary": "same skill",
            })
            SkillOptimizer._append_lesson(path, {
                "strategy": "b", "summary": "generic",
            })
            opt = make_optimizer(
                lesson_file=path,
                lesson_retrieval="hybrid",
                lesson_rag_pipeline="quality_diverse",
                lesson_embed_batch_size=2,
            )
            with (
                patch.object(opt, "_embed", new=AsyncMock(return_value=[1.0, 0.0])),
                patch.object(opt, "_embed_many_sync", side_effect=RuntimeError("batch down")),
            ):
                lessons = await opt._prepare_lessons(
                    "---\nname: writer\n---", [{"id": 1, "input": "draft"}], [], [],
                )
        self.assertEqual([lesson["strategy"] for lesson in lessons], ["a", "b"])


class TestLessonDashScopeKeySeparation(unittest.TestCase):
    """Generation credentials must stay separate from DashScope RAG calls."""

    @staticmethod
    def _fake_dashscope():
        embedding_response = SimpleNamespace(
            status_code=200,
            output={"embeddings": [{"text_index": 0, "embedding": [1.0, 0.0]}]},
        )
        rerank_response = SimpleNamespace(
            status_code=200,
            output={"results": [{"index": 0, "relevance_score": 0.9}]},
        )
        return SimpleNamespace(
            TextEmbedding=SimpleNamespace(call=MagicMock(return_value=embedding_response)),
            TextReRank=SimpleNamespace(call=MagicMock(return_value=rerank_response)),
        )

    def test_dashscope_env_key_wins_for_all_rag_calls(self):
        opt = make_optimizer(api_key="generation-key", model="glm-4-flash")
        fake_dashscope = self._fake_dashscope()
        with (
            patch.dict("sys.modules", {"dashscope": fake_dashscope}),
            patch.dict(os.environ, {"DASHSCOPE_API_KEY": "rag-key"}, clear=False),
        ):
            opt._embed_sync("single")
            opt._embed_many_sync(["batch"])
            opt._rerank_sync("query", ["document"])

        embedding_keys = [
            call.kwargs["api_key"]
            for call in fake_dashscope.TextEmbedding.call.call_args_list
        ]
        self.assertEqual(embedding_keys, ["rag-key", "rag-key"])
        self.assertEqual(
            fake_dashscope.TextReRank.call.call_args.kwargs["api_key"], "rag-key"
        )

    def test_rag_calls_fall_back_to_optimizer_key_without_env(self):
        opt = make_optimizer(api_key="generation-key", model="glm-4-flash")
        fake_dashscope = self._fake_dashscope()
        with (
            patch.dict("sys.modules", {"dashscope": fake_dashscope}),
            patch.dict(os.environ, {"DASHSCOPE_API_KEY": ""}, clear=False),
        ):
            opt._embed_sync("single")
            opt._embed_many_sync(["batch"])
            opt._rerank_sync("query", ["document"])

        embedding_keys = [
            call.kwargs["api_key"]
            for call in fake_dashscope.TextEmbedding.call.call_args_list
        ]
        self.assertEqual(embedding_keys, ["generation-key", "generation-key"])
        self.assertEqual(
            fake_dashscope.TextReRank.call.call_args.kwargs["api_key"],
            "generation-key",
        )

    def test_codex_generation_does_not_change_dashscope_embedding_route(self):
        opt = make_optimizer(api_key="", model="gpt-5.6-sol")
        fake_dashscope = self._fake_dashscope()
        with (
            patch.dict("sys.modules", {"dashscope": fake_dashscope}),
            patch.dict(os.environ, {"DASHSCOPE_API_KEY": "rag-only-key"}, clear=False),
            patch("llm_client._CodexAppServerClient.call") as mock_codex,
        ):
            vector = opt._embed_sync("lesson metadata")
        self.assertEqual(vector, [1.0, 0.0])
        self.assertEqual(
            fake_dashscope.TextEmbedding.call.call_args.kwargs["model"],
            "text-embedding-v3",
        )
        self.assertEqual(
            fake_dashscope.TextEmbedding.call.call_args.kwargs["api_key"],
            "rag-only-key",
        )
        mock_codex.assert_not_called()


class TestLessonQualityGate(unittest.IsolatedAsyncioTestCase):
    """经验沉淀质量门槛：LESSON_MIN_GAIN / LESSON_MIN_FINAL（OR 语义）。"""

    def test_defaults_disabled(self):
        opt = make_optimizer()
        self.assertEqual(opt.lesson_min_gain, 0.0)
        self.assertEqual(opt.lesson_min_final, 0.0)

    def test_env_override(self):
        with patch.dict(os.environ, {"LESSON_MIN_GAIN": "15", "LESSON_MIN_FINAL": "85"}, clear=False):
            opt = make_optimizer()
        self.assertEqual(opt.lesson_min_gain, 15.0)
        self.assertEqual(opt.lesson_min_final, 85.0)

    def test_gain_gate_or_semantics(self):
        opt = make_optimizer(lesson_min_gain=15.0, lesson_min_final=85.0)
        # 幅度达标（50→75, gain 25）→ 沉淀
        self.assertTrue(opt._lesson_qualifies(50.0, 75.0))
        # 水位达标（98→100）→ 沉淀（OR 命中 final）
        self.assertTrue(opt._lesson_qualifies(98.0, 100.0))
        # 双不达标（50→60）→ 不沉淀
        self.assertFalse(opt._lesson_qualifies(50.0, 60.0))

    def test_only_final_gate(self):
        opt = make_optimizer(lesson_min_final=85.0)
        self.assertFalse(opt._lesson_qualifies(50.0, 75.0))   # 水位 75 < 85
        self.assertTrue(opt._lesson_qualifies(50.0, 90.0))    # 水位 90 >= 85

    def test_only_gain_gate(self):
        opt = make_optimizer(lesson_min_gain=20.0)
        self.assertFalse(opt._lesson_qualifies(90.0, 92.0))   # gain 2 < 20
        # 只开 gain 时最终水位不参与判断：90→100 的 gain=10 < 20 → 不沉淀。
        self.assertFalse(opt._lesson_qualifies(90.0, 100.0))
        self.assertTrue(opt._lesson_qualifies(50.0, 80.0))    # gain 30 >= 20

    def test_disabled_always_true(self):
        opt = make_optimizer()
        self.assertTrue(opt._lesson_qualifies(50.0, 52.0))    # 旧行为：任何 kept 都沉淀
        self.assertTrue(opt._lesson_qualifies(0.0, 100.0))

    async def test_optimize_skips_low_quality_persistence(self):
        # kept 但未达门槛 → 经验库不新增；达标 → 新增。
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lessons.db")
            opt = make_optimizer(lesson_min_gain=15.0, lesson_min_final=85.0)
            opt.lesson_file = path
            low_baseline = {"passed": 2, "total": 4, "per_eval": [], "details": []}   # 50%
            low_kept = {"passed": 2, "total": 4, "per_eval": [], "details": []}       # 50%（不提升→不 kept？）
            # 用 50→60 构造 kept 但不过门槛：baseline 50，improved 60。
            improved60 = {"passed": 3, "total": 5, "per_eval": [], "details": []}      # 60%
            analysis = {"diagnosis": "d", "mutation_strategy": "add_example",
                        "target_section": "s", "suggested_change": "c"}
            mutation = {"description": "m", "reasoning": "r", "new_skill_md": "# New"}
            with (
                patch.object(opt, "_score_skill", new=AsyncMock(side_effect=[low_baseline, improved60])),
                patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=analysis)),
                patch.object(opt, "_mutate_skill", new=AsyncMock(return_value=mutation)),
            ):
                await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=1)
            lessons_after_low = SkillOptimizer._load_lessons(path)
        # 50→60（gain 10 < 15，final 60 < 85）→ 不沉淀。
        self.assertEqual(len(lessons_after_low), 0)


class TestStopRequested(unittest.IsolatedAsyncioTestCase):
    """C5: cooperative stop raises between rounds."""

    async def test_stop_raises_stop_optimization_error(self):
        opt = make_optimizer()
        opt.stop_requested = True
        with self.assertRaises(StopOptimizationError):
            opt.check_stop()

    async def test_stop_requested_during_optimization(self):
        opt = make_optimizer()
        baseline = {"passed": 2, "total": 4, "per_eval": [], "details": []}
        with (
            patch.object(opt, "_score_skill", new=AsyncMock(return_value=baseline)),
        ):
            # 开始前未置位 → 不抛；置位后（模拟轮间用户停止）→ 抛错。
            opt.check_stop()
            opt.stop_requested = True
            with self.assertRaises(StopOptimizationError):
                await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=3)


class TestStopProvider(unittest.IsolatedAsyncioTestCase):
    """C5 增强：stop_provider 回调让 /api/stop 真正到达优化器（轮间生效）。"""

    async def test_stop_provider_true_raises(self):
        opt = make_optimizer(stop_provider=lambda: True)
        with self.assertRaises(StopOptimizationError):
            opt.check_stop()

    async def test_stop_provider_false_does_not_raise(self):
        opt = make_optimizer(stop_provider=lambda: False)
        opt.check_stop()  # 不抛

    async def test_stop_provider_during_optimization(self):
        opt = make_optimizer(stop_provider=lambda: True)
        baseline = {"passed": 2, "total": 4, "per_eval": [], "details": []}
        with patch.object(opt, "_score_skill", new=AsyncMock(return_value=baseline)) as mock_score:
            with self.assertRaises(StopOptimizationError):
                await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=3)
        # 停止在 baseline 评分前生效（与手写版 L598 的 check_stop 一致）。
        mock_score.assert_not_awaited()

    async def test_stop_provider_exception_ignored(self):
        # stop_provider 自身抛异常视为未停止，不打断优化。
        def bad_provider():
            raise RuntimeError("provider broken")
        opt = make_optimizer(stop_provider=bad_provider)
        opt.check_stop()  # 不抛


class TestCheckpointer(unittest.IsolatedAsyncioTestCase):
    """阶段 5：SqliteSaver 断点续跑（默认关闭；thread_id 隔离；key 不入库）。"""

    async def test_disabled_by_default(self):
        opt = make_optimizer()
        self.assertIsNone(opt.checkpoint_file)
        # 无 checkpoint 时 optimize 正常执行（无 config）。
        baseline = {"passed": 4, "total": 4, "per_eval": [], "details": []}
        with patch.object(opt, "_score_skill", new=AsyncMock(return_value=baseline)):
            result = await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=5)
        self.assertEqual(result["baseline_score"], 100.0)

    async def test_resume_after_interruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = os.path.join(tmp, "cp.db")
            opt = make_optimizer(checkpoint_file=cp)
            # 恒定 50%（不提升、不早停），确保第 2 轮复评时触发中断。
            baseline = {"passed": 2, "total": 4, "per_eval": [], "details": []}
            analysis = {"diagnosis": "d", "mutation_strategy": "add_example",
                        "target_section": "s", "suggested_change": "c"}
            mutation = {"description": "m", "reasoning": "r", "new_skill_md": "# New"}
            calls = {"n": 0}
            interrupted = {"on": True}

            async def score(*a, **kw):
                if interrupted["on"]:
                    calls["n"] += 1
                    if calls["n"] == 3:  # baseline + r1 复评之后 → 第 2 轮复评中断
                        raise RuntimeError("interrupted mid-round-2")
                return baseline

            events = []
            async def cb(ev):
                events.append(ev["type"])

            with (
                patch.object(opt, "_score_skill", new=AsyncMock(side_effect=score)),
                patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=analysis)),
                patch.object(opt, "_mutate_skill", new=AsyncMock(return_value=mutation)),
            ):
                with self.assertRaises(RuntimeError):
                    await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=5, thread_id="t1", callback=cb)
                # 中断前已产出 baseline + 第 1 轮完整事件 + 第 2 轮开始事件。
                self.assertIn("baseline", events)
                interrupted["on"] = False
                events.clear()
                # 同一 thread_id 续跑：跳过已 emit 事件，从第 2 轮复评继续。
                result = await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=5, thread_id="t1", callback=cb)

            # 续跑不重复 baseline（只 emit 中断点之后的新事件）。
            self.assertNotIn("baseline", events)
            # 最终结果完整：基线 50% + 5 轮（中断点续跑后跑满 max_rounds）。
            self.assertEqual(result["baseline_score"], 50.0)
            self.assertEqual(result["final_score"], 50.0)
            self.assertEqual(len(result["score_history"]), 6)
            self.assertEqual(len(result["mutation_log"]), 5)

    async def test_thread_id_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = os.path.join(tmp, "cp.db")
            opt = make_optimizer(checkpoint_file=cp)
            baseline = {"passed": 2, "total": 4, "per_eval": [], "details": []}
            analysis = {"diagnosis": "d", "mutation_strategy": "add_example",
                        "target_section": "s", "suggested_change": "c"}
            mutation = {"description": "m", "reasoning": "r", "new_skill_md": "# New"}
            calls = {"n": 0}
            interrupted = {"on": True}

            async def score(*a, **kw):
                if interrupted["on"]:
                    calls["n"] += 1
                    if calls["n"] == 3:
                        raise RuntimeError("interrupted")
                return baseline

            events = []
            async def cb(ev):
                events.append(ev["type"])

            with (
                patch.object(opt, "_score_skill", new=AsyncMock(side_effect=score)),
                patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=analysis)),
                patch.object(opt, "_mutate_skill", new=AsyncMock(return_value=mutation)),
            ):
                with self.assertRaises(RuntimeError):
                    await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=5, thread_id="t1")
                interrupted["on"] = False
                events.clear()
                # 新 thread_id：全新开始 → baseline 事件重新产生。
                await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=5, thread_id="t2", callback=cb)
            self.assertIn("baseline", events)

    async def test_api_key_not_in_checkpoint_file(self):
        # 运行一次带 key 的优化（中断后留有 checkpoint），检查库文件内容：
        # api_key 绝不入 checkpoint（prompt/模型响应同样不入库；图状态含
        # 正在优化的 skill_md 是执行必需，允许）。
        with tempfile.TemporaryDirectory() as tmp:
            cp = os.path.join(tmp, "cp.db")
            opt = make_optimizer(api_key="sk-test-secret-xyz", checkpoint_file=cp)
            baseline = {"passed": 2, "total": 4, "per_eval": [], "details": []}
            analysis = {"diagnosis": "d", "mutation_strategy": "add_example",
                        "target_section": "s", "suggested_change": "c"}
            mutation = {"description": "m", "reasoning": "r", "new_skill_md": "# New"}
            with (
                patch.object(opt, "_score_skill", new=AsyncMock(return_value=baseline)),
                patch.object(opt, "_analyze_failures", new=AsyncMock(return_value=analysis)),
                patch.object(opt, "_mutate_skill", new=AsyncMock(side_effect=RuntimeError("stop-here"))),
            ):
                with self.assertRaises(RuntimeError):
                    await opt.optimize({"SKILL.md": "# S"}, [], [], max_rounds=5, thread_id="t1")
            with open(cp, "rb") as f:
                content = f.read()
            self.assertNotIn(b"sk-test-secret-xyz", content)


class TestFrontmatterParsing(unittest.TestCase):
    """Folded YAML frontmatter parsing for example skills."""

    def test_folded_description_parsed(self):
        content = (
            "---\n"
            "name: my-skill\n"
            "description: >-\n"
            "  First line of a long description\n"
            "  that continues over multiple lines\n"
            "license: Apache-2.0\n"
            "---\n"
            "# Body"
        )
        meta = parse_skill_frontmatter(content)
        self.assertEqual(meta["name"], "my-skill")
        self.assertEqual(
            meta["description"],
            "First line of a long description that continues over multiple lines",
        )
        self.assertEqual(meta["license"], "Apache-2.0")

    def test_plain_values_unchanged(self):
        content = "---\nname: x\ndescription: short\n---\n# Body"
        meta = parse_skill_frontmatter(content)
        self.assertEqual(meta["description"], "short")

    def test_no_frontmatter_returns_empty(self):
        self.assertEqual(parse_skill_frontmatter("# No frontmatter"), {})


class TestExamplesListing(unittest.IsolatedAsyncioTestCase):
    """Examples endpoint reads skill-examples/*.zip (fixes dead selector)."""

    async def test_list_examples_contains_bundled_packs(self):
        ex = await list_examples()
        paths = {e["path"] for e in ex["examples"]}
        self.assertTrue({
            "project-graveyard",
            "commit-archaeologist",
            "dependency-doctor",
            "thinking-out-loud",
        }.issubset(paths))

    async def test_load_example_creates_session(self):
        ex = await list_examples()
        first = ex["examples"][0]["path"]
        loaded = await load_example(first)
        self.assertIn("session_id", loaded)
        self.assertGreater(len(loaded["file_list"]), 0)
        # 文件清单里应包含 SKILL.md，且 metadata 解析出了技能名。
        self.assertTrue(any(f.endswith("SKILL.md") for f in loaded["file_list"]))
        self.assertTrue(loaded["metadata"].get("name"))

    async def test_load_example_rejects_traversal(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            await load_example("..%2F..%2Fetc")
        self.assertEqual(ctx.exception.status_code, 400)


class TestLLMClient(unittest.IsolatedAsyncioTestCase):
    """DashScope 直调薄封装：文本提取 / 重试策略 / JSON mode / enable_thinking。"""

    @staticmethod
    def _resp(text="ok", status=200, code=None, message=None):
        return SimpleNamespace(
            status_code=status, code=code, message=message,
            output={"text": text},
        )

    def test_sync_call_returns_text(self):
        client = LLMClient(api_key="sk-test-1234", model="qwen-plus")
        with patch("llm_client.dashscope.Generation.call", return_value=self._resp("hello")) as mock_call:
            text = client.sync_call("sys", "hi")
        self.assertEqual(text, "hello")
        # 参数对齐：system/user 消息 + temperature 0.2 + api_key 直传。
        kwargs = mock_call.call_args.kwargs
        self.assertEqual(kwargs["model"], "qwen-plus")
        self.assertEqual(kwargs["messages"][0], {"role": "system", "content": "sys"})
        self.assertEqual(kwargs["messages"][1], {"role": "user", "content": "hi"})
        self.assertEqual(kwargs["temperature"], 0.2)
        self.assertEqual(kwargs["api_key"], "sk-test-1234")
        self.assertNotIn("response_format", kwargs)  # 默认非 JSON mode

    def test_non_200_raises_runtime_error(self):
        client = LLMClient(api_key="sk-test-1234", model="qwen-plus", max_attempts=1)
        with patch("llm_client.dashscope.Generation.call",
                   return_value=self._resp(status=400, code="InvalidParam", message="bad request")):
            with self.assertRaises(RuntimeError):
                client.sync_call("sys", "hi")

    def test_retry_on_connection_error(self):
        client = LLMClient(api_key="sk-test-1234", model="qwen-plus", max_attempts=3)
        with (
            patch("llm_client.dashscope.Generation.call", side_effect=ConnectionError("connection reset")) as mock_call,
            patch("llm_client.time.sleep"),  # 跳过真实退避等待
        ):
            with self.assertRaises(ConnectionError):
                client.sync_call("sys", "hi")
        self.assertEqual(mock_call.call_count, 3)  # 网络类异常重试到上限

    def test_retry_then_success(self):
        client = LLMClient(api_key="sk-test-1234", model="qwen-plus", max_attempts=3)
        with (
            patch("llm_client.dashscope.Generation.call",
                  side_effect=[ConnectionError("temporary failure"), self._resp("ok")]) as mock_call,
            patch("llm_client.time.sleep"),
        ):
            text = client.sync_call("sys", "hi")
        self.assertEqual(text, "ok")
        self.assertEqual(mock_call.call_count, 2)  # 第二次成功，不再重试

    def test_non_retryable_error_raised_immediately(self):
        client = LLMClient(api_key="sk-test-1234", model="qwen-plus")
        with patch("llm_client.dashscope.Generation.call", side_effect=RuntimeError("invalid argument")) as mock_call:
            with self.assertRaises(RuntimeError):
                client.sync_call("sys", "hi")
        self.assertEqual(mock_call.call_count, 1)  # 解析/业务类错误不重试

    def test_json_mode_sets_response_format(self):
        client = LLMClient(api_key="sk-test-1234", model="qwen-plus")
        with patch("llm_client.dashscope.Generation.call", return_value=self._resp('{"a":1}')) as mock_call:
            text = client.sync_call("sys", "hi", json_mode=True)
        self.assertEqual(text, '{"a":1}')
        self.assertEqual(mock_call.call_args.kwargs["response_format"], {"type": "json_object"})

    def test_choices_structure_supported(self):
        # preview 系列模型：顶层 output.text 为 None，回答在 choices[0].message.content。
        client = LLMClient(api_key="k", model="qwen3.7-max-preview")
        resp = SimpleNamespace(
            status_code=200, code=None, message=None,
            output={"text": None, "finish_reason": "stop", "choices": [
                {"message": {"role": "assistant", "content": "hello-from-choices"}},
            ]},
        )
        with patch("llm_client.dashscope.Generation.call", return_value=resp):
            text = client.sync_call("sys", "hi")
        self.assertEqual(text, "hello-from-choices")

    def test_empty_text_raises_runtime_error(self):
        # 200 但两种结构都取不到文本 → 视为失败（不静默返回 None）。
        client = LLMClient(api_key="k", model="m", max_attempts=1)
        resp = SimpleNamespace(status_code=200, code=None, message=None,
                               output={"text": None, "choices": []})
        with patch("llm_client.dashscope.Generation.call", return_value=resp):
            with self.assertRaises(RuntimeError):
                client.sync_call("sys", "hi")

    def test_deepseek_model_routes_to_deepseek_api(self):
        # deepseek- 前缀模型 → DeepSeek OpenAI 兼容 API，解析 choices[0].message.content；
        # key 优先用实例持有的（前端直传）。
        client = LLMClient(api_key="sk-frontend-ds", model="deepseek-chat")
        fake_resp = SimpleNamespace(
            status_code=200,
            json=lambda: {"choices": [{"message": {"role": "assistant", "content": "deepseek-reply"}}]},
            text="",
        )
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-env-ds"}, clear=False):
            with patch("llm_client.requests.post", return_value=fake_resp) as mock_post:
                text = client.sync_call("sys", "hi")
        self.assertEqual(text, "deepseek-reply")
        # 请求打到 DeepSeek URL，Bearer 用前端直传的 key（不打印 key）。
        args, kwargs = mock_post.call_args
        self.assertIn("api.deepseek.com", args[0])
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer sk-frontend-ds")
        self.assertEqual(kwargs["json"]["model"], "deepseek-chat")

    def test_deepseek_api_key_field_takes_priority(self):
        # 双输入框场景：deepseek_api_key 字段优先于通用 api_key 与环境变量。
        client = LLMClient(api_key="sk-qwen", model="deepseek-chat", deepseek_api_key="sk-ds-front")
        fake_resp = SimpleNamespace(
            status_code=200,
            json=lambda: {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            text="",
        )
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-ds-env"}, clear=False):
            with patch("llm_client.requests.post", return_value=fake_resp) as mock_post:
                client.sync_call("sys", "hi")
        self.assertEqual(mock_post.call_args.kwargs["headers"]["Authorization"], "Bearer sk-ds-front")

    def test_deepseek_env_fallback_when_no_instance_key(self):
        # 实例无 key（前端未传/为空）时回退 DEEPSEEK_API_KEY 环境变量。
        client = LLMClient(api_key="", model="deepseek-chat")
        fake_resp = SimpleNamespace(
            status_code=200,
            json=lambda: {"choices": [{"message": {"role": "assistant", "content": "env-key-reply"}}]},
            text="",
        )
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-env-ds"}, clear=False):
            with patch("llm_client.requests.post", return_value=fake_resp) as mock_post:
                text = client.sync_call("sys", "hi")
        self.assertEqual(text, "env-key-reply")
        self.assertEqual(mock_post.call_args.kwargs["headers"]["Authorization"], "Bearer sk-env-ds")

    def test_deepseek_json_mode_passthrough(self):
        client = LLMClient(api_key="k", model="deepseek-chat")
        fake_resp = SimpleNamespace(
            status_code=200,
            json=lambda: {"choices": [{"message": {"role": "assistant", "content": '{"a":1}'}}]},
            text="",
        )
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-ds"}, clear=False):
            with patch("llm_client.requests.post", return_value=fake_resp) as mock_post:
                text = client.sync_call("sys", "hi", json_mode=True)
        self.assertEqual(text, '{"a":1}')
        self.assertEqual(mock_post.call_args.kwargs["json"]["response_format"], {"type": "json_object"})

    def test_deepseek_missing_api_key_raises(self):
        # 实例与环境变量都没有 key → 明确报错（不静默、不走 DashScope）。
        client = LLMClient(api_key="", model="deepseek-chat", max_attempts=1)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                client.sync_call("sys", "hi")
        self.assertIn("API key", str(ctx.exception))

    def test_deepseek_non_200_raises(self):
        client = LLMClient(api_key="k", model="deepseek-chat", max_attempts=1)
        fake_resp = SimpleNamespace(status_code=429, text='{"error":{"message":"rate limited"}}', json=lambda: {})
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-ds"}, clear=False):
            with patch("llm_client.requests.post", return_value=fake_resp):
                with self.assertRaises(RuntimeError) as ctx:
                    client.sync_call("sys", "hi")
        self.assertIn("429", str(ctx.exception))

    def test_non_deepseek_model_ignores_deepseek_env(self):
        # qwen 模型不读 DEEPSEEK_API_KEY：即使设置了也走 DashScope（dashscope 被 mock）。
        client = LLMClient(api_key="sk-qwen", model="qwen-plus")
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-ds"}, clear=False):
            with patch("llm_client.dashscope.Generation.call", return_value=self._resp("qwen-reply")) as mock_call:
                text = client.sync_call("sys", "hi")
        self.assertEqual(text, "qwen-reply")
        self.assertEqual(mock_call.call_args.kwargs["api_key"], "sk-qwen")

    def test_glm_model_routes_to_zhipu_and_prefers_env_key(self):
        client = LLMClient(api_key="caller-key", model="glm-4-flash")
        fake_resp = SimpleNamespace(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "glm-reply"}}]},
            text="",
        )
        with (
            patch.dict(os.environ, {"ZHIPU_API_KEY": "zhipu-env-key"}, clear=False),
            patch("llm_client.requests.post", return_value=fake_resp) as mock_post,
            patch("llm_client.dashscope.Generation.call") as mock_dashscope,
        ):
            text = client.sync_call("system", "user")

        self.assertEqual(text, "glm-reply")
        args, kwargs = mock_post.call_args
        self.assertIn("open.bigmodel.cn", args[0])
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer zhipu-env-key")
        self.assertEqual(kwargs["json"]["model"], "glm-4-flash")
        self.assertEqual(kwargs["json"]["messages"], [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ])
        self.assertEqual(kwargs["timeout"], GLM_REQUEST_TIMEOUT)
        mock_dashscope.assert_not_called()

    def test_glm_falls_back_to_caller_key_and_supports_json_mode(self):
        client = LLMClient(api_key="caller-key", model="glm-4-flash")
        fake_resp = SimpleNamespace(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": '{"ok":true}'}}]},
            text="",
        )
        with (
            patch.dict(os.environ, {"ZHIPU_API_KEY": ""}, clear=False),
            patch("llm_client.requests.post", return_value=fake_resp) as mock_post,
        ):
            text = client.sync_call("system", "user", json_mode=True)

        self.assertEqual(text, '{"ok":true}')
        kwargs = mock_post.call_args.kwargs
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer caller-key")
        self.assertEqual(
            kwargs["json"]["response_format"], {"type": "json_object"}
        )

    def test_glm_missing_key_raises_without_network(self):
        client = LLMClient(api_key="", model="glm-4-flash", max_attempts=1)
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("llm_client.requests.post") as mock_post,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                client.sync_call("system", "user")
        self.assertIn("API key", str(ctx.exception))
        mock_post.assert_not_called()

    def test_glm_non_200_error_does_not_echo_response_body(self):
        client = LLMClient(api_key="caller-key", model="glm-4-flash", max_attempts=1)
        fake_resp = SimpleNamespace(
            status_code=401,
            text="provider echoed sensitive-request-material",
            json=lambda: {},
        )
        with (
            patch.dict(os.environ, {"ZHIPU_API_KEY": ""}, clear=False),
            patch("llm_client.requests.post", return_value=fake_resp),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                client.sync_call("system", "user")
        message = str(ctx.exception)
        self.assertIn("401", message)
        self.assertNotIn("sensitive-request-material", message)

    def test_glm_transient_http_status_retries_without_echoing_body(self):
        client = LLMClient(api_key="caller-key", model="glm-4-flash", max_attempts=2)
        failed = SimpleNamespace(
            status_code=500,
            text="provider echoed sensitive-request-material",
            json=lambda: {},
        )
        succeeded = SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"choices": [{"message": {"content": "recovered"}}]},
        )
        with (
            patch.dict(os.environ, {"ZHIPU_API_KEY": ""}, clear=False),
            patch(
                "llm_client.requests.post", side_effect=[failed, succeeded]
            ) as mock_post,
            patch("llm_client.time.sleep"),
        ):
            text = client.sync_call("system", "user")
        self.assertEqual(text, "recovered")
        self.assertEqual(mock_post.call_count, 2)

    def test_non_glm_model_ignores_zhipu_env(self):
        client = LLMClient(api_key="qwen-key", model="qwen-plus")
        with (
            patch.dict(os.environ, {"ZHIPU_API_KEY": "zhipu-env-key"}, clear=False),
            patch(
                "llm_client.dashscope.Generation.call",
                return_value=self._resp("qwen-reply"),
            ) as mock_dashscope,
            patch("llm_client.requests.post") as mock_post,
        ):
            text = client.sync_call("system", "user")
        self.assertEqual(text, "qwen-reply")
        self.assertEqual(mock_dashscope.call_args.kwargs["api_key"], "qwen-key")
        mock_post.assert_not_called()

    def test_enable_thinking_passed_only_when_on(self):
        on = LLMClient(api_key="k", model="m", enable_thinking=True)
        off = LLMClient(api_key="k", model="m", enable_thinking=False)
        with patch("llm_client.dashscope.Generation.call", return_value=self._resp()) as mock_call:
            on.sync_call("s", "u")
            self.assertTrue(mock_call.call_args.kwargs.get("enable_thinking"))
        with patch("llm_client.dashscope.Generation.call", return_value=self._resp()) as mock_call:
            off.sync_call("s", "u")
            self.assertNotIn("enable_thinking", mock_call.call_args.kwargs)

    async def test_ask_bridges_through_thread(self):
        client = LLMClient(api_key="k", model="m")
        with patch("llm_client.asyncio.to_thread", new=AsyncMock(return_value="hello")) as mock_bridge:
            text = await client.ask("sys", "hi")
        self.assertEqual(text, "hello")
        # 线程桥接必须把 (sync_call, system, user) 传过去。
        self.assertEqual(mock_bridge.call_args[0][0], client.sync_call)
        self.assertEqual(mock_bridge.call_args[0][1], "sys")
        self.assertEqual(mock_bridge.call_args[0][2], "hi")


class _FakeCodexProcess:
    """In-memory newline JSON process used by Codex App Server route tests."""

    def __init__(self, messages):
        self.stdin = io.StringIO()
        encoded = "".join(json.dumps(item) + "\n" for item in messages)
        self.stdout = io.StringIO(encoded)
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode or 0


class TestCodexAppServerRoute(unittest.TestCase):
    """Codex ChatGPT 路由：认证、模型、隔离、结构化输出与 fail-closed。"""

    @staticmethod
    def _model_entry(model="gpt-5.6-sol", efforts=None):
        efforts = efforts or ["low", "medium", "high"]
        return {
            "id": model,
            "model": model,
            "displayName": "GPT test",
            "hidden": False,
            "isDefault": True,
            "defaultReasoningEffort": "low",
            "supportedReasoningEfforts": [
                {"reasoningEffort": effort, "description": effort}
                for effort in efforts
            ],
        }

    @classmethod
    def _success_messages(cls, text='{"result":"{\\"ok\\":true}"}'):
        return [
            {"id": 1, "result": {"userAgent": "test"}},
            {"id": 2, "result": {"account": {"type": "chatgpt", "planType": "pro"}}},
            {"id": 3, "result": {"data": [cls._model_entry()], "nextCursor": None}},
            {
                "id": 4,
                "result": {
                    "thread": {
                        "id": "thr_test",
                        "sessionId": "thr_test",
                        "ephemeral": True,
                    }
                },
            },
            {"id": 5, "result": {"turn": {"id": "turn_test", "status": "inProgress"}}},
            {
                "method": "item/completed",
                "params": {
                    "item": {
                        "id": "msg_test",
                        "type": "agentMessage",
                        "phase": "final_answer",
                        "text": text,
                    }
                },
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"id": "turn_test", "status": "completed"}},
            },
        ]

    @staticmethod
    def _sent_messages(process):
        return [json.loads(line) for line in process.stdin.getvalue().splitlines()]

    def _call_with_process(self, process, *, json_mode=True, env=None):
        client = LLMClient(
            api_key="provider-secret-must-not-propagate",
            model="gpt-5.6-sol",
            max_attempts=1,
        )
        env = {"CODEX_REASONING_EFFORT": "", **(env or {})}
        with (
            patch.dict(os.environ, env, clear=False),
            patch("llm_client._resolve_codex_cli", return_value="/fake/codex"),
            patch("llm_client.subprocess.Popen", return_value=process) as mock_popen,
            patch("llm_client.dashscope.Generation.call") as mock_dashscope,
            patch("llm_client.requests.post") as mock_http,
        ):
            text = client.sync_call("system-role", "user-prompt", json_mode=json_mode)
        mock_dashscope.assert_not_called()
        mock_http.assert_not_called()
        return text, mock_popen

    def test_gpt_prefix_is_fixed_codex_route(self):
        self.assertTrue(_is_codex_model("gpt-5.6-sol"))
        self.assertFalse(_is_codex_model("qwen-plus"))
        self.assertFalse(_is_codex_model("deepseek-chat"))
        self.assertFalse(_is_codex_model("glm-4-flash"))

    def test_chatgpt_route_is_ephemeral_text_only_and_uses_output_schema(self):
        process = _FakeCodexProcess(self._success_messages())
        env = {
            "OPENAI_API_KEY": "openai-platform-secret",
            "QWEN_API_KEY": "qwen-environment-secret",
            "DASHSCOPE_API_KEY": "dashscope-secret",
            "DEEPSEEK_API_KEY": "deepseek-secret",
            "ZHIPU_API_KEY": "zhipu-secret",
            "CODEX_REASONING_EFFORT": "medium",
        }
        text, mock_popen = self._call_with_process(process, env=env)
        self.assertEqual(text, '{"ok":true}')

        sent = self._sent_messages(process)
        methods = [item.get("method") for item in sent]
        self.assertEqual(
            methods,
            ["initialize", "initialized", "account/read", "model/list", "thread/start", "turn/start"],
        )
        thread_params = next(item["params"] for item in sent if item.get("method") == "thread/start")
        self.assertTrue(thread_params["ephemeral"])
        self.assertEqual(thread_params["approvalPolicy"], "never")
        self.assertEqual(thread_params["sandbox"], "read-only")
        self.assertEqual(thread_params["config"], {"mcp_servers": {}})

        turn_params = next(item["params"] for item in sent if item.get("method") == "turn/start")
        self.assertEqual(turn_params["effort"], "medium")
        self.assertEqual(turn_params["sandboxPolicy"], {"type": "readOnly", "networkAccess": False})
        self.assertEqual(turn_params["outputSchema"]["type"], "object")
        self.assertFalse(turn_params["outputSchema"]["additionalProperties"])
        self.assertEqual(turn_params["outputSchema"]["required"], ["result"])

        popen_env = mock_popen.call_args.kwargs["env"]
        for key in (
            "OPENAI_API_KEY",
            "QWEN_API_KEY",
            "DASHSCOPE_API_KEY",
            "DEEPSEEK_API_KEY",
            "ZHIPU_API_KEY",
        ):
            self.assertNotIn(key, popen_env)
        self.assertNotIn("provider-secret-must-not-propagate", process.stdin.getvalue())

    def test_plain_text_turn_omits_output_schema(self):
        process = _FakeCodexProcess(self._success_messages(text="plain reply"))
        text, _ = self._call_with_process(process, json_mode=False)
        self.assertEqual(text, "plain reply")
        sent = self._sent_messages(process)
        turn_params = next(item["params"] for item in sent if item.get("method") == "turn/start")
        self.assertNotIn("outputSchema", turn_params)

    def test_api_key_login_fails_before_prompt_submission(self):
        process = _FakeCodexProcess([
            {"id": 1, "result": {}},
            {"id": 2, "result": {"account": {"type": "apiKey"}}},
        ])
        client = LLMClient(api_key="provider-secret", model="gpt-5.6-sol", max_attempts=1)
        with (
            patch("llm_client._resolve_codex_cli", return_value="/fake/codex"),
            patch("llm_client.subprocess.Popen", return_value=process),
        ):
            with self.assertRaisesRegex(RuntimeError, "ChatGPT login required"):
                client.sync_call("system-secret", "prompt-secret")
        sent = self._sent_messages(process)
        self.assertNotIn("thread/start", [item.get("method") for item in sent])
        self.assertNotIn("prompt-secret", process.stdin.getvalue())

    def test_unavailable_model_fails_before_prompt_submission(self):
        process = _FakeCodexProcess([
            {"id": 1, "result": {}},
            {"id": 2, "result": {"account": {"type": "chatgpt"}}},
            {"id": 3, "result": {"data": [self._model_entry("gpt-other")] }},
        ])
        client = LLMClient(api_key="", model="gpt-5.6-sol", max_attempts=1)
        with (
            patch("llm_client._resolve_codex_cli", return_value="/fake/codex"),
            patch("llm_client.subprocess.Popen", return_value=process),
        ):
            with self.assertRaisesRegex(RuntimeError, "not available"):
                client.sync_call("system-secret", "prompt-secret")
        self.assertNotIn("prompt-secret", process.stdin.getvalue())

    def test_unsupported_reasoning_effort_fails_closed(self):
        process = _FakeCodexProcess([
            {"id": 1, "result": {}},
            {"id": 2, "result": {"account": {"type": "chatgpt"}}},
            {"id": 3, "result": {"data": [self._model_entry(efforts=["low"])]}},
        ])
        client = LLMClient(api_key="", model="gpt-5.6-sol", max_attempts=1)
        with (
            patch.dict(os.environ, {"CODEX_REASONING_EFFORT": "ultra"}, clear=False),
            patch("llm_client._resolve_codex_cli", return_value="/fake/codex"),
            patch("llm_client.subprocess.Popen", return_value=process),
        ):
            with self.assertRaisesRegex(RuntimeError, "not supported"):
                client.sync_call("system", "prompt")

    def test_forbidden_tool_item_aborts_generation(self):
        messages = self._success_messages()
        messages[5] = {
            "method": "item/started",
            "params": {
                "item": {
                    "id": "cmd_test",
                    "type": "commandExecution",
                    "command": "printenv",
                }
            },
        }
        process = _FakeCodexProcess(messages)
        client = LLMClient(api_key="", model="gpt-5.6-sol", max_attempts=1)
        with (
            patch("llm_client._resolve_codex_cli", return_value="/fake/codex"),
            patch("llm_client.subprocess.Popen", return_value=process),
        ):
            with self.assertRaisesRegex(RuntimeError, "forbidden tool"):
                client.sync_call("system", "prompt")

    def test_server_error_body_is_not_exposed(self):
        process = _FakeCodexProcess([
            {"id": 1, "result": {}},
            {"id": 2, "error": {"message": "provider echoed prompt-secret"}},
        ])
        client = LLMClient(api_key="", model="gpt-5.6-sol", max_attempts=1)
        with (
            patch("llm_client._resolve_codex_cli", return_value="/fake/codex"),
            patch("llm_client.subprocess.Popen", return_value=process),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                client.sync_call("system", "prompt-secret")
        self.assertNotIn("prompt-secret", str(ctx.exception))

    def test_app_server_retry_notification_does_not_abort_turn(self):
        messages = self._success_messages()
        messages.insert(5, {
            "method": "error",
            "params": {
                "willRetry": True,
                "error": {"message": "temporary upstream issue"},
            },
        })
        process = _FakeCodexProcess(messages)
        text, _ = self._call_with_process(process)
        self.assertEqual(text, '{"ok":true}')


if __name__ == "__main__":
    unittest.main()
