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
import json
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

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
    """Verify the final batch of a sync Qwen response generator is extracted."""

    # 同步生成器取最后一轮的 assistant 文本（前面若干批是流式中间结果）。
    def test_final_batch_text_is_extracted(self):
        agent = FakeAgent([
            [{"role": "assistant", "content": "partial stream"}],
            [{"role": "assistant", "content": ""}],
            [{"role": "assistant", "content": "final answer"}],
        ])
        opt = make_optimizer()
        self.assertEqual(opt._run_agent_sync(agent, "hi"), "final answer")

    # content 是片段列表时拼接成整段文本。
    def test_list_content_is_joined(self):
        agent = FakeAgent([[{"role": "assistant", "content": [{"text": "a"}, {"text": "b"}]}]])
        opt = make_optimizer()
        self.assertEqual(opt._run_agent_sync(agent, "hi"), "ab")

    # reasoning_content（推理过程）必须被忽略，只取 content。
    def test_reasoning_content_is_ignored(self):
        agent = FakeAgent([
            [{"role": "assistant", "content": "only this", "reasoning_content": "hidden"}],
        ])
        opt = make_optimizer()
        self.assertEqual(opt._run_agent_sync(agent, "hi"), "only this")

    # 完全没有文本时抛 RuntimeError，由上层 catch 兜底。
    def test_no_text_raises_runtime_error(self):
        agent = FakeAgent([[{"role": "assistant", "content": ""}]])
        opt = make_optimizer()
        with self.assertRaises(RuntimeError):
            opt._run_agent_sync(agent, "hi")


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
    """Verify tolerant JSON extraction (code fences / extra text)."""

    # 宽容解析 1：模型把 JSON 包在 ```json 代码块里也能提取。
    async def test_code_fenced_json_is_parsed(self):
        payload = json.dumps({"scenarios": [{"id": 1}], "evals": []})
        text = f"Here you go:\n```json\n{payload}\n```\nDone."
        opt = make_optimizer()
        agent = FakeAgent([[{"role": "assistant", "content": text}]])
        result = await opt._ask_json(agent, "prompt")
        self.assertEqual(result["scenarios"], [{"id": 1}])

    # 宽容解析 2：JSON 前后有额外文字也能定位并解析。
    async def test_extra_text_before_json_is_handled(self):
        text = 'Sure!\n{"a": 1}\nmore text'
        opt = make_optimizer()
        agent = FakeAgent([[{"role": "assistant", "content": text}]])
        result = await opt._ask_json(agent, "prompt")
        self.assertEqual(result, {"a": 1})

    # 完全不是 JSON 时返回 fallback（兜底值），而不是抛异常。
    async def test_non_json_uses_fallback(self):
        opt = make_optimizer()
        agent = FakeAgent([[{"role": "assistant", "content": "I cannot do that."}]])
        result = await opt._ask_json(agent, "prompt", fallback={"results": []})
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
        agent = FakeAgent([[{"role": "assistant", "content": json.dumps(valid)}]])
        result = await opt._ask_json(
            agent, "prompt",
            fallback={"diagnosis": "fb", "mutation_strategy": "add_constraint",
                      "target_section": "x", "suggested_change": "y"},
            schema=FailureAnalysis,
        )
        self.assertEqual(result["diagnosis"], "missing examples")

    # 缺少必填字段 → schema 校验失败 → 使用 fallback（结构化输出兜底）。
    async def test_analyst_schema_validation_falls_back(self):
        opt = make_optimizer()
        # Missing required fields -> schema validation must fail -> fallback
        agent = FakeAgent([[{"role": "assistant", "content": json.dumps({"diagnosis": "only"})}]])
        fallback = {"diagnosis": "fb", "mutation_strategy": "add_constraint",
                    "target_section": "x", "suggested_change": "y"}
        result = await opt._ask_json(agent, "prompt", fallback=fallback, schema=FailureAnalysis)
        self.assertEqual(result, fallback)

    # Mutator 返回符合 SkillMutation schema 时正常通过。
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

    # Mutator 返回非 JSON → fallback（此时 new_skill_md 兜底为原内容）。
    async def test_mutator_schema_non_json_falls_back(self):
        opt = make_optimizer()
        agent = FakeAgent([[{"role": "assistant", "content": "not json at all"}]])
        fallback = {"description": "fb", "reasoning": "", "new_skill_md": "orig"}
        result = await opt._ask_json(agent, "prompt", fallback=fallback, schema=SkillMutation)
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

    # 默认模型必须是 qwen-plus。
    def test_model_defaults_to_qwen_plus(self):
        opt = make_optimizer()
        self.assertEqual(opt.model, "qwen-plus")

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
        agent = FakeAgent([
            [{"role": "assistant", "content": "some output"}],
            [{"role": "assistant", "content": json.dumps({"results": [
                {"eval_id": 1, "passed": True},
                {"eval_id": 2, "passed": False},
            ]})}],
        ])
        result = await opt._score_skill(skill_md, scenarios, evals, agent=agent)
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
        agent = FakeAgent([
            [{"role": "assistant", "content": "some output"}],
            [{"role": "assistant", "content": json.dumps({"results": [
                {"eval_id": 1, "passed": True},
                {"eval_id": 2, "passed": False},
            ]})}],
        ])
        result = await opt._score_skill("# S", scenarios, evals, agent=agent)
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
        agent = FakeAgent([
            [{"role": "assistant", "content": "some output"}],
            [{"role": "assistant", "content": json.dumps({"results": [
                {"eval_id": 1, "passed": True},
                {"eval_id": 2, "passed": False},
            ]})}],
        ])
        result = await opt._score_skill("# S", scenarios, evals, agent=agent)
        # 与旧行为一致：全走 LLM 打分，passed 来自 FakeAgent 返回的完整结果。
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

    async def test_list_examples_returns_four_packs(self):
        ex = await list_examples()
        self.assertEqual(len(ex["examples"]), 4)
        paths = {e["path"] for e in ex["examples"]}
        self.assertIn("project-graveyard", paths)
        self.assertIn("commit-archaeologist", paths)

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


if __name__ == "__main__":
    unittest.main()
