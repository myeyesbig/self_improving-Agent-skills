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

import json
import os
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


def make_optimizer(api_key="sk-test-1234"):
    # 构造测试用的 SkillOptimizer；默认使用假密钥，绝不触碰真实凭据。
    return SkillOptimizer(api_key=api_key)


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
