# =============================================================================
# 【文件头】qwen_optimizer.py —— 后端优化算法的核心
# 职责：用三角色（Executor 评分 / Analyst 诊断 / Mutator 变异）协作，对一个
#       SKILL.md 反复执行"评估 → 诊断 → 修改 → 复评"，最终给出改进后的技能文件。
#       编排层已显式化为 LangGraph 状态图（optimize_graph.py）；本文件仍持有
#       全部算法逻辑（评分/诊断/变异/回归守卫/经验库）与 LLM 调用薄封装。
# 接收：技能文件内容 dict（含 SKILL.md 与 references/ 参考文件）、测试场景
#       scenarios、评估标准 evals、最大轮数 max_rounds，以及进度回调 callback。
# 输出：优化结果 dict（baseline_score / final_score / improved_skill_md /
#       score_history / mutation_log），并通过 callback 逐条推送进度事件。
# 建议先看：SkillOptimizer.optimize()（切图执行）→ _score_skill()（评分）→
#       _run_agent_sync()（模型调用的线程桥接）。
# 【注意】本文件所有调用都是"同步 + 线程桥接"：DashScope SDK 是同步的，
#       必须放进工作线程（llm_client 的 to_thread），避免阻塞 FastAPI 事件循环。
# =============================================================================

"""Multi-Agent Skill Optimizer with fixed Codex/Qwen/DeepSeek/GLM routes.

Three stateless model roles work together to improve agent skills:
  Executor: runs the skill against test scenarios, scores outputs, analyzes skills
  Analyst: diagnoses why evals failed, picks a mutation strategy
  Mutator: makes one targeted fix per round
"""

import asyncio
import datetime
import json
import math
import os
import re
from typing import Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from llm_client import DEFAULT_MODEL, LLMClient


class _RoleMessage:
    """三角色 system prompt 的轻量持有者（替代 Qwen-Agent Assistant）。

    阶段 4 移除 qwen-agent 后，self.executor/analyst/mutator 等属性仍保留，
    但只承载 system_message —— LLM 调用统一走 llm_client（无状态），
    不再需要 Assistant 实例或每槽独立实例池的线程安全约定。
    """

    def __init__(self, name: str, system_message: str, role: Optional[str] = None):
        self.name = name
        self.system_message = system_message
        # ``role`` stays stable for parallel slot names such as ``mutator-1``.
        # It is used only to select the role-specific LLM client.
        self.role = role or name


# -- Mutation strategy templates ----------------------------------------------
# 【主流程】C1 可配置变异策略：每项描述"何时用"以及给 Mutator 的提示词片段。
# 策略池可被环境变量 MUTATION_STRATEGIES 或请求字段 strategy_pool 白名单过滤。
# 原有 4 个策略（add_example / add_constraint / restructure / add_edge_case）
# 是行为基线，新增 3 个扩展策略只是更多选择，不改变"一次变异只改一处"。

STRATEGY_TEMPLATES: Dict[str, dict] = {
    "add_example": {
        "description": "Add a concrete example to illustrate usage",
        "when": "The skill lacks examples, or outputs fail because users need a worked example",
        "mutator_hint": "Add ONE concrete example that demonstrates the expected output format.",
    },
    "add_constraint": {
        "description": "Add an explicit constraint or boundary",
        "when": "Outputs overstep scope, or a failing criterion needs a hard rule",
        "mutator_hint": "Add ONE explicit constraint or boundary that addresses the failure.",
    },
    "restructure": {
        "description": "Reorder or reorganize sections for clarity",
        "when": "The skill is confusingly ordered, and eval failures trace to missing guidance",
        "mutator_hint": "Reorder or reorganize ONE section so the guidance flows logically.",
    },
    "add_edge_case": {
        "description": "Add handling for a specific edge case",
        "when": "Failures occur on unusual inputs that the skill does not cover",
        "mutator_hint": "Add ONE edge-case rule that covers the failing input pattern.",
    },
    "add_reference": {
        "description": "Point to a reference file or external resource",
        "when": "The skill needs deeper context that belongs in references/",
        "mutator_hint": "Add ONE pointer to an existing reference file or a short inline reference.",
    },
    "rewrite_section": {
        "description": "Rewrite a single section for precision",
        "when": "A specific section is vague and causes repeated failures",
        "mutator_hint": "Rewrite exactly ONE section to be precise, keeping every other section unchanged.",
    },
    "fix_format": {
        "description": "Fix formatting, frontmatter, or YAML issues",
        "when": "Frontmatter or formatting problems break parsing or execution",
        "mutator_hint": "Fix exactly ONE formatting or frontmatter issue, keeping content otherwise intact.",
    },
}

DEFAULT_STRATEGY_POOL = list(STRATEGY_TEMPLATES.keys())

# 【目标 4】弱维度 → 优先尝试的单点编辑策略。只做排序提示，仍受调用方
# strategy_pool 白名单约束，不会创建新角色或绕过回归/严格提升门槛。
WEAK_DIMENSION_STRATEGIES: Dict[str, List[str]] = {
    "correctness": ["add_constraint", "add_edge_case", "add_example"],
    "clarity": ["rewrite_section", "restructure", "add_example"],
    "executability": ["add_example", "add_constraint", "restructure"],
    "maintainability": ["restructure", "rewrite_section", "add_reference"],
    "quality": ["rewrite_section", "add_example", "add_constraint"],
    "semantic_preservation": ["add_constraint", "rewrite_section"],
}


def _active_strategy_pool(strategy_pool: Optional[List[str]] = None) -> List[str]:
    """Resolve the effective strategy pool: request override, env, or all templates."""
    if strategy_pool:
        valid = [s for s in strategy_pool if s in STRATEGY_TEMPLATES]
        if valid:
            return valid
    env_pool = os.getenv("MUTATION_STRATEGIES")
    if env_pool:
        valid = [s.strip() for s in env_pool.split(",") if s.strip() in STRATEGY_TEMPLATES]
        if valid:
            return valid
    return list(DEFAULT_STRATEGY_POOL)


class StopOptimizationError(Exception):
    """Raised between rounds when the user requests a cooperative stop."""


# -- Model-facing prompt policies --------------------------------------------
# 以下三个 system prompt 保持英文原版（与基线 1cd7b9a 一致）；JSON 字段名、
# 策略枚举、YAML 键和代码标识是程序协议，不参与本地化。

EXECUTOR_SYSTEM_PROMPT = (
    "You are a versatile skill execution agent. You have three modes:\n\n"
    "1. EXECUTE MODE: Given a skill's instructions and a user request, "
    "produce the output that skill would generate. Follow the instructions "
    "exactly. No meta-commentary.\n\n"
    "2. ANALYZE MODE: Given a skill definition, generate test scenarios "
    "and evaluation criteria. Return valid JSON.\n\n"
    "3. SCORE MODE: Given an output and evaluation criteria, score the "
    "output against each criterion. Return valid JSON."
)

ANALYST_SYSTEM_PROMPT = (
    "You diagnose why agent skill evaluations fail. "
    "Given failed eval results, identify the root cause and suggest "
    "a specific fix. Pick one mutation_strategy from the allowed list "
    "you are given in the prompt."
)

MUTATOR_SYSTEM_PROMPT = (
    "You edit agent skill files. Given a SKILL.md and a diagnosis, "
    "make exactly ONE targeted change. Keep the YAML frontmatter and "
    "overall structure intact. Return the complete updated SKILL.md."
)


# -- Pydantic schemas for structured agent output ----------------------------
# 【主流程】schema（结构约定）用来约束模型返回的 JSON 形状：Analyst 必须返回
# 诊断 + 修改策略，Mutator 必须返回完整的新 SKILL.md。校验失败时会走 fallback。

class FailureAnalysis(BaseModel):
    diagnosis: str = Field(description="Root cause of failures")
    mutation_strategy: str = Field(
        description="One of the allowed mutation strategies from the prompt"
    )
    target_section: str = Field(description="Which part of the skill to change")
    suggested_change: str = Field(description="What specific change to make")
    target_dimension: Optional[str] = Field(
        default=None, description="Optional weak evaluation dimension targeted by this fix"
    )


class SkillMutation(BaseModel):
    description: str = Field(description="Short description of the change made")
    reasoning: str = Field(description="Why this change should help")
    new_skill_md: str = Field(description="The full updated SKILL.md content")


# -- Optimizer ---------------------------------------------------------------
# 【主流程】SkillOptimizer 持有三个助手实例和一个共享的 LLM 配置（模型 + 密钥）。
# 密钥只存在于这个配置对象里，不写进程环境变量、不落日志、不存 session。

class SkillOptimizer:
    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        parallelism: Optional[int] = None,
        dimension_weights: Optional[Dict[str, float]] = None,
        improvement_threshold: Optional[float] = None,
        regression_check: Optional[bool] = None,
        noise_floor: Optional[float] = None,
        edit_limit: Optional[float] = None,
        patience: Optional[int] = None,
        saturation_exit: Optional[bool] = None,
        final_confirm: Optional[bool] = None,
        candidate_confirm_runs: Optional[int] = None,
        lesson_file: Optional[str] = None,
        lesson_retrieval: Optional[str] = None,
        lesson_threshold: Optional[float] = None,
        lesson_top_k: Optional[int] = None,
        lesson_rerank: Optional[bool] = None,
        lesson_rerank_pool: Optional[int] = None,
        lesson_rag_pipeline: Optional[str] = None,
        lesson_candidate_pool: Optional[int] = None,
        lesson_diversity: Optional[float] = None,
        lesson_dedup_threshold: Optional[float] = None,
        lesson_context_chars: Optional[int] = None,
        lesson_signal_weights: Optional[Dict[str, float]] = None,
        lesson_failure_context: Optional[bool] = None,
        lesson_embed_batch_size: Optional[int] = None,
        lesson_min_gain: Optional[float] = None,
        lesson_min_final: Optional[float] = None,
        tie_dimension_lessons: Optional[bool] = None,
        tie_dimension_min_gain: Optional[float] = None,
        cross_round_tie_dimension_lessons: Optional[bool] = None,
        cross_round_tie_dimension_min_gain: Optional[float] = None,
        weak_dimension_focus: Optional[bool] = None,
        weak_dimension_threshold: Optional[float] = None,
        weak_dimension_max: Optional[int] = None,
        search_policy: Optional[str] = None,
        search_exploration: Optional[float] = None,
        checkpoint_file: Optional[str] = None,
        stop_provider: Optional[Callable[[], bool]] = None,
        deepseek_api_key: Optional[str] = None,
    ):
        # The key is passed directly into the LLM configuration; it is never
        # written to the process environment, stored in sessions, or logged.
        # 【临时默认】生成走本机 Codex App Server 的 gpt-5.6-sol；沿用既有
        # QWEN_MODEL 名称保持部署兼容，可显式覆盖回 Qwen/DeepSeek/GLM。
        self.model = model or os.getenv("QWEN_MODEL", DEFAULT_MODEL)
        self._call_id = 0

        # 【C4】回归保护开关：默认开启，可用 REGRESSION_CHECK=0 关闭做实验。
        if regression_check is None:
            regression_check = os.getenv("REGRESSION_CHECK", "1") != "0"
        self.regression_check = regression_check

        # 【C4】提升阈值：新分必须严格高于"基线分 + 阈值"才保留；默认 0.0
        # 即保留原"严格高于"语义。可用 IMPROVEMENT_THRESHOLD 环境变量覆盖。
        if improvement_threshold is None:
            try:
                improvement_threshold = float(os.getenv("IMPROVEMENT_THRESHOLD", "0.0"))
            except ValueError:
                improvement_threshold = 0.0
        self.improvement_threshold = max(0.0, float(improvement_threshold))

        # 【P1 噪声地板】候选提升必须超过"基线 + 阈值 + 噪声地板"才保留，
        # 防止评分波动被当成真进步；默认 0.0 即保持原"严格高于"语义。
        # 可用 NOISE_FLOOR 环境变量覆盖。
        if noise_floor is None:
            try:
                noise_floor = float(os.getenv("NOISE_FLOOR", "0.0"))
            except ValueError:
                noise_floor = 0.0
        self.noise_floor = max(0.0, float(noise_floor))

        # 【P1 编辑幅度】单次变异相对原文本的变化比例上限，超过即拒绝，
        # 防止大改漂移；默认 0.0 表示关闭。可用 EDIT_LIMIT 环境变量覆盖。
        if edit_limit is None:
            try:
                edit_limit = float(os.getenv("EDIT_LIMIT", "0.0"))
            except ValueError:
                edit_limit = 0.0
        self.edit_limit = max(0.0, float(edit_limit))

        # 【P3 耐心早停】连续 N 轮未提升提前终止；默认 0 表示关闭（只保留
        # 原 100% 早停）。可用 PATIENCE 环境变量覆盖。
        if patience is None:
            try:
                patience = int(os.getenv("PATIENCE", "0"))
            except ValueError:
                patience = 0
        self.patience = max(0, int(patience))

        # 【P3 饱和快速退出】基线无可提升带（100% 或 0 分全失败）时跳过全部
        # 轮次；默认 0 关闭（100% 的旧早停始终生效）。可用 SATURATION_EXIT。
        if saturation_exit is None:
            saturation_exit = os.getenv("SATURATION_EXIT", "0") != "0"
        self.saturation_exit = saturation_exit

        # 【P3 胜出候选复核】complete 前对最终版本独立再评一次，若复核分未
        # 超过基线则回退为不采纳（防单点幸运）；默认 0 关闭。FINAL_CONFIRM。
        if final_confirm is None:
            final_confirm = os.getenv("FINAL_CONFIRM", "0") != "0"
        self.final_confirm = final_confirm

        # 【候选配对复核】默认 0 完整保留单次评分路径。开启后，仅对初评分
        # 已严格提升的 provisional winner 追加 incumbent/challenger 成对复评；
        # 每一对都必须继续严格胜出，因此该门只会拒绝，不会放松接受条件。
        if candidate_confirm_runs is None:
            try:
                candidate_confirm_runs = int(os.getenv("CANDIDATE_CONFIRM_RUNS", "0"))
            except ValueError:
                candidate_confirm_runs = 0
        self.candidate_confirm_runs = max(0, min(int(candidate_confirm_runs), 3))

        # 【P4 跨会话经验库】SKILL_LESSONS_FILE 指向 jsonl/db 文件路径；不设=
        # 关闭。保留的修改会被沉淀为经验，下次优化时注入 Analyst/Mutator。
        if lesson_file is None:
            lesson_file = os.getenv("SKILL_LESSONS_FILE") or None
        self.lesson_file = lesson_file

        # 【RAG 检索模式】LESSON_RETRIEVAL：off（默认，最近 N 条旧行为）/
        # tag（同技能硬过滤）/ semantic（embedding 语义检索）/
        # hybrid（dense+sparse 混合）。非法值回退 off。
        if lesson_retrieval is None:
            lesson_retrieval = os.getenv("LESSON_RETRIEVAL", "off")
        if lesson_retrieval not in ("off", "tag", "semantic", "hybrid"):
            lesson_retrieval = "off"
        self.lesson_retrieval = lesson_retrieval

        # 【RAG 检索参数】semantic 余弦阈值与检索注入条数（默认 0.3 / 5）；
        # LESSON_N 管"读取条数"（off/tag 模式），LESSON_TOP_K 管"检索注入条数"。
        if lesson_threshold is None:
            try:
                lesson_threshold = float(os.getenv("LESSON_THRESHOLD", "0.3"))
            except ValueError:
                lesson_threshold = 0.3
        self.lesson_threshold = max(0.0, min(float(lesson_threshold), 1.0))
        if lesson_top_k is None:
            try:
                lesson_top_k = int(os.getenv("LESSON_TOP_K", "5"))
            except ValueError:
                lesson_top_k = 5
        self.lesson_top_k = max(1, int(lesson_top_k))

        # 【RAG 重排】LESSON_RERANK=1 时对 semantic/hybrid 的候选池调用 DashScope
        # qwen3-rerank 重排后取 top-K；默认 0 关闭（保持现有排序）。候选池大小
        # 由 LESSON_RERANK_POOL 控制（默认 20）。
        if lesson_rerank is None:
            lesson_rerank = os.getenv("LESSON_RERANK", "0") != "0"
        self.lesson_rerank = lesson_rerank
        if lesson_rerank_pool is None:
            try:
                lesson_rerank_pool = int(os.getenv("LESSON_RERANK_POOL", "20"))
            except ValueError:
                lesson_rerank_pool = 20
        self.lesson_rerank_pool = max(2, int(lesson_rerank_pool))

        # 【目标 5：经验 RAG v2】classic（默认）逐字保留原检索路径；只有显式
        # 开启 quality_diverse 才启用多信号质量排序、去重和 MMR 式多样性选择。
        # 所有参数仅影响 lesson store，不改变优化器的严格提升/回归守卫语义。
        if lesson_rag_pipeline is None:
            lesson_rag_pipeline = os.getenv("LESSON_RAG_PIPELINE", "classic")
        if lesson_rag_pipeline not in ("classic", "quality_diverse"):
            lesson_rag_pipeline = "classic"
        self.lesson_rag_pipeline = lesson_rag_pipeline

        if lesson_candidate_pool is None:
            try:
                lesson_candidate_pool = int(os.getenv("LESSON_CANDIDATE_POOL", "20"))
            except ValueError:
                lesson_candidate_pool = 20
        self.lesson_candidate_pool = max(2, int(lesson_candidate_pool))

        if lesson_diversity is None:
            try:
                lesson_diversity = float(os.getenv("LESSON_DIVERSITY", "0.2"))
            except ValueError:
                lesson_diversity = 0.2
        self.lesson_diversity = max(0.0, min(float(lesson_diversity), 1.0))

        if lesson_dedup_threshold is None:
            try:
                lesson_dedup_threshold = float(os.getenv("LESSON_DEDUP_THRESHOLD", "0.92"))
            except ValueError:
                lesson_dedup_threshold = 0.92
        self.lesson_dedup_threshold = max(
            0.0, min(float(lesson_dedup_threshold), 1.0)
        )

        if lesson_context_chars is None:
            try:
                lesson_context_chars = int(os.getenv("LESSON_CONTEXT_CHARS", "6000"))
            except ValueError:
                lesson_context_chars = 6000
        self.lesson_context_chars = max(500, int(lesson_context_chars))

        default_signal_weights = {
            "semantic": 0.35,
            "sparse": 0.15,
            "skill": 0.12,
            "domain": 0.08,
            "dimension": 0.12,
            "quality": 0.13,
            "recency": 0.05,
        }
        if lesson_signal_weights is None:
            env_signal_weights = os.getenv("LESSON_SIGNAL_WEIGHTS")
            if env_signal_weights:
                try:
                    lesson_signal_weights = json.loads(env_signal_weights)
                except (json.JSONDecodeError, TypeError):
                    lesson_signal_weights = None
        valid_signal_weights = {}
        if isinstance(lesson_signal_weights, dict):
            for name in default_signal_weights:
                try:
                    weight = float(lesson_signal_weights.get(name, 0.0))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(weight) and weight >= 0:
                    valid_signal_weights[name] = weight
        weight_total = sum(valid_signal_weights.values())
        if weight_total <= 0:
            self.lesson_signal_weights = default_signal_weights
        else:
            self.lesson_signal_weights = {
                name: valid_signal_weights.get(name, 0.0) / weight_total
                for name in default_signal_weights
            }

        # 【后续改进：失败感知查询】默认关闭。开启后，RAG 查询会加入失败
        # eval 的 criterion/question/pass_condition 及对应失败场景；内容仅用于
        # 当次 embedding/rerank 请求，不进入 lesson store 或任何返回结构。
        if lesson_failure_context is None:
            lesson_failure_context = os.getenv("LESSON_FAILURE_CONTEXT", "0") != "0"
        self.lesson_failure_context = lesson_failure_context

        # 【后续改进：批量 embedding】官方 text-embedding-v3 同步接口每批
        # 最多 10 条。默认 1 保持原逐条调用；显式调大可降低经验库冷启动请求数。
        if lesson_embed_batch_size is None:
            try:
                lesson_embed_batch_size = int(os.getenv("LESSON_EMBED_BATCH_SIZE", "1"))
            except ValueError:
                lesson_embed_batch_size = 1
        self.lesson_embed_batch_size = max(1, min(int(lesson_embed_batch_size), 10))

        # 【经验沉淀质量门槛】LESSON_MIN_GAIN（提升幅度）/ LESSON_MIN_FINAL
        # （最终水位），OR 语义：任一启用的维度达标才沉淀经验；全 0=关闭=
        # 旧行为（任何 kept 都沉淀）。只影响经验库质量，不动优化保留语义。
        if lesson_min_gain is None:
            try:
                lesson_min_gain = float(os.getenv("LESSON_MIN_GAIN", "0.0"))
            except ValueError:
                lesson_min_gain = 0.0
        self.lesson_min_gain = max(0.0, float(lesson_min_gain))
        if lesson_min_final is None:
            try:
                lesson_min_final = float(os.getenv("LESSON_MIN_FINAL", "0.0"))
            except ValueError:
                lesson_min_final = 0.0
        self.lesson_min_final = max(0.0, float(lesson_min_final))

        # 【目标 3：平局维度经验】默认关闭。开启后，总分与胜者相同但在至少
        # 一个评分维度更优的未选候选会沉淀“维度优势”元数据；候选本身仍不
        # 被保留，严格提升与单一 current_md 不变量完全不动。
        if tie_dimension_lessons is None:
            tie_dimension_lessons = os.getenv("TIE_DIMENSION_LESSONS", "0") != "0"
        self.tie_dimension_lessons = tie_dimension_lessons
        if tie_dimension_min_gain is None:
            try:
                tie_dimension_min_gain = float(os.getenv("TIE_DIMENSION_MIN_GAIN", "0.0"))
            except ValueError:
                tie_dimension_min_gain = 0.0
        self.tie_dimension_min_gain = max(0.0, float(tie_dimension_min_gain))

        # 【跨轮平局维度经验】使用独立开关，避免现有同轮开关在
        # 升级后静默产生额外经验。该配置只决定是否提取元数据，不参与
        # 候选保留判定。最小增益独立于同轮阈值，必须严格超过才命中。
        if cross_round_tie_dimension_lessons is None:
            cross_round_tie_dimension_lessons = (
                os.getenv("CROSS_ROUND_TIE_DIMENSION_LESSONS", "0") != "0"
            )
        self.cross_round_tie_dimension_lessons = cross_round_tie_dimension_lessons
        if cross_round_tie_dimension_min_gain is None:
            try:
                cross_round_tie_dimension_min_gain = float(
                    os.getenv("CROSS_ROUND_TIE_DIMENSION_MIN_GAIN", "0.0")
                )
            except (TypeError, ValueError):
                cross_round_tie_dimension_min_gain = 0.0
        try:
            cross_round_tie_dimension_min_gain = float(
                cross_round_tie_dimension_min_gain
            )
        except (TypeError, ValueError):
            cross_round_tie_dimension_min_gain = 0.0
        if not math.isfinite(cross_round_tie_dimension_min_gain):
            cross_round_tie_dimension_min_gain = 0.0
        self.cross_round_tie_dimension_min_gain = max(
            0.0, cross_round_tie_dimension_min_gain
        )

        # 【目标 4：弱维度专项】默认关闭。开启时，把低于阈值的维度按分数
        # 排序并显式注入 Analyst / Mutator，同时重排候选策略与经验检索查询。
        if weak_dimension_focus is None:
            weak_dimension_focus = os.getenv("WEAK_DIMENSION_FOCUS", "0") != "0"
        self.weak_dimension_focus = weak_dimension_focus
        if weak_dimension_threshold is None:
            try:
                weak_dimension_threshold = float(os.getenv("WEAK_DIMENSION_THRESHOLD", "50.0"))
            except ValueError:
                weak_dimension_threshold = 50.0
        self.weak_dimension_threshold = max(0.0, min(100.0, float(weak_dimension_threshold)))
        if weak_dimension_max is None:
            try:
                weak_dimension_max = int(os.getenv("WEAK_DIMENSION_MAX", "2"))
            except ValueError:
                weak_dimension_max = 2
        self.weak_dimension_max = max(1, int(weak_dimension_max))

        # 【安全】仅内存持有 key 引用，用于 embedding/rerank 的 DashScope 调用；
        # 绝不打印、不落日志、不写环境变量、不序列化进任何返回结构。
        self._api_key = api_key
        # 内存 embedding 缓存（同文本不重复调用）。
        self._embed_cache = {}

        # 【阶段 1 薄封装】LLM 直调客户端：Codex / DashScope / DeepSeek / GLM
        # 固定路由 + to_thread 桥接 + 指数退避重试 + 超时 + JSON mode。
        # 【双 key】deepseek_api_key 是可选 DeepSeek key（前端双输入框场景）；
        # DeepSeek 分支优先用它，未传则回退 api_key / DEEPSEEK_API_KEY 环境变量。
        enable_thinking = os.getenv("QWEN_ENABLE_THINKING", "0") != "0"
        self._llm = LLMClient(
            api_key=api_key,
            model=self.model,
            temperature=0.2,
            enable_thinking=enable_thinking,
            deepseek_api_key=deepseek_api_key,
        )

        # 【目标 1：角色模型分工】三个角色可分别覆盖基础模型。所有变量均未
        # 设置时，各角色继续复用上面的单一客户端，调用序列与旧行为一致。
        # 路由仍由 LLMClient 的显式规则决定：gpt- 走 Codex ChatGPT 登录，
        # deepseek-/glm- 走既有例外，其余模型走 DashScope；不引入通用抽象。
        self.executor_model = os.getenv("EXECUTOR_MODEL") or self.model
        self.analyst_model = os.getenv("ANALYST_MODEL") or self.model
        self.mutator_model = os.getenv("MUTATOR_MODEL") or self.model
        self._role_llms = {}
        for role, role_model in (
            ("executor", self.executor_model),
            ("analyst", self.analyst_model),
            ("mutator", self.mutator_model),
        ):
            if role_model != self.model:
                self._role_llms[role] = LLMClient(
                    api_key=api_key,
                    model=role_model,
                    temperature=0.2,
                    enable_thinking=enable_thinking,
                    deepseek_api_key=deepseek_api_key,
                )

        # 【C2】维度权重：评分按 eval.dimension 分组后加权；无权重时与旧
        # 通过率完全一致。可用 ANALYST_DIMENSION_WEIGHTS（JSON）环境变量覆盖。
        if dimension_weights is None:
            env_weights = os.getenv("ANALYST_DIMENSION_WEIGHTS")
            if env_weights:
                try:
                    dimension_weights = json.loads(env_weights)
                except json.JSONDecodeError:
                    dimension_weights = None
        self.dimension_weights = dimension_weights or None

        # 【阶段 5 断点续跑】SKILL_CHECKPOINT_FILE 指向 langgraph SqliteSaver
        # 数据库文件；不设=关闭（默认行为不变）。恢复按 thread_id 隔离。
        if checkpoint_file is None:
            checkpoint_file = os.getenv("SKILL_CHECKPOINT_FILE") or None
        self.checkpoint_file = checkpoint_file

        # 【C5 停止源】stop_provider 回调（如 app.py 的 session 标志）与
        # self.stop_requested 任一置位即抛 StopOptimizationError；默认 None
        # 保持旧行为（只有 self.stop_requested 生效）。
        self.stop_provider = stop_provider

        # 【C3】并行变异数：默认 1（与旧行为一致）；可用 MUTATION_PARALLELISM
        # 环境变量或请求字段 parallel_mutations 开启并行，上限 3。
        if parallelism is None:
            try:
                parallelism = int(os.getenv("MUTATION_PARALLELISM", "1"))
            except ValueError:
                parallelism = 1
        self.parallelism = max(1, min(int(parallelism), 3))

        # 【目标 2：自适应策略组合】默认 classic 完整保留旧路径；adaptive
        # 才让并行候选使用不同策略，并按历史实测增益的 UCB 分数分配槽位。
        # 该机制只决定“尝试哪种单点编辑”，绝不改变严格提升保留门槛。
        if search_policy is None:
            search_policy = os.getenv("OPTIMIZATION_SEARCH", "classic")
        if search_policy not in ("classic", "adaptive"):
            search_policy = "classic"
        self.search_policy = search_policy
        if search_exploration is None:
            try:
                search_exploration = float(os.getenv("SEARCH_EXPLORATION", "1.0"))
            except ValueError:
                search_exploration = 1.0
        self.search_exploration = max(0.0, float(search_exploration))

        # 三个角色只需各自的 system prompt（LLM 调用已收敛进 llm_client）；
        # 保留 executor/analyst/mutator 与并行槽位池属性，兼容既有调用点
        # 与测试引用（_ask 通过 getattr(agent, "system_message") 取提示词）。
        # Executor：三种模式 —— 执行技能 / 生成测试场景与评分标准 / 给输出打分。
        self.executor = _RoleMessage("executor", EXECUTOR_SYSTEM_PROMPT, role="executor")
        # Analyst：根据失败的评估结果定位根因，并选择一种修改策略。
        self.analyst = _RoleMessage("analyst", ANALYST_SYSTEM_PROMPT, role="analyst")
        # Mutator：根据诊断对 SKILL.md 做"恰好一处"针对性修改，返回完整新内容。
        self.mutator = _RoleMessage("mutator", MUTATOR_SYSTEM_PROMPT, role="mutator")

        # 【C3 并行】槽位池保留（每槽独立角色对象，语义对齐旧 Assistant 池）；
        # llm_client 无会话状态，天然支持并发，无需再担心共享实例并发问题。
        self.mutator_pool = [self.mutator] + [
            _RoleMessage(f"mutator-{i}", MUTATOR_SYSTEM_PROMPT, role="mutator")
            for i in range(1, self.parallelism)
        ]
        self.executor_pool = [self.executor] + [
            _RoleMessage(f"executor-{i}", EXECUTOR_SYSTEM_PROMPT, role="executor")
            for i in range(1, self.parallelism)
        ]

    # -- Agent runner helpers ------------------------------------------------
    # 【异步】以下三个辅助函数是"同步调用 ↔ 异步接口"的桥接层：
    # _run_agent_sync 在普通线程里执行同步 DashScope 调用，_ask 用
    # asyncio.to_thread 把它交给线程池，从而不阻塞 FastAPI 事件循环
    # （轮询/SSE 才能继续响应）。签名保留以兼容既有调用点与测试 mock 点。

    def _run_agent_sync(self, agent, prompt: str, json_mode: bool = False) -> str:
        """Run the LLM synchronously through the fixed-route client (thin shell).

        ``agent`` is kept in the signature for backward compatibility: the system
        message is read from it (real assistants carry ``system_message``; objects
        without that attribute fall back to an empty system prompt). The old
        Qwen-Agent sync-generator walk is gone — the SDK returns final text
        directly, so this is a single plain call with retry inside llm_client.
        """
        system = getattr(agent, "system_message", "") or ""
        role = getattr(agent, "role", "") or ""
        client = self._role_llms.get(role, self._llm)
        return client.sync_call(system=system, user=prompt, json_mode=json_mode)

    async def _ask(self, agent, prompt: str, json_mode: bool = False) -> str:
        """Run the LLM with a prompt, return text response.

        The model request is bridged through a worker thread so it never blocks
        the FastAPI event loop handling status polling / SSE.
        """
        # 【异步】asyncio.to_thread 把阻塞的同步调用丢进线程池执行，返回可 await
        # 的结果；这样模型在思考时，事件循环仍能处理其他 HTTP 请求。
        return await asyncio.to_thread(self._run_agent_sync, agent, prompt, json_mode=json_mode)

    async def _ask_json(self, agent, prompt: str, fallback=None, schema=None):
        """Run a model role and parse the JSON response.

        When a Pydantic ``schema`` is provided, it is appended to the prompt and
        the parsed payload is validated with ``model_validate()``. Tolerant JSON
        extraction is kept, and both JSON and schema-validation failures fall
        back to ``fallback`` when one is provided.
        """
        if schema is not None:
            # 【主流程】把 schema 的 JSON 定义拼进 prompt，让模型"按图说话"。
            prompt = (
                f"{prompt}\n\n"
                f"请只返回严格符合下列 JSON Schema 的有效 JSON：\n"
                f"{json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)}\n"
                f"不要使用 Markdown 代码块，也不要在 JSON 前后添加任何说明文字。"
            )
        # 【阶段 2 可靠性】结构化输出请求开启 dashscope 原生 JSON mode
        # （response_format={"type":"json_object"}）：从协议层约束模型输出为
        # 合法 JSON，降低解析失败率；宽容解析链（raw_decode 截取 + Pydantic
        # 校验 + fallback）原样保留作兜底，行为不变量不变。
        text = await self._ask(agent, prompt, json_mode=True)

        # 【注意】宽容解析：模型可能把 JSON 包在代码块里或前后夹杂废话。
        # 先整体 json.loads，失败则用 raw_decode 定位第一个 { 或 [ 再截取。
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Try raw_decode to handle extra data after valid JSON
            decoder = json.JSONDecoder()
            idx = text.find("{")
            if idx == -1:
                idx = text.find("[")
            parsed = None
            if idx != -1:
                try:
                    parsed, _ = decoder.raw_decode(text, idx)
                except json.JSONDecodeError:
                    parsed = None

        # 【主流程】若有 schema，用 Pydantic 校验字段；不合法就当作解析失败，
        # 走 fallback（兜底值），保证优化循环不会因为一次坏响应而中断。
        if schema is not None:
            try:
                return schema.model_validate(parsed).model_dump()
            except Exception:
                parsed = None

        if parsed is None:
            if fallback is not None:
                return fallback
            raise json.JSONDecodeError(
                "Could not parse JSON from model response", text, 0
            )
        return parsed

    # -- Public API -----------------------------------------------------------
    # 【主流程】analyze_skill 是优化开始前的"分析阶段"：让 Executor 根据技能
    # 文件生成测试场景（scenarios）和评估标准（evals）。

    async def analyze_skill(self, skill_files: dict) -> dict:
        """Generate test scenarios and eval criteria from skill files."""
        # 从上传的文件里挑出 SKILL.md，以及 references/ 目录下的参考文件，
        # 把它们拼进提示词，让模型在完整上下文里分析。
        skill_md = next(
            (v for k, v in skill_files.items() if k.endswith("SKILL.md")), ""
        )
        refs = {k: v for k, v in skill_files.items() if "references/" in k}
        ref_text = ""
        if refs:
            ref_text = "\n\n# 参考文件\n" + "\n---\n".join(
                f"## {k}\n{v}" for k, v in refs.items()
            )

        prompt = (
            f"Analyze this agent skill and generate test scenarios "
            f"with evaluation criteria.\n\n"
            f"# SKILL.md\n{skill_md}\n{ref_text}\n\n"
            f"Generate:\n"
            f"1. 3-4 diverse test scenarios (realistic user inputs)\n"
            f"2. 4-6 binary yes/no evaluation criteria\n\n"
            f"If a criterion is machine-checkable (keyword presence, exact format, "
            f"length), set check_type to one of keyword|regex|yaml_or_json|length "
            f"with its required fields; otherwise omit check_type.\n\n"
            f"Also return a short domain label for the skill, one of "
            f"coding|writing|ops|data|other.\n\n"
            f"Return JSON:\n"
            f'{{"domain": "coding", '
            f'"scenarios": [{{"id": 1, "name": "short name", "description": "short name", '
            f'"input": "the user request to test"}}], '
            f'"evals": [{{"id": 1, "name": "what to check", "criterion": "what to check", '
            f'"question": "yes/no question about the output", '
            f'"pass_condition": "what yes looks like", "fail_condition": "what no looks like", '
            f'"dimension": "one of correctness|clarity|executability|maintainability|quality|semantic_preservation"}}]}}'
        )
        result = await self._ask_json(self.executor, prompt)
        # 【P0 修复】归一化 evals：模型可能漏填 dimension，代码兜底补
        # correctness，保证 C2 维度加权评分始终有维度可依（_score_skill 的
        # .get("dimension", "correctness") 兜底保留，双重保护）。
        evals = result.get("evals") if isinstance(result, dict) else None
        if isinstance(evals, list):
            for e in evals:
                if isinstance(e, dict) and not e.get("dimension"):
                    e["dimension"] = "correctness"
        # 【B 选项】归一化 domain：模型可能漏填，兜底 other。
        if isinstance(result, dict) and not result.get("domain"):
            result["domain"] = "other"
        return result

    async def optimize(
        self,
        skill_files: dict,
        scenarios: list,
        evals: list,
        max_rounds: int = 5,
        callback: Optional[Callable] = None,
        parallel_mutations: Optional[int] = None,
        strategy_pool: Optional[List[str]] = None,
        patience: Optional[int] = None,
        domain: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> dict:
        """Run the optimization loop as a LangGraph state graph.

        【C2/C3/C4/C5/P1-P3】可选参数均为"默认即旧行为"，与手写版一致：
        - parallel_mutations：并行变异数（None → 用构造时的 parallelism）
        - strategy_pool：Analyst 可用策略白名单（None → 环境变量/全量）
        - patience：耐心早停轮数（None → 用构造时的 PATIENCE，默认 0 关闭）
        - thread_id：阶段 5 断点续跑的会话标识（None → 无 checkpoint 行为）
        核心不变量不变：每个候选只改一处；仅当分数严格高于
        当前最佳 + improvement_threshold（+ noise_floor）时才保留。
        控制流已从嵌套 if 循环显式化为 optimize_graph 的条件边 + Send 扇出；
        公共签名、返回结构、进度事件逐字不变。
        """
        # 【主流程】callback 是"事件出口"：图节点把事件写进 pending_events，
        # 这里消费 LangGraph 原生 astream 后逐个回调，由 app.py 写入 session。
        # 事件源从闭包 callback 换为图流，前端轮询接口与事件形状零改动。
        async def emit(event):
            if callback:
                await callback(event)

        skill_md = next(
            (v for k, v in skill_files.items() if k.endswith("SKILL.md")), ""
        )
        # 生效的并行数：请求参数优先，否则用构造时的 parallelism（默认 1）。
        n_candidates = parallel_mutations or self.parallelism
        n_candidates = max(1, min(int(n_candidates), 3))
        # 生效的策略池。
        pool = _active_strategy_pool(strategy_pool)
        # 【P3 耐心早停】连续 N 轮未提升即终止；0 表示关闭。
        effective_patience = patience if patience is not None else self.patience

        # 【主流程】延迟导入避免与 optimize_graph 的循环依赖（optimize_graph
        # 顶层 import 本模块的 STRATEGY_TEMPLATES）。建图后流式执行：事件按
        # 节点完成顺序产出（baseline → experiment_start → experiment_result
        # ×N → complete），与手写版一致；StopOptimizationError 由 check_stop
        # 节点抛出并经 astream 冒泡（协作停止语义不变，不使用 interrupt）。
        from optimize_graph import build_optimize_graph

        initial_state: dict = {
            "skill_md": skill_md,
            "domain": domain or "",
            "n_candidates": n_candidates,
            "effective_patience": effective_patience,
            "current_md": "",
            "baseline_pct": 0.0,
            "current_details": [],
            "current_dimension_scores": {},
            "weak_dimensions": [],
            "score_history": [],
            "mutation_log": [],
            "round_memory": [],
            "lessons": [],
            "round_idx": 0,
            "no_improve": 0,
            "saturated": False,
            "analysis": None,
            "mutations": [],
            "candidates": [],
            "rescored": [],
            "confirmation": {},
            "kept": False,
            "final_pct": 0.0,
            "pending_events": [],
            "final_result": {},
        }

        # 【阶段 5 断点续跑】SKILL_CHECKPOINT_FILE 设置时挂载 SqliteSaver
        # （checkpoint-sqlite 3.x 的 from_conn_string 是 context manager，
        # 用 with 打开获得 saver）；thread_id 按会话隔离，同一 thread 再次
        # 调用自动从上次 checkpoint 续跑（跳过已 emit 的事件，前端不重复）。
        # 默认关闭=旧行为。
        async def run_graph(checkpointer, config):
            graph = build_optimize_graph(
                self,
                max_rounds=max_rounds,
                n_candidates=n_candidates,
                strategy_pool=pool,
                effective_patience=effective_patience,
                scenarios=scenarios,
                evals=evals,
                checkpointer=checkpointer,
            )
            # 【续跑】checkpointer 开启且该 thread 已有历史 checkpoint 时，
            # 以 None 作为输入从上次中断点继续；否则全新开始。
            already_emitted = 0
            if checkpointer is not None:
                # AsyncSqliteSaver 须用异步接口（aget_state）。
                snapshot = await graph.aget_state(config)
                if snapshot.values:
                    already_emitted = len(snapshot.values.get("pending_events", []))
                    stream = graph.astream(None, config=config, stream_mode="updates")
                else:
                    stream = graph.astream(initial_state, config=config, stream_mode="updates")
            else:
                stream = graph.astream(initial_state, stream_mode="updates")

            outcome: dict = {}
            seen = 0
            async for updates in stream:
                for update in updates.values():
                    for event in update.get("pending_events", []):
                        # 续跑模式跳过 checkpoint 之前已 emit 的事件。
                        if seen >= already_emitted:
                            await emit(event)
                        seen += 1
                    # finalize 节点返回最终结果（含 final_confirm 回退后的值）。
                    if update.get("final_result"):
                        outcome = update["final_result"]
            return outcome

        if self.checkpoint_file:
            # 【阶段 5】astream 是异步执行，必须用 AsyncSqliteSaver（同步版
            # SqliteSaver 只支持 sync API）；from_conn_string 返回 async
            # context manager，async with 打开后获得 saver。
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            config = {"configurable": {"thread_id": thread_id or "default"}}
            async with AsyncSqliteSaver.from_conn_string(self.checkpoint_file) as checkpointer:
                return await run_graph(checkpointer, config)
        return await run_graph(None, None)

    @staticmethod
    def _diff_summary(original_md: str, new_md: str, limit: int = 4000) -> str:
        """【C6】生成 SKILL.md 的 unified diff 摘要（限长，避免全量文本上线）。"""
        if not original_md or not new_md:
            return ""
        import difflib
        diff_lines = list(difflib.unified_diff(
            original_md.splitlines(), new_md.splitlines(),
            fromfile="before", tofile="after", lineterm="",
        ))
        summary = "\n".join(diff_lines)
        return summary[:limit]

    # -- Internal helpers -----------------------------------------------------
    # 【主流程】这三个私有方法是 optimize() 的三个环节，也是"三个 Agent 各司
    # 其职"的具体实现：Executor 打分、Analyst 诊断、Mutator 修改。

    async def _score_skill(self, skill_md, scenarios, evals, agent=None):
        """Executor runs all scenarios, then scores outputs.

        【C2 多维加权评分】除了原有的二进制通过率（passed/total），还派生
        ``dimension_scores``（按 eval 的 dimension 字段分组，默认 correctness）
        与 ``weighted_pct``。未配置维度权重时 weighted_pct 与旧通过率完全一致，
        因此旧行为与旧测试不受影响。
        """
        all_results = []
        total_passed = 0
        total_checks = 0
        per_eval = {e["id"]: {"passed": 0, "total": 0, "dimension": e.get("dimension", "correctness")} for e in evals}
        # 维度统计：每个评估项按其 dimension 归组累加。
        dim_stats = {}
        for e in evals:
            dim = e.get("dimension", "correctness")
            dim_stats.setdefault(dim, {"passed": 0, "total": 0})
        # 选择本次评分使用的 Executor 实例（并行时用对应槽位）。
        scorer = agent or self.executor

        # 【P1 规则锚点】把可程序化校验的 eval（带 check_type）与需要 LLM
        # 语义打分的 eval 分开：规则型用 Python 判定，LLM 型仍走原打分路径。
        # 全部 eval 都无 check_type 时，llm_evals == evals，调用序列与旧行为
        # 完全一致（每场景仍 1 次执行 + 1 次打分），旧测试不受影响。
        rule_evals = [e for e in evals if e.get("check_type")]
        llm_evals = [e for e in evals if not e.get("check_type")]

        for sc in scenarios:
            # Executor runs the skill (free-form text)
            # 【主流程】第一步：让 Executor 模拟执行技能，产出结果（自由文本）。
            output = await self._ask(
                scorer,
                f"Execute this skill:\n\n{skill_md}\n\nUser request:\n{sc['input']}",
            )
            # Executor scores the output (JSON)
            # 【主流程】第二步：让 Executor 对照 llm_evals 给输出逐条打分（JSON）。
            # fallback 保证模型不给合法 JSON 时，这一场景按"全部未通过"处理。
            if llm_evals:
                scoring = await self._ask_json(
                    scorer,
                    (
                        f"Evaluate this output against the criteria.\n\n"
                        f"Input: {sc['input']}\n\n"
                        f"Output: {output}\n\n"
                        f"Criteria:\n{json.dumps(llm_evals, indent=2)}\n\n"
                        f'Return JSON: {{"results": [{{"eval_id": 1, "passed": true, "reason": "..."}}]}}'
                    ),
                    fallback={"results": []},
                )
            else:
                scoring = {"results": []}
            scores = scoring.get("results", []) if isinstance(scoring, dict) else scoring

            # 累加每个评估项的通过与总次数，同时保留逐条明细给 Analyst 用。
            for s in scores:
                eid = s.get("eval_id")
                passed = s.get("passed", False)
                if passed:
                    total_passed += 1
                total_checks += 1
                if eid in per_eval:
                    per_eval[eid]["total"] += 1
                    if passed:
                        per_eval[eid]["passed"] += 1
                    dim = per_eval[eid]["dimension"]
                    if dim in dim_stats:
                        dim_stats[dim]["total"] += 1
                        if passed:
                            dim_stats[dim]["passed"] += 1
                all_results.append({**s, "scenario_id": sc["id"]})

            # 【P1 规则锚点】规则型 eval 用 Python 判定，不进 LLM 打分。
            for e in rule_evals:
                passed, reason = self._rule_check(e, output)
                eid = e["id"]
                if passed:
                    total_passed += 1
                total_checks += 1
                if eid in per_eval:
                    per_eval[eid]["total"] += 1
                    if passed:
                        per_eval[eid]["passed"] += 1
                    dim = per_eval[eid]["dimension"]
                    if dim in dim_stats:
                        dim_stats[dim]["total"] += 1
                        if passed:
                            dim_stats[dim]["passed"] += 1
                all_results.append({
                    "eval_id": eid, "passed": passed, "reason": reason,
                    "scenario_id": sc["id"],
                })

        # 【C2】派生维度分数：按 dimension 分组，保留每个维度的通过率。
        dimension_scores = {
            dim: {
                "passed": stats["passed"],
                "total": stats["total"],
                "pct": round(100 * stats["passed"] / max(stats["total"], 1), 1),
            }
            for dim, stats in dim_stats.items()
        }

        # 【C2】加权分数：有权重时按维度加权平均；无权重时与旧通过率完全一致。
        weights = self.dimension_weights or {}
        if weights and dim_stats:
            # 只对出现在维度统计里的权重求和，避免除零。
            w_total = sum(w for d, w in weights.items() if d in dim_stats and w >= 0)
            if w_total > 0:
                w_passed = sum(
                    weights[d] * (stats["passed"] / max(stats["total"], 1))
                    for d, stats in dim_stats.items()
                    if d in weights
                )
                weighted_pct = round(100 * w_passed / w_total, 1)
            else:
                weighted_pct = round(100 * total_passed / max(total_checks, 1), 1)
        else:
            weighted_pct = round(100 * total_passed / max(total_checks, 1), 1)

        return {
            "passed": total_passed,
            "total": total_checks,
            "per_eval": [
                {"eval_id": k, **v, "pass_rate": round(v["passed"] / max(v["total"], 1) * 100, 1)}
                for k, v in per_eval.items()
            ],
            "details": all_results,
            "dimension_scores": dimension_scores,
            "weighted_pct": weighted_pct,
        }

    @staticmethod
    def _rule_check(eval_def: dict, output: str):
        """【P1 规则锚点】用 Python 规则判定可程序化校验的 eval。

        Returns ``(passed, reason)``。支持 4 种 check_type：
        keyword（输出须包含全部关键词）/ regex（正则匹配）/
        yaml_or_json（可解析且含 required_key）/ length（长度区间）。
        未知或字段缺失的 check_type 按未通过处理（保守）。
        """
        ct = eval_def.get("check_type")
        if ct == "keyword":
            kws = eval_def.get("keywords") or []
            passed = all(k in output for k in kws)
            return passed, ("keyword:ok" if passed else "keyword:missing")
        if ct == "regex":
            try:
                pattern = eval_def.get("pattern") or ""
                flags = re.IGNORECASE if "i" in (eval_def.get("flags") or "") else 0
                passed = re.search(pattern, output, flags) is not None
            except re.error:
                passed = False
            return passed, ("regex:match" if passed else "regex:no_match")
        if ct == "yaml_or_json":
            parsed = None
            try:
                parsed = json.loads(output)
            except Exception:
                try:
                    import yaml
                    parsed = yaml.safe_load(output)
                except Exception:
                    parsed = None
            key = eval_def.get("required_key")
            passed = parsed is not None and (key is None or key in parsed)
            return passed, ("format:valid" if passed else "format:invalid")
        if ct == "length":
            lo = eval_def.get("min_length")
            hi = eval_def.get("max_length")
            n = len(output)
            passed = (lo is None or n >= lo) and (hi is None or n <= hi)
            return passed, f"length:{n}"
        return False, f"unknown_check_type:{ct}"

    @staticmethod
    def _regression_check(original_md: str, new_md: str):
        """【C4】纯 Python 回归守卫：变异后先做结构完整性检查。

        Returns ``(ok, reason)``. 命中任一规则即拒绝该变异并跳过复评（省 token）。
        守卫只可能比原逻辑更严格，绝不会让分数退化的变异被采纳。
        """
        if not new_md or not new_md.strip():
            return False, "regression:empty"
        if not original_md:
            return True, ""
        # frontmatter 丢失：原文件有 --- 头而新文件没有。
        if original_md.lstrip().startswith("---") and not new_md.lstrip().startswith("---"):
            return False, "regression:frontmatter"
        # name 字段丢失。
        if re.search(r"(?m)^name\s*:", original_md) and not re.search(r"(?m)^name\s*:", new_md):
            return False, "regression:name_missing"
        # 多数标题被删（超过一半的 #/## 标题消失）。
        orig_heads = len(re.findall(r"(?m)^#{1,2} ", original_md))
        if orig_heads > 0:
            new_heads = len(re.findall(r"(?m)^#{1,2} ", new_md))
            if new_heads < orig_heads / 2:
                return False, "regression:headings_removed"
        # 体积缩减超过 40%，视为大改或截断。
        if len(new_md) < 0.6 * len(original_md):
            return False, "regression:content_shrunk"
        # 【P2 大小上限】体积膨胀超过 15KB 拒绝（GEPA size limits），防技能无限膨胀。
        if len(new_md) > 15000:
            return False, "regression:too_large"
        return True, ""

    def check_stop(self):
        """【C5】协作式取消检查：在轮间调用，stop 置位即抛异常。

        停止来源有两个（任一置位即停）：实例属性 ``self.stop_requested``
        （兼容旧行为/测试），以及构造时传入的 ``stop_provider`` 回调
        （如 app.py 的 session["stop_requested"]，让 /api/stop 真正生效）。
        """
        if getattr(self, "stop_requested", False):
            raise StopOptimizationError("Optimization stopped by user request")
        if self.stop_provider is not None:
            try:
                if self.stop_provider():
                    raise StopOptimizationError("Optimization stopped by user request")
            except StopOptimizationError:
                raise
            except Exception:
                pass  # stop_provider 自身异常视为未停止，不影响优化

    async def _analyze_failures(self, skill_md, scenarios, evals, details, strategy_pool=None,
                                round_memory=None, mutation_log=None, lessons=None,
                                weak_dimensions=None):
        """Analyst agent diagnoses the worst failures.

        【C1】strategy_pool 白名单：只把允许的策略模板及其描述拼进提示词，
        让 Analyst 只能在可用的策略里选择；未指定时用环境变量/全量模板。
        【P2】round_memory：注入最近几轮的"诊断+结果"，避免反复修同一根因；
        mutation_log：把已尝试未提升的策略标注 blocked，并列出被拒编辑摘要，
        防止重复提出相同修改（SkillOpt rejected-edit buffer 思想）。
        """
        # 没有失败项时无需诊断，直接返回"无操作"占位，让 Mutator 拿到安全默认值。
        failed = [d for d in details if not d.get("passed")]
        if not failed:
            return {
                "diagnosis": "All passed",
                "mutation_strategy": "add_constraint",
                "target_section": "N/A",
                "suggested_change": "none",
                "target_dimension": None,
            }

        # 解析生效的策略池，并生成"策略 → 描述"的提示片段。
        pool = _active_strategy_pool(strategy_pool)
        # 【P2 策略黑名单】复评过但未提升（非回归拒绝）的策略标注 blocked，
        # 只标注不移除，保底 pool[0]，防止策略枯竭。
        blocked = {
            m.get("strategy_type") for m in (mutation_log or [])
            if not m.get("kept") and not (m.get("reason") or "").startswith("regression")
            and not (m.get("reason") or "") == "edit_limit_exceeded"
        }
        strategy_desc = "\n".join(
            f"- {s}: {STRATEGY_TEMPLATES[s]['description']} ({STRATEGY_TEMPLATES[s]['when']})"
            + (" (blocked: tried without improvement)" if s in blocked else "")
            for s in pool
        )

        prompt = (
            f"Diagnose these failures and suggest ONE fix.\n\n"
            f"Skill:\n{skill_md[:2000]}\n\n"
            f"Scenarios:\n{json.dumps(scenarios, indent=2)}\n\n"
            f"Criteria:\n{json.dumps(evals, indent=2)}\n\n"
            f"Failures:\n{json.dumps(failed[:5], indent=2)}\n\n"
            f"Allowed mutation strategies (pick exactly one):\n{strategy_desc}"
        )

        # 【目标 4】弱维度专项诊断：只注入维度统计与相关失败标准，要求本轮
        # 仍只提出一个、能直接改善最弱维度的修改。
        if weak_dimensions:
            weak_names = [d["dimension"] for d in weak_dimensions]
            eval_dimensions = {e.get("id"): e.get("dimension", "correctness") for e in evals}
            focused_failures = [
                d for d in failed if eval_dimensions.get(d.get("eval_id")) in weak_names
            ][:5]
            preferred = self._focused_strategy_pool(pool, weak_dimensions)
            prompt += (
                f"\n\nPriority weak dimensions (lowest first):\n"
                f"{json.dumps(weak_dimensions, ensure_ascii=False)}\n"
                f"Failures in those dimensions:\n"
                f"{json.dumps(focused_failures, ensure_ascii=False)}\n"
                f"Preferred strategies for this weakness: {json.dumps(preferred)}\n"
                f"Set target_dimension to one listed weak dimension. The ONE suggested "
                f"change must directly improve it without sacrificing other dimensions."
            )

        # 【P2 轮间记忆】最近 MEMORY_ROUNDS 条历史注入，默认 3。
        if round_memory:
            try:
                memory_rounds = int(os.getenv("MEMORY_ROUNDS", "3"))
            except ValueError:
                memory_rounds = 3
            memory_slice = round_memory[-max(1, memory_rounds):]
            prompt += (
                f"\n\nRecent attempts (previous rounds):\n"
                f"{json.dumps(memory_slice, ensure_ascii=False)}"
            )

        # 【P2 被拒编辑缓冲】被拒候选的编辑摘要注入，避免重复提出相同修改。
        rejected_edits = [
            (m.get("description") or "")[:100] for m in (mutation_log or [])
            if not m.get("kept") and m.get("description")
        ]
        if rejected_edits:
            prompt += (
                f"\n\nAvoid repeating these rejected edits:\n"
                f"{json.dumps(rejected_edits, ensure_ascii=False)}"
            )

        # 【P2 语义保持】建议的修改不得偏离技能原始用途。
        prompt += "\n\nKeep the skill's original purpose intact; do not drift from its intent."

        # 【P4 跨会话经验】历史成功修复作为 few-shot 示例注入。
        if lessons:
            prompt += (
                f"\n\nLessons from past successful fixes:\n"
                f"{json.dumps(lessons, ensure_ascii=False)}"
            )

        # 【主流程】把技能内容与最近的失败明细喂给 Analyst，并要求返回符合
        # FailureAnalysis schema 的 JSON；schema 校验失败会走 fallback。
        return await self._ask_json(
            self.analyst,
            prompt,
            fallback={
                "diagnosis": "Unable to determine root cause from failures",
                "mutation_strategy": pool[0] if pool else "add_constraint",
                "target_section": "TBD",
                "suggested_change": f"Apply the {pool[0] if pool else 'add_constraint'} strategy to address the failing criteria",
                "target_dimension": (
                    weak_dimensions[0]["dimension"] if weak_dimensions else None
                ),
            },
            schema=FailureAnalysis,
        )

    async def _mutate_skill(self, skill_md, analysis, agent=None, strategy_hint=None, lessons=None):
        """Mutator agent makes one targeted change.

        【C1】strategy_hint：把选中策略的 mutator_hint 追加进提示词，引导
        Mutator 按该策略做"恰好一处"修改。
        【C3】agent：并行时传入对应槽位的 Mutator 实例；默认用主实例。
        【P4】lessons：跨会话成功经验作为 few-shot 注入。
        """
        mutator = agent or self.mutator
        strategy = analysis.get("mutation_strategy", "")
        hint = strategy_hint or (
            STRATEGY_TEMPLATES[strategy].get("mutator_hint", "")
            if strategy in STRATEGY_TEMPLATES
            else ""
        )
        prompt = (
            f"Apply this fix to the skill. Make ONE change only.\n"
            f"Preserve the skill's original purpose and overall structure.\n\n"
            f"SKILL.md:\n{skill_md}\n\n"
            f"Diagnosis: {analysis.get('diagnosis')}\n"
            f"Strategy: {strategy}\n"
            f"Target: {analysis.get('target_section')}\n"
            f"Change: {analysis.get('suggested_change')}"
        )
        if analysis.get("target_dimension"):
            prompt += (
                f"\nTarget evaluation dimension: {analysis['target_dimension']}\n"
                f"Use the ONE edit to improve this dimension while preserving all others."
            )
        if hint:
            prompt += f"\nStrategy guidance: {hint}"
        if lessons:
            prompt += (
                f"\n\nLessons from past successful fixes:\n"
                f"{json.dumps(lessons, ensure_ascii=False)}"
            )
        # 【主流程】把诊断结论作为修改指令交给 Mutator；prompt 明确要求"只改
        # 一处"，返回完整的新 SKILL.md。失败时兜底返回原内容（等于不改）。
        return await self._ask_json(
            mutator,
            prompt,
            fallback={
                "description": "No change applied",
                "reasoning": "",
                "new_skill_md": skill_md,
            },
            schema=SkillMutation,
        )

    @staticmethod
    def _strategy_stats(mutation_log):
        """按修改策略统计"总尝试数 / 被保留数"，用于结果页的策略汇总。"""
        stats = {}
        for m in mutation_log:
            s = m.get("strategy_type", "unknown")
            if s not in stats:
                stats[s] = {"total": 0, "kept": 0}
            stats[s]["total"] += 1
            if m.get("kept"):
                stats[s]["kept"] += 1
        return stats

    def _select_candidate_strategies(self, strategy_pool, mutation_log, n, primary_strategy):
        """Select a deterministic explore/exploit portfolio for one round.

        Untried strategies are explored first (the Analyst's choice wins ties).
        Once tried, strategies are ranked by normalized mean positive score gain
        plus a UCB exploration bonus. Candidate evaluation remains unchanged;
        only a strictly improving best candidate can update ``current_md``.
        """
        pool = [s for s in strategy_pool if s in STRATEGY_TEMPLATES]
        if not pool:
            return [primary_strategy or "add_constraint"] * max(1, n)

        stats = {s: {"attempts": 0, "reward": 0.0} for s in pool}
        for entry in mutation_log or []:
            strategy = entry.get("strategy_type")
            if strategy not in stats:
                continue
            stats[strategy]["attempts"] += 1
            gain = max(0.0, float(entry.get("score_after", 0.0)) - float(entry.get("score_before", 0.0)))
            stats[strategy]["reward"] += gain / 100.0
        total_attempts = sum(s["attempts"] for s in stats.values())

        def rank(item):
            index, strategy = item
            attempts = stats[strategy]["attempts"]
            # Deterministic cold-start: honor the Analyst first, then pool order.
            if attempts == 0:
                return (1, 1 if strategy == primary_strategy else 0, 0.0, -index)
            mean_reward = stats[strategy]["reward"] / attempts
            bonus = self.search_exploration * math.sqrt(
                math.log(max(total_attempts, 1) + 1) / attempts
            )
            analyst_bonus = 0.05 if strategy == primary_strategy else 0.0
            return (0, 0, mean_reward + bonus + analyst_bonus, -index)

        ordered = [s for _, s in sorted(enumerate(pool), key=rank, reverse=True)]
        count = max(1, int(n))
        return [ordered[i % len(ordered)] for i in range(count)]

    def _identify_weak_dimensions(self, dimension_scores):
        """Return scored dimensions at/below the configured weakness threshold."""
        if not self.weak_dimension_focus:
            return []
        weak = []
        for dimension, stats in (dimension_scores or {}).items():
            if not isinstance(stats, dict):
                continue
            try:
                total = int(stats.get("total", 0))
                passed = int(stats.get("passed", 0))
                pct = float(stats.get("pct", 0.0))
            except (TypeError, ValueError):
                continue
            if total > 0 and pct <= self.weak_dimension_threshold:
                weak.append({
                    "dimension": dimension,
                    "pct": pct,
                    "passed": passed,
                    "total": total,
                })
        weak.sort(key=lambda d: (d["pct"], d["dimension"]))
        return weak[:self.weak_dimension_max]

    @staticmethod
    def _focused_strategy_pool(strategy_pool, weak_dimensions):
        """Reorder an allowed strategy pool around the weakest dimensions."""
        pool = [s for s in strategy_pool if s in STRATEGY_TEMPLATES]
        preferred = []
        for item in weak_dimensions or []:
            for strategy in WEAK_DIMENSION_STRATEGIES.get(item.get("dimension"), []):
                if strategy in pool and strategy not in preferred:
                    preferred.append(strategy)
        return preferred + [s for s in pool if s not in preferred]

    # -- P4 跨会话经验库（SQLite / jsonl 双存储）--------------------------------
    # 【P4】经验库只记录模型生成的策略/诊断/摘要与分数，绝不落 skill_md /
    # scenario / output / api_key（与已返回前端的 mutation_log 同级，安全）。
    # 读写一律 try/except 静默失败，绝不因经验库问题中断优化。
    # 【RAG】.db/.sqlite 后缀走 SQLite（可存 embedding，上限 1000）；其他后缀
    # 保持 jsonl 兼容（上限 1000）。SQLite 由 Python 内置 sqlite3 提供，零依赖。

    @staticmethod
    def _lesson_is_sqlite(path):
        return bool(path and path.lower().endswith((".db", ".sqlite", ".sqlite3")))

    @staticmethod
    def _normalize_lesson_metadata(lesson):
        """Normalize additive lesson metadata without rewriting persisted data."""
        normalized = dict(lesson)
        if (
            normalized.get("lesson_type") == "tie_dimension"
            and not normalized.get("comparison_scope")
        ):
            # Records created before cross-round learning were necessarily
            # same-round comparisons. Normalize only at read time so old JSONL
            # and SQLite stores remain usable without a destructive rewrite.
            normalized["comparison_scope"] = "same_round"
        return normalized

    @staticmethod
    def _embedding_to_blob(embedding):
        """numpy 向量 → float32 bytes（SQLite BLOB）；None 原样返回。"""
        if embedding is None:
            return None
        import numpy as np
        return np.asarray(embedding, dtype=np.float32).tobytes()

    @staticmethod
    def _load_lessons(path, limit=5):
        """读取最近 ``limit`` 条经验；按后缀选 SQLite 或 jsonl；异常返回 []。"""
        if not path:
            return []
        if SkillOptimizer._lesson_is_sqlite(path):
            return SkillOptimizer._load_lessons_sqlite(path, limit)
        return SkillOptimizer._load_lessons_jsonl(path, limit)

    @staticmethod
    def _load_lessons_jsonl(path, limit=5):
        lessons = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        lesson = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(lesson, dict):
                        lesson = SkillOptimizer._normalize_lesson_metadata(lesson)
                    lessons.append(lesson)
        except OSError:
            return []
        return lessons[-max(1, limit):]

    @staticmethod
    def _load_lessons_sqlite(path, limit=5):
        try:
            import sqlite3
            conn = sqlite3.connect(path)
            try:
                columns = {
                    row[1] for row in conn.execute("PRAGMA table_info(lessons)").fetchall()
                }
                lesson_type_col = "lesson_type" if "lesson_type" in columns else "NULL"
                dimension_gains_col = "dimension_gains" if "dimension_gains" in columns else "NULL"
                target_dimension_col = (
                    "target_dimension" if "target_dimension" in columns else "NULL"
                )
                comparison_scope_col = (
                    "comparison_scope" if "comparison_scope" in columns else "NULL"
                )
                rows = conn.execute(
                    "SELECT id, skill_name, domain, skill_description, strategy, diagnosis, "
                    "summary, score_before, score_after, created_at, embedding, "
                    f"{lesson_type_col}, {dimension_gains_col}, {target_dimension_col}, "
                    f"{comparison_scope_col} "
                    "FROM lessons ORDER BY id DESC LIMIT ?",
                    (max(1, limit),),
                ).fetchall()
                lessons = []
                for r in reversed(rows):  # 反序恢复"旧→新"
                    lesson = {
                        # 【RAG】_id 用于惰性 embedding 写回；注入前会被剥离。
                        "_id": r[0],
                        "skill_name": r[1], "domain": r[2], "skill_description": r[3],
                        "strategy": r[4], "diagnosis": r[5], "summary": r[6],
                        "score_before": r[7], "score_after": r[8], "created_at": r[9],
                    }
                    if r[10]:
                        import numpy as np
                        lesson["embedding"] = np.frombuffer(r[10], dtype=np.float32)
                    if r[11]:
                        lesson["lesson_type"] = r[11]
                    if r[12]:
                        try:
                            lesson["dimension_gains"] = json.loads(r[12])
                        except (json.JSONDecodeError, TypeError):
                            pass
                    if r[13]:
                        lesson["target_dimension"] = r[13]
                    if r[14]:
                        lesson["comparison_scope"] = r[14]
                    lessons.append(SkillOptimizer._normalize_lesson_metadata(lesson))
                return lessons
            finally:
                conn.close()
        except Exception:
            return []

    @staticmethod
    def _append_lesson(path, lesson):
        """追加一条经验（SQLite 或 jsonl）；上限 1000；异常静默。"""
        if not path:
            return
        if SkillOptimizer._lesson_is_sqlite(path):
            SkillOptimizer._append_lesson_sqlite(path, lesson)
        else:
            SkillOptimizer._append_lesson_jsonl(path, lesson)

    @staticmethod
    def _append_lesson_jsonl(path, lesson):
        try:
            lines = []
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            lines.append(line)
            lines.append(json.dumps(lesson, ensure_ascii=False))
            if len(lines) > 1000:
                lines = lines[-1000:]
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except OSError:
            pass

    @staticmethod
    def _append_lesson_sqlite(path, lesson):
        try:
            import sqlite3
            conn = sqlite3.connect(path)
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS lessons ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "skill_name TEXT, domain TEXT, skill_description TEXT,"
                    "strategy TEXT, diagnosis TEXT, summary TEXT,"
                    "score_before REAL, score_after REAL, created_at TEXT,"
                    "embedding BLOB, lesson_type TEXT, dimension_gains TEXT,"
                    "target_dimension TEXT, comparison_scope TEXT)"
                )
                # 兼容既有数据库：CREATE IF NOT EXISTS 不会补列，按需迁移。
                columns = {
                    row[1] for row in conn.execute("PRAGMA table_info(lessons)").fetchall()
                }
                if "lesson_type" not in columns:
                    conn.execute("ALTER TABLE lessons ADD COLUMN lesson_type TEXT")
                if "dimension_gains" not in columns:
                    conn.execute("ALTER TABLE lessons ADD COLUMN dimension_gains TEXT")
                if "target_dimension" not in columns:
                    conn.execute("ALTER TABLE lessons ADD COLUMN target_dimension TEXT")
                if "comparison_scope" not in columns:
                    conn.execute("ALTER TABLE lessons ADD COLUMN comparison_scope TEXT")
                conn.execute(
                    "INSERT INTO lessons (skill_name, domain, skill_description, strategy, "
                    "diagnosis, summary, score_before, score_after, created_at, embedding, "
                    "lesson_type, dimension_gains, target_dimension, comparison_scope) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        lesson.get("skill_name"), lesson.get("domain"),
                        lesson.get("skill_description"), lesson.get("strategy"),
                        lesson.get("diagnosis"), lesson.get("summary"),
                        lesson.get("score_before"), lesson.get("score_after"),
                        lesson.get("created_at"),
                        SkillOptimizer._embedding_to_blob(lesson.get("embedding")),
                        lesson.get("lesson_type"),
                        (
                            json.dumps(lesson.get("dimension_gains"), ensure_ascii=False)
                            if lesson.get("dimension_gains") else None
                        ),
                        lesson.get("target_dimension"),
                        lesson.get("comparison_scope"),
                    ),
                )
                # 上限 1000：删除最旧的超量行。
                conn.execute(
                    "DELETE FROM lessons WHERE id NOT IN "
                    "(SELECT id FROM lessons ORDER BY id DESC LIMIT 1000)"
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass

    # -- P4 检索模式（RAG）-----------------------------------------------------
    # 【RAG】LESSON_RETRIEVAL 决定经验如何注入：off=最近 N 条（旧行为）；
    # tag=同技能硬过滤+通用兜底（零依赖）；semantic=embedding 语义检索；
    # hybrid=dense+sparse 混合（RRF 融合）。embedding 用 DashScope
    # text-embedding-v3（Qwen 生态），失败静默降级到 tag，绝不中断优化。

    @staticmethod
    def _update_lesson_embedding(path, row_id, embedding):
        """惰性算出的 embedding 写回 SQLite（跨会话复用，避免重复调用）。"""
        if not path or row_id is None or not SkillOptimizer._lesson_is_sqlite(path):
            return
        try:
            import sqlite3
            conn = sqlite3.connect(path)
            try:
                conn.execute(
                    "UPDATE lessons SET embedding=? WHERE id=?",
                    (SkillOptimizer._embedding_to_blob(embedding), row_id),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass

    @staticmethod
    def _skill_name_from_md(skill_md):
        """从 SKILL.md frontmatter 提取 name（无则空串）。"""
        m = re.search(r"(?m)^name\s*:\s*(.+?)\s*$", skill_md or "")
        return m.group(1).strip() if m else ""

    @staticmethod
    def _lesson_index_text(lesson):
        """经验 → 检索文本（模型建议 + 安全的结构化维度元数据）。"""
        parts = [
            lesson.get("strategy", ""),
            lesson.get("diagnosis", ""),
            lesson.get("summary", ""),
            lesson.get("comparison_scope", ""),
            lesson.get("target_dimension", ""),
        ]
        if lesson.get("dimension_gains"):
            parts.append(json.dumps(lesson["dimension_gains"], ensure_ascii=False, sort_keys=True))
        return " ".join(str(p) for p in parts if p)

    @staticmethod
    def _build_query(
        skill_name,
        domain,
        scenarios,
        evals,
        current_details,
        weak_dimensions=None,
        include_failure_context=False,
    ):
        """Build a bounded retrieval query from tags and observed failures."""
        parts = []
        if domain:
            parts.append(f"domain: {domain}")
        if skill_name:
            parts.append(f"skill: {skill_name}")
        for item in weak_dimensions or []:
            parts.append(f"weak dimension: {item.get('dimension')} {item.get('pct')} percent")
        failed = [d for d in (current_details or []) if not d.get("passed")]
        for d in failed[:3]:
            parts.append(str(d.get("reason") or ""))

        if include_failure_context:
            eval_by_id = {
                str(item.get("id")): item
                for item in (evals or [])
                if isinstance(item, dict) and item.get("id") is not None
            }
            scenario_by_id = {
                str(item.get("id")): item
                for item in (scenarios or [])
                if isinstance(item, dict) and item.get("id") is not None
            }
            used_scenarios = set()
            for detail in failed[:3]:
                eval_def = eval_by_id.get(str(detail.get("eval_id")))
                if eval_def:
                    fields = [
                        eval_def.get("dimension"),
                        eval_def.get("name"),
                        eval_def.get("criterion"),
                        eval_def.get("question"),
                        eval_def.get("pass_condition"),
                    ]
                    context = " | ".join(str(value) for value in fields if value)
                    if context:
                        parts.append(f"failed evaluation: {context[:360]}")
                scenario_id = str(detail.get("scenario_id"))
                if scenario_id not in used_scenarios and scenario_id in scenario_by_id:
                    scenario_input = str(scenario_by_id[scenario_id].get("input", ""))[:160]
                    if scenario_input:
                        parts.append(f"failed scenario: {scenario_input}")
                    used_scenarios.add(scenario_id)
        else:
            # Classic/default query shape is preserved unless explicitly enabled.
            for scenario in (scenarios or [])[:2]:
                parts.append(str(scenario.get("input", ""))[:80])
        return " ".join(p for p in parts if p)

    @staticmethod
    def _tag_filter(lessons, skill_name, domain, limit):
        """同技能经验优先 → 同领域次之 → 通用兜底；各自取最近。"""
        same = [l for l in lessons if l.get("skill_name") == skill_name]
        same_domain = [
            l for l in lessons
            if l.get("skill_name") != skill_name and l.get("domain") == domain and domain
        ]
        generic = [l for l in lessons if not l.get("skill_name") and not l.get("domain")]
        pool = (same + same_domain + generic)
        return pool[-max(1, limit):]

    @staticmethod
    def _cosine(a, b):
        import numpy as np
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def _sparse_jaccard(query, text):
        """轻量稀疏分：ASCII 词 + CJK 单字/双字 Jaccard，无新增分词依赖。"""
        q = SkillOptimizer._lesson_tokens(query)
        t = SkillOptimizer._lesson_tokens(text)
        if not q or not t:
            return 0.0
        return len(q & t) / len(q | t)

    @staticmethod
    def _lesson_tokens(text):
        """Tokenize English identifiers and Chinese text for sparse/dedup scoring."""
        value = (text or "").lower()
        tokens = set(re.findall(r"[a-z0-9_]+", value))
        cjk = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", value)
        tokens.update(cjk)
        tokens.update("".join(cjk[i:i + 2]) for i in range(len(cjk) - 1))
        return tokens

    @staticmethod
    def _rrf_fuse(rank_lists, k=60):
        """RRF 融合多个排序：score = Σ 1/(k+rank)。返回 (item, score)。"""
        scores = {}
        for ranks in rank_lists:
            for rank, item in enumerate(ranks):
                key = id(item)
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        by_id = {id(it): it for ranks in rank_lists for it in ranks}
        return [(by_id[k], s) for k, s in ordered]

    def _embed_sync(self, text):
        """DashScope text-embedding-v3（同步；在 _embed 的线程桥接里执行）。

        key 来源：DASHSCOPE_API_KEY 环境变量优先（CLI/benchmark 用 DeepSeek/
        GLM 生成时，主 key 不是百炼 key，检索侧需单独取百炼 key），否则主 key。
        """
        import dashscope
        resp = dashscope.TextEmbedding.call(
            model="text-embedding-v3", input=text,
            api_key=os.getenv("DASHSCOPE_API_KEY") or self._api_key, dimensions=512,
        )
        if getattr(resp, "status_code", 500) != 200:
            raise RuntimeError(
                f"embedding failed: {getattr(resp, 'code', '')} {getattr(resp, 'message', '')}"
            )
        return resp.output["embeddings"][0]["embedding"]

    async def _embed(self, text):
        """embedding（内存缓存 + 线程桥接，不阻塞事件循环）。"""
        if text not in self._embed_cache:
            self._embed_cache[text] = await asyncio.to_thread(self._embed_sync, text)
        return self._embed_cache[text]

    def _embed_many_sync(self, texts):
        """Batch DashScope text-embedding-v3 call; response is restored by text_index."""
        import dashscope
        resp = dashscope.TextEmbedding.call(
            model="text-embedding-v3",
            input=texts,
            api_key=os.getenv("DASHSCOPE_API_KEY") or self._api_key,
            dimensions=512,
        )
        if getattr(resp, "status_code", 500) != 200:
            raise RuntimeError(
                f"embedding failed: {getattr(resp, 'code', '')} {getattr(resp, 'message', '')}"
            )
        raw = resp.output.get("embeddings", [])
        ordered = [None] * len(texts)
        for position, item in enumerate(raw):
            try:
                index = int(item.get("text_index", position))
            except (TypeError, ValueError):
                index = position
            if 0 <= index < len(ordered):
                ordered[index] = item.get("embedding")
        if any(vector is None for vector in ordered):
            raise RuntimeError("embedding failed: incomplete batch response")
        return ordered

    async def _embed_many(self, texts):
        """Embed unique texts in bounded batches while preserving cache/order semantics."""
        unique_missing = []
        seen = set()
        for text in texts:
            if text not in self._embed_cache and text not in seen:
                unique_missing.append(text)
                seen.add(text)

        if self.lesson_embed_batch_size <= 1:
            for text in unique_missing:
                # Do not rely on a patched/custom _embed implementation to mutate
                # our cache as a side effect; its return value is the interface.
                self._embed_cache[text] = await self._embed(text)
        else:
            size = self.lesson_embed_batch_size
            for start in range(0, len(unique_missing), size):
                batch = unique_missing[start:start + size]
                vectors = await asyncio.to_thread(self._embed_many_sync, batch)
                if len(vectors) != len(batch):
                    raise RuntimeError("embedding failed: batch size mismatch")
                self._embed_cache.update(zip(batch, vectors))
        return [self._embed_cache[text] for text in texts]

    async def _ensure_lesson_embeddings(self, lessons):
        """Populate missing lesson vectors and persist them when SQLite row IDs exist."""
        missing = []
        for lesson in lessons:
            if lesson.get("embedding") is None:
                missing.append((lesson, self._lesson_index_text(lesson)))
        if not missing:
            return
        vectors = await self._embed_many([text for _, text in missing])
        for (lesson, _), vector in zip(missing, vectors):
            lesson["embedding"] = vector
            if lesson.get("_id") is not None:
                self._update_lesson_embedding(self.lesson_file, lesson["_id"], vector)

    def _rerank_sync(self, query, docs):
        """DashScope qwen3-rerank（同步；在 _rerank 的线程桥接里执行）。

        key 来源同 _embed_sync：DASHSCOPE_API_KEY 环境变量优先，否则主 key。
        """
        import dashscope
        resp = dashscope.TextReRank.call(
            model="qwen3-rerank", query=query, documents=docs,
            api_key=os.getenv("DASHSCOPE_API_KEY") or self._api_key, top_n=len(docs),
        )
        if getattr(resp, "status_code", 500) != 200:
            raise RuntimeError(
                f"rerank failed: {getattr(resp, 'code', '')} {getattr(resp, 'message', '')}"
            )
        results = resp.output.get("results", [])
        # results: [{"index": 0, "relevance_score": 0.9}, ...] → 按分数降序。
        return sorted(results, key=lambda r: r.get("relevance_score", 0.0), reverse=True)

    async def _rerank(self, query, docs):
        """rerank（线程桥接，不阻塞事件循环）。"""
        return await asyncio.to_thread(self._rerank_sync, query, docs)

    @staticmethod
    def _lesson_quality_score(lesson):
        """Map historical gain/final score to a bounded quality prior."""
        try:
            before = float(lesson.get("score_before", 0.0))
            after = float(lesson.get("score_after", 0.0))
        except (TypeError, ValueError):
            return 0.0
        gain = max(0.0, min((after - before) / 100.0, 1.0))
        final = max(0.0, min(after / 100.0, 1.0))
        return 0.6 * gain + 0.4 * final

    @staticmethod
    def _lesson_metadata_signals(lesson, context):
        """Return soft skill/domain/weak-dimension matches for RAG v2."""
        context = context or {}
        skill_name = context.get("skill_name") or ""
        domain = context.get("domain") or ""
        lesson_skill = lesson.get("skill_name") or ""
        lesson_domain = lesson.get("domain") or ""

        if skill_name:
            skill_score = 1.0 if lesson_skill == skill_name else (0.25 if not lesson_skill else 0.0)
        else:
            skill_score = 0.5 if not lesson_skill else 0.0
        if domain:
            domain_score = 1.0 if lesson_domain == domain else (0.25 if not lesson_domain else 0.0)
        else:
            domain_score = 0.5 if not lesson_domain else 0.0

        weak = {
            str(item.get("dimension"))
            for item in context.get("weak_dimensions", [])
            if isinstance(item, dict) and item.get("dimension")
        }
        lesson_dimensions = set((lesson.get("dimension_gains") or {}).keys())
        if lesson.get("target_dimension"):
            lesson_dimensions.add(str(lesson["target_dimension"]))
        dimension_score = 1.0 if weak and weak.intersection(lesson_dimensions) else 0.0
        return skill_score, domain_score, dimension_score

    async def _retrieve_lessons_quality_diverse(self, lessons, query, mode, limit, context):
        """RAG v2: multi-signal ranking, optional rerank, dedup, then MMR diversity."""
        qvec = await self._embed(query)
        await self._ensure_lesson_embeddings(lessons)
        records = []
        count = max(1, len(lessons))
        for index, lesson in enumerate(lessons):
            vec = lesson.get("embedding")
            semantic = max(0.0, min(self._cosine(qvec, vec), 1.0))
            if mode == "semantic" and semantic < self.lesson_threshold:
                continue
            sparse = self._sparse_jaccard(query, self._lesson_index_text(lesson))
            skill, domain, dimension = self._lesson_metadata_signals(lesson, context)
            components = {
                "semantic": semantic,
                "sparse": sparse,
                "skill": skill,
                "domain": domain,
                "dimension": dimension,
                "quality": self._lesson_quality_score(lesson),
                "recency": (index + 1) / count,
            }
            score = sum(
                self.lesson_signal_weights[name] * value
                for name, value in components.items()
            )
            reasons = []
            if skill >= 1.0:
                reasons.append("same_skill")
            if domain >= 1.0:
                reasons.append("same_domain")
            if dimension >= 1.0:
                reasons.append("weak_dimension")
            records.append({
                "lesson": lesson,
                "score": score,
                "components": components,
                "reasons": reasons,
                "tokens": self._lesson_tokens(self._lesson_index_text(lesson)),
                "order": index,
            })
        if not records:
            return []

        records.sort(key=lambda r: (-r["score"], r["order"]))
        pool_size = max(limit, self.lesson_candidate_pool)
        if self.lesson_rerank:
            pool_size = max(pool_size, self.lesson_rerank_pool)
        records = records[:pool_size]

        # Cross-encoder remains optional. Its rank is blended with the deterministic
        # metadata/quality score so a transient reranker cannot erase all safeguards.
        if self.lesson_rerank and records:
            try:
                reranked = await self._rerank(
                    query, [self._lesson_index_text(r["lesson"]) for r in records]
                )
                rerank_position = {
                    int(item.get("index", -1)): rank
                    for rank, item in enumerate(reranked)
                    if 0 <= int(item.get("index", -1)) < len(records)
                }
                denom = max(1, len(records))
                for original_index, record in enumerate(records):
                    rank = rerank_position.get(original_index, len(records))
                    rerank_score = max(0.0, 1.0 - rank / denom)
                    record["score"] = 0.7 * record["score"] + 0.3 * rerank_score
                    record["components"]["rerank"] = rerank_score
                records.sort(key=lambda r: (-r["score"], r["order"]))
            except Exception:
                pass

        # Remove exact/near-duplicate advice before selection. Similarity is based
        # only on model-generated lesson metadata; no skill/scenario/output is stored.
        deduped = []
        for record in records:
            if any(
                self._token_jaccard(record["tokens"], kept["tokens"])
                >= self.lesson_dedup_threshold
                for kept in deduped
            ):
                continue
            deduped.append(record)

        # Greedy maximal-marginal-relevance selection avoids injecting five versions
        # of the same fix while keeping the strongest item first.
        selected = []
        remaining = list(deduped)
        while remaining and len(selected) < max(1, limit):
            best_index = 0
            best_value = -float("inf")
            for index, record in enumerate(remaining):
                redundancy = max(
                    (self._token_jaccard(record["tokens"], item["tokens"]) for item in selected),
                    default=0.0,
                )
                value = record["score"] - self.lesson_diversity * redundancy
                if value > best_value:
                    best_index, best_value = index, value
            selected.append(remaining.pop(best_index))

        output = []
        for record in selected:
            lesson = dict(record["lesson"])
            lesson["_retrieval"] = {
                "pipeline": "quality_diverse",
                "score": round(record["score"], 4),
                "matched": record["reasons"],
                "signals": {
                    name: round(value, 4)
                    for name, value in record["components"].items()
                },
            }
            output.append(lesson)
        return output

    @staticmethod
    def _token_jaccard(left, right):
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    def _compact_lessons_for_prompt(self, lessons):
        """Whitelist and bound v2 lesson context; embeddings/internal ids never enter prompts."""
        allowed = (
            "skill_name", "domain", "strategy", "diagnosis", "summary",
            "score_before", "score_after", "lesson_type", "target_dimension",
            "dimension_gains", "comparison_scope", "_retrieval",
        )
        compact = []
        for lesson in lessons:
            item = {name: lesson[name] for name in allowed if lesson.get(name) is not None}
            for text_field in ("diagnosis", "summary"):
                if text_field in item:
                    item[text_field] = str(item[text_field])[:300]
            candidate = compact + [item]
            if len(json.dumps(candidate, ensure_ascii=False)) > self.lesson_context_chars:
                if compact:
                    break
                # Always try to retain one useful lesson, but shrink verbose fields
                # until the configured budget is genuinely respected.
                if "diagnosis" in item:
                    item["diagnosis"] = item["diagnosis"][:120]
                if "summary" in item:
                    item["summary"] = item["summary"][:120]
                retrieval = item.get("_retrieval")
                if (
                    len(json.dumps([item], ensure_ascii=False)) > self.lesson_context_chars
                    and isinstance(retrieval, dict)
                ):
                    retrieval.pop("signals", None)
                if len(json.dumps([item], ensure_ascii=False)) > self.lesson_context_chars:
                    for optional_field in (
                        "skill_name", "domain", "score_before", "score_after", "lesson_type"
                    ):
                        item.pop(optional_field, None)
                if len(json.dumps([item], ensure_ascii=False)) > self.lesson_context_chars:
                    item = {
                        name: item[name]
                        for name in ("strategy", "summary", "target_dimension")
                        if name in item
                    }
                if len(json.dumps([item], ensure_ascii=False)) > self.lesson_context_chars:
                    item = {"strategy": str(item.get("strategy", ""))[:120]}
            compact.append(item)
        return compact

    async def _retrieve_lessons(self, lessons, query, mode, limit, context=None):
        """semantic/hybrid 检索：embedding 余弦（+sparse/RRF）；可选 qwen3-rerank 重排。

        【D 选项】LESSON_RERANK 开启时对候选池（前 max(limit, rerank_pool) 条）
        调用 qwen3-rerank 重排后取 top-K；rerank 失败静默降级到原始排序。
        """
        if self.lesson_rag_pipeline == "quality_diverse":
            return await self._retrieve_lessons_quality_diverse(
                lessons, query, mode, limit, context or {}
            )
        qvec = await self._embed(query)
        await self._ensure_lesson_embeddings(lessons)
        scored = []
        for l in lessons:
            vec = l.get("embedding")
            scored.append((l, self._cosine(qvec, vec)))
        if mode == "hybrid":
            dense_rank = [l for l, _ in sorted(scored, key=lambda x: x[1], reverse=True)]
            sparse_rank = sorted(
                lessons,
                key=lambda l: self._sparse_jaccard(query, self._lesson_index_text(l)),
                reverse=True,
            )
            fused = self._rrf_fuse([dense_rank, sparse_rank])
            ranked = [l for l, _ in fused]
        else:  # semantic：余弦 >= 阈值；无达标返回空（触发外层降级）。
            ranked = [
                l for l, s in sorted(scored, key=lambda x: x[1], reverse=True)
                if s >= self.lesson_threshold
            ]
        if not ranked:
            return []
        # 【D 选项】可选 rerank 重排候选池后取 top-K；失败降级保持原始排序。
        if self.lesson_rerank:
            pool = max(limit, self.lesson_rerank_pool)
            candidates = ranked[:pool]
            try:
                reranked = await self._rerank(
                    query, [self._lesson_index_text(l) for l in candidates]
                )
                ranked = [candidates[i.get("index", 0)] for i in reranked]
            except Exception:
                pass  # 降级：保持原始排序
        return ranked[:max(1, limit)]

    def _lesson_n(self):
        try:
            return max(1, int(os.getenv("LESSON_N", "5")))
        except ValueError:
            return 5

    def _lesson_qualifies(self, score_before, score_after):
        """经验沉淀质量门槛（LESSON_MIN_GAIN / LESSON_MIN_FINAL，OR 语义）。

        任一启用的维度达标即沉淀；两个都关闭（0）时总是沉淀（旧行为）。
        只影响经验库质量，不改变优化保留语义。
        """
        gain = score_after - score_before
        if self.lesson_min_gain > 0 and gain >= self.lesson_min_gain:
            return True
        if self.lesson_min_final > 0 and score_after >= self.lesson_min_final:
            return True
        return self.lesson_min_gain <= 0 and self.lesson_min_final <= 0

    def _cross_round_tie_lesson_qualifies(self, score):
        """Apply only the final-score quality gate to a zero-total-gain lesson.

        A cross-round tie has an honest total-score gain of zero, so
        ``LESSON_MIN_GAIN`` is deliberately irrelevant. Its dimension-gain
        threshold is the lesson's own quality gate; ``LESSON_MIN_FINAL`` remains
        an optional additional persistence floor.
        """
        try:
            score = float(score)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(score):
            return False
        return self.lesson_min_final <= 0 or score >= self.lesson_min_final

    @staticmethod
    def _tie_dimension_target(dimension_gains):
        """Choose the largest-gain dimension, breaking ties by dimension name."""
        if not isinstance(dimension_gains, dict) or not dimension_gains:
            return None
        ranked = []
        for dimension, metadata in dimension_gains.items():
            if not isinstance(metadata, dict):
                continue
            try:
                gain = float(metadata.get("gain"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(gain):
                ranked.append((-gain, str(dimension)))
        if not ranked:
            return None
        return min(ranked)[1]

    def _dimension_advantages(self, candidate_scores, winner_scores, min_gain=None):
        """Return finite, shared dimensions where a tied candidate is better."""
        if not isinstance(candidate_scores, dict) or not isinstance(winner_scores, dict):
            return {}
        if min_gain is None:
            min_gain = self.tie_dimension_min_gain
        try:
            min_gain = float(min_gain)
        except (TypeError, ValueError):
            min_gain = 0.0
        if not math.isfinite(min_gain):
            min_gain = 0.0
        min_gain = max(0.0, min_gain)
        advantages = {}
        for dimension, candidate in candidate_scores.items():
            if not isinstance(candidate, dict) or "pct" not in candidate:
                continue
            winner = winner_scores.get(dimension)
            if not isinstance(winner, dict) or "pct" not in winner:
                continue
            try:
                candidate_pct = float(candidate["pct"])
                winner_pct = float(winner["pct"])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(candidate_pct) or not math.isfinite(winner_pct):
                continue
            gain = candidate_pct - winner_pct
            if gain > min_gain:
                advantages[dimension] = {
                    "candidate_pct": candidate_pct,
                    "winner_pct": winner_pct,
                    "gain": round(gain, 1),
                }
        return advantages

    async def _prepare_lessons(self, skill_md, scenarios, evals, current_details, domain=None,
                               weak_dimensions=None):
        """按 LESSON_RETRIEVAL 模式准备注入的经验；任何检索失败降级不中断。"""
        if not self.lesson_file:
            return []
        mode = self.lesson_retrieval
        read_n = self._lesson_n()
        top_k = self.lesson_top_k
        if mode == "off":
            lessons = self._load_lessons(self.lesson_file, limit=read_n)
        else:
            all_lessons = self._load_lessons(self.lesson_file, limit=1000)
            if not all_lessons:
                return []
            skill_name = self._skill_name_from_md(skill_md)
            if mode == "tag":
                lessons = self._tag_filter(all_lessons, skill_name, domain, read_n)
            else:
                query = self._build_query(
                    skill_name,
                    domain,
                    scenarios,
                    evals,
                    current_details,
                    weak_dimensions,
                    include_failure_context=self.lesson_failure_context,
                )
                if not query:
                    lessons = self._tag_filter(all_lessons, skill_name, domain, read_n)
                else:
                    try:
                        lessons = await self._retrieve_lessons(
                            all_lessons,
                            query,
                            mode,
                            top_k,
                            context={
                                "skill_name": skill_name,
                                "domain": domain or "",
                                "weak_dimensions": weak_dimensions or [],
                            },
                        )
                    except Exception:
                        # 降级链：semantic/hybrid 失败 → tag 过滤（尽力而为）。
                        lessons = self._tag_filter(all_lessons, skill_name, domain, read_n)
        # 【RAG】剥离内部检索字段；embedding 只用于排序/SQLite 复用，绝不进入
        # Analyst/Mutator prompt。v2 进一步白名单化并限制总字符预算。
        for l in lessons:
            l.pop("_id", None)
            l.pop("embedding", None)
        if self.lesson_rag_pipeline == "quality_diverse":
            lessons = self._compact_lessons_for_prompt(lessons)
        return lessons
