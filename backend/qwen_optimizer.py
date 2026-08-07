# =============================================================================
# 【文件头】qwen_optimizer.py —— 后端优化算法的核心
# 职责：用三个 Qwen-Agent 助手（Executor / Analyst / Mutator）协作，对一个
#       SKILL.md 反复执行"评估 → 诊断 → 修改 → 复评"，最终给出改进后的技能文件。
# 接收：技能文件内容 dict（含 SKILL.md 与 references/ 参考文件）、测试场景
#       scenarios、评估标准 evals、最大轮数 max_rounds，以及进度回调 callback。
# 输出：优化结果 dict（baseline_score / final_score / improved_skill_md /
#       score_history / mutation_log），并通过 callback 逐条推送进度事件。
# 建议先看：SkillOptimizer.optimize()（主循环）→ _score_skill()（评分）→
#       _run_agent_sync()（模型调用的线程桥接）。
# 【注意】本文件所有调用都是"同步 + 线程桥接"：Qwen-Agent 的 Assistant.run()
#       是同步生成器，必须放进工作线程，避免阻塞 FastAPI 的事件循环。
# =============================================================================

"""Multi-Agent Skill Optimizer using Qwen-Agent and Alibaba Cloud Model Studio (DashScope).

3 Qwen-Agent assistants work together to improve agent skills:
  Executor: runs the skill against test scenarios, scores outputs, analyzes skills
  Analyst: diagnoses why evals failed, picks a mutation strategy
  Mutator: makes one targeted fix per round
"""

import asyncio
import datetime
import json
import os
import re
from typing import Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from qwen_agent.agents import Assistant


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


class SkillMutation(BaseModel):
    description: str = Field(description="Short description of the change made")
    reasoning: str = Field(description="Why this change should help")
    new_skill_md: str = Field(description="The full updated SKILL.md content")


# -- Text extraction helper ---------------------------------------------------
# 【初学者提示】Qwen-Agent 的消息 content 可能是纯字符串，也可能是由多个片段
# （dict / str）组成的列表，这里统一提取成一段纯文本。

def _extract_text(content) -> str:
    """Extract plain text from a Qwen-Agent message content.

    Only the ``content`` field is read; ``reasoning_content`` is ignored.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                for key in ("text", "content"):
                    value = item.get(key)
                    if isinstance(value, str):
                        parts.append(value)
                        break
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


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
        lesson_file: Optional[str] = None,
        lesson_retrieval: Optional[str] = None,
        lesson_threshold: Optional[float] = None,
        lesson_top_k: Optional[int] = None,
        lesson_rerank: Optional[bool] = None,
        lesson_rerank_pool: Optional[int] = None,
        lesson_min_gain: Optional[float] = None,
        lesson_min_final: Optional[float] = None,
    ):
        # The key is passed directly into the LLM configuration; it is never
        # written to the process environment, stored in sessions, or logged.
        # 【安全】默认模型 qwen-plus，可用环境变量 QWEN_MODEL 覆盖后端默认值。
        self.model = model or os.getenv("QWEN_MODEL", "qwen-plus")
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
        # gte-rerank 重排后取 top-K；默认 0 关闭（保持现有排序）。候选池大小
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

        # 【安全】仅内存持有 key 引用，用于 embedding/rerank 的 DashScope 调用；
        # 绝不打印、不落日志、不写环境变量、不序列化进任何返回结构。
        self._api_key = api_key
        # 内存 embedding 缓存（同文本不重复调用）。
        self._embed_cache = {}

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

        # 【C3】并行变异数：默认 1（与旧行为一致）；可用 MUTATION_PARALLELISM
        # 环境变量或请求字段 parallel_mutations 开启并行，上限 3。
        if parallelism is None:
            try:
                parallelism = int(os.getenv("MUTATION_PARALLELISM", "1"))
            except ValueError:
                parallelism = 1
        self.parallelism = max(1, min(int(parallelism), 3))

        # 三个助手共用同一份 LLM 配置：模型类型固定为 qwen_dashscope（阿里云
        # 百炼），默认关闭思考模式并压低 temperature，让输出更稳定、便于解析。
        # 【模型兼容】部分模型（如 qwen3.7-max-2026-05-17 日期快照）强制要求
        # enable_thinking=True，可用 QWEN_ENABLE_THINKING=1 打开；默认 0 保持
        # 旧行为（关闭）。thinking 输出含 reasoning_content，_extract_text
        # 只取 content，解析路径不受影响。
        llm_config = {
            "model": self.model,
            "model_type": "qwen_dashscope",
            "api_key": api_key,
            "generate_cfg": {
                "enable_thinking": os.getenv("QWEN_ENABLE_THINKING", "0") != "0",
                "temperature": 0.2,
            },
        }

        # Executor：三种模式 —— 执行技能 / 生成测试场景与评分标准 / 给输出打分。
        self.executor = Assistant(
            name="executor",
            system_message=EXECUTOR_SYSTEM_PROMPT,
            llm=llm_config,
        )
        # Analyst：根据失败的评估结果定位根因，并选择一种修改策略。
        self.analyst = Assistant(
            name="analyst",
            system_message=ANALYST_SYSTEM_PROMPT,
            llm=llm_config,
        )
        # Mutator：根据诊断对 SKILL.md 做"恰好一处"针对性修改，返回完整新内容。
        self.mutator = Assistant(
            name="mutator",
            system_message=MUTATOR_SYSTEM_PROMPT,
            llm=llm_config,
        )

        # 【C3 并行】每个变异槽位持有独立的 Mutator / Executor 实例。并行
        # 运行时必须用各自独立的 Assistant，避免在同一实例上并发 run() 引发
        # 线程安全问题。parallelism=1 时只建主实例（与旧行为完全一致）。
        self.mutator_pool = [self.mutator] + [
            Assistant(
                name=f"mutator-{i}",
                system_message=MUTATOR_SYSTEM_PROMPT,
                llm=llm_config,
            )
            for i in range(1, self.parallelism)
        ]
        self.executor_pool = [self.executor] + [
            Assistant(
                name=f"executor-{i}",
                system_message=EXECUTOR_SYSTEM_PROMPT,
                llm=llm_config,
            )
            for i in range(1, self.parallelism)
        ]

    # -- Agent runner helpers ------------------------------------------------
    # 【异步】以下三个辅助函数是"同步调用 ↔ 异步接口"的桥接层：
    # _run_agent_sync 在普通线程里执行同步生成器，_ask 用 asyncio.to_thread
    # 把它交给线程池，从而不阻塞 FastAPI 事件循环（轮询/SSE 才能继续响应）。

    def _run_agent_sync(self, agent: Assistant, prompt: str) -> str:
        """Run a Qwen-Agent assistant synchronously, returning the final assistant text.

        ``Assistant.run()`` is a synchronous generator; the final batch of messages
        holds the complete answer. Only the ``content`` of the last assistant turn
        is kept (``reasoning_content`` is ignored).
        """
        responses = None
        # 【初学者提示】run() 是同步生成器：一边流式产出中间结果，一边 yield。
        # 用循环把它"跑完"，最后一次迭代拿到的 responses 就是完整回答。
        for responses in agent.run(
            [{"role": "user", "content": prompt}],
            stream=False,
        ):
            pass
        if not responses:
            raise RuntimeError("Qwen-Agent returned no response")
        # 从最后一轮 assistant 消息中取文本；reasoning_content 被忽略。
        for msg in reversed(responses):
            if msg.get("role") == "assistant":
                text = _extract_text(msg.get("content"))
                if text:
                    return text
        raise RuntimeError("Qwen-Agent returned no text response")

    async def _ask(self, agent: Assistant, prompt: str) -> str:
        """Run a Qwen-Agent assistant with a prompt, return text response.

        The model request is bridged through a worker thread so it never blocks
        the FastAPI event loop handling status polling / SSE.
        """
        # 【异步】asyncio.to_thread 把阻塞的同步调用丢进线程池执行，返回可 await
        # 的结果；这样模型在思考时，事件循环仍能处理其他 HTTP 请求。
        return await asyncio.to_thread(self._run_agent_sync, agent, prompt)

    async def _ask_json(self, agent: Assistant, prompt: str, fallback=None, schema=None):
        """Run a Qwen-Agent assistant and parse the JSON response.

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
        text = await self._ask(agent, prompt)

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
    ) -> dict:
        """Run the optimization loop with 3 Qwen-Agent assistants.

        【C2/C3/C4/C5/P1-P3】新增可选参数均为"默认即旧行为"：
        - parallel_mutations：并行变异数（None → 用构造时的 parallelism）
        - strategy_pool：Analyst 可用策略白名单（None → 环境变量/全量）
        - patience：耐心早停轮数（None → 用构造时的 PATIENCE，默认 0 关闭）
        核心不变量不变：每个候选只改一处；仅当分数严格高于
        当前最佳 + improvement_threshold（+ noise_floor）时才保留。
        """

        # 【主流程】callback 是"事件出口"：每次评估/每轮结束都向它推送事件，
        # 由 app.py 把事件写进 session 的事件队列或 experiments 列表。
        async def emit(event):
            if callback:
                await callback(event)

        skill_md = next(
            (v for k, v in skill_files.items() if k.endswith("SKILL.md")), ""
        )
        current_md = skill_md
        score_history = []
        mutation_log = []
        # 【P2 轮间记忆】每轮结束后追加一条"诊断+结果"，供下一轮 Analyst 参考。
        round_memory = []
        # 生效的并行数：请求参数优先，否则用构造时的 parallelism（默认 1）。
        n_candidates = parallel_mutations or self.parallelism
        n_candidates = max(1, min(int(n_candidates), 3))
        # 生效的策略池。
        pool = _active_strategy_pool(strategy_pool)

        # 分数归一化辅助：有权重时用加权分，否则用通过率（与旧版一致）。
        def pct_of(result):
            if self.dimension_weights and result.get("weighted_pct") is not None:
                return float(result["weighted_pct"])
            return round(100 * result["passed"] / max(result["total"], 1), 1)

        # -- Baseline ---------------------------------------------------------
        # 【主流程】先给原始技能打分，得到基线分数（baseline）；后续每轮修改
        # 只有分数"严格更高"才会被保留，否则丢弃。
        self.check_stop()
        baseline = await self._score_skill(current_md, scenarios, evals)
        baseline_pct = pct_of(baseline)
        # 当前版本的失败明细：每轮 Analyst 基于它诊断；保留新版本后随之更新。
        current_details = baseline["details"]
        score_history.append(baseline_pct)

        await emit({
            "type": "baseline",
            "data": {
                "score": baseline_pct,
                "passed": baseline["passed"],
                "total": baseline["total"],
                "per_eval": baseline["per_eval"],
                "dimension_scores": baseline.get("dimension_scores", {}),
            },
        })

        # 【P4/RAG】按 LESSON_RETRIEVAL 模式准备注入的经验（off/tag/semantic/
        # hybrid）；依赖基线失败明细构建检索查询，故放在 baseline 之后。
        lessons = await self._prepare_lessons(
            skill_md, scenarios, evals, current_details, domain=domain,
        )

        # -- Rounds -----------------------------------------------------------
        # 【P3 饱和快速退出】基线无可提升带（100%，或 0 分且所有评估全失败）
        # 时跳过全部轮次，避免空耗；默认关闭，开启后与 100% 早停同源。
        saturated = (
            self.saturation_exit
            and (baseline_pct >= 100.0 or (baseline_pct <= 0.0 and baseline.get("total", 0) > 0))
        )
        # 【P3 耐心早停】连续 N 轮未提升即终止；0 表示关闭。
        effective_patience = patience if patience is not None else self.patience
        no_improve = 0

        # 【主流程】循环主体：诊断 → 修改 → 复评 → 决定保留/丢弃，最多 max_rounds 轮。
        for rnd in range(1, max_rounds + 1):
            # 【主流程】进度 100% 提前终止：当前最佳分数已达满分（所有评估
            # 全部通过），后续轮次不可能再提升，立即跳出循环，不再空转消耗
            # token。用 >= 100.0 判定，兼容浮点边界，避免漏判。
            if baseline_pct >= 100.0 or saturated:
                break
            # 【P3 耐心早停】连续 effective_patience 轮未提升 → 提前终止。
            if effective_patience > 0 and no_improve >= effective_patience:
                break
            self.check_stop()
            await emit({"type": "experiment_start", "data": {"round": rnd}})

            # Analyst diagnoses worst failure
            # 第 1 个 Agent（Analyst）：分析当前失败项，给出根因与修改策略。
            # 【P2/P4】把轮间记忆、被拒编辑历史与跨会话经验一并交给 Analyst。
            analysis = await self._analyze_failures(
                current_md, scenarios, evals, current_details,
                strategy_pool=pool,
                round_memory=round_memory,
                mutation_log=mutation_log,
                lessons=lessons,
            )

            # 【C3 并行变异】同一次诊断产出 N 个候选：每个槽位用独立的 Mutator
            # 实例，并给不同策略角度 hint，让候选彼此差异化；随后并行复评，
            # 取分数最优的那个。n_candidates=1 时与旧版逐轮流程完全一致。
            if n_candidates > 1:
                mutations = await asyncio.gather(*[
                    self._mutate_skill(
                        current_md,
                        analysis,
                        agent=self.mutator_pool[i % len(self.mutator_pool)],
                        strategy_hint=(
                            STRATEGY_TEMPLATES[pool[i % len(pool)]].get("mutator_hint")
                            if pool else None
                        ),
                        lessons=lessons,
                    )
                    for i in range(n_candidates)
                ])
            else:
                mutations = [await self._mutate_skill(current_md, analysis, lessons=lessons)]

            # 【C4 回归守卫】先对每个候选做结构完整性检查，命中即拒绝并跳过
            # 复评（省 token）；随后并行复评通过检查的候选。
            candidates = []
            for i, mutation in enumerate(mutations):
                new_md = mutation.get("new_skill_md", current_md)
                if self.regression_check:
                    ok, reason = self._regression_check(current_md, new_md)
                    if not ok:
                        candidates.append({
                            "candidate_id": i,
                            "new_md": None,
                            "description": mutation.get("description", ""),
                            "reasoning": mutation.get("reasoning", ""),
                            "rejected": reason,
                            "score_after": baseline_pct,
                        })
                        continue
                # 【P1 编辑幅度】开启时单次变异相对原文本变化比例超限即拒绝，
                # 复用回归拒绝管道（只更严，不改变结构守卫的判定）。
                if self.edit_limit > 0 and current_md:
                    change_ratio = abs(len(new_md) - len(current_md)) / max(len(current_md), 1)
                    if change_ratio > self.edit_limit:
                        candidates.append({
                            "candidate_id": i,
                            "new_md": None,
                            "description": mutation.get("description", ""),
                            "reasoning": mutation.get("reasoning", ""),
                            "rejected": "edit_limit_exceeded",
                            "score_after": baseline_pct,
                        })
                        continue
                candidates.append({
                    "candidate_id": i,
                    "new_md": new_md,
                    "description": mutation.get("description", ""),
                    "reasoning": mutation.get("reasoning", ""),
                    "rejected": None,
                    "score_after": baseline_pct,
                })

            # 并行复评所有未拒绝的候选。
            to_score = [c for c in candidates if c["new_md"] is not None]
            if to_score:
                results = await asyncio.gather(*[
                    self._score_skill(
                        c["new_md"], scenarios, evals,
                        agent=self.executor_pool[c["candidate_id"] % len(self.executor_pool)],
                    )
                    for c in to_score
                ])
                for c, res in zip(to_score, results):
                    c["score_after"] = pct_of(res)
                    c["per_eval"] = res["per_eval"]
                    c["dimension_scores"] = res.get("dimension_scores", {})
                    c["details"] = res["details"]

            # 【主流程】选出本轮最优候选：分数最高者；被回归拒绝的永不入选。
            scored = [c for c in candidates if c["rejected"] is None and c["new_md"] is not None]
            best = max(scored, key=lambda c: c["score_after"], default=None)

            # 【主流程】保留条件：最优候选的分必须严格高于"当前基线 + 阈值
            # + 噪声地板"（kept = True 才采纳）；等于或低于基线的 mutation
            # 都会被丢弃。这是本项目的核心行为不变量，避免模型乱改导致技能
            # 退化；noise_floor 默认 0.0，仅在显式开启时抬高门槛（更严格）。
            kept = (
                best is not None
                and best["score_after"] > baseline_pct + self.improvement_threshold + self.noise_floor
            )
            new_pct = best["score_after"] if best else baseline_pct

            # 【C3】每个候选各记一条 mutation_log（含 candidate_id / reason）；
            # 只有最优且严格提升的候选才成为新的当前版本。
            for c in candidates:
                is_winner = kept and best is not None and c["candidate_id"] == best["candidate_id"]
                entry = {
                    "round": rnd,
                    "candidate_id": c["candidate_id"],
                    "strategy_type": analysis.get("mutation_strategy", "unknown"),
                    "diagnosis": analysis.get("diagnosis", ""),
                    "description": c["description"],
                    "score_before": baseline_pct,
                    "score_after": c["score_after"],
                    "kept": is_winner,
                    "reason": c["rejected"] or ("best" if is_winner else "not_best"),
                }
                mutation_log.append(entry)

            # 【P2 轮间记忆】压缩本轮"诊断+结果"为一条记录，供下一轮 Analyst 参考。
            if best is not None:
                outcome = "rejected" if best["rejected"] else ("kept" if kept else "discarded")
                mem_reason = best["rejected"] or ("best" if kept else "not_best")
            else:
                outcome, mem_reason = "discarded", "no_candidate"
            round_memory.append({
                "round": rnd,
                "strategy": analysis.get("mutation_strategy", "unknown"),
                "diagnosis": (analysis.get("diagnosis") or "")[:200],
                "target": analysis.get("target_section", ""),
                "outcome": outcome,
                "score_before": baseline_pct,
                "score_after": new_pct,
                "reason": mem_reason,
            })

            if kept and best is not None:
                # 【P4】保留的修改沉淀为跨会话经验（只记模型生成内容，不落
                # skill_md / scenario / output / api_key，安全）。skill_name 供
                # tag/semantic 检索模式按技能过滤，created_at 记录时间。
                # 【质量门槛】仅当提升幅度或最终水位达标时才沉淀，过滤小修
                # 噪音（_lesson_qualifies，默认关闭=任何 kept 都沉淀）。
                if self._lesson_qualifies(baseline_pct, best["score_after"]):
                    self._append_lesson(self.lesson_file, {
                        "skill_name": self._skill_name_from_md(skill_md),
                        "domain": domain or "",
                        "skill_description": "",
                        "strategy": analysis.get("mutation_strategy", "unknown"),
                        "diagnosis": (analysis.get("diagnosis") or "")[:200],
                        "summary": (best.get("description") or "")[:200],
                        "score_before": baseline_pct,
                        "score_after": best["score_after"],
                        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
                    })
                current_md = best["new_md"]
                baseline_pct = best["score_after"]
                current_details = best.get("details", current_details)

            # 【P3 耐心早停】计数连续未提升轮数；保留即清零。
            no_improve = 0 if kept else no_improve + 1

            score_history.append(baseline_pct)

            await emit({
                "type": "experiment_result",
                "data": {
                    "round": rnd,
                    "score": new_pct,
                    "kept": kept,
                    "status": "kept" if kept else "discarded",
                    "description": best["description"] if best else "",
                    "strategy": analysis.get("mutation_strategy", ""),
                    "per_eval": best["per_eval"] if best else [],
                    "candidates": [
                        {
                            "candidate_id": c["candidate_id"],
                            "description": c["description"],
                            "score": c["score_after"],
                            "reason": c["rejected"] or ("best" if (kept and best and c["candidate_id"] == best["candidate_id"]) else "not_best"),
                        }
                        for c in candidates
                    ],
                    "dimension_scores": best["dimension_scores"] if best else {},
                    "diff_summary": (
                        self._diff_summary(current_md, best["new_md"])
                        if kept and best is not None else ""
                    ),
                },
            })

        # -- Done -------------------------------------------------------------
        # 【P3 胜出候选复核】开启时对最终版本独立再评一次：若复核分未超过
        # 基线（含阈值/噪声地板），说明此前的提升可能是单点幸运，回退为原始
        # 技能（不采纳）。默认关闭。
        if self.final_confirm and current_md != skill_md:
            confirm = await self._score_skill(current_md, scenarios, evals)
            confirm_pct = pct_of(confirm)
            if confirm_pct <= score_history[0] + self.improvement_threshold + self.noise_floor:
                current_md = skill_md
                baseline_pct = score_history[0]

        # 【主流程】全部轮次结束，推送 complete 事件并返回汇总结果。
        final_pct = baseline_pct
        await emit({
            "type": "complete",
            "data": {
                "baseline_score": score_history[0],
                "final_score": final_pct,
                "improved_skill_md": current_md,
                "score_history": score_history,
                "mutation_log": mutation_log,
                "strategy_stats": self._strategy_stats(mutation_log),
            },
        })

        return {
            "baseline_score": score_history[0],
            "final_score": final_pct,
            "improved_skill_md": current_md,
            "score_history": score_history,
            "mutation_log": mutation_log,
        }

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
        """【C5】协作式取消检查：在轮间调用，stop_requested 置位即抛异常。"""
        if getattr(self, "stop_requested", False):
            raise StopOptimizationError("Optimization stopped by user request")

    async def _analyze_failures(self, skill_md, scenarios, evals, details, strategy_pool=None,
                                round_memory=None, mutation_log=None, lessons=None):
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
                        lessons.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return []
        return lessons[-max(1, limit):]

    @staticmethod
    def _load_lessons_sqlite(path, limit=5):
        try:
            import sqlite3
            conn = sqlite3.connect(path)
            try:
                rows = conn.execute(
                    "SELECT id, skill_name, domain, skill_description, strategy, diagnosis, "
                    "summary, score_before, score_after, created_at, embedding "
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
                    lessons.append(lesson)
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
                    "embedding BLOB)"
                )
                conn.execute(
                    "INSERT INTO lessons (skill_name, domain, skill_description, strategy, "
                    "diagnosis, summary, score_before, score_after, created_at, embedding) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        lesson.get("skill_name"), lesson.get("domain"),
                        lesson.get("skill_description"), lesson.get("strategy"),
                        lesson.get("diagnosis"), lesson.get("summary"),
                        lesson.get("score_before"), lesson.get("score_after"),
                        lesson.get("created_at"),
                        SkillOptimizer._embedding_to_blob(lesson.get("embedding")),
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
        """经验 → 检索文本（strategy + diagnosis + summary 拼接）。"""
        parts = [
            lesson.get("strategy", ""),
            lesson.get("diagnosis", ""),
            lesson.get("summary", ""),
        ]
        return " ".join(str(p) for p in parts if p)

    @staticmethod
    def _build_query(skill_name, domain, scenarios, evals, current_details):
        """检索查询：领域 + 技能名 + 最近失败原因 + 场景输入摘要（规则拼接）。"""
        parts = []
        if domain:
            parts.append(f"domain: {domain}")
        if skill_name:
            parts.append(f"skill: {skill_name}")
        failed = [d for d in (current_details or []) if not d.get("passed")]
        for d in failed[:3]:
            parts.append(str(d.get("reason") or ""))
        for s in (scenarios or [])[:2]:
            parts.append(str(s.get("input", ""))[:80])
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
        """极简稀疏分：查询与文本的词集合 Jaccard（术语精确匹配兜底）。"""
        q = set(re.findall(r"[a-z0-9]+", (query or "").lower()))
        t = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
        if not q or not t:
            return 0.0
        return len(q & t) / len(q | t)

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
        """DashScope text-embedding-v3（同步；在 _embed 的线程桥接里执行）。"""
        import dashscope
        resp = dashscope.TextEmbedding.call(
            model="text-embedding-v3", input=text,
            api_key=self._api_key, dimensions=512,
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

    def _rerank_sync(self, query, docs):
        """DashScope gte-rerank（同步；在 _rerank 的线程桥接里执行）。"""
        import dashscope
        resp = dashscope.TextReRank.call(
            model="gte-rerank", query=query, documents=docs,
            api_key=self._api_key, top_n=len(docs),
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

    async def _retrieve_lessons(self, lessons, query, mode, limit):
        """semantic/hybrid 检索：embedding 余弦（+sparse/RRF）；可选 gte-rerank 重排。

        【D 选项】LESSON_RERANK 开启时对候选池（前 max(limit, rerank_pool) 条）
        调用 gte-rerank 重排后取 top-K；rerank 失败静默降级到原始排序。
        """
        qvec = await self._embed(query)
        scored = []
        for l in lessons:
            vec = l.get("embedding")
            if vec is None:
                vec = await self._embed(self._lesson_index_text(l))
                l["embedding"] = vec
                # 【RAG】惰性 embedding 写回 SQLite，下次会话直接复用。
                if l.get("_id") is not None:
                    self._update_lesson_embedding(self.lesson_file, l["_id"], vec)
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

    async def _prepare_lessons(self, skill_md, scenarios, evals, current_details, domain=None):
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
                query = self._build_query(skill_name, domain, scenarios, evals, current_details)
                if not query:
                    lessons = self._tag_filter(all_lessons, skill_name, domain, read_n)
                else:
                    try:
                        lessons = await self._retrieve_lessons(all_lessons, query, mode, top_k)
                    except Exception:
                        # 降级链：semantic/hybrid 失败 → tag 过滤（尽力而为）。
                        lessons = self._tag_filter(all_lessons, skill_name, domain, read_n)
        # 【RAG】剥离内部 _id（仅用于 embedding 写回），不让其进入注入内容。
        for l in lessons:
            l.pop("_id", None)
        return lessons
