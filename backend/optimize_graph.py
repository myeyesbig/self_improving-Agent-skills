# =============================================================================
# 【文件头】optimize_graph.py —— LangGraph 状态图：把 SkillOptimizer 的手写
#       轮循环显式化为"图即文档"的编排层。
# 职责：只做编排——节点调用 qwen_optimizer.py 中可独立测试的实例方法
#       （_score_skill / _analyze_failures / _mutate_skill / _regression_check /
#       _prepare_lessons 等），自身不含算法逻辑；并行变异/复评用 Send API
#       map-reduce 扇出扇入；早停/耐心/协作停止用条件边声明式路由。
# 状态：OptimizeState（扁平 TypedDict）；mutation_log / round_memory /
#       score_history / pending_events 用 add reducer 追加。
# 事件：节点把要 emit 的事件写进 pending_events，optimize() 消费 astream 时
#       逐个回调，事件顺序与手写版逐字一致（baseline → experiment_start →
#       experiment_result ×N → complete）。
# 【注意】本文件只使用 LangGraph，不导入禁止的上层编排命名空间。
# =============================================================================

"""LangGraph StateGraph for the SkillForge optimization loop.

Nodes call the optimizer's existing testable methods; the graph itself only
orchestrates. Parallel candidate generation and rescoring use the Send API
(map-reduce), and early exits (100% score / saturation / patience / cooperative
stop) are declared as conditional edges instead of nested ``if`` blocks.
"""

import copy
import datetime
import math
from operator import add
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from qwen_optimizer import STRATEGY_TEMPLATES


def _append_or_reset(old: list, new: list) -> list:
    """Send 汇聚 reducer：非空新值追加到旧值；空列表表示"重置本字段"。

    LangGraph 以 ``reducer(old, new)`` 调用（与 operator.add 的 old+new 一致）；
    并行 Send 子任务对同一字段的写入会依次 apply。轮首由 analyst 节点返回
    空列表显式清空，防止 mutations/rescored 这些"每轮中间产物"跨轮累积
    （计划红线：Send 汇聚隔离）。
    """
    if not new:
        return []
    return (old or []) + new


class OptimizeState(TypedDict):
    # -- 输入与配置（只读，由 optimize() 注入）--------------------------------
    skill_md: str                     # 原始技能内容（不变）
    domain: str
    n_candidates: int                 # 本轮并行变异数（clamp 1-3）
    effective_patience: int           # 耐心早停轮数（0=关闭）
    # -- 跨轮状态 -----------------------------------------------------------
    current_md: str                   # 当前最优 SKILL.md
    baseline_pct: float               # 当前最优分数
    current_details: list             # 当前版本的失败明细
    current_dimension_scores: dict    # 当前版本各评分维度统计
    weak_dimensions: list             # 目标 4：当前低分维度（默认空）
    score_history: Annotated[list, add]   # 每轮结束追加当前分
    mutation_log: Annotated[list, add]    # 每候选一条（含 kept/reason）
    round_memory: Annotated[list, add]    # 每轮一条"诊断+结果"（P2 轮间记忆）
    lessons: list                     # 跨会话经验注入（P4/RAG）
    round_idx: int                    # 已完成的轮次数
    no_improve: int                   # 连续未提升轮数（P3 耐心早停）
    saturated: bool                   # 基线无可提升带（P3 饱和退出）
    # -- 每轮中间产物（不跨轮共享；analyst 每轮开头重置）-----------------------
    analysis: Optional[dict]          # Analyst 输出
    mutations: Annotated[list, _append_or_reset]  # Send 汇聚：各槽位 raw mutation
    candidates: list                  # 回归守卫后的候选（覆盖式）
    rescored: Annotated[list, _append_or_reset]  # Send 汇聚：复评后的候选
    confirmation: dict                 # 可选 provisional winner 配对复核结果
    kept: bool                        # 本轮是否保留最优候选
    final_pct: float                  # 本轮最优分（事件用）
    # -- 事件与结果出口 ------------------------------------------------------
    pending_events: Annotated[list, add]  # 节点产生的进度事件（optimize() 消费）
    final_result: dict                # finalize 节点返回的最终结果（覆盖式）


def build_optimize_graph(
    optimizer,
    *,
    max_rounds: int,
    n_candidates: int,
    strategy_pool: list,
    effective_patience: int,
    scenarios: list,
    evals: list,
    checkpointer=None,
):
    """Build and compile the optimization StateGraph.

    ``optimizer`` is the SkillOptimizer instance; every node calls its methods
    so that unit tests patching those methods keep working unchanged.
    ``scenarios`` / ``evals`` are immutable inputs captured by closure, so
    Send sub-nodes (which receive only their payload) can reuse them safely.
    ``checkpointer`` (optional, stage 5) enables resume-after-restart keyed by
    ``thread_id``; when set, the graph requires a config at stream time.
    """
    opt = optimizer
    pool = strategy_pool

    # 分数归一化：有权重用加权分，否则用通过率（与手写版逐字一致）。
    def pct_of(result: dict) -> float:
        if opt.dimension_weights and result.get("weighted_pct") is not None:
            return float(result["weighted_pct"])
        return round(100 * result["passed"] / max(result["total"], 1), 1)

    # -- 节点：baseline -------------------------------------------------------
    async def node_baseline(state: OptimizeState) -> dict:
        # 【C5】开始前的协作停止检查（等价手写版在评分前调用 check_stop）。
        opt.check_stop()
        baseline = await opt._score_skill(state["skill_md"], scenarios, evals)
        baseline_pct = pct_of(baseline)
        # 【P3 饱和快速退出】基线无可提升带（100%，或 0 分且全部失败）。
        saturated = (
            opt.saturation_exit
            and (baseline_pct >= 100.0 or (baseline_pct <= 0.0 and baseline.get("total", 0) > 0))
        )
        event = {
            "type": "baseline",
            "data": {
                "score": baseline_pct,
                "passed": baseline["passed"],
                "total": baseline["total"],
                "per_eval": baseline["per_eval"],
                "dimension_scores": baseline.get("dimension_scores", {}),
            },
        }
        dimension_scores = baseline.get("dimension_scores", {})
        return {
            "current_md": state["skill_md"],
            "baseline_pct": baseline_pct,
            "current_details": baseline["details"],
            "current_dimension_scores": dimension_scores,
            "weak_dimensions": opt._identify_weak_dimensions(dimension_scores),
            "score_history": [baseline_pct],
            "saturated": saturated,
            "pending_events": [event],
        }

    # -- 节点：经验准备（P4/RAG）---------------------------------------------
    async def node_prepare_lessons(state: OptimizeState) -> dict:
        prepare_kwargs = {"domain": state["domain"] or None}
        if state.get("weak_dimensions"):
            prepare_kwargs["weak_dimensions"] = state["weak_dimensions"]
        lessons = await opt._prepare_lessons(
            state["skill_md"], scenarios, evals,
            state["current_details"], **prepare_kwargs,
        )
        return {"lessons": lessons}

    # -- 条件路由：早停 / 耐心 / 达最大轮数 → finalize，否则下一轮 -------------
    def route_check(state: OptimizeState) -> str:
        if state["round_idx"] >= max_rounds:
            return "exit"
        if state["baseline_pct"] >= 100.0 or state["saturated"]:
            return "exit"
        if state["effective_patience"] > 0 and state["no_improve"] >= state["effective_patience"]:
            return "exit"
        return "continue"

    # -- 节点：轮间协作停止 + 轮开始事件 ---------------------------------------
    async def node_check_stop(state: OptimizeState) -> dict:
        # 【C5】轮间协作取消：置位即抛 StopOptimizationError（在途调用已完成），
        # 由 astream 冒泡到 app.py；语义与手写版逐字一致，不用 interrupt。
        opt.check_stop()
        rnd = state["round_idx"] + 1
        return {"pending_events": [{"type": "experiment_start", "data": {"round": rnd}}]}

    # -- 节点：Analyst 诊断 ---------------------------------------------------
    async def node_analyst(state: OptimizeState) -> dict:
        analyze_kwargs = {
            "strategy_pool": pool,
            "round_memory": state.get("round_memory", []),
            "mutation_log": state.get("mutation_log", []),
            "lessons": state.get("lessons", []),
        }
        if state.get("weak_dimensions"):
            analyze_kwargs["weak_dimensions"] = state["weak_dimensions"]
        analysis = await opt._analyze_failures(
            state["current_md"], scenarios, evals, state["current_details"],
            **analyze_kwargs,
        )
        # 【隔离红线】每轮开头重置 Send 汇聚中间产物（mutations/rescored），
        # 防止上一轮残留候选混入本轮（_append_or_reset 的空列表语义）。
        return {"analysis": analysis, "mutations": [], "rescored": [], "confirmation": {}}

    # -- 扇出路由：并行变异（Send map-reduce，LangGraph 1.x 由条件边返回）------
    def route_mutate(state: OptimizeState) -> list:
        """Analyst 之后：把本轮变异扇出到 N 个独立 mutate_slot 子任务。"""
        n = state["n_candidates"]
        analysis = state["analysis"]
        current_md = state["current_md"]
        lessons = state.get("lessons", [])
        weak_dimensions = state.get("weak_dimensions", [])
        candidate_pool = (
            opt._focused_strategy_pool(pool, weak_dimensions)
            if weak_dimensions else pool
        )
        primary_strategy = analysis.get("mutation_strategy", pool[0] if pool else "add_constraint")
        if opt.search_policy == "adaptive":
            slot_strategies = opt._select_candidate_strategies(
                candidate_pool, state.get("mutation_log", []), n, primary_strategy,
            )
        else:
            # classic 完整保留旧行为：所有候选沿用同一份 Analyst 策略。
            slot_strategies = [primary_strategy] * n
        sends = []
        for i in range(n):
            strategy = slot_strategies[i]
            slot_analysis = copy.deepcopy(analysis)
            # adaptive 下每个槽位得到自洽的策略字段与提示；classic 不改诊断。
            if opt.search_policy == "adaptive":
                slot_analysis["mutation_strategy"] = strategy
            if weak_dimensions:
                weak_names = {item["dimension"] for item in weak_dimensions}
                if slot_analysis.get("target_dimension") not in weak_names:
                    slot_analysis["target_dimension"] = weak_dimensions[0]["dimension"]
            # 【C1】classic 并行沿用原轮换提示；adaptive 使用分配策略的提示。
            hint = None
            if opt.search_policy == "adaptive" and strategy in STRATEGY_TEMPLATES:
                hint = STRATEGY_TEMPLATES[strategy].get("mutator_hint")
            elif n > 1 and candidate_pool:
                hint = STRATEGY_TEMPLATES[candidate_pool[i % len(candidate_pool)]].get("mutator_hint")
            # 【隔离红线】Send 子状态输入深拷贝，禁止共享可变对象。
            sends.append(Send("mutate_slot", {
                "slot": i,
                "current_md": copy.deepcopy(current_md),
                "analysis": slot_analysis,
                "candidate_strategy": strategy,
                "strategy_hint": hint,
                "lessons": copy.deepcopy(lessons),
            }))
        return sends

    async def node_mutate_slot(payload: dict) -> dict:
        mutation = await opt._mutate_skill(
            payload["current_md"], payload["analysis"],
            strategy_hint=payload["strategy_hint"],
            lessons=payload.get("lessons", []),
        )
        # 带槽位号，guard 汇聚时按槽位定序聚合（保持日志顺序确定）。
        mutation["_slot"] = payload["slot"]
        mutation["_strategy_type"] = payload["candidate_strategy"]
        mutation["_target_dimension"] = payload["analysis"].get("target_dimension")
        return {"mutations": [mutation]}

    # -- 节点：回归守卫 + 编辑幅度（纯拒绝管道，只更严不放松）------------------
    async def node_guard(state: OptimizeState) -> dict:
        current_md = state["current_md"]
        baseline_pct = state["baseline_pct"]
        # 按槽位号排序恢复确定性顺序（Send 完成顺序不定）。
        mutations = sorted(state.get("mutations", []), key=lambda m: m.get("_slot", 0))
        candidates = []
        for i, mutation in enumerate(mutations):
            new_md = mutation.get("new_skill_md", current_md)
            entry = {
                "candidate_id": mutation.get("_slot", i),
                "new_md": new_md,
                "description": mutation.get("description", ""),
                "reasoning": mutation.get("reasoning", ""),
                "strategy_type": mutation.get(
                    "_strategy_type", (state.get("analysis") or {}).get("mutation_strategy", "unknown")
                ),
                "target_dimension": mutation.get(
                    "_target_dimension", (state.get("analysis") or {}).get("target_dimension")
                ),
                "rejected": None,
                "score_after": baseline_pct,
            }
            # 【C4 回归守卫】结构完整性检查，命中即拒绝并跳过复评（省 token）。
            if opt.regression_check:
                ok, reason = opt._regression_check(current_md, new_md)
                if not ok:
                    entry["new_md"] = None
                    entry["rejected"] = reason
                    candidates.append(entry)
                    continue
            # 【P1 编辑幅度】变化比例超限即拒绝，复用拒绝管道（只更严）。
            if opt.edit_limit > 0 and current_md:
                change_ratio = abs(len(new_md) - len(current_md)) / max(len(current_md), 1)
                if change_ratio > opt.edit_limit:
                    entry["new_md"] = None
                    entry["rejected"] = "edit_limit_exceeded"
                    candidates.append(entry)
                    continue
            candidates.append(entry)
        return {"candidates": candidates}

    # -- 扇出路由：并行复评（Send map-reduce；全被拒时直接 decide）-------------
    def route_reeval(state: OptimizeState) -> list:
        """guard 之后：把未拒绝的候选扇出到 reeval_slot；无候选则走 decide。"""
        sends = []
        for c in state.get("candidates", []):
            if c["new_md"] is not None:
                sends.append(Send("reeval_slot", {"candidate": copy.deepcopy(c)}))
        if sends:
            return sends
        return ["skip"]  # 全部被回归拒绝 → 跳过复评，直接 decide

    async def node_reeval_slot(payload: dict) -> dict:
        c = payload["candidate"]
        res = await opt._score_skill(c["new_md"], scenarios, evals)
        c["score_after"] = pct_of(res)
        c["per_eval"] = res["per_eval"]
        c["dimension_scores"] = res.get("dimension_scores", {})
        c["details"] = res["details"]
        return {"rescored": [c]}

    def merged_candidates(state: OptimizeState) -> list:
        """Merge deterministic Send results without mutating checkpoint state in place."""
        rescored = {c["candidate_id"]: c for c in state.get("rescored", [])}
        candidates = []
        for original in state.get("candidates", []):
            candidate = dict(original)
            result = rescored.get(candidate["candidate_id"])
            if result is not None:
                candidate.update(result)
            candidates.append(candidate)
        return candidates

    # -- 节点：暂定胜者与当轮 incumbent 配对复核（默认关闭）-------------------
    async def node_confirm_candidate(state: OptimizeState) -> dict:
        candidates = merged_candidates(state)
        scored = [
            candidate for candidate in candidates
            if candidate["rejected"] is None and candidate["new_md"] is not None
        ]
        best = max(scored, key=lambda candidate: candidate["score_after"], default=None)
        threshold = opt.improvement_threshold + opt.noise_floor
        if (
            opt.candidate_confirm_runs <= 0
            or best is None
            or best["score_after"] <= state["baseline_pct"] + threshold
        ):
            return {"confirmation": {}}

        incumbent_scores = []
        challenger_scores = []
        passed = True
        for _ in range(opt.candidate_confirm_runs):
            # 同一任务集、相邻调用形成 paired comparison；协作停止仍只在
            # 轮间 check_stop 生效，不在复核半途打断当轮。
            incumbent = await opt._score_skill(state["current_md"], scenarios, evals)
            challenger = await opt._score_skill(best["new_md"], scenarios, evals)
            incumbent_pct = pct_of(incumbent)
            challenger_pct = pct_of(challenger)
            incumbent_scores.append(incumbent_pct)
            challenger_scores.append(challenger_pct)
            if challenger_pct <= incumbent_pct + threshold:
                passed = False

        # 采用初评分与所有确认评分中的最低值，避免复核本身再次引入乐观偏差。
        confirmed_score = min([best["score_after"], *challenger_scores])
        if confirmed_score <= state["baseline_pct"] + threshold:
            passed = False
        return {
            "confirmation": {
                "candidate_id": best["candidate_id"],
                "passed": passed,
                "runs": opt.candidate_confirm_runs,
                "incumbent_scores": incumbent_scores,
                "challenger_scores": challenger_scores,
                "confirmed_score": confirmed_score,
            }
        }

    # -- 节点：选优 + 保留判定 + 记忆/日志/经验沉淀 + 轮结果事件 -----------------
    async def node_decide(state: OptimizeState) -> dict:
        baseline_pct = state["baseline_pct"]
        candidates = merged_candidates(state)
        analysis = state.get("analysis") or {}

        scored = [c for c in candidates if c["rejected"] is None and c["new_md"] is not None]
        best = max(scored, key=lambda c: c["score_after"], default=None)
        confirmation = state.get("confirmation") or {}
        confirmation_failed = False
        if best is not None and confirmation.get("candidate_id") == best["candidate_id"]:
            confirmation_failed = not confirmation.get("passed", False)
            if not confirmation_failed:
                best["score_after"] = confirmation["confirmed_score"]

        # 【核心不变量】严格提升才保留：score > baseline + threshold + noise_floor。
        kept = (
            best is not None
            and not confirmation_failed
            and best["score_after"] > baseline_pct + opt.improvement_threshold + opt.noise_floor
        )
        new_pct = best["score_after"] if best else baseline_pct
        rnd = state["round_idx"] + 1

        # 【目标 3】总分并列不改变 winner（max 的确定性首胜规则保持不变）；
        # 仅提取未选候选相对 winner 的维度优势，供轮间记忆/经验库学习。
        tie_dimension_records = []
        if opt.tie_dimension_lessons and kept and best is not None:
            for c in scored:
                if c["candidate_id"] == best["candidate_id"]:
                    continue
                if not math.isclose(c["score_after"], best["score_after"], abs_tol=1e-9):
                    continue
                advantages = opt._dimension_advantages(
                    c.get("dimension_scores", {}), best.get("dimension_scores", {})
                )
                if advantages:
                    tie_dimension_records.append({
                        "candidate_id": c["candidate_id"],
                        "strategy": c.get("strategy_type", analysis.get("mutation_strategy", "unknown")),
                        "summary": (c.get("description") or "")[:200],
                        "dimension_gains": advantages,
                    })

        # 每候选一条 mutation_log（增量返回，经 add reducer 追加）。
        log_entries = []
        for c in candidates:
            is_winner = kept and best is not None and c["candidate_id"] == best["candidate_id"]
            confirm_rejected = (
                confirmation_failed and best is not None
                and c["candidate_id"] == best["candidate_id"]
            )
            log_entries.append({
                "round": rnd,
                "candidate_id": c["candidate_id"],
                "strategy_type": c.get("strategy_type", analysis.get("mutation_strategy", "unknown")),
                "diagnosis": analysis.get("diagnosis", ""),
                "description": c["description"],
                "score_before": baseline_pct,
                "score_after": c["score_after"],
                "kept": is_winner,
                "reason": c["rejected"] or (
                    "confirmation_failed" if confirm_rejected
                    else ("best" if is_winner else "not_best")
                ),
            })

        # 每轮一条 round_memory（P2 轮间记忆）。
        if best is not None:
            outcome = "rejected" if best["rejected"] else ("kept" if kept else "discarded")
            mem_reason = best["rejected"] or (
                "confirmation_failed" if confirmation_failed
                else ("best" if kept else "not_best")
            )
        else:
            outcome, mem_reason = "discarded", "no_candidate"
        mem_entry = {
            "round": rnd,
            "strategy": (
                best.get("strategy_type", analysis.get("mutation_strategy", "unknown"))
                if best else analysis.get("mutation_strategy", "unknown")
            ),
            "diagnosis": (analysis.get("diagnosis") or "")[:200],
            "target": analysis.get("target_section", ""),
            "outcome": outcome,
            "score_before": baseline_pct,
            "score_after": new_pct,
            "reason": mem_reason,
        }
        if tie_dimension_records:
            mem_entry["tie_dimension_lessons"] = tie_dimension_records

        # 保留则更新当前版本 + 经验沉淀（只记模型生成内容，安全）。
        current_md = state["current_md"]
        current_details = state["current_details"]
        current_dimension_scores = state.get("current_dimension_scores", {})
        weak_dimensions = state.get("weak_dimensions", [])
        if kept and best is not None:
            # 【质量门槛】提升幅度/最终水位达标才沉淀（_lesson_qualifies）。
            if opt._lesson_qualifies(baseline_pct, best["score_after"]):
                lesson = {
                    "skill_name": opt._skill_name_from_md(state["skill_md"]),
                    "domain": state.get("domain") or "",
                    "skill_description": "",
                    "strategy": best.get(
                        "strategy_type", analysis.get("mutation_strategy", "unknown")
                    ),
                    "diagnosis": (analysis.get("diagnosis") or "")[:200],
                    "summary": (best.get("description") or "")[:200],
                    "score_before": baseline_pct,
                    "score_after": best["score_after"],
                    "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
                }
                # 【目标 5】仅新版经验管线持久化模型产出的目标维度元数据；
                # 不保存技能正文、场景、执行输出或凭据。classic 的记录形状不变。
                if opt.lesson_rag_pipeline == "quality_diverse" and analysis.get("target_dimension"):
                    lesson["target_dimension"] = analysis["target_dimension"]
                opt._append_lesson(opt.lesson_file, lesson)
            # 平局候选只沉淀元数据，不保留其 SKILL.md，也不改变 winner。
            if opt._lesson_qualifies(baseline_pct, best["score_after"]):
                for tie in tie_dimension_records:
                    tie_lesson = {
                        "skill_name": opt._skill_name_from_md(state["skill_md"]),
                        "domain": state.get("domain") or "",
                        "skill_description": "",
                        "strategy": tie["strategy"],
                        "diagnosis": (analysis.get("diagnosis") or "")[:200],
                        "summary": tie["summary"],
                        "score_before": baseline_pct,
                        "score_after": best["score_after"],
                        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
                        "lesson_type": "tie_dimension",
                        "dimension_gains": tie["dimension_gains"],
                    }
                    if opt.lesson_rag_pipeline == "quality_diverse" and tie["dimension_gains"]:
                        tie_lesson["target_dimension"] = next(iter(tie["dimension_gains"]))
                    opt._append_lesson(opt.lesson_file, tie_lesson)
            current_md = best["new_md"]
            baseline_pct = best["score_after"]
            current_details = best.get("details", current_details)
            current_dimension_scores = best.get("dimension_scores", current_dimension_scores)
            weak_dimensions = opt._identify_weak_dimensions(current_dimension_scores)

        no_improve = 0 if kept else state["no_improve"] + 1

        event = {
            "type": "experiment_result",
            "data": {
                "round": rnd,
                "score": new_pct,
                "kept": kept,
                "status": "kept" if kept else "discarded",
                "description": best["description"] if best else "",
                "strategy": (
                    best.get("strategy_type", analysis.get("mutation_strategy", ""))
                    if best else analysis.get("mutation_strategy", "")
                ),
                "per_eval": best["per_eval"] if best else [],
                "candidates": [
                    {
                        "candidate_id": c["candidate_id"],
                        "description": c["description"],
                        "score": c["score_after"],
                        "reason": c["rejected"] or (
                            "confirmation_failed"
                            if (
                                confirmation_failed and best
                                and c["candidate_id"] == best["candidate_id"]
                            )
                            else (
                                "best"
                                if (kept and best and c["candidate_id"] == best["candidate_id"])
                                else "not_best"
                            )
                        ),
                    }
                    for c in candidates
                ],
                "dimension_scores": best["dimension_scores"] if best else {},
                "diff_summary": (
                    opt._diff_summary(current_md, best["new_md"])
                    if kept and best is not None else ""
                ),
            },
        }
        return {
            "current_md": current_md,
            "baseline_pct": baseline_pct,
            "current_details": current_details,
            "current_dimension_scores": current_dimension_scores,
            "weak_dimensions": weak_dimensions,
            "score_history": [baseline_pct],
            "mutation_log": log_entries,
            "round_memory": [mem_entry],
            "no_improve": no_improve,
            "round_idx": state["round_idx"] + 1,
            "kept": kept,
            "final_pct": new_pct,
            "pending_events": [event],
        }

    # -- 节点：final_confirm 复核 + complete 事件 + 最终结果 --------------------
    async def node_finalize(state: OptimizeState) -> dict:
        skill_md = state["skill_md"]
        current_md = state["current_md"]
        baseline_pct = state["baseline_pct"]
        score_history = list(state.get("score_history", []))
        mutation_log = state.get("mutation_log", [])
        # 【P3 胜出候选复核】开启时独立再评一次，不达标回退原始技能。
        if opt.final_confirm and current_md != skill_md:
            confirm = await opt._score_skill(current_md, scenarios, evals)
            confirm_pct = pct_of(confirm)
            if confirm_pct <= score_history[0] + opt.improvement_threshold + opt.noise_floor:
                current_md = skill_md
                baseline_pct = score_history[0]
        final_pct = baseline_pct
        baseline_score = score_history[0] if score_history else 0.0
        event = {
            "type": "complete",
            "data": {
                "baseline_score": baseline_score,
                "final_score": final_pct,
                "improved_skill_md": current_md,
                "score_history": score_history,
                "mutation_log": mutation_log,
                "strategy_stats": opt._strategy_stats(mutation_log),
            },
        }
        final_result = {
            "baseline_score": baseline_score,
            "final_score": final_pct,
            "improved_skill_md": current_md,
            "score_history": score_history,
            "mutation_log": mutation_log,
        }
        return {"pending_events": [event], "final_result": final_result}

    # -- 建图 ----------------------------------------------------------------
    g = StateGraph(OptimizeState)
    g.add_node("baseline", node_baseline)
    g.add_node("prepare_lessons", node_prepare_lessons)
    g.add_node("check_stop", node_check_stop)
    g.add_node("analyst", node_analyst)
    g.add_node("mutate_slot", node_mutate_slot)
    g.add_node("guard", node_guard)
    g.add_node("reeval_slot", node_reeval_slot)
    g.add_node("confirm_candidate", node_confirm_candidate)
    g.add_node("decide", node_decide)
    g.add_node("finalize", node_finalize)

    g.add_edge(START, "baseline")
    g.add_edge("baseline", "prepare_lessons")
    # baseline 后首轮路由：可优化才进轮，否则直接收尾。
    g.add_conditional_edges(
        "prepare_lessons", route_check,
        {"continue": "check_stop", "exit": "finalize"},
    )
    g.add_edge("check_stop", "analyst")
    # analyst 后扇出到 N 个 mutate_slot（Send map-reduce）；guard 是汇聚点。
    g.add_conditional_edges("analyst", route_mutate)
    g.add_edge("mutate_slot", "guard")
    # guard 后扇出复评到 reeval_slot；全被拒则直接 decide。
    g.add_conditional_edges("guard", route_reeval, {"skip": "decide"})
    # reeval_slot 是初评汇聚点；可选配对复核后再进入 decide。
    g.add_edge("reeval_slot", "confirm_candidate")
    g.add_edge("confirm_candidate", "decide")
    # 轮末路由：继续下一轮（check_stop）或收尾（finalize）。
    g.add_conditional_edges(
        "decide", route_check,
        {"continue": "check_stop", "exit": "finalize"},
    )
    g.add_edge("finalize", END)

    # 【阶段 5】可选 checkpointer：compile 时挂载；调用方通过 config 的
    # thread_id 持久化/恢复执行（断点续跑），不设置则与旧行为完全一致。
    return g.compile(checkpointer=checkpointer)
