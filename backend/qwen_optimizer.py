"""Multi-Agent Skill Optimizer using Qwen-Agent and Alibaba Cloud Model Studio (DashScope).

3 Qwen-Agent assistants work together to improve agent skills:
  Executor: runs the skill against test scenarios, scores outputs, analyzes skills
  Analyst: diagnoses why evals failed, picks a mutation strategy
  Mutator: makes one targeted fix per round
"""

import asyncio
import json
import os
from typing import Callable, List, Optional

from pydantic import BaseModel, Field

from qwen_agent.agents import Assistant


# -- Pydantic schemas for structured agent output ----------------------------


class FailureAnalysis(BaseModel):
    diagnosis: str = Field(description="Root cause of failures")
    mutation_strategy: str = Field(
        description="One of: add_example, add_constraint, restructure, add_edge_case"
    )
    target_section: str = Field(description="Which part of the skill to change")
    suggested_change: str = Field(description="What specific change to make")


class SkillMutation(BaseModel):
    description: str = Field(description="Short description of the change made")
    reasoning: str = Field(description="Why this change should help")
    new_skill_md: str = Field(description="The full updated SKILL.md content")


# -- Text extraction helper ---------------------------------------------------


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


class SkillOptimizer:
    def __init__(self, api_key: str, model: Optional[str] = None):
        # The key is passed directly into the LLM configuration; it is never
        # written to the process environment, stored in sessions, or logged.
        self.model = model or os.getenv("QWEN_MODEL", "qwen-plus")
        self._call_id = 0

        llm_config = {
            "model": self.model,
            "model_type": "qwen_dashscope",
            "api_key": api_key,
            "generate_cfg": {
                "enable_thinking": False,
                "temperature": 0.2,
            },
        }

        self.executor = Assistant(
            name="executor",
            system_message=(
                "You are a versatile skill execution agent. You have three modes:\n\n"
                "1. EXECUTE MODE: Given a skill's instructions and a user request, "
                "produce the output that skill would generate. Follow the instructions "
                "exactly. No meta-commentary.\n\n"
                "2. ANALYZE MODE: Given a skill definition, generate test scenarios "
                "and evaluation criteria. Return valid JSON.\n\n"
                "3. SCORE MODE: Given an output and evaluation criteria, score the "
                "output against each criterion. Return valid JSON."
            ),
            llm=llm_config,
        )
        self.analyst = Assistant(
            name="analyst",
            system_message=(
                "You diagnose why agent skill evaluations fail. "
                "Given failed eval results, identify the root cause and suggest "
                "a specific fix. Pick one mutation_strategy from: "
                "add_example, add_constraint, restructure, or add_edge_case."
            ),
            llm=llm_config,
        )
        self.mutator = Assistant(
            name="mutator",
            system_message=(
                "You edit agent skill files. Given a SKILL.md and a diagnosis, "
                "make exactly ONE targeted change. Keep the YAML frontmatter and "
                "overall structure intact. Return the complete updated SKILL.md."
            ),
            llm=llm_config,
        )

    # -- Agent runner helpers ------------------------------------------------

    def _run_agent_sync(self, agent: Assistant, prompt: str) -> str:
        """Run a Qwen-Agent assistant synchronously, returning the final assistant text.

        ``Assistant.run()`` is a synchronous generator; the final batch of messages
        holds the complete answer. Only the ``content`` of the last assistant turn
        is kept (``reasoning_content`` is ignored).
        """
        responses = None
        for responses in agent.run(
            [{"role": "user", "content": prompt}],
            stream=False,
        ):
            pass
        if not responses:
            raise RuntimeError("Qwen-Agent returned no response")
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
        return await asyncio.to_thread(self._run_agent_sync, agent, prompt)

    async def _ask_json(self, agent: Assistant, prompt: str, fallback=None, schema=None):
        """Run a Qwen-Agent assistant and parse the JSON response.

        When a Pydantic ``schema`` is provided, it is appended to the prompt and
        the parsed payload is validated with ``model_validate()``. Tolerant JSON
        extraction is kept, and both JSON and schema-validation failures fall
        back to ``fallback`` when one is provided.
        """
        if schema is not None:
            prompt = (
                f"{prompt}\n\n"
                f"Return ONLY valid JSON that conforms exactly to this JSON schema:\n"
                f"{json.dumps(schema.model_json_schema(), indent=2)}\n"
                f"Do not wrap it in markdown code fences and do not add any text "
                f"before or after the JSON."
            )
        text = await self._ask(agent, prompt)

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

    async def analyze_skill(self, skill_files: dict) -> dict:
        """Generate test scenarios and eval criteria from skill files."""
        skill_md = next(
            (v for k, v in skill_files.items() if k.endswith("SKILL.md")), ""
        )
        refs = {k: v for k, v in skill_files.items() if "references/" in k}
        ref_text = ""
        if refs:
            ref_text = "\n\nReference files:\n" + "\n---\n".join(
                f"## {k}\n{v}" for k, v in refs.items()
            )

        prompt = (
            f"Analyze this agent skill and generate test scenarios "
            f"with evaluation criteria.\n\n"
            f"# SKILL.md\n{skill_md}\n{ref_text}\n\n"
            f"Generate:\n"
            f"1. 3-4 diverse test scenarios (realistic user inputs)\n"
            f"2. 4-6 binary yes/no evaluation criteria\n\n"
            f"Return JSON:\n"
            f'{{"scenarios": [{{"id": 1, "name": "short name", "description": "short name", '
            f'"input": "the user request to test"}}], '
            f'"evals": [{{"id": 1, "name": "what to check", "criterion": "what to check", '
            f'"question": "yes/no question about the output", '
            f'"pass_condition": "what yes looks like", "fail_condition": "what no looks like"}}]}}'
        )
        return await self._ask_json(self.executor, prompt)

    async def optimize(
        self,
        skill_files: dict,
        scenarios: list,
        evals: list,
        max_rounds: int = 5,
        callback: Optional[Callable] = None,
    ) -> dict:
        """Run the optimization loop with 3 Qwen-Agent assistants."""

        async def emit(event):
            if callback:
                await callback(event)

        skill_md = next(
            (v for k, v in skill_files.items() if k.endswith("SKILL.md")), ""
        )
        current_md = skill_md
        score_history = []
        mutation_log = []

        # -- Baseline ---------------------------------------------------------
        baseline = await self._score_skill(current_md, scenarios, evals)
        baseline_pct = round(100 * baseline["passed"] / max(baseline["total"], 1), 1)
        score_history.append(baseline_pct)

        await emit({
            "type": "baseline",
            "data": {
                "score": baseline_pct,
                "passed": baseline["passed"],
                "total": baseline["total"],
                "per_eval": baseline["per_eval"],
            },
        })

        # -- Rounds -----------------------------------------------------------
        for rnd in range(1, max_rounds + 1):
            await emit({"type": "experiment_start", "data": {"round": rnd}})

            # Analyst diagnoses worst failure
            analysis = await self._analyze_failures(
                current_md, scenarios, evals, baseline["details"]
            )

            # Mutator applies fix
            mutation = await self._mutate_skill(current_md, analysis)
            new_md = mutation.get("new_skill_md", current_md)

            # Re-score
            result = await self._score_skill(new_md, scenarios, evals)
            new_pct = round(100 * result["passed"] / max(result["total"], 1), 1)

            kept = new_pct > baseline_pct
            entry = {
                "round": rnd,
                "strategy_type": analysis.get("mutation_strategy", "unknown"),
                "diagnosis": analysis.get("diagnosis", ""),
                "description": mutation.get("description", ""),
                "score_before": baseline_pct,
                "score_after": new_pct,
                "kept": kept,
            }
            mutation_log.append(entry)

            if kept:
                current_md = new_md
                baseline = result
                baseline_pct = new_pct

            score_history.append(baseline_pct)

            await emit({
                "type": "experiment_result",
                "data": {
                    "round": rnd,
                    "score": new_pct,
                    "kept": kept,
                    "status": "kept" if kept else "discarded",
                    "description": mutation.get("description", ""),
                    "strategy": analysis.get("mutation_strategy", ""),
                    "per_eval": result["per_eval"],
                },
            })

        # -- Done -------------------------------------------------------------
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

    # -- Internal helpers -----------------------------------------------------

    async def _score_skill(self, skill_md, scenarios, evals):
        """Executor runs all scenarios, then scores outputs."""
        all_results = []
        total_passed = 0
        total_checks = 0
        per_eval = {e["id"]: {"passed": 0, "total": 0} for e in evals}

        for sc in scenarios:
            # Executor runs the skill (free-form text)
            output = await self._ask(
                self.executor,
                f"Execute this skill:\n\n{skill_md}\n\nUser request:\n{sc['input']}",
            )
            # Executor scores the output (JSON)
            scoring = await self._ask_json(
                self.executor,
                (
                    f"Evaluate this output against the criteria.\n\n"
                    f"Input: {sc['input']}\n\n"
                    f"Output: {output}\n\n"
                    f"Criteria:\n{json.dumps(evals, indent=2)}\n\n"
                    f"Return JSON: {{\"results\": [{{\"eval_id\": 1, \"passed\": true, \"reason\": \"...\"}}]}}"
                ),
                fallback={"results": []},
            )
            scores = scoring.get("results", []) if isinstance(scoring, dict) else scoring

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
                all_results.append({**s, "scenario_id": sc["id"]})

        return {
            "passed": total_passed,
            "total": total_checks,
            "per_eval": [
                {"eval_id": k, **v, "pass_rate": round(v["passed"] / max(v["total"], 1) * 100, 1)}
                for k, v in per_eval.items()
            ],
            "details": all_results,
        }

    async def _analyze_failures(self, skill_md, scenarios, evals, details):
        """Analyst agent diagnoses the worst failures."""
        failed = [d for d in details if not d.get("passed")]
        if not failed:
            return {
                "diagnosis": "All passed",
                "mutation_strategy": "add_constraint",
                "target_section": "N/A",
                "suggested_change": "none",
            }

        return await self._ask_json(
            self.analyst,
            (
                f"Diagnose these failures and suggest ONE fix.\n\n"
                f"Skill:\n{skill_md[:2000]}\n\n"
                f"Failed evals:\n{json.dumps(failed[:5], indent=2)}"
            ),
            fallback={
                "diagnosis": "Unable to diagnose",
                "mutation_strategy": "add_constraint",
                "target_section": "unknown",
                "suggested_change": "unclear",
            },
            schema=FailureAnalysis,
        )

    async def _mutate_skill(self, skill_md, analysis):
        """Mutator agent makes one targeted change."""
        return await self._ask_json(
            self.mutator,
            (
                f"Apply this fix to the skill. Make ONE change only.\n\n"
                f"SKILL.md:\n{skill_md}\n\n"
                f"Diagnosis: {analysis.get('diagnosis')}\n"
                f"Strategy: {analysis.get('mutation_strategy')}\n"
                f"Target: {analysis.get('target_section')}\n"
                f"Change: {analysis.get('suggested_change')}"
            ),
            fallback={
                "description": "Failed to parse mutation",
                "reasoning": "",
                "new_skill_md": skill_md,
            },
            schema=SkillMutation,
        )

    @staticmethod
    def _strategy_stats(mutation_log):
        stats = {}
        for m in mutation_log:
            s = m.get("strategy_type", "unknown")
            if s not in stats:
                stats[s] = {"total": 0, "kept": 0}
            stats[s]["total"] += 1
            if m.get("kept"):
                stats[s]["kept"] += 1
        return stats
