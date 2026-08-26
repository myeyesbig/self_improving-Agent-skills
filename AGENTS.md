# AGENTS.md

## Scope

These instructions apply to the `skillforge` project (formerly `self-improving-agent-skills`) and all files below this directory.

This is a Qwen-primary application. DashScope (Alibaba Cloud Model Studio), accessed directly through the `dashscope` SDK, is the default and primary AI stack. DeepSeek and Zhipu GLM are the only approved optional alternatives and are selected exclusively by the fixed `deepseek-` / `glm-` model-name prefixes described below. The optimization loop is orchestrated by LangGraph. Do not add Gemini, Google ADK, OpenAI as a provider, Anthropic, Ollama, vLLM, or generic provider abstractions unless the user explicitly expands the scope. LangChain (`langchain`/`langchain-community`) must not be imported in code.

## Architecture

- `backend/app.py` owns the FastAPI routes, in-memory sessions, upload validation, polling/SSE progress, and download packaging.
- `backend/qwen_optimizer.py` owns the Executor → Analyst → Mutator algorithm logic: role/model selection, scoring, diagnosis, single-spot mutation, strategy allocation, regression helpers, and the cross-session lesson store/RAG implementation.
- `backend/llm_client.py` is the thin model-call layer. It owns the direct DashScope SDK call, the explicit `deepseek-` HTTP branch, exponential-backoff retry, timeouts, and JSON mode. Async callers bridge its synchronous calls with `asyncio.to_thread`.
- `backend/optimize_graph.py` owns the LangGraph orchestration: baseline, initial/per-round lesson preparation, Analyst, Send-API mutation/evaluation fan-out, protective guard, optional provisional-winner confirmation, strict decide, and finalization nodes.
- `frontend/src/app/page.tsx` owns the four-step page state.
- `frontend/src/components/` owns upload, configuration, running, and results UI behavior.
- `frontend/src/lib/api.ts` is the single frontend API client — do not add per-component `API_BASE`/`fetch` URL construction.
- `skill-examples/*.zip` are bundled example skill packs read by `/api/examples`.
- `README.md` is the user-facing source of truth for setup and API usage.

Keep provider-specific code inside `backend/qwen_optimizer.py` and `backend/llm_client.py`. Do not spread DashScope SDK or DeepSeek/GLM HTTP calls across API routes or graph nodes.

## Behavioral Invariants

- Preserve the three roles: Executor, Analyst, and Mutator.
- The roles may use different models through `EXECUTOR_MODEL`, `ANALYST_MODEL`, and `MUTATOR_MODEL`. Each variable defaults to `QWEN_MODEL`, so unset role overrides preserve the original single-model path. A role using a `deepseek-` model follows only the existing explicit DeepSeek route.
- A mutation must make exactly one targeted change to `SKILL.md` — including parallel mutations, each candidate changes exactly one spot.
- Keep a mutation only when its score is strictly higher than the current best score plus `improvement_threshold` plus `NOISE_FLOOR`. Both additions default to 0.0, preserving strict `>` semantics.
- The regression guard (`REGRESSION_CHECK`) is strictly protective — it can only reject mutations, never accept a non-improving one. Do not weaken it.
- `CANDIDATE_CONFIRM_RUNS` is also strictly protective: when enabled, every incumbent/challenger confirmation pair must clear the same strict threshold, and the accepted score is the lowest challenger observation. Confirmation can only reject an otherwise eligible provisional winner; it can never accept a tie or regression.
- Parallel mutations use the LangGraph Send API with deep-copied per-slot payloads and deterministic ordering at the guard/confirm/decide reduce path; the per-slot `mutator_pool`/`executor_pool` attributes are kept for compatibility but the stateless `llm_client` makes sharing safe.
- `OPTIMIZATION_SEARCH=adaptive` may change which allowed single-spot mutation strategy each slot tries by using deterministic UCB-style exploration/exploitation over observed in-session gains. It must retain one incumbent/parent, must not introduce crossover or a retained population, and must not alter candidate evaluation or acceptance gates. The default is `classic`.
- `TIE_DIMENSION_LESSONS` may extract dimension-gain metadata from a same-round, total-score-tied non-winner relative to that round's selected winner. It must not retain that candidate or its `SKILL.md`, mark it kept, or allow it to become the next-round parent.
- `CROSS_ROUND_TIE_DIMENSION_LESSONS` may extract dimension-gain metadata from any guard-passing, actually scored candidate whose total score ties the round-start incumbent — the most recent version accepted by the strict improvement gate. It must compare against the frozen round-start score/dimensions before any incumbent update, exclude unscored, guard-rejected, edit-limit-rejected, and confirmation-rejected candidates, aggregate a candidate's winning dimensions into one lesson, and never retain or mark the tied candidate as kept. `CROSS_ROUND_TIE_DIMENSION_MIN_GAIN` is an independent strict dimension-gain threshold. Same-round and cross-round records use `comparison_scope=same_round` and `comparison_scope=cross_round`; within a round, the same candidate/dimension signal is stored only once, with same-round extraction taking precedence.
- `WEAK_DIMENSION_FOCUS` may identify measured low-scoring dimensions, reorder only the allowed strategy pool, target Analyst/Mutator prompts, and influence lesson retrieval. A local dimension gain must never compensate for a non-improving total score.
- Lesson retrieval and all lesson metadata are advisory inputs only. Retrieval, rerank, persistence, or lesson-quality decisions must never bypass the regression guard, strict improvement gate, or candidate confirmation gate. `LESSON_MIN_GAIN` does not apply to a cross-round total-score tie; its own strict dimension-gain threshold is the gain-quality gate. When `LESSON_MIN_FINAL` is enabled, it remains an additional persistence floor for cross-round tie lessons. The first lesson preparation remains immediately after baseline; with cross-round learning enabled, each subsequent round refreshes lessons after the cooperative stop check and before Analyst so the prior round's metadata can be reused immediately. Embedding/retrieval failure retains the existing tag fallback, rerank failure preserves the pre-rerank order, and neither may block optimization.
- All optimization and lesson features added on 2026-08-12 are environment-variable gated and default to the pre-change behavior: `OPTIMIZATION_SEARCH=classic`, `TIE_DIMENSION_LESSONS=0`, `CROSS_ROUND_TIE_DIMENSION_LESSONS=0`, `CROSS_ROUND_TIE_DIMENSION_MIN_GAIN=0.0`, `WEAK_DIMENSION_FOCUS=0`, `LESSON_RAG_PIPELINE=classic`, `LESSON_FAILURE_CONTEXT=0`, `LESSON_EMBED_BATCH_SIZE=1`, and `CANDIDATE_CONFIRM_RUNS=0`.
- Preserve endpoint paths, response objects, progress events, download layout, upload limits, and session TTL unless the task explicitly changes them.
- The required primary credential request field is `qwen_api_key`. The existing optional `deepseek_api_key` field is allowed only for explicitly routed `deepseek-` models; it must not replace or alias the primary field. Do not restore or alias `gemini_api_key`.
- The default model is `qwen-plus`; `QWEN_MODEL` may override it on the backend, and role-specific variables may override it per role.
- DashScope is the default and primary model service. DeepSeek and Zhipu GLM are supported only through the fixed-prefix exceptions; no runtime provider registry or general OpenAI-compatible provider layer is allowed.
- DashScope SDK calls and DeepSeek/GLM HTTP calls are synchronous and must run through `asyncio.to_thread` (via `llm_client`) when invoked from async FastAPI code.
- Structured Analyst and Mutator outputs must be Pydantic-validated. Keep tolerant JSON extraction and explicit fallbacks.
- `/api/stop` is a cooperative cancel: it takes effect between rounds (the in-flight model call finishes). Keep this documented behavior. Do not implement it with LangGraph `interrupt`.

## Security

- Never commit, print, log, persist, or return API keys.
- Pass request keys directly into `LLMClient` and from there into the DashScope call configuration or DeepSeek/GLM authorization header; do not copy request keys into process-wide environment variables.
- Keep both frontend key values only in component memory. Do not place either key in sessions, checkpoints, lesson stores, progress events, downloads, prompts, or model-generated metadata.
- Preserve upload size, extension, and path-traversal checks.
- Do not include uploaded skill contents or model prompts in production logs.

## Change Discipline

- Prefer the smallest change that satisfies the requested behavior.
- Do not rewrite the optimization algorithm during provider or branding changes. The algorithm-evolution exception below is limited to its named mechanisms and does not authorize unrelated rewrites.
- Do not add tools, RAG, MCP, databases, queues, authentication, or deployment infrastructure without explicit scope.
  - **Approved exception (2026-08-07, user-explicit)**: the cross-session lesson store (`SKILL_LESSONS_FILE`) may use JSONL or a lightweight SQLite DB (`.db`/`.sqlite` suffix, Python built-in `sqlite3`) and retrieval-augmented selection (`LESSON_RETRIEVAL` = `off`/`tag`/`semantic`/`hybrid`) backed by DashScope `text-embedding-v3` and optional `gte-rerank`. `LESSON_RAG_PIPELINE=quality_diverse` may combine semantic, bilingual sparse, skill, domain, weak-dimension, historical-quality, and recency signals, followed by near-duplicate removal and MMR-style selection. `LESSON_FAILURE_CONTEXT` may add the current failed eval/scenario context to the transient retrieval query only, and `LESSON_EMBED_BATCH_SIZE` may batch at most 10 texts per official API limits. Hard constraints: all enhanced paths default off/classic; only model-generated lesson metadata and embeddings may persist; skill content, scenarios, outputs, prompts, failure-query context, candidate IDs, and API keys must never enter the lesson store; internal IDs/embeddings must not enter prompts; any embedding/retrieval failure silently degrades to tag filtering and rerank failure preserves the pre-rerank order. Tie-dimension records may persist `comparison_scope`, `target_dimension`, and aggregate `dimension_gains`; SQLite schema changes must use built-in `sqlite3`, migrate existing databases compatibly, and interpret legacy tie lessons without `comparison_scope` as `same_round`. This exception implies no other RAG or database usage.
  - **Approved exception (2026-08-11, user-explicit)**: DeepSeek may be used as an optional alternative model service for `backend/llm_client.py`. Selection is explicit and non-generic: only model names with the `deepseek-` prefix route to `https://api.deepseek.com/chat/completions`. Authentication priority for the DeepSeek branch is the optional request-body `deepseek_api_key`, then the required `qwen_api_key` for backward compatibility, then `DEEPSEEK_API_KEY`. Request keys are never logged or stored in sessions. Generation and lesson-store retrieval keys are separate: the lesson store's DashScope embedding/rerank calls prefer `DASHSCOPE_API_KEY` over the caller key (so DeepSeek/GLM generation runs keep full hybrid RAG). All other model names use DashScope. No generic provider abstraction or additional OpenAI-compatible provider is authorized.
  - **Approved exception (2026-08-17, user-explicit)**: Zhipu GLM may be used as an optional low-cost alternative model service for `backend/llm_client.py`, following the exact DeepSeek precedent. Selection is explicit and non-generic: only model names with the `glm-` prefix route to `https://open.bigmodel.cn/api/paas/v4/chat/completions` (Zhipu OpenAI-compatible endpoint). Authentication priority for the GLM branch is `ZHIPU_API_KEY` (CLI/benchmark runs that also use a DashScope key for the lesson-store embeddings/rerank keep generation and embedding keys separate), then the caller-passed `api_key`. Request keys are never logged or stored in sessions. All other model names use DashScope (or the `deepseek-` exception). No generic provider abstraction or additional OpenAI-compatible provider is authorized beyond these two fixed-prefix exceptions.
  - **Approved exception (2026-08-11, user-explicit)**: LangGraph is the orchestration layer for the optimization loop (`backend/optimize_graph.py`), and the LLM layer uses direct DashScope SDK calls plus the fixed DeepSeek exception (`backend/llm_client.py`). LangChain itself (`langchain`/`langchain-community`) must NOT be used — no ChatTongyi, LangChain prompts, memory, or callbacks. A lightweight SQLite checkpointer (`langgraph-checkpoint-sqlite`, Python built-in `sqlite3`) may persist optimization graph state keyed by `thread_id = session_id`, enabled only by `SKILL_CHECKPOINT_FILE` (default off). Checkpoints may contain execution-required skill text but never prompts, model responses, or API keys.
  - **Approved exception (2026-08-12, user-explicit)**: the optimizer may evolve only through the following bounded mechanisms: independent role-model selection; `OPTIMIZATION_SEARCH=adaptive` strategy allocation over the existing mutation-strategy whitelist; same-round and cross-round tied-candidate dimension metadata; weak-dimension targeting; the lesson RAG enhancements above, including per-round refresh while cross-round tie learning is enabled; and provisional-winner paired confirmation. Every mechanism must be environment-variable gated with the old behavior as default. None may change the three roles, retain more than one parent, permit crossover, make more than one targeted edit per candidate, accept a total-score tie/regression, weaken `REGRESSION_CHECK`, or change cooperative stop semantics.
- Keep the existing English UI and documentation unless localization is requested.
- `next/font/google` is a font import and is allowed; Google AI SDKs and Gemini branding are not.
- Update README examples whenever an API field, model default, command, or environment variable changes.

## Commands

From the project root:

```bash
python -m venv backend/.venv
source backend/.venv/bin/activate
pip install -r backend/requirements.txt
(cd backend && python -m unittest discover -p 'test_*.py')
(cd backend && python -c "from qwen_optimizer import SkillOptimizer; print('OK')")
(cd frontend && npm install)
(cd frontend && npm run build)
```

Run the backend with:

```bash
(cd backend && python app.py)
```

Run the frontend with:

```bash
(cd frontend && npm run dev)
```

## Required Verification

Before handing off a change:

1. Run backend unit tests and the optimizer import smoke test.
2. Run the production frontend build.
3. Confirm this scan has no matches:

   ```bash
   rg -n "Google ADK|Gemini|gemini_api_key|GOOGLE_API_KEY|google\.adk|google\.genai|qwen_agent|langchain" --glob '*.py' .
   ```

   (The `langgraph` / `langchain-core` entries in requirements are install-time
   transitive dependencies; Python code must never import the `langchain`
   namespace. The `qwen_agent` keyword is allowed in comments/history docs only.)

4. For changes touching model execution, perform a live `max_rounds=1` smoke test against every affected route. DashScope/default-route changes require a DashScope smoke; `deepseek-` branch changes additionally require a DeepSeek smoke. One provider's result must not be presented as verification of the other; if an affected credential is unavailable, report that route as unverified.
5. Verify that no API key appears in logs, session responses, generated ZIP files, or git diffs.
