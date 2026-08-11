# AGENTS.md

## Scope

These instructions apply to the `skillforge` project (formerly `self-improving-agent-skills`) and all files below this directory.

This is a Qwen-only application. The supported AI stack is DashScope (Alibaba Cloud Model Studio), accessed directly through the `dashscope` SDK. The optimization loop is orchestrated by LangGraph. Do not add Gemini, Google ADK, OpenAI, Anthropic, Ollama, vLLM, or generic provider abstractions unless the user explicitly expands the scope. LangChain (`langchain`/`langchain-community`) must not be imported in code.

## Architecture

- `backend/app.py` owns the FastAPI routes, in-memory sessions, upload validation, polling/SSE progress, and download packaging.
- `backend/qwen_optimizer.py` owns all DashScope integration, the Executor → Analyst → Mutator algorithm logic (scoring, diagnosis, mutation, regression guard, lesson store), and the LLM-call bridge.
- `backend/llm_client.py` is the thin DashScope wrapper: `asyncio.to_thread` bridging, exponential-backoff retry, timeouts, and JSON mode.
- `backend/optimize_graph.py` owns the LangGraph orchestration: the StateGraph with conditional-edge routing and Send-API parallel fan-out; nodes call `qwen_optimizer.py` methods.
- `frontend/src/app/page.tsx` owns the four-step page state.
- `frontend/src/components/` owns upload, configuration, running, and results UI behavior.
- `frontend/src/lib/api.ts` is the single frontend API client — do not add per-component `API_BASE`/`fetch` URL construction.
- `skill-examples/*.zip` are bundled example skill packs read by `/api/examples`.
- `README.md` is the user-facing source of truth for setup and API usage.

Keep provider-specific code inside `backend/qwen_optimizer.py` and `backend/llm_client.py`. Do not spread DashScope SDK calls across API routes.

## Behavioral Invariants

- Preserve the three roles: Executor, Analyst, and Mutator.
- A mutation must make exactly one targeted change to `SKILL.md` — including parallel mutations, each candidate changes exactly one spot.
- Keep a mutation only when its score is strictly higher than the current best score (optionally plus `improvement_threshold`; default 0.0 preserves strict `>` semantics).
- The regression guard (`REGRESSION_CHECK`) is strictly protective — it can only reject mutations, never accept a non-improving one. Do not weaken it.
- Parallel mutations use the LangGraph Send API with deep-copied per-slot payloads and deterministic ordering at the reduce point (guard/decide); the per-slot `mutator_pool`/`executor_pool` attributes are kept for compatibility but the stateless `llm_client` makes sharing safe.
- Preserve endpoint paths, response objects, progress events, download layout, upload limits, and session TTL unless the task explicitly changes them.
- The credential request field is `qwen_api_key`. Do not restore or alias `gemini_api_key`.
- The default model is `qwen-plus`; `QWEN_MODEL` may override it on the backend.
- DashScope is the only supported model service.
- DashScope SDK calls are synchronous and must run through `asyncio.to_thread` (via `llm_client`) when invoked from async FastAPI code.
- Structured Analyst and Mutator outputs must be Pydantic-validated. Keep tolerant JSON extraction and explicit fallbacks.
- `/api/stop` is a cooperative cancel: it takes effect between rounds (the in-flight model call finishes). Keep this documented behavior. Do not implement it with LangGraph `interrupt`.

## Security

- Never commit, print, log, persist, or return API keys.
- Pass the key directly in the Qwen-Agent LLM configuration; do not assign it to a process-wide environment variable.
- Keep the frontend key only in component memory.
- Preserve upload size, extension, and path-traversal checks.
- Do not include uploaded skill contents or model prompts in production logs.

## Change Discipline

- Prefer the smallest change that satisfies the requested behavior.
- Do not rewrite the optimization algorithm during provider or branding changes.
- Do not add tools, RAG, MCP, databases, queues, authentication, or deployment infrastructure without explicit scope.
  - **Approved exception (2026-08-07, user-explicit)**: the cross-session lesson store (`SKILL_LESSONS_FILE`) may use a lightweight SQLite DB (`.db`/`.sqlite` path suffix, Python built-in `sqlite3`) and retrieval-augmented selection (`LESSON_RETRIEVAL` = `tag`/`semantic`/`hybrid`) backed by DashScope `text-embedding-v3` / `gte-rerank` (Qwen ecosystem only). Hard constraints: default is `off` (unchanged behavior); only model-generated lesson metadata is persisted (never skill content, scenarios, outputs, or API keys); any embedding/rerank failure silently degrades to tag filtering; scope is limited to the lesson store — no other RAG/DB usage is implied.
  - **Approved exception (2026-08-11, user-explicit)**: DeepSeek may be used as an optional alternative model service for the LLM call layer (`backend/llm_client.py`). Selection is explicit and non-generic: model names with the `deepseek-` prefix are routed to the DeepSeek OpenAI-compatible API (`https://api.deepseek.com/chat/completions`), authenticated with the key passed from the frontend (same request path as `qwen_api_key`; never logged, never stored in sessions), falling back to the `DEEPSEEK_API_KEY` environment variable when no key was passed; all other model names continue to use DashScope, which remains the default and primary service. No generic provider-abstraction layer is introduced. All existing behavioral invariants (three roles, single-spot mutation, strict improvement, regression guard, cooperative stop between rounds) remain unchanged.
  - **Approved exception (2026-08-11, user-explicit)**: LangGraph may be adopted as the orchestration layer for the optimization loop (`backend/optimize_graph.py`), and the LLM call layer may be replaced by direct DashScope SDK calls (`backend/llm_client.py`; same SDK family as the lesson-store embedding/rerank). LangChain itself (`langchain`/`langchain-community`) must NOT be used — no ChatTongyi, no LangChain prompts/memory/callbacks; DashScope remains the only model service and `qwen_api_key` remains the only credential field. A lightweight SQLite checkpointer (`langgraph-checkpoint-sqlite`, Python built-in `sqlite3`) may persist optimization-run checkpoints keyed by `thread_id = session_id`, enabled only via `SKILL_CHECKPOINT_FILE` (default off, unchanged behavior). Checkpoints store the optimization graph state (including the skill text being optimized, which is execution-required); they never store prompts, model responses, or API keys, and they may enable resume-after-restart. All existing behavioral invariants (three roles, single-spot mutation, strict improvement, regression guard, cooperative stop between rounds) remain unchanged.
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

4. For changes touching model execution, perform one live DashScope smoke test with `max_rounds=1`.
5. Verify that no API key appears in logs, session responses, generated ZIP files, or git diffs.
