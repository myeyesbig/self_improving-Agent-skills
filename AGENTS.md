# AGENTS.md

## Scope

These instructions apply to the `self-improving-agent-skills` project and all files below this directory.

This is a Qwen-only application. The supported AI stack is Qwen-Agent with Alibaba Cloud Model Studio (DashScope). Do not add Gemini, Google ADK, OpenAI, Anthropic, Ollama, vLLM, or generic provider abstractions unless the user explicitly expands the scope.

## Architecture

- `backend/app.py` owns the FastAPI routes, in-memory sessions, upload validation, polling/SSE progress, and download packaging.
- `backend/qwen_optimizer.py` owns all Qwen-Agent integration and the Executor → Analyst → Mutator optimization loop.
- `frontend/src/app/page.tsx` owns the four-step page state.
- `frontend/src/components/` owns upload, configuration, running, and results UI behavior.
- `README.md` is the user-facing source of truth for setup and API usage.

Keep provider-specific code inside `backend/qwen_optimizer.py`. Do not spread Qwen SDK calls across API routes.

## Behavioral Invariants

- Preserve the three roles: Executor, Analyst, and Mutator.
- A mutation must make exactly one targeted change to `SKILL.md`.
- Keep a mutation only when its score is strictly higher than the current best score.
- Preserve endpoint paths, response objects, progress events, download layout, upload limits, and session TTL unless the task explicitly changes them.
- The credential request field is `qwen_api_key`. Do not restore or alias `gemini_api_key`.
- The default model is `qwen-plus`; `QWEN_MODEL` may override it on the backend.
- DashScope is the only supported model service.
- Qwen-Agent calls are synchronous and must run through `asyncio.to_thread` when invoked from async FastAPI code.
- Structured Analyst and Mutator outputs must be Pydantic-validated. Keep tolerant JSON extraction and explicit fallbacks.

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
   rg -n "Google ADK|Gemini|gemini_api_key|GOOGLE_API_KEY|google\.adk|google\.genai" .
   ```

4. For changes touching model execution, perform one live DashScope smoke test with `max_rounds=1`.
5. Verify that no API key appears in logs, session responses, generated ZIP files, or git diffs.
