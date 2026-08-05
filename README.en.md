# 🔥 SkillForge — Forge Better Agent Skills

**SkillForge** automatically optimizes your agent skills using a multi-agent system built with **Qwen-Agent** and **Qwen via Alibaba Cloud Model Studio (DashScope)**. Upload a skill, let the agents generate test scenarios and evaluation criteria, then watch as three specialized Qwen-Agent assistants collaborate to improve your skill through iterative optimization — every mutation is hammered on the anvil, scored, and kept only when it is genuinely stronger.

> A personal project inspired by Karpathy's autoresearch methodology: instead of hand-tuning prompts, define success criteria and let the AI improve itself.


## How It Works

This app implements an automated skill improvement loop inspired by Karpathy's autoresearch methodology, powered by a team of Qwen-Agent assistants:

1. **Upload**: Drop in your skill folder (following [agentskills.io](https://agentskills.io) spec)
2. **Configure**: The Executor agent generates test scenarios and evaluation criteria. Edit, add, or regenerate as needed
3. **Optimize**: Three Qwen-Agent assistants collaborate — one executes and scores, one diagnoses failures, one applies fixes
4. **Results**: Download your improved skill with a detailed changelog

### The Qwen-Agent Team

| Agent | Role | What It Does |
|-------|------|-------------|
| **Executor** | Skill Runner & Scorer | Executes the skill against test scenarios, scores outputs against evaluation criteria, and generates initial test scenarios during analysis |
| **Analyst** | Failure Diagnostician | Examines failed evaluations, identifies root causes, and recommends a mutation strategy from a configurable strategy pool. Uses Pydantic validation for guaranteed structured JSON |
| **Mutator** | Prompt Editor | Makes exactly ONE targeted change to the skill prompt based on the analyst's diagnosis. Uses Pydantic validation for guaranteed structured JSON |

### The Optimization Loop

- The **Executor** agent runs the skill against all test scenarios
- The **Executor** then scores each output against binary yes/no evaluation criteria (optionally dimension-weighted)
- The **Analyst** agent diagnoses failure patterns and picks a strategy from the configurable pool (`add_example`, `add_constraint`, `restructure`, `add_edge_case`, `add_reference`, `rewrite_section`, `fix_format`)
- The **Mutator** agent applies ONE surgical fix to the skill prompt; with parallel mutations enabled, multiple candidates are generated per round
- A **regression guard** checks structural integrity (frontmatter / headings / size) before re-scoring, skipping rejected candidates
- The **Executor** re-runs and re-scores the modified skill (best candidate wins when parallel)
- Changes are kept only when strictly above "baseline + improvement threshold", reverted otherwise
- Repeats until the target pass rate is reached or max rounds hit

## Architecture

```
skillforge/
├── backend/                 # FastAPI server + Qwen-Agent optimization engine
│   ├── app.py              # REST API endpoints + SSE streaming
│   ├── qwen_optimizer.py   # Multi-agent optimizer (Executor, Analyst, Mutator)
│   └── requirements.txt
├── frontend/               # Next.js + React + Tailwind
│   ├── src/
│   │   ├── app/            # Main page + layout + icon.svg (brand)
│   │   ├── components/     # Upload, Config, Running, Results steps + Logo
│   │   └── lib/api.ts      # Single frontend API client
│   ├── package.json
│   └── *.config.ts
├── skill-examples/         # Bundled example skill packs (.zip, read by /api/examples)
└── README.md
```

## Tech Stack

- **Backend**: Python 3.10+, FastAPI, Qwen-Agent, Pydantic
- **Frontend**: Next.js 15, React 19, Tailwind CSS v4, Recharts
- **AI**: Qwen-Agent multi-agent system with Qwen via DashScope (`qwen-plus`) — structured output via Pydantic validation on Analyst and Mutator agents
- **Real-time**: Server-Sent Events (SSE) for live optimization progress

## Quick Start

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run server
python app.py
# Server runs on http://localhost:8891
```

The backend uses `qwen-plus` by default. You may override it with the optional `QWEN_MODEL` environment variable, e.g. `QWEN_MODEL=qwen-max python app.py`.

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
# App runs on http://localhost:3000
```

### Usage

1. Get a DashScope API key from [Alibaba Cloud Model Studio (百炼)](https://bailian.console.aliyun.com/)
2. Open http://localhost:3000
3. Upload a skill folder as a .zip file (or try an example)
4. Enter your DashScope API key
5. Review and edit the generated test scenarios and evaluation criteria
6. Click "Start Optimization" and watch the agents collaborate to improve your skill
7. Download your improved skill when complete

## Skill Format

Skills follow the [agentskills.io](https://agentskills.io) specification:

```
my-skill/
├── SKILL.md           # Required: YAML frontmatter + instructions
├── scripts/           # Optional: executable code
├── references/        # Optional: additional docs
└── assets/            # Optional: templates, resources
```

Example SKILL.md:

```markdown
---
name: my-skill
description: What this skill does and when to use it
license: MIT
metadata:
  author: your-name
  version: "1.0"
---

# My Skill

Your skill instructions here...
```

## Trying it

Zip any skill folder and upload it. The repo ships with 4 bundled example
skill packs (`skill-examples/*.zip`), and the app's "examples" picker reads
them automatically — real skills, not toys:

```bash
# Bundled examples (one-click in the app, no manual zip needed):
# project-graveyard / thinking-out-loud / dependency-doctor / commit-archaeologist

# Zip your own skill:
zip -r my-skill.zip my-skill/
```

## How the Multi-Agent Optimization Works

### 1. Analysis Phase
The **Executor** agent analyzes your skill and generates:
- 3-4 diverse test scenarios
- 4-6 binary evaluation criteria (yes/no questions)

You can edit, add, or remove scenarios and criteria before optimization begins.

### 2. Baseline Run
The **Executor** agent runs the skill against all scenarios and scores each output against all evaluation criteria. This establishes the starting score.

### 3. Optimization Loop
For each round, the three agents collaborate:
1. **Executor** runs the skill against all test scenarios and scores the outputs (optionally dimension-weighted)
2. **Analyst** examines failures, identifies root cause, and selects a mutation strategy from the configurable pool (returns structured JSON validated against a Pydantic schema)
3. **Mutator** applies ONE specific change to improve the skill (returns structured JSON validated against a Pydantic schema); with parallel mutations enabled, multiple candidates are generated per round
4. **Regression guard** checks structural integrity (frontmatter / headings / size) for each candidate, rejecting and skipping re-scoring on failure
5. **Executor** re-runs and re-scores the modified skill (best candidate wins when parallel)
6. Score is compared — keep only if strictly above "baseline + improvement threshold", revert otherwise
7. Repeat until target pass rate or max rounds reached

### 4. Output
- Improved SKILL.md with all successful changes applied
- Detailed changelog of what changed and why (strategy used, parallel candidate details)
- Performance comparison (baseline vs final, with per-dimension score trends)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload skill zip file (max 10MB, text files only) |
| `POST` | `/api/upload-files` | Upload multiple files (folder upload) |
| `POST` | `/api/analyze` | Generate scenarios and evals (requires DashScope API key) |
| `POST` | `/api/regenerate` | Regenerate scenarios and evals |
| `POST` | `/api/update-config` | Save user's selected/edited config |
| `POST` | `/api/start/{session_id}` | Start optimization |
| `GET` | `/api/stream/{session_id}` | SSE stream of optimization progress |
| `POST` | `/api/stop/{session_id}` | Stop optimization |
| `GET` | `/api/download/{session_id}` | Download improved skill |
| `GET` | `/api/examples` | List available example skills |
| `POST` | `/api/examples/{name}/load` | Load an example skill |
| `GET` | `/api/status/{session_id}` | Poll-based status endpoint |
| `GET` | `/health` | Health check |

## Configuration

### Backend

The DashScope API key is passed from the frontend with each request (field `qwen_api_key`) and used directly in the Qwen-Agent model configuration — it is never written to a process-wide environment variable, stored in sessions, or logged. The model defaults to `qwen-plus` and can be overridden with the optional `QWEN_MODEL` environment variable. Server runs on port **8891**.

Upload limits:
- **10MB** max total upload size
- **1MB** max per file
- **50** max files per upload
- Text files only (`.md`, `.txt`, `.json`, `.yaml`, `.py`, `.js`, `.ts`, etc.)

Sessions expire after **1 hour** automatically.

### Frontend

API key is entered in the UI, stored in component state (not persisted), and sent with each request as `qwen_api_key`.

### Optimization Parameters

SkillForge exposes a set of **optional** optimization knobs, all defaulting to classic behavior (not passing them behaves exactly like the base version):

| Knob | Where | Default | Description |
|------|-------|---------|-------------|
| `max_rounds` | request field | 20 (≤50) | Max optimization rounds |
| `parallel_mutations` | request field | backend `MUTATION_PARALLELISM` (default 1, max 3) | How many candidate mutations per round |
| `strategy_pool` | request field | all strategy templates | Whitelist of mutation strategies for the Analyst |
| `improvement_threshold` | request field / `IMPROVEMENT_THRESHOLD` | 0.0 | New score must be strictly above "baseline + threshold" to keep |
| Dimension weights | `ANALYST_DIMENSION_WEIGHTS` (JSON) | none (equals old pass-rate) | Weight scoring by eval `dimension` field |
| `REGRESSION_CHECK` | env var | 1 (on) | Structural integrity guard before re-scoring; 0 disables |

> **Security & invariants**: the credential field is always `qwen_api_key`; the key lives only in component memory and the request body — never in logs, sessions, zips, or git. A mutation still changes exactly one spot in SKILL.md; only strictly-improving mutations are kept, and the regression guard is strictly protective — it can never let a skill degrade.

In `qwen_optimizer.py`, adjust the model:

```python
def __init__(self, api_key: str, model: Optional[str] = None):  # defaults to QWEN_MODEL or "qwen-plus"
```

## Development

### Backend Tests

```bash
cd backend
python -m unittest discover -p 'test_*.py'
python -c "from qwen_optimizer import SkillOptimizer; print('OK')"
```

### Frontend Build

```bash
cd frontend
npm run build
```

### Live Development

Both servers support hot reload. Edit code and see changes immediately.

## Based on Karpathy's Autoresearch

This tool applies Andrej Karpathy's autoresearch methodology (using LLMs to iteratively improve their own prompts) to agent skills. The key insight: rather than manually tweaking prompts, define success criteria and let the AI optimize itself — now powered by a team of specialized Qwen-Agent assistants.

Original concept: [https://github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch)
