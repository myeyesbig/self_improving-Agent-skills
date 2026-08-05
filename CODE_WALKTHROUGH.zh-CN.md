# CODE_WALKTHROUGH（中文学习导览）

> 面向零基础读者的全项目导读。本文件只解释"代码在做什么、为什么这样做、数据接下来去哪"，不修改任何代码。
> 配套说明：源码中的中文注释统一使用少量标签 —— 【主流程】【初学者提示】【异步】【安全】【注意】。
> 本应用是 Qwen-only 应用：技术栈为 Qwen-Agent + 阿里云百炼 DashScope，凭据字段为 `qwen_api_key`。

---

## 一、推荐阅读顺序

按"数据流动"的顺序读代码，比按目录顺序读更容易理解：

1. **前端四步状态机** — `frontend/src/app/page.tsx`（先建立整体框架：上传 → 配置 → 优化 → 结果）
2. **上传路由** — `backend/app.py` 的 `/api/upload` 与 `/api/upload-files`（session 从哪来）
3. **内存 session** — `backend/app.py` 的 `create_session_from_files()`（session 里存了什么）
4. **Qwen 分析** — `backend/qwen_optimizer.py` 的 `analyze_skill()`（生成测试场景与评估标准）
5. **优化循环** — `backend/qwen_optimizer.py` 的 `optimize()`（三 Agent 协作的核心）
6. **回调 / 轮询** — `backend/app.py` 的 `run_optimization()` 闭包 callback 与 `/api/status`，配合 `frontend/src/components/RunningStep.tsx`
7. **结果下载** — `frontend/src/components/ResultsStep.tsx` 与 `/api/download/{session_id}`
8. **单元测试** — `backend/test_qwen_optimizer.py`（理解 fake agent 与 mock 的测试思路）

---

## 二、端到端数据流图

```
┌────────────┐   ①上传 zip/文件夹   ┌────────────────────────────────────────┐
│  浏览器     │ ────────────────────▶ │             FastAPI (backend/app.py)       │
│ (Next.js)  │                      │                                        │
│            │   ②返回 session_id    │   /api/upload → create_session_from_files │
│ page.tsx   │ ◀──────────────────── │        │                              │
│            │                      │        ▼                              │
│ UploadStep │   ③POST /api/analyze  │   内存 sessions 字典                    │
│ ConfigStep │ ────────────────────▶ │   { session_id: { skill_files,       │
│ RunningStep│                      │       scenarios, evals, experiments,  │
│ ResultsStep│   ④返回 scenarios/     │       final_result, ... } }           │
│            │     evals            │        ▲                              │
│            │ ◀──────────────────── │        │ 读取/写入                     │
│            │                      │   ┌────┴────┐                         │
│            │   ⑤POST /api/start    │   │SkillOptimizer                     │
│            │ ────────────────────▶ │   │(qwen_optimizer.py)                │
│            │                      │   │  Executor ◀──────┐                │
│            │   ⑥每3秒 GET /api/    │   │  Analyst  ◀──────┤ 同步调用经       │
│            │     status (轮询)     │   │  Mutator  ◀──────┤ asyncio.to_thread│
│            │ ◀──────────────────── │   │                 │ 桥接进工作线程    │
│            │                      │   └────────┬────────┘                 │
│            │   ⑦GET /api/download  │            │ callback 事件             │
│            │ ────────────────────▶ │            ▼                          │
│            │   ⑧下载 improved_skill│   session.experiments / final_result   │
└────────────┘     .zip             └────────────────────────────────────────┘
       │
       ▼
 Qwen-Agent + DashScope（三个 Assistant：Executor / Analyst / Mutator）
```

数据方向速记：**浏览器 ↔ FastAPI（JSON/文件）↔ 内存 session ↔ 优化器 ↔ Qwen 模型**。

---

## 三、三张速查表

### 表 1：核心文件职责

| 文件 | 职责 | 建议先看 |
| --- | --- | --- |
| `backend/app.py` | FastAPI 路由、内存 session、上传校验、后台任务、进度存储、下载打包 | `/api/upload` → `/api/start` → `/api/status` |
| `backend/qwen_optimizer.py` | 三 Agent 优化算法、LLM 配置、JSON 宽容解析、Pydantic 校验与 fallback | `optimize()` 与 `_score_skill()` |
| `backend/test_qwen_optimizer.py` | 单元测试（fake agent + mock，无需真实 Key） | `FakeAgent` 与 `TestApiKeyHandling` |
| `frontend/src/lib/api.ts` | 前端与后端通信的唯一出口：集中 API_BASE 与全部端点函数 | `uploadZip()` / `startOptimization()` |
| `frontend/src/app/page.tsx` | 四步页面状态机、共享状态、回调推进 | `handleUploadComplete` → `handleOptimizationComplete` |
| `frontend/src/components/UploadStep.tsx` | 上传、API Key 输入、分析触发 | `uploadZip()` 与 `handleAnalyze()` |
| `frontend/src/components/ConfigStep.tsx` | 测试场景 / 评估标准的编辑与保存 | `handleContinue()` |
| `frontend/src/components/RunningStep.tsx` | 启动优化、3 秒轮询、进度展示 | `startOptimization()` 里的轮询循环 |
| `frontend/src/components/ResultsStep.tsx` | 结果展示、文本 diff、下载 | `handleDownload()` 与 diff 计算 |
| `frontend/src/components/StepIndicator.tsx` | 顶部四步进度条（纯展示） | 条件渲染三段判断 |
| `frontend/src/components/ThemeToggle.tsx` | 明暗主题切换 + localStorage 持久化 | `useEffect` 恢复 + `toggle()` |
| `frontend/src/app/layout.tsx` | 根布局、主题首帧脚本、hydration 处理 | `<head>` 内联脚本 |
| `frontend/src/app/globals.css` | Tailwind 层、明暗主题覆盖、动画 | `@layer` 与 `html.light` 覆盖块 |

### 表 2：session 关键字段

| 字段 | 类型 | 含义 | 写入时机 |
| --- | --- | --- | --- |
| `skill_files` | dict | 上传的技能文件（含 SKILL.md 与 references/） | 上传时 |
| `file_list` | list | 文件名清单 | 上传时 |
| `metadata` | dict | SKILL.md frontmatter 解析出的名称 / 描述 | 上传时 |
| `status` | str | uploaded / analyzed / configured / running / complete / error / stopped | 各路由 |
| `scenarios` / `evals` | list | 测试场景与评估标准（前端可编辑） | analyze / update-config |
| `experiments` | list | 每轮实验的快照（baseline / keep / discard，含 candidates / dimension_scores / diff_summary） | 优化 callback |
| `final_result` | dict | 前端 ResultsStep 期望的完整结果 | 优化完成 callback |
| `current_skill_md` | str | 当前最优的 SKILL.md（改进后） | 优化完成 |
| `original_skill_md` | str | 上传时的原始 SKILL.md | 上传时 |
| `created_at` | float | 创建时间（用于过期清理） | 上传时 |

### 表 3：API / 进度事件及其前端消费者

| 端点 | 方法 | 用途 | 前端消费者 |
| --- | --- | --- | --- |
| `/api/upload` | POST | ZIP 上传，创建 session | `UploadStep.uploadZip` |
| `/api/upload-files` | POST | 文件夹上传，创建 session | `UploadStep.uploadMultipleFiles` |
| `/api/analyze` | POST | 生成 scenarios / evals | `UploadStep.handleAnalyze` |
| `/api/regenerate` | POST | 重新生成 scenarios / evals | `ConfigStep.handleRegenerate` |
| `/api/update-config` | POST | 保存用户编辑后的配置 | `ConfigStep.handleContinue` |
| `/api/start/{id}` | POST | 启动后台优化任务 | `RunningStep.startOptimization` |
| `/api/status/{id}` | GET | 轮询进度（当前 UI 实际使用） | `RunningStep` 轮询循环 |
| `/api/stream/{id}` | GET | SSE 事件流（存在但**未被当前 UI 消费**） | 无（保留接口） |
| `/api/stop/{id}` | POST | 停止（协作式取消：轮间生效，当次模型调用返回后停止） | `RunningStep.handleStop` |
| `/api/download/{id}` | GET | 下载改进后的 zip | `ResultsStep.handleDownload` |
| `/api/examples` | GET | 列出示例技能 | 预留（示例按钮） |
| `/api/examples/{name}/load` | POST | 加载示例技能为 session | `UploadStep.handleExampleSelect` |
| `/health` | GET | 健康检查 | 无 |

优化器内部进度事件（经 callback 写入 session）：`baseline`、`experiment_start`、`experiment_result`、`complete`、`error`、`stopped`；`experiment_result` 额外携带 `candidates`（并行候选）、`dimension_scores`（逐维度得分）、`diff_summary`（保留轮次的 diff）。

---

## 四、零基础术语表

| 术语 | 含义 |
| --- | --- |
| 组件（component） | 前端 UI 的可复用块。本项目每个 `.tsx` 文件导出一个组件 |
| props | 父组件传给子组件的数据（只读），例如 `sessionId`、`onComplete` |
| state | 组件内部的可变数据，通过 `useState` 声明，修改后触发重新渲染 |
| hook | React 提供的"钩子"函数，如 `useState`、`useEffect`、`useRef`，让函数组件拥有状态与副作用 |
| 异步（async） | 不阻塞等待的操作。`async/await` 让代码"等结果的同时让出控制权" |
| 事件循环（event loop） | asyncio 的调度核心，循环处理事件与回调；被阻塞时其他请求无法响应 |
| 后台任务（background task） | `asyncio.create_task(...)` 启动、不阻塞当前请求的协程，例如优化任务 |
| 轮询（polling） | 客户端每隔固定时间主动询问服务器，本项目前端每 3 秒查一次 `/api/status` |
| SSE | Server-Sent Events，服务器单向推送事件流的技术；本项目的 SSE 路由保留但当前 UI 未使用 |
| schema | 数据的结构约定；Pydantic `BaseModel` 用于校验模型输出是否符合预期形状 |
| fallback | 兜底值：解析或校验失败时返回的默认数据，保证流程不中断 |
| prompt | 发给模型的提示词，本项目的 prompt 中会嵌入技能内容与 JSON 结构要求 |
| mutation | 变异/修改：Mutator 对 SKILL.md 做的一处针对性改动 |
| baseline | 基线：优化前原始技能的评分，作为每轮"是否保留修改"的比较基准 |

---

## 五、四个学习检查点

读完代码后，试着用自己的话解释以下四个问题（能讲清楚即说明主线已打通）：

1. **上传文件后 `session_id` 如何产生和传递？**
   `backend/app.py` 的 `create_session_from_files()` 用 `uuid.uuid4()` 生成 `session_id`，连同技能文件一起存入内存 `sessions` 字典，然后把 `session_id` 返回给前端；前端 `UploadStep` 把它存进 state，之后所有请求（analyze / start / status / download）都带上它，后端凭它找到对应 session。

2. **同步的 Qwen-Agent 调用为什么需要放进工作线程？**
   Qwen-Agent 的 `Assistant.run()` 是**同步生成器**，会阻塞当前线程。FastAPI 的接口跑在**事件循环**上，如果直接在协程里调用同步阻塞代码，整个事件循环会被卡住，其他请求（比如正在轮询的 `/api/status`）就无法响应。所以 `_ask()` 用 `asyncio.to_thread(...)` 把它丢进线程池执行，再 `await` 结果。

3. **三个 Agent 如何完成一次优化轮次？**
   每轮先由 **Analyst** 分析当前失败项，输出根因与修改策略；再由 **Mutator** 按诊断对 SKILL.md 做"恰好一处"修改（开启并行时每轮生成多个候选，各用独立 Mutator 实例），返回完整新内容；最后由 **Executor** 对修改后的技能重新执行所有测试场景并打分（并行时取最优候选）。`optimize()` 把这三步循环 `max_rounds` 次。

4. **为什么低分或同分的 mutation 会被丢弃？**
   `optimize()` 里 `kept = best_score > baseline_pct + improvement_threshold` —— 只有分数**严格高于**当前基线（默认阈值 0.0）才保留（更新 `current_md` 与基线），等于或低于基线的修改都会被丢弃并记为 `discarded`。此外变异在复评前先过**回归守卫**（frontmatter / 标题 / 体积完整性检查），命中即拒绝。这是本项目的行为不变量：宁可少改，也不允许模型乱改导致技能退化。

---

## 六、当前实现边界（重要）

以下行为是**当前实现**的真实情况，阅读代码时请与"理想设计"区分：

- **session 只存在于单进程内存**：`sessions` 是进程内 dict，后端重启即全部丢失；没有数据库、没有跨进程共享。
- **API Key 不持久化**：`qwen_api_key` 只存在于前端组件内存与后端请求体中，不写 localStorage、不落日志、不进环境变量、不持久化。
- **前端采用轮询（polling）**：`RunningStep` 每 3 秒调 `/api/status`；`/api/stream` 的 SSE 路由虽然存在，但当前 UI 没有消费它。
- **停止操作是协作式取消**：`/api/stop` 把 `stop_requested` 置位，优化器在**轮间**检查并抛出 `StopOptimizationError`，后端把状态置为 `stopped`。当次正在执行的模型调用无法中途中断，会在返回后生效 —— 这是 `asyncio.to_thread` 下的实际最优解（不换供应商）。
- **上传限制**：总量 10MB、单文件 1MB、最多 50 个文件，仅允许白名单文本扩展名；路径穿越（`..`、绝对路径）会被拒绝。

---

## 七、不注释文件说明

以下文件不属于教学注释范围，其作用在此统一说明：

- `backend/requirements.txt` — Python 依赖清单（FastAPI、uvicorn、pydantic、qwen-agent 等）。
- `frontend/package.json` / `package-lock.json` — 前端依赖与锁定版本（Next.js、React、Recharts、diff 等）。
- `frontend/tsconfig.json` / `next.config.*` / `postcss.config.*` — TypeScript 与构建配置。
- `frontend/src/app/favicon.ico` 等静态资源 — 无需注释。
- README（`README.md` / `README.en.md`）— 面向用户的使用与部署文档（保持英文，不含教学注释）。
