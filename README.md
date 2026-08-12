# 🔥 SkillForge —— 自我进化型 Agent 技能锻造厂

**SkillForge** 基于 **LangGraph** 状态图编排与 **阿里云百炼（DashScope）** 直调构建的技能自优化系统，自动改进你的 Agent 技能。上传一个技能，让智能体生成测试场景与评估标准，然后观察三个角色（Executor / Analyst / Mutator）通过迭代优化协作改进你的技能 —— 每一次变异都在砧台上被锤打、评估，只保留真正更强的版本。

> 一个受 Karpathy「自动研究」方法论启发的个人项目：与其手动调提示词，不如定义成功标准，让 AI 自我进化。


## 工作原理

本应用实现了受 Karpathy「自动研究」（autoresearch）方法论启发的自动化技能改进循环，由三个专业角色驱动：

1. **上传**：拖入你的技能文件夹（遵循 [agentskills.io](https://agentskills.io) 规范）
2. **配置**：Executor 智能体生成测试场景与评估标准，可按需编辑、添加或重新生成
3. **优化**：三角色协作 —— 一个执行并评分，一个诊断失败原因，一个应用修复
4. **结果**：下载改进后的技能及详细变更日志

### 智能体团队

| 智能体 | 角色 | 职责 |
|-------|------|-------------|
| **Executor** | 技能执行与评分 | 针对测试场景执行技能，按评估标准对输出评分，并在分析阶段生成初始测试场景 |
| **Analyst** | 失败诊断师 | 检查失败评估、定位根因并推荐变异策略。通过 Pydantic `output_schema` 保证结构化 JSON 输出 |
| **Mutator** | 提示词编辑 | 基于分析师诊断，对技能提示词做出**恰好一处**针对性修改。通过 Pydantic `output_schema` 保证结构化 JSON 输出 |

### 优化循环

- **Executor** 智能体针对所有测试场景运行技能
- **Executor** 再对每个输出按二值（是/否）评估标准打分，可选**按维度加权**（correctness / clarity / executability 等）；带 `check_type` 的可程序化标准（keyword / regex / yaml_or_json / length）直接用 Python 规则判定，不消耗 LLM 调用
- **Analyst** 智能体诊断失败模式，并从**可配置策略池**（`add_example`、`add_constraint`、`restructure`、`add_edge_case`、`add_reference`、`rewrite_section`、`fix_format`）中选择策略；会看到最近几轮的修改记忆，已尝试未提升的策略会被标注 blocked，被拒的编辑摘要会被注入避免重复
- **Mutator** 智能体对技能提示词应用一处精准修复；开启并行时每轮同时生成多个候选变异
- **Executor** 重新运行并重新评估修改后的技能（并行时取分数最优的候选）
- 变异先过**回归守卫**（frontmatter / 标题 / 体积完整性检查），再按「严格高于基线 + 提升阈值（+ 可选的噪声地板）」决定保留，否则回滚；开启 `EDIT_LIMIT` 时超幅度的变异在复评前即被拒绝
- 评分达到 100%（全部评估通过）立即停止循环，否则循环直至达到最大轮数

## 架构

```
skillforge/
├── backend/                 # FastAPI 服务端 + LangGraph 优化引擎
│   ├── app.py              # REST API 端点 + SSE 流式推送
│   ├── qwen_optimizer.py   # 算法核心（评分/诊断/变异/回归守卫/经验库）
│   ├── optimize_graph.py    # LangGraph 状态图编排（条件边 + Send 并行）
│   ├── llm_client.py        # DashScope 直调薄封装（桥接/重试/JSON mode）
│   └── requirements.txt
├── frontend/               # Next.js + React + Tailwind
│   ├── src/
│   │   ├── app/            # 主页面 + 布局 + icon.svg（品牌标识）
│   │   ├── components/     # 上传、配置、运行、结果四个步骤 + Logo
│   │   └── lib/api.ts      # 前端与后端通信的唯一 API 出口
│   ├── package.json
│   └── *.config.ts
├── skill-examples/         # 内置示例技能包（.zip，/api/examples 自动读取）
└── README.md
```

## 技术栈

- **后端**：Python 3.10+、FastAPI、LangGraph、DashScope、Pydantic
- **前端**：Next.js 15、React 19、Tailwind CSS v4、Recharts
- **AI**：LangGraph 状态图编排的三角色循环，DashScope SDK 直调 Qwen（`qwen-plus`）；结构化输出走协议层 JSON mode + Pydantic 校验 + 宽容解析兜底
- **实时通信**：Server-Sent Events（SSE）实时推送优化进度

## 快速开始

> **一键启动**：项目根目录执行 `./start-dev.sh` 可同时拉起前后端（单终端，Ctrl+C 全部停止；端口已占用时自动跳过）。默认模型 `qwen3.7-max-2026-05-17` 并开启思考模式，可用 `QWEN_MODEL` / `QWEN_ENABLE_THINKING` 环境变量覆盖。

### 后端配置

```bash
cd backend

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows 下: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 启动服务
python app.py
# 服务运行在 http://localhost:8891
```

后端默认使用 `qwen-plus` 模型，可通过可选的 `QWEN_MODEL` 环境变量覆盖，例如 `QWEN_MODEL=qwen-max python app.py`。

### 前端配置

```bash
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
# 应用运行在 http://localhost:3000
```

### 使用步骤

1. 在[阿里云百炼控制台](https://bailian.console.aliyun.com/)获取 DashScope API Key
2. 打开 http://localhost:3000
3. 以 .zip 格式上传技能文件夹（或试用示例）
4. 输入你的 DashScope API Key
5. 审阅并编辑生成的测试场景与评估标准
6. 点击「Start Optimization」，观看智能体协作改进你的技能
7. 完成后下载改进后的技能

## 技能格式

技能遵循 [agentskills.io](https://agentskills.io) 规范：

```
my-skill/
├── SKILL.md           # 必需：YAML frontmatter + 指令
├── scripts/           # 可选：可执行代码
├── references/        # 可选：补充文档
└── assets/            # 可选：模板、资源
```

SKILL.md 示例：

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

## 试用

将任意技能文件夹打包为 zip 后上传。仓库自带 4 个内置示例技能包（`skill-examples/*.zip`），应用中的「examples」选择器会自动读取它们 —— 都是真实技能，而非玩具示例：

```bash
# 内置示例（应用内一键加载，无需手动打包）
# project-graveyard / thinking-out-loud / dependency-doctor / commit-archaeologist

# 手动打包自己的技能：
zip -r my-skill.zip my-skill/
```

## 多智能体优化是如何运作的

### 1. 分析阶段
**Executor** 智能体分析你的技能并生成：
- 3-4 个多样化的测试场景
- 4-6 条二值评估标准（是/否问题）

在优化开始前，你可以编辑、添加或删除场景与标准。

### 2. 基线运行
**Executor** 智能体针对所有场景运行技能，并按所有评估标准对每个输出评分，确立起始分数。

### 3. 优化循环
每一轮，三个智能体协作：
1. **Executor** 针对所有测试场景运行技能并对输出评分（可选按维度加权）
2. **Analyst** 检查失败项、定位根因并从策略池中选择变异策略（返回经 Pydantic schema 校验的结构化 JSON）
3. **Mutator** 应用一处针对性修改以改进技能（返回经 Pydantic schema 校验的结构化 JSON）；开启并行变异时每轮同时生成多个候选
4. **回归守卫** 检查每个候选的结构完整性（frontmatter、标题、体积），命中即拒绝且跳过复评
5. **Executor** 重新运行并重新评估修改后的技能（并行时取分数最优的候选）
6. 比较分数 —— 严格高于「基线 + 提升阈值」则保留，否则回滚
7. 可选地从总分同分但局部维度更优的未保留候选中提炼经验元数据；同分候选仍然回滚，绝不成为下一轮版本
8. 评分达到 100%（全部评估通过）立即停止循环，否则重复直至最大轮数

### 4. 输出
- 应用了全部成功修改的改进版 SKILL.md
- 详细的变更日志（改了什么、为什么改、用了什么策略、并行候选明细）
- 性能对比（基线 vs 最终，含逐维度得分趋势）

### 模型输出语言

模型侧的分析、评分、诊断和修改提示词均以中文原生编写。测试场景、评估标准、评分理由、诊断结果和变更说明默认使用自然、简洁的简体中文，不采用“先用英文生成、再强制翻译”的方式。

- 如果 `SKILL.md` 或用户请求明确指定英文或其他语言，Executor 会优先遵守该显式要求。
- Mutator 每轮仍只修改一个目标点；对英文技能不会先做全文翻译，只会让本轮新增或改写的正文默认使用中文。
- JSON 字段名、`mutation_strategy` 枚举、YAML 键、代码、命令和专有标识保持原样，以保证 API 与下载文件兼容。

## API 端点

| 方法 | 端点 | 说明 |
|--------|----------|-------------|
| `POST` | `/api/upload` | 上传技能 zip（最大 10MB，仅文本文件） |
| `POST` | `/api/upload-files` | 上传多个文件（文件夹上传） |
| `POST` | `/api/analyze` | 生成场景与评估标准（需要 DashScope API Key） |
| `POST` | `/api/regenerate` | 重新生成场景与评估标准 |
| `POST` | `/api/update-config` | 保存用户选择/编辑的配置 |
| `POST` | `/api/start/{session_id}` | 启动优化 |
| `GET` | `/api/stream/{session_id}` | SSE 实时推送优化进度 |
| `POST` | `/api/stop/{session_id}` | 停止优化 |
| `GET` | `/api/download/{session_id}` | 下载改进后的技能 |
| `GET` | `/api/examples` | 列出可用示例技能 |
| `POST` | `/api/examples/{name}/load` | 加载示例技能 |
| `GET` | `/api/status/{session_id}` | 轮询式状态端点 |
| `GET` | `/health` | 健康检查 |

## 配置

### 后端

DashScope API Key 随每次请求由前端传入（字段 `qwen_api_key`），并直接用于 DashScope SDK 调用配置 —— 绝不写入进程级环境变量、不存入会话、不记入日志。模型默认使用 `qwen-plus`，可通过可选的 `QWEN_MODEL` 环境变量覆盖；Executor、Analyst、Mutator 还可分别用 `EXECUTOR_MODEL`、`ANALYST_MODEL`、`MUTATOR_MODEL` 覆盖。服务运行在 **8891** 端口。

上传限制：
- 单次上传最大 **10MB**
- 单文件最大 **1MB**
- 单次最多 **50** 个文件
- 仅限文本文件（`.md`、`.txt`、`.json`、`.yaml`、`.py`、`.js`、`.ts` 等）

会话 **1 小时**后自动过期。

### 前端

API Key 在界面中输入，仅保存在组件内存中（不持久化），并以 `qwen_api_key` 字段随每次请求发送。

### 优化参数

SkillForge 提供一组**可选**优化旋钮，全部默认等价于经典行为（不传即与基础版一致）：

| 旋钮 | 位置 | 默认 | 说明 |
|------|------|------|------|
| `max_rounds` | 请求字段 | 20（≤50） | 最大优化轮数 |
| `parallel_mutations` | 请求字段 | 后端 `MUTATION_PARALLELISM`（默认 1，上限 3） | 每轮并行生成几个候选变异 |
| `strategy_pool` | 请求字段 | 全量策略模板 | Analyst 可用的变异策略白名单 |
| `improvement_threshold` | 请求字段 / `IMPROVEMENT_THRESHOLD` | 0.0 | 新分必须严格高于「基线 + 阈值」才保留 |
| 维度权重 | `ANALYST_DIMENSION_WEIGHTS`（JSON） | 无（等价旧通过率） | 按 eval 的 `dimension` 字段加权评分 |
| `REGRESSION_CHECK` | 环境变量 | 1（开启） | 变异先过结构完整性守卫，0 关闭 |
| `NOISE_FLOOR` | 环境变量 | 0.0 | 候选提升须超过「基线 + 阈值 + 噪声地板」才保留，防止评分波动被当成真进步 |
| `EDIT_LIMIT` | 环境变量 | 0.0 | 单次变异相对原文本的变化比例上限，超限直接拒绝（`edit_limit_exceeded`），0 关闭 |
| `EXECUTOR_MODEL` | 环境变量 | 继承 `QWEN_MODEL` | Executor 的执行、测试生成与评分模型；适合配置为成本较低的模型 |
| `ANALYST_MODEL` | 环境变量 | 继承 `QWEN_MODEL` | Analyst 的失败诊断模型；可配置为推理能力更强的模型 |
| `MUTATOR_MODEL` | 环境变量 | 继承 `QWEN_MODEL` | Mutator 的单点编辑模型；可配置为指令遵循更强的模型 |
| `OPTIMIZATION_SEARCH` | 环境变量 | classic | `classic` 保持原贪心路径；`adaptive` 为并行槽位分配不同策略，并按历史实测增益自适应探索/利用 |
| `SEARCH_EXPLORATION` | 环境变量 | 1.0 | `adaptive` 策略的 UCB 探索系数；0 只利用历史平均增益 |
| `TIE_DIMENSION_LESSONS` | 环境变量 | 0（关闭） | 总分与胜者并列、但某维度更优的未选候选沉淀维度经验；候选本身仍不保留 |
| `TIE_DIMENSION_MIN_GAIN` | 环境变量 | 0.0 | 平局候选相对胜者的维度提升须严格超过该百分点才沉淀 |
| `CROSS_ROUND_TIE_DIMENSION_LESSONS` | 环境变量 | 0（关闭） | 扫描本轮所有通过守卫且完成评分的候选；若总分与轮首 incumbent 相同但某维度更优，则提炼跨轮经验；候选仍不保留 |
| `CROSS_ROUND_TIE_DIMENSION_MIN_GAIN` | 环境变量 | 0.0 | 跨轮同分候选相对轮首 incumbent 的维度提升须严格超过该百分点才学习；非法值回退、负值钳制为 0.0 |
| `WEAK_DIMENSION_FOCUS` | 环境变量 | 0（关闭） | 识别低分维度并注入 Analyst/Mutator，重排策略与经验检索，仍只做一个单点修改 |
| `WEAK_DIMENSION_THRESHOLD` | 环境变量 | 50.0 | `pct` 小于等于该值的已测量维度进入专项列表 |
| `WEAK_DIMENSION_MAX` | 环境变量 | 2 | 每轮最多专项关注的弱维度数（按分数升序） |
| `QWEN_ENABLE_THINKING` | 环境变量 | 0（关闭） | 打开模型思考模式；部分模型（如 `qwen3.7-max-2026-05-17`）强制要求开启 |
| `MEMORY_ROUNDS` | 环境变量 | 3 | 注入给 Analyst 的最近轮次记忆条数（轮间经验，避免反复修同一根因） |
| `PATIENCE` | 环境变量 | 0（关闭） | 连续 N 轮未提升即提前终止优化（耐心早停） |
| `SATURATION_EXIT` | 环境变量 | 0（关闭） | 基线无可提升带（100% 或 0 分全失败）时跳过全部轮次 |
| `FINAL_CONFIRM` | 环境变量 | 0（关闭） | 完成前对最终版本独立复核，未超过基线则回退（防单点幸运） |
| `CANDIDATE_CONFIRM_RUNS` | 环境变量 | 0（关闭） | 初评胜者与当轮当前版本追加 1–3 次配对复评；每次都须继续严格胜出，并采用最低确认分，尤其适合并行候选 |
| `SKILL_LESSONS_FILE` | 环境变量 | 无（关闭） | 跨会话经验库路径（jsonl 或 SQLite）：保留修改和启用后的同分维度元数据可沉淀为经验，再注入 Analyst/Mutator |
| `LESSON_N` | 环境变量 | 5 | 每次优化读取的最近经验条数 |
| `LESSON_RETRIEVAL` | 环境变量 | off | 经验注入方式：`off`（最近 N 条）/ `tag`（同技能硬过滤）/ `semantic`（embedding 语义检索）/ `hybrid`（dense+sparse 混合）。semantic/hybrid 需要经验库使用 `.db` 后缀（SQLite）并调用 DashScope text-embedding-v3 |
| `LESSON_THRESHOLD` | 环境变量 | 0.3 | semantic 检索的余弦相似度阈值（0~1） |
| `LESSON_TOP_K` | 环境变量 | 5 | semantic/hybrid 检索后注入的经验条数（与 LESSON_N 读取条数独立） |
| `LESSON_RERANK` | 环境变量 | 0（关闭） | 开启后对检索候选池调用 DashScope gte-rerank 重排再取 top-K；失败时保留重排前顺序 |
| `LESSON_RERANK_POOL` | 环境变量 | 20 | rerank 的候选池大小 |
| `LESSON_RAG_PIPELINE` | 环境变量 | classic | `classic` 保持原排序；`quality_diverse` 开启语义/词面/技能/领域/弱维度/历史收益/时序多信号排序，再去重并做多样性选择 |
| `LESSON_CANDIDATE_POOL` | 环境变量 | 20 | `quality_diverse` 在去重与多样性选择前保留的候选数 |
| `LESSON_DIVERSITY` | 环境变量 | 0.2 | `quality_diverse` 的重复惩罚（0–1；越高越偏好多样经验） |
| `LESSON_DEDUP_THRESHOLD` | 环境变量 | 0.92 | `quality_diverse` 的近重复 Jaccard 阈值（0–1） |
| `LESSON_CONTEXT_CHARS` | 环境变量 | 6000 | `quality_diverse` 注入经验的总字符预算（至少 500） |
| `LESSON_SIGNAL_WEIGHTS` | 环境变量 | 内置权重 | 可选 JSON，覆盖并自动归一化 `semantic/sparse/skill/domain/dimension/quality/recency` 权重 |
| `LESSON_FAILURE_CONTEXT` | 环境变量 | 0（关闭） | 开启后把失败 eval 的标准、期望和对应失败场景加入当次检索查询；不持久化这些内容 |
| `LESSON_EMBED_BATCH_SIZE` | 环境变量 | 1 | 未缓存经验的 DashScope embedding 批量大小（1–10）；1 保持逐条调用，10 降低大经验库冷启动请求数 |
| `LESSON_MIN_GAIN` | 环境变量 | 0（关闭） | 普通/同轮经验的提升幅度门槛（与 `LESSON_MIN_FINAL` 为 OR；推荐 15）；跨轮同分经验如实为零总分增益，忽略此门槛 |
| `LESSON_MIN_FINAL` | 环境变量 | 0（关闭） | 最终水位门槛（推荐 85）；普通/同轮经验仍与 `MIN_GAIN` 为 OR，跨轮同分经验启用该值后必须达到此水位才持久化 |

> **安全与不变量**：凭据字段恒为 `qwen_api_key`；密钥仅存组件内存与请求体，绝不落日志/会话/zip/git。一次变异仍只改 SKILL.md 一处；只有「严格提升」的变异才会被保留，回归守卫只会更严格，绝不会让技能退化。

同分维度经验分为两个互不隐式联动的范围：

- `same_round`：`TIE_DIMENSION_LESSONS=1` 时，比较同一轮中总分与已保留 winner 相同的未选候选，参照维度是该 winner。
- `cross_round`：`CROSS_ROUND_TIE_DIMENSION_LESSONS=1` 时，比较本轮所有通过结构/编辑守卫且真正完成评分的候选，参照物是**截至本轮开始最近一次通过严格提升门被保留的 incumbent**（第一轮为初始版本）。连续数轮没有保留新版本时，参照物仍是更早那次最近被保留的版本，而不是上一轮的临时候选。

两类记录都使用 `lesson_type=tie_dimension`，并用 `comparison_scope` 区分；一个候选的多个优势维度聚合到同一条 `dimension_gains`，`target_dimension` 稳定选择增益最大的维度（同增益按维度名排序）。同一候选/维度同时命中两种比较时只记录一次，同轮信号优先。lesson 与轮次经验只含 strategy、diagnosis、summary、分数、目标维度等模型生成元数据，不含候选或 incumbent 的 SKILL.md、场景、执行/评分原文、Prompt、内部候选 ID或 API Key；既有 mutation log/API 中用于展示候选明细的 `candidate_id` 保持不变。

跨轮候选的 `score_before` 与 `score_after` 都记录真实同分，不伪造总分提升；维度增益必须严格超过 `CROSS_ROUND_TIE_DIMENSION_MIN_GAIN`。因此 `LESSON_MIN_GAIN` 不参与其质量判断；若启用了 `LESSON_MIN_FINAL`，总分还必须达到该水位才写入跨会话经验库。未达到持久化水位的元数据仍可留在本会话 `round_memory`，但候选的 `current_md`、基线分、维度基线和 `kept=false` 均不改变。

首轮仍在 baseline 后准备经验；开启跨轮学习后，后续每轮先检查协作式 stop，再在 Analyst 前刷新经验检索，使刚学到的维度信号可立即用于下一轮的弱维度定向。`off`/`tag` 只增加本地读取；`semantic`/`hybrid` 可能每轮增加查询 embedding，启用 rerank 时还可能增加 rerank 调用。embedding/检索失败会降级到 tag/轮内记忆，rerank 失败则保留重排前顺序，都不会影响优化主循环。JSONL 直接保存新字段；SQLite 用内置 `sqlite3` 向后兼容补充 `comparison_scope`，旧记录缺失该字段时按 legacy `same_round` 读取。默认开关为 0，因此不开启时调用路径与成本不变。

角色模型示例（未设置三个角色变量时，行为与原来完全相同）：

```bash
EXECUTOR_MODEL=qwen-plus \
ANALYST_MODEL=qwen-max \
MUTATOR_MODEL=qwen-max \
python backend/app.py
```

任一角色模型名以 `deepseek-` 开头时，仅该角色走现有 DeepSeek 显式路由；密钥仍只通过请求体直传。

在 `qwen_optimizer.py` 中调整模型：

```python
def __init__(self, api_key: str, model: Optional[str] = None):  # 默认取 QWEN_MODEL 或 "qwen-plus"
```

## 开发

### 后端测试

```bash
cd backend
python -m unittest discover -p 'test_*.py'
python -c "from qwen_optimizer import SkillOptimizer; print('OK')"
```

### 前端构建

```bash
cd frontend
npm run build
```

### 本地开发

两个服务均支持热重载，改完代码即可看到效果。

## 基于 Karpathy 的自动研究（Autoresearch）

本工具将 Andrej Karpathy 的自动研究方法论（利用 LLM 迭代改进自身提示词）应用于 Agent 技能。核心洞察是：与其手动调整提示词，不如定义成功标准，让 AI 自我优化 —— 优化循环由 LangGraph 状态图编排、DashScope 直调驱动。

原始概念：[https://github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch)
