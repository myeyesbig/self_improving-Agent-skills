# SkillForge × LangGraph 集成方案

> 版本：v3（2026-08-11）｜ 状态：**已实施**（v2 方案已按执行计划落地，见 /Users/luboru/.workbuddy/plans/stellar-thunder-curie.md 与执行总结；本文档保留为设计依据）
>
> **已拍板的方向**：用 **LangGraph** 替换 Qwen-Agent 的编排层；**明确不使用 LangChain**——不引入 `langchain`/`langchain-community`，不用 ChatTongyi / ChatPromptTemplate / with_structured_output 等组件。LLM 调用直接走 **DashScope SDK**（与经验库 embedding/rerank 同一家 SDK，依赖收敛）；模型服务仍只有阿里云百炼，Qwen-only 约束不变。
>
> v1 → v2 变更说明：初版方案依赖 langchain-community 的 ChatTongyi 作为模型层，v2 按用户决策移除全部 LangChain 组件，仅保留 LangGraph 做状态图编排。

---

## 0. TL;DR

| 问题 | 结论 |
|---|---|
| 接哪里？ | 核心是**优化循环编排**（手写轮循环 → LangGraph 状态图）；配套：LLM 调用层薄封装、可靠性（重试/超时）、节点级进度事件 |
| 用什么？ | **LangGraph**（StateGraph + Send API + astream）+ **dashscope SDK 直调**；不碰 LangChain |
| 收益？ | 控制流显式化（图即文档）、并行变异 Send API 取代手写实例池、依赖链变短、1551 行测试基本零改动 |
| 风险？ | DashScope SDK 同步调用仍需 to_thread 桥接（保留薄层）、重试需手写极简实现、langgraph 传递依赖 langchain-core（仅安装层面） |
| 怎么落地？ | 5 阶段：依赖冒烟 → LLM 薄封装 → 可靠性增强 → LangGraph 化 → 收尾清理 |

**前置条件**：本方案与现行 `AGENTS.md` 存在冲突（见第 6 章），实施前需您批准修订 AGENTS.md。

---

## 1. 各环节评估：接 LangGraph 还是保持现状（基于真实代码）

对 `backend/qwen_optimizer.py`（优化核心）与 `backend/app.py`（API 层）逐环节评估：

| # | 环节 | 现状（代码证据） | 决策 |
|---|---|---|---|
| ① | **优化循环编排** | `optimize()` 手写轮循环：baseline → analyst → 并行变异（两处 `asyncio.gather`，:661/:720 + 手写实例池 :386-401）→ 回归守卫 → 复评 → kept 判定 → 早停/满分/`check_stop` 退出 | ✅ **接 LangGraph**（核心价值点） |
| ② | **LLM 调用层** | 三个 `qwen_agent.agents.Assistant`（:365-381）共享 `llm_config`（`qwen_dashscope`，T=0.2，:354-362） | ✅ **替换为 dashscope SDK 直调**（`dashscope.Generation.call`），不用 ChatTongyi |
| ③ | **同步↔异步桥接** | `_run_agent_sync`（:408-431）+ `_ask` 的 `asyncio.to_thread`（:433-441） | ⚠️ **保留但收窄**。dashscope SDK 同样是同步的，桥接仍是必需的；但从"消费同步生成器"简化为"包装一次函数调用"，代码量减半 |
| ④ | **Prompt 管理** | 三条 system prompt 模块级常量（:104-121 等），用户 prompt 硬编码 f-string | ⚠️ **保持 f-string 不变**。不引 LangChain 就没有 ChatPromptTemplate；现有 f-string 直白可靠，模板化收益不足以引入新抽象。可选轻量改进：把长 prompt 抽成模块级函数，纯 Python |
| ⑤ | **结构化输出** | `_ask_json`（:443-470+）：schema 拼进 prompt → `json.loads` → `raw_decode` 截取 → `model_validate` → fallback | ⚠️ **保留现有解析链**。可选增强：dashscope 原生支持 `response_format={"type": "json_object"}` 参数，直传即可，无需 with_structured_output |
| ⑥ | **可靠性** | **没有重试、超时、限流处理** | ✅ **新增手写极简 retry**（约 15 行指数退避）+ 超时参数；不引入 tenacity 等新依赖 |
| ⑦ | **进度回调** | 闭包 callback 推送事件 → `app.py` 会话 dict → `/api/status` 轮询 | ✅ **升级为 LangGraph 原生事件**：`graph.astream(stream_mode="updates")` 按节点产出进度，不依赖 LangChain callbacks |
| ⑧ | **经验检索（RAG）** | `SKILL_LESSONS_FILE` jsonl/SQLite 双存储 + `LESSON_RETRIEVAL` 四档 + text-embedding-v3 + gte-rerank + 静默降级链 | ❌ **保留自研不动**（已批准例外，降级链复杂且被测试覆盖） |

**明确不做的事**：
- 不引入 `langchain` / `langchain-community` 的任何组件（无 ChatTongyi、无 ChatPromptTemplate、无 with_structured_output、无 LangChain callbacks/memory）。
- 不需要 Agent/Tool 抽象：Executor 是对 skill 文本的模拟执行与打分（纯文本进、文本/JSON 出），没有外部工具调用场景。
- 不需要 ChatMemory：`round_memory`/`mutation_log` 是优化算法的领域状态，直接作为 LangGraph State 字段。
- 前端与 `app.py` 的轮询接口、会话 TTL、上传校验、下载打包全部不动。

---

## 2. LangGraph 状态图设计（核心）

### 2.1 State schema

```python
from typing import Annotated, TypedDict
from operator import add

class OptimizeState(TypedDict):
    skill_md: str                      # 当前最优 SKILL.md
    scenarios: list                    # 测试场景
    evals: list                        # 评估标准
    baseline_pct: float                # 基线分
    score_history: list
    round_idx: int
    round_memory: Annotated[list, add] # reducer 追加，对应现 optimize() 同名局部变量
    mutation_log: Annotated[list, add]
    lessons: list                      # 经验库检索结果（few-shot 注入）
    analysis: FailureAnalysis | None
    candidates: list[SkillMutation]
    kept: bool
    best: dict
    stop_flag: bool
    progress_events: Annotated[list, add]
```

### 2.2 节点与边

```
baseline → check_stop → analyst → mutate_fanout (Send API) → guard
   ↑                                                            ↓
   └──────────── decide ← reeval_fanout (Send API) ←───────────┘
                 ↓ (满分/饱和/耐心早停/stop_flag/达 max_rounds)
              finalize（final_confirm 复核 + 经验沉淀）
```

- `baseline`：封装 `_score_skill`。
- `analyst`：策略池白名单 + blocked 标注 + round_memory + rejected-edits + lessons few-shot，全部保留为 prompt 数据注入。
- `mutate_fanout`：LangGraph **Send API** map-reduce——每个候选 `Send("mutate_slot", {...})` 进入独立子状态，复评同理；`guard`/`decide` 为 reduce 汇聚点。
- `guard`：`_regression_check`（frontmatter/name/标题/缩水 40%/超 15KB）+ edit_limit 预筛；被拒候选打 `rejected` 标记直接进 decide 汇总。
- `decide`：kept 判定 `score > baseline + improvement_threshold + noise_floor`。
- **条件边**：`check_stop`/`decide` 后路由——满分 100、饱和/耐心早停、`stop_flag` 置位、达到 max_rounds → `finalize`；否则回边继续轮循环。

### 2.3 两个关键语义映射

1. **协作取消不用 interrupt**。LangGraph 的 interrupt 是"挂起等待人工输入"语义，与 `/api/stop` 不符。做法：`check_stop` 节点读取现有 `self._stop_event`，置 `stop_flag` 由条件边退出——轮间生效、在途模型调用完成，语义与现状完全一致。
2. **并行独立实例语义天然保持**。原实现靠"每槽独立 Assistant 实例"避免并发 `run()` 的线程安全问题；改为 dashscope SDK 直调后，LLM 客户端无会话状态，Send 子状态深拷贝输入、禁止共享可变对象即可等价。`decide` 按原 `asyncio.gather` 结果顺序聚合，保证日志/round_memory 顺序确定。

### 2.4 LLM 调用层薄封装（替代 ChatTongyi 的部分）

新增 `backend/llm_client.py`，职责收敛为三件事：

```python
# 伪代码示意
async def ask(system: str, prompt: str, *, json_mode: bool = False) -> str:
    """DashScope 同步 SDK 的 async 薄封装：to_thread 桥接 + 重试 + 超时。"""
    return await asyncio.to_thread(_call_with_retry, system, prompt, json_mode)

def _call_with_retry(system, prompt, json_mode, max_attempts=3):
    """指数退避重试；仅对网络/限流类异常重试，解析错误不重试。"""
    ...
```

- 底层调用 `dashscope.Generation.call(model=self.model, messages=[...], temperature=0.2, ...)`；`QWEN_MODEL` 环境变量覆盖逻辑、`QWEN_ENABLE_THINKING` 行为逐一对齐现状。
- `json_mode=True` 时传 `response_format={"type": "json_object"}`（dashscope 原生参数），配合保留的 `_ask_json` 宽容解析链 + Pydantic 校验 + fallback。
- API key 沿用现有做法：直传调用配置，绝不写进程环境变量/日志。

### 2.5 进度事件

图执行用 `async for event in graph.astream(state, stream_mode="updates")`：每个节点完成即产出一个事件，事件处理器把"节点名/轮次/得分"写入 `app.py` 会话 dict——`/api/status` 轮询接口与前端零改动，只是事件源从闭包 callback 换成图流。

---

## 3. 行为不变量保护策略

- **公共接口不动**：`SkillOptimizer.optimize(...)` 签名、返回结构、`check_stop()`、全部环境变量旋钮（`MUTATION_STRATEGIES` / `REGRESSION_CHECK` / `IMPROVEMENT_THRESHOLD` / `NOISE_FLOOR` / `EDIT_LIMIT` / `FINAL_CONFIRM` / `LESSON_RETRIEVAL` / `SKILL_LESSONS_FILE`）原样保留，内部实现从手写循环换成图编译执行。
- **纯函数下沉**：`_score_skill`、`_regression_check`、`_active_strategy_pool`、kept 判定表达式提取为模块级纯函数，图节点只做编排。1551 行测试中针对这些函数与旋钮的用例**零改动**。
- **prompt 逐字保留**：三条角色 prompt 文本不改一字；`FailureAnalysis`/`SkillMutation` 模型复用；`_ask_json` 宽容解析链原样保留。
- **测试影响面收敛**：保留 `_run_agent_sync`/`_ask`/`_ask_json` 的函数签名作为薄壳（内部改调 `llm_client.ask`），使现有 mock 点不变；仅少数涉及 Assistant 实例池构造的用例需调整。

---

## 4. 各集成点预期收益

| 集成点 | 预期收益 |
|---|---|
| LangGraph 状态图（核心） | 优化循环控制流显式化——`graph.get_graph().draw_mermaid()` 直接生成流程图（图即文档）；早停/停止/回归拒绝从嵌套 if 变为声明式条件边；Send API 取代手写 gather + 实例池，并行槽位隔离语义由框架保证 |
| dashscope SDK 直调替换 Assistant | 与经验库 embedding/rerank 同一家 SDK，**依赖收敛**（qwen-agent 0.0.34 依赖声明不完整、需手动补装 numpy/soundfile/python-dateutil 的问题随之消失）；同步生成器消费简化为单次函数调用，桥接层代码量减半 |
| 手写极简 retry + 超时 | 网络抖动/限流自动恢复；20 轮 × 每轮多次模型调用的长任务中途失败率显著下降；纯增益、无行为变化、零新依赖 |
| dashscope `response_format` JSON mode | 从协议层约束输出，`FailureAnalysis`/`SkillMutation` 解析成功率提升、fallback 触发减少；仅需直传参数，不依赖 LangChain |
| `graph.astream` 节点级事件 | 进度事件天然带节点粒度（"当前在 Analyst 诊断中"）；事件流是 LangGraph 原生能力，无需 LangChain callbacks |
| 保留自研经验检索 / prompt / 解析链 | 已验证的行为零风险；重写成本为零 |

综合：**自动化程度**（重试自愈、JSON mode 免解析失败）、**可维护性**（控制流声明式、依赖链变短）、**可观测性**（节点级进度事件）三线并进，且把第三方框架依赖压缩到只剩 LangGraph 一个。

---

## 5. 潜在风险与缓解

| # | 风险 | 说明 | 缓解措施 |
|---|---|---|---|
| 1 | **DashScope SDK 仍是同步的** | 不用 ChatTongyi 意味着失去原生 async，`to_thread` 桥接必须保留 | 桥接收窄为 `llm_client.py` 内一处薄封装（约 15 行）；这是"无 LangChain"决策的已知代价，换来零框架耦合 |
| 2 | **重试需手写** | 无 `with_retry` 可用 | 手写约 15 行指数退避（仅对网络/限流异常重试，上限 3 次）；逻辑简单、可控、被单测覆盖 |
| 3 | **langgraph 传递依赖 langchain-core** | LangGraph 构建在 langchain-core 的 runnable 协议上，安装时会带入 | 仅安装层面的传递依赖；**代码零 import 任何 langchain 命名空间**；requirements 只声明 `langgraph` 并锁版本界；禁词扫描改为强制约束代码中不出现 `langchain` import |
| 4 | **SDK 行为差异** | qwen-agent 的 Assistant 对 dashscope 有默认包装（消息格式、重试），直调 SDK 后参数需显式管理 | temperature/enable_thinking/max_tokens 显式对齐现状；`QWEN_ENABLE_THINKING` 行为逐项核对；`max_rounds=1` 真实冒烟**对比替换前后**的 FailureAnalysis/SkillMutation 解析成功率 |
| 5 | **LangGraph 学习曲线与调试可见性** | 状态图调试比线性代码难 | 图用 `draw_mermaid()` 固化进文档；`astream` 事件本身就是调试视图 |
| 6 | **并行变异隔离语义被破坏** | Send 子状态若共享可变对象会引入隐蔽 bug | 子状态输入深拷贝；禁止共享可变对象；decide 按 gather 顺序聚合保确定性 |
| 7 | **测试改造成本** | 1551 行测试多处 mock `Assistant.run`/`_run_agent_sync` | 薄壳先行（签名不变、内部换实现），测试主体零改动；再逐步把 mock 点迁移到 `llm_client` |
| 8 | **成本与延迟** | 重试增加少量 token 消耗；JSON mode 输出略慢 | 重试上限 3 次且仅对网络/限流生效；解析成功率提升抵消开销；轮次数与判定逻辑不变，成本量级不变 |
| 9 | **与 AGENTS.md 的冲突** | 现行约束"Qwen-only、不新增依赖设施" | 实施前需您批准修订 AGENTS.md（见第 6 章建议文本）；方案仍保持 DashScope 唯一模型服务 |

---

## 6. 分阶段实施路径

每阶段验证标准：**全量单测绿 + import 冒烟 + `max_rounds=1` 真实 DashScope 冒烟**（阶段 0 代码零改动）。

### 阶段 0：依赖与冒烟（零代码改动）
- requirements.txt 新增 `langgraph`（锁版本界），顺手补声明既有隐性依赖 `dashscope` / `numpy` / `pyyaml`。
- import 冒烟：`python -c "import langgraph; from langgraph.graph import StateGraph"`。
- ✅ 验证：全量单测绿（代码未动，理应全绿）。

### 阶段 1：LLM 薄封装
- 新增 `backend/llm_client.py`：dashscope SDK 直调 + `to_thread` 桥接 + 手写指数退避重试 + 超时 + JSON mode 参数。
- `_run_agent_sync`/`_ask`/`_ask_json` 签名保留为薄壳，内部改调 `llm_client`。
- ✅ 验证：单测绿 + `max_rounds=1` 真实冒烟产出合法 baseline 评分。

### 阶段 2：可靠性与 JSON mode
- Analyst/Mutator 调用开启 `response_format` JSON mode；验证现有宽容解析 + fallback 兜底兼容。
- 重试参数（次数/退避基数）收敛为模块级常量。
- ✅ 验证：单测绿 + 冒烟中 FailureAnalysis/SkillMutation 解析成功率**不低于**替换前。

### 阶段 3：LangGraph 化 optimize()
- 新增 `backend/optimize_graph.py`，按第 2 节建图；节点复用下沉的纯函数；进度事件改走 `astream`。
- `optimize()` 内部切换为图执行，公共签名不变。
- ✅ 验证：单测绿（含早停/满分/回归拒绝/edit_limit/kept 阈值/协作停止全部用例）+ `max_rounds=1` 与多轮冒烟行为对齐。

### 阶段 4：收尾清理
- 移除 `qwen-agent` 依赖与残留 import，删除薄壳中的死代码。
- 禁词扫描：`qwen_agent|qwen-agent` 仅允许出现在文档/历史注释；代码中零 `langchain` import；原有禁词扫描（`Google ADK|Gemini|gemini_api_key|...`）照常。
- 更新 AGENTS.md / CODE_WALKTHROUGH.zh-CN.md / README（文档保持英文惯例，中文文档单独维护）。
- ✅ 验证：全量单测 + import 冒烟 + 前端 `npm run build` + 禁词扫描零命中。

---

## 7. 与 AGENTS.md 的关系（实施前置条件）

现行 AGENTS.md 与本方案冲突的条款及建议修订文本：

1. **原条款**："Do not add … or generic provider abstractions…" 与 "Do not add tools, RAG, MCP, databases, queues…"
   **建议追加**："Approved exception (2026-08-11, user-explicit): LangGraph may be adopted as the orchestration layer for the optimization loop. LangChain itself (`langchain`/`langchain-community`) must NOT be used — no ChatTongyi, no LangChain prompts/memory/callbacks; model calls go directly through the DashScope SDK, which remains the only model service."
2. **原条款**：依赖清单与"Commands"中的 qwen-agent 相关内容。
   **建议**：阶段 4 完成后替换为 langgraph 与新的冒烟命令。
3. **不变量条款全部保留**：三角色、单点变异、严格提升、回归守卫只严不松、并行独立实例语义、协作式停止——第 3 节已逐条给出保护策略，修订时原样保留。

---

## 附：关键文件清单

| 文件 | 角色 |
|---|---|
| `backend/qwen_optimizer.py` | 改造主体（三角色、循环、评分、守卫、经验库） |
| `backend/app.py` | 仅进度事件源微调（callback → graph.astream） |
| `backend/requirements.txt` | 依赖增删（+langgraph，−qwen-agent，补 dashscope/numpy/pyyaml） |
| `backend/test_qwen_optimizer.py` | 1551 行回归安全网 |
| `backend/llm_client.py`（新增，阶段 1） | dashscope 直调薄封装（桥接/重试/超时/JSON mode） |
| `backend/optimize_graph.py`（新增，阶段 3） | LangGraph 状态图定义 |
| `AGENTS.md` / `README.md` / `CODE_WALKTHROUGH.zh-CN.md` | 阶段 4 同步更新 |
