# 同类项目调研与融合思路（SkillForge）

> 调研时间：2026-08-07。方法：GitHub API 检索（self-improving agent / skill optimizer / self-evolving agent）+ 网络搜索 + 项目 README/论文深读。
> 目的：对照同类项目，验证 SkillForge 现有设计，并吸收可融合的机制。本文是分析文档，不含代码改动。

## 一、同类项目全景

| 项目 | Star | 定位 | 与 SkillForge 的相关度 |
|---|---|---|---|
| **SkillOpt**（微软） | ~8.4k | 把 SKILL.md 当"可训练参数"，用神经网络训练纪律（epoch/batch/learning rate/validation gate）优化技能文本 | ★★★★★ 目标完全一致 |
| **Hermes Agent Self-Evolution**（Nous Research, GEPA） | 数千 | 进化技能的 SKILL.md / 工具描述 / 系统提示，双信号门控 + 闭环验证，ICLR 2026 Oral | ★★★★★ 门控/信号方法论最先进 |
| HyperAgents（Meta FAIR） | 2.7k | 自指（self-referential）自我改进，可优化任何可计算任务 | ★★★ 学术性强，落地距离远 |
| dgm（Darwin Gödel Machine） | 2.2k | 开放式进化：Agent 递归改写自身代码 | ★★ 代码级自我重写，超出技能文本优化范围 |
| SIA（hexo-ai） | 2.1k | 自改进框架：在基准任务上自主提升任意 AI 系统 | ★★★ 抽象度高于技能层 |
| GPTSwarm | 1.0k | RL / Prompt 优化的自我改进 Agent | ★★ 需要 RL 设施，与 Qwen-only/无训练约束冲突 |
| prime-agent / letta / autocontext / OS-Copilot / future-agi | 1-24k | 记忆平台 / 自改进 harness / 具身 agent / eval 平台 | ★ 更偏平台层，机制参考价值低 |

**结论**：真正值得深读融合的只有前两个（SkillOpt 与 GEPA）。其余要么学术过重、要么平台过宽、要么引入 RL/训练设施（违反 AGENTS.md）。

## 二、SkillOpt：训练技能像训练神经网络

GitHub: https://github.com/CloudEngineHub/SkillOpt（微软 microsoft/SkillOpt）｜论文 arXiv:2605.23904｜项目页 https://microsoft.github.io/SkillOpt/

核心机制（论文默认路径，GPT-5.5 上无技能→+23.5 分，52 个评测单元全部最佳或并列最佳）：

1. **技能文档 = 可训练状态，目标模型冻结**。不调权重、不改推理代码，只优化一个 Markdown 文件（最终产物 best_skill.md，300–2000 tokens），部署期零额外模型调用。
2. **训练循环**：rollout（目标模型执行任务）→ reflect（批判模型分析失败轨迹）→ aggregate（汇总）→ select（选编辑）→ update（应用）→ evaluate（验证门控）。
3. **有界编辑**：单独的 optimizer model 把打分后的轨迹变成**有界 add / delete / replace 编辑，作用在单个技能文档上** —— 与 SkillForge"一次变异只改一处"完全同构。
4. **验证门控**：候选编辑**只在 held-out 验证集上严格提升才被接受** —— 与 SkillForge"严格提升才保留"完全一致。
5. **文本学习率预算**（textual learning-rate budget）：控制每次编辑的幅度，防止大改漂移。
6. **拒绝编辑缓冲**（rejected-edit buffer）：记录被拒的编辑，防止优化器重复提出相同修改。
7. **epoch 级 slow/meta 更新**：多轮小改整合，稳定训练。
8. **SkillOpt-Sleep**（v0.2.0）：夜间离线自我进化引擎，harvest（收割历史会话）→ mine（挖掘模式）→ replay（回放复现）→ consolidate（held-out 门控后整合技能）。
9. **多后端**：openai_chat / claude_chat / **qwen_chat** / minimax_chat / openai_compatible 等 —— 但走 OpenAI 兼容协议，SkillForge 用 qwen_agent SDK，不可直接引入（AGENTS.md 禁 provider abstraction），只能借鉴机制。
10. **6 个内置 benchmark + WebUI 监控面板**。

## 三、Hermes Agent Self-Evolution（GEPA）：信号与门控

GitHub: https://github.com/jramos/agent-self-evolution（Nous Research teknium1 主导，GEPA 获 ICLR 2026 Oral）

**最重要的一句话**：*"the value is the signal + the gate, not the search."* —— 作者实验表明 GEPA（反思式提示进化）在自有语料上被简单 best-of-N 重采样追平，真正的种群进化（imbue darwinian_evolver）被 best-of-N 严格支配（6/23 vs 8/23）。**即：复杂的搜索算法（种群/交叉/变异池）收益有限，评分信号质量和门控严格性才是决定性因素。**

可借鉴的机制：

1. **噪声感知部署门**（noise-aware deploy gate）：真实 Agent A/B 测试带 **A/A 噪声基线**（noise floor）—— 提升幅度落在噪声内的一律不许部署；配对 **bootstrap 90% 置信区间**，区间排除零才判定为真实信号；另设 futility floor 与 baseline-diff regression floor。
2. **饱和预检**（saturation pre-flight）：基线没有提升空间时拒绝消耗预算 —— 实证发现二元 pass/fail 任务上强模型要么 1.00 要么 0.00，不存在 0.2–0.7 的"可提升带"，此时进化是浪费。
3. **三维 judge 评分**：LLM-as-judge 多维 rubric（正确性 / 语义保持 / 输出质量）。
4. **闭环行为套件**（closed-loop behavioral suite）：在真实 Agent 上跑任务套件确认行为真的改变了，而非只在合成数据集得分提升。
5. **双信号门**：合成留出（synthetic holdout）快速筛选 + 闭环最终确认；synth 打平以闭环为准 —— 防止"在小评估集上碰运气"的候选通过。
6. **验证套件门**：pytest 100% 通过、**大小限制**（skills ≤15KB、工具描述 ≤500 字符）、缓存兼容（会话中不产生变更）、**语义保持**（不偏离原始用途）、**人类 PR 审查**（永不直接提交）。
7. **部署纪律**：只提 PR 不直接提交，propose-only 默认。

## 四、对 SkillForge 的启示：已有设计被背书，改进应聚焦"信号+门控"

**SkillForge 的既有行为不变量获得了外部背书，无需动摇**：
- "一次变异只改一处" ≈ SkillOpt 的有界编辑；
- "严格提升才保留" ≈ SkillOpt 的验证门控 / GEPA 的 deploy gate；
- "回归守卫只更严不弱化" ≈ GEPA 的验证套件门 + 语义保持。

**改进方向应遵循 GEPA 结论**：不要引入种群进化/交叉（此前的 P5 多样性是边际项），把精力放在——
1. **信号质量**（评分更可信）：规则锚点（已计划）、维度加权真正生效（已计划）、去噪。
2. **门控严格性**（区分真提升与幸运）：A/A 噪声地板、显著性判断（bootstrap 的轻量替代）、语义保持维度、大小上限。

## 五、融合思路（对照已有计划 stellar-nebula-newton.md）

### A. 已有计划被强化的部分（照做）
- P0 修断链（improvement_threshold / dimension）—— 不变，维度加权未来可作为"三维评分"（正确性/语义保持/质量）的载体。
- P1 规则锚点评分 —— 正是"信号质量"；SkillOpt 的 benchmark 方法论建议 held-out 验证，可作为远期扩展。
- P2 轮间记忆 + 策略黑名单 —— **升级**：黑名单可进一步参考 SkillOpt 的 rejected-edit buffer，不只记策略名，还记"被拒编辑的摘要"，防止 Mutator 反复生成相同/相似修改（比纯策略黑名单更强，实现仍只是 prompt 注入 + 内存列表）。
- P3 耐心早停 —— 对应 GEPA 的饱和预检：除"连续 N 轮无提升"外，可加"基线处于无可提升带"（如 baseline 0 且所有 eval 全 fail）的快速退出。
- P4 跨会话经验库 —— 对应 SkillOpt-Sleep 的 harvest 雏形；远期可演进为"收割会话→挖掘→回放→整合"。

### B. 新融合点（本次调研新增，按优先级）

| 优先级 | 融合点 | 来源 | 落地方式（默认=旧行为） |
|---|---|---|---|
| P1 | **噪声地板/显著性门控**：候选提升需超过评分噪声估计（如提升 ≥ 2 个 eval 通过数，或对接近阈值的候选复评取平均）才保留 | GEPA noise-aware gate | 新旋钮 `NOISE_FLOOR`（默认 0=关闭，保持严格 `>` 语义字面不变；开启时在"严格提升"之上加最小可识别提升） |
| P1 | **编辑幅度控制（文本学习率）**：限定单次编辑的文本变化量（如改动 ≤15% 内容），防大改漂移 | SkillOpt textual learning-rate | Mutator 输出校验，超限即拒绝；env `EDIT_LIMIT` 默认关 |
| P2 | **语义保持维度**：C2 维度加权增加 semantic_preservation（变异不得偏离技能原始用途） | GEPA semantic preservation | evals 增加该维度 + Analyst/Mutator prompt 强调"保持原始意图"，配合 P0 修好的 dimension 加权 |
| P2 | **SKILL.md 大小上限**：回归守卫增加体积上限（如 15KB）防膨胀 | GEPA size limits | `_regression_check` 加一条规则（只更严，符合不变量） |
| P3 | **拒绝编辑缓冲**：被拒候选的编辑摘要进入黑名单上下文，Mutator 不再重复 | SkillOpt rejected-edit buffer | 与 P2 策略黑名单合并实现 |
| P3 | **胜出候选复核（双信号轻量版）**：最终胜出的 best 候选再评一次确认，防单点幸运 | GEPA dual-signal | 可选开关 `FINAL_CONFIRM` |
| 远期 | **双模型分工**：Analyst/Mutator 用强模型（qwen3.7-max），Executor 用默认模型 —— 对应 SkillOpt optimizer/target 分离；已有实证（8-05 日志：多智能体任务是 qwen-plus 弱项） | SkillOpt optimizer≠target | env `OPTIMIZER_MODEL`（注意 QWEN_MODEL 现有语义：整后端覆盖，需设计不冲突） |
| 远期 | **held-out 验证集**：scenarios 拆 train/holdout，防过拟合 | SkillOpt | 需场景数量足够，单技能场景少，列为远期 |
| 远期 | **SkillOpt-Sleep 式夜间回放**：历史会话→挖掘→回放→整合 | SkillOpt v0.2.0 | 超出当前单会话优化范围，仅记录方向 |

### C. 明确不采纳
- **种群进化/交叉/复杂搜索算法**（GEPA 实证被 best-of-N 支配；且与"严格提升"不变量、最小改动纪律冲突）。
- **RL/训练设施**（GPTSwarm、prime-agent 路线；违反 AGENTS.md Qwen-only/无训练约束）。
- **代码级自我重写**（dgm/Darwin Gödel Machine；超出"优化 SKILL.md"定位，风险不可控）。
- **多后端/provider abstraction**（SkillOpt 的 openai_compatible 路线；违反 AGENTS.md）。

## 六、融合后的建议实施路线

```
P0 修断链（improvement_threshold / dimension 字段）          [已计划, 不变]
P1 规则锚点评分 + 噪声地板 NOISE_FLOOR + 编辑幅度 EDIT_LIMIT  [已计划 + 新增B]
P2 轮间记忆 + 策略黑名单(+被拒编辑缓冲) + 语义保持维度 + 大小上限 [已计划 + 新增B]
P3 耐心早停(+饱和快速退出) + 胜出候选复核 FINAL_CONFIRM       [已计划 + 新增B]
P4 跨会话经验库（harvest 雏形）                               [已计划, 不变]
远期 OPTIMIZER_MODEL 双模型分工 / held-out / 夜间回放          [待评估]
```

每阶段保持"默认=旧行为"，沿用 AGENTS.md 验证流程（unittest + 导入冒烟 + live DashScope smoke + rg 扫描 + 前端 build）。

## 七、参考链接
- SkillOpt: https://github.com/CloudEngineHub/SkillOpt ｜ 论文 https://arxiv.org/abs/2605.23904 ｜ 微软研究博客 "Agent Skills as Trainable Parameters"
- Hermes Agent Self-Evolution (GEPA): https://github.com/jramos/agent-self-evolution
- HyperAgents: https://github.com/facebookresearch/HyperAgents
- Darwin Gödel Machine: https://github.com/jennyzzt/dgm
- SIA: https://github.com/hexo-ai/sia
- GPTSwarm: https://github.com/metauto-ai/GPTSwarm

## 八、2026-08-12 补充调研：从单一贪心到自适应策略组合

本轮通过 GitHub 连接器复核了几条更贴近当前问题的主线：

- [GEPA](https://github.com/gepa-ai/gepa) 使用执行轨迹反思、并行 proposal 与 Pareto-aware selection；可借鉴的是“利用诊断信号制造有目的的候选多样性”，而不是直接复制其依赖或允许劣化版本进入 SkillForge 主链。
- [DSPy MIPROv2](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/mipro_optimizer_v2.py) 将候选生成与指标驱动的搜索分开；可借鉴的是用历史试验结果指导下一批提案。
- [EvoPrompt](https://github.com/beeevita/EvoPrompt) 维护 prompt population，并用不同演化算子产生候选；可借鉴的是“不同候选必须来自真正不同的算子”，而非只改变一句提示。
- [PromptWizard](https://github.com/microsoft/PromptWizard) 组合多种 mutation style、critique/refine 与 top-N 选择；再次支持“候选多样性 + 反馈闭环”的方向。

SkillForge 选择轻量的 `OPTIMIZATION_SEARCH=adaptive`：保持当前最优版本作为唯一父节点，每个候选仍只做一处修改，严格提升门槛完全不变；仅在候选生成前，用确定性的 UCB 策略组合分配不同 mutation strategy。未尝试策略优先探索，已有策略按平均正增益与探索奖励排序。默认 `classic`，因此不开开关时调用路径逐字保持旧行为。

在多维评分上，采用两个比“保存第二名全文”更安全的补充：`TIE_DIMENSION_LESSONS` 只提炼并列候选相对胜者的维度增益元数据；`WEAK_DIMENSION_FOCUS` 把最低维度、相关失败项和适配的单点策略显式交给 Analyst/Mutator。前者保留 Pareto 信号但不保留候选，后者把反馈变成可执行的专项修改；两者都默认关闭，也都不能绕过总分严格提升门槛。

## 九、目标 5：成功经验 RAG 从“召回”升级为“可用上下文”

本轮复核了三个可直接映射到 SkillForge、且无需引入框架的主来源：

- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) 的 BGE-M3 路线把 dense、sparse 与 multi-vector 视为互补信号，并建议在初筛后使用 cross-encoder reranker。SkillForge 保留现有 DashScope embedding / `gte-rerank`，但把“词面与语义互补”落实为轻量多信号排序。
- [Qdrant](https://github.com/qdrant/qdrant) 同时提供 dense+sparse hybrid、RRF/DBSF、payload filtering、MMR 与 relevance feedback。SkillForge 借鉴 metadata soft boost、候选融合和 MMR 多样性，不引入向量数据库，也不持久化用户内容或反馈。
- [Haystack](https://github.com/deepset-ai/haystack) 的多检索器路径会并行汇总、去重并用 RRF 融合。SkillForge 对应地在一个本地函数内完成候选池、近重复删除与确定性归并，避免引入新依赖。

此前管线的主要问题不是“没有 embedding”，而是排序目标太单薄：同技能、同领域、弱维度、历史收益与经验重复度都没有进入最终选择；中文词面又会被 ASCII-only tokenizer 忽略；SQLite embedding 还可能被一起注入 prompt。新版 `LESSON_RAG_PIPELINE=quality_diverse` 因此采用：

1. embedding 语义、中文/英文词面、技能、领域、弱维度、历史质量和时序七信号加权；
2. `gte-rerank` 仍为可选的候选级交叉编码器，并与确定性质量分融合；
3. 近重复过滤后用 MMR 式贪心选择，避免 top-K 都是同一个建议的改写；
4. 只把白名单元数据注入模型，并设置总字符预算；内部 ID 与 embedding 永不进入 prompt；
5. 任一 embedding/rerank 异常仍静默降级 tag，默认 `classic` 完全保留旧路径。

没有照搬 relevance feedback：反馈若落库会扩展数据类型与隐私边界，超出本次仅持久化模型生成 lesson 元数据的授权。评测也从“同技能或同领域即相关”改为显式 `relevant_ids`，同时报告 Top-1、MRR、nDCG、Recall、Precision、误注入率和多样性，避免用偏乐观的单一指标证明自己。

## 十、RAG v2 后续审计：失败上下文与向量化冷启动

实现后复查发现，`_build_query` 虽然接收 `evals`，此前却没有读取它：检索查询只有 `keyword:missing` 等失败 reason 和固定前两个场景，可能遗漏真正失败标准及对应场景。因此新增默认关闭的 `LESSON_FAILURE_CONTEXT`，按 `current_details.eval_id/scenario_id` 精确关联失败 eval 和场景，并把 criterion、question、pass condition、dimension 作为当次查询上下文；它们不进入 lesson store。

另一处瓶颈是未缓存经验逐条 embedding。阿里云百炼的[文本向量同步 API](https://help.aliyun.com/en/model-studio/text-embedding-synchronous-api)明确支持 `TextEmbedding.call(input=batch)`，并给出每批最多 10 条的示例；[RAGFlow 的 DashScope 实现](https://github.com/infiniflow/ragflow/blob/main/rag/llm/embedding_model.py)也采用分批调用并依据 `text_index` 恢复顺序。SkillForge 因此新增 `LESSON_EMBED_BATCH_SIZE`（1–10，默认 1），先对重复文本去重，再批量调用、恢复原顺序并沿用 SQLite 惰性写回；失败仍降级 tag。

## 十一、优化门控补强：暂定胜者与 incumbent 配对复核

RAG 完成后的主循环审计发现：并行候选各评分一次后直接取最大值，会产生典型的“赢家诅咒”——候选越多，偶然高估者越可能成为 winner。`NOISE_FLOOR` 只能过滤固定幅度，`FINAL_CONFIRM` 则到整个流程结束才与最初版本比较，二者都不能证明每轮 challenger 稳定胜过当轮 incumbent。

三个主来源给出了互补依据：

- [GEPA FAQ](https://github.com/gepa-ai/gepa/blob/main/docs/docs/guides/faq.md)采用 minibatch 初筛，只有候选先改善才进入完整验证，说明昂贵复核应只花在 provisional winner 上。
- [SkillOpt](https://github.com/microsoft/SkillOpt/blob/main/README.md)用严格 validation gate 比较 candidate 与 current skill，hard gate 要求严格更高。
- [Optuna WilcoxonPruner](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.WilcoxonPruner.html)强调按相同 step 配对候选观测；这里借鉴“同任务集成对比较”的证据结构，但不在 3–4 个小场景上冒充统计显著性检验。

SkillForge 新增 `CANDIDATE_CONFIRM_RUNS=0..3`，默认 0 保持原调用路径。开启后仅当初评分已经越过 `improvement_threshold + noise_floor` 时，才将暂定胜者与当轮当前版本在同一 scenarios/evals 上成对复评；每一对都必须继续严格胜出，最终状态分取初评和所有 challenger 复评分中的最低值。任何一对打平或失败都将本轮标记为 `confirmation_failed`，候选不保留、不沉淀经验。该门不会接受原本不合格的候选，也不改变单点变异、回归守卫或轮间停止语义。

本轮没有直接引入 held-out split：当前分析阶段默认只生成 3–4 个场景，自动切分会让训练或验证侧只剩 1–2 条，且现有 API 没有独立选择集。配对复核先解决评分噪声；真正的 held-out gate 应在未来同时扩充场景数量并显式区分 train/selection 数据后再做，避免用过小验证集制造虚假的安全感。

## 十二、跨轮同分维度优势：扩展信号，不扩展接受集合

同轮 `TIE_DIMENSION_LESSONS` 只能回答“这轮另一个候选相对已保留 winner，是否在局部维度更好”。它遗漏了另一类有价值信号：某轮可能没有任何总分严格提升，但一个已完成评分的候选与当前 incumbent 总分相同，并显著修复了 incumbent 的弱维度。按照 GEPA 的“signal + gate”取舍，这类候选的局部信号值得学习，但接受门不应因此扩大。

SkillForge 因此把比较范围明确拆成两类：

| `comparison_scope` | 参照物 | 候选范围 | 是否可替换当前版本 |
|---|---|---|---|
| `same_round` | 本轮通过严格提升门并被保留的 winner | 与 winner 总分相同的其他本轮有效候选 | 否 |
| `cross_round` | 截至本轮开始最近一次通过严格提升门被保留的 incumbent | 本轮所有通过守卫且真实完成评分的候选 | 否 |

“最近一次被保留的 incumbent”是比“上一轮版本”更准确的定义：如果连续几轮没有严格提升，上一轮只有被丢弃的临时候选，真正参照物仍是更早那次被保留的版本。由于 SkillForge 始终只有一个父节点且只接受严格提升，该 incumbent 同时也是截至轮首的历史最高总分版本。比较必须冻结轮首的 `current_md`、总分和维度成绩，并在任何本轮状态更新前完成，避免把新 winner 误当成历史参照。

跨轮学习使用独立的 `CROSS_ROUND_TIE_DIMENSION_LESSONS=0`，不会因既有同轮开关为 1 而在升级后静默产生新经验；`CROSS_ROUND_TIE_DIMENSION_MIN_GAIN=0.0` 控制严格维度增益门。总分按现有浮点容差判定同分，维度增益必须严格大于阈值。低于 incumbent 的候选没有同分信号，高于 incumbent 的候选走原来的严格提升和可选配对复核；同分候选不会触发 `CANDIDATE_CONFIRM_RUNS`，也永远不会成为 `current_md` 或得到 `kept=true`。回归守卫、编辑幅度守卫、未完成评分和 confirmation 拒绝均先于经验提取生效。

经验形状遵循“最小可复用元数据”：同一候选的多个优势维度聚合为一条 `tie_dimension` lesson，包含 scope、目标维度和 `dimension_gains`，目标维度取最大 gain（同 gain 按维度名稳定排序），但不保存候选/incumbent 的 SKILL.md、场景、eval/执行原文、输出或 Prompt。候选 ID在 tie 提取流程中只用于本轮 `(candidate_id, dimension)` 去重，不持久化或注入 Prompt；既有 mutation log/API 候选明细不变。同一信号同时命中两类比较时由 `same_round` 优先，避免重复沉淀。跨轮记录的 `score_before == score_after`，明确表达“总分没有进步”，而非制造虚假 gain。

这也要求质量门分工清晰：跨轮经验以自身的严格维度增益阈值判断信号质量，不使用针对总分提升设计的 `LESSON_MIN_GAIN`；若启用了 `LESSON_MIN_FINAL`，它继续作为额外的持久化总分水位。未过持久化水位的元数据仍可进入当次会话的 `round_memory`。因此 lesson 质量判断只决定“是否记住”，不能改变“是否接受”。

为了让跨轮信号真的影响下一轮，首轮仍在 baseline 后准备经验；开关启用时，后续轮在协作式 stop 检查之后、Analyst 之前刷新经验检索。`target_dimension` / `dimension_gains` 继续参与 `quality_diverse` 的弱维度 signal，`comparison_scope` 进入受控 Prompt 白名单；内部 ID与 embedding 仍被剥离。代价是 semantic/hybrid 模式可能在后续每轮增加查询 embedding，rerank 开启时还可能增加重排调用；embedding/检索失败仍降级 tag/轮内记忆，rerank 失败保留原排序。开关关闭时不改变原检索调用次数。

JSONL 可直接承载新字段；SQLite 只用内置 `sqlite3` 增加可空 `comparison_scope` 并兼容旧库，历史 tie lesson 缺少 scope 时按 `same_round` 解读。该迁移没有扩大持久化数据边界，也没有新增依赖、provider 或保留种群。
