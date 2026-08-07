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
