# 🔥 SkillForge — 关于这个个人项目

> *"Self-forging agent skills with Qwen."*

SkillForge 是作者的个人项目：把一个开源的「自我改进型 Agent 技能」应用，重塑成带个人标识的技能锻造厂。名字取自锻造（forge）—— 每一次变异都像在砧台上锤打，高温淬炼，只留下真正更硬的版本。

## 个人风格

- **品牌**：青绿渐变（`#22d3ee → #14b8a6`）+ 深色科技感，砧台 Logo（`frontend/src/app/icon.svg`）
- **核心信念**：与其手动调提示词，不如定义成功标准，让 AI 自我进化（致敬 Karpathy 的 autoresearch）
- **工程纪律**：Qwen-only、最小改动、行为不变量优先 —— 三 Agent 角色、严格提升才保留、回归守卫只更严

## 特色功能（相对上游的增量）

| 能力 | 说明 |
|------|------|
| **可配置变异策略池** | 7 种策略模板（原 4 种 + add_reference / rewrite_section / fix_format），Analyst 只在白名单内选择 |
| **多维加权评分** | 按 eval 的 `dimension` 字段分组打分（默认 correctness），无权重时与经典通过率完全一致 |
| **每轮并行变异** | 每轮同时生成多个候选（独立 Assistant 实例），并行复评取最优，严格提升才保留 |
| **回归守卫** | frontmatter / 标题 / 体积完整性检查，命中即拒绝并跳过复评 |
| **协作式停止** | `/api/stop` 轮间生效，不再只是"假停" |
| **过程可视化** | 实时展示 mutation→evaluate→decide 链路、逐维度趋势线、每次变异 diff |
| **示例技能修复** | `/api/examples` 真正读取 `skill-examples/*.zip`，前端动态拉取 |

## 环境与安全

- 仅支持 Qwen-Agent + 阿里云百炼 DashScope；凭据字段恒为 `qwen_api_key`
- 密钥只在组件内存与请求体中流动，绝不落日志 / 会话 / zip / git
- 默认模型 `qwen-plus`，可用 `QWEN_MODEL` 覆盖

## 维护约定

见 [`AGENTS.md`](AGENTS.md)：行为不变量、验证流程（后端单测 + 前端构建 + 非 Qwen 提供商痕迹扫描 + 真实冒烟测试）全部保留并扩展。

---

_2026-08 · SkillForge_
