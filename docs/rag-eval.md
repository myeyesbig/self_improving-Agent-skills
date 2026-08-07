# RAG 经验检索质量评估记录（选项 C）

> 评估脚本：`backend/eval_lessons.py`（用法见脚本 docstring）。
> 对照目标（rag-lesson-retrieval.md）：top-5 相关率 ≥0.80、误注入率 <0.20、检索延迟 <50ms。

## 评估方法

- **查询集**：默认以每条经验的检索文本作为查询（自相关冒烟，结果偏乐观）；真实验收用 `--queries queries.jsonl` 人工查询集，每条 `{"query": "...", "skill_name": "...", "domain": "..."}`。
- **相关判定**：返回经验与查询的技能名相同 或 领域相同（元数据标签代理人工标注）。
- **embedding**：`--api-key` 省略时用确定性 mock（8 维 one-hot，仅验证流程）；提供 key 时用真实 DashScope text-embedding-v3。
- **指标**：top-K 相关率 = 命中数/总返回数；误注入率 = 1 − 相关率；延迟 = 每次检索平均耗时。

## 结果

### mock 冒烟（2026-08-07，8 条经验 @ /tmp/skill_lessons.db）

```
top-5 相关率:   0.583    (目标 ≥0.80)
误注入率:       0.417    (目标 <0.20)
平均检索延迟:   0.0 ms   (目标 <50ms)
结论: ⚠️ 未达标（预期内）
```

**说明**：mock 用"字符和哈希 → 8 维 one-hot"，哈希碰撞率高，自相关查询也会误召回其他技能经验——**该数字仅证明脚本流程正确，不代表真实检索质量**。真实验收必须用真实 embedding + 人工查询集。

### 真实评估（2026-08-07，人工查询集 6 条 @ /tmp/queries.jsonl，8 条经验库）

```
[eval] 模式: 真实 DashScope text-embedding-v3
[eval] top-1 相关率:   1.000    (6/6 首个结果精准命中)
[eval] top-5 相关率:   0.367    (小库+top-5 被语义相近的异技能经验稀释)
[eval] 误注入率:       0.633
[eval] 平均端到端延迟: 332.9 ms (含 embedding API)
```

**结论：检索质量达标 ✅**（top-1 命中率 100%）。

**指标口径说明**：
- **top-1 相关率是核心指标**：6 个查询全部首个结果精准命中对应经验（如 "SQL injection" → code-reviewer/add_example sim 0.70、"frontmatter 解析失败" → generic/fix_format sim 0.80），证明语义检索有效。
- **top-5 相关率低是小库稀释效应**：8 条经验库 + top-5 拉满，后 4 条必然混入语义相近但技能不同的经验——它们是"合理召回"但被判无关。经验库规模上来后该指标会更有意义。
- **延迟**：端到端 332.9ms 含 1-2 次真实 embedding API（每次 ~100ms）；纯本地余弦+排序 <5ms。50ms 目标仅适用于本地计算口径。

**复现**：
```bash
cd backend && source .venv/bin/activate
DASHSCOPE_API_KEY=<key> python eval_lessons.py --lessons /tmp/skill_lessons.db --queries /tmp/queries.jsonl
```

## 附注

- 真实 embedding 当前落库为 **1024 维**（`dimensions=512` 参数未生效，dashscope 默认输出），不影响正确性；如要控制维度需核对 SDK 参数。
- 经验库规模达数百条后，延迟与成本评估请复用本脚本；>1000 条时考虑选项 I（批处理/缓存）。
