# RAG 经验检索质量评估记录（目标 5）

评估脚本为 `backend/eval_lessons.py`。它不接受、读取或输出 API key，使用固定词法哈希向量与显式相关标签，保证 classic / quality_diverse 对照可离线复现。真实模型连通性另走 API 请求体 smoke test。

## 评测口径

- 固定经验集：`backend/fixtures/lesson_eval_lessons.jsonl`，10 条，包含高/低收益、重复、跨领域与弱维度干扰项。
- 固定查询集：`backend/fixtures/lesson_eval_queries.jsonl`，6 条，每条用 `relevant_ids` 明确标注相关经验，不再把“同技能或同领域”一律当作正确。
- 排序指标：Top-1 accuracy、MRR、nDCG@K、Recall@K、Precision@K 和 Misinjection@K。
- 上下文指标：策略多样性与经验文本的平均两两差异。
- 延迟：只统计本地检索；不混入网络 embedding/rerank 延迟。

复现命令：

```bash
cd backend
.venv/bin/python eval_lessons.py \
  --lessons fixtures/lesson_eval_lessons.jsonl \
  --queries fixtures/lesson_eval_queries.jsonl \
  --top-k 3
```

## 2026-08-12 结果

| 指标 | classic | quality_diverse | 差值 |
|---|---:|---:|---:|
| Top-1 accuracy | 1.000 | 1.000 | +0.000 |
| MRR | 1.000 | 1.000 | +0.000 |
| nDCG@3 | 0.936 | 1.000 | +0.064 |
| Recall@3 | 0.917 | 1.000 | +0.083 |
| Precision@3 | 0.333 | 0.389 | +0.056 |
| Misinjection@3 | 0.667 | 0.611 | -0.056 |
| Strategy diversity | 0.944 | 0.889 | -0.056 |
| Pairwise diversity | 0.851 | 0.903 | +0.052 |

结论：在固定小样本上，新管线保持 Top-1/MRR 不退化，同时把 nDCG@3 与 Recall@3 提升到 1.0，平均文本多样性也更高。策略多样性略降是因为某些查询的两个正确经验恰好使用同一策略；这项指标只作诊断，不作为相关性门槛。Precision 较低仍反映 top-3 在多数单答案查询中的数学上限，因此主要验收指标采用 MRR、nDCG 与 Recall。

脚本在 compare 模式下要求新版 MRR 和 nDCG 均不低于 classic，否则以非零状态退出。该 fixture 是可重复的回归基线，不替代真实历史库上的人工标注评估；经验规模扩大后，应另建代表性查询集并继续使用相同指标。

## 后续性能与查询质量守卫

- `TestLessonFailureContextAndBatching` 验证默认查询形状和逐条 embedding 行为不变。
- 开启失败上下文后，查询只选取实际失败的 eval 定义和对应场景，不再误用“前两个场景”。
- 批量向量化会先对相同 lesson 文本去重，再按 `LESSON_EMBED_BATCH_SIZE` 分批，并按输入顺序恢复结果。
- 任一批次异常仍由 `_prepare_lessons` 捕获并降级 tag，不影响优化主流程。
