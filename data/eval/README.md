## `enterprise_tasks.jsonl`（Phase 5 统一企业任务 Benchmark）

- **~490 条企业任务**（10 类 task_type + easy/medium/hard 难度 + 单/双/三/四工具
  组合），是**整个项目正式的统一评测集**，驱动 Phase 5 端到端评测
  （`scripts/run_enterprise_eval.py`）与 Baseline / Ablation；
- 由 `scripts/build_enterprise_tasks.py` 生成，`scripts/validate_enterprise_tasks.py`
  校验；完整说明见 [`README_enterprise_tasks.md`](./README_enterprise_tasks.md)；
- Ground Truth 由**独立** SQL / KB 包含 / 人工声明产生（`provenance.verifier`
  记录来源），**绝不取自** Router / Verification / Answer 任一系统输出；
- 本文件不属于上列任何阶段性实验集；阶段性结果不得替代它。

**边界**：正式 430+ 条 Benchmark 已落地；数值型业务结论来自**合成**业务种子
数据，只用于验证系统流程与 GT 一致性，不得当作真实经营指标对外引用。

## `hybrid_eval.jsonl`

- **10 个 query**，覆盖：精确关键词 / 中文短问题 / 中文长问题 / 多关键词 /
  同义表达 / 数字规则 / 跨句语义（每类至少一条）；
- 用于 **Phase 2.2 Hybrid RAG**（Dense + BM25 + RRF）的局部检索实验，
  语料为 35-chunk 企业知识库（`data/knowledge_base/*.md`）；
- 每条记录含 `id / query / type / expected_chunk_ids / source / title /
  category`；其中 **`expected_chunk_ids` 为人工核验的 Ground Truth**
  （与 `scripts/build_kb.py` 的 chunk_id 对齐，先于实验结果固定，
  不因结果而修改）；
- 由 `scripts/build_hybrid_eval.py` 生成；
- 检索指标（Recall@1/3/5、MRR）由 `scripts/run_hybrid_eval.py` 自动计算，
  结果写入 `artifacts/evaluation/`（本地保留，不进 Git）。

**边界**：这是 Phase 2.2 的阶段性实验集，**不属于**正式 430 条评估集；
不得把 10-query 结果当作最终系统性能对外引用。

## `retrieval_eval.jsonl`

- **56 个 query**，覆盖 8 类：精确关键词 / 同义表达 / 多关键词 / 长自然语言 /
  跨句语义 / 数字规则 / 多实体 / 易混淆问题；
- 用于 **Phase 2.3 Reranker** 的局部检索实验，语料为扩大后的
  27-doc / 190-chunk 企业知识库（`data/knowledge_base/*.md`）；
- 每条记录含 `id / query / type / difficulty (easy|medium|hard) /
  expected_chunk_ids / source / category`；其中 **`expected_chunk_ids`
  为人工核验的 Ground Truth**（与 `scripts/build_kb.py` 的 chunk_id
  对齐，先于实验结果固定，不因结果而修改）；
- 由 `scripts/build_retrieval_eval.py` 生成；
- 检索指标（Recall@1/3/5、MRR、NDCG@5）由 `scripts/run_reranker_eval.py`
  自动计算（对比 Dense / BM25 / Hybrid / Hybrid+Rerank 四种模式），
  结果写入 `artifacts/evaluation/`（本地保留，不进 Git）。

**边界**：这是 Phase 2.3 的阶段性实验集，**不属于**正式 430 条评估集；
不得把 56-query 结果当作最终系统性能对外引用。
