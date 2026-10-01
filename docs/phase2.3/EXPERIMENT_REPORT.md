# Phase 2.3 Experiment Report

> **定位**：Phase 2.3 的**正式阶段性实验记录**，用于支撑毕业论文的
> "实验设计 / 实验结果 / 消融"章节。
> **数据来源**：本文所有指标均生成自
> [`artifacts/phase2.3/retrieval_eval_summary.json`](../../artifacts/phase2.3/retrieval_eval_summary.json)
> （由 `scripts/run_retrieval_eval.py` 真实运行产出，**不进 Git**，本地保留）。
> **本文是"由实验 artifact 生成的记录"**，不手工维护孤立数字；
> 逐 query 明细见 `artifacts/phase2.3/retrieval_eval_results.json`。

---

## 1. Goal

在已完成 Hybrid 召回（Dense + BM25 + RRF）的前提下，验证
**BGE-Reranker-v2-M3 是否能进一步改善最终 Top-K 排序质量**。
对比 4 种检索模式：Dense / BM25 / Hybrid / Hybrid + Reranker，
统一在同一语料、同一 query 集、同一 final_k 下评测。

**这是局部检索实验**，不等价于最终 487 条 Evaluation Benchmark
（Phase 5 构建）。

## 2. Dataset

- **语料**：企业知识库，**27 篇文档 / 190 chunks**（`data/knowledge_base/*.md`），
  全部标注 `synthetic: true`，由 `scripts/build_kb.py` 切分、BGE-M3 嵌入
  后写入 Milvus `enterprise_knowledge`（`entity_count = 190`）。
- **query**：**56 条**（`data/eval/retrieval_eval.jsonl`），
  覆盖 8 类：`exact_keyword / synonym / multi_keyword / long_question /
  cross_sentence / numeric_rule / multi_entity / confusable`，
  并带 `difficulty ∈ {easy, medium, hard}`（easy 9 / medium 25 / hard 22）。
- **Ground Truth**：人工核验的 `expected_chunk_ids`，**独立于任何检索
  系统输出**（不得来自 Dense / BM25 / Hybrid / Reranker），固定先于实验。

## 3. Retrieval Pipeline

| 通道 | 实现 | 位置 |
|---|---|---|
| Dense | BGE-M3（1024-dim）→ Milvus | `src/core/vector_store.py` |
| BM25 | 进程内 Okapi BM25（k1=1.5, b=0.75），与 Milvus 同一批 chunk | `src/core/bm25_store.py` |
| Fusion | RRF（k=60，基于 rank） | `src/core/retrieval_types.py` |
| Reranker | BGE-Reranker-v2-M3 cross-encoder，**仅重排候选** | `src/core/reranker.py` |

**Reranker 是重排阶段（reranking stage），不是召回阶段（retrieval
stage）**：它接收 RRF 融合后的 `candidate_k` 条候选，重新排序后取
`final_k`；不参与 Dense / BM25 原始召回，也不扫描整个知识库。

## 4. Compared Methods

1. **Dense** — BGE-M3 + Milvus，Top-5
2. **BM25** — 关键词召回，Top-5
3. **Hybrid** — Dense + BM25 → RRF，Top-5
4. **Hybrid + Reranker** — Dense + BM25 → RRF → `candidate_k=20` →
   BGE-Reranker-v2-M3 → `final_k=5`

四模式使用**相同 query、相同语料、相同 final_k**；Reranker 不参与召回。

## 5. Metrics

| 指标 | 定义 |
|---|---|
| Recall@K | \|Relevant ∩ Retrieved@K\| / \|Relevant\|（单条 query），对 56 条求均值 |
| MRR | 所有 query 的 1/rank(首个相关结果) 均值；无相关命中记 0 |
| NDCG@5 | 二值相关（`chunk ∈ expected_chunk_ids` ? 1 : 0）的标准 DCG@5 / IDCG@5 |
| Latency | 单 query 端到端检索耗时（ms）；工程指标，不替代检索质量 |

## 6. Overall Results（真实运行）

| Method | Recall@1 | Recall@3 | Recall@5 | MRR | NDCG@5 |
|---|---:|---:|---:|---:|---:|
| Dense | 0.5908 | 0.8259 | 0.8988 | 0.9062 | 0.8657 |
| BM25 | 0.6473 | 0.8705 | 0.9077 | 0.9345 | 0.8965 |
| Hybrid | 0.6622 | 0.8735 | 0.9286 | 0.9500 | 0.9120 |
| **Hybrid + Reranker** | **0.6711** | 0.8646 | 0.9271 | **0.9613** | 0.9087 |

**Reranker 相对 Hybrid 的差值**：Recall@5 **−0.0015**、MRR **+0.0113**、
NDCG@5 **−0.0033**。

- 56/56 query 全部完成，**0 条失败**。
- 所有指标无 NaN；失败 query（若出现）会保留在分母中，不静默删除。

## 7. Query Type Analysis（Recall@5 / MRR / NDCG@5，摘录）

| Method | 最强类型 | 最弱类型 |
|---|---|---|
| Dense | confusable (0.95 / 0.80 / 0.83) | cross_sentence (0.83 / 0.83 / 0.83) |
| BM25 | confusable (1.00 / 0.95 / 0.96) | cross_sentence (0.83 / 0.75 / 0.78) |
| Hybrid | confusable (1.00 / 1.00 / 0.99) | exact_keyword (0.82 / 1.00 / 0.85) |
| Hybrid + Rerank | confusable (1.00 / 1.00 / 1.00)、long_question (1.00) | multi_entity (0.92 / 0.92 / 0.85) |

类型级波动较大；**不能据此断言 Reranker 在所有类型上均有效**
（如 `synonym` 类 MRR 由 Hybrid 的 0.92 降至 Rerank 的 0.83）。

## 8. Difficulty Analysis（Recall@5）

| Method | easy (n=9) | medium (n=25) | hard (n=22) |
|---|---:|---:|---:|
| Dense | 0.8611 | 0.9133 | 0.8977 |
| BM25 | 0.8611 | 0.8933 | 0.9432 |
| Hybrid | 0.8611 | 0.9267 | 0.9583 |
| Hybrid + Rerank | 0.8611 | **0.9333** | 0.9470 |

- Reranker 在 **medium** 难度上 Recall@5 小幅改善（0.9267 → 0.9333）；
- 在 **hard** 难度上 Recall@5 轻微下降（0.9583 → 0.9470），**未表现出
  稳定净收益**。

## 9. Latency Analysis（CPU，per query）

| Method | Avg(ms) | P50(ms) | P95(ms) |
|---|---:|---:|---:|
| Dense | 178 | 182 | 201 |
| BM25 | 4 | 4 | 5 |
| Hybrid | 180 | 187 | 201 |
| Hybrid + Rerank | 5713 | 5731 | 6515 |

Reranker 在 CPU 上对 20 条候选重排约 5.7s/query，为**已验证的稳态**
耗时（非 warmup）；正式 GPU 环境下预计显著降低。

## 10. Failure Cases

由 `scripts/run_retrieval_eval.py` 自动挖掘的 4 条代表性
跨模式分歧 / rerank 重排案例（完整见
`artifacts/phase2.3/retrieval_failure_cases.md`）：

1. **rq-35（hard, cross_sentence）**：Dense 与 BM25 单通道均未召回，
   Hybrid RRF 将 `kb-d96806441a` 拉至第 5；Reranker 进一步提升至第 3。
2. **rq-10（medium, long_question）**：Dense / BM25 各命中不同相关 chunk，
   Hybrid 相关 chunk 在第 2；Reranker 提至第 1。
3. **rq-19（medium, numeric_rule）**：BM25 首位命中，Reranker 维持首位。
4. **rq-49（medium, synonym）**：Hybrid 已第 1，Reranker 反而降至第 2
   ——**Reranker 在此 query 上产生负向影响**（如实保留）。

## 11. Findings（仅限本实验数据集）

1. **Dense 与 BM25 均能完成基本企业知识检索**（Recall@5 ≈ 0.90）。
2. **Hybrid 的 Recall@5（0.9286）与 NDCG@5（0.9120）高于单一
   Dense / BM25**——RRF 融合在本数据集上有稳定召回收益。
3. **Reranker 提高了 MRR（0.9500 → 0.9613）**，说明首个相关结果的位置
   有所改善（排序改善）。
4. **Reranker 未提高 Recall@5（0.9286 → 0.9271），也未提高 NDCG@5
   （0.9120 → 0.9087）**。
5. 因此当前实验更支持：**"Reranker 主要改善候选排序，而非扩大召回
   覆盖率"**（Hybrid 已在 Top-5 内召回候选本身较完整）。
6. **在 hard query 上 Reranker 未表现出稳定净收益**（Recall@5 略降）。
7. **CPU 环境下 Reranker 显著增加延迟**（~5.7s/query）。

> 以上均为**实验现象总结**，不是对 Reranker 模型的普遍性结论。
> 所有结论限定于"在本实验数据集 / 当前 190-chunk synthetic enterprise
> corpus 上"。

## 12. Limitations

- **语料规模**：190 chunk 偏小，Reranker 在更大语料上的表现不可外推；
- **query 数量**：56 条，单条异常对均值影响偏大；
- **合成企业知识**：非真实企业内部文档，结论仅限合成语料；
- **设备**：CPU 推理（~5.7s/query），正式 GPU 环境下延迟会显著降低；
- **实验范围**：仅检索层，未接入 Answer Generation / Verification；
- 本实验是 Phase 2.3 局部检索实验，**不等价于**最终 487 条
  Evaluation Benchmark（第 1 节，Phase 5 构建）。

## 13. Reproducibility

| 项 | 取值 |
|---|---|
| Python 版本 | 3.11.9 |
| Embedding 模型 | BGE-M3（本地 `EMBEDDING_MODEL`，1024-dim） |
| Reranker 模型 | BGE-Reranker-v2-M3（本地路径，经 `RERANKER_MODEL` 配置项 / `--rerank-model` 传入） |
| 向量库 | Milvus（`enterprise_knowledge`，HNSW / IP） |
| 设备 | CPU（embedding 与 reranker；本机无 CUDA，自动降 CPU 而非静默失败） |
| 语料规模 | 27 docs / 190 chunks |
| query 数 | 56 |
| candidate_k / final_k | 20 / 5 |
| RRF k | 60 |
| 实验脚本 | `scripts/run_retrieval_eval.py`（本地模型路径通过 `--rerank-model` 指定，**不写死在代码或文档**） |
| 结果 artifact | `artifacts/phase2.3/`（gitignore，本地保留） |

> 本地模型路径是**环境变量 / 配置项**（`RERANKER_MODEL`），不属于通用
> 路径；复现时替换为本机实际路径即可。逐 query 结果、失败案例与
> 延迟明细均保留在 `artifacts/phase2.3/`，不复制到本文。
