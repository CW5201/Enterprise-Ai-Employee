# 评估设计（EVALUATION）

> 状态：Phase 2.2 已更新（2026-09-30）。
> 本文档定义评估协议。正式系统级指标在 Phase 5 由
> `src/evaluation/runner.py` 实际运行产出，当前不填写未经验证的系统级数值。
> Phase 2.2 完成了 Hybrid RAG 的**局部检索实验**（10 query），结果如实记录于
> 第 14 节，属于阶段性实验，不等价于最终 430 条 Benchmark。

---

## 1. Eval Dataset

- 文件：`data/eval_dataset.jsonl`（当前为空，Phase 5 构建）；
- 规模目标：约 430 条；
- 类别：Information Query / Knowledge QA / Text-to-SQL / Data Analysis /
  Multi-source Reasoning / Complex Agent Task；
- 字段定义与构建规则见 [`DATASET.md`](DATASET.md) 第 11 节。

评估集加载与版本校验由 `src/evaluation/dataset.py` 实现，要求：
- 校验必需字段与枚举合法性；
- 校验 `evidence_id` 唯一且稳定；
- 记录数据集版本，实验结果必须带版本信息。

## 2. Benchmark

| 层级 | 数据集 | 目的 |
|---|---|---|
| 主评估 | Enterprise AI Employee Task Dataset（约 430 条） | 完整系统、Baseline、消融 |
| 外部对照 | Spider, BIRD | Text-to-SQL 能力对照 |
| 外部对照 | HotpotQA | 多跳检索与证据支持对照 |
| 外部对照 | BFCL, ToolBench | 工具调用能力对照 |
| 外部对照 | GAIA | 复杂 Agent 任务对照 |

规则：
- 外部 Benchmark 只按其许可与访问规则使用，结果需注明版本；
- 外部 Benchmark 用于**验证方法不是只在自建集上有效**，不替代自建集；
- 不得把外部 Benchmark 的答案并入提示材料。

## 3. Task Success Rate

**定义**：任务完成且结论达到判定标准的比例。

判定标准按类别区分（需在 Phase 5 固定并写入配置，评估时不得临时调整）：

| 类别 | 判定标准 |
|---|---|
| Information Query | 结果值与 Ground Truth 一致 |
| Knowledge QA | 结论与参考要点一致且引用到正确文档 |
| Text-to-SQL | 执行结果集与参考结果一致（顺序无关） |
| Data Analysis | 关键统计量与结论方向一致 |
| Multi-source Reasoning | 使用了必要的多个来源，且结论与 Ground Truth 一致 |
| Complex Agent Task | 交付物（答案/图表/报告）满足任务要求的必需要素 |

**统计方式**：总体 + 按类别 + 按难度分组报告；只报总体会掩盖路由与融合的差异。

## 4. Routing Accuracy

**定义**：系统选择的工具集合与参考工具集合 `expected_tools` 的一致程度。

报告以下三个数值，避免单一口径掩盖问题：
- **Exact Match**：工具集合完全相同；
- **Over-selection**：多选了工具（浪费，且可能引入噪声证据）；
- **Under-selection**：漏选工具（能力缺失，通常直接导致任务失败）。

## 5. Retrieval Recall@K

**定义**：Top-K 检索结果中覆盖参考证据的比例。

- K 取值需固定并记录（默认报告 K = 1, 3, 5, 10）；
- 需分别报告 **Dense / BM25 / Hybrid / Hybrid+Rerank** 四个配置，
  否则无法说明混合检索与 Rerank 的贡献；
- 判定"命中"需明确粒度：文档级命中与片段级命中分别报告（片段级更严格）。

## 6. SQL Execution Accuracy

**定义**：生成的 SQL 在数据库中执行后，结果与参考结果一致的比例。

- 采用**执行结果比对**而非字符串比对（字符串比对会低估正确性）；
- 结果集比对需处理列顺序差异与行顺序差异；
- 同时记录两类失败：**执行失败**（语法/字段错误）与**结果不符**（语义错误），
  两者成因不同，必须分开统计。

## 7. Evidence Support Rate

**定义**：生成的 claim 中被证据支持的比例。

```
Evidence Support Rate = supported claims / total claims
```

- claim 判定类别：`supported` / `partially_supported` / `unsupported` / `contradicted`；
- `partially_supported` 在报告中单列，不与 `supported` 混同；
- 无依据结论比例（`unsupported + contradicted` 占比）也需单列，
  这是创新点 3 的核心受益指标；
- 判定由 `src/nodes/verification.py` 产出，评估侧抽样人工复核并记录一致性。

## 8. Average Latency

**定义**：端到端平均耗时（毫秒）。

- 必须同时报告**分项耗时**：意图理解 / 路由 / 各工具 / 融合 / 验证；
- 报告 P50 / P90 / P95，只用平均值会掩盖长尾；
- 必须记录运行环境（CPU/GPU、模型是否本地、数据库是否同机），
  否则不同实验之间的延迟不可比较；
- 说明局限：单机环境的延迟不能外推到生产部署。

## 9. Baseline

| 编号 | 系统 | 配置要点 |
|---|---|---|
| Baseline 1 | LLM | 无检索、无工具 |
| Baseline 2 | LLM + RAG | 仅文档检索 |
| Baseline 3 | Hybrid RAG + SQL | 文档 + 结构化数据 |
| Baseline 4 | RAG + SQL + KG | 加图谱，固定流程，无验证 |
| Ours | 完整系统 | 动态路由 + 多源融合 + 结论验证 |

**公平性要求**：同一 LLM、同一 Embedding / Reranker、同一评估集版本、
同一 Prompt 版本记录、同一运行环境；只允许系统结构不同。

## 10. Ablation

| 实验 | 变更 | 与 Ours 的关系 |
|---|---|---|
| A | 移除 Task-Adaptive Routing（改为固定 RAG→SQL→Answer） | 仅此一处不同 |
| B | 移除 Knowledge Graph | 仅此一处不同 |
| C | 移除 Claim-Evidence Verification | 仅此一处不同 |

报告要求：
- 每个消融实验都需给出指标变化与**分类型任务的变化**（例如移除 KG 主要影响 relation_query 与 multi_source_reasoning）；
- 需说明该模块带来的是"完成率提升"还是"效率提升"还是"可信度提升"，
  三者对应的指标不同（Task Success Rate / Latency / Evidence Support Rate）。

## 11. Error Analysis

流程：
1. 收集所有失败与低分样本；
2. 按 [`RESEARCH.md`](RESEARCH.md) 第 9 节的七类归因打标签；
3. 统计每类错误占比与分布（按任务类别分组）；
4. 对占比最高的 2–3 类做案例分析，登记到 [`FAILURE_HANDBOOK.md`](../FAILURE_HANDBOOK.md)；
5. 针对高频错误提出改进，并加入回归测试集，防止改进引入回退。

报告要求：
- 错误案例必须附真实输入与真实输出，不允许编造；
- 必须报告**验证模块的漏判率与误判率**，这是评估验证机制本身是否可靠的关键；
- 必须说明未能解释的失败样本数量。

## 12. 评估框架实现

| 文件 | 职责 |
|---|---|
| `src/evaluation/dataset.py` | 评估集加载与版本校验 |
| `src/evaluation/metrics.py` | 六项指标的计算实现 |
| `src/evaluation/runner.py` | 运行 Baseline / Ours / 消融并落盘结果 |
| `src/evaluation/reporter.py` | 生成 Markdown / HTML 报告 |
| `scripts/run_eval.py` | 命令行入口 |

输出要求：
- 每次运行产出一个结果目录：`outputs/eval/<run_id>/`，包含配置快照、逐条结果、指标汇总、错误归因；
- 报告必须包含：数据集版本、配置版本、Prompt 版本、运行环境、指标数值与置信区间（若适用）；
- **禁止手填数值**：报告中的每个数字都必须能追溯到该目录下的逐条结果。

## 13. CI 评估门禁（Phase 5 启用）

- `.github/workflows/ci.yml` 中的 evaluation gate 当前为注释状态；
- 启用后，门禁与**已存档的 baseline 结果**比较，而不是写死阈值；
- 门禁失败需给出是哪一项指标、哪个任务类别退化的具体信息。

## 14. Phase 2.2 Hybrid Retrieval Experiment

> **定位**：本节是 Phase 2.2 的**局部检索实验**（Dense vs BM25 vs Hybrid），
> 用于验证混合检索链路是否正确接通，**不等于**第 1 节的 430 条
> 正式 Evaluation Benchmark（正式集在 Phase 5 构建）。
> 所有指标由 `scripts/run_hybrid_eval.py` 真实运行产生，**禁止手填**；
> 结果如实记录，包含对"Hybrid 未超过 Dense"的负向结论。

### 14.1 Experimental Setup

| 项 | 取值 |
|---|---|
| 语料 | 当前企业知识库，**35 chunks**（`data/knowledge_base/*.md`） |
| Queries | **10**（`data/eval/hybrid_eval.jsonl`） |
| Top-K | **5**（三模式同一 top_k） |
| Retrieval Modes | Dense / BM25 / Hybrid |
| Embedding | **BGE-M3**（真实模型，1024 维） |
| Vector DB | **Milvus**（`enterprise_knowledge` collection） |
| 关键词召回 | **BM25**（in-process，与 Milvus 同一批 chunk，`chunk_id` 对齐） |
| Fusion | **RRF**，k = 60（基于 rank，不直接相加原始 score） |
| 生成脚本 | `scripts/run_hybrid_eval.py` |

三模式在**同一 query、同一语料、同一 top_k** 下对比，公平性由 runner 保证；
hybrid 内部融合窗口（更宽的候选池）已在输出中记录。

### 14.2 Ground Truth

- 10 个 query 使用当前知识库人工核验建立 `expected_chunk_ids`，写入
  `data/eval/hybrid_eval.jsonl`（每条含 `query / expected_chunk_ids /
  source / title / category`）；
- **Ground Truth 不因实验结果而修改**——它先于结果固定，是判定召回的基准；
- 生成脚本为 `scripts/build_hybrid_eval.py`。

### 14.3 Metrics

- **Recall@1 / Recall@3 / Recall@5**：Top-K 命中 `expected_chunk_ids` 的比例；
- **MRR**：首个命中 chunk 的 1/rank。

### 14.4 Results（真实运行）

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| Dense | 0.3833 | 0.8500 | 0.9500 | 0.9500 |
| BM25 | 0.4500 | 0.7667 | 1.0000 | 0.9500 |
| Hybrid | 0.3833 | 0.8500 | 0.9500 | 0.9500 |

逐 query 明细：`artifacts/evaluation/phase2.2_hybrid_results.json`（JSON）
与 `.csv`（本地保留，不进 Git）。

### 14.5 Analysis（如实记录）

1. **Dense 已具备较高 Recall@3/5**（0.85 / 0.95）——BGE-M3 在制度类
   小语料上的语义召回本身就较强。
2. **Hybrid 与 Dense 指标完全相同**（Recall@1=0.3833、@3=0.85、@5=0.95、
   MRR=0.95），在本次小规模实验上 **Hybrid 没有表现出对 Dense 的增益**。
   这是**有效实验结果，不是失败**，必须如实保留。
3. **BM25 的 Recall@5（1.0）高于 Dense（0.95）**，说明关键词匹配对
   当前制度类小语料有一定价值（专有名词、编号等精确词命中）。
4. **当前语料仅 35 chunks、query 区分度高**，样本量与多样性都不足以证明
   Hybrid 在一般企业知识库上普遍有效，**不得据此下"Hybrid 显著优于 Dense"
   的结论**。
5. 下一阶段**扩大语料规模并引入 Reranker（Phase 2.3）后继续评估**，
   届时在更大、更难的语料上重测 Hybrid vs Dense。

### 14.6 Experimental Limitation

- 本实验是 **Phase 2.2 的局部检索实验**（10 query / 35 chunk），
  **不等价于**最终 430 条 Evaluation Benchmark（第 1 节，Phase 5 构建）。
- 语料规模小、query 少，指标对单条结果敏感，**不能把 10-query 结果外推为
  最终系统性能**；正式系统级指标仍以 Phase 5 的真实运行为准。
- 单点 MRR=0.95 在三模式上相同，主要由 query 区分度高导致，非模型能力的
  全面指标。
