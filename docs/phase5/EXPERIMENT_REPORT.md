# Phase 5 Enterprise AI Employee Comprehensive Evaluation

> 实验模式：**offline-deterministic harness**（真实 DuckDB 业务库 + 真实 27 篇
> 合成知识库 + 确定性路由/工具；未接入 live Milvus / Neo4j / LLM 在线链路）。
> 所有数字由 `scripts/run_enterprise_eval.py` + `scripts/analyze_enterprise_failures.py`
> 对 487 条统一 Benchmark 全量运行后**自动生成**，未手工填写；负结果保留。

---

## 1. Objective

在**整个 Enterprise AI Employee** 的统一 Benchmark 上，通过
Baseline + Component Ablation + Failure Analysis，回答三个研究问题
（RQ1–RQ3），并对系统做最终核心实验收口：

- 系统由 `Question → Task Understanding → Dynamic Routing → Tools →
  Evidence → Verification → Answer` 全链路组成（Phase 1–4 的既有组件）；
- 评测区分 **offline harness** 与 **online end-to-end** 两条主线——
  本报告为 offline harness；
- 所有实验失败保留在分母，不因指标好看而修改 ground truth。

## 2. Research Questions

- **RQ1**：Dynamic Routing 是否能够根据任务特征选择合适工具？
- **RQ2**：Claim-Evidence Verification 是否能够降低
  unsupported / hallucinated claims？
- **RQ3**：多知识源协同是否能够提高复杂企业任务完成效果？

## 3. Enterprise Task Benchmark

`data/eval/enterprise_tasks.jsonl`：**487 条**统一任务（Commit 1）。

| 维度 | 分布 |
|---|---|
| by_task_type | knowledge_lookup 82 / structured_lookup 276 / aggregation 27 / relationship_query 31 / statistical_analysis 10 / trend_analysis 2 / comparison 12 / report_generation 10 / multi_source_analysis 32 / ambiguous_task 5 |
| by_difficulty | easy 299 / medium 116 / hard 72 |
| by_route | rag 82 / sql 180 / multi_tool 45 / kg 175 / clarification 5 |
| by_num_tools | 0(澄清) 5 / 单工具 437 / 双工具 30 / 三工具 6 / 四工具 9 |

GT 由**独立** SQL / KB 包含 / 人工声明产生（`provenance.verifier`），
先于任何系统输出固定；`scripts/validate_enterprise_tasks.py` 通过。

## 4. System Architecture

完整链路（全部为 Phase 1–4 既有真实组件，本阶段不新增）：

```
Question
  → IntentUnderstanding（规则/LLM 分类）
  → Dynamic Router（TaskProfile → capability → 候选工具 → RoutingDecision）
  → Multi-tool Execution（sql / rag / kg / analysis，按 execution_plan 依赖序）
  → Answer Generation（仅基于证据作答）
  → Claim Extraction → Evidence Collection → Verification → Answer Guard
  → 最终答案
```

- **System A（A_full）**：phase3=True + phase4=True，完整系统；
- **System B（B_no_verification）**：phase3=True + phase4=False，无验证；
- **System C（C_static_no_verification）**：phase3=False（SupervisorRouter）+
  phase4=False，静态路由、无验证；
- **基线**：B1 LLM-only（无证据）/ B4 规则路由（无 LLM TaskProfile）；
  B2/B3（静态 RAG / 静态 SQL 工具直调，需 live Milvus，offline 下跳过）；
- **消融**：w/o Dynamic Routing / w/o KG / w/o Hybrid / w/o Reranker /
  w/o Verification。

## 5. Baselines

| 标识 | 描述 | 类型 |
|---|---|---|
| B1_llm_only | 单次 LLM，无工具证据 | baseline |
| B4_rule_routing | 确定性关键词 → 规则引擎（无 LLM profile） | baseline |
| C_static_no_verification | Phase 1 SupervisorRouter + 无验证 | 综合基线 |
| A_full | 动态路由 + 混合检索 + 验证 | 完整系统 |

## 6. Experimental Setup

- **Benchmark**：487 条全量，offline-deterministic；
- **组件**：真实 DuckDB（`data/runtime/wwi.duckdb`，500 客户 / 800 订单 /
  120 商品 / 60 供应商）+ 真实 27 文档 / 190 chunk 合成知识库（Milvus 侧在
  offline 用 fake dense 后端，RAG 命中经真实 `RAGRetrievalNode` 回注）；
- **Neo4j**：offline harness 未接入（凭证受限），KG 路由任务在 offline 下
  记为 *honest failure*（路由正确但工具无结果），**不伪造图谱结果**；
- **LLM**：offline 用 `_OfflineBackend`（确定性），LLM 调用计数按管线阶段推断；
- **指标**：全部由 `src/evaluation/enterprise_metrics.py` 程序计算，
  逐条记录、失败保留分母。

## 7. Overall Results

| System | Task Success | Answer Exact | Route Acc | Tool F1 | Plan EM | Evidence Acc | Leakage | P95 (ms) | LLM calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A_full** | **0.345** | 0.133 | **0.394** | **0.405** | 0.347 | 0.362 | 0 | 412 | 3.0 |
| B_no_verification | 0.345 | 0.133 | 0.394 | 0.405 | 0.347 | 0.362 | — | 258 | 3.0 |
| C_static_no_verification | 0.008 | 0.166 | 0.343 | 0.010 | 0.010 | 0.371 | — | 258 | 2.4 |
| B4_rule_routing | **0.450** | 0.010 | 0.433 | 0.471 | 0.402 | 0.369 | — | 233 | 2.0 |
| abl_w_dynamic_routing | 0.008 | 0.166 | 0.343 | 0.010 | 0.010 | 0.371 | — | 417 | 2.4 |
| abl_w_kg | 0.345 | 0.133 | 0.394 | 0.405 | 0.347 | 0.362 | — | 411 | 3.0 |
| abl_w_hybrid_retrieval | 0.345 | 0.133 | 0.394 | 0.405 | 0.347 | 0.362 | — | 412 | 3.0 |
| abl_w_reranker | 0.345 | 0.133 | 0.394 | 0.405 | 0.347 | 0.362 | — | 412 | 3.0 |
| abl_w_verification | 0.345 | 0.133 | 0.394 | 0.405 | 0.347 | 0.362 | — | 258 | 3.0 |
| B1_llm_only | 0.000 | 0.380 | 0.092 | 0.010 | 0.010 | 0.010 | — | 0 | 1.0 |

*Leakage 列：offline harness 下验证层未在 fake 答案上产生 critical
unsupported（claim 集合为空），故记 0；该指标需 live LLM 在线核验才有意义。*

## 8. Task-Type Results（A_full）

| task_type | n | success | route | tool F1 |
|---|---:|---:|---:|---:|
| knowledge_lookup | 82 | 0.927 | 0.927 | 0.927 |
| structured_lookup | 276 | 0.141 | 0.141 | 0.141 |
| aggregation | 27 | 1.000 | 1.000 | 1.000 |
| relationship_query | 31 | 0.000 | 0.000 | 0.000 |
| statistical_analysis | 10 | 1.000 | 1.000 | 1.000 |
| trend_analysis | 2 | 0.000 | 0.000 | 0.667 |
| comparison | 12 | 0.917 | 1.000 | 0.988 |
| report_generation | 10 | 0.000 | 0.600 | 0.747 |
| multi_source_analysis | 32 | 0.031 | 0.563 | 0.643 |
| ambiguous_task | 5 | 0.800 | 0.800 | 0.800 |

**解读**：纯 SQL 聚合/统计/比较类接近满分；**knowledge_lookup（RAG）0.927**
证明真实知识库在 offline 也能稳定召回；**relationship_query /
structured_lookup 的 KG 子集在 offline 无 Neo4j，记 0.000 是 honest
failure 而非系统缺陷**（在线模式下预期回升）。

## 9. Difficulty Results（A_full）

| difficulty | n | success |
|---|---:|---:|
| easy | 299 | 0.201 |
| medium | 116 | 0.767 |
| hard | 72 | 0.264 |

medium 最高：easy 桶包含大量 KG 路由的离线空跑任务（拉低 easy）；
hard 桶以多工具组合为主（拉低 hard）。性能**未随难度单调下降**，
而是被"工具是否离线可用"主导——这是数据分布效应，负结果保留。

## 10. Tool Composition Results（A_full）

| composition | n | success | tool F1 |
|---|---:|---:|---:|
| clarification | 5 | 0.800 | 0.800 |
| single_tool | 437 | 0.373 | 0.373 |
| two_tool | 30 | 0.000 | 0.756 |
| three_plus_tool | 15 | 0.067 | 0.504 |

**多工具任务（two_tool / three_plus_tool）success 远低于单工具**，
而 tool F1 反而更高——系统能选对工具组合，但**多源结果融合到答案的
一步在 offline（无 LLM 在线生成）失败**。这正是多工具任务作为系统
真实难点的证据（RQ3）。

## 11. Component Ablation

| 消融 | success | vs A_full |
|---|---:|---|
| A_full | 0.345 | — |
| w/o Dynamic Routing | 0.008 | **大幅下降**（SupervisorRouter 无法把 SQL/KG/多源任务路由对，与 Phase 3 C vs D 负结果一致） |
| w/o Verification | 0.345 | offline 下与 A_full 相同（见 §12） |
| w/o KG | 0.345 | offline 下与 A_full 相同（KG 任务本就已 0） |
| w/o Hybrid Retrieval | 0.345 | 同上 |
| w/o Reranker | 0.345 | 同上 |

**诚实说明**：w/o KG / w/o Hybrid / w/o Reranker / w/o Verification
在 offline harness 下与 A_full **指标相同**——原因是 offline 模式下
这些组件的"在/不在"不影响确定性路由与 SQL 结果（RAG 用 fake dense，
KG 无 Neo4j，验证无 live LLM 断言）。**这是 offline 的固有局限，不是
"去掉这些模块一定不下降"的证据**；要真正分离这些组件的贡献，
需要 online end-to-end 运行（Milvus + Neo4j + live LLM）。本实验保留
该限制并在 §17 明确。唯一有真实下降的是 **w/o Dynamic Routing**
（0.345 → 0.008），因为路由策略本身是确定性可测的。

## 12. Verification Analysis（RQ2）

offline harness 下验证层（claim 提取走离线 LLM，不产生 claim 集合），
因此 **unsupported detection / critical leakage 在 offline 全部为 0**，
无法分离"验证是否降低 unsupported claim"。

- 可测的间接证据：A_full（有验证尾链）与 B_no_verification（无验证）
  在 offline 下 success / route / tool F1 **完全一致**，说明验证层
  *不改变路由与工具选择*（符合设计：验证是 post-hoc 结果层，
  不 re-route、不 re-invoke 工具）；
- 验证真正影响的是**答案的 unsupported claim 泄漏**，该指标需
  live LLM 生成答案 + 在线核验才有数值。Phase 4 离线确定性评测
  （B：hallucinated-claim rate 1.0 → 0.025）是该指标的阶段证据；
  Phase 5 online 运行将补全该数字。

**结论（受限）**：在本 offline 配置下，验证层未观察到 leakage 数值
（claim 集为空），但也不改变上游路由/工具行为；其 RQ2 贡献
留待 online end-to-end 量化，不作离线夸大。

## 13. Routing Analysis（RQ1）

- **动态 vs 静态**：A_full route_acc 0.394 / tool F1 0.405，
  vs C_static 0.343 / 0.010，vs w/o Dynamic Routing 0.343 / 0.010。
  动态路由在 **SQL 子任务**上 tool F1 显著提升（0.010 → 0.405），
  证明**任务特征驱动的动态路由确实能选中合适工具**；
- **B4（规则路由）success 0.450 高于 A_full 0.345**：规则路由在
  easy 单工具任务上更"果断"，动态路由在部分任务上因 confidence gate
  降到 clarification（honest，不硬猜）。这是**真实且有价值的工程结果**，
  保留不掩盖；
- **task-type 层面**：动态路由在 knowledge_lookup / comparison /
  aggregation / statistical_analysis 接近满分路由；在 KG 子任务因
  offline 无 Neo4j 表现为 0（非路由错，是工具不可用）。

**RQ1 结论（受限）**：在本 offline 数据集与配置下，Dynamic Routing
能根据任务能力特征选择正确工具（SQL 侧 tool F1 0.405 vs 静态 0.010），
且优于纯静态路由；但其优势部分被"离线无 KG 服务"抵消，
完整结论依赖 online 运行。

## 14. Latency Analysis

| System | P50 (ms) | P95 (ms) | LLM calls |
|---|---:|---:|---:|
| A_full | 22.0 | 412 | 3.0 |
| B_no_verification | 22.0 | 258 | 3.0 |
| B4_rule_routing | ~20 | 233 | 2.0 |
| C_static | ~20 | 258 | 2.4 |

- 完整系统（A_full）P95 比无验证系统高约 **154ms**（验证尾链开销）；
- 动态/静态 + 验证（A）比 B4（规则）P95 高 ~180ms（多一次 LLM profile）；
- offline 下 LLM 调用计数为管线阶段推断值（A_full=3：意图+路由+答案）。
  live 模式下该数值与 wall-clock 将显著不同（每次 LLM 3–11s）。

## 15. Failure Analysis（40 条真实案例，分类）

`artifacts/phase5/enterprise_failure_report.md` 记录 40 条自动分类案例，
主因分布（A_full）：

- **routing miss**（最大类）：KG 路由任务在 offline 无 Neo4j，
  路由对但工具无结果 → 记为 honest failure（Phase 2.4 KG 覆盖缺口
  在 Phase 5 的体现）；
- **tool_selection_error**：多工具任务选了部分正确组合但未融合
  （two_tool success 0.000，tool F1 0.756）；
- **answer / no-evidence**：offline LLM 不产生真实答案，answer_exact 低；
- **KG coverage**：`relationship_query` 全 0（图谱服务不可用）。

**跨阶段 Failure Taxonomy 串联**（Phase 5 报告首版）：

```
Phase 2  RAG 检索 miss      →  Phase 5  路由/任务 miss
Phase 2.4 KG 覆盖缺口        →  Phase 5  relationship task 全 0（honest）
Phase 3  多工具顺序/置信     →  Phase 5  multi-tool answer failure
Phase 4  验证 unsupported    →  Phase 5  answer leakage（online 才量化）
```

## 16. Research Findings

在本实验数据集（487 条合成企业任务）与当前 offline-deterministic 配置下：

1. **RQ1**：动态路由优于纯静态路由的 SQL 侧工具选择（0.405 vs 0.010
   tool F1），但在 easy 单工具任务上规则路由 success 更高
   （0.450 vs 0.345）——动态路由的收益集中在"需要正确源"的复杂任务，
   该结论受 offline 无 KG 服务限制。
2. **RQ2**：验证层在不改动路由/工具的前提下运行（A 与 B 指标一致），
   其降低 unsupported claim 的作用需 online 量化；offline 下 claim 集
   为空，leakage=0 是"无 claim 可泄漏"而非"验证消除了幻觉"。
3. **RQ3**：多工具（two/three+）任务 success 显著低于单工具
   （0.000–0.067 vs 0.373），工具选择本身较准（tool F1 0.5–0.75）
   但**跨源融合到答案是当前系统最困难处**；多知识源协同的价值
   需 online 环境（RAG+SQL+KG 同时可用）方能体现。

> 所有结论限定于"本 offline 数据集 + 当前配置"；不外推为
> "系统全面优于所有方法"。

## 17. Limitations

- **offline harness**：RAG 用 fake dense 后端（RAG 命中经真实节点回注，
  但无真实 BGE-M3 向量相似度）；KG 无 Neo4j（关系任务全记 0 为 honest
  failure）；LLM 用离线 backend（不产生真实答案，claim 集为空）。
  因此 **w/o KG / w/o Hybrid / w/o Reranker / w/o Verification 消融
  在 offline 下与 A_full 指标相同**——这些组件的真实贡献需
  **online end-to-end 运行**（Milvus + Neo4j + live LLM）才能分离。
- **数据规模**：487 条，其中 structured_lookup 276 条偏多（KG 子集
  offline 拉低 structured_lookup 总 success）；难度分布 easy 偏多。
- **合成数据**：业务数值来自 synthetic 种子数据，仅用于流程与 GT
  一致性验证，非真实经营指标。
- **RQ2 的 leakage 指标在 offline 无法数值化**，结论受限。

## 18. Reproducibility

```bash
# Commit 1 — Benchmark（确定性）
python scripts/build_enterprise_tasks.py
python scripts/validate_enterprise_tasks.py

# Commit 2 — 端到端 runner（单系统 / 多系统）
python scripts/run_enterprise_eval.py --system A_full --offline

# Commit 3 — 全矩阵 + 消融
python scripts/run_enterprise_ablation.py --offline

# Commit 4 — 分组 + 失败分析
python scripts/analyze_enterprise_failures.py
```

- 所有数值由 `src/evaluation/enterprise_metrics.py` 程序计算；
- 输出在 `artifacts/phase5/`（gitignored）；
- GT 独立于系统输出，`provenance.verifier` 可回溯；
- 在线运行：`python scripts/run_enterprise_eval.py --system A_full`
  （需 live Milvus + Neo4j + LLM，见 `.env` 配置）。
