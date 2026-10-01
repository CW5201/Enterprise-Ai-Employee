# Phase 3 Task-Adaptive Routing Experiment

> **定位**：Phase 3 的**正式阶段性实验记录**，用于支撑毕业论文
> "路由策略设计与实验"章节。
> **数据来源**：本文所有指标均生成自
> [`artifacts/phase3/routing_eval_summary.json`](../../artifacts/phase3/routing_eval_summary.json)
> （由 `scripts/run_routing_eval.py` 真实运行产出，**不进 Git**，本地保留）。
> 逐任务明细见 `artifacts/phase3/routing_eval_results.json`。

---

## 1. Objective

验证**任务自适应路由（Task-Adaptive Routing）**：企业 AI 任务能否按
任务特征被动态路由到合适的知识源与工具（RAG / SQL / KG / Analysis /
多工具协同），且动态路由相对静态策略是否有可测量的增益。

## 2. Research Question

**RQ1** — How can enterprise AI tasks be dynamically routed to
appropriate knowledge sources and tools according to task characteristics?

本阶段回答"系统是否能够根据任务特征动态选择不同知识源与工具"
（capability-aware 路由的可行性与代价），**不回答**"多源融合是否已被
证明有效"——那需 Phase 4（Claim-Evidence Verification）与 Phase 5
（487 任务正式 Benchmark，Phase 5 已构建）。

## 3. Task Taxonomy

10 类任务（`config/routing_rules.yaml` task_types）：

| task_type | 说明 | 主要能力信号 |
|---|---|---|
| knowledge_lookup | 文档/政策/制度 | 非结构化知识 |
| structured_lookup | 业务表精确事实 | 结构化数据 |
| aggregation | COUNT/SUM/AVG/GROUP BY | 结构化数据 |
| relationship_query | 实体关系/多跳 | 关系遍历 |
| statistical_analysis | 派生统计量 | 统计分析 |
| trend_analysis | 时序变化/增长 | 统计 + 结构化 |
| comparison | 并排/排名对比 | 结构化 + 统计 |
| report_generation | 格式化多部分报告 | 多源 |
| multi_source_analysis | ≥2 类知识源协同 | 多源 |
| ambiguous_task | 无法消解，需澄清 | 无 |

## 4. Routing Architecture

```
用户任务
  ↓
Intent 理解（Phase 1 节点，规则快车道 + LLM 车道）
  ↓
TaskRouterNode（Phase 3）
  ├─ 规则车道：确定性意图 → 能力标志
  ├─ LLM 车道：结构化 TaskProfile（能力标志 + 歧义度 + 难度）
  ↓
候选工具生成（capability_rules，config/routing_rules.yaml）
  ↓
路由规则 + 一致性校验（RoutingDecision.model_validate）
  ↓
RoutingDecision（route_type + tools + execution_plan + confidence）
```

- **LLM 提议、规则裁决**：LLM 输出结构化 TaskProfile，规则引擎生成
  候选并裁决；非法工具 / 非法 route / 一致性冲突被 Pydantic 拒绝，
  **绝不静默 fallback 到 RAG**。
- **confidence gate**：低于 0.3 的非 clarification 决策降级为
  clarification（不猜测）。
- **routing_trace**：完整审计（profile、candidates、selected tools、
  reason、timestamp、latency_ms），不含 secret。

## 5. Tool Capability Model

`config/tool_registry.yaml` v0.2 为 6 个工具声明能力模型：

| 工具 | 能力 | 说明 |
|---|---|---|
| rag | unstructured_knowledge, document_qa, policy_rules | 非结构化知识 |
| sql | structured_data, exact_facts, aggregation, filtering, grouping | 业务表 |
| kg | entity_relationship, multi_hop_traversal, cross_entity_query | 关系/多跳 |
| analysis | statistical_calculation, trend_analysis, comparison, derived_metrics | 派生计算 |
| chart | visualization | 可视化（伴随工具） |
| report | formatted_report, evidence_aggregation | 报告生成 |

capability → tool 候选规则（带权重）：`requires_structured_data→sql`、
`requires_unstructured_knowledge→rag`、`requires_relationship_reasoning→kg`、
`requires_statistical_analysis→analysis`、`requires_visualization→chart(0.5)`、
`requires_multi_source→report(0.3)`。

## 6. Evaluation Dataset

`data/eval/routing_eval.jsonl`：**100 任务**（8 route 类型 ×
10 task_type × easy/medium/hard）。Ground truth 按任务语义**人工声明**
（`expected_route` / `expected_tools` / `expected_order`），**不由
任何 router 生成**。含 10 条 clarification 任务与 44 条 multi_tool。
hard KG 关系任务的 KG 模板覆盖缺口已在 `notes` 中显式标记
（Phase 2.4 遗留 limitation，本实验不掩盖）。

## 7. Baselines

| 编号 | 系统 | 说明 |
|---|---|---|
| A | Static RAG | 所有任务统一走 RAG |
| B | Static SQL | 所有任务统一走 SQL |
| C | Rule-based | 确定性关键词能力标志 → 规则引擎（无 LLM） |
| D | LLM Task-Adaptive | 真实 `TaskRouterNode`（LLM 提议 + 规则校验 + confidence gate） |
| E | LLM without capability validation | 消融：LLM 能力标志直接映射 route，**跳过**澄清门/一致性校验/多工具触发 |

## 8. Metrics

| 指标 | 定义 |
|---|---|
| Route Accuracy | `route_type` 与期望精确匹配 |
| Tool Selection P/R/F1 | 预测工具集 vs 期望工具集（集合，顺序不敏感） |
| Plan Exact Match | 执行顺序与 `expected_order` 精确一致 |
| Clarification Accuracy | 两侧是否一致判定需要澄清 |
| Task Success | 单源任务：期望工具被覆盖（multi_tool 额外须分解正确）；澄清任务：双方均判澄清 |
| Latency | mean / p50 / p95（每任务路由决策墙钟时间） |

预测失败（LLM 报错 / 不可路由）**保留在分母**，记 0 分。

## 9. Overall Results（真实运行，100 任务）

| 系统 | Route Acc | Tool F1 | Plan EM | Clarif Acc | Task Success | 延迟 mean (ms) | 失败数 |
|---|---|---|---|---|---|---|---|
| A Static RAG | 0.110 | 0.172 | 0.210 | — | 0.110 | 0.0 | 0 |
| B Static SQL | 0.220 | 0.508 | 0.320 | — | 0.220 | 0.0 | 0 |
| C Rule-based | **0.630** | **0.790** | 0.590 | — | **0.570** | 6.5 | 0 |
| D LLM Adaptive | 0.680 | 0.781 | 0.520 | **0.880** | 0.520 | 8605 | 0 |
| E LLM unvalidated | 0.720 | 0.814 | 0.600 | 0.920 | 0.580 | 7997 | 6 |

> E 的 6 次失败全部是 **LLM 429 限流**（`Qwen request failed after
> 3 attempts: 429 Too Many Requests`），属运行环境瞬态故障，非路由
> 逻辑缺陷；E 的 route_acc 略高于 D 是因为**跳过了澄清门与一致性
> 校验**（更少"保守降级"），代价是失去 guardrail 且无结构化决策
> 保证——这正是消融要隔离的能力验证层代价。

## 10. Task Type Results（Route Acc / Tool F1，摘录）

| 系统 | knowledge_lookup | structured_lookup | aggregation | relationship_query | statistical_analysis | multi_source |
|---|---|---|---|---|---|---|
| C | 高 | 高 | 高 | 中 | 中 | 中 |
| D | 高 | 中 | 中 | 中 | 中 | 中 |

（完整数字见 `routing_eval_summary.json` 的 `by_task_type`；
C 在结构化/聚合类上最稳，D 在关系类上与 C 接近但延迟高 3 个数量级。）

## 11. Difficulty Results

| 系统 | easy | medium | hard（route_acc） |
|---|---|---|---|
| C | 最高 | 中 | 最低 |
| D | 高 | 中 | 最低 |

route_acc 随难度单调下降；hard 任务（多跳/跨源/澄清占比高）失分最多。
失败集中在 multi_tool 的顺序拆解与 KG 模板覆盖缺口。

## 12. Ablation Study

**核心对比**：

1. **动态 > 静态**：C（0.630）/ D（0.680）的 route_acc 远高于
   A（0.110）/ B（0.220）——任务特征驱动的候选生成对多类型任务
   有显著增益，验证 RQ1 的可行性。
2. **LLM 在 route_type 上未超确定性规则**：D（0.680）仅略高于
   C（0.630），但 D 的澄清精度（0.880）与对模糊任务的稳健性
   更高；C 无澄清门（对 "帮我查一下" 类会误判为某单一工具）。
   **LLM 的价值在歧义消解，不在 route_type 本身**（route_type
   由能力标志 + 规则引擎确定性导出）。
3. **capability validation 的代价**：E（无校验）route_acc 0.720
   高于 D 0.680，但 E 丢失了结构化决策、一致性保证与澄清安全网，
   且 6 次 429 限流全部记 0。验证层的价值是**可审计 + 不越权**，
   不是 route_acc 本身——这是安全/可复现性 vs 精度小代价的权衡。
4. **multi_tool 拆解不稳定**：D 的 plan_exact_match（0.520）低于
   route_acc（0.680）——工具集合对，但执行顺序/依赖拆解仍经常错。
   这是当前动态路由的主要失分点。

## 13. Failure Analysis（真实失败案例，10 个）

以下取自 `routing_eval_results.json` 的 D（LLM adaptive）真实结果：

1. **rt-067/068/070（SQL+KG multi_tool）— wrong execution order**：
   期望 `kg → sql`（先关系遍历再取指标），D 多预测为 `sql → kg`。
   原因：TaskProfile 能力标志不携带"指标优先还是关系优先"的序信息。
2. **rt-037/038（hard relationship_query）— KG coverage limitation**：
   期望纯 `kg`，D 判 `multi_tool[kg, sql]`。真实原因：`customer_orders`
   等 KG 模板返回的是"订单列表"，无法单模板直接回答"哪些客户买了
   供应商 X 的商品"（Phase 2.4 记录的 `supplier_customers` 反向模板
   缺口）。**该任务理论上需 KG，但当前 KG Tool 不支持对应 query**
   ——路由把它补成 multi_tool 是"能力不足的诚实补偿"，已在
   `routing_eval.jsonl` notes 中标记。
3. **rt-081/083/086/088（ambiguous_task）— clarification success**：
   D 正确判 clarification（4/4），C 把它们误判为某单一工具（route
   0）——证明 LLM 车道在歧义消解上的真实价值。
4. **rt-041/045（trend_analysis）— missing analysis step**：D 有时
   只预测 `sql` 漏掉 `analysis`（tool recall 下降），导致
   增长率类问题缺派生计算步骤。
5. **rt-061/065（RAG+SQL policy 类）— wrong tool selection**：
   "按销售政策统计违规订单"需 `rag + sql`，D 偶发只预测 `sql`
   （漏政策检索），因 TaskProfile 的 `requires_unstructured_knowledge`
   对"隐含引用政策"的显式化不稳定。
6. **rt-073/074/075（三源 report_generation）— missing multi-tool**：
   期望 `rag+sql+kg`，D 常缺 `kg`（关系遍历在报告生成中不被
   显式触发），tool F1 下降。
7. **rt-016/029/054/070/071/093（仅 E）— LLM service failure**：
   429 限流，E 记 0；D 未触发（运行窗口错峰）。属运行环境瞬态故障，
   非路由逻辑缺陷，如实记录。

**归类**：wrong execution order（1）/ KG coverage limitation（2）/
clarification 正确（3，正面）/ missing analysis step（4）/
wrong tool selection（5）/ missing multi-tool（6）/ LLM service
failure（7，环境）。**没有一例源于 Neo4j / DuckDB 数据层缺陷**。

## 14. Latency

| 系统 | mean | p50 | p95 |
|---|---|---|---|
| A / B | 0.0 ms | 0.0 | 0.0 |
| C | 6.5 ms | ~6 | ~8 |
| D | 8605 ms | ~7900 | ~23217 |
| E | 7997 ms | ~7300 | ~21500 |

LLM 车道（D/E）比确定性规则（C）慢约 3 个数量级——动态路由的精度
来自 LLM 任务理解，代价是每任务一次 LLM 调用。生产环境应按任务
置信度做"规则快车道优先、LLM 兜底"的混合策略（本阶段未实现）。

## 15. Findings（基于实际结果，限定本数据集）

- **动态路由相对静态有可测量增益**：C/D route_acc（0.63/0.68）
  显著高于 A/B（0.11/0.22），回答 RQ1 的"可行性"——任务特征
  驱动的工具选择能覆盖多类型企业任务，静态策略不能。
- **LLM 的边际价值在歧义消解而非 route_type**：D 仅比 C 略高
  （0.68 vs 0.63），但澄清精度 0.88、对模糊任务稳健性更高；
  route_type 本身由能力标志 + 规则确定性导出。
- **multi_tool 顺序拆解是当前主要失分点**：D 的 plan_exact
  （0.52）低于 route_acc（0.68），工具集合对但执行顺序常错。
- **capability validation 是安全换精度的权衡**：E（无校验）route_acc
  略高（0.72）但丢结构化/一致性/澄清安全网，且 6 次 429 全记 0。
- **KG 覆盖缺口直接造成路由失败**：Phase 2.4 记录的 `supplier_customers`
  反向模板缺失，使"客户-供应商"类关系任务被迫走 multi_tool 补偿，
  是跨阶段（2.4→3）的真实 limitation 传导。
- **LLM 车道延迟 ~8.6s/任务**（p95 ~23s），确定性规则 6.5ms——
  精度与延迟的权衡需 Phase 5 在正式 Benchmark 上重测。

> 以上结论**限定于本 100 任务 / WWI 样例 + 项目 schema / 单模型
> Qwen 代理**，不外推为"动态路由普遍优于静态"或"多源融合已证明
> 有效"。

## 16. Limitations

- **route_acc 对单源任务偏严**：GT 把"需 sql+analysis"记为
  `route: sql`，D 预测 `multi_tool` 在 route_acc 上记错，但
  task_success（宽松语义）记对——route_acc 低估了真实能力；
- **KG 模板覆盖缺口**（Phase 2.4 遗留）传导为路由失败；
- **顺序信息缺失**：TaskProfile 能力标志不携带工具执行优先级；
- **E 的 429 限流**是运行环境瞬态故障，非路由缺陷；
- **单模型**：只用一个 LLM 代理（Qwen via Agnes），未做多模型对比；
- **未做混合"规则快车道 + LLM 兜底"策略**：D/E 全走 LLM，
  延迟未优化；
- 数据仍为 WWI + 项目转换，非真实企业多源语料。

## 17. Reproducibility

| 项 | 值 |
|---|---|
| LLM | Qwen（Agnes apihub 代理，OpenAI-compatible），经 `src/core/llm_client.py` |
| Python | 3.11（本机） |
| 路由 | `src/nodes/task_router.py` + `src/core/routing_rules.py` |
| 评测 | `python scripts/run_routing_eval.py`（baseline A–E） |
| 消融 E | `artifacts/phase3/abl_e_results.json`（本地保留） |
| 数据集 | `data/eval/routing_eval.jsonl`（100 任务，`scripts/build_routing_eval.py`） |
| 指标 | `src/evaluation/routing_metrics.py` |
| 路由层不涉及 | Neo4j / Milvus / 模型权重 / 本地 runtime（不进 Git） |

**复现命令**：

```bash
# 全部 baseline
python scripts/run_routing_eval.py
# 单 baseline
python scripts/run_routing_eval.py --baseline D_llm_adaptive
```
