# 研究设计（RESEARCH）

> 状态：Phase 2.4。§11 / §12 为阶段性研究证据（Phase 2.3 / 2.4
> 真实实验结果）；其余章节的方法学设计数值仍随实验填充。
> **未经真实运行的数字不得写入本文档。**

---

## 1. 研究背景

企业中的知识分布在三种异构载体上：制度与流程文档（非结构化）、业务数据库（结构化）、
组织与业务关系（图结构）。企业员工提出的一条任务，往往需要同时使用这三者。

例如："上个季度华东区的报销总额是多少？超出差旅标准的部分有多少？哪些部门超标最多？"

- "报销总额"需要查业务数据库（DuckDB）；
- "差旅标准"需要查制度文档（RAG）；
- "部门归属与汇报关系"需要查组织关系（知识图谱）；
- "超标金额与排序"需要计算（Analysis）；
- 最终结论必须能指回具体数据与文档条款，否则不可信（Verification）。

现有系统很少能端到端完成这类任务。

## 2. 现有企业 RAG / Agent 的局限

| 局限 | 表现 | 后果 |
|---|---|---|
| **单一知识源** | 只做文档 RAG，或只做 Text-to-SQL | 需要跨源的任务直接失败，或只回答一半 |
| **固定流水线** | 所有任务都走同一条链（总是先 RAG 再 LLM） | 数值问题被文档描述误导；关系问题无从回答；简单任务被过度处理 |
| **缺少口径约束** | 生成 SQL 时不带业务指标定义 | 语法正确但业务含义错误（"销售额"口径漂移） |
| **无结果验证** | 生成即输出 | 无依据结论（unsupported claim）无法被发现，可信度不可度量 |
| **不可评估** | 只有 Demo 与主观演示 | 无法回答"去掉某个模块会怎样" |
| **框架即创新** | 以"用了 X 框架"作为贡献 | 缺少可被检验的研究问题 |

本项目针对前四条局限设计方法，第五条是本项目的自我约束。

## 3. 研究问题

**RQ1 — 路由**：如何根据企业任务类型动态选择 RAG、SQL、知识图谱和分析工具？
固定流水线在混合型任务上必然浪费或缺失能力；需要一套可解释、可评估的路由策略。

**RQ2 — 验证**：如何验证 AI 数字员工生成的结论是否真正有数据或文档证据支持？
需要把"结论"拆解为可核对的原子事实，并逐条与证据比对。

**RQ3 — 协同**：多源知识协同相比单一知识源，对复杂企业任务的完成效果有何影响？
需要统一证据表示与融合机制，并通过消融实验量化其增益。

## 4. 方法

### 4.1 总体方法

```
任务 → 意图理解 → 任务自适应路由 → 多工具执行 → 统一证据表示
     → 多源融合 → 结论抽取与证据验证 → 答案 / 图表 / 报告
```

### 4.2 方法要点

1. **意图到工具的策略映射**：把任务类型映射为工具集合与优先顺序，
   而不是让 LLM 在无约束空间中自由选择工具。
2. **统一证据模型**：四种来源（Milvus chunk / DuckDB 结果集 / Neo4j 路径 / Analysis 输出）
   归一为同一 `Evidence` 结构，使融合、引用与验证有共同基础。
3. **口径字典约束生成**：业务指标定义集中在 `data/schemas/data_dictionary.yaml`，
   SQL 生成与结论验证共用同一定义。
4. **模板化图谱查询**：图谱查询限定为预定义 Cypher 模板 + 参数，
   保证语义可控、结果可复现。
5. **结论级验证**：以 claim 为最小验证单元，而非对整段答案给一个模糊评分。

### 4.3 关键设计取舍

- 路由策略**显式声明**在配置中（可读、可审计、可消融），而不是完全交给 LLM 隐式决策。
- 验证结果**参与输出**（引用与支持状态展示给用户），而不是内部静默过滤。
- 冲突证据**显式呈现**，不静默取舍。

## 5. 创新点

### 创新点 1：Task-Adaptive Routing

- 内容：依据任务类型、数据结构与证据需求动态选择工具组合与执行顺序；
  支持失败接管与证据不足时的路径升级。
- 研究问题：动态路由相比固定流程是否可以提高复杂企业任务的完成率和执行效率？
- 度量：Task Success Rate、Routing Accuracy、Average Latency。
- 消融：Ablation A（移除动态路由，退化为固定流水线）。

### 创新点 2：Multi-source Knowledge Fusion

- 内容：非结构化知识、结构化业务数据、图谱关系与 Python 分析结果在统一证据空间中融合，
  包括去重、加权与冲突显式化。
- 研究问题：多源知识协同相比单一知识源，对复杂任务是否具有更好的完成效果？
- 度量：Task Success Rate（按任务类型分组，重点看 multi_source_reasoning 与 complex_agent_task）、
  Evidence Support Rate。
- 消融：Ablation B（移除 Knowledge Graph）。

### 创新点 3：Claim-Evidence Verification

- 内容：`Answer → Claim Extraction → Evidence Matching → Support Check → Pass / Retry / Correct`。
- 研究问题：结果验证是否能够提升答案可信度、证据支持率并降低无依据结论？
- 度量：Evidence Support Rate、Task Success Rate、无依据结论比例（错误分析中统计）。
- 消融：Ablation C（移除验证节点）。

> 三个创新点各自独立可移除、可度量，且都不是"使用了某个框架"。

## 6. Baseline

| 编号 | 系统 | 说明 |
|---|---|---|
| Baseline 1 | LLM | 纯模型回答，无检索、无工具 |
| Baseline 2 | LLM + RAG | 仅文档检索增强 |
| Baseline 3 | Hybrid RAG + SQL | 文档 + 结构化数据 |
| Baseline 4 | RAG + SQL + KG | 再加知识图谱，但仍是固定流程、无验证 |
| **Ours** | Task-Adaptive Routing + RAG + SQL + KG + Analysis + Claim-Evidence Verification | 完整系统 |

Baseline 与 Ours 使用**同一 LLM、同一 Embedding、同一评估集、同一 Prompt 版本记录**，
只有系统结构不同，否则比较不成立。

## 7. Ablation

| 实验 | 移除的模块 | 希望回答 |
|---|---|---|
| Ablation A | Task-Adaptive Routing（改为固定 RAG→SQL→Answer） | 动态路由贡献了什么 |
| Ablation B | Knowledge Graph | 图谱关系贡献了什么 |
| Ablation C | Claim-Evidence Verification | 结果验证贡献了什么 |

每个消融实验只改变一个变量，其余配置与 Ours 完全一致。

## 8. 评价指标

| 指标 | 定义要点 |
|---|---|
| Task Success Rate | 任务完成且结论满足判定标准的比例（按任务类别分组统计） |
| Routing Accuracy | 路由选择的工具集合与参考工具集合的一致程度 |
| Retrieval Recall@K | Top-K 检索结果覆盖参考证据的比例 |
| SQL Execution Accuracy | 生成 SQL 的执行结果与参考结果一致的比例 |
| Evidence Support Rate | 被证据支持的 claim 占比 |
| Average Latency | 端到端平均耗时（路由、各工具、验证分别记录分项耗时） |

指标定义与计算方式详见 [`EVALUATION.md`](EVALUATION.md)。**当前无任何数值。**

## 9. Error Analysis

错误按阶段归因，便于定位是"理解错"还是"执行错"还是"表述错"：

1. **意图理解错误**：意图分类错误或槽位抽取错误。
2. **路由错误**：选了工具但选错（漏选 / 多选 / 顺序错误）。
3. **检索错误**：召回缺失、召回噪声、Rerank 排序错误、Metadata Filter 误过滤。
4. **生成错误**：SQL 语义错误、图谱模板误选、分析函数误用。
5. **融合错误**：多源冲突处理不当、证据去重过度。
6. **验证错误**：漏判（无依据结论通过了验证）、误判（正确结论被判为不支持）。
7. **系统错误**：超时、服务失败、状态丢失。

每类错误登记到 [`FAILURE_HANDBOOK.md`](../FAILURE_HANDBOOK.md)，包含检测方式与回归测试计划。

## 10. 研究边界

**本项目不研究**：
- 模型预训练 / 微调 / 对齐；
- 多模态（图像、音频）任务；
- 通用对话能力与闲聊质量；
- 分布式系统与大规模性能工程；
- 生产级安全合规（等保、审计、复杂 RBAC）。

**本项目明确不做**：
- 伪造任何实验结果或 Benchmark 数值；
- 声称使用了某家真实企业的内部数据；
- 把第三方框架的使用包装成研究贡献。

**已知的效度限制**（需在论文中说明）：
- 企业知识库语料来自公开文档与合成文本，与真实企业内部语料的分布存在差异；
- 业务数据来自公开示例库（WideWorldImporters / AdventureWorks），规模小于真实企业；
- 评估集部分由受控生成得到，需说明生成规则与人工校验比例；
- 单机环境的延迟指标不能直接外推到生产部署条件。

## 11. Phase 2.3 Research Evidence（Retrieval Quality 阶段性证据）

**对应研究问题**（沿用第 3 节编号，未新增 RQ）：

- **RQ3 — 协同 / 检索机制**：在多源（多通道）知识检索场景下，
  不同检索机制（Dense / BM25 / 混合 RRF / 混合 + Reranker）
  对**召回质量**与**排序质量**的影响。

Phase 2.3 在该研究问题下的阶段性证据（详见
[`EVALUATION.md`](EVALUATION.md) §15 与
[`phase2.3/EXPERIMENT_REPORT.md`](phase2.3/EXPERIMENT_REPORT.md)）：

1. 在 190-chunk 合成企业语料、56 条人工核验 query 上，
   **Hybrid（Dense + BM25 + RRF）的 Recall@5 / NDCG@5 高于单一
   Dense 与 BM25**——多通道协同在本数据集上有稳定召回收益；
2. 在 Hybrid 基础上加入 **BGE-Reranker-v2-M3 重排**后，
   **MRR 提升（0.9500 → 0.9613）**而 **Recall@5 与 NDCG@5 未提升
   （0.9286 → 0.9271 / 0.9120 → 0.9087）**——重排主要改善
   "首个相关结果的位置"（排序），而非扩大召回覆盖；
3. 在 **hard 难度 query** 上 Reranker 未表现出稳定净收益
   （Recall@5 0.9583 → 0.9470），负结果如实保留；
4. 上述结论**仅限本实验数据集（190-chunk synthetic enterprise
   corpus / 56 queries / CPU 推理）**，不外推为 Reranker 在一般
   企业知识库上的普遍结论；更大语料与 430 条正式 Benchmark
   的验证留给 Phase 5。

## 12. Phase 2.4 Research Evidence（Knowledge Graph 阶段性证据）

**对应研究问题**（沿用第 3 节编号，未新增 RQ）：

- **RQ3 — 协同 / 多源知识表示**：关系型业务数据转换为知识图谱后，
  能否作为**第三种知识源**支撑跨实体的关系遍历查询；为后续
  多源融合（创新点 2）提供"图结构证据"来源。

Phase 2.4 在该研究问题下的阶段性证据（详见
[`phase2.4/EXPERIMENT_REPORT.md`](phase2.4/EXPERIMENT_REPORT.md)）：

1. 将 WWI 关系型数据（DuckDB）转换为 Neo4j 图谱
   （4,723 节点 / 3,880 关系），**entity lookup 与 one-hop /
   two-hop 关系遍历在 58 任务评测集上得到可复现的正面结果**
   （two_hop F1 0.825 / path accuracy 0.958；one_hop P 0.889）；
2. **multi-hop / cross-entity 复合查询是当前模板层的瓶颈**：
   需要集合交集或双边聚合的问题（kg-rq-054 / 055 / 056）超出
   预定义模板能力，exact match 为 0——负结果如实保留；
3. **路径准确率（0.924）显著高于实体集合 exact match（0.293）**：
   图遍历本身可靠，失分集中在远端实体集合的精确匹配与复合答案口径；
4. 评测集 58 条任务的 ground truth 全部由 DuckDB SQL 独立核验、
   **先于**图谱查询生成，避免"用 Neo4j 输出当答案"的评测污染。

**边界声明**：本阶段**不能**证明"知识图谱证明了多源融合有效"——
那需要 Phase 3 的 Dynamic Routing 与 Phase 4 的 Claim-Evidence
Verification 把图谱作为独立证据源接入融合/验证链路。Phase 2.4 只能
证明"知识图谱作为关系型知识源，可以支撑企业业务实体之间的多跳关系
查询"（限定于 WWI 样例 + 项目定义 schema）。

## 13. Phase 3 Research Evidence（Task-Adaptive Routing 阶段性证据）

**对应研究问题**（沿用第 3 节编号，未新增 RQ）：

- **RQ1 — 路由**：企业任务能否按任务特征（能力标志 + 歧义度）
  被动态路由到合适的知识源与工具。

Phase 3 在该研究问题下的阶段性证据（详见
[`phase3/EXPERIMENT_REPORT.md`](phase3/EXPERIMENT_REPORT.md)）：

1. 在 100 条人工声明 GT 的路由评测集上，**动态路由（规则基 C /
   LLM 基 D）route accuracy（0.630 / 0.680）显著高于静态策略
   （A Static RAG 0.110 / B Static SQL 0.220）**——任务特征驱动
   的候选生成对多类型任务有可测量增益，验证 RQ1 的可行性；
2. **LLM 的边际价值在歧义消解而非 route_type 本身**：D（0.680）
   仅略高于 C（0.630），但澄清精度 0.880、对模糊任务稳健性更高；
   route_type 由能力标志 + 规则引擎确定性导出；
3. **capability validation 是安全换精度的权衡**：无校验消融 E
   route_acc 0.720 略高，但失去结构化决策 / 一致性保证 / 澄清安全网；
4. **multi_tool 顺序拆解是当前主要失分点**：D 的 plan exact
   match（0.520）低于 route_acc（0.680）；
5. **KG 模板覆盖缺口（Phase 2.4 遗留）直接传导为路由失败**：
   "客户-供应商"类关系任务被迫走 multi_tool 补偿。

**边界声明**：本阶段证明"系统能够根据任务特征动态选择不同知识源与
工具"（限定 100 任务 / WWI 样例 + 项目 schema / 单模型 Qwen 代理），
**不能**证明"多源融合已被证明有效"——多源融合的最终证据需
Phase 4（Claim-Evidence Verification）与 Phase 5（430 任务正式
Benchmark）。
