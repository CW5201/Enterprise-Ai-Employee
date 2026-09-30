# Architecture Decision Records (ADR)

本文件记录项目的**架构决策**。规则：

- 任何新增技术、框架或组件，**必须先在此新增一条 ADR**，才能进入代码；
- 每条 ADR 一经记录不删除；若决策被推翻，新增一条 ADR 说明取代关系；
- ADR 只描述"为什么这样选"，不描述"代码怎么实现"。

统一结构：**Context / Decision / Alternatives / Consequences**。

---

## ADR-001: Why LangGraph?

### Context

企业任务需要多步执行：意图理解 → 路由 → 多工具调用 → 融合 → 验证 → 输出。
这条链包含条件分支（路由选择）、循环（验证失败重试）、并行（多工具同时调用）
与状态持久化（长任务中断恢复）。用普通函数串联会导致控制流散落在各处，
且无法在缺少任何一步时清晰地回答"状态是什么"。

### Decision

采用 **LangGraph** 作为工作流编排层，使用 `StateGraph` 构建，共用统一的 `AgentState`。
LangChain 仅作为必要的基础组件引入，不引入其高层 Agent 抽象。

### Alternatives

| 方案 | 未采用原因 |
|---|---|
| 手写控制流（if/else + 函数调用） | 分支与循环增多后不可维护，状态隐式传递，难以做中间态观察 |
| LangChain AgentExecutor | 抽象层次过高，路由逻辑被隐藏在内部，无法满足"路由可解释、可消融"的研究需求 |
| 自研状态机框架 | 重复造轮子，且毕业设计的工作量应投入研究问题而非基础设施 |
| 传统工作流引擎（Airflow 等） | 面向批处理调度，不适合交互式、带 LLM 推理的短链路任务 |

### Consequences

正面：
- 显式图结构使"任务自适应路由"可以被声明、被观察、被单独移除（消融 A）；
- 状态持久化与中断恢复天然支持长任务；
- 中间态可直接用于前端展示与评估记录。

负面：
- 引入框架依赖，版本升级需回归测试；
- 图结构调试比普通函数栈复杂；
- 需注意"框架即创新"的陷阱——**LangGraph 的使用本身不是研究贡献**，
  研究贡献是 Task-Adaptive Routing、Multi-source Knowledge Fusion 与 Claim-Evidence Verification。

---

## ADR-002: Why Task-Adaptive Routing?

### Context

企业任务类型差异极大：制度问答只需要文档检索，营收统计只需要 SQL，
组织关系需要图谱，而"上季度超标报销最多的部门及其制度依据"需要三者协同。
固定流水线（例如总是 RAG → SQL → LLM）在简单任务上浪费、在复杂任务上缺失能力。
这正是 RQ1 要回答的问题。

### Decision

引入 **Task-Adaptive Routing**：由 `Supervisor Router` 依据意图、槽位、路由规则与
已有证据，动态选择工具集合与执行顺序；支持失败接管与证据不足时的路径升级。
路由策略**显式声明**在 `config/routing_rules.yaml`，而不是完全依赖 LLM 隐式决策。

### Alternatives

| 方案 | 未采用原因 |
|---|---|
| 固定流水线 | 无法同时满足多类任务，效率与完成率都受限；作为消融对照保留 |
| 完全由 LLM 自由选择工具 | 决策不可复现、不可审计，且难以做错误归因 |
| 纯关键词规则匹配 | 无法处理"看起来像知识问答其实是数值计算"的任务 |
| 训练一个专用路由分类器 | 需要标注数据与训练成本，超出本项目范围，且不利于解释 |

### Consequences

正面：
- 每类任务只调用必要的工具，降低延迟与噪声证据；
- 路由决策可记录（`routing_history`），可计算 Routing Accuracy，可做消融 A；
- 失败接管与升级策略让系统在部分工具失效时仍能完成任务。

负面：
- 路由本身可能出错（漏选/多选），成为新的错误来源，需纳入错误分析；
- 路由规则需要维护，任务类型扩展时需要同步更新配置与文档；
- 需要额外的路由判定耗时。

---

## ADR-003: Why Hybrid RAG?

### Context

企业制度类文本的特点：既包含大量专有名词、编号、表单名（"差旅费报销单 A-102"），
也存在语义相近但用词不同的表述（"请假流程" / "休假审批"）。
纯稠密检索对专有名词与编号的精确匹配较弱；纯稀疏（BM25）检索对语义改写不敏感。
此外，制度文档有时效性，必须能按部门、文档类型、生效日期过滤。

### Decision

采用 **Hybrid Retrieval**：
`Metadata Filter → Dense(Milvus) + BM25 → RRF 融合 → BGE-Reranker-v2-M3 重排 → Top-K`。

### Alternatives

| 方案 | 未采用原因 |
|---|---|
| 纯稠密检索 | 专有名词、编号召回差 |
| 纯 BM25 | 语义改写召回差，无法处理同义表述 |
| 仅稠密 + Rerank | Rerank 只能重排已召回结果，无法弥补第一阶段的漏召 |
| 使用 Chroma 等轻量向量库 | Milvus 更适合本项目对元数据过滤与索引参数可控性的需求，且避免双向量库并存 |
| 图检索增强（GraphRAG 全量替代） | 成本高，本项目已将图能力独立为 KG 工具，职责更清晰 |

### Consequences

正面：
- 召回率与排序质量优于单一通道（由 Retrieval Recall@K 分配置验证）；
- Metadata Filtering 使"现行有效"类问题不会命中已废止版本；
- Rerank 提升进入 LLM 上下文的质量，间接改善 Evidence Support Rate。

负面：
- 组件增多（BM25 索引需与向量索引同步维护）；
- 融合权重与 RRF 参数需要调参，评估时必须分配置报告；
- 多一次 Rerank 推理，增加延迟。

---

## ADR-004: Why Knowledge Graph?

### Context

企业问题中存在一类无法用文档或标量查询回答的问题：组织汇报关系、部门层级、
客户-订单-产品的关系路径、"某人的上级所属部门还负责哪些客户"等。
这类问题在关系数据库中需要递归 join，表达困难且不利于 LLM 生成。

### Decision

引入 **Neo4j + Cypher**，并且**只允许使用预定义查询模板**（`template_id` + 参数），
由 `config/tool_registry.yaml` 中 `llm_generated_cypher: false` 强制约束。

### Alternatives

| 方案 | 未采用原因 |
|---|---|
| 在 DuckDB 中用递归 CTE 实现 | 语义表达笨拙，LLM 生成递归 SQL 错误率高 |
| 让 LLM 直接生成任意 Cypher | 性能不可控、可能越权、语义不可复现、无法评估 |
| 不建图谱，交给文档描述 | 关系类问题的答案不精确且不可验证 |
| 使用 RDF / SPARQL | 生态与工具链复杂度更高，收益不明显 |

### Consequences

正面：
- 关系类查询语义明确、结果可复现，可精确记录证据（模板 ID + 路径）；
- 模板化使查询可审计，避免图数据库被滥用为通用计算引擎；
- 为创新点 2（多源融合）提供第三种独立证据来源，并支撑消融 B。

负面：
- 需额外维护图谱构建流水线与实体对齐逻辑（实体匹配错误是已知失败模式）；
- 模板覆盖面有限，新问题类型可能需要新增模板；
- 引入一个额外的存储组件与运维成本。

---

## ADR-005: Why DuckDB?

### Context

需要结构化业务数据来支撑 Text-to-SQL 与数据分析。约束：单机毕业设计环境、
不希望引入需要独立部署的数据库服务、需要与分析工具（Pandas）高效互操作、
查询必须严格只读。

### Decision

采用 **DuckDB** 作为业务数据库，**只读**访问，语句白名单仅允许 `SELECT` / `WITH`，
带行数上限与超时。

### Alternatives

| 方案 | 未采用原因 |
|---|---|
| PostgreSQL / MySQL | 需独立服务与运维，本地开发与 Docker 编排成本更高 |
| SQLite | 分析型（列式、聚合）性能与 Parquet/DataFrame 互操作不如 DuckDB |
| ClickHouse | 面向大规模 OLAP 集群，本项目数据规模远未达到，属于过度设计 |
| 直接对 CSV/Pandas 做操作 | 缺失 SQL 语义，无法支撑 Text-to-SQL 研究目标 |
| Spark | 分布式计算，与单机毕业设计场景不匹配 |

### Consequences

正面：
- 零运维，文件型数据库便于复现与分发（数据文件本身不入库）；
- 列式执行 + 与 Pandas 零拷贝互操作，分析类任务效率高；
- 只读连接（`read_only=true`）配合语句白名单，把写入风险降到最低；
- WideWorldImporters / AdventureWorks 可导入为本地库，支持 Text-to-SQL 评估。

负面：
- 并发写入能力弱（本项目只读，无影响）；
- 方言与 PostgreSQL 等存在差异，SQL 生成需针对 DuckDB 校准；
- 数据规模上限受单机资源限制。

---

## ADR-006: Why Claim-Evidence Verification?

### Context

LLM 生成的答案中可能包含看似合理但无数据或文档依据的结论。
企业场景下这类"无依据结论"代价很高，而仅凭最终答案文本无法判断其是否可信。
现有做法多是对整段答案给一个模糊置信度，无法定位到底哪一句有问题。

### Decision

引入 **Claim-Evidence Verification**：

```
Answer → Claim Extraction → Evidence Matching → Support Check → Pass / Retry / Correct
```

以 **claim（原子事实性结论）** 为最小验证单元，逐条给出
`supported / partially_supported / unsupported / contradicted` 判定与支持证据 ID；
验证结果进入最终输出的引用信息，未通过验证的结论显式标注而非静默过滤。

### Alternatives

| 方案 | 未采用原因 |
|---|---|
| 让 LLM 自评置信度 | 自评与事实正确性相关性弱，且不可追溯 |
| 整段答案打分 | 粒度太粗，无法定位问题结论，也无法计算表述级指标 |
| 只做引用展示（要求模型给引用） | 模型可能给出错误引用（citation mismatch），需要独立的匹配与判定环节 |
| 用外部检索器做事实核查 | 需额外训练/依赖，且企业领域证据已在系统内部，无需外部核查器 |
| 不做验证，靠更好的 Prompt | 属于"希望模型不犯错"，无法度量也无法消融 |

### Consequences

正面：
- 产出可度量的 **Evidence Support Rate**，直接支撑 RQ2 与消融 C；
- 验证失败可触发重试或修正，形成闭环改进；
- 用户可展开查看每条结论的证据，系统可信度可被外部检查。

负面：
- 增加一次（或多次）LLM 调用，延迟与成本上升；
- 验证模块自身会漏判与误判，必须单独评估其可靠性（否则"验证"只是转移了不可信）；
- claim 抽取质量直接影响后续所有判定，是新的错误来源。

---

## ADR-007: Why Unified Evaluation Framework?

### Context

项目的核心主张是"动态路由、多源协同、结论验证有效"。
这类主张必须有对照实验支撑：4 个 Baseline + 3 组消融 + 6 项指标。
如果评估靠临时脚本与手工记录，会出现：指标口径不一致、数据集版本混乱、
Prompt 变更后结果不可比、无法追溯某个数字怎么来的。

### Decision

建立**统一的评估框架**（`src/evaluation/` + `scripts/run_eval.py`），要求：

- 指标计算集中实现（`metrics.py`），Baseline / Ours / 消融共用同一套实现；
- 每次运行落盘完整快照：配置、数据集版本、Prompt 版本、运行环境、逐条结果、指标汇总；
- **禁止手填任何数值**，报告中的每个数字必须能追溯到逐条结果；
- CI 评估门禁与**存档 baseline 结果**比较，而非写死阈值。

### Alternatives

| 方案 | 未采用原因 |
|---|---|
| 临时脚本 + 手工记录 | 口径漂移、不可复现、无法审计 |
| 仅使用第三方评测框架 | 不支持本项目的路由准确率与证据支持率等定制指标 |
| 只保留最终指标不做逐条落盘 | 无法做错误归因与案例分析 |
| 手工整理结果表格 | 存在填错与"美化"的风险，违反项目纪律 |
| 引入 DVC / MLflow 等进行实验管理 | 超出本项目必要范围（见技术栈禁止清单），当前用目录约定 + 版本字段即可 |

### Consequences

正面：
- Baseline 与 Ours 的比较在方法学上成立（同实现、同数据集、同环境）；
- 错误分析有据可依，失败案例可沉淀到 `FAILURE_HANDBOOK.md`；
- 论文中的所有数字可回溯到具体运行目录，可复现。

负面：
- 前期投入大，Phase 5 之前无法产出任何指标；
- 框架本身的正确性也需要维护（指标实现错误会污染所有结论）；
- 逐条落盘带来存储开销。

---

## ADR-008: Reranker 阶段的设计约束（Phase 2.3）

### Context

Phase 2.3 在 Hybrid 检索之上接入 BGE-Reranker-v2-M3。需要明确：
Reranker 在检索链路中的位置、候选规模、默认开关、fake backend 边界，
以及负结果的记录纪律。

### Decision

1. **Reranker is a reranking stage, not a retrieval stage.**
   Reranker 只接收 RRF 融合后的候选（`candidate_k` 条），对候选重排，
   **不参与** Dense / BM25 的原始召回，也不扫描整个知识库。
   流程严格为：`Dense + BM25 → RRF → candidate_k → Rerank → final_k`。
2. **候选规模 `candidate_k=20` / `final_k=5`。**
   20 条候选给 Reranker 足够多样性以纠正 RRF 排序，同时控制推理成本；
   5 条与其余模式的最终 top-k 对齐，保证指标可比。
3. **默认 `reranker.enabled=false`。**
   生产默认关闭以保持 legacy dense / bm25 / hybrid 行为不变；
   正式实验时显式置 `true`。模型不可用时抛出
   `RerankerUnavailableError`，**禁止静默 fallback** 到 fake / 随机打分。
4. **Fake backend 仅限单元测试。**
   `FakeVectorStore` 与 `FakeScorer`（词法重叠打分）显式标注
   `backend="fake"`，只允许在测试 / 离线场景使用，
   绝不作为正式模型加载失败的替身。
5. **保留负结果。**
   实验若显示 Reranker 未改善（甚至略降）某指标，如实记录，
   不得修改 Ground Truth、挑选 query 或调参后只留最优结果。

### Consequences

正面：
- "先召回、后重排"的职责边界清晰，Reranker 不会掩盖召回缺陷；
- 默认关闭使 Phase 2.2 的三模式行为零改动，向后兼容；
- 负结果保留使论文结论可被审计与复现。

负面：
- 候选扩大带来推理延迟（CPU 上 ~5.7s/query），生产需按设备选型；
- `candidate_k` / `final_k` 是经验值，正式 Benchmark（Phase 5）需分配置重测。
---

## ADR-009: Phase 2.4 Knowledge Graph 的设计约束

### Context

Phase 2.4 将 WWI 关系型数据（DuckDB）转换为 Neo4j 知识图谱，并提供
预定义 Cypher 模板的只读查询能力。ADR-004 已确立"只允许预定义模板、
禁止 LLM 生成任意 Cypher"的方向；本 ADR 记录 Phase 2.4 落地时
追加的工程约束与负结果纪律。

### Decision

1. **凭证只走环境变量。** Neo4j 密码 / 用户名 / URI 由
   `.env` / 环境变量注入，`config/settings.yaml` 与 yaml schema
   文件**不存密码**；异常消息做 credential-safe 映射，不泄露
   bolt URI / 密码 / 堆栈给 API 调用方。
2. **命名空间隔离。** 图谱节点 / 关系均打 `__graph='eae'` 标记；
   `--reset` 只删该命名空间，**绝不**无条件删除整个 Neo4j 数据库，
   避免污染共享实例。
3. **派生关系显式溯源。** `Sales_OrderLines` 在 WWI 样例中为空，
   Order→StockItem 经 `Sales_Invoices` + `Sales_InvoiceLines` 派生
   （HAS_LINE 边），`__source` 标记 `derived:Sales_InvoiceLines`，
   不得伪装成原生 Sales_OrderLines 边。
4. **评测 GT 独立于图谱输出。** `data/eval/kg_eval.jsonl` 的
   ground truth 全部由 DuckDB SQL 独立核验、**先于** Neo4j 查询
   生成，避免"用待评估系统输出当答案"的评测污染；负结果任务
   （空集答案）保留，不因指标难看而删除。
5. **保留负结果与瓶颈声明。** 58 任务 exact match 仅 0.2931，
   multi-hop / cross-entity 复合查询是**当前模板层瓶颈**，如实记录；
   不得外推为"图谱全面优于 SQL"——那需 Phase 3 / 4 才能证明。

### Alternatives

| 方案 | 未采用原因 |
|---|---|
| 让 LLM 直接生成任意 Cypher | ADR-004 已否决：不可控、越权、不可复现 |
| `--reset` 直接 `MATCH () DETACH DELETE` 清空全库 | 破坏共享实例中其他数据；改用命名空间标记 |
| 用 Neo4j 查询输出反向构造 GT | 评测污染；GT 必须独立（DuckDB SQL） |

### Consequences

正面：
- 图谱可复现构建（幂等 + 命名空间 + 溯源），评测口径干净；
- 只读模板 + 校验器使 KG 工具可安全暴露给 agent 而不具备写能力。

负面：
- 派生关系（HAS_LINE）语义依赖 invoice 表，若将来 WWI 数据补全
  `Sales_OrderLines` 需切换边来源并重建图谱；
- 复合 / 负例查询受限于预定义模板覆盖，exact match 偏低是
  **模板层**而非图数据库的局限，需 Phase 3 路由 + 模板扩展才能改善。
