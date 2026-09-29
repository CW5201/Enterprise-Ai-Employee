# 系统架构设计（ARCHITECTURE）

> 状态：Phase 2.1 已更新（2026-09-29）。
> 本文档描述系统架构与约束。Phase 2.1 已实现 RAG 真实链路（BGE-M3 + Milvus），
> 其余模块（Neo4j KG、Hybrid RAG / Rerank、Claim-Evidence Verification）仍为
> 目标架构占位，尚未实现。

---

## 1. 分层架构总览

系统分为五层，层间只能自上而下调用，禁止跨层反向依赖。

```
┌─────────────────────────────────────────────────────────────┐
│ L1 用户交互层                                                 │
│    Vue3 + TypeScript + Element Plus + ECharts                │
│    对话 / 任务提交 / 执行过程 / 证据与图表 / 报告              │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP + SSE（流式）
┌──────────────────────────▼──────────────────────────────────┐
│ L2 AI 数字员工核心层                                          │
│    LangGraph StateGraph + AgentState                         │
│    Intent → Router → 执行 → 融合 → 验证 → 输出                │
└──────────────────────────┬──────────────────────────────────┘
                           │ Tool Calling（白名单 + 权限治理）
┌──────────────────────────▼──────────────────────────────────┐
│ L3 Tool Layer                                                │
│    rag / sql / kg / analysis / chart / report                │
└──────────────────────────┬──────────────────────────────────┘
                           │ 各存储的专用客户端
┌──────────────────────────▼──────────────────────────────────┐
│ L4 企业知识与数据层                                           │
│    Milvus（向量检索） DuckDB（业务数据） Neo4j（关系）          │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│ L5 Infrastructure Layer                                      │
│    LLM(Qwen) / Embedding(BGE-M3) / Reranker / 配置 / 日志      │
│    Trace / 异常 / 工具注册 / 评估                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 用户交互层（L1）

- 技术：Vue3 + TypeScript + Element Plus + ECharts。
- 通过 FastAPI 的 HTTP 接口提交任务，通过 **SSE** 接收流式执行过程。
- 必须展示的不只是最终答案，还包括：命中的工具、检索到的证据、SQL 语句、
  图谱路径、验证结论（哪条 claim 被支持 / 不被支持）。
- **可见性要求**：证据与验证结果是一等公民，因为系统的核心主张是"结论可追溯"。
- 目录：`src/frontend/Vue3/`（Phase 6 实现）。

## 3. AI 数字员工核心层（L2）

以 **LangGraph StateGraph** 编排，所有节点读写同一个 `AgentState`。

节点清单（`src/nodes/`）：

| 节点 | 职责 |
|---|---|
| `intent_understanding` | 解析意图、槽位、约束、置信度 |
| `supervisor_router` | **任务自适应路由**：选择工具集合与执行顺序 |
| `rag_retrieval` | 混合检索企业知识 |
| `sql_execution` | Text-to-SQL 生成、安全校验、执行 |
| `tool_invocation` | 统一工具选择与调用入口 |
| `clarification` | 任务信息不足时生成澄清问题 |
| `answer_generation` | 多源证据融合与答案生成 |
| `verification` | **Claim-Evidence Verification** |
| `feedback_learning` | 反馈收集与失败样例沉淀 |

工作流构建与状态持久化位于 `src/graph/builder.py` 与 `src/graph/checkpointer.py`。

## 4. Tool Layer（L3）

Agent 的"手"。**Agent 只能通过 Tool Registry 中登记的工具执行实际操作**（白名单，非黑名单）。

| 工具 | 职责 | 后端 |
|---|---|---|
| `rag_tool` | 企业知识检索 | Milvus |
| `sql_tool` | 只读业务数据查询 | DuckDB |
| `kg_tool` | 关系与路径查询（预定义 Cypher 模板） | Neo4j |
| `analysis_tool` | 受控 Pandas / NumPy 分析 | 内存 |
| `chart_tool` | 生成 ECharts 所需数据结构 | 内部 |
| `report_tool` | 组装结构化报告 | 内部 |

治理规则（`config/tool_registry.yaml`）：
- `whitelist_only: true`，未登记的工具有调用即失败；
- 每个工具声明 `timeout` 与 `permissions`（是否只读、是否可写、是否可联网）；
- 所有调用必须记录输入、耗时与结果，供评估与失败分析使用；
- **不允许执行任意用户 Python 代码**；
- **不允许 LLM 直接生成任意 Cypher**。

## 5. 企业知识与数据层（L4）

| 存储 | 唯一职责 | 明确不负责 |
|---|---|---|
| **Milvus** | 非结构化知识的向量检索（Dense / Hybrid 的向量侧） | 业务数据聚合、关系查询 |
| **DuckDB** | 业务数据、Text-to-SQL、统计分析（**只读**） | 向量检索、图遍历 |
| **Neo4j** | 实体关系、路径、组织与业务关系 | 业务数据的聚合计算 |

**职责不得混用。** 例如"某部门报销总额"必须走 DuckDB，
"某部门下有哪些岗位、汇报关系如何"走 Neo4j，
"报销制度规定上限是多少"走 Milvus——三者不能相互替代。

## 6. Infrastructure Layer（L5）

| 组件 | 文件 | 约束 |
|---|---|---|
| LLM 客户端 | `src/core/llm_client.py` | 唯一调用入口；业务代码不得直接耦合模型 SDK |
| Embedding | `src/core/embedder.py` | 统一封装 BGE-M3 |
| Reranker | 由检索侧统一封装 | BGE-Reranker-v2-M3 |
| 向量库接口 | `src/core/vector_store.py` | 统一封装 Milvus |
| SQL 执行器 | `src/core/sql_executor.py` | 只读、语句白名单、行数上限、超时 |
| 配置加载 | `src/core/config_loader.py` | 读取 `config/*.yaml`；密钥来自环境变量 |
| 工具注册 | `src/core/tool_registry.py` | 白名单与权限校验 |
| 可观测性 | `src/core/observability.py` | 结构化日志、Trace、耗时 |
| 异常体系 | `src/core/exceptions.py` | 统一异常类型，便于失败归因 |

---

## 7. AgentState

`AgentState` 是贯穿全流程的唯一任务状态（定义于 `src/core/state.py`，Phase 1 实现）。
所有节点读取并更新它，禁止在节点之间传递隐式状态。

计划的字段分组：

| 分组 | 字段（计划） | 说明 |
|---|---|---|
| 任务 | `task_id`, `user_task`, `conversation` | 输入与上下文 |
| 理解 | `intent`, `slots`, `constraints`, `intent_confidence` | 意图理解结果 |
| 路由 | `selected_tools`, `routing_plan`, `routing_confidence`, `routing_history` | 路由决策与轨迹 |
| 执行 | `tool_calls`, `tool_results`, `errors` | 工具调用记录 |
| 证据 | `evidence` | 统一证据列表（见下） |
| 分析 | `analysis_results` | 受控分析输出 |
| 答案 | `answer`, `citations`, `chart_spec`, `report` | 生成结果 |
| 验证 | `claims`, `claim_verdicts`, `support_score`, `verification_status` | 验证结果 |
| 控制 | `iteration`, `status`, `next_action`, `latency_ms` | 流程控制与终止条件 |

**统一证据模型（计划）**：

```python
Evidence = {
    "evidence_id": str,      # 稳定 ID，跨重跑可复现
    "source_type": str,      # "milvus" | "duckdb" | "neo4j" | "analysis"
    "source_ref": str,       # chunk id / sql 结果集 / 图谱路径 / 分析函数
    "content": str,          # 供 LLM 阅读的证据文本
    "payload": dict,         # 结构化原始数据
    "score": float,          # 检索或计算得到的相关度
    "metadata": dict,        # 文档元数据 / 表名 / 时间范围等
}
```

统一证据模型是**多源融合（创新点 2）的前提**：只有把四种来源放进同一个结构，
才能做去重、加权、引用与验证。

## 8. Intent Understanding

- 输入：用户任务、对话上下文、意图分类体系（`config/routing_rules.yaml` 的 `intent` 段）。
- 输出：`intent`、`slots`、`constraints`（时间范围 / 部门 / 阈值）、`confidence`。
- 意图体系：`information_query`、`knowledge_qa`、`text_to_sql`、`data_analysis`、
  `relation_query`、`multi_source_reasoning`、`complex_agent_task`、`chitchat_or_out_of_scope`。
- **不猜测**：置信度低于阈值或属于超范围请求时，转交 `clarification` 节点或明确拒答。

## 9. Supervisor Router

- 输入：`intent`、`slots`、路由规则、工具注册表、当前已有证据。
- 输出：`selected_tools`、`routing_plan`、`confidence`。
- 支持多工具组合，默认最小工具集原则（能一个工具解决就不调用第二个）。
- 记录 `routing_history`，使路由决策可以被评估（Routing Accuracy）与消融。

## 10. Task-Adaptive Routing（创新点 1）

路由不是"关键词匹配到工具"，而是依据**任务类型 + 数据结构 + 证据需求**选择执行路径。

策略声明于 `config/routing_rules.yaml`：

| intent | 优先工具链 |
|---|---|
| information_query | sql |
| knowledge_qa | rag |
| text_to_sql | sql → analysis |
| data_analysis | sql → analysis → chart |
| relation_query | kg → sql |
| multi_source_reasoning | rag + sql + kg + analysis |
| complex_agent_task | rag + sql + kg + analysis + chart + report |

关键策略点：
- 数值型问题优先结构化数据，定义型问题优先文档，关系型问题优先图谱；
- 工具失败或结果为空时按 `fallback` 策略接管，而不是直接失败；
- 证据不足时**升级**为多工具路径（`on_low_evidence: escalate_to_multi_tool`）；
- 对 `text_to_sql / data_analysis / multi_source_reasoning / complex_agent_task`
  强制要求进入验证节点。

与固定流水线的差异，正是消融实验 A 要量化的问题。

## 11. RAG

**Phase 2.1 已实现的 dense 检索链路（真实 BGE-M3 + 真实 Milvus）**：

```
Query → BGE-M3 (1024-dim, L2-normalised) → Milvus Top-K → RetrievalResult (source preserved)
```

- Collection（名称 / metric / index 均来自 `config/settings.yaml`）：
  `chunk_id / document_id / title / source / category / text / metadata_json / vector`。
- 每个 chunk 保留 `source`（文档路径）与 `title`（文档标题），下游答案可引用。
- 正式后端为 Milvus；fake 后端（进程内 hash 向量）仅限单元测试 / 测试模式，
  且必须显式选择（`create_vector_store(backend="fake")`），Milvus 失败时**禁止**静默降级。
- 输出进入统一证据模型（`source_type = "milvus"`，`source_ref = milvus:chunk_id`）。

**Phase 2.2+ 目标扩展（尚未实现）**：

- Hybrid 检索：`Query 改写 → Metadata Filter → Dense + BM25 稀疏 → RRF 融合 →
  Rerank(BGE-Reranker-v2-M3) → Top-K`。
- Metadata Filtering 使用文档前置元数据（部门、文档类型、生效日期）。

## 12. SQL

链路：`问题 + Schema + 数据字典 + Few-shot → LLM 生成 SQL → 安全校验 → 只读执行 → 结果集`

- **安全校验**（`src/core/sql_executor.py`）：仅允许 `SELECT` / `WITH`，
  禁用 `INSERT/UPDATE/DELETE/DROP/CREATE/ALTER/ATTACH/COPY/PRAGMA` 等，
  强制行数上限与超时。
- **数据字典约束**：`data/schemas/data_dictionary.yaml` 定义指标口径（如"活跃客户"），
  使 SQL 生成与后续验证共用同一定义，避免"口径漂移"。
- 歧义登记表（`ambiguity_register`）声明哪些问法必须先澄清（如"大客户"）。
- 输出进入统一证据模型（`source_type = "duckdb"`）。

## 13. Knowledge Graph

- Neo4j + Cypher，**仅使用预定义模板**（`kg_tool` 的 `template_id` + 参数），
  由 `config/tool_registry.yaml` 中 `llm_generated_cypher: false` 强制约束。
- 原因：LLM 生成任意 Cypher 在生产环境中不可控（性能、越权、错误语义），
  且模板化后查询语义可被评估与复现。
- 适用范围：组织与汇报关系、客户-订单-产品关系、部门-岗位关系、实体路径。
- 输出进入统一证据模型（`source_type = "neo4j"`，`source_ref` 记录模板 ID 与路径）。

## 14. Tool Calling

- 所有工具调用经 `src/core/tool_registry.py` 统一入口，执行前校验白名单、权限与超时。
- 调用记录写入 `AgentState.tool_calls`：工具名、参数、耗时、状态、错误。
- 失败处理：单工具失败不终止任务，按 `routing_rules.yaml` 的 `on_tool_error: record_and_continue`
  记录并尝试替代路径；全部失败才进入降级答复。
- 并发上限 `max_parallel_calls: 4`，避免本地模型与数据库过载。

## 15. Result Aggregation

- 将 `AgentState.evidence` 中的多源证据归一化为统一证据列表：
  去重（同一 chunk / 同一结果集）、按来源与相关度加权、按时间与部门过滤冲突项。
- 冲突处理原则：结构化数据 > 文档描述（当文档可能过期时），
  但冲突本身必须被记录并在答案中显式说明，不允许静默取舍。
- 融合结果同时供答案生成与验证节点使用。

## 16. Claim-Evidence Verification（创新点 3）

流程：

```
Answer
  → Claim Extraction        拆分为原子事实性结论
  → Evidence Matching       为每条 claim 匹配候选证据
  → Support Check           判定 supported / partially_supported / unsupported / contradicted
  → Pass / Retry / Correct  通过则输出；不通过则重试或修正，超过上限则降级并标注
```

- 输出 `claims`、`claim_verdicts`、`support_score`、`verification_status`
  并进入最终答案的引用信息，前端可展开查看。
- 阈值与失败动作由 `config/settings.yaml` 的 `verification` 段控制
  （`min_support_score`、`max_retry`、`on_failure`）。
- 评估指标 **Evidence Support Rate** 即由此模块产出，消融实验 C 验证其必要性。

## 17. Final Output

三种输出形态：

1. **Answer**：带内联引用的自然语言结论，引用指向 `evidence_id`。
2. **Chart**：`chart_tool` 生成的 ECharts 数据结构，由前端渲染。
3. **Report**：`report_tool` 组装的结构化报告（结论、数据、证据、图表、方法说明）。

未通过验证的结论必须以显式标注的形式输出（例如"缺少证据支持"），
**不允许把未验证结论伪装成已验证结论**。

## 18. 端到端主线

```
用户任务
  ↓
Intent Understanding
  ↓
Supervisor Router（Task-Adaptive Routing）
  ↓
┌──────────────┬──────────────┬──────────────┐
│  RAG Tool    │  SQL Tool    │  KG Tool     │
└──────────────┴──────────────┴──────────────┘
  ↓
Analysis Tool
  ↓
Multi-source Evidence Fusion
  ↓
Claim-Evidence Verification
  ↓
Answer / Chart / Report
```

## 19. 工程边界（当前阶段的硬约束）

- `config/` 只承载行为配置，不写业务逻辑；
- 业务代码不得直接调用模型 SDK、不得直接实例化 Milvus / Neo4j 客户端，
  必须经 `src/core/` 的统一封装；
- 不允许 LLM 生成任意 Cypher，不允许执行任意用户 Python 代码；
- 新增技术或框架前必须先更新 `ADR.md`，不允许擅自扩张技术栈。
