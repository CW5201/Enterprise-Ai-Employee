# 开发路线图（ROADMAP）

> 状态：Phase 2.1 完成并通过验收（2026-09-29）；Phase 1 完成并通过验收（2026-09-29）。
> 阶段顺序固定，不跳阶段。每个阶段有明确的完成标准，未达标不进入下一阶段。

---

## Phase 0 — 项目初始化 ✅

**目标**：把项目骨架、研究叙事和工程边界锁死。

内容：
- 目录结构、许可证、`.gitignore`、`.env.example`；
- Python 项目配置（`pyproject.toml`、`requirements.txt`、`requirements-dev.txt`）；
- 文档：`README.md`、`docs/ARCHITECTURE.md`、`docs/RESEARCH.md`、`docs/DATASET.md`、
  `docs/EVALUATION.md`、`docs/ROADMAP.md`；
- `ADR.md`、`FAILURE_HANDBOOK.md`、`THIRD_PARTY_LICENSES.md`；
- `config/*.yaml` 配置结构；
- `src/**` 接口占位（无业务逻辑）。

**完成标准**：所有关键文件存在；目录职责在文档中说清楚；技术栈全仓库一致。

**明确不做**：任何业务逻辑、任何实验、任何数据下载。

---

## Phase 1 — 最小 Agent 闭环 ✅

**目标**：跑通 `Intent → Router → RAG/SQL → Answer`。

内容（全部已实现）：
- `src/core/`：`state.py`（`AgentState` + 统一 `EvidenceItem` 模型 + append reducer）、
  `config_loader.py`（settings.yaml + routing_rules.yaml 的 typed 加载，支持 `${VAR}` 环境变量覆盖）、
  `llm_client.py`（OpenAI 兼容端点：Qwen / Agnes apihub 等，离线 backend 兜底；
  唯一 LLM 入口，支持本地 `.env` 注入密钥）、
  `exceptions.py`、`observability.py`（分项耗时）、`sql_executor.py`（只读 DuckDB + 语句白名单 + 行数上限 + 超时）、
  `vector_store.py`（Phase 1 进程内最小向量索引，Phase 2 切换 pymilvus 后端）；
- `src/nodes/`：`intent_understanding.py`（规则 fast lane + LLM 分类）、
  `supervisor_router.py`（最小策略）、`sql_execution.py`（Text-to-SQL + Schema + 数据字典 + Few-shot 上下文 + 安全执行）、
  `rag_retrieval.py`、`answer_generation.py`（仅基于证据作答 + 无证据诚实降级）、`clarification.py`；
- `src/tools/`：`base.py`（工具白名单注册表 + ToolResult）、`sql_tool.py`、
  `rag_tool.py`（dense 检索 + 本地语料 `data/knowledge_base/`，Phase 2 接 BGE-M3 + Milvus + Rerank）；
- `src/graph/builder.py`：最小 StateGraph + `DigitalEmployee` 入口；
- `src/api/main.py`：FastAPI `POST /chat` 最小接口；
- `scripts/import_wwi.py`：WideWorldImporters 官方 T-SQL DDL → DuckDB 方言转换 + 种子数据导入
  （官方维度表 3,364 行真实数据），自动生成 `data/schemas/*.yaml`（Schema / 数据字典 / Few-shot）；
- `scripts/generate_synthetic_seed.py`：合成业务种子数据（Sales_Customers / Purchasing_Suppliers /
  Warehouse_StockItems / Sales_Orders / Sales_Invoices / Sales_InvoiceLines，共 3,200 行，
  **明确标注 synthetic**，仅用于让 Text-to-SQL 在真实表结构上可查，不得当作真实业务指标）；
- 端到端测试 `tests/integration/test_sql_e2e.py` 与 `test_rag_e2e.py`。

**完成标准**：✅ 给定一条中文业务问题，系统走完全流程并返回带 SQL 结果 / RAG 证据的答案；
配置来自 `config/*.yaml`；LLM 调用只出现在 `llm_client.py`；12 项测试通过，ruff 全绿。

**Phase 1 边界（明确不做，留给 Phase 2+）**：
- 不使用 Neo4j 知识图谱、不做 Claim-Evidence 验证、不建评估框架、不做前端；
- RAG 用进程内最小向量索引 + 本地语料，BGE-M3 真实 Embedding 与 Milvus 服务端接入在 Phase 2；
  当前 `embedder.py` 支持本地 BGE-M3 模型路径（`EMBEDDING_MODEL` 环境变量），
  `EMBEDDING_FORCE_OFFLINE=1` 时走 hash fallback（CI 离线场景）；
- Text-to-SQL 走 LLM 生成（离线 backend 可 mock 跑通），SQL 安全校验已在 `sql_executor.py` 到位；
- `finance_expenses` 表为**明确标注的合成数据**；WWI 真实维度数据（Application_* 等 14 张表）
  已由 `import_wwi.py` 导入，业务大表（Customers/Orders/Invoices 等）由
  `generate_synthetic_seed.py` 合成补齐——两者来源不同，报告中需区分。

---

## Phase 2.1 — 真实 RAG（BGE-M3 + Milvus）✅

**目标**：把 RAG 真正接通——知识库文档 → Chunk → BGE-M3 → Milvus → Top-K 检索 →
Qwen 生成带来源的回答。本阶段使用**真实** BGE-M3 与 **真实** Milvus，
禁止用 hash/fake 兜底冒充正式功能。

内容（全部已实现）：
- `src/core/embedder.py`：BGE-M3 真实加载（支持本地模型目录，维度实测 1024）；
  正式模式模型加载失败时抛出 `EmbeddingUnavailableError`，**不做** hash 兜底；
  hash 嵌入器仅作为 `backend="fake"` 的测试模式显式保留；
- `src/core/vector_store.py`：`MilvusVectorStore`（pymilvus 3.x `MilvusClient`）+
  `FakeVectorStore`（测试模式）；统一 schema：
  `chunk_id / document_id / title / source / category / text / metadata_json / vector`；
  `connect / health_check / create_collection / insert / search / delete / size / drop`；
- `scripts/build_kb.py`：`data/knowledge_base/*.md` → 标题/段落切分 → BGE-M3 →
  Milvus 入库（collection 名称 / metric / index 全部读自 `config/settings.yaml`）；
- `src/tools/rag_tool.py`：`query → BGE-M3 → Milvus Top-K → RetrievalResult`，
  结果保留 `source / title / category / score`；
- `src/nodes/rag_retrieval.py`：读 `state.user_query` → 调 RAG Tool →
  写 `state.retrieved_context` / `state.evidence`；
- `src/nodes/answer_generation.py`：知识类回答只依据 `retrieved_context`，
  引用文档标题与 `chunk_id`；`response` 保留
  `evidence: [{source, chunk_id, score}]`；
- `GET /health`（`src/api/main.py`）：返回 `llm / embedding / milvus / database`
  分项健康状态，Milvus 分项含 `collection_name / entity_count / dimension /
  metric_type / index_type`；
- 测试：`tests/unit/test_embedder.py`、`tests/unit/test_vector_store.py`、
  `tests/unit/test_rag_tool.py`（fake 后端 + 参数校验 + 空结果 + 维度 + 结果格式），
  `tests/integration/test_real_milvus_rag.py`（真实 Milvus + BGE-M3 + LLM 全链路；
  Milvus 未启动时**明确 SKIP**，不以 mock 冒充）；
- 新增 `data/knowledge_base/hr/hr-0001-remote-work-policy.md`（合成）使"远程办公"
  类问题在知识库中可被真实召回。

**完成标准**（全部满足）：
- ✅ BGE-M3 真实加载（维度 1024 实测一致，向量已归一化）；
- ✅ Milvus 真实连接（`http://localhost:19530`，collection `enterprise_knowledge`）；
- ✅ collection 已创建（HNSW / IP）；现有 5 篇文档、35 个 chunk，`entity_count = 35 > 0`；
- ✅ 5 个真实 query 全部真实召回正确文档：差旅报销审批 / 可报销范围 / 采购流程 /
  远程办公规定 / 信息安全要求；
- ✅ 检索 → BGE-M3 → Milvus → Top-K → LLM 全链路无 hardcode 答案、无 fake 分数/来源；
- ✅ unit tests 通过；real integration 通过；文档已同步。

**本阶段明确不做**（留给 Phase 2.2+）：
- 不做 Hybrid RAG（BM25 稀疏检索、RRF 融合）；
- 不做 Reranker（BGE-Reranker-v2-M3 仅保留配置项，未启用）；
- 不做 Neo4j 知识图谱 / GraphRAG；
- 不做 Claim-Evidence Verification；
- 不修改 Phase 1 已稳定的 SQL / Intent / Router 核心逻辑。

---

## Phase 2 — Milvus RAG / DuckDB Text-to-SQL / Neo4j KG

**目标**：三种知识源的专用能力到位，职责边界不混用。

内容：
- `src/core/embedder.py`（BGE-M3）、`vector_store.py`（Milvus）；
- `scripts/build_kb.py`：知识库切分、元数据写入、向量入库；
- `src/nodes/rag_retrieval.py`：Dense + BM25 + Metadata Filter + RRF 融合 + Rerank；
- `src/nodes/sql_execution.py`：Text-to-SQL（Schema + 数据字典 + Few-shot）+ 安全校验；
- `src/core/sql_executor.py`：只读执行器（语句白名单、行数上限、超时）；
- `src/tools/kg_tool.py`：预定义 Cypher 模板（禁止 LLM 生成任意 Cypher）；
- 知识图谱构建：从 WideWorldImporters 抽取实体与关系导入 Neo4j；
- `src/core/tool_registry.py`：工具白名单与权限校验；
- 验证 `data/schemas/*.yaml` 与真实数据库一致，并修正声明。

**完成标准**：三类问题（文档问答 / 数值查询 / 关系查询）分别由对应存储正确回答，
且每条答案都能给出证据来源。

**不做**：动态路由策略、验证、评估框架、前端。

---

## Phase 3 — Task-Adaptive Routing（创新点 1）

**目标**：路由从"固定链"变成"按任务类型动态选择"。

内容：
- 完善 `src/nodes/supervisor_router.py`：意图 → 工具集合 → 执行计划；
- 落地 `config/routing_rules.yaml` 全部策略：优先级、失败接管、证据不足升级；
- 多工具编排与依赖管理（`src/nodes/tool_invocation.py`）；
- 记录 `routing_history`，为 Routing Accuracy 评估做准备；
- 固定流程版本（供消融 A 使用）作为对照配置。

**完成标准**：路由决策可解释、可记录、可通过配置切换为固定流程。

---

## Phase 4 — Claim-Evidence Verification（创新点 3）

**目标**：结论在输出前被逐条验证。

内容：
- `src/nodes/verification.py`：claim 抽取、证据匹配、支持性判定、Pass/Retry/Correct 控制；
- 统一证据模型与融合（`Result Aggregation`，支撑创新点 2）；
- 验证结果进入答案引用信息，前端可展开；
- `verification` 配置项（阈值、重试上限、失败动作）；
- 验证模块自身的漏判 / 误判测试。

**完成标准**：每条输出结论都带支持状态；无依据结论被标记而不是静默输出。

---

## Phase 5 — Evaluation（Baseline + Ablation）

**目标**：用真实实验证明模块有效性。

内容：
- 构建 `data/eval_dataset.jsonl`（约 430 条）与 Ground Truth / Evidence；
- `src/evaluation/`：`dataset.py`、`metrics.py`、`runner.py`、`reporter.py`；
- `scripts/run_eval.py`；
- Baseline 1–4 与 Ours 的实现与运行；
- 消融 A / B / C；
- 错误分析与 `FAILURE_HANDBOOK.md` 案例填充；
- 启用 CI 评估门禁（与存档 baseline 比较）。

**完成标准**：六项指标全部由真实运行产出，报告可追溯到逐条结果。

**禁止**：手填任何数值、伪造任何结果。

---

## Phase 6 — Vue3 工作台

**目标**：企业员工可用的工作台。

内容：
- `src/frontend/Vue3/`：Vue3 + TypeScript + Element Plus；
- 对话与任务提交、SSE 流式执行过程展示；
- 证据面板（命中来源、SQL、图谱路径）、验证结论面板；
- ECharts 图表渲染、报告导出；
- 与 `src/api/` 完整对接。

**完成标准**：非技术用户可独立完成任务提交、过程查看与结果导出。

---

## Phase 7 — Docker + CI

**目标**：一条命令启动全栈，CI 覆盖质量门禁。

内容：
- `docker/docker-compose.yml`：Milvus、Neo4j、API、前端；
- `docker/Dockerfile.api` 与前端构建；
- 环境变量与数据卷配置；
- CI 完善：ruff / mypy / pytest / 评估门禁；
- 启动文档与最小复现说明。

**完成标准**：干净环境下按文档可启动并跑通一条端到端任务。

---

## Phase 8 — 论文与答辩

**目标**：把研究讲清楚。

内容：
- 论文章节：背景 → 相关工作 → 方法（三个创新点）→ 实验（Baseline / Ablation）→ 结论与局限；
- 结果表格与图表全部来自 Phase 5 的真实运行；
- 局限与效度讨论（数据来源、规模、单机环境）；
- 演示脚本与答辩材料。

**完成标准**：论文中的每个数字都能指向仓库中的一次具体运行。

---

## 阶段依赖与纪律

```
Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7 → Phase 8
                     └──────── 可并行：Phase 6 前端可在 Phase 3 后提前启动 ────────┘
```

- 每个阶段结束更新 `README.md` 的 Roadmap 状态；
- 新增技术 / 框架必须先更新 `ADR.md`；
- 任何阶段都不得跳过文档更新，尤其是评估相关结论。
