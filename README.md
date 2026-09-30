# Enterprise AI Employee

**A knowledge-augmented AI digital employee for enterprise data analysis and task execution.**

企业 AI 数字员工是一个面向企业数据分析与任务执行的知识增强型 AI Agent 系统，通过任务理解、动态路由、RAG、Text-to-SQL、知识图谱、工具调用和结果验证，将自然语言任务转化为可执行的数据查询、知识检索、分析推理和报告生成流程。

> **项目状态：Phase 3 — Task-Adaptive Routing ✅ 已完成（2026-09-30）。**
> Phase 0 完成骨架与文档；Phase 1 完成 `Intent → Router → RAG/SQL → Answer` 最小闭环；
> Phase 2.1 已将 RAG 接通（BGE-M3 + 真实 Milvus）；
> Phase 2.2 将 RAG 升级为 Hybrid 检索（Dense + BM25 + RRF）；
> Phase 2.3 在 Hybrid 之上接入 BGE-Reranker-v2-M3 重排阶段（4 模式 × 56 query 检索实验）；
> Phase 2.4 接入 Neo4j 知识图谱（WWI → Neo4j 图谱构建 + 预定义 Cypher 模板
> KG Tool + 58 任务真实评测，GT 经 DuckDB SQL 独立核验）；
> **Phase 3 实现 Task-Adaptive Routing：任务特征 → 结构化 TaskProfile（LLM 提议）→
> 候选工具（能力规则）→ 路由决策（RoutingDecision 强校验 + confidence gate）→
> 多工具执行与结果融合；100 任务真实评测（A/B/C/D 四 baseline），
> 详见 `docs/phase3/EXPERIMENT_REPORT.md`。**
> 真实实验环境：BGE-M3 + Real Milvus + BM25 + RRF + BGE-Reranker-v2-M3 +
> Neo4j 5.26 + Qwen（LLM 路由车道）。
> **实测结果如实记录**：动态路由（C rule 0.630 / D LLM 0.680）route accuracy
> 显著高于静态策略（A 0.110 / B 0.220）；multi_tool 顺序拆解（plan EM 0.520）
> 与 KG 模板覆盖缺口为主要失分点（负结果保留）。
> 尚未实现：Claim-Evidence 验证、正式 430 任务评估框架与前端
> （分别为 Phase 4 / 5 / 6，见 Roadmap）。

---

## 1. 项目简介

AI 数字员工的定位不是"能回答问题的聊天机器人"，而是"能完成企业任务的执行体"。
系统接收一条自然语言企业任务，理解意图后**动态选择**知识检索（RAG）、结构化查询（Text-to-SQL）、
知识图谱（KG）与分析工具的组合，融合多源证据生成结论，并在输出前对结论逐条做
**Claim-Evidence Verification**（结论-证据验证），最终产出答案、图表或报告。

核心闭环：

```
自然语言任务
  → Intent Understanding
  → Task-Adaptive Routing
  → RAG / SQL / Knowledge Graph / Analysis
  → Tool Calling
  → Multi-source Evidence Fusion
  → Claim-Evidence Verification
  → Answer / Chart / Report
```

## 2. 项目背景

企业内的数据分散在三种形态中，而现有系统通常只覆盖其中一种：

| 数据形态 | 典型载体 | 常见系统 | 局限 |
|---|---|---|---|
| 非结构化知识 | 制度、手册、流程文档 | RAG 问答 | 无法回答需要计算的业务问题 |
| 结构化数据 | 业务数据库 | Text-to-SQL | 缺少业务口径与制度约束，易生成"正确但无意义"的查询 |
| 关系型知识 | 组织、客户、产品关系 | 知识图谱 | 覆盖面窄，单独使用价值有限 |

企业真实任务往往**同时需要三者**，并且要求结论可被追溯验证。
本项目研究的正是这一层：如何为任务动态选择知识源、如何协同多源证据、如何验证结论是否真有依据。

## 3. 项目目标

1. **任务自适应**：同一系统对不同类型的任务使用不同的工具组合，而不是固定流水线。
2. **多源协同**：文档、数据、关系、分析结果在同一证据空间中融合，而非各自为政。
3. **结果可信**：生成结论前逐条检查证据支持情况，降低无依据结论。
4. **可评估**：所有能力提升必须由自建评估框架与消融实验证明，而不是 Demo 演示。

**非目标**：不做通用聊天机器人；不做单一 RAG 问答系统；不为使用新技术而堆叠组件。

## 4. 核心能力

| 能力 | 说明 | 承载模块 |
|---|---|---|
| 意图理解 | 将自然语言任务解析为意图、槽位与约束 | `src/nodes/intent_understanding.py` |
| 任务自适应路由 | 依据任务类型动态选择工具集合与执行顺序 | `src/nodes/supervisor_router.py` |
| 企业知识检索 | Dense + BM25 + RRF 混合检索 + BGE-Reranker-v2-M3 重排（Phase 2.3） | `src/nodes/rag_retrieval.py`, `src/tools/rag_tool.py`, `src/core/reranker.py` |
| 结构化数据查询 | 只读 Text-to-SQL 与安全执行 | `src/nodes/sql_execution.py`, `src/tools/sql_tool.py` |
| 关系查询 | 基于预定义 Cypher 模板的图谱查询（Phase 2.4，禁止 LLM 生成任意 Cypher） | `src/tools/kg_tool.py`, `src/core/cypher_validator.py`, `src/core/neo4j_client.py` |
| 数据分析 | 受控 Pandas / NumPy 统计与分析函数 | `src/tools/analysis_tool.py` |
| 多源证据融合 | 统一证据模型与去重、加权 | `src/nodes/answer_generation.py` |
| 结论验证 | Claim 抽取 → 证据匹配 → 支持性判定 → Pass/Retry/Correct | `src/nodes/verification.py` |
| 结果输出 | 答案、ECharts 图表数据、结构化报告 | `src/tools/chart_tool.py`, `src/tools/report_tool.py` |
| 评估 | 统一评估框架、Baseline、Ablation | `src/evaluation/` |

## 5. 系统架构

```
┌───────────────────────────────────────────────────────────┐
│ 用户交互层        Vue3 + TypeScript + Element Plus          │
│                  对话 / 任务 / 报告 / 证据展示               │
└───────────────────────────┬───────────────────────────────┘
                            │ HTTP / SSE (FastAPI)
┌───────────────────────────▼───────────────────────────────┐
│ AI 数字员工核心层（LangGraph StateGraph + AgentState）       │
│  Intent → Router → RAG / SQL / KG → Analysis               │
│         → Evidence Fusion → Verification → Answer          │
└───────────────────────────┬───────────────────────────────┘
                            │ Tool Calling（白名单 + 权限治理）
┌───────────────────────────▼───────────────────────────────┐
│ Tool Layer  rag / sql / kg / analysis / chart / report     │
└───────────────────────────┬───────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────┐
│ 知识 / 数据层   Milvus    DuckDB    Neo4j    Pandas         │
└───────────────────────────┬───────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────┐
│ Infrastructure  LLM(Qwen) / Embedding(BGE-M3) / Reranker   │
│                 配置 / 日志 / Trace / 异常 / 评估            │
└───────────────────────────────────────────────────────────┘
```

详细设计见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

**数据库职责严格区分，不得混用：**

| 存储 | 唯一职责 |
|---|---|
| Milvus | 向量知识检索（非结构化知识） |
| DuckDB | 业务数据、Text-to-SQL、数据分析（只读） |
| Neo4j | 知识图谱与关系查询（仅预定义模板） |

## 6. 技术路线

意图理解 → 任务自适应路由 → 多工具执行 → 多源证据融合 → 结论-证据验证 → 答案/图表/报告。

每一步都是可单独评估的模块，且每个模块都必须回答"去掉它会怎样"（见第 12 节消融实验）。

## 7. 研究问题

- **RQ1** 如何根据企业任务类型动态选择 RAG、SQL、知识图谱和分析工具？
- **RQ2** 如何验证 AI 数字员工生成的结论是否真正有数据或文档证据支持？
- **RQ3** 多源知识协同相比单一知识源，对复杂企业任务的完成效果有何影响？

## 8. 创新点

研究贡献固定为以下三点，**不把第三方框架的使用当作创新**：

1. **Task-Adaptive Routing** — 依据任务类型动态选择工具组合（对应 RQ1）。
2. **Multi-source Knowledge Fusion** — 非结构化知识、结构化数据、图谱关系、分析结果的协同（对应 RQ3）。
3. **Claim-Evidence Verification** — 结论抽取、证据匹配与支持性判定（对应 RQ2）。

详见 [`docs/RESEARCH.md`](docs/RESEARCH.md)。

## 9. 数据来源

| 类别 | 来源 | 用途 |
|---|---|---|
| 企业业务数据（主） | WideWorldImporters | DuckDB、Text-to-SQL、分析、图谱构建基础 |
| 企业业务数据（辅） | AdventureWorks | 同上，用于补充多样性 |
| 企业知识库 | 公开企业手册 / 政策 / 安全规范 / 运营规范 / 业务文档（*Public Enterprise Policy Corpus*） | RAG 语料 |
| Text-to-SQL Benchmark | Spider, BIRD | 能力对照 |
| RAG / Evidence Benchmark | HotpotQA | 检索与证据对照 |
| Tool Calling Benchmark | BFCL, ToolBench | 工具调用能力对照 |
| Agent Benchmark | GAIA | 复杂任务对照 |
| 自建评估集 | Enterprise AI Employee Task Dataset（约 430 条） | 主评估集 |

**不伪造"某家真实企业的内部数据"。** 知识库语料一律标注为公开文档或合成企业知识文本。
第三方资源按原许可证使用，登记于 [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md)，详见 [`docs/DATASET.md`](docs/DATASET.md)。

## 10. Evaluation

自建统一评估框架，核心指标：

1. Task Success Rate
2. Routing Accuracy
3. Retrieval Recall@K
4. SQL Execution Accuracy
5. Evidence Support Rate
6. Average Latency

> 最终系统级指标（Task Success Rate 等）在 Phase 5 由 `src/evaluation/runner.py`
> 真实运行产生；在 Phase 5 之前**没有系统级实验数值**。
> 检索层的阶段性实验数值（Recall@K / MRR / NDCG@5）由
> `scripts/run_hybrid_eval.py`（Phase 2.2）与 `scripts/run_retrieval_eval.py`
> （Phase 2.3）生成，结果写入 `artifacts/`（本地保留，不进 Git）。

详见 [`docs/EVALUATION.md`](docs/EVALUATION.md)。

### Phase 2.2 Hybrid Retrieval 局部实验（真实数值）

Phase 2.2 在**当前真实实验环境**下完成了 Hybrid RAG 的阶段性检索实验
（35 chunks / 10 queries，后续语料扩大后重跑结果见
`docs/EVALUATION.md` §14）：

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| Dense | 0.3833 | 0.8500 | 0.9500 | 0.9500 |
| BM25 | 0.4500 | 0.7667 | 1.0000 | 0.9500 |
| Hybrid | 0.3833 | 0.8500 | 0.9500 | 0.9500 |

**如实说明（不做无依据结论）：**

- 在 35-chunk 小规模语料上，**Hybrid 与 Dense 指标一致，未表现出普遍增益**；
  语料扩大后（190 chunks / 56 queries）重测，Hybrid 已优于单一通道（§15）。
- 该实验是 **Phase 2.2 的局部检索实验**，**不等价于最终 430 条 Evaluation
  Benchmark**（后者在 Phase 5 构建）。

完整实验设置、Ground Truth、指标与局限见
[`docs/EVALUATION.md`](docs/EVALUATION.md) 的 "Phase 2.2 Hybrid Retrieval
Experiment" 一节；逐 query 结果由 `scripts/run_hybrid_eval.py` 生成。

### Phase 2.3 Reranker 局部实验（真实数值）

Phase 2.3 在扩大后的知识库（**27 docs / 190 chunks**）与 56 条人工核验
Ground Truth 上，对比 4 种检索模式（同一 query / 同一语料 / 同一 final_k=5）：

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR | NDCG@5 |
|---|---:|---:|---:|---:|---:|
| Dense | 0.5908 | 0.8259 | 0.8988 | 0.9062 | 0.8657 |
| BM25 | 0.6473 | 0.8705 | 0.9077 | 0.9345 | 0.8965 |
| Hybrid (Dense+BM25+RRF) | 0.6622 | 0.8735 | 0.9286 | 0.9500 | 0.9120 |
| **Hybrid + Reranker** | **0.6711** | 0.8646 | 0.9271 | **0.9613** | 0.9087 |

**如实说明（保留负结果，不做普遍化结论）：**

- 在本 190-chunk 合成企业语料上，**Hybrid 的 Recall@5 / NDCG@5 高于单一
  Dense 与 BM25**，RRF 融合有稳定收益。
- **Reranker 提高了 MRR（+0.0113），但未提高 Recall@5（−0.0015）与
  NDCG@5（−0.0033）**——说明其作用是改善候选排序位置，而非扩大召回覆盖。
- hard 难度 query 上 Reranker 未表现出稳定净收益；CPU 环境下
  Reranker 单 query 延迟约 5.7s（工程指标，不替代检索质量）。
- 完整设置、按类型/难度切片与失败案例分析见
  [`docs/EVALUATION.md`](docs/EVALUATION.md) §15 与
  [`docs/phase2.3/EXPERIMENT_REPORT.md`](docs/phase2.3/EXPERIMENT_REPORT.md)。

### Phase 2.4 知识图谱局部实验（真实数值）

58 任务评测集（`data/eval/kg_eval.jsonl`，8 类 task_type ×
easy/medium/hard，GT 全部由 DuckDB SQL 独立核验）；
真实 Neo4j 5.26 + `KGTool` 预定义模板（禁止 LLM 生成任意 Cypher）：

| 指标 | 值 |
|---|---|
| Exact Match（overall） | 0.2931 |
| Avg Precision / Recall / F1 | 0.6257 / 0.4993 / 0.5167 |
| Path Accuracy（已定义任务） | 0.9238 |
| 失败（unroutable，保留在分母） | 5 |
| 延迟 mean / p50 / p95 | 3.36 / 1.98 / 3.22 ms |

- two_hop 表现最好（exact 0.625 / F1 0.825 / path 0.958）；
- multi-hop / cross-entity 复合查询（集合交集 / 双边聚合）是
  当前模板层瓶颈，exact 为 0（**负结果保留**）；
- Path Accuracy 显著高于 Exact Match：图遍历可靠，失分集中在
  远端实体集合精确匹配与复合答案口径。
- 完整失败分析（6 个真实案例）见
  [`docs/phase2.4/EXPERIMENT_REPORT.md`](docs/phase2.4/EXPERIMENT_REPORT.md) §10 与
  [`FAILURE_HANDBOOK.md`](FAILURE_HANDBOOK.md) §4（FH-KG-001/002/003）。

### Phase 3 任务自适应路由局部实验（真实数值）

100 任务路由评测集（`data/eval/routing_eval.jsonl`，8 route 类型 ×
10 task_type × easy/medium/hard，GT 人工按任务语义声明）；
4 baseline：A Static RAG / B Static SQL / C Rule-based / D LLM Adaptive
（Qwen 经 Agnes 代理）：

| 系统 | Route Acc | Tool F1 | Plan EM | Clarif Acc | Task Success | 延迟 mean |
|---|---|---|---|---|---|---|
| A Static RAG | 0.110 | 0.172 | 0.210 | — | 0.110 | 0 ms |
| B Static SQL | 0.220 | 0.508 | 0.320 | — | 0.220 | 0 ms |
| C Rule-based | 0.630 | 0.790 | 0.590 | — | 0.570 | 6.5 ms |
| D LLM Adaptive | 0.680 | 0.781 | 0.520 | 0.880 | 0.520 | 8605 ms |

- 动态路由（C/D）route accuracy 显著高于静态（A/B），验证 RQ1 可行性；
- LLM 的边际价值在歧义消解（澄清精度 0.880）而非 route_type 本身；
- multi_tool 顺序拆解不稳定（plan EM 0.520 < route_acc 0.680）；
- KG 模板覆盖缺口（Phase 2.4 遗留）传导为部分关系任务的路由失败
  （**负结果保留**）。
- 完整消融（含 E 无校验消融）与 10 个真实失败案例见
  [`docs/phase3/EXPERIMENT_REPORT.md`](docs/phase3/EXPERIMENT_REPORT.md)。

## 11. Baseline

| 编号 | 系统 |
|---|---|
| Baseline 1 | LLM |
| Baseline 2 | LLM + RAG |
| Baseline 3 | Hybrid RAG + SQL |
| Baseline 4 | RAG + SQL + KG |
| Ours | Task-Adaptive Routing + RAG + SQL + KG + Analysis + Claim-Evidence Verification |

## 12. Ablation

| 实验 | 移除模块 | 回答的问题 |
|---|---|---|
| Ablation A | Task-Adaptive Routing | 动态路由是否必要 |
| Ablation B | Knowledge Graph | 图谱关系是否必要 |
| Ablation C | Claim-Evidence Verification | 结果验证是否必要 |

## 13. 技术栈

| 层 | 选型 |
|---|---|
| 语言 | Python 3.11+ / TypeScript |
| LLM | Qwen（统一经 `src/core/llm_client.py` 调用） |
| Agent 编排 | LangGraph（StateGraph），LangChain 仅作必要基础组件 |
| Embedding | BGE-M3 |
| Reranker | BGE-Reranker-v2-M3（Phase 2.3 已实现，默认 `reranker.enabled=false`，正式实验时开启） |
| RAG | Dense + BM25 + RRF + Reranker（Phase 2.2 Hybrid；Phase 2.3 重排） |
| 向量库 | Milvus |
| 知识图谱 | Neo4j + Cypher（仅预定义模板） |
| 业务数据库 | DuckDB（只读） |
| 分析 | Pandas + NumPy（受控函数） |
| 后端 | FastAPI + Pydantic + SSE |
| 前端 | Vue3 + TypeScript + Element Plus + ECharts |
| 评估 | pytest + 自建 Evaluation Framework |
| 部署 | Docker + Docker Compose |
| 工程 | Git / GitHub / GitHub Actions / ruff / mypy |
| 许可证 | Apache License 2.0 |

**明确不使用**：Kubernetes、Helm、Kafka、ClickHouse、LakeFS、DVC、Airflow、Chroma、
Redis 作为核心数据库、LLM 多主模型集群、Prometheus/Grafana/Loki/Tempo 全家桶、
复杂微服务、自动微调、复杂多模态、复杂 RBAC、CDC、等保级合规系统。

## 14. 项目结构

```
enterprise-ai-employee/
├── .github/workflows/ci.yml      CI：lint / type-check / test（评估门禁 Phase 5 启用）
├── config/                       系统配置中心（只控制行为，不含业务逻辑）
│   ├── settings.yaml             全局配置
│   ├── routing_rules.yaml        Agent 路由规则（创新点 1 的策略声明）
│   ├── tool_registry.yaml        Tool 白名单与治理配置
│   └── prompt_templates.yaml     Prompt 模板及版本
├── data/                         数据层
│   ├── knowledge_base/           企业知识库文档（hr / finance / operations / security / business）
│   ├── schemas/                  数据库 Schema、数据字典、SQL Few-shot
│   ├── synthetic/                任务生成 / Ground Truth 生成 / 受控扩充
│   └── eval_dataset.jsonl        统一评估集（约 430 条）
├── docs/                         研究与设计文档
│   ├── ARCHITECTURE.md           系统架构
│   ├── RESEARCH.md               研究问题、创新点、方法
│   ├── DATASET.md                数据来源与构建
│   ├── EVALUATION.md             指标、Baseline、消融
│   └── ROADMAP.md                开发路线图
├── src/
│   ├── core/                     系统底座（LLM / Embedding / Milvus / DuckDB / State / 日志）
│   ├── nodes/                    数字员工的大脑（理解 / 路由 / 检索 / 执行 / 生成 / 验证）
│   ├── graph/                    LangGraph 工作流编排与状态持久化
│   ├── tools/                    数字员工的执行能力（"手"）
│   ├── evaluation/               评估系统
│   ├── api/                      FastAPI 接口（供前端调用）
│   └── frontend/Vue3/            企业 AI 工作台
├── tests/                        unit / integration
├── scripts/                      build_kb.py / run_eval.py
├── docker/docker-compose.yml     本地统一运行环境
├── README.md  ADR.md  FAILURE_HANDBOOK.md  THIRD_PARTY_LICENSES.md  LICENSE
└── .env.example  .gitignore  requirements.txt  requirements-dev.txt  pyproject.toml
```

### 目录职责

| 目录 | 职责 |
|---|---|
| `data/` | 数字员工使用的业务数据、知识库、评估数据与数据库 Schema |
| `config/` | 控制系统行为的配置文件，**不承载核心业务逻辑** |
| `src/core/` | 系统底座：LLM、Embedding、Milvus、DuckDB、State、日志、异常 |
| `src/nodes/` | 数字员工的大脑：理解、路由、检索、执行、生成、验证 |
| `src/graph/` | LangGraph 工作流编排与任务状态持久化 |
| `src/tools/` | 数字员工的执行能力（"手"）：Agent 决策后通过 Tool 完成实际工作 |
| `src/evaluation/` | 用于**证明系统有效**，而不是只展示 Demo |
| `src/api/` | 给前端提供 HTTP / SSE 接口 |
| `src/frontend/` | 企业员工使用数字员工的工作台 |
| `tests/` | 单元测试与完整链路测试 |
| `scripts/` | 数据 / 知识库构建与评估辅助脚本 |
| `docker/` | 本地统一运行环境 |
| `docs/` | 项目研究、架构、数据与实验文档 |

## 15. Roadmap

| 阶段 | 内容 | 状态 |
|---|---|---|
| Phase 0 | 项目初始化：骨架、文档、配置、边界 | ✅ 已完成 |
| Phase 1 | 最小 Agent 闭环：Intent → Router → RAG/SQL → Answer | ✅ 已完成 |
| Phase 2.1 | 真实 RAG：BGE-M3 + Milvus 端到端接通 | ✅ 已完成 |
| Phase 2.2 | Hybrid RAG：Dense + BM25 + RRF | ✅ 已完成 |
| Phase 2.3 | Reranker：知识库扩大 + BGE-Reranker-v2-M3 + 4 模式检索实验 | ✅ 已完成 |
| Phase 2.4 | Neo4j Knowledge Graph：图谱构建 + KG Tool + 安全 Cypher + 58 任务评测 | ✅ 已完成 |
| Phase 3 | Task-Adaptive Routing：Task Profile + 动态路由 + 多工具融合 + 100 任务评测 | ✅ 已完成 |
| Phase 4 | Claim-Evidence Verification | ⬜ 未开始 |
| Phase 5 | Evaluation：Baseline + Ablation | ⬜ 未开始 |
| Phase 6 | Vue3 工作台 | ⬜ 未开始 |
| Phase 7 | Docker + CI | ⬜ 未开始 |
| Phase 8 | 论文与答辩 | ⬜ 未开始 |

详见 [`docs/ROADMAP.md`](docs/ROADMAP.md)。

## 16. License

项目源码采用 **Apache License 2.0**，见 [`LICENSE`](LICENSE)。

第三方数据、模型与依赖继续遵循其原始许可证。

## 17. Third-party Resources

所有第三方资源（数据集、模型、Python / JS 依赖）的来源、许可证、用途与署名要求，
统一登记于 [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md)。

> 说明：LangGraph、Milvus、Neo4j 等框架是本项目的**工程基础设施**，不是研究贡献。
> 本项目的创新点在 Task-Adaptive Routing、Multi-source Knowledge Fusion 与
> Claim-Evidence Verification。
