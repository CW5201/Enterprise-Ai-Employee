# 数据集与数据来源（DATASET）

> 状态：Phase 2.2 已更新（2026-09-30）。
>
> **当前仓库数据资产（已入库）**：
> - `data/raw/wwi_ddl/` — WideWorldImporters 官方 T-SQL DDL 与种子脚本
>   （本地下载，已被 `.gitignore` 忽略，**不进 Git**）；
> - `data/runtime/wwi.duckdb` — 由 `scripts/import_wwi.py` 生成的 DuckDB 运行时数据库
>   （含官方维度数据 3,364 行 + 由 `scripts/generate_synthetic_seed.py` 补齐的
>   合成业务数据 3,200 行，**同样不入库**）；
> - `data/schemas/*.yaml` — 数据库 Schema、数据字典、Text-to-SQL Few-shot（**已入库**，
>   Few-shot 状态为 `verified`，即已在真实数据库上执行验证）；
> - `data/knowledge_base/*/` — 企业知识库文档，全部为**合成语料**（synthetic），
>   文档头部显式标注"非真实企业制度"，**不得描述为任何真实企业的内部数据**。
>   Phase 2.1 新增 `hr/hr-0001-remote-work-policy.md`（员工远程办公管理办法），
>   与 finance / operations / security 文档一同由 `scripts/build_kb.py`
>   切分、BGE-M3 嵌入后写入 Milvus（`enterprise_knowledge` collection，
>   当前 5 篇文档、35 个 chunk，`entity_count > 0`）；
> - `data/eval/hybrid_eval.jsonl` — Phase 2.2 Hybrid Retrieval 的**局部实验集**
>   （10 个 query + 人工核验的 `expected_chunk_ids`），由
>   `scripts/build_hybrid_eval.py` 生成，**仅用于 35-chunk 语料上的
>   Dense / BM25 / Hybrid 对比**，**不是**最终 430 条 Evaluation Dataset
>   （后者在 Phase 5 由 `data/eval_dataset.jsonl` 构建）。
>
> **合成数据红线**：
> - `generate_synthetic_seed.py` 生成的 Sales_Customers / Purchasing_Suppliers /
>   Warehouse_StockItems / Sales_Orders / Sales_Invoices / Sales_InvoiceLines
>   数据为确定性合成（synthetic），仅用于让 Text-to-SQL 全链路在真实表结构上跑通；
>   任何统计结论必须标注"基于合成种子数据"，不得当作真实业务指标引用；
> - WideWorldImporters 官方维度表（Application_* / 部分 Warehouse_*）为微软公开示例库
>   的真实数据，与合成表来源不同，报告中需区分。
>
> 本文档其余部分（来源、许可证、评估集构建规则）为 Phase 0 边界声明，
> 后续 Phase 的数据下载与评估集构建仍以本文件为准。

---

## 0. Phase 2.2 局部检索实验集（`data/eval/`）

| 项 | 说明 |
|---|---|
| 文件 | `data/eval/hybrid_eval.jsonl`（10 query + `expected_chunk_ids`） |
| 语料 | 当前企业知识库（35 chunks，`data/knowledge_base/*.md`） |
| 用途 | 仅用于 Phase 2.2 的 Hybrid Retrieval 阶段实验（Dense / BM25 / Hybrid 对比） |
| 生成 | `scripts/build_hybrid_eval.py`；指标由 `scripts/run_hybrid_eval.py` 计算 |
| Ground Truth | 人工核验的 `expected_chunk_ids`，**不因实验结果而修改** |

**边界（重要）**：

- 这只是 **Hybrid Retrieval 的阶段性实验集**（10 query），
  **不属于正式 430 条 Evaluation Dataset**；
- 正式 430 条 Evaluation Dataset 在 **Phase 5** 由 `data/eval_dataset.jsonl`
  构建（见第 11 节）；
- 不得把 10-query 的召回指标当作最终系统性能对外引用。

---

## 1. 数据来源总览

| 类别 | 名称 | 本项目用途 | 当前状态 |
|---|---|---|---|
| 业务数据（主） | WideWorldImporters | DuckDB、Text-to-SQL、Analysis、KG 构建基础 | 已导入 DuckDB（`data/runtime/wwi.duckdb`，本地生成，不入库） |
| 业务数据（辅） | AdventureWorks | 补充表结构与任务多样性 | 未下载 |
| 企业知识库 | Public Enterprise Policy Corpus | RAG 语料（Milvus） | 未收集 |
| Text-to-SQL | Spider | 能力对照 | 未下载 |
| Text-to-SQL | BIRD | 能力对照 | 未下载 |
| RAG / Evidence | HotpotQA | 检索与证据对照 | 未下载 |
| Tool Calling | BFCL | 工具调用能力对照 | 未下载 |
| Tool Calling | ToolBench | 工具调用能力对照 | 未下载 |
| Agent | GAIA | 复杂任务对照 | 未下载 |
| 自建评估集 | Enterprise AI Employee Task Dataset | 主评估集（约 430 条） | 未构建（Phase 5） |
| Phase 2.2 局部检索实验集 | `data/eval/hybrid_eval.jsonl` | 35-chunk 语料上 Dense/BM25/Hybrid 对比 | 已生成（10 query，已入库） |

许可证与署名要求统一登记于 [`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md)。

---

## 2. WideWorldImporters（主业务数据源）

- 内容：微软发布的示例企业数据库（销售、采购、库存、客户、员工、城市等）。
- 用途：
  - 导入 DuckDB，作为唯一的结构化业务数据来源；
  - Text-to-SQL 的目标数据库；
  - 数据分析任务（趋势、对比、分布、相关性）的事实来源；
  - 知识图谱实体与关系的构建基础（客户、产品、员工、部门、城市）。
- 许可证：见 `THIRD_PARTY_LICENSES.md`（按其原始许可证使用，本项目不重新授权）。
- 注意：本项目**不修改**其原始语义；如有派生表（如统一的指标视图），
  必须在 `data/schemas/data_dictionary.yaml` 中显式声明派生规则。

## 3. AdventureWorks（辅助业务数据源）

- 内容：微软发布的示例企业数据库（生产、销售、人力资源）。
- 用途：补充 WideWorldImporters 未覆盖的表结构（如生产、人力资源），
  增加 SQL 与分析任务的多样性。
- 边界：与 WideWorldImporters 分库导入，**不混合成单一 schema**，
  任务中必须标明目标数据库，避免 Schema 混淆带来的评估噪声。

## 4. Public Enterprise Policy Corpus（企业知识库）

企业知识库是一个**公开文档语料库**，由以下类型的公开材料构成：

- 公开的企业员工手册；
- 公开的企业政策与制度文本；
- 公开的信息安全规范 / 标准 / 指南；
- 公开的运营规范与流程说明；
- 公开的产品与业务文档。

按部门归类到 `data/knowledge_base/{hr,finance,operations,security,business}/`。

**命名与表述纪律（重要）**：
- 该语料统一称为 **Public Enterprise Policy Corpus** 或 **Synthetic Enterprise Knowledge**；
- **绝不声称这些文档来自某一家真实企业的内部数据**；
- 若为合成文本，必须在文档元数据中标注 `synthetic: true`；
- 每份文档必须登记 `doc_id / source / source_url / license / attribution / effective_date`。

## 5. Spider

- 类型：跨域 Text-to-SQL 基准（多数据库、多 schema）。
- 用途：作为 Text-to-SQL 能力的**外部对照**，检验方法不是只对本项目 schema 过拟合。
- 使用约束：按其原始许可证与访问规则使用；不重新分发、不重新授权。

## 6. BIRD

- 类型：大规模、贴近真实场景的 Text-to-SQL 基准（含外部知识、较多脏数据）。
- 用途：补充 Spider 之外的难度与真实度对照；尤其关注"需要业务知识才能写对 SQL"的场景，
  与本研究的数据字典约束思路直接相关。
- 使用约束：同上。

## 7. HotpotQA

- 类型：多跳问答数据集，带有支持性事实（supporting facts）标注。
- 用途：
  - 作为 **证据支持判定（Evidence Support Rate）** 的外部对照，
    因为其自带 supporting facts，可检验验证模块在标准数据上的表现；
  - 多跳检索能力的对照。
- 使用约束：同上。仅使用英文原始数据，不翻译后重新分发。

## 8. BFCL

- 类型：Berkeley Function Calling Leaderboard，工具/函数调用评测。
- 用途：检验工具选择与参数填充能力的外部对照，与创新点 1（路由）相关。
- 使用约束：同上，遵循其排行榜与数据使用规则。

## 9. ToolBench

- 类型：大规模工具调用指令数据（含多工具、多步调用）。
- 用途：多步工具编排能力的对照。
- 使用约束：同上。

## 10. GAIA

- 类型：通用 AI 助手基准，需要多步推理、工具使用与信息整合。
- 用途：复杂 Agent 任务的外部对照，检验系统在开放式复杂任务上的表现上界与差距。
- 使用约束：同上；其部分答案为隐藏答案，评测遵循官方提交规则，
  **不得把答案集并入本项目训练/提示材料**。

---

## 11. Enterprise AI Employee Task Dataset（自建主评估集）

### 11.1 定位

基于 WideWorldImporters / AdventureWorks 的 Schema 与公开企业知识库构建的企业任务数据集，
目标规模约 430 条，用于评估完整系统、Baseline 与消融实验。

### 11.2 任务类别

| 类别 | 说明 | 主要涉及工具 |
|---|---|---|
| Information Query | 单点事实查询 | sql |
| Knowledge QA | 制度、流程、规范问答 | rag |
| Text-to-SQL | 自然语言转 SQL 查询 | sql |
| Data Analysis | 趋势、对比、分布、相关性 | sql + analysis |
| Multi-source Reasoning | 需跨文档 / 数据 / 关系 | rag + sql + kg |
| Complex Agent Task | 多步任务，产出报告或图表 | 全部 + chart/report |

### 11.3 数据结构

`data/eval_dataset.jsonl`，每行一条 JSON：

```json
{
  "id": "task-0001",
  "category": "multi_source_reasoning",
  "difficulty": "hard",
  "question": "...",
  "expected_tools": ["rag", "sql", "analysis"],
  "ground_truth": "...",
  "evidence": [
    {"evidence_id": "ev-0001-1", "source_type": "duckdb", "source_ref": "sql:...", "content": "..."}
  ],
  "metadata": {
    "database": "wide_world_importers",
    "doc_ids": ["finance-0003"],
    "requires_verification": true,
    "created_by": "rule|manual|benchmark-derived",
    "dataset_version": "v0.0.0"
  }
}
```

字段约束：
- `expected_tools` 是评估 Routing Accuracy 的参考值，必须由规则或人工确定并记录来源；
- `evidence` 中的 `evidence_id` 必须稳定可复现，否则验证模块无法评估；
- `metadata.created_by` 记录任务来源，便于统计"生成 vs 人工"比例。

### 11.4 构建方式

1. **规则生成**（`data/synthetic/generate_tasks.py`）：依据 Schema、数据字典口径、
   文档元数据组合生成任务模板；
2. **Ground Truth 生成**（`data/synthetic/generate_ground_truth.py`）：
   SQL 类任务必须**真实执行**参考查询得到答案；文档类任务必须绑定到具体 doc_id 与片段；
3. **受控扩充**（`data/synthetic/augment_data.py`）：改写、时间范围偏移、部门偏移、
   难度升级，扩充后必须重新验证；
4. **人工校验**：抽样人工检查语义正确性与答案可验证性，并记录抽样比例与一致性结果。

批量生成的任务若无法验证，一律丢弃而不是保留待定。

## 12. License

| 主体 | 许可证 |
|---|---|
| 本项目源码 | Apache License 2.0 |
| 第三方数据 / 模型 / 依赖 | 各自原始许可证（登记于 `THIRD_PARTY_LICENSES.md`） |

原则：
- **不把第三方数据打包后以本项目名义重新授权**；
- 不在仓库中提交体积大或再分发受限的原始数据，改为提供下载脚本与来源说明；
- 派生数据必须说明派生规则与原始许可证约束。

## 13. 数据版本

- 评估集带版本号（`metadata.dataset_version`，当前声明为 `v0.0.0`）；
- 每一轮实验报告必须记录：评估集版本、Schema 版本、知识库快照版本、Prompt 版本；
- 版本不一致的实验结果不得直接对比。

## 14. Ground Truth

Ground Truth 只允许以下三种来源：

1. **真实数据库执行结果**（SQL 类任务）：必须实际运行并保存结果快照；
2. **公开文档原文**（知识类任务）：必须绑定 doc_id 与片段，可回溯到原文；
3. **明确记录的生成规则**（组合类任务）：规则本身写入数据集元数据。

禁止：手工臆造答案、用模型输出作为 Ground Truth、用待评估系统的输出反向构造答案。

## 15. 数据泄漏风险

| 风险 | 控制措施 |
|---|---|
| Few-shot 示例出现在评估集 | `data/schemas/sql_fewshots.yaml` 中的示例必须从评估集排除 |
| 扩充任务的改写家族跨越训练/评估 | 按改写家族分组切分，同族不同时出现在两侧 |
| 提示词中泄露参考证据 | 提示只给 Schema 与字典，不给参考答案 |
| 使用第三方 Benchmark 时答案泄露（如 GAIA 隐藏答案） | 遵循官方评测规则，不并入提示材料 |
| 用模型自己生成评估集再用同一模型评估 | 关键任务人工校验并记录校验比例 |

## 16. 第三方资源管理

- 每个第三方资源在 `THIRD_PARTY_LICENSES.md` 中登记 `Resource / Type / Source / License / Usage / Notes`；
- 下载脚本（Phase 2+）必须记录来源 URL 与获取日期，不提交原始数据到仓库；
- 使用第三方 Benchmark 的结果在论文中必须注明其名称、版本与许可证；
- 新增第三方资源前，先确认许可证允许该用途，并在文档中登记。
