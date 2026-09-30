# `data/eval/enterprise_tasks.jsonl` — 统一企业任务 Benchmark（Phase 5）

本文件是**整个项目正式的统一评测集**，由 `scripts/build_enterprise_tasks.py`
生成、`scripts/validate_enterprise_tasks.py` 校验。它驱动 Phase 5 的端到端
评测（`scripts/run_enterprise_eval.py`）与 Baseline / Ablation 实验。

> 与 `retrieval_eval.jsonl` / `kg_eval.jsonl` / `routing_eval.jsonl` /
> `verification_eval.jsonl` 等**阶段性实验集不同**——那些是各 Phase 的局部
> 实验集，本文件是正式 430+ 条的完整 Benchmark，不得用阶段性结果代替。

## 规模与分布（当前 487 条，`python scripts/build_enterprise_tasks.py` 生成）

- **by_task_type**：knowledge_lookup / structured_lookup / aggregation /
  relationship_query / statistical_analysis / trend_analysis / comparison /
  report_generation / multi_source_analysis / ambiguous_task 全部覆盖；
- **by_difficulty**：easy / medium / hard；
- **by_route**：rag / sql / kg / multi_tool / clarification；
- **by_num_tools**：0（clarification）/ 单工具 / 双工具 / 三工具 / 四工具。

## 每条记录字段

```json
{
  "id": "ent-0001",
  "question": "...",
  "domain": "customers | suppliers | orders | ...",
  "task_type": "knowledge_lookup | ... | ambiguous_task",
  "difficulty": "easy | medium | hard",
  "expected_route": "rag | sql | kg | analysis | multi_tool | clarification",
  "expected_tools": ["sql"],
  "expected_tool_order": ["sql"],
  "expected_clarification": false,
  "clarification_question": null,
  "expected_answer": "...",
  "expected_evidence": ["sql"],
  "expected_claims": [{"text": "...", "claim_type": "numerical",
                       "importance": "critical", "value": 500}],
  "provenance": {"verifier": "independent_sql", "sql": "...", "value": 500}
}
```

## Ground Truth 独立验证（不来自任何系统输出）

所有 GT 由三类**独立**验证器产生，绝不取自 Router / Verification / Answer
任一环的输出（ADR-009 纪律）：

| `verifier` | 说明 |
|---|---|
| `independent_sql` | 业务数值 GT：由**独立** DuckDB 连接用手写 SQL 计算，构建时断言与 live 表一致，DB 漂移时构建失败 |
| `independent_sql_crosscheck` | 关系型 GT：KG 声明与独立 SQL 双向核对，防止图谱单独幻觉 |
| `independent_sql+kb_rule` | 多源 GT：SQL 数值 + KB 关键词包含规则组合 |
| `kb_containment` | RAG GT：知识库语料关键词包含断言（非检索结果），记录 `expected_chunk` |
| `manual` | 模糊任务 GT：人工声明需澄清 |

## 数据 provenance（必须保留）

- 企业知识：`data/knowledge_base/*.md`（**合成**，明确标注 synthetic）；
- 企业业务：WWI 真实维度表 + `generate_synthetic_seed.py` 合成的业务大表
  （`data/runtime/wwi.duckdb`，**明确标注 synthetic**）；
- 图谱：WWI → Neo4j（`scripts/build_kg.py`，命名空间 `__graph='eae'`）。

每条任务通过 `provenance` 字段记录其 GT 的验证来源，可回溯。

## 命令

```bash
python scripts/build_enterprise_tasks.py          # 生成并校验
python scripts/build_enterprise_tasks.py --check  # 只重校验 GT（live 漂移检测）
python scripts/validate_enterprise_tasks.py        # 结构 + 无泄漏校验
```

## 边界

- 本文件是**唯一**正式企业任务 Benchmark；阶段性实验集不得替代；
- 数值型业务结论（销售额 / 数量 / 排名）来自**合成**业务种子数据，只用于
  验证系统流程与 GT 一致性，不得当作真实经营指标对外引用；
- 任务 GT 先于任何系统运行固定，实验结果不得反向修改 GT。
