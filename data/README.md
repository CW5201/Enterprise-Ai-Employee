# data/ — 数据层

数字员工使用的**业务数据、知识库、评估数据和数据库 Schema**。

| 子目录 | 职责 |
|---|---|
| `knowledge_base/` | 企业非结构化知识文档（人事、财务、运营、安全、业务），是 RAG 的唯一语料来源 |
| `schemas/` | 企业数据库 Schema、数据字典、Text-to-SQL Few-shot，是 SQL 生成的结构依据 |
| `synthetic/` | 任务生成、Ground Truth 生成、受控数据扩充脚本（当前仅为占位） |
| `eval_dataset.jsonl` | 统一评估集，目标约 430 条企业任务 |

## 数据边界（必须遵守）

- **不伪造真实企业内部数据。** `knowledge_base/` 使用公开企业手册 / 政策 / 规范文档，统一定义为
  *Public Enterprise Policy Corpus*，绝不声称其来自某家真实企业的内部。
- **业务数据来自公开示例库**：WideWorldImporters（主）、AdventureWorks（辅）。
- **第三方 Benchmark**（Spider / BIRD / HotpotQA / BFCL / ToolBench / GAIA）只按原许可证使用，
  不打包重新授权。

详见 `docs/DATASET.md` 与 `THIRD_PARTY_LICENSES.md`。

## Ground Truth 规则

Ground Truth 只能来自：
1. 真实数据库查询结果；
2. 公开文档原文；
3. 明确记录的生成规则。

禁止手工臆造答案，禁止手填任何未经实验得到的指标数值。
