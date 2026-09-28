# 第三方许可证说明（THIRD_PARTY_LICENSES）

本项目**源码**采用 **Apache License 2.0**（见 `LICENSE`）。
第三方数据、模型与依赖**继续遵循其原始许可证**，不在本项目名义下重新授权、重新分发。

登记规则：
- 每引入一个第三方资源，必须在此表登记，再进入代码或数据目录；
- 大型或再分发受限的原始数据**不提交进仓库**，只保留下载脚本与来源说明；
- "Notes" 中记录使用限制（如仅限研究、禁止商用、须署名等）。

## 资源登记表

| Resource | Type | Source | License | Usage | Notes |
|---|---|---|---|---|---|
| WideWorldImporters | 数据库 | Microsoft (github.com/microsoft/WideWorldImporters) | Microsoft Sample Code / MS-LPL（按其仓库条款） | DuckDB 导入，Text-to-SQL、分析、KG 构建基础（主业务数据源） | 需记录下载版本；不修改原始语义；派生表须在数据字典声明 |
| AdventureWorks | 数据库 | Microsoft (docs / 官方示例) | Microsoft 示例数据条款 | 补充表结构与任务多样性（辅业务数据源） | 与 WWI 分库，不混合 schema；按原条款使用 |
| Spider | 数据集 | Yu et al., NAACL 2018 | 原始研究数据集条款 | Text-to-SQL 能力外部对照 | 仅研究用途；不复分发；注明版本 |
| BIRD | 数据集 | 官方仓库 (bird-benchmark) | 按仓库条款 | 真实场景 Text-to-SQL 对照 | 注意其中部分数据访问限制 |
| HotpotQA | 数据集 | Yang et al. (HotpotQA) | 按原始授权（研究用途） | 多跳检索与证据支持对照 | 使用原始英文数据，不翻译后重新分发 |
| BFCL | 数据集/榜单 | Berkeley Function Calling Leaderboard | 按官方条款 | 工具调用能力对照 | 遵循其评测与数据使用规则 |
| ToolBench | 数据集 | Qin et al. (ToolBench) | 按仓库条款 | 多工具、多步调用对照 | 不复分发 |
| GAIA | 数据集 | GAIA 基准 | 按官方条款 | 复杂 Agent 任务对照 | 隐藏答案不得并入提示/训练材料 |
| 公开企业制度文档（Public Enterprise Policy Corpus） | 文档 | 各公开来源（手册/政策/规范） | 各来源原始许可证 | RAG 语料（`data/knowledge_base/`） | 每份文档单独登记 `source / license / attribution`；合成文本标注 `synthetic: true`；绝不声称来自真实企业内部 |
| BGE-M3 | 模型 | BAAI (Hugging Face: BAAI/bge-m3) | Apache-2.0 | Embedding（经 `src/core/embedder.py` 封装） | 本地或下载；注明版本 |
| BGE-Reranker-v2-M3 | 模型 | BAAI (Hugging Face: BAAI/bge-reranker-v2-m3) | 模型条款（以官方为准） | Rerank | 注明版本与来源 |
| Qwen | 模型 | 阿里 Qwen（API 或指定版本） | 依所用渠道条款 | 主 LLM（仅经 `src/core/llm_client.py` 调用） | 依渠道（API / 本地）确认许可；记录模型版本 |
| Python 依赖（requirements.txt） | 依赖 | 各 PyPI 包 | 各包自身许可证 | 运行时依赖 | 安装时生成依赖清单；不得引入未列入锁定技术栈的框架 |
| JS 依赖（前端） | 依赖 | npm（Vue3/Element Plus/ECharts 等） | 各包自身许可证 | 前端工作台 | Phase 6 引入时登记 |
| 公开企业文档（补充） | 文档 | 各公开来源 | 各来源原始许可证 | RAG 语料 | 同上，逐份登记 |

## 说明与纪律

- 本表为 **Phase 0 预留结构**：具体"Version / 下载日期 / 文件清单"在各资源真正引入时补全，
  不允许预先填写未经确认的数值或版本。
- 项目源码：Apache-2.0；第三方资源各自遵循原许可证——二者独立，不因本项目打包而改变。
- 任何"将第三方数据打包重新授权"的行为均被禁止。
- 评估中引用第三方 Benchmark 时，论文与报告必须注明其名称、版本与许可证。