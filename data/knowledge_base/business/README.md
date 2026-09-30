# knowledge_base/business — 业务与产品文档语料

存放产品、业务与流程类文档，是 RAG（Milvus）检索的语料来源之一。

内容范围（示例）：
- 产品说明与功能定义
- 业务流程与规则说明
- 客户、订单、合同相关业务说明
- 指标口径与统计规则定义
- 常见业务问题解答（FAQ）

## 数据来源约束

- 只允许使用**公开**的业务文档，或明确标注为合成的企业知识文本。
- 每份文档必须登记到 `THIRD_PARTY_LICENSES.md`。
- 禁止放入任何真实企业的未公开内部文件。

## 文档前置元数据

见 `data/knowledge_base/hr/README.md` 中的元数据模板（`department: business`）。

当前阶段已录入合成企业制度文档（见 `synthetic: true` 标记）；正式评估语料在 Phase 5 引入。
