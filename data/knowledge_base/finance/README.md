# knowledge_base/finance — 财务制度语料

存放与财务、费用、报销相关的企业制度类文档，是 RAG（Milvus）检索的语料来源之一。

内容范围（示例）：
- 费用报销制度与标准
- 差旅、招待、采购审批流程
- 预算管理与资金审批权限
- 发票与合规要求
- 付款与结算规则

## 数据来源约束

- 只允许使用**公开**的企业制度文档，或明确标注为合成的企业知识文本。
- 每份文档必须登记到 `THIRD_PARTY_LICENSES.md`。
- 禁止放入任何真实企业的未公开内部文件。

## 文档前置元数据

见 `data/knowledge_base/hr/README.md` 中的元数据模板（`department: finance`）。

当前阶段已录入合成企业制度文档（见 `synthetic: true` 标记）；正式评估语料在 Phase 5 引入。
