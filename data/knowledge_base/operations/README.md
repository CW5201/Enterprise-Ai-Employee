# knowledge_base/operations — 运营规范语料

存放企业运营、流程与协作规范类文档，是 RAG（Milvus）检索的语料来源之一。

内容范围（示例）：
- 标准作业流程（SOP）
- 客户服务与工单规范
- 供应链、库存、物流流程
- 项目管理与跨部门协作规范
- 会议、文档、知识管理规范

## 数据来源约束

- 只允许使用**公开**的运营规范文档，或明确标注为合成的企业知识文本。
- 每份文档必须登记到 `THIRD_PARTY_LICENSES.md`。
- 禁止放入任何真实企业的未公开内部文件。

## 文档前置元数据

见 `data/knowledge_base/hr/README.md` 中的元数据模板（`department: operations`）。

当前阶段已录入合成企业制度文档（见 `synthetic: true` 标记）；正式评估语料在 Phase 5 引入。
