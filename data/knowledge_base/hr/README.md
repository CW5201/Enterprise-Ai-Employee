# knowledge_base/hr — 人事制度语料

存放与人力资源相关的企业制度类文档，是 RAG（Milvus）检索的语料来源之一。

内容范围（示例）：
- 员工手册
- 考勤、请假、加班、调休制度
- 招聘与入职、转正、离职流程
- 绩效考核与晋升规则
- 薪酬福利、社保公积金说明

## 数据来源约束

- 只允许使用**公开**的企业手册 / 政策 / 规范文档，或明确标注为合成的企业知识文本。
- 每份文档必须登记到 `THIRD_PARTY_LICENSES.md`（Source / License / Usage / Attribution）。
- 禁止放入任何真实企业的未公开内部文件。

## 文档前置元数据（Phase 2 建库时写入 Milvus metadata）

```yaml
doc_id: hr-0001
title: 员工请假管理制度
department: hr
doc_type: policy
source: <公开来源>
source_url: <链接>
license: <许可证>
effective_date: <生效日期>
language: zh
```

当前阶段仅建立目录，不放入任何实际文档。
