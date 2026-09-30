# 失败案例手册（FAILURE HANDBOOK）

> 状态：Phase 0。**本文件当前只建立模板，不包含任何真实案例结果。**
> Phase 2 起，每次发现的真实失败案例按模板登记在对应条目下。
> 禁止编造案例、禁止编造统计数据。

## 使用方式

1. 评估或人工测试中发现失败 → 按阶段归因（见 `docs/RESEARCH.md` 第 9 节）；
2. 找到对应模板条目，在 `## 案例记录` 下追加一条**真实**记录（含原始输入与原始输出）；
3. 补充或更新 `Detection` 与 `Recovery`，并将可自动化的场景写入 `Regression Test`；
4. 回归测试进入评估集或测试套件，防止修复后回退。

## 案例记录字段说明

- **Problem**：问题是什么（现象级描述）。
- **Trigger**：在什么条件下触发（任务类型、输入特征、数据状态）。
- **Expected Behavior**：系统本应做什么。
- **Failure Behavior**：系统实际做了什么。
- **Detection**：如何自动或人工发现（日志、指标、验证节点、用户反馈）。
- **Recovery**：发现后如何恢复（重试、降级、澄清、替换工具）。
- **Regression Test**：用哪个测试防止复发。

---

## 1. SQL Generation Error

- **Problem**：生成的 SQL 语法正确但语义错误（错误的表、错误的过滤条件、错误的口径）。
- **Trigger**：问题涉及多表连接、时间窗口、业务口径歧义（如"销售额"未指明口径）。
- **Expected Behavior**：依据 Schema 与数据字典生成符合业务口径的只读查询，必要时先澄清。
- **Failure Behavior**：生成字段存在但语义不符的查询，返回"看起来合理"的错误数字。
- **Detection**：SQL Execution Accuracy 比对结果集；`ambiguity_register` 命中检查；执行报错日志。
- **Recovery**：命中歧义登记表则转 `clarification`；执行失败或结果为空则按路由 `fallback` 重试；仍失败则降级并标注。
- **Regression Test**：将该问题加入评估集 `text_to_sql` 类别，并写 `tests/integration/` 用例固定期望结果。

### 案例记录

_（Phase 2 起填充真实案例）_

---

## 2. RAG Retrieval Error

- **Problem**：应召回的文档片段未被召回（漏召），或召回了大量无关片段（噪声）。
- **Trigger**：问题使用口语化表述；专有名词被改写；问题需要跨文档多跳；Metadata Filter 过严。
- **Expected Behavior**：Hybrid 检索召回相关片段并通过 Rerank 排序到 Top-K。
- **Failure Behavior**：Top-K 中无正确片段，答案基于无关文档生成。
- **Detection**：Retrieval Recall@K（分 Dense / BM25 / Hybrid / +Rerank 配置对比）；人工抽查引用是否相关。
- **Recovery**：放宽或移除 Metadata Filter；扩展查询改写数量；升级为多工具路径补充结构化证据。
- **Regression Test**：将漏召问题加入检索评估子集，固定 K 值与期望命中片段。

### 案例记录

_（Phase 2 起填充真实案例）_

---

## 3. Reranker Error

- **Problem**：正确片段被 Rerank 排在 Top-K 之外（排序错误），或 Rerank 后多样性下降导致关键片段被挤出。
- **Trigger**：多个片段语义高度相似；正确片段表述与问题差异较大。
- **Expected Behavior**：Rerank 后正确片段仍位于 Top-K 内。
- **Failure Behavior**：召回阶段命中的正确片段在重排后丢失，答案质量下降。
- **Detection**：对比 "Hybrid" 与 "Hybrid + Rerank" 的 Recall@K 与最终 Task Success Rate；记录重排前后位置变化。
- **Recovery**：调整 `top_k` 与 `final_top_k`；在融合阶段保留一定比例的召回原始顺序结果；必要时关闭 Rerank。
- **Regression Test**：建立"重排前后位置变动"的监控测试，对已知易错样本固定期望。

### 案例记录

_（Phase 2 起填充真实案例）_

---

## 4. KG Entity Matching Error

- **Problem**：问题中的实体名无法匹配到图谱节点（同义、缩写、别名、大小写、翻译差异），导致模板查询返回空。
- **Trigger**：用户使用简称或口语名称；数据库实体名与图谱节点名不一致。
- **Expected Behavior**：实体归一化/别名映射后正确匹配到节点并返回关系结果。
- **Failure Behavior**：匹配失败返回空结果，或匹配到错误实体返回错误关系。
- **Detection**：图谱查询空结果率统计；模板查询结果的实体与问题实体一致性检查。
- **Recovery**：启用别名表与模糊匹配回退；匹配失败则转 SQL 工具或向用户澄清实体名称。
- **Regression Test**：为已知同义词/缩写建立映射测试用例。

### 案例记录

#### FH-KG-001（Phase 2.4，真实）

- **Problem**：存在性任务（"Did customer 2 place order 502?"）被 runner
  路由到 `customer_orders`（返回订单列表），无法把"负例无订单"判为正确
  的 no；KGTool 已有 `relationship_exists` 模板但未被启用。
- **Trigger**：existence 类任务 + 负例期望（正确答案为"no"）。
- **Expected Behavior**：走 `relationship_exists` 模板，返回
  `exists: true/false`，负例判为 no 即正确。
- **Failure Behavior**：路由到列表型模板，`predicted_ids` 非空（该客户
  的其他订单），exact match 判错。
- **Detection**：`kg_eval_results.json` 中 `error="unroutable"` 且
  task_type=existence 的条目。
- **Recovery**：runner 对 existence 任务优先路由到 `relationship_exists`
  （Commit 5 改进项；当前保留为已知缺口）。
- **Regression Test**：`tests/integration/test_kg_tool.py::test_relation_existence`
  已覆盖 `relationship_exists` 的正/负例；existence 路由待补 runner 用例。

#### FH-KG-002（Phase 2.4，真实）

- **Problem**：跨实体复合问题（"Which customers ordered **both** items
  supplier 1 supplies?"，正确答案为空集）无对应模板——单模板只能做
  单个供应商×单个客户的过滤，不能做两个 item 集合的交集。
- **Trigger**：cross_entity hard 任务，期望为"两个实体集合的交集"或
  "空集"。
- **Expected Behavior**：返回正确交集/空集，判为 correct。
- **Failure Behavior**：`error="unroutable"`（`kg-rq-054`），记为失败
  （保留在分母，不剔除）。
- **Detection**：`kg_eval_results.json` 中 task_type=cross_entity 且
  含 "both items" 措辞的条目。
- **Recovery**：新增 `customer_items_via_supplier` 的"双 item 交集"
  变体模板，或允许 runner 组合两次单模板调用求交集（Commit 5 改进项）。
- **Regression Test**：`kg-rq-051`（空集）与 `kg-rq-052`（单 item 命中）
  在 `data/eval/kg_eval.jsonl` 中作为负例/正例锚点。

#### FH-KG-003（Phase 2.4，真实）

- **Problem**：多跳/跨实体复合答案需同时返回两组实体（如 056 期望
  同时给 supplier 与 city），单模板只返回一组，exact match 失分。
- **Trigger**：期望答案含多组实体 ID（supplier 组 + city 组）。
- **Expected Behavior**：返回全部期望实体组，exact match 通过。
- **Failure Behavior**：`predicted_ids` 只含一组（F1 0.667，exact=False，
  见 `kg-rq-030`/`032`）。
- **Detection**：`avg_f1` 在 multi_hop/cross_entity 低于 two_hop；
  `path_accuracy` 接近 1 而 `exact_match` 为 0 的条目。
- **Recovery**：复合问题拆成多次单模板调用后在 runner 侧合并实体组
  （Commit 5 改进项）。
- **Regression Test**：`kg-rq-056` 作为"多组实体"锚点。

_（Phase 2 起填充真实案例）_

---

## 5. Tool Selection Error

- **Problem**：路由选择了错误的工具集合：漏选（能力不足）、多选（引入噪声与延迟）、选错（用文档回答数值问题）。
- **Trigger**：意图分类错误；任务混合了多种类型但只识别出一种；路由规则未覆盖该任务模式。
- **Expected Behavior**：选出的工具集合与任务实际所需一致。
- **Failure Behavior**：任务失败或结果不完整，或延迟显著上升。
- **Detection**：Routing Accuracy（Exact Match / Over-selection / Under-selection 三项分列）；`routing_history` 分析。
- **Recovery**：证据不足时按 `on_low_evidence: escalate_to_multi_tool` 升级路径；低置信度转澄清。
- **Regression Test**：按意图类别建立路由期望测试；新增任务模式时同步更新 `routing_rules.yaml`。

### 案例记录

_（Phase 3 起填充真实案例）_

---

## 6. Task Planning Error

- **Problem**：多步骤任务的执行顺序或依赖关系错误，导致中间结果不可用或重复计算。
- **Trigger**：任务需要先检索口径再执行查询；后一步依赖前一步的输出格式。
- **Expected Behavior**：按依赖关系排序执行，复用中间结果。
- **Failure Behavior**：顺序颠倒导致后一步缺少必要输入；或重复调用同一工具浪费资源。
- **Detection**：执行轨迹（`tool_calls` 序列）人工与规则检查；复杂任务类别的完成率。
- **Recovery**：在路由计划中显式声明依赖；对重复调用做结果复用；超过 `max_iterations` 时终止并报告。
- **Regression Test**：为复杂任务类别建立执行轨迹断言（顺序与调用次数）。

### 案例记录

_（Phase 3 起填充真实案例）_

---

## 7. Unsupported Claim

- **Problem**：答案中包含无任何数据或文档依据的结论（本项目的核心失败模式）。
- **Trigger**：证据不足时模型仍给出结论；模型使用自身先验知识补充；证据与问题相关性弱。
- **Expected Behavior**：证据不足时明确说明缺少什么，而不是给出结论。
- **Failure Behavior**：输出看似合理但无法追溯的结论，用户无法识别。
- **Detection**：Claim-Evidence Verification 判定为 `unsupported` / `contradicted`；评估中统计无依据结论比例。
- **Recovery**：触发重试（补充检索/换工具）；超过 `max_retry` 则输出时显式标注"缺少证据支持"。
- **Regression Test**：构建"证据不足"负样本集，期望系统标注而非编造。

### 案例记录

_（Phase 4 起填充真实案例）_

---

## 8. Citation Mismatch

- **Problem**：答案给出的引用与结论内容不匹配：引用存在但不支持该结论，或引用了错误的文档/结果集。
- **Trigger**：多源证据混杂；证据 ID 编号与内容对应关系在融合阶段被打乱；模型复述引用时出错。
- **Expected Behavior**：每条结论的引用精确指向支持它的证据项。
- **Failure Behavior**：结论与引用脱节，形式上"有引用"但实质不可核验。
- **Detection**：逐条检查 citation → evidence 的支持关系；统计引用不匹配率；验证节点交叉检查。
- **Recovery**：由验证节点重新匹配证据；无法匹配则移除该引用或降级该结论的表述。
- **Regression Test**：对每条输出断言"引用 ID 必须存在于本轮证据集合中且被判定为支持"。

### 案例记录

_（Phase 4 起填充真实案例）_

---

## 9. Tool Timeout

- **Problem**：工具调用超过超时时间未返回，任务被阻塞或中断。
- **Trigger**：SQL 扫描行数过大；向量检索集合过大或 Rerank 批处理过载；图谱查询路径爆炸；本地模型推理慢。
- **Expected Behavior**：超时被捕获并按路由策略接管，任务不中断。
- **Failure Behavior**：任务挂起，或直接抛出未处理异常导致整体失败。
- **Detection**：分项耗时监控（意图/路由/各工具/融合/验证）；超时计数日志。
- **Recovery**：按 `config/tool_registry.yaml` 的超时设置中断并记录；尝试替代工具或降低召回规模；必要时向用户说明降级。
- **Regression Test**：以超小超时值构造超时场景，断言系统仍返回可解释结果。

### 案例记录

_（Phase 2 起填充真实案例）_

---

## 10. LLM Service Failure

- **Problem**：LLM 服务不可用、限流、返回格式不可解析或返回内容被截断。
- **Trigger**：网络异常；服务端限流；上下文过长导致截断；结构化输出解析失败。
- **Expected Behavior**：重试与降级策略生效，向用户给出明确状态而不是错误堆栈。
- **Failure Behavior**：节点抛出未处理异常，整个任务失败且用户看到内部错误。
- **Detection**：LLM 调用错误率与重试次数监控（`src/core/observability.py`）。
- **Recovery**：按 `llm.max_retries` 重试；仍失败则降级为"仅返回已获取的证据与执行结果，不做生成"；记录到失败日志。
- **Regression Test**：以 mock 的失败响应构造用例，断言重试与降级路径均被触发且不抛未处理异常。

### 案例记录

_（Phase 1 起填充真实案例）_

---

## 记录纪律

- 案例必须来自**真实运行**，含真实输入、真实输出、时间与配置版本；
- 不允许用"设想中的失败"充数；
- 每修复一个失败模式，必须同时补充回归测试，否则视为未修复；
- 案例统计（各类错误占比）只能来自评估运行的错误分析输出，不得手工填写。