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

#### FH-ROUTE-001（Phase 3，真实）

- **Problem**：multi_tool 任务（期望 `kg → sql`）被预测为 `sql → kg`，
  工具集合正确但执行顺序错误（plan_exact_match 失分）。
- **Trigger**：任务同时需要关系遍历与结构化指标，且 TaskProfile 能力标志
  不携带"指标优先还是关系优先"的序信息（rt-067/068/070）。
- **Expected Behavior**：按任务语义拆解出正确的工具执行顺序。
- **Failure Behavior**：`routing_decision.execution_plan` 顺序颠倒；
  下游 `MultiToolExecutionNode` 会按错误顺序执行。
- **Detection**：`routing_eval_results.json` 中 `route_correct=true`
  但 `plan_exact=false` 的 multi_tool 条目。
- **Recovery**：在 TaskProfile 增加工具优先级信号（Commit 5 改进项，
  未实现）；当前保留为已知失分点。
- **Regression Test**：rt-067/068/070 作为"顺序敏感"锚点留在
  `routing_eval.jsonl`。

#### FH-ROUTE-002（Phase 3，真实）

- **Problem**：LLM 把结构化 TaskProfile JSON 包在 markdown 围栏
  （```json …```）里返回，`generate_structured` 解析失败 → 全部任务
  降级 clarification（基线 D 首轮 100/100 全错）。
- **Trigger**：provider（Qwen via Agnes）的习惯性围栏输出。
- **Expected Behavior**：结构化输出被正常解析，路由生效。
- **Failure Behavior**：`LLMError: Model did not return valid JSON`，
  任务 100% 落入 clarification。
- **Detection**：路由 smoke test 中 predicted 全为 clarification 且
  无 LLM 报错（降级静默）；或 `routing_trace` 中 profile 全为
  `ambiguous_task/ambiguity=1.0`。
- **Recovery**：`src/core/llm_client.py::_parse_json_payload` 增加
  围栏剥离（Commit 4 修复）；修复后 D 基线 route_acc 0.000 → 0.680。
- **Regression Test**：`tests/unit/test_llm_client.py`（如已存在）
  或路由 smoke（rt-011 等单源任务应路由到 sql 而非 clarification）。

#### FH-ROUTE-003（Phase 3，真实）

- **Problem**：hard 关系任务（"哪些客户买了供应商 X 的商品"）期望纯
  KG 路由，但 KG Tool 缺反向模板（Phase 2.4 记录的
  `supplier_customers` 缺口），路由只能判 multi_tool[kg, sql]
  补偿，route_acc 失分。
- **Trigger**：关系查询方向为"供应商 → 客户"（KG 现有模板
  均为客户/订单/发票正向遍历）。
- **Expected Behavior**：单 KG 模板回答。
- **Failure Behavior**：路由决策 multi_tool；若执行，KG 步骤会
  `skipped_no_template`（MultiToolExecutionNode 的诚实失败）。
- **Detection**：`routing_eval.jsonl` notes 中
  "KG-coverage note" 标记 + `routing_eval_results.json` 中
  对应 rt-037/038/070/071/072 的 route_correct=false。
- **Recovery**：Commit 5 改进项：补 `supplier_customers` 反向模板
  （未实现）；当前作为跨阶段 limitation 记录。
- **Regression Test**：rt-037/038/070/071/072 保留在评测集。

#### FH-ROUTE-004（Phase 3，真实）

- **Problem**：统计/趋势任务（"算增长率"）漏预测 `analysis` 工具，
  只路由到 `sql`，tool recall 下降。
- **Trigger**：TaskProfile 的 `requires_statistical_analysis` 对
  "增长率/环比"措辞的识别不稳定（LLM 车道）。
- **Expected Behavior**：multi_tool[sql, analysis]。
- **Failure Behavior**：predicted tools = [sql]，缺派生计算步骤
  （rt-041/045 等）。
- **Detection**：`routing_eval_results.json` 中 task_type
  statistical_analysis/trend_analysis 且 tool_recall < 1.0。
- **Recovery**：在 routing_rules.yaml 的 capability_rules 增加
  统计类关键词的强化规则（Commit 5 改进项，未实现）。
- **Regression Test**：rt-041/045/049/050 作为"必须带 analysis"
  锚点。

_（Phase 3 起填充真实案例）_

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

- **FH-VER-001**（Phase 4，真实运行）：`ver-031`「本月费用总额为 5000 元」
  （真实值 4600.50，GT `expected_supported=False`）。B_rule_only 基线在
  exact 层将 5000 误判为 supported（计入 hallucinated-claim rate）；
  C_full 基线经 conflict 分支正确拒绝。详见
  `docs/phase4/FAILURE_CASES.md` §7。
- **FH-VER-002**（Phase 4，真实运行）：`ver-105`「费用类型共 4 种」
  （真实 2 种，GT `expected_supported=False`）。C_full 基线 rule 层因
  表层 token 重合误支持，计入 unsupported-claim leakage。详见
  `docs/phase4/FAILURE_CASES.md` §8。

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

- **FH-VER-003**（Phase 4，真实运行）：`ver-064` / `ver-065`「费用增长率
  30% vs 116%」（两个 analysis 源口径冲突，GT `expected_conflict=True`）。
  C_full 基线未触发 conflict 分支（cross-source disagreement 未传达到
  exact 层），计入 conflict-not-detected。详见
  `docs/phase4/FAILURE_CASES.md` §6。

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

## 11. Phase 5 End-to-End Failure Taxonomy（真实运行，487 条全量）

> 来源：`artifacts/phase5/enterprise_failure_report.md`（由
> `scripts/analyze_enterprise_failures.py` 自动生成）。以下为四类
> **跨阶段串联**的真实失败模式（Phase 5 将 Phase 2 / 3 / 4 的失败
> 首次串到同一张分类图上）。

### 案例记录

#### FH-P5-001（Phase 5，offline harness，KG 路由 honest failure）

- **触发**：`relationship_query` / `structured_lookup`（KG 子集）任务在
  offline harness 运行，Neo4j 不可用（凭证受限），KG 工具无结果。
- **路由 / 工具选择**：动态路由**正确**选了 KG 路由（route_acc 对 KG
  子集接近 1.0），但工具在 offline 无 Neo4j 时返回空 → task_success = 0。
- **根因**：**环境限制**，非路由或系统缺陷——KG 链路本身已在 Phase 2.4
  验证，offline 只是未接入。
- **恢复 / 回归**：online 运行（`NEO4J_PASSWORD` 配置）后该子集预期回升；
  失败类别 `kg_coverage` 保留在分母。

#### FH-P5-002（Phase 5，多工具任务 success 低但 tool F1 高）

- **触发**：`two_tool` / `three_plus_tool` 任务（RAG+SQL+KG+Analysis 组合）。
- **现象**：tool F1 0.5–0.75（**工具组合选对了**）但 task_success
  0.000–0.067（**跨源融合到答案失败**）。
- **根因**：offline 无 live LLM 在线生成答案，多源融合的最后一步
  （`AnswerGenerationNode` 只基于 `sql_result` + `retrieved_context`，
  不读 KG / Analysis 的 `tool_results`）在 offline 下无法产出完整答案。
- **恢复 / 回归**：Phase 4 多工具执行已把各源结果写入 `tool_results`
  通道；online 下 `AnswerGenerationNode` + `MultiToolExecutionNode`
  融合后 success 预期上升。负结果保留，不外推。

#### FH-P5-003（Phase 5，B4 规则路由 success 高于动态路由——保留负结果）

- **现象**：B4（确定性关键词路由）success 0.450 > A_full（动态路由）
  0.345。
- **根因**：动态路由的 confidence gate 将部分 easy 单工具任务降为
  clarification（honest，不硬猜）；B4 对这类任务"果断"路由因而 success
  更高。这是**真实且有价值的工程结果**，保留不掩盖。
- **结论**：动态路由的收益集中在需要正确源判断的复杂任务
  （SQL 侧 tool F1 0.405 vs 静态 0.010），不在 easy 单工具上。

#### FH-P5-004（Phase 5，验证层 offline 下 leakage 无数值）

- **现象**：A_full 与 B_no_verification 的 success / route / tool F1
  **完全相同**；unsupported-claim leakage 在 offline 下全 0。
- **根因**：offline LLM backend 不产生 claim 集合（`_OfflineBackend`
  的 claim-extraction 返回空 claims list），验证层对空集合运行 →
  无 leakage 可计。
- **恢复 / 回归**：RQ2 的 leakage 指标需 online 运行（live LLM 生成
  真实答案 → claim 抽取 → 验证）才有数值；Phase 4 离线确定性评测
  （B：hallucinated-claim rate 1.0 → 0.025）是该指标的阶段性证据，
  Phase 5 online 将补全。

#### FH-P5-005（Phase 5，w/o KG / w/o Hybrid / w/o Reranker 消融无指标差）

- **现象**：四项消融（w/o KG / w/o Hybrid / w/o Reranker /
  w/o Verification）在 offline 下与 A_full **success / route / tool F1
  完全相同**。
- **根因**：offline 环境下这些组件的"在/不在"不影响确定性路由与
  SQL 结果（RAG 用 fake dense、KG 无 Neo4j、验证无 live LLM），
  消融未触发任何实际差异。
- **结论**：这是 **offline harness 的固有局限**，不是"去掉这些
  模块一定不下降"的证据。要分离这些组件的真实贡献，需 online
  end-to-end 运行（live Milvus + Neo4j + LLM）。


## 记录纪律

- 案例必须来自**真实运行**，含真实输入、真实输出、时间与配置版本；
- 不允许用"设想中的失败"充数；
- 每修复一个失败模式，必须同时补充回归测试，否则视为未修复；
- 案例统计（各类错误占比）只能来自评估运行的错误分析输出，不得手工填写。