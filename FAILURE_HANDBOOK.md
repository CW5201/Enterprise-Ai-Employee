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

#### FH-ROUTE-005（Phase 6 修复期引入的回归，真实运行，已修复）

- **Problem**：歧义登记表 `config/ambiguity_register.yaml` 被填入了**有确定
  口径的度量词**（销售额 / 金额 / 数量 / 平均 / 比例 / 增长率 / 环比 / 同比 /
  订单量 / 利润 / 营收），而 `classify_by_rules` 把歧义判定放在数据 / 知识
  关键词判定**之前**，命中即返回 `clarification`（置信度 0.95）并跳过 LLM
  车道 → 大量正常业务问题被拒绝回答。
- **Trigger**：问题文本包含上述任一子串。50 个字符以内的高频词（尤其"数量"
  "金额"）覆盖率极高。
- **Expected Behavior**：`统计最近三个月的销售额` → `complex_analysis` →
  走 SQL + analysis；`按订单数量排名，第 1 位的客户是谁？` → `data_query` → 走 SQL。
- **Failure Behavior**：实测
  `classify_by_rules("统计最近三个月的销售额")` →
  `('clarification', 0.95, [], 'rule: ambiguity pattern detected')`；
  端到端 `DigitalEmployee(phase3=True, phase4=True)` →
  `route=clarification, status=clarified_out_of_scope`，
  答案为"该任务意图不明确或不属于…范围"。
- **Detection**：`data/eval/enterprise_tasks.jsonl` 中 30/487 条、
  `data/eval/routing_eval.jsonl` 中 26/100 条命中登记表；查
  `classify_by_rules` 的返回值即可发现（无需跑全量）。
- **Recovery**：登记表只保留**口径未定义**的问法（大客户 / 重要客户 /
  大额订单 / 异常订单 …），并在文件头写明登记标准；`task_router._RULE_LANE`
  新增 `clarification → ambiguous_task(ambiguity=1.0)`，使快车道的澄清裁决
  在 phase3 链路上也被确定性地执行（不再依赖 LLM 车道"救回"）。
- **Regression Test**：`tests/unit/test_intent_ambiguity.py`
  （口径守卫 + 度量词必须留在数据车道 + **487 条 Benchmark 中不得有任何
  期望真实路由的任务被登记表拒绝**）、
  `tests/unit/test_task_router.py::test_rule_lane_clarification_is_decisive`。

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


## 12. API / SSE Delivery Defect（交付缺陷）

- **Problem**：节点算得对，但接口层把结果丢了、改了或发错形态——前端看到的
  与后端真实状态不一致（本轮为 SSE 与报告落库）。
- **Trigger**：状态通道里放的是 **Pydantic 模型**而非 dict；接口自己折叠
  节点 delta；用固定状态字符串当"成功"判据。
- **Expected Behavior**：接口只做投影（脱敏 + 序列化），不改变语义；上报的
  状态与图内真实终态一致；报告只由真实运行产生且内容完整。
- **Failure Behavior**：见下方四条真实案例。
- **Detection**：用**真实 graph**（非手写 dict）驱动接口的测试；
  对比 `/api/chat` 与 `/api/chat/stream` 同一问题的 evidence / 报告条数。
- **Recovery**：投影统一走 `_to_plain` / `_strip_raw_content`；delta 折叠按
  `src/core/state.py::APPEND_CHANNELS` 合并；成功判据改为终态状态词表
  （`reports.should_record_report`）。
- **Regression Test**：`tests/unit/test_phase6_stream_evidence.py`
  （真实 EvidenceItem / ToolCallRecord 走完整条链路直到报告存储）。

### 案例记录

#### FH-SSE-001（Phase 6.7 修复期，真实运行，已修复）

- **Problem**：`_sanitize_evidence()` 用 `ev.items()` 展开证据项，但状态里
  是 `EvidenceItem`（pydantic v2 模型，没有 `items()`）。
- **Trigger**：任何**产出证据**的任务走 `POST /api/chat/stream`。
- **Expected Behavior**：输出 `completed` 事件，payload 带脱敏证据。
- **Failure Behavior**：`AttributeError: 'EvidenceItem' object has no attribute
  'items'` → 生成器被外层兜底捕获 → 客户端只收到 `error` 事件
  （"任务执行失败，请稍后重试或联系管理员"），随后**没有任何报告入库**。
- **Detection**：服务端日志 `stream pipeline failed` + traceback；
  或对真实 graph 跑一次流式请求（旧的 `test_phase6_sse.py` 全用手写 dict，
  所以一直是绿的）。
- **Recovery**：新增 `_to_plain()`（模型 → dict，递归）并在
  `_sanitize_evidence` / `_safe_payload` / `_sse_event` 统一使用。
- **Regression Test**：`test_streamed_run_reaches_completed_with_evidence`。

#### FH-SSE-002（Phase 6.7 修复期，真实运行，已修复）

- **Problem**：`json.dumps(payload, default=str)` 把 payload 里的模型
  （`SQLResultPayload` / `ToolCallRecord` / 验证结果）序列化成**字符串
  repr**，前端拿到的是 `"tool='sql' ok=True …"` 这类文本而不是对象。
- **Trigger**：任何 payload 含模型的节点事件。
- **Expected Behavior**：字段以 JSON 对象/数组下发。
- **Failure Behavior**：前端类型契约（`ToolCallRecord` / `SQLResultPayload`）
  与实际不符，字段访问得到 undefined。
- **Detection**：检查 SSE 帧中 `tool_calls` / `sql_result` 的 JSON 类型。
- **Recovery**：`_sse_event` 先 `_to_plain()` 再序列化。
- **Regression Test**：`test_streamed_run_reaches_completed_with_evidence`
  断言 `evidence[0]` / `tool_calls[0]` 必须是 dict。

#### FH-SSE-003（Phase 6.7 修复期，真实运行，已修复）

- **Problem**：SSE 路由用 `final_state[k] = v` 折叠每节点 delta，而
  `AgentState` 的 list 通道是 **append reducer**（节点只返回"新增项"）。
  覆盖赋值只留下**最后一个写该通道的节点**的内容。
- **Trigger**：任何多节点写同一通道的任务（`evidence` / `tool_calls` /
  `claims` / `verification_results` / `tool_results` …）。
- **Expected Behavior**：折叠结果等于图内真实终态。
- **Failure Behavior**：流式路径落库的报告只剩 answer 节点的 RAG 证据子集；
  纯 SQL 任务 `evidence_count=0`，报告正文写"未提供证据"。
- **Detection**：同一问题分别走 `/api/chat` 与 `/api/chat/stream`，
  比较 `evidence` 条数与报告 `evidence_count`。
- **Recovery**：`_fold_delta()` 按 `APPEND_CHANNELS`（由 `AgentState`
  注解推导，防漂移）合并列表通道。
- **Regression Test**：`test_streamed_run_records_a_report_with_all_its_evidence`。

#### FH-SSE-004（Phase 6.7 修复期，真实运行，已修复）

- **Problem**：报告门禁写成 `status == "completed"`，但 Phase 4 尾链会覆写
  `status`（answer_generation → claim_extraction → answer_guard），终态是
  `claims_empty` / `claims_extracted` / `verified` / `guarded` /
  `guard_blocked`，**永远不会是 `completed`**。
- **Trigger**：应用默认配置 `phase3=True, phase4=True` 下的每一次运行。
- **Expected Behavior**：每次产生答案的运行都在「报告中心」出现一条记录。
- **Failure Behavior**：`GET /api/reports` 恒为空（实测 `status='claims_empty'`
  且 `assert []` 成立）——整个 Phase 6.7 报告能力在真实配置下等于没有生效。
- **Detection**：跑一条真实任务后查 `/api/reports`（或 README 的
  "在「AI 任务」页成功执行一条任务后，报告中心才会出现对应报告"）。
- **Recovery**：`reports.should_record_report(status, answer)`：只对
  "产生了答案"的终态登记（含 guarded / guard_blocked），澄清与失败不登记；
  两个入口（chat / stream）共用同一判据。
- **Regression Test**：`test_chat_run_records_a_report`、
  `test_streamed_run_records_a_report_with_all_its_evidence`。

#### FH-TASK-001（Phase 6.7 修复期，真实运行，已修复）

- **Problem**：任务历史只有 `POST /api/tasks` 一个写入口，而工作台的
  **默认执行模式是流式**（`POST /api/chat/stream`）——流式运行不写任务库。
- **Trigger**：在「AI 任务」页以默认模式（流式执行）提交任务。
- **Expected Behavior**：任务出现在「任务历史」列表里（与报告中心一致）。
- **Failure Behavior**：任务历史恒为空；同一次运行在「报告中心」有记录，
  两个内存存储口径不一致，用户以为任务没跑。
- **Detection**：流式跑一条任务后查 `GET /api/tasks`（修复前返回 `[]`）。
- **Recovery**：`tasks.record_task()` 作为统一写入点，SSE 路由在 `completed`
  之前调用；状态由 `task_status_for_run()` 映射（有答案 → `completed`，
  澄清 → `clarification`，其余 → `failed`）；任务库同时补上 `MAX_TASKS`
  上限与 `seq` 排序键（`created_at` 只有秒级精度，同秒任务排序不稳定）。
- **Regression Test**：`test_streamed_run_lands_in_task_history`。

#### FH-UI-001（Phase 6.7 修复期，静态审查确认，已修复）

- **Problem**：工作台多个页面把接口失败**静默吞掉**：`load()` 只有
  `try/finally` 没有 `catch`，异常变成未处理的 Promise rejection，页面显示
  "暂无数据/未检索到文档"，与"真的没有数据"无法区分；`AnalyticsView`
  甚至把错误写进 `error` 变量却不渲染；`SettingsView` 的"运行时状态"
  是写死的字符串（API 实际跑在内存回退后端时仍显示 milvus）。
- **Trigger**：API 未启动、超时、或返回 5xx。
- **Expected Behavior**：与 Tasks / Reports 两个页面同一口径——错误必须显示出来。
- **Failure Behavior**：`GraphView.loadEntities` / `selectEntity`、
  `KnowledgeView.load`、`EvaluationView.load` 静默失败；
  `AnalyticsView` 只显示"数据集不可用"；`SettingsView` 展示与运行状态不符的信息。
- **Detection**：关掉 API 后逐个页面查看（或读代码看有无 `catch`）。
- **Recovery**：四处补 `catch` + 页面级错误提示；`AnalyticsView` 渲染
  `error`；`SettingsView` 改为读取 `GET /health`（运行时状态）并把
  "检索模式 / 动态路由 / 结果验证"明确标注为服务端代码固定的构建配置；
  `KnowledgeView` 的正文加载从裸 `fetch` 改为共享 `client`（base URL 与
  错误归一化与其他页面一致）。
- **Regression Test**：前端只做逻辑单测（Phase 6 边界，不做组件渲染测试），
  因此这条以人工复核（关闭 API 逐页查看）为准，不虚构自动化测试。

---

## 记录纪律

- 案例必须来自**真实运行**，含真实输入、真实输出、时间与配置版本；
- 不允许用"设想中的失败"充数；
- 每修复一个失败模式，必须同时补充回归测试，否则视为未修复；
- 案例统计（各类错误占比）只能来自评估运行的错误分析输出，不得手工填写。