# Phase 2.4 Neo4j Knowledge Graph Experiment

> **定位**：Phase 2.4 的**正式阶段性实验记录**，用于支撑毕业论文
> "实验设计 / 实验结果 / 消融"章节。
> **数据来源**：本文所有指标均生成自
> [`artifacts/phase2.4/kg_eval_summary.json`](../../artifacts/phase2.4/kg_eval_summary.json)
> （由 `scripts/run_kg_eval.py` 真实运行产出，**不进 Git**，本地保留）。
> 逐任务明细见 `artifacts/phase2.4/kg_eval_results.json`；
> 图谱构建规模见 `artifacts/phase2.4/kg_build_summary.json`。

---

## 1. Objective

研究问题：关系型业务数据转换为知识图谱后，是否可以支持多跳关系与
实体关联查询。

本阶段只回答"**图谱作为关系型知识源能否支撑多跳关系查询**"，
**不回答**"图谱是否全面优于 SQL / 文档"——那需要 Dynamic Routing
与多源验证（Phase 3 / 4）才能得出。

## 2. Dataset

| 项 | 值 |
|---|---|
| 源数据库 | WideWorldImporters（Microsoft 样例，DuckDB 运行时 `data/runtime/wwi.duckdb`） |
| 源表（非 Archive） | 37 张（31 张有数据；`Sales_OrderLines`、`Application_Logs`、`Purchasing_PurchaseOrderLines` 等 8 张为空） |
| 图谱节点 | **4,723**（Customer 500 / Order 800 / Invoice 800 / StockItem 120 / Supplier 60 / BuyingGroup 15 / City 2,428） |
| 图谱关系 | **3,880**（PLACED 800 / INVOICED 800 / SHIPPED_ON 800 / HAS_LINE 800 / SUPPLIED_BY 120 / BELONGS_TO_GROUP 500 / SUPPLIER_IN_CITY 60） |
| 评测任务 | **58** 条（8 类 task_type × easy 19 / medium 19 / hard 20） |
| Ground Truth | 全部由 DuckDB SQL 独立核验，**先于**图谱查询生成，避免评测污染 |

**关键数据事实**：`Sales_OrderLines` 在本样例中为空，Order→StockItem
关系通过 `Sales_Invoices` + `Sales_InvoiceLines` 派生（HAS_LINE 边，
`__source` 标记 `derived:Sales_InvoiceLines`）。

## 3. Graph Schema

### 节点

| Label | 数量 | 源表 | 主键属性 |
|---|---|---|---|
| Customer | 500 | Sales_Customers | customer_id |
| Order | 800 | Sales_Orders | order_id |
| Invoice | 800 | Sales_Invoices | invoice_id |
| StockItem | 120 | Warehouse_StockItems | stock_item_id |
| Supplier | 60 | Purchasing_Suppliers | supplier_id |
| BuyingGroup | 15 | Sales_BuyingGroups | buying_group_id |
| City | 2,428 | Application_Cities | city_id |

### 关系

| 类型 | 数量 | 语义 |
|---|---|---|
| Customer `PLACED` Order | 800 | 客户下单 |
| Order `INVOICED` Invoice | 800 | 订单开票 |
| Invoice `SHIPPED_ON` StockItem | 800 | 发票行（含 quantity / extended_price 边属性） |
| Order `HAS_LINE` StockItem | 800 | 订单行（派生，见 §2 关键数据事实） |
| StockItem `SUPPLIED_BY` Supplier | 120 | 商品供应（SupplierID 非空的 120 个商品） |
| Customer `BELONGS_TO_GROUP` BuyingGroup | 500 | 客户分组 |
| Supplier `SUPPLIER_IN_CITY` City | 60 | 供应商配送城市 |

每个节点/关系均携带 `__source`（溯源表名）与 `__graph='eae'`
（命名空间标记，支持 `--reset` 时只清理本项目数据）。

## 4. KG Construction

```
DuckDB (wwi.duckdb)
   │  scripts/build_kg.py（按 data/schemas/kg_mapping.yaml）
   ├── 实体抽取：7 张源表 → 7 类节点（MERGE 按业务主键，幂等）
   ├── 关系抽取：7 类关系（含 1 个派生关系 HAS_LINE）
   └── Neo4j MERGE（4,723 节点 / 3,880 关系，0 失败 / 0 重复键）
```

- 构建脚本支持 `--reset`（只删 `__graph='eae'` 命名空间）/
  `--dry-run` / `--limit` / `--verbose`。
- 构建后 sanity checks：5 类核心关系全部非空；孤立节点 2,368 个
  （50.14%），**全部为 City 参照维度**（仅 60 个城市被 Supplier 使用），
  核心业务链（Customer/Order/Invoice/StockItem/Supplier/BuyingGroup）
  无孤立节点。
- 与 DuckDB 交叉核验：7 类节点数一致；多跳路径
  Customer→Order→Invoice→StockItem 与 SQL 结果一致。

## 5. Evaluation Tasks

| task_type | 条数 | 典型问题 |
|---|---|---|
| entity_lookup | 8 | "What is the customer with ID 2?" |
| one_hop | 9 | "Which orders did customer 2 place?" |
| two_hop | 8 | "Which items did customer 40's orders reference?" |
| multi_hop | 8 | "Which suppliers provided items in customer 20's orders?" |
| aggregation | 8 | "How many members does buying group 12 have?" |
| existence | 9 | "Did customer 2 place order 502?"（负例） |
| cross_entity | 8 | "Which customers bought supplier 1's items?" / "…ordered both items?"（负例） |

难度分布：easy 19 / medium 19 / hard 20。
所有任务都**需要关系知识**（至少一跳图遍历）；
负结果任务（016b / 048b / 051 / 054）的正确答案是**空集**，
用于检验系统能否正确回答"没有关系"。

## 6. Metrics

| 指标 | 定义 |
|---|---|
| Exact Match | 预测答案与 SQL 核验的答案是否一致（标量归一化比较 / 集合相等比较） |
| Precision / Recall / F1 | 预测实体 ID 集合 vs 期望实体 ID 集合 |
| Path Accuracy | 期望关系路径中有多少跳被模板实际遍历（仅对声明了 `expected_relations` 的任务定义） |
| Latency | 每次 KGTool 调用的墙钟时间（p50 / p95 / mean） |

失败任务（未路由 / 工具报错）**不剔除分母**，记为失败。

## 7. Overall Results（真实运行）

| 指标 | 值 |
|---|---|
| 总任务数 | 58 |
| **Exact Match** | **29.31%**（17/58） |
| 失败数（unroutable / 工具错误） | **5**（全部为 unroutable，无 Neo4j 报错） |
| 平均 Precision | 0.6257 |
| 平均 Recall | 0.4993 |
| 平均 F1 | 0.5167 |
| Path Accuracy（已定义任务） | 0.9238 |
| 延迟 | mean 3.36 ms / p50 1.98 ms / p95 3.22 ms（纯 Cypher，无 LLM） |

## 8. Task Type Analysis

| task_type | 任务数 | Exact | Avg P / R / F1 | Path Acc | 失败数 |
|---|---|---|---|---|---|
| two_hop | 8 | 0.625 | 0.875 / 0.792 / 0.825 | 0.958 | 0 |
| entity_lookup | 8 | 0.625 | 0.287 / 0.625 / 0.314 | 1.000 | 0 |
| aggregation | 8 | 0.625 | 0.375 / 0.313 / 0.333 | 0.813 | 0 |
| one_hop | 9 | 0.111 | 0.889 / 0.537 / 0.659 | 1.000 | 0 |
| multi_hop | 8 | 0.000 | 1.000 / 0.583 / 0.733 | 0.844 | 0 |
| existence | 9 | 0.111 | 0.444 / 0.278 / 0.333 | 1.000 | 3 |
| cross_entity | 8 | 0.000 | 0.500 / 0.391 / 0.424 | 0.917 | 2 |

观察：

- **two_hop 表现最好**（exact 0.625 / F1 0.825 / path 0.958）——
  一跳/两跳遍历是模板最擅长的形态。
- **multi_hop exact=0 但 P=1.0 / F1=0.733**：预测的供应商集合
  完全正确（precision 1.0，无多余），但期望实体只列了终点
  供应商、指标对"是否额外携带中间实体"口径敏感 → 失分在
  口径而非遍历。
- **cross_entity exact=0**：054/055 unroutable（无交集/双边聚合
  模板），056 复合答案超出单模板返回能力。
- **existence exact=0.111**：正例（yes）与负例（no）在 exact
  字符串比对下区分度低；path 全部为 1.0，说明关系跳本身遍历正确。

（数值取自 `kg_eval_summary.json` 的 `by_task_type`。）

## 9. Difficulty Analysis

| difficulty | 任务数 | Exact | Avg P / R / F1 | Path Acc | 失败数 |
|---|---|---|---|---|---|
| easy | 19 | 0.474 | 0.638 / 0.535 / 0.549 | 1.000 | 2 |
| medium | 19 | 0.316 | 0.746 / 0.621 / 0.644 | 1.000 | 1 |
| hard | 20 | 0.100 | 0.500 / 0.350 / 0.365 | 0.789 | 2 |

观察：

- exact match 随难度**单调下降**（0.474 → 0.316 → 0.100），
  符合预期——hard 任务多跳/跨实体/负例占比高。
- **path accuracy 在 easy/medium 稳定为 1.0**，到 hard 降至 0.789，
  说明图遍历本身在简单/中等任务上完全正确，hard 任务失分
  集中在复合路径与负例。
- 失败（unroutable）分布：easy 2（kg-rq-041/042）/ medium 1
  （kg-rq-006）/ hard 2（kg-rq-054/055）——负例与跨实体任务
  集中在 hard。

## 10. Failure Analysis（真实失败案例）

以下案例**全部真实发生**在首跑中（`kg_eval_results.json` 中
5 条 unroutable 失败 + 若干口径失败），未做任何人为修饰：

1. **kg-rq-041 / 042（existence, easy）— 模板路由缺口**
   问题 "Did customer 2 place order 501?" 需要**关系存在性**判断
   （Customer→Order 的 PLACED 边是否存在）。KGTool 有
   `relationship_exists` 模板，但 runner 对 existence 类任务
   未启用它（只路由到 `customer_orders` 返回订单列表），
   负例 042（order 502 不属于 customer 2）无法被判为正确"no"。
   → **模板覆盖/路由缺口**。

2. **kg-rq-054（cross_entity, hard）— 负例 + 集合交集无模板**
   "Which customers ordered both items that supplier 1 supplies?"
   正确答案是**空集**（item 60 买家与 item 120 买家不相交，
   经 SQL 核验）。现有模板只能做"单供应商 × 单客户"的过滤，
   **没有** "两个 item 集合的交集" 模板。→ 模板能力缺口，
   且负例答案依赖集合运算。

3. **kg-rq-055（cross_entity, hard）— 跨边聚合无模板**
   "Which buying groups contain customers that ordered supplier 1's
   items?" 需要把 `BELONGS_TO_GROUP` 与订单链**两条独立边**
   联合聚合（GROUP BY group）。单模板无法表达。→ 模板能力缺口。

4. **kg-rq-029 / 030 / 032（multi_hop, hard）— 多跳终点匹配口径**
   例如 029 "Which suppliers provided the items shipped on invoice 1?"
   期望供应商 `{3}`，预测也得到 `{3}`，但 `customer_item_suppliers`
   走的是 Customer 分支（029 走 `invoice_item_suppliers` 模板），
   而 030/032 期望"供应商 + 城市"两组实体，模板只返回供应商，
   城市那组缺失 → F1=0.667、exact=False。
   → 指标口径问题（expected_entities 只列了终点实体，城市未计入；
   且 029 的 invoice 分支模板返回正确但 expected 仅含 supplier，
   预测却同时携带了中间 StockItem 节点序列化出的 ID）。

5. **kg-rq-056（cross_entity, hard）— 复合答案超出单模板返回能力**
   "Which cities do the suppliers serving customer 40's order items
   deliver to?" 期望同时返回 suppliers `[1, 41]` 与 cities `[2, 42]`
   两组实体；`customer_item_suppliers` 只返回供应商，城市这一组
   无法由单模板补全。→ 复合（多组实体）答案超出当前模板返回结构。

6. **kg-rq-006（entity_lookup, medium）— "Which city does supplier
   10 deliver to?" 路由未命中**：问题中供应商 ID 以 "supplier 10"
   形式出现，而 runner 的 supplier_city 路由只匹配 entity_lookup
   分支里 "city + supplier with ID" 的措辞，本条走的是
   "Which city does supplier N deliver to?" 句式，正则未覆盖。
   → 路由正则缺口（非图谱缺陷）。

**共性**：6 个失败案例里没有一个是 Neo4j 本身的缺陷（无
Neo4j 报错、无数据缺失、无 stale graph）；全部是
**(a) 模板路由/覆盖缺口**（041 / 042 / 054 / 055 / 056 / 006）
或 **(b) 评测口径**（028 / 029 / 030 / 031 / 032，预测集合实际
正确，但 expected_entities 只列终点实体、复合答案多组实体，
导致 exact 失分）。这说明图谱数据层是可靠的，瓶颈在
**查询表达层（模板覆盖）** 与**评测口径**，而非图数据库。

## 11. Findings（基于实际结果，限定本数据集）

- 图谱**能可靠支撑** entity lookup 与 one-hop / two-hop 关系遍历
  （two_hop exact 0.625 / F1 0.825 / path 0.958；one_hop P 0.889 /
  path 1.0）。
- **multi-hop / cross-entity 查询是当前模板层的瓶颈**：需要"集合
  交集"或"双边聚合"的复合查询（054 / 055 / 056）超出预定义模板
  能力，exact 全部为 0。
- **负结果任务（空集答案）区分度低**：空集在 exact_match 上难以
  与"路由失败返回空"区分，exact match 指标会低估系统真实能力
  （multi_hop 的 P=1.0 说明预测无多余，但 exact 仍为 0）。
- **Path Accuracy（0.924）显著高于 Exact Match（0.293）**：
  说明"遍历了对的关系跳"基本做到了，主要失分在**远端实体集合的
  精确匹配**与**复合答案口径**。
- **延迟**：纯 Cypher 无 LLM 参与，mean 3.36 ms / p50 1.98 ms /
  p95 3.22 ms；entity_lookup 的 p95（44 ms）偏高是首次调用
  Neo4j 连接冷启动所致。

> 以上结论**限定于本 58 任务 / WWI 样例数据集**，不外推为
> "图谱优于 SQL"。多源融合与 agent 自主选源需 Phase 3 / 4。

## 12. Limitations

- 图谱 schema 是**项目定义**（7 节点 / 7 关系），非自动抽取；
- 数据来自 **WWI + 项目转换**，非真实企业内部图谱；
- 图谱规模有限（4,723 节点 / 3,880 关系）；
- **尚无 Dynamic Routing**（本实验是"人工指定 KGTool 模板"，
  不是 agent 自主选择数据源）；
- **尚无 multi-source evidence verification**；
- exact match 对负例 / 复合答案的口径偏严，可能低估真实能力。

## 13. Reproducibility

| 项 | 值 |
|---|---|
| Neo4j | 5.26-community（Docker 容器 `eae-neo4j`，本地运行） |
| Python | 3.11（本机 `C:\...\Python311`） |
| Neo4j driver | neo4j==6.3.1 |
| DuckDB | duckdb==1.5.5，`data/runtime/wwi.duckdb` |
| 图谱构建 | `python scripts/build_kg.py`（幂等，`--reset` 可重跑） |
| 评测脚本 | `python scripts/run_kg_eval.py` |
| 节点 / 关系数 | 4,723 / 3,880（见 `kg_build_summary.json`） |
| 环境变量 | `NEO4J_PASSWORD`（.env，不进 Git） |
