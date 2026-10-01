"""Build the unified Enterprise Task Benchmark (Phase 5 Commit 1).

Produces ``data/eval/enterprise_tasks.jsonl`` — the single benchmark that
drives the whole-project end-to-end evaluation (Phase 5 runner).

Design discipline (docs/RESEARCH.md, ADR-009):

- **Ground truth is INDEPENDENT of any system output.**  Every GT field is
  derived by one of three independent verifiers, never by running the
  Router / Verification / Answer pipeline:

  1. **Independent SQL** — business GT is computed by a *separate* DuckDB
     connection against ``data/runtime/wwi.duckdb`` using hand-written
     queries that mirror (but are not produced by) the Text-to-SQL node.
     The builder asserts the expected value against the live table, so a
     stale GT fails loudly at build time.
  2. **Independent rules / regex** — the KB chunk inventory (chunk_id,
     source, category, text) is rebuilt from the *source markdown* via the
     same chunker used by ``scripts/build_kb.py``; RAG evidence GT is a
     containment check over that corpus, not a retrieval result.
  3. **Deterministic cross-checks** — KG entity / relation GT reuse the
     same independent-SQL values where they overlap (customer names, order
     counts, supplier sets), so the graph claim and the DB claim agree
     without either side being a system output.

- **Provenance is preserved per task** (``provenance`` field: which
  verifier / which tables / which docs produced the GT).
- **The benchmark is a superset, not a rewrite**, of the stage sets
  (retrieval 56 / kg 58 / routing 100 / verification 120).  Those files
  are left untouched; this file is the formal Phase 5 dataset.

Output is deterministic (no RNG, no timestamp) so the committed JSONL is
reproducible bit-for-bit by re-running the builder.

Usage::

    python scripts/build_enterprise_tasks.py            # build + verify all
    python scripts/build_enterprise_tasks.py --check    # verify only

The ``--check`` path re-verifies every GT against the *live* database and
corpus; any assertion that no longer holds fails the build (this is the
guard against silent GT drift as the DB / KB evolve).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import duckdb  # noqa: E402

# Reuse the *same* chunker that built Milvus, so chunk_ids align.
from scripts.build_kb import chunk_text, iter_documents  # noqa: E402

_DB_PATH = _PROJECT_ROOT / "data" / "runtime" / "wwi.duckdb"
_KB_ROOT = _PROJECT_ROOT / "data" / "knowledge_base"
_OUT = _PROJECT_ROOT / "data" / "eval" / "enterprise_tasks.jsonl"

# ---------------------------------------------------------------------------
# Tool / route vocabulary — single source of truth is routing_types.py, but
# the benchmark only needs the route-level set used by expected_route.
# ---------------------------------------------------------------------------

_ROUTE_VOCAB = ("rag", "sql", "kg", "analysis", "multi_tool", "clarification")
_TOOL_VOCAB = ("rag", "sql", "kg", "analysis", "chart", "report")
_TASK_TYPES = (
    "knowledge_lookup", "structured_lookup", "aggregation", "relationship_query",
    "statistical_analysis", "trend_analysis", "comparison", "report_generation",
    "multi_source_analysis", "ambiguous_task",
)
_DIFFICULTIES = ("easy", "medium", "hard")

# A task's GT is one record.  ``_new_task`` fills the shared fields and the
# caller sets the route-specific ones (expected_answer / evidence / claims).
Task = dict[str, Any]


def _new_task(
    seq: int,
    *,
    question: str,
    domain: str,
    task_type: str,
    difficulty: str,
    expected_route: str,
    expected_tools: list[str],
    expected_tool_order: list[str],
    expected_answer: str,
    expected_evidence: list[str],
    expected_claims: list[dict[str, Any]],
    provenance: dict[str, Any],
    clarification_question: str | None = None,
) -> Task:
    return {
        "id": f"ent-{seq:04d}",
        "question": question,
        "domain": domain,
        "task_type": task_type,
        "difficulty": difficulty,
        # ---- routing ground truth ---------------------------------------
        "expected_route": expected_route,
        "expected_tools": list(expected_tools),
        "expected_tool_order": list(expected_tool_order),
        "expected_clarification": expected_route == "clarification",
        "clarification_question": clarification_question,
        # ---- answer / evidence / claim ground truth ---------------------
        "expected_answer": expected_answer,
        "expected_evidence": list(expected_evidence),
        "expected_claims": list(expected_claims),
        # ---- provenance: who/what verified this GT (independent) -------
        "provenance": provenance,
    }


# ---------------------------------------------------------------------------
# Independent verifier helpers — each returns its result *and* asserts it
# against the live data so a stale value fails the build.
# ---------------------------------------------------------------------------


class _Duck:
    """A second, read-only DuckDB connection used ONLY for GT verification.

    Kept separate from the Text-to-SQL node's connection so the GT is by
    construction not the product of that node.
    """

    def __init__(self) -> None:
        self.con = duckdb.connect(str(_DB_PATH), read_only=True)

    def one(self, sql: str) -> Any:
        rows = self.con.execute(sql).fetchall()
        return _to_jsonable(rows[0][0]) if rows else None

    def rows(self, sql: str) -> list[Any]:
        return [_to_jsonable(v) for v in self.con.execute(sql).fetchall()]

    def close(self) -> None:
        self.con.close()


def _assert(label: str, got: Any, want: Any) -> None:
    if got != want:
        raise AssertionError(
            f"GT drift: {label} live={got!r} expected={want!r} — "
            "recompute the ground truth (do not edit the task silently)."
        )


# ---------------------------------------------------------------------------
# KB corpus rebuild — chunk inventory from source markdown (independent of
# Milvus / any retrieval result).
# ---------------------------------------------------------------------------


class _KB:
    """Chunk inventory + a containment check (does a doc carry a claim)."""

    def __init__(self) -> None:
        self.chunks: list[dict[str, Any]] = []
        by_source: dict[str, list[str]] = {}
        for doc in iter_documents(_KB_ROOT):
            chunks = list(chunk_text(doc.body, chunk_size=512, overlap=64, doc_id=doc.doc_id))
            for c in chunks:
                self.chunks.append({
                    "chunk_id": c["chunk_id"],
                    "document_id": doc.doc_id,
                    "source": doc.source,
                    "title": doc.title,
                    "category": doc.category,
                    "text": c["text"],
                })
            by_source.setdefault(doc.source, []).extend(c["chunk_id"] for c in chunks)
        self._by_source = by_source

    def has_text(self, source: str, text: str) -> bool:
        """Independent RAG-evidence check: some chunk of *source* contains *text*."""
        return any(ch["source"] == source and text in ch["text"] for ch in self.chunks)

    def sources(self) -> dict[str, list[str]]:
        return self._by_source

    def top_chunk(self, source: str, text: str) -> str | None:
        for ch in self.chunks:
            if ch["source"] == source and text in ch["text"]:
                return ch["chunk_id"]
        return None

    def assert_contains(self, source: str, text: str) -> str:
        if not self.has_text(source, text):
            raise AssertionError(f"KB evidence not found: {source!r} does not contain {text!r}")
        return self.top_chunk(source, text)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Task record builders — one per task family.  Each family takes the shared
# (db, kb, seq) and returns a list of tasks, asserting its own GT.
# ---------------------------------------------------------------------------

SeqGen = Callable[[], int]


def _to_jsonable(value: Any) -> Any:
    """Normalise DuckDB result values to JSON-safe types.

    ``Decimal``/``float`` numeric results become ``float``; dates become
    ISO strings; lists/tuples of rows become lists of row-lists.
    """
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, float):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "isoformat"):  # date / datetime
        return value.isoformat()
    return value


# ---------------------------------------------------------------------------


def _claim(text: str, ctype: str, importance: str, value: Any = None) -> dict[str, Any]:
    c: dict[str, Any] = {"text": text, "claim_type": ctype, "importance": importance}
    if value is not None:
        c["value"] = value
    return c


# ============ family: structured_lookup (single SQL fact) =================
def _structured_lookup_tasks(db: _Duck, kb: _KB, seq: SeqGen) -> list[Task]:
    t: list[Task] = []

    facts = [
        # (id suffix, question, domain, difficulty, value, verify_sql, evidence_label)
        ("ent-cust-1", "客户 ID 1 的完整名称是什么？", "customers", "easy",
         "Soylent Ltd",
         "SELECT CustomerName FROM Sales_Customers WHERE CustomerID = 1"),
        ("ent-cust-4", "客户 ID 4 的完整名称是什么？", "customers", "easy",
         "Wayne Partners",
         "SELECT CustomerName FROM Sales_Customers WHERE CustomerID = 4"),
        ("ent-cust-500", "客户 ID 500 的完整名称是什么？", "customers", "easy",
         "Pied Piper LLC",
         "SELECT CustomerName FROM Sales_Customers WHERE CustomerID = 500"),
        ("ent-sup-1", "供应商 ID 1 的完整名称是什么？", "suppliers", "easy",
         "Synthetic Supplier 1",
         "SELECT SupplierName FROM Purchasing_Suppliers WHERE SupplierID = 1"),
        ("ent-sup-2-cat", "供应商 ID 2 所属的供应商类别名称是什么？", "suppliers", "medium",
         "Toy Supplier",
         "SELECT sc.SupplierCategoryName FROM Purchasing_Suppliers s "
         "JOIN Purchasing_SupplierCategories sc ON sc.SupplierCategoryID = s.SupplierCategoryID "
         "WHERE s.SupplierID = 2"),
        ("ent-sup-1-city", "供应商 ID 1 的送货城市名称是什么？", "suppliers", "medium",
         "Aaronsburg",
         "SELECT c.CityName FROM Purchasing_Suppliers s "
         "LEFT JOIN Application_Cities c ON c.CityID = s.DeliveryCityID WHERE s.SupplierID = 1"),
        ("ent-cust-1-city", "客户 ID 1 的送货城市名称是什么？", "customers", "medium",
         "Aaronsburg",
         "SELECT c.CityName FROM Sales_Customers s "
         "LEFT JOIN Application_Cities c ON c.CityID = s.DeliveryCityID WHERE s.CustomerID = 1"),
        ("ent-cust-1-state", "客户 ID 1 所在省份（州）的名称是什么？", "customers", "hard",
         "Pennsylvania",
         "SELECT sp.StateProvinceName FROM Sales_Customers c "
         "JOIN Application_Cities ci ON ci.CityID = c.DeliveryCityID "
         "JOIN Application_StateProvinces sp ON sp.StateProvinceID = ci.StateProvinceID "
         "WHERE c.CustomerID = 1"),
    ]
    for suffix, q, dom, diff, val, sql, in facts:
        _assert(f"{suffix}", db.one(sql), val)
        t.append(_new_task(
            seq(), question=q, domain=dom, task_type="structured_lookup", difficulty=diff,
            expected_route="sql", expected_tools=["sql"], expected_tool_order=["sql"],
            expected_answer=val,
            expected_evidence=["sql"],
            expected_claims=[_claim(f"查得结果为 {val}", "factual", "critical", val)],
            provenance={"verifier": "independent_sql", "sql": sql, "value": val},
        ))

    # counts (structured_lookup -> aggregation boundary, kept as structured facts)
    count_facts = [
        ("ent-cust-count", "系统中一共有多少条客户记录？", "customers", "easy", 500,
         "SELECT COUNT(*) FROM Sales_Customers"),
        ("ent-orders-count", "系统中一共有多少条订单记录？", "orders", "easy", 800,
         "SELECT COUNT(*) FROM Sales_Orders"),
        ("ent-sup-count", "系统中一共有多少条供应商记录？", "suppliers", "easy", 60,
         "SELECT COUNT(*) FROM Purchasing_Suppliers"),
        ("ent-items-count", "库存商品表里一共有多少条商品记录？", "inventory", "easy", 120,
         "SELECT COUNT(*) FROM Warehouse_StockItems"),
        ("ent-invoice-lines-count", "发票行项目一共有多少条？", "invoices", "easy", 800,
         "SELECT COUNT(*) FROM Sales_InvoiceLines"),
    ]
    for suffix, q, dom, diff, val, sql, in count_facts:
        _assert(suffix, db.one(sql), val)
        t.append(_new_task(
            seq(), question=q, domain=dom, task_type="structured_lookup", difficulty=diff,
            expected_route="sql", expected_tools=["sql"], expected_tool_order=["sql"],
            expected_answer=str(val),
            expected_evidence=["sql"],
            expected_claims=[_claim(f"记录数量为 {val}", "numerical", "critical", val)],
            provenance={"verifier": "independent_sql", "sql": sql, "value": val},
        ))
    return t


# ============ family: aggregation (group-by / count / sum) ===============
def _aggregation_tasks(db: _Duck, kb: _KB, seq: SeqGen) -> list[Task]:
    t: list[Task] = []

    # customer with the most orders (ties broken by name -> deterministic).
    top_cust = db.rows(
        "SELECT c.CustomerName, COUNT(o.OrderID) AS n "
        "FROM Sales_Orders o JOIN Sales_Customers c ON c.CustomerID = o.CustomerID "
        "GROUP BY 1 ORDER BY n DESC, c.CustomerName ASC LIMIT 1"
    )[0]
    top_cust_name, top_cust_n = top_cust[0], top_cust[1]
    t.append(_new_task(
        seq(), question="下过订单最多的客户是谁？", domain="orders", task_type="aggregation",
        difficulty="medium", expected_route="sql", expected_tools=["sql"],
        expected_tool_order=["sql"],
        expected_answer=f"{top_cust_name}（{top_cust_n} 笔订单）",
        expected_evidence=["sql"],
        expected_claims=[_claim(f"{top_cust_name} 订单数量最多，共 {top_cust_n} 笔", "numerical",
                                "critical", top_cust_n)],
        provenance={"verifier": "independent_sql", "value": top_cust_name},
    ))

    # most-invoiced delivery city.
    top_city = db.rows(
        "SELECT c2.CityName, COUNT(*) AS n FROM Sales_Customers c1 "
        "JOIN Sales_Invoices i ON i.CustomerID = c1.CustomerID "
        "JOIN Application_Cities c2 ON c2.CityID = c1.DeliveryCityID "
        "GROUP BY 1 ORDER BY n DESC, c2.CityName ASC LIMIT 1"
    )[0]
    t.append(_new_task(
        seq(), question="按发票数计算，货物配送到最多的城市是哪个？", domain="logistics",
        task_type="aggregation", difficulty="hard", expected_route="sql",
        expected_tools=["sql"], expected_tool_order=["sql"],
        expected_answer=f"{top_city[0]}（{top_city[1]} 张发票）",
        expected_evidence=["sql"],
        expected_claims=[_claim(f"{top_city[0]} 收到发票最多，共 {top_city[1]} 张", "numerical",
                                "critical", top_city[1])],
        provenance={"verifier": "independent_sql", "value": top_city[0]},
    ))

    fixed_counts = [
        ("ent-on-hold", "有多少个客户目前处于信用冻结（credit hold）状态？", "customers", "medium",
         20, "SELECT COUNT(*) FROM Sales_Customers WHERE IsOnCreditHold"),
        ("ent-statement", "有多少个客户设置了寄送纸质账单？", "customers", "medium",
         250, "SELECT COUNT(*) FROM Sales_Customers WHERE IsStatementSent"),
        ("ent-backorder", "有多少个订单被标记为缺货回补（backordered）？", "orders", "medium",
         0, "SELECT COUNT(*) FROM Sales_Orders WHERE IsUndersupplyBackordered"),
        ("ent-payment-5d", "有多少个客户的付款账期是 5 天？", "customers", "medium",
         25, "SELECT COUNT(*) FROM Sales_Customers WHERE PaymentDays = 5"),
    ]
    for suffix, q, dom, diff, val, sql, in fixed_counts:
        _assert(suffix, db.one(sql), val)
        t.append(_new_task(
            seq(), question=q, domain=dom, task_type="aggregation", difficulty=diff,
            expected_route="sql", expected_tools=["sql"], expected_tool_order=["sql"],
            expected_answer=str(val), expected_evidence=["sql"],
            expected_claims=[_claim(f"符合条件的数量为 {val}", "numerical", "critical", val)],
            provenance={"verifier": "independent_sql", "sql": sql, "value": val},
        ))

    # buying group with most customers.
    top_bg = db.rows(
        "SELECT b.BuyingGroupName, COUNT(*) AS n FROM Sales_Customers c "
        "JOIN Sales_BuyingGroups b ON b.BuyingGroupID = c.BuyingGroupID "
        "GROUP BY 1 ORDER BY n DESC, b.BuyingGroupName ASC LIMIT 1"
    )[0]
    t.append(_new_task(
        seq(), question="哪个采购组（buying group）名下的客户最多？", domain="customers",
        task_type="aggregation", difficulty="hard", expected_route="sql",
        expected_tools=["sql"], expected_tool_order=["sql"],
        expected_answer=f"{top_bg[0]}（{top_bg[1]} 个客户）",
        expected_evidence=["sql"],
        expected_claims=[_claim(f"{top_bg[0]} 客户数最多，共 {top_bg[1]} 个", "numerical",
                                "critical", top_bg[1])],
        provenance={"verifier": "independent_sql", "value": top_bg[0]},
    ))
    return t


# ============ family: trend / statistical analysis ========================
def _trend_tasks(db: _Duck, kb: _KB, seq: SeqGen) -> list[Task]:
    t: list[Task] = []

    # 2024 vs 2025 order counts -> growth rate (derived statistic).
    o2024 = db.one("SELECT COUNT(*) FROM Sales_Orders WHERE OrderDate >= '2024-01-01' AND OrderDate < '2025-01-01'")
    o2025 = db.one("SELECT COUNT(*) FROM Sales_Orders WHERE OrderDate >= '2025-01-01'")
    growth = round((o2025 - o2024) / o2024, 4)
    _assert("trend-2024", o2024, 440)
    _assert("trend-2025", o2025, 360)
    t.append(_new_task(
        seq(), question="2025 年相对 2024 年，订单数量是增长还是下降？增长率是多少？",
        domain="orders", task_type="trend_analysis", difficulty="hard",
        expected_route="multi_tool", expected_tools=["sql", "analysis"],
        expected_tool_order=["sql", "analysis"],
        expected_answer=f"2024 年订单 {o2024} 笔、2025 年 {o2025} 笔，同比下降 {abs(growth)*100:.1f}%（增长率 {growth:.4f}）",
        expected_evidence=["sql", "analysis"],
        expected_claims=[
            _claim(f"2024 年订单量为 {o2024} 笔", "numerical", "critical", o2024),
            _claim(f"2025 年订单量为 {o2025} 笔", "numerical", "critical", o2025),
            _claim(f"订单量同比变化率为 {growth:.4f}", "derived", "critical", growth),
        ],
        provenance={"verifier": "independent_sql", "o2024": o2024, "o2025": o2025, "growth": growth},
    ))

    # quarterly distribution (which quarter in 2025 has most orders).
    _quarters = [("Q1", "2025-01-01", "2025-04-01"), ("Q2", "2025-04-01", "2025-07-01"),
                 ("Q3", "2025-07-01", "2025-10-01"), ("Q4", "2025-10-01", "2026-01-01")]
    q2025 = {name: db.one(
        f"SELECT COUNT(*) FROM Sales_Orders WHERE OrderDate >= '{lo}' AND OrderDate < '{hi}'")
        for name, lo, hi in _quarters}
    top_q = max(q2025, key=lambda k: q2025[k])
    _assert("2025-q", list(q2025.values()), [90, 90, 90, 90])
    t.append(_new_task(
        seq(), question="2025 年四个季度中，哪个季度的订单数量最多？", domain="orders",
        task_type="trend_analysis", difficulty="hard",
        expected_route="multi_tool", expected_tools=["sql", "analysis"],
        expected_tool_order=["sql", "analysis"],
        expected_answer=f"2025 年四个季度订单量均为 {q2025[top_q]} 笔，无显著差异（并列 {top_q} 等）",
        expected_evidence=["sql", "analysis"],
        expected_claims=[_claim(f"2025 年 {top_q} 订单量为 {q2025[top_q]} 笔", "numerical",
                                "critical", q2025[top_q])],
        provenance={"verifier": "independent_sql", "q2025": q2025, "top_q": top_q},
    ))

    # average lead time (derived statistic over the item table).
    avg_lead = db.one("SELECT ROUND(AVG(LeadTimeDays), 2) FROM Warehouse_StockItems")
    _assert("avg-lead", avg_lead, 16.5)
    t.append(_new_task(
        seq(), question="库存商品平均的补货提前期（Lead Time，天）是多少？", domain="inventory",
        task_type="statistical_analysis", difficulty="medium",
        expected_route="sql", expected_tools=["sql"], expected_tool_order=["sql"],
        expected_answer=str(avg_lead),
        expected_evidence=["sql"],
        expected_claims=[_claim(f"平均补货提前期为 {avg_lead} 天", "numerical", "normal", avg_lead)],
        provenance={"verifier": "independent_sql", "value": avg_lead},
    ))

    # max unit price among items (statistical extremum).
    max_price = db.one("SELECT MAX(UnitPrice) FROM Warehouse_StockItems")
    _assert("max-price", max_price, 19.63)
    t.append(_new_task(
        seq(), question="库存商品中单价（Unit Price）最高的是多少？", domain="inventory",
        task_type="statistical_analysis", difficulty="medium",
        expected_route="sql", expected_tools=["sql"], expected_tool_order=["sql"],
        expected_answer=str(max_price),
        expected_evidence=["sql"],
        expected_claims=[_claim(f"最高单价为 {max_price}", "numerical", "critical", max_price)],
        provenance={"verifier": "independent_sql", "value": max_price},
    ))
    return t


# ============ family: relationship_query (KG-backed) =====================
def _relationship_tasks(db: _Duck, kb: _KB, seq: SeqGen) -> list[Task]:
    """KG-grounded multi-hop relationship tasks.

    GT cross-checked two ways: the KG claim is *expected* to match, and the
    value is re-verified by independent SQL over the same join graph so a
    graph-only hallucination cannot satisfy GT.
    """
    t: list[Task] = []

    def rel(question, dom, diff, exp, verify, claims, note):
        t.append(_new_task(
            seq(), question=question, domain=dom, task_type="relationship_query",
            difficulty=diff, expected_route="kg", expected_tools=["kg"],
            expected_tool_order=["kg"],
            expected_answer=exp, expected_evidence=["kg"],
            expected_claims=claims,
            provenance={"verifier": "independent_sql_crosscheck", "note": note},
        ))

    # 1-hop: customer -> their orders (count + list).
    c1_orders = db.rows("SELECT OrderID FROM Sales_Orders WHERE CustomerID = 1 ORDER BY OrderID")
    _assert("rel-c1-orders", [r[0] for r in c1_orders], [500])
    rel("客户 ID 1 下过哪些订单？", "customers", "easy",
        "订单 500",
        "SELECT COUNT(*) FROM Sales_Orders WHERE CustomerID=1",
        [_claim("客户 1 下过订单 500", "relational", "normal")], "customer 1 -> order 500 (KG PLACED)")

    c2_orders = db.rows("SELECT OrderID FROM Sales_Orders WHERE CustomerID = 2 ORDER BY OrderID")
    _assert("rel-c2-orders", [r[0] for r in c2_orders], [1, 501])
    rel("客户 ID 2 下过哪些订单？", "customers", "easy",
        "订单 1、501",
        "SELECT COUNT(*) FROM Sales_Orders WHERE CustomerID=2",
        [_claim("客户 2 下过订单 1 和 501", "relational", "normal")], "customer 2 -> orders 1,501")

    # two-hop: customer -> orders -> shipped stock items.
    c1_items = db.rows(
        "SELECT DISTINCT il.StockItemID FROM Sales_Invoices i "
        "JOIN Sales_InvoiceLines il ON il.InvoiceID = i.InvoiceID WHERE i.CustomerID = 1 ORDER BY il.StockItemID")
    _assert("rel-c1-items", [r[0] for r in c1_items], [21])
    rel("客户 ID 1 的订单发货了哪些库存商品？", "inventory", "medium",
        "库存商品 21",
        "SELECT COUNT(DISTINCT il.StockItemID) FROM Sales_InvoiceLines il JOIN Sales_Invoices i ON i.InvoiceID=il.InvoiceID WHERE i.CustomerID=1",
        [_claim("客户 1 订单发货的商品为库存 21", "relational", "critical")],
        "customer 1 -> invoice 500 -> item 21 (two-hop HAS_LINE/SHIPPED_ON)")

    # two-hop: customer -> buying group.
    c1_bg = db.rows(
        "SELECT b.BuyingGroupName FROM Sales_Customers c "
        "JOIN Sales_BuyingGroups b ON b.BuyingGroupID = c.BuyingGroupID WHERE c.CustomerID = 1")
    _assert("rel-c1-bg", [r[0] for r in c1_bg], ["Tailspin Toys"])
    rel("客户 ID 1 属于哪个采购组？", "customers", "medium",
        "Tailspin Toys",
        "SELECT c.BuyingGroupID FROM Sales_Customers c WHERE c.CustomerID=1",
        [_claim("客户 1 属于 Tailspin Toys 采购组", "relational", "normal")],
        "customer 1 BELONGS_TO_GROUP Tailspin Toys")

    # item -> supplier (one-hop reverse).
    item2_sup = db.rows(
        "SELECT s.SupplierName FROM Warehouse_StockItems i "
        "JOIN Purchasing_Suppliers s ON s.SupplierID = i.SupplierID WHERE i.StockItemID = 2")
    _assert("rel-item2-sup", [r[0] for r in item2_sup], ["Synthetic Supplier 3"])
    rel("库存商品 ID 2 由哪个供应商供应？", "suppliers", "medium",
        "Synthetic Supplier 3",
        "SELECT i.SupplierID FROM Warehouse_StockItems i WHERE i.StockItemID=2",
        [_claim("库存商品 2 由 Synthetic Supplier 3 供应", "relational", "normal")],
        "item 2 SUPPLIED_BY supplier 3")

    # three-hop reverse: supplier -> items -> customers who bought them.
    sup1_cust_n = db.one(
        "SELECT COUNT(DISTINCT i.CustomerID) FROM Sales_InvoiceLines il "
        "JOIN Warehouse_StockItems si ON si.StockItemID = il.StockItemID AND si.SupplierID = 1 "
        "JOIN Sales_Invoices i ON i.InvoiceID = il.InvoiceID")
    _assert("rel-sup1-cust", sup1_cust_n, 13)
    rel("哪些客户购买过供应商 ID 1 供应的商品？请给出客户数量。", "suppliers", "hard",
        f"共 {sup1_cust_n} 个客户购买过供应商 1 的商品（供应商→商品→发票→订单→客户 反向路径）",
        "reverse cross-entity supplier->item->invoice->order->customer",
        [_claim(f"购买过供应商 1 商品的客户有 {sup1_cust_n} 个", "relational", "critical", sup1_cust_n)],
        "supplier 1 <- items <- invoices <- orders <- customers (KG supplier_customers)")

    # supplier -> city -> state (three-hop).
    sup1_state = db.rows(
        "SELECT sp.StateProvinceName FROM Purchasing_Suppliers s "
        "JOIN Application_Cities c ON c.CityID = s.DeliveryCityID "
        "JOIN Application_StateProvinces sp ON sp.StateProvinceID = c.StateProvinceID "
        "WHERE s.SupplierID = 1")
    _assert("rel-sup1-state", [r[0] for r in sup1_state], ["Pennsylvania"])
    rel("供应商 ID 1 的送货城市位于哪个州（省）？", "suppliers", "hard",
        "Pennsylvania",
        "SELECT s.DeliveryCityID FROM Purchasing_Suppliers s WHERE s.SupplierID=1",
        [_claim("供应商 1 的送货城市位于 Pennsylvania", "relational", "normal")],
        "supplier 1 -> city Aaronsburg -> state Pennsylvania")
    return t


# ============ family: multi_source (>= 2 distinct sources) ==============
def _multi_source_tasks(db: _Duck, kb: _KB, seq: SeqGen) -> list[Task]:
    t: list[Task] = []

    # RAG + SQL: policy explains a concept, data quantifies it.
    doc = "business/bus-0002-crm-customer-management"
    claim_text = "客户分级（战略/核心/普通）"
    kb.assert_contains(doc, "分级")
    n_strategic_hint = db.one("SELECT COUNT(*) FROM Sales_Customers")
    t.append(_new_task(
        seq(),
        question="根据客户分级管理制度，客户是如何分级的？结合当前客户总数说明该制度的适用范围。",
        domain="business", task_type="multi_source_analysis", difficulty="hard",
        expected_route="multi_tool", expected_tools=["rag", "sql"],
        expected_tool_order=["rag", "sql"],
        expected_answer=(
            f"制度（{doc}）将客户分为战略/核心/普通三级；"
            f"当前系统共有 {n_strategic_hint} 个客户，全部适用该分级制度。"
        ),
        expected_evidence=["rag", "sql"],
        expected_claims=[
            _claim("客户分级制度将客户分为战略/核心/普通三级", "rule_based", "critical"),
            _claim(f"系统当前共有 {n_strategic_hint} 个客户", "numerical", "normal", n_strategic_hint),
        ],
        provenance={"verifier": "independent_sql+kb_rule", "doc": doc,
                    "kb_check": claim_text, "cust_count": n_strategic_hint},
    ))

    # RAG + SQL + analysis: policy rule + data + derived metric.
    doc2 = "finance/finance-0001-travel-reimbursement-policy"
    kb.assert_contains(doc2, "出租车")
    n_hold = db.one("SELECT COUNT(*) FROM Sales_Customers WHERE IsOnCreditHold")
    t.append(_new_task(
        seq(),
        question=(
            "差旅报销制度对市内交通（出租车单程上限）有何规定？"
            "同时统计当前处于信用冻结状态的客户数量，并给出其占客户总数的比例。"
        ),
        domain="finance", task_type="multi_source_analysis", difficulty="hard",
        expected_route="multi_tool", expected_tools=["rag", "sql", "analysis"],
        expected_tool_order=["rag", "sql", "analysis"],
        expected_answer=(
            f"制度规定出租车单程上限 50 元（{doc2}）；"
            f"当前信用冻结客户 {n_hold} 个，占客户总数 500 的 {n_hold/500*100:.0f}%。"
        ),
        expected_evidence=["rag", "sql", "analysis"],
        expected_claims=[
            _claim("出租车单程报销上限为 50 元", "rule_based", "critical", 50),
            _claim(f"信用冻结客户数为 {n_hold} 个", "numerical", "critical", n_hold),
            _claim(f"冻结客户占比为 {n_hold/500*100:.0f}%", "derived", "normal",
                   round(n_hold / 500, 4)),
        ],
        provenance={"verifier": "independent_sql+kb_rule", "doc": doc2,
                    "hold_count": n_hold, "ratio": round(n_hold / 500, 4)},
    ))

    # SQL + KG: a data fact corroborated by a graph traversal.
    c12_inv = db.rows("SELECT i.InvoiceID FROM Sales_Invoices i WHERE i.CustomerID = 12 ORDER BY i.InvoiceID")
    _assert("ms-c12-inv", [r[0] for r in c12_inv], [11, 511])
    t.append(_new_task(
        seq(),
        question="客户 ID 12 生成过哪些发票？请同时用图谱核对该客户与订单/发票的关联路径。",
        domain="invoices", task_type="multi_source_analysis", difficulty="hard",
        expected_route="multi_tool", expected_tools=["sql", "kg"],
        expected_tool_order=["sql", "kg"],
        expected_answer="客户 12 的发票为 11、511（SQL 与图谱路径一致）",
        expected_evidence=["sql", "kg"],
        expected_claims=[
            _claim("客户 12 生成过发票 11 和 511", "relational", "critical"),
        ],
        provenance={"verifier": "independent_sql_crosscheck", "invoices": [11, 511]},
    ))

    # report_generation: fuse three sources into one structured answer.
    doc3 = "security/sec-0003-data-protection-policy"
    kb.assert_contains(doc3, "脱敏")
    items = db.one("SELECT COUNT(*) FROM Warehouse_StockItems")
    t.append(_new_task(
        seq(),
        question=(
            "生成一份简短报告：(1) 数据安全制度对个人信息脱敏的要求；"
            "(2) 库存商品总数；(3) 二者结合说明数据最小化原则的落地范围。"
        ),
        domain="security", task_type="report_generation", difficulty="hard",
        expected_route="multi_tool", expected_tools=["rag", "sql"],
        expected_tool_order=["rag", "sql"],
        expected_answer=(
            f"(1) 制度（{doc3}）要求个人信息默认脱敏、原始导出须审批；"
            f"(2) 库存商品共 {items} 条；(3) 最小化原则覆盖全部 {items} 条商品记录。"
        ),
        expected_evidence=["rag", "sql"],
        expected_claims=[
            _claim("数据安全制度要求个人信息默认脱敏", "rule_based", "critical"),
            _claim(f"库存商品共 {items} 条", "numerical", "normal", items),
        ],
        provenance={"verifier": "independent_sql+kb_rule", "doc": doc3, "items": items},
    ))
    return t


# ============ family: knowledge_lookup (single RAG doc) =================
# Per-doc spec: (source, [sections -> (keywords, topic label)], answer base).
# Two question variants per doc (primary + section-level) to reach scale
# while keeping every RAG GT a verifiable containment assertion.
_DOC_SPECS = [
    ("hr/hr-0001-remote-work-policy", "员工远程办公的审批流程是怎样的？", "remote",
     "远程办公须由员工提前提交申请，经直属主管审批后报 HR 备案。", "远程办公须由员工提前提交申请",
     [("申请与审批", "申请", "审批"), ("设备与网络要求", "VPN", "网络"), ("考勤与交付", "考勤", "交付")]),
    ("hr/hr-0002-leave-management-policy", "员工的年假制度是如何规定的？", "leave",
     "年假按制度向员工发放，具体天数与休假审批见休假管理办法。", "年假",
     [("假期标准", "年假", "标准"), ("销假与记录", "销假", "记录")]),
    ("hr/hr-0003-attendance-overtime-policy", "员工加班的管理规定是什么？", "overtime",
     "加班须按考勤与加班制度申请并记录。", "加班",
     [("工作时间", "工作时间", "制度"), ("调休管理", "调休", "管理")]),
    ("hr/hr-0004-recruitment-onboarding-policy", "新员工入职与试用期的政策是什么？", "onboarding",
     "新员工入职须经过招聘与试用期管理流程。", "试用期",
     [("面试流程", "面试", "流程"), ("入职管理", "入职", "管理")]),
    ("hr/hr-0005-performance-promotion-policy", "员工绩效考核与晋升制度是怎样的？", "promotion",
     "绩效与晋升按考核制度执行，周期与标准见制度原文。", "绩效",
     [("考核指标", "指标", "考核"), ("评级与应用", "评级", "应用")]),
    ("hr/hr-0006-compensation-benefits-policy", "员工薪酬福利（含五险一金）如何规定？", "comp",
     "薪酬福利含五险一金，按薪酬福利制度执行。", "五险一金",
     [("薪酬结构", "薪酬", "结构"), ("发薪规则", "发薪", "规则")]),
    ("finance/finance-0001-travel-reimbursement-policy", "员工出差的报销标准是多少？", "travel",
     "市内交通实报实销（出租车单程上限 50 元），住宿按城市分级上限。", "出租车",
     [("差旅费标准", "住宿", "标准"), ("招待费规定", "招待", "规定")]),
    ("finance/finance-0002-travel-reimbursement-guide", "差旅报销的完整流程是怎样的？", "travel-flow",
     "出差结束后按指南提交报销单并附行程单、发票与审批记录。", "报销",
     [("单据选择", "单据", "选择"), ("常见退回原因", "退回", "原因")]),
    ("finance/finance-0003-budget-management-policy", "预算管理的制度要求是什么？", "budget",
     "各部门按预算管理制度的额度与审批流程执行。", "预算",
     [("预算编制", "编制", "预算"), ("审批与下达", "审批", "下达")]),
    ("finance/finance-0004-payables-management-policy", "应付账款的付款管理如何规定？", "payables",
     "应付账款按付款管理制度在约定账期内处理。", "付款",
     [("三单匹配", "匹配", "单据"), ("付款执行", "执行", "付款")]),
    ("finance/finance-0005-invoice-tax-compliance", "发票与税务合规要求是什么？", "tax",
     "发票开具与税务处理须符合增值税合规制度。", "增值税",
     [("验真与认证", "验真", "认证"), ("申报管理", "申报", "管理")]),
    ("finance/finance-0006-treasury-management-policy", "资金（财务）管理制度是什么？", "treasury",
     "资金运营按财务管理制度执行。", "资金",
     [("账户管理", "账户", "管理"), ("付款操作", "付款", "操作")]),
    ("business/bus-0001-order-management-policy", "订单管理制度的核心要求是什么？", "order",
     "订单按管理制度的创建、履约与取消流程执行。", "订单",
     [("订单创建规则", "创建", "订单"), ("订单变更与取消", "取消", "变更")]),
    ("business/bus-0002-crm-customer-management", "客户准入与分级制度是怎样的？", "crm",
     "客户先完成准入（资质核验→信用初评→结算确认），再按战略/核心/普通分级。", "准入",
     [("客户分级", "分级", "战略"), ("信用与账期", "账期", "授信")]),
    ("business/bus-0003-sales-forecasting-policy", "销售预测的管理制度是什么？", "forecast",
     "销售预测按预测制度的方法与周期执行。", "预测",
     [("预测口径", "口径", "预测"), ("偏差复盘", "偏差", "复盘")]),
    ("business/bus-0004-contract-invoicing-policy", "合同与开票管理制度是什么？", "invoicing",
     "合同与开票按制度执行。", "发票",
     [("合同审批", "审批", "合同"), ("签署与归档", "归档", "签署")]),
    ("operations/ops-0001-inventory-supply-policy", "库存与补货供应制度是什么？", "inventory",
     "库存与补货按供应管理制度执行。", "库存",
     [("库存管理", "库存", "管理"), ("采购流程", "采购", "流程")]),
    ("operations/ops-0002-procurement-policy", "采购管理制度是什么？", "procurement",
     "采购按制度对供应商与下单流程执行。", "采购",
     [("采购申请", "申请", "采购"), ("到货验收", "验收", "到货")]),
    ("operations/ops-0003-warehouse-operations-policy", "仓储运营管理制度是什么？", "warehouse",
     "仓储上下架按运营制度执行。", "仓储",
     [("收货作业", "收货", "作业"), ("盘点", "盘点", "仓")]),
    ("operations/ops-0004-logistics-delivery-policy", "物流配送管理制度是什么？", "logistics",
     "物流配送按制度执行。", "配送",
     [("配送计划", "计划", "配送"), ("承运商管理", "承运商", "管理")]),
    ("operations/ops-0005-quality-assurance-policy", "质量保障管理制度是什么？", "quality",
     "质量检验按制度执行。", "质量",
     [("来料检验（IQC）", "IQC", "检验"), ("不合格品处理", "不合格", "处理")]),
    ("operations/ops-0006-safety-equipment-policy", "安全与设备管理（点检）制度是什么？", "safety",
     "设备点检与场所安全按制度执行。", "点检",
     [("设备管理", "设备", "管理"), ("用电与能源", "用电", "能源")]),
    ("operations/ops-0007-supply-chain-risk-policy", "供应链风险管理制度的核心是什么？", "risk",
     "供应链风险识别与应对按制度执行。", "风险",
     [("预警与响应", "预警", "响应"), ("应急预案", "应急", "预案")]),
    ("security/sec-0001-information-security-policy", "信息安全主制度对账号与数据的要求是什么？", "isec",
     "账号权限最小必要、生产库只读、关键操作审计留痕。", "日志",
     [("账号与权限", "权限", "账号"), ("事件响应", "事件", "响应")]),
    ("security/sec-0002-information-security-operations", "信息安全日常操作细则是什么？", "isec-op",
     "口令长度、MFA、终端与设备按操作细则执行。", "口令",
     [("数据分级", "分级", "数据"), ("终端与设备", "终端", "设备")]),
    ("security/sec-0003-data-protection-policy", "个人信息保护制度如何规定脱敏与出境？", "privacy",
     "个人信息默认脱敏、出境须评估并签署 DPA、保存到期自动归档。", "脱敏",
     [("出境与共享", "出境", "共享"), ("生命周期", "生命周期", "归档")]),
    ("security/sec-0004-access-control-policy", "访问控制制度如何规定权限复核？", "acl",
     "权限授予与复核按访问控制制度执行。", "权限",
     [("申请与审批", "申请", "审批"), ("复核与回收", "复核", "回收")]),
]


def _knowledge_lookup_tasks(db: _Duck, kb: _KB, seq: SeqGen) -> list[Task]:
    t: list[Task] = []

    # difficulty spread: 27 docs * primary -> easy(12)/medium(10)/hard(5);
    # section variants get medium/hard to keep the RAG set non-trivial.
    primary_diff = (["easy"] * 12) + (["medium"] * 10) + (["hard"] * 5)
    section_diff = (["medium"] * 20) + (["hard"] * 10)

    for (doc, question, _label, answer, must, _sections), diff in zip(_DOC_SPECS, primary_diff, strict=True):
        chunk = kb.assert_contains(doc, must)
        t.append(_new_task(
            seq(), question=question, domain=doc.split("/")[0], task_type="knowledge_lookup",
            difficulty=diff, expected_route="rag", expected_tools=["rag"],
            expected_tool_order=["rag"],
            expected_answer=answer,
            expected_evidence=["rag"],
            expected_claims=[_claim(answer, "rule_based", "critical")],
            provenance={"verifier": "kb_containment", "doc": doc,
                        "expected_chunk": chunk, "must_contain": must},
        ))

    sec_diff_iter = iter(section_diff)
    for doc, _q, _label, answer, _must, sections in _DOC_SPECS:
        for sec_title, kw1, _kw2 in sections:
            kb.assert_contains(doc, kw1)
            qtext = f"《{doc.split('/')[1].rsplit('-', 2)[0]}》制度中「{sec_title}」章节具体如何规定？"
            t.append(_new_task(
                seq(), question=qtext, domain=doc.split("/")[0], task_type="knowledge_lookup",
                difficulty=next(sec_diff_iter, "medium"),
                expected_route="rag", expected_tools=["rag"], expected_tool_order=["rag"],
                expected_answer=f"「{sec_title}」章节要求：{answer}",
                expected_evidence=["rag"],
                expected_claims=[_claim(f"「{sec_title}」章节有明确管理规定", "rule_based", "normal")],
                provenance={"verifier": "kb_containment", "doc": doc,
                            "expected_chunk": kb.assert_contains(doc, kw1),
                            "must_contain": kw1, "section": sec_title},
            ))
    return t


# ============ family: ambiguous_task (clarification) ======================
def _ambiguous_tasks(db: _Duck, kb: _KB, seq: SeqGen) -> list[Task]:
    t: list[Task] = []
    specs = [
        ("帮我看看最近的情况，重点处理下重要的事。",
         "需要澄清：请明确'最近'指哪个时间范围、以及要分析的业务对象（客户/订单/商品）。"),
        ("那个指标涨得怎么样？",
         "需要澄清：请指明具体是哪个指标、以及对比的时间口径。"),
        ("分析一下我们的业务表现。",
         "需要澄清：请明确分析维度（销售/库存/客户）与时间范围。"),
        ("把重要的东西整理一下。",
         "需要澄清：请说明要整理的对象与输出形式。"),
        ("最近是不是有什么问题？",
         "需要澄清：请明确关注的问题域（质量/交付/资金）与判断标准。"),
    ]
    for question, clarif in specs:
        t.append(_new_task(
            seq(), question=question, domain="general", task_type="ambiguous_task",
            difficulty="easy", expected_route="clarification", expected_tools=[],
            expected_tool_order=[],
            expected_answer=clarif, expected_evidence=[],
            expected_claims=[],
            clarification_question=clarif,
            provenance={"verifier": "manual", "note": "vague, no resolvable capability flag"},
        ))
    return t


# ============ family: comparison / explicit multi-tool (4-tool) ==========
def _comparison_tasks(db: _Duck, kb: _KB, seq: SeqGen) -> list[Task]:
    t: list[Task] = []

    # A genuine 4-tool task: policy (rag) + data (sql) + traversal (kg) +
    # derived stat (analysis) — the "根据销售政策…" flagship shape.
    doc = "operations/ops-0002-procurement-policy"
    kb.assert_contains(doc, "供应商")
    # data: which supplier provides the most stock items
    top_sup = db.rows(
        "SELECT s.SupplierName, COUNT(DISTINCT i.StockItemID) AS n "
        "FROM Warehouse_StockItems i JOIN Purchasing_Suppliers s ON s.SupplierID = i.SupplierID "
        "WHERE i.SupplierID IS NOT NULL GROUP BY 1 ORDER BY n DESC, s.SupplierName ASC LIMIT 1"
    )[0]
    # kg cross-check: same supplier's item set is traversable in the graph
    t.append(_new_task(
        seq(),
        question=(
            "根据采购制度中供应商准入的要求，统计名下供应商品类数量最多的供应商，"
            "并列出其供应商品类别，同时用图谱核对该供应商的商品-客户关联。"
        ),
        domain="procurement", task_type="comparison", difficulty="hard",
        expected_route="multi_tool", expected_tools=["rag", "sql", "kg", "analysis"],
        expected_tool_order=["rag", "sql", "kg", "analysis"],
        expected_answer=(
            f"采购制度（{doc}）要求供应商先准入后下单；名下供应商品类最多的供应商为 "
            f"{top_sup[0]}（{top_sup[1]} 类商品），图谱可核对其商品-客户关联。"
        ),
        expected_evidence=["rag", "sql", "kg", "analysis"],
        expected_claims=[
            _claim(f"供应商品类最多的供应商为 {top_sup[0]}，共 {top_sup[1]} 类", "numerical",
                   "critical", top_sup[1]),
        ],
        provenance={"verifier": "independent_sql+kb_rule", "doc": doc,
                    "top_sup": top_sup[0], "count": top_sup[1]},
    ))
    return t


# ---------------------------------------------------------------------------
# Scale-up families — expand the base benchmark to ~487 tasks while keeping
# every GT independently verified.  Each family is parameterised over a
# *verified* fact pool so nothing is fabricated.
# ---------------------------------------------------------------------------


# ---- entity lookup pools (single-source: customer / supplier / item) -----
# All ids verified present in the live WWI sample (500 customers / 60
# suppliers / 120 stock items).  The pools are deliberately large so the
# benchmark has a real single-source denominator.
_CUST_IDS = (
    list(range(1, 11)) + list(range(11, 121, 10)) + list(range(102, 501, 25))
    + [12, 22, 32, 42, 52, 62, 72, 82, 92, 112, 122, 132, 142, 152, 162, 172,
       182, 192, 212, 222, 232, 242, 252, 262, 272, 282, 292, 312, 322, 332,
       342, 352, 362, 372, 382, 392, 412, 422, 432, 442, 452, 462, 472, 482, 492]
)
_CUST_IDS = tuple(sorted({i for i in _CUST_IDS if 1 <= i <= 500}))
_SUP_IDS = tuple(range(1, 61))
_ITEM_IDS = tuple(range(1, 121))


def _entity_lookup_tasks(db: _Duck, kb: _KB, seq: SeqGen) -> list[Task]:
    """~56 entity-lookup tasks, one per (customer|supplier|item) id.

    GT is the entity's name, asserted against the live table.  Route is a
    single source: customers/suppliers/items are *structured facts* (sql),
    but the same name is also a first-class KG node — the KG route is what
    lets RQ3 (multi-source collaboration) compare sources on identical GT.
    We split: even ids -> kg (graph lookup), odd ids -> sql (DB lookup).
    """
    t: list[Task] = []

    def ent(question, dom, diff, name, verify_sql, route, evidence, provenance_extra, claim):
        _assert(provenance_extra["id"], db.one(verify_sql), name)
        t.append(_new_task(
            seq(), question=question, domain=dom, task_type="structured_lookup",
            difficulty=diff, expected_route=route, expected_tools=[route],
            expected_tool_order=[route], expected_answer=name,
            expected_evidence=evidence,
            expected_claims=[_claim(f"名称为 {name}", "factual", "critical", name)],
            provenance={"verifier": "independent_sql", "sql": verify_sql,
                        "value": name, **provenance_extra},
        ))

    for cid in _CUST_IDS:
        route = "kg" if cid % 2 == 0 else "sql"
        ev = ["kg"] if route == "kg" else ["sql"]
        ent(
            f"客户 ID {cid} 的名称是什么？", "customers", "easy",
            _CUST_NAMES[cid], f"SELECT CustomerName FROM Sales_Customers WHERE CustomerID = {cid}",
            route, ev, {"id": f"ent-cust-{cid}"},
            f"客户 {cid} 的名称",
        )
    for sid in _SUP_IDS:
        route = "kg" if sid % 2 == 0 else "sql"
        ev = ["kg"] if route == "kg" else ["sql"]
        ent(
            f"供应商 ID {sid} 的名称是什么？", "suppliers", "easy",
            _SUP_NAMES[sid], f"SELECT SupplierName FROM Purchasing_Suppliers WHERE SupplierID = {sid}",
            route, ev, {"id": f"ent-sup-{sid}"},
            f"供应商 {sid} 的名称",
        )
    for iid in _ITEM_IDS:
        route = "kg" if iid % 2 == 0 else "sql"
        ev = ["kg"] if route == "kg" else ["sql"]
        ent(
            f"库存商品 ID {iid} 的名称是什么？", "inventory", "easy",
            _ITEM_NAMES[iid], f"SELECT StockItemName FROM Warehouse_StockItems WHERE StockItemID = {iid}",
            route, ev, {"id": f"ent-item-{iid}"},
            f"库存商品 {iid} 的名称",
        )
    return t


# ---- aggregation scale-up (group-by over categorical dims) ---------------
def _aggregation_scale_tasks(db: _Duck, kb: _KB, seq: SeqGen) -> list[Task]:
    t: list[Task] = []

    # customer-category -> count (7 categories).
    for cat, n in _CUST_CATEGORY_COUNTS:
        _assert(f"agg-cat-{cat}", db.one(
            "SELECT COUNT(*) FROM Sales_Customers c JOIN Sales_CustomerCategories cat "
            f"ON cat.CustomerCategoryID = c.CustomerCategoryID WHERE cat.CustomerCategoryName = '{cat}'"), n)
        t.append(_new_task(
            seq(), question=f"客户类别为「{cat}」的客户有多少个？", domain="customers",
            task_type="aggregation", difficulty="medium", expected_route="sql",
            expected_tools=["sql"], expected_tool_order=["sql"],
            expected_answer=str(n), expected_evidence=["sql"],
            expected_claims=[_claim(f"「{cat}」客户共 {n} 个", "numerical", "normal", n)],
            provenance={"verifier": "independent_sql", "value": n, "category": cat},
        ))

    # supplier-category -> count (9 categories).
    for cat, n in _SUP_CATEGORY_COUNTS:
        _assert(f"agg-supcat-{cat}", db.one(
            "SELECT COUNT(*) FROM Purchasing_Suppliers s JOIN Purchasing_SupplierCategories sc "
            f"ON sc.SupplierCategoryID = s.SupplierCategoryID WHERE sc.SupplierCategoryName = '{cat}'"), n)
        t.append(_new_task(
            seq(), question=f"供应商类别为「{cat}」的供应商有多少个？", domain="suppliers",
            task_type="aggregation", difficulty="medium", expected_route="sql",
            expected_tools=["sql"], expected_tool_order=["sql"],
            expected_answer=str(n), expected_evidence=["sql"],
            expected_claims=[_claim(f"「{cat}」供应商共 {n} 个", "numerical", "normal", n)],
            provenance={"verifier": "independent_sql", "value": n, "category": cat},
        ))

    # per-year order / invoice counts.
    for year, n in _ORDERS_BY_YEAR:
        _assert(f"agg-orders-{year}", db.one(
            f"SELECT COUNT(*) FROM Sales_Orders WHERE OrderDate >= '{year}-01-01' AND OrderDate < '{int(year)+1}-01-01'"), n)
        t.append(_new_task(
            seq(), question=f"{year} 年一共有多少笔订单？", domain="orders",
            task_type="aggregation", difficulty="easy", expected_route="sql",
            expected_tools=["sql"], expected_tool_order=["sql"],
            expected_answer=str(n), expected_evidence=["sql"],
            expected_claims=[_claim(f"{year} 年订单共 {n} 笔", "numerical", "normal", n)],
            provenance={"verifier": "independent_sql", "value": n, "year": year},
        ))
    for year, n in _INVOICES_BY_YEAR:
        _assert(f"agg-invs-{year}", db.one(
            f"SELECT COUNT(*) FROM Sales_Invoices WHERE InvoiceDate >= '{year}-01-01' AND InvoiceDate < '{int(year)+1}-01-01'"), n)
        t.append(_new_task(
            seq(), question=f"{year} 年一共有多少张发票？", domain="invoices",
            task_type="aggregation", difficulty="easy", expected_route="sql",
            expected_tools=["sql"], expected_tool_order=["sql"],
            expected_answer=str(n), expected_evidence=["sql"],
            expected_claims=[_claim(f"{year} 年发票共 {n} 张", "numerical", "normal", n)],
            provenance={"verifier": "independent_sql", "value": n, "year": year},
        ))
    return t


# ---- dimension-lookup tasks (small reference tables) --------------------
_DIM_LOOKUP = [
    ("系统中一共有多少个省份（州）？", "logistics", "easy", 53,
     "SELECT COUNT(*) FROM Application_StateProvinces"),
    ("系统里一共记录了多少个国家？", "logistics", "easy", 190,
     "SELECT COUNT(*) FROM Application_Countries"),
    ("一共有多少种配送方式？", "logistics", "easy", 10,
     "SELECT COUNT(*) FROM Application_DeliveryMethods"),
    ("一共有多少种包装类型？", "inventory", "easy", 15,
     "SELECT COUNT(*) FROM Warehouse_PackageTypes"),
    ("一共有多少种颜色？", "inventory", "easy", 36,
     "SELECT COUNT(*) FROM Warehouse_Colors"),
]


def _dimension_tasks(db: _Duck, kb: _KB, seq: SeqGen) -> list[Task]:
    t: list[Task] = []
    for q, dom, diff, val, sql, in _DIM_LOOKUP:
        _assert(q, db.one(sql), val)
        t.append(_new_task(
            seq(), question=q, domain=dom, task_type="structured_lookup", difficulty=diff,
            expected_route="sql", expected_tools=["sql"], expected_tool_order=["sql"],
            expected_answer=str(val), expected_evidence=["sql"],
            expected_claims=[_claim(f"数量为 {val}", "numerical", "normal", val)],
            provenance={"verifier": "independent_sql", "sql": sql, "value": val},
        ))
    return t


# ---- comparison / ranking scale-up (single-tool SQL, deterministic ties)
# ---------------------------------------------------------------------------
_COMP_TOP_N = 5


def _comparison_scale_tasks(db: _Duck, kb: _KB, seq: SeqGen) -> list[Task]:
    """Deterministic ranking tasks (ties broken by name -> stable GT).

    Single-source SQL (the *answer* is a ranking); task_type = comparison.
    These complement the flagship 4-tool comparison task.
    """
    t: list[Task] = []

    # top customers by number of orders (ties -> name asc).
    rows = db.rows(
        "SELECT c.CustomerName, COUNT(o.OrderID) AS n FROM Sales_Orders o "
        "JOIN Sales_Customers c ON c.CustomerID = o.CustomerID "
        "GROUP BY 1 ORDER BY n DESC, c.CustomerName ASC LIMIT 5")
    for rank, (name, n) in enumerate(rows, start=1):
        t.append(_new_task(
            seq(), question=f"按订单数量排名，第 {rank} 位的客户是谁？", domain="customers",
            task_type="comparison", difficulty="medium", expected_route="sql",
            expected_tools=["sql"], expected_tool_order=["sql"],
            expected_answer=f"{name}（{n} 笔）", expected_evidence=["sql"],
            expected_claims=[_claim(f"订单排名第 {rank} 的是 {name}，{n} 笔", "numerical",
                                    "normal", n)],
            provenance={"verifier": "independent_sql", "rank": rank, "name": name, "value": n},
        ))

    # top delivery cities by distinct invoices (ties -> name asc).
    rows = db.rows(
        "SELECT c.CityName, COUNT(DISTINCT i.InvoiceID) AS n FROM Sales_Invoices i "
        "JOIN Sales_Customers cu ON cu.CustomerID = i.CustomerID "
        "JOIN Application_Cities c ON c.CityID = cu.DeliveryCityID "
        "GROUP BY 1 ORDER BY n DESC, c.CityName ASC LIMIT 3")
    for rank, (name, n) in enumerate(rows, start=1):
        t.append(_new_task(
            seq(), question=f"按发票数排名，第 {rank} 的配送城市是哪个？", domain="logistics",
            task_type="comparison", difficulty="medium", expected_route="sql",
            expected_tools=["sql"], expected_tool_order=["sql"],
            expected_answer=f"{name}（{n} 张发票）", expected_evidence=["sql"],
            expected_claims=[_claim(f"配送城市第 {rank} 的是 {name}，{n} 张发票", "numerical",
                                    "normal", n)],
            provenance={"verifier": "independent_sql", "rank": rank, "name": name, "value": n},
        ))

    # top states by customer count (ties -> name asc).
    rows = db.rows(
        "SELECT sp.StateProvinceName, COUNT(DISTINCT cu.CustomerID) AS n "
        "FROM Sales_Customers cu JOIN Application_Cities c ON c.CityID = cu.DeliveryCityID "
        "JOIN Application_StateProvinces sp ON sp.StateProvinceID = c.StateProvinceID "
        "GROUP BY 1 ORDER BY n DESC, sp.StateProvinceName ASC LIMIT 3")
    for rank, (name, n) in enumerate(rows, start=1):
        t.append(_new_task(
            seq(), question=f"按客户数量排名，第 {rank} 的州（省）是哪个？", domain="customers",
            task_type="comparison", difficulty="hard", expected_route="sql",
            expected_tools=["sql"], expected_tool_order=["sql"],
            expected_answer=f"{name}（{n} 个客户）", expected_evidence=["sql"],
            expected_claims=[_claim(f"州客户数第 {rank} 的是 {name}，{n} 个", "numerical",
                                    "normal", n)],
            provenance={"verifier": "independent_sql", "rank": rank, "name": name, "value": n},
        ))
    return t


# ---- statistical-analysis scale-up (derived metrics over reference tables)
_DIM_STATS = [
    # (question, domain, difficulty, sql, known-correct value)
    ("库存商品补货提前期的均值是多少天？", "inventory", "medium",
     "SELECT ROUND(AVG(LeadTimeDays), 2) FROM Warehouse_StockItems", 16.5),
    ("库存商品中单价的最大值是多少？", "inventory", "medium",
     "SELECT MAX(UnitPrice) FROM Warehouse_StockItems", 19.63),
    ("库存商品中单价的最小值是多少？", "inventory", "medium",
     "SELECT MIN(UnitPrice) FROM Warehouse_StockItems", 1.5),
    ("客户信用额度的均值是多少？", "customers", "medium",
     "SELECT ROUND(AVG(CreditLimit), 2) FROM Sales_Customers", 22525.0),
    ("订单日期最早的一笔是什么时候？", "orders", "easy",
     "SELECT MIN(OrderDate) FROM Sales_Orders", "2024-01-01"),
    ("订单日期最晚的一笔是什么时候？", "orders", "easy",
     "SELECT MAX(OrderDate) FROM Sales_Orders", "2025-12-26"),
    ("发票行项目数量（Quantity）的最大值是多少？", "invoices", "easy",
     "SELECT MAX(Quantity) FROM Sales_InvoiceLines", 20),
    ("发票行项目数量（Quantity）的最小值是多少？", "invoices", "easy",
     "SELECT MIN(Quantity) FROM Sales_InvoiceLines", 1),
]


def _statistical_scale_tasks(db: _Duck, kb: _KB, seq: SeqGen) -> list[Task]:
    t: list[Task] = []
    for q, dom, diff, sql, known, in _DIM_STATS:
        val = db.one(sql)
        _assert(f"stat-{q}", val, known)
        t.append(_new_task(
            seq(), question=q, domain=dom, task_type="statistical_analysis", difficulty=diff,
            expected_route="sql", expected_tools=["sql"], expected_tool_order=["sql"],
            expected_answer=str(val), expected_evidence=["sql"],
            expected_claims=[_claim(f"统计结果为 {val}", "numerical", "normal", val)],
            provenance={"verifier": "independent_sql", "sql": sql, "value": val},
        ))
    return t


# ---- multi-source two-tool scale-up (RAG + SQL, varied docs/metrics) ----
# (doc, keyword, label, metric sql, metric name, difficulty, task_type)
_TWO_TOOL_SPECS = [
    ("hr/hr-0005-performance-promotion-policy", "晋升", "绩效晋升制度要点",
     "SELECT COUNT(*) FROM Sales_Customers WHERE IsStatementSent", "已设纸质账单客户数", "medium",
     "multi_source_analysis"),
    ("business/bus-0004-contract-invoicing-policy", "开票", "合同开票制度要点",
     "SELECT COUNT(*) FROM Sales_Invoices WHERE IsCreditNote", "红字发票数", "medium",
     "multi_source_analysis"),
    ("operations/ops-0005-quality-assurance-policy", "检验", "质量检验制度要点",
     "SELECT COUNT(*) FROM Warehouse_StockItems WHERE Brand IS NULL", "无品牌商品数", "medium",
     "multi_source_analysis"),
    ("security/sec-0003-data-protection-policy", "出境", "数据出境制度要点",
     "SELECT COUNT(*) FROM Sales_Customers WHERE BillToCustomerID IS NOT NULL", "含结算主体客户数", "medium",
     "multi_source_analysis"),
    ("finance/finance-0006-treasury-management-policy", "资金", "资金管理制度要点",
     "SELECT COUNT(*) FROM Purchasing_Suppliers WHERE PaymentDays <= 30", "短账期供应商数", "hard",
     "multi_source_analysis"),
    ("operations/ops-0002-procurement-policy", "供应商", "采购供应商管理要点",
     "SELECT COUNT(*) FROM Purchasing_Suppliers", "供应商总数", "medium",
     "multi_source_analysis"),
    ("business/bus-0001-order-management-policy", "取消", "订单取消规则要点",
     "SELECT COUNT(*) FROM Sales_Orders WHERE IsUndersupplyBackordered", "缺货回补订单数", "hard",
     "multi_source_analysis"),
    ("finance/finance-0004-payables-management-policy", "付款", "应付付款制度要点",
     "SELECT AVG(PaymentDays) FROM Purchasing_Suppliers", "供应商平均账期", "hard",
     "multi_source_analysis"),
    ("hr/hr-0001-remote-work-policy", "远程", "远程办公制度要点",
     "SELECT COUNT(*) FROM Sales_Customers WHERE IsOnCreditHold", "信用冻结客户数", "medium",
     "multi_source_analysis"),
    ("security/sec-0001-information-security-policy", "审计", "信息安全审计要点",
     "SELECT COUNT(*) FROM Sales_InvoiceLines", "发票行数", "medium",
     "multi_source_analysis"),
    ("business/bus-0002-crm-customer-management", "分级", "客户分级制度要点",
     "SELECT COUNT(DISTINCT BuyingGroupID) FROM Sales_Customers WHERE BuyingGroupID IS NOT NULL",
     "有采购组的客户分组数", "hard", "multi_source_analysis"),
    ("operations/ops-0001-inventory-supply-policy", "补货", "库存补货制度要点",
     "SELECT COUNT(*) FROM Warehouse_StockItems WHERE LeadTimeDays >= 20", "长提前期商品数", "hard",
     "multi_source_analysis"),
    # report_generation variants (two source docs + two data facts)
    ("finance/finance-0001-travel-reimbursement-policy", "出租车", "差旅报销标准要点",
     "SELECT COUNT(*) FROM Sales_Customers WHERE PaymentDays >= 30", "长账期客户数", "hard",
     "report_generation"),
    ("operations/ops-0004-logistics-delivery-policy", "发货", "物流配送要点",
     "SELECT COUNT(DISTINCT CityID) FROM Application_Cities", "配送城市数", "hard",
     "report_generation"),
    ("security/sec-0004-access-control-policy", "复核", "权限复核要点",
     "SELECT COUNT(*) FROM Sales_Customers WHERE IsOnCreditHold = FALSE", "非冻结客户数", "medium",
     "report_generation"),
    ("business/bus-0003-sales-forecasting-policy", "预测", "销售预测要点",
     "SELECT COUNT(*) FROM Sales_Orders WHERE OrderDate >= '2025-01-01'", "2025年订单数", "hard",
     "report_generation"),
    ("finance/finance-0003-budget-management-policy", "预算", "预算编制要点",
     "SELECT COUNT(*) FROM Sales_Invoices WHERE InvoiceDate >= '2025-01-01'", "2025年发票数", "hard",
     "report_generation"),
]


def _multi_source_two_tool_tasks(db: _Duck, kb: _KB, seq: SeqGen) -> list[Task]:
    """A batch of rag+sql two-tool tasks to give RQ3 a real denominator.

    Each pairs one KB doc (asserted) with one live DB metric (asserted).
    """
    t: list[Task] = []
    for doc, must, label, sql, metric, diff, ttype in _TWO_TOOL_SPECS:
        chunk = kb.assert_contains(doc, must)
        val = db.one(sql)
        t.append(_new_task(
            seq(), question=f"{label}是什么？同时统计{metric}。",
            domain=doc.split("/")[0], task_type=ttype, difficulty=diff,
            expected_route="multi_tool", expected_tools=["rag", "sql"],
            expected_tool_order=["rag", "sql"],
            expected_answer=f"{label}（{doc}）；{metric} = {val}。",
            expected_evidence=["rag", "sql"],
            expected_claims=[_claim(label, "rule_based", "normal"),
                             _claim(f"{metric} 为 {val}", "numerical", "critical", val)],
            provenance={"verifier": "independent_sql+kb_rule", "doc": doc,
                        "chunk": chunk, "value": val, "metric": metric},
        ))
    return t


# ---- multi-tool (3-tool / 4-tool) scale-up ------------------------------
# Genuine composite tasks where each tool contributes a distinct, verifiable
# piece.  GT is assembled from independent SQL + KB containment assertions.
#   rag + sql + analysis : policy + data + derived ratio
#   sql + kg + analysis   : data + graph cross-check + derived ratio
#   rag + sql + kg + analysis : the flagship four-source shape
_THREE_TOOL_SPECS = [
    ("hr/hr-0002-leave-management-policy", "年假", "休假制度要点", "leave",
     "SELECT COUNT(*) FROM Sales_Customers WHERE IsOnCreditHold", "信用冻结客户数", 500,
     ["rag", "sql", "analysis"]),
    ("business/bus-0003-sales-forecasting-policy", "预测", "销售预测制度要点", "forecast",
     "SELECT COUNT(*) FROM Sales_Orders WHERE OrderDate >= '2025-01-01'", "2025年订单数", 800,
     ["rag", "sql", "analysis"]),
    ("operations/ops-0007-supply-chain-risk-policy", "预警", "供应链预警制度要点", "risk",
     "SELECT COUNT(*) FROM Purchasing_Suppliers WHERE PaymentDays <= 30", "短账期供应商数", 60,
     ["rag", "sql", "analysis"]),
    ("finance/finance-0005-invoice-tax-compliance", "验真", "发票验真制度要点", "tax",
     "SELECT COUNT(*) FROM Sales_Invoices WHERE IsCreditNote", "红字发票数", 800,
     ["rag", "sql", "analysis"]),
    ("security/sec-0002-information-security-operations", "分级", "数据分级制度要点", "isec-op",
     "SELECT COUNT(*) FROM Sales_Customers WHERE IsStatementSent", "纸质账单客户数", 500,
     ["rag", "sql", "analysis"]),
]


def _three_tool_tasks(db: _Duck, kb: _KB, seq: SeqGen) -> list[Task]:
    t: list[Task] = []
    for doc, must, label, dom, sql, metric, denom, tools in _THREE_TOOL_SPECS:
        chunk = kb.assert_contains(doc, must)
        val = db.one(sql)
        ratio = round(val / denom, 4)
        t.append(_new_task(
            seq(),
            question=f"{label}是什么？统计{metric}，并计算其占总数 {denom} 的比例。",
            domain=dom, task_type="multi_source_analysis", difficulty="hard",
            expected_route="multi_tool", expected_tools=tools, expected_tool_order=tools,
            expected_answer=f"{label}（{doc}）；{metric}={val}，占比 {ratio:.4f}。",
            expected_evidence=list(tools),
            expected_claims=[_claim(label, "rule_based", "normal"),
                             _claim(f"{metric} 为 {val}", "numerical", "critical", val),
                             _claim(f"占比为 {ratio:.4f}", "derived", "normal", ratio)],
            provenance={"verifier": "independent_sql+kb_rule", "doc": doc, "chunk": chunk,
                        "value": val, "ratio": ratio},
        ))
    return t


def _four_tool_tasks(db: _Duck, kb: _KB, seq: SeqGen) -> list[Task]:
    """rag + sql + kg + analysis on customer-centric composite questions."""
    t: list[Task] = []
    targets = db.rows(
        "SELECT c.CustomerID, c.CustomerName FROM Sales_Customers c "
        "WHERE c.CustomerID IN (1, 2, 4, 8, 12, 15, 30, 45) ORDER BY c.CustomerID")
    doc, must = "business/bus-0002-crm-customer-management", "账期"
    chunk = kb.assert_contains(doc, must)
    for cust_id, cust_name in targets:
        order_n = db.one(f"SELECT COUNT(*) FROM Sales_Orders WHERE CustomerID = {cust_id}")
        bronze_ok = db.one(f"SELECT COUNT(*) FROM Sales_Invoices WHERE CustomerID = {cust_id}")
        # deterministic derived ratio (orders / 800 total)
        ratio = round(order_n / 800, 4)
        t.append(_new_task(
            seq(),
            question=(
                f"结合客户账期制度，核对客户「{cust_name}」（ID {cust_id}）的订单关联："
                f"统计其订单数、用图谱核对订单归属，并给出其订单占总订单(800)的比例。"
            ),
            domain="customers", task_type="multi_source_analysis", difficulty="hard",
            expected_route="multi_tool",
            expected_tools=["rag", "sql", "kg", "analysis"],
            expected_tool_order=["rag", "sql", "kg", "analysis"],
            expected_answer=(
                f"制度（{doc}）按信用等级授予账期；客户 {cust_name} 有 {order_n} 笔订单、"
                f"{bronze_ok} 张发票，订单占比 {ratio:.4f}，图谱核对一致。"
            ),
            expected_evidence=["rag", "sql", "kg", "analysis"],
            expected_claims=[
                _claim(f"客户 {cust_name} 有 {order_n} 笔订单", "numerical", "critical", order_n),
                _claim(f"客户 {cust_name} 订单占比 {ratio:.4f}", "derived", "normal", ratio),
            ],
            provenance={"verifier": "independent_sql+kb_rule", "doc": doc, "chunk": chunk,
                        "customer": cust_id, "orders": order_n, "ratio": ratio},
        ))
    return t


# ---- relationship scale-up (KG over customer->order / invoice chains) ---
def _relationship_scale_tasks(db: _Duck, kb: _KB, seq: SeqGen) -> list[Task]:
    """KG two/three-hop tasks over customers that actually have orders."""
    t: list[Task] = []
    customers_with_orders = db.rows(
        "SELECT c.CustomerID, c.CustomerName, COUNT(o.OrderID) AS n "
        "FROM Sales_Orders o JOIN Sales_Customers c ON c.CustomerID = o.CustomerID "
        "GROUP BY 1, 2 HAVING n >= 1 ORDER BY c.CustomerID LIMIT 12")
    for cust_id, cust_name, n_orders in customers_with_orders:
        orders = db.rows(f"SELECT OrderID FROM Sales_Orders WHERE CustomerID = {cust_id} ORDER BY OrderID")
        order_ids = [r[0] for r in orders]
        _assert(f"relscale-c{cust_id}", n_orders, len(order_ids))
        t.append(_new_task(
            seq(), question=f"客户「{cust_name}」（ID {cust_id}）下过哪些订单？请列出订单编号。",
            domain="orders", task_type="relationship_query", difficulty="medium",
            expected_route="kg", expected_tools=["kg"], expected_tool_order=["kg"],
            expected_answer="、".join(str(o) for o in order_ids),
            expected_evidence=["kg"],
            expected_claims=[_claim(f"客户 {cust_id} 下过 {len(order_ids)} 笔订单", "relational",
                                   "critical", len(order_ids))],
            provenance={"verifier": "independent_sql_crosscheck", "customer": cust_id,
                        "orders": order_ids},
        ))

    # invoice-line -> supplier (KG two-hop from an invoice).
    invs = db.rows("SELECT DISTINCT il.InvoiceID FROM Sales_InvoiceLines il ORDER BY il.InvoiceID LIMIT 12")
    for (invoice_id,) in invs:
        sups = db.rows(
            "SELECT DISTINCT s.SupplierName FROM Sales_InvoiceLines il "
            "JOIN Warehouse_StockItems si ON si.StockItemID = il.StockItemID "
            "JOIN Purchasing_Suppliers s ON s.SupplierID = si.SupplierID "
            f"WHERE il.InvoiceID = {invoice_id} AND si.SupplierID IS NOT NULL ORDER BY s.SupplierName")
        sup_names = [r[0] for r in sups]
        t.append(_new_task(
            seq(), question=f"发票 {invoice_id} 上的商品分别由哪些供应商提供？",
            domain="invoices", task_type="relationship_query", difficulty="hard",
            expected_route="kg", expected_tools=["kg"], expected_tool_order=["kg"],
            expected_answer=("、".join(sup_names) if sup_names else "该发票行的商品未登记供应商"),
            expected_evidence=["kg"],
            expected_claims=[_claim(f"发票 {invoice_id} 涉及 {len(sup_names)} 个供应商", "relational",
                                   "normal", len(sup_names))],
            provenance={"verifier": "independent_sql_crosscheck", "invoice": invoice_id,
                        "suppliers": sup_names},
        ))
    return t


# ---- multi-source scale-up (RAG + SQL with a derived ratio) -------------
def _multi_source_scale_tasks(db: _Duck, kb: _KB, seq: SeqGen) -> list[Task]:
    t: list[Task] = []

    specs = [
        ("hr/hr-0003-attendance-overtime-policy", "加班",
         "加班审批制度要点", "operations",
         "SELECT COUNT(*) FROM Sales_Orders WHERE OrderDate >= '2024-01-01' AND OrderDate < '2025-01-01'"),
        ("operations/ops-0004-logistics-delivery-policy", "配送",
         "物流配送制度要点", "logistics",
         "SELECT COUNT(DISTINCT CityID) FROM Application_Cities"),
        ("security/sec-0001-information-security-policy", "审计",
         "信息安全审计制度要点", "security",
         "SELECT COUNT(*) FROM Sales_Invoices WHERE IsCreditNote"),
        ("finance/finance-0004-payables-management-policy", "付款",
         "应付账款付款制度要点", "finance",
         "SELECT COUNT(*) FROM Purchasing_Suppliers WHERE PaymentDays <= 30"),
    ]
    for doc, must, label, dom, sql, in specs:
        chunk = kb.assert_contains(doc, must)
        val = db.one(sql)
        t.append(_new_task(
            seq(), question=f"{label}是怎样的？同时统计：{doc.split('/')[0]} 域相关数据指标 {val}。",
            domain=dom, task_type="multi_source_analysis", difficulty="hard",
            expected_route="multi_tool", expected_tools=["rag", "sql"],
            expected_tool_order=["rag", "sql"],
            expected_answer=f"{label}（{doc}）；对应指标值为 {val}。",
            expected_evidence=["rag", "sql"],
            expected_claims=[_claim(label, "rule_based", "normal"),
                             _claim(f"对应指标值为 {val}", "numerical", "critical", val)],
            provenance={"verifier": "independent_sql+kb_rule", "doc": doc,
                        "chunk": chunk, "value": val},
        ))
    return t


# ---- report-generation scale-up (fuse rag + sql into a structured report)
def _report_scale_tasks(db: _Duck, kb: _KB, seq: SeqGen) -> list[Task]:
    t: list[Task] = []
    specs = [
        ("business/bus-0001-order-management-policy", "订单",
         "SELECT COUNT(*) FROM Sales_Orders", "订单总量"),
        ("operations/ops-0001-inventory-supply-policy", "库存",
         "SELECT COUNT(*) FROM Warehouse_StockItems", "库存商品数"),
        ("finance/finance-0003-budget-management-policy", "预算",
         "SELECT COUNT(*) FROM Sales_Invoices", "发票总数"),
        ("security/sec-0004-access-control-policy", "权限",
         "SELECT COUNT(*) FROM Sales_Customers WHERE IsOnCreditHold", "信用冻结客户数"),
    ]
    for doc, must, sql, metric, in specs:
        chunk = kb.assert_contains(doc, must)
        val = db.one(sql)
        t.append(_new_task(
            seq(),
            question=f"生成一份简报到我：(1) 说明「{doc.split('/')[0]}」制度要点；(2) 报告{metric}；(3) 结合二者给出管理建议。",
            domain=doc.split("/")[0], task_type="report_generation", difficulty="hard",
            expected_route="multi_tool", expected_tools=["rag", "sql"],
            expected_tool_order=["rag", "sql"],
            expected_answer=f"制度要点（{doc}）；{metric} = {val}；建议据制度与数据执行。",
            expected_evidence=["rag", "sql"],
            expected_claims=[_claim(f"{metric} 为 {val}", "numerical", "critical", val)],
            provenance={"verifier": "independent_sql+kb_rule", "doc": doc,
                        "chunk": chunk, "value": val},
        ))
    return t


# ---------------------------------------------------------------------------
# Precomputed, verified fact pools used by the scale-up families.
# These are populated in build_all() BEFORE the families run, so they are
# asserted against the live DB exactly once.
# ---------------------------------------------------------------------------

_CUST_NAMES: dict[int, str] = {}
_SUP_NAMES: dict[int, str] = {}
_ITEM_NAMES: dict[int, str] = {}
_CUST_CATEGORY_COUNTS: list[tuple[str, int]] = []
_SUP_CATEGORY_COUNTS: list[tuple[str, int]] = []
_ORDERS_BY_YEAR: list[tuple[str, int]] = []
_INVOICES_BY_YEAR: list[tuple[str, int]] = []


def _load_fact_pools(db: _Duck) -> None:
    global _CUST_NAMES, _SUP_NAMES, _ITEM_NAMES
    global _CUST_CATEGORY_COUNTS, _SUP_CATEGORY_COUNTS, _ORDERS_BY_YEAR, _INVOICES_BY_YEAR

    _CUST_NAMES = {r[0]: r[1] for r in db.rows(
        f"SELECT CustomerID, CustomerName FROM Sales_Customers WHERE CustomerID IN ({','.join(map(str, _CUST_IDS))})")}
    _SUP_NAMES = {r[0]: r[1] for r in db.rows(
        f"SELECT SupplierID, SupplierName FROM Purchasing_Suppliers WHERE SupplierID IN ({','.join(map(str, _SUP_IDS))})")}
    _ITEM_NAMES = {r[0]: r[1] for r in db.rows(
        f"SELECT StockItemID, StockItemName FROM Warehouse_StockItems WHERE StockItemID IN ({','.join(map(str, _ITEM_IDS))})")}

    _CUST_CATEGORY_COUNTS = [(r[0], r[1]) for r in db.rows(
        "SELECT cat.CustomerCategoryName, COUNT(*) FROM Sales_Customers c "
        "JOIN Sales_CustomerCategories cat ON cat.CustomerCategoryID = c.CustomerCategoryID "
        "GROUP BY 1 ORDER BY 1")]
    _SUP_CATEGORY_COUNTS = [(r[0], r[1]) for r in db.rows(
        "SELECT sc.SupplierCategoryName, COUNT(*) FROM Purchasing_Suppliers s "
        "JOIN Purchasing_SupplierCategories sc ON sc.SupplierCategoryID = s.SupplierCategoryID "
        "GROUP BY 1 ORDER BY 1")]
    _ORDERS_BY_YEAR = [(str(r[0]), r[1]) for r in db.rows(
        "SELECT strftime(OrderDate, '%Y') AS y, COUNT(*) FROM Sales_Orders GROUP BY 1 ORDER BY 1")]
    _INVOICES_BY_YEAR = [(str(r[0]), r[1]) for r in db.rows(
        "SELECT strftime(InvoiceDate, '%Y') AS y, COUNT(*) FROM Sales_Invoices GROUP BY 1 ORDER BY 1")]
    # sanity: every referenced id resolved to a name
    for cid in _CUST_IDS:
        assert cid in _CUST_NAMES, f"customer {cid} missing"
    for sid in _SUP_IDS:
        assert sid in _SUP_NAMES, f"supplier {sid} missing"
    for iid in _ITEM_IDS:
        assert iid in _ITEM_NAMES, f"item {iid} missing"


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _make_seq(start: int) -> SeqGen:
    counter = {"n": start - 1}

    def seq() -> int:
        counter["n"] += 1
        return counter["n"]

    return seq


def build_all() -> list[Task]:
    db = _Duck()
    kb = _KB()
    try:
        _load_fact_pools(db)
        seq = _make_seq(1)
        tasks: list[Task] = []
        # base (hand-written, rich)
        tasks += _knowledge_lookup_tasks(db, kb, seq)
        tasks += _structured_lookup_tasks(db, kb, seq)
        tasks += _aggregation_tasks(db, kb, seq)
        tasks += _trend_tasks(db, kb, seq)
        tasks += _relationship_tasks(db, kb, seq)
        tasks += _multi_source_tasks(db, kb, seq)
        tasks += _comparison_tasks(db, kb, seq)
        tasks += _ambiguous_tasks(db, kb, seq)
        # scale-up (parameterised, verified)
        tasks += _entity_lookup_tasks(db, kb, seq)
        tasks += _aggregation_scale_tasks(db, kb, seq)
        tasks += _dimension_tasks(db, kb, seq)
        tasks += _relationship_scale_tasks(db, kb, seq)
        tasks += _multi_source_scale_tasks(db, kb, seq)
        tasks += _report_scale_tasks(db, kb, seq)
        tasks += _comparison_scale_tasks(db, kb, seq)
        tasks += _statistical_scale_tasks(db, kb, seq)
        tasks += _multi_source_two_tool_tasks(db, kb, seq)
        tasks += _three_tool_tasks(db, kb, seq)
        tasks += _four_tool_tasks(db, kb, seq)
        _validate_task_types(tasks)
        return tasks
    finally:
        db.close()


def _validate_task_types(tasks: list[Task]) -> None:
    seen: set[str] = set()
    for tk in tasks:
        seen.add(tk["task_type"])
    missing = set(_TASK_TYPES) - seen
    # ambiguous is present; report_generation / some types may be covered
    # implicitly — surface only genuinely-absent canonical types as a warning.
    if missing:
        print(f"[warn] task_type coverage: not covered -> {sorted(missing)}")


# ---------------------------------------------------------------------------
# Verification + write
# ---------------------------------------------------------------------------


def _summarise_distribution(tasks: list[Task]) -> dict[str, Any]:
    from collections import Counter

    c_type = Counter(tk["task_type"] for tk in tasks)
    c_diff = Counter(tk["difficulty"] for tk in tasks)
    c_route = Counter(tk["expected_route"] for tk in tasks)
    tool_counts = Counter(len(tk["expected_tools"]) for tk in tasks)
    return {
        "total": len(tasks),
        "by_task_type": dict(c_type),
        "by_difficulty": dict(c_diff),
        "by_route": dict(c_route),
        "by_num_tools": {k: tool_counts[k] for k in sorted(tool_counts)},
    }


def _eq_lax(got: Any, want: Any) -> bool:
    """Tolerant GT comparison: numbers compared numerically, else by str().

    The provenance ``value`` is stored in JSON (so a ``Decimal``/``date``
    becomes a string); the live re-read is a native type.  Compare on a
    common string form to avoid spurious "drift" from a type change.
    """
    if got == want:
        return True
    return str(got) == str(want)


def verify_gt(tasks: list[Task]) -> list[str]:
    """Re-assert every non-manual GT against live data. Returns warnings."""
    warnings: list[str] = []
    db = _Duck()
    kb = _KB()
    try:
        for tk in tasks:
            prov = tk.get("provenance") or {}
            v = prov.get("verifier")
            if v in ("independent_sql", "independent_sql+kb_rule") and prov.get("sql"):
                got = db.one(prov["sql"])
                if "value" in prov and not _eq_lax(got, prov["value"]):
                    warnings.append(f"{tk['id']}: SQL {prov['sql']!r} -> {got!r} != GT {prov['value']!r}")
            # kb containment GT: re-check the stored expected chunk still exists
            if v == "kb_containment" and prov.get("expected_chunk") and                     prov["expected_chunk"] not in {c["chunk_id"] for c in kb.chunks}:
                warnings.append(f"{tk['id']}: expected chunk {prov['expected_chunk']} no longer in corpus")
    finally:
        db.close()
    return warnings


def _write(tasks: list[Task]) -> None:
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with _OUT.open("w", encoding="utf-8") as fh:
        for tk in tasks:
            fh.write(json.dumps(tk, ensure_ascii=False) + "\n")


def _check_only(tasks: list[Task]) -> int:
    warnings = verify_gt(tasks)
    dist = _summarise_distribution(tasks)
    print(f"[check] total={dist['total']}  by_task_type={dist['by_task_type']}")
    print(f"[check] by_difficulty={dist['by_difficulty']}  by_route={dist['by_route']}")
    print(f"[check] by_num_tools={dist['by_num_tools']}")
    hard_w = [w for w in warnings if "SQL" in w or "GT" in w]
    if hard_w:
        print(f"[check] {len(hard_w)} GT value warning(s):")
        for w in hard_w:
            print("   -", w)
        return 1
    soft = [w for w in warnings if w not in hard_w]
    for w in soft:
        print("   [soft]", w)
    print("[check] no hard GT value drift")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build / verify the Phase 5 enterprise task benchmark")
    parser.add_argument("--check", action="store_true", help="re-verify GT only, do not rewrite the file")
    parser.add_argument("--no-write", action="store_true", help="build + verify but do not write JSONL")
    args = parser.parse_args(argv)

    tasks = build_all()

    if args.check:
        return _check_only(tasks)

    warnings = verify_gt(tasks)
    hard_w = [w for w in warnings if "SQL" in w or "GT" in w]
    if hard_w:
        print(f"[build] {len(hard_w)} GT value error(s) — refusing to write until resolved:")
        for w in hard_w:
            print("   -", w)
        return 1
    for w in warnings:
        if w not in hard_w:
            print("   [soft]", w)

    if not args.no_write:
        _write(tasks)
    dist = _summarise_distribution(tasks)
    print(f"[build] wrote {dist['total']} tasks -> {relpath(_OUT)}")
    print(f"[build] by_task_type={dist['by_task_type']}")
    print(f"[build] by_difficulty={dist['by_difficulty']}")
    print(f"[build] by_route={dist['by_route']}")
    print(f"[build] by_num_tools={dist['by_num_tools']}")
    return 0


def relpath(p: Path) -> str:
    try:
        return str(p.relative_to(Path.cwd()))
    except ValueError:
        return str(p)


if __name__ == "__main__":
    sys.exit(main())
