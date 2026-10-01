"""API route: analytics - BI-style aggregations over the DuckDB dataset.

Phase 6.5: the Analytics page shows KPI / trends / supplier ranking /
order statistics.  All numbers come directly from the DuckDB dataset
(Sales_Invoices / Sales_InvoiceLines / Warehouse_StockItems /
Purchasing_Suppliers / Sales_Customers) — no hardcoded figures in the
frontend.  When the dataset is not present the endpoints return
``{"available": false}`` so the page shows a correct empty state.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from pathlib import Path
from typing import Any

from fastapi import APIRouter

router = APIRouter(tags=["analytics"])
logger = logging.getLogger("eae.api.analytics")

_CACHE: dict[str, dict[str, Any]] = {}


def _db_path() -> Path:
    """Same database the SQL pipeline uses (data/runtime/wwi.duckdb)."""
    from src.core.sql_executor import default_database_path

    return default_database_path()


def _run_sql(query: str) -> list[dict[str, Any]]:
    """Run one read-only query against the DuckDB dataset.

    Uses a short-lived, explicitly closed connection per request — the
    route is a thin read-only surface, not a long-lived app resource.
    """
    import duckdb

    path = _db_path()
    conn = duckdb.connect(str(path), read_only=True)
    try:
        cur = conn.execute(query)
        columns = [d[0] for d in cur.description]
        return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]
    finally:
        conn.close()


def _dataset_available() -> bool:
    """True only when the expected tables actually exist in the dataset."""
    path = _db_path()
    if not path.exists():
        return False
    try:
        import duckdb

        conn = duckdb.connect(str(path), read_only=True)
        try:
            rows = conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name IN ('Sales_Invoices', 'Sales_InvoiceLines')"
            ).fetchall()
            return len(rows) >= 2
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return False


def _f(v: Any) -> float:
    return float(v) if isinstance(v, (int, float, Decimal)) else 0.0


@router.get("/analytics/summary")
def analytics_summary() -> dict[str, Any]:
    """KPI snapshot: totals, counts, averages over the invoice data."""
    if "summary" in _CACHE:
        return _CACHE["summary"]
    if not _dataset_available():
        return {"available": False, "reason": "DuckDB dataset not found"}

    try:
        rows = _run_sql(
            "SELECT "
            "COUNT(*) AS order_count, "
            "COUNT(DISTINCT i.CustomerID) AS customer_count, "
            "COUNT(DISTINCT l.StockItemID) AS product_count, "
            "SUM(l.ExtendedPrice) AS total_sales, "
            "AVG(l.ExtendedPrice) AS avg_line_amount "
            "FROM Sales_InvoiceLines l "
            "JOIN Sales_Invoices i ON l.InvoiceID = i.InvoiceID"
        )
        r = rows[0]
        kpis = [
            {"label": "订单行数", "value": int(r["order_count"]), "unit": "行"},
            {"label": "客户数", "value": int(r["customer_count"]), "unit": "家"},
            {"label": "产品数", "value": int(r["product_count"]), "unit": "种"},
            {"label": "销售额", "value": round(_f(r["total_sales"]), 2), "unit": "元"},
            {"label": "行均价", "value": round(_f(r["avg_line_amount"]), 2), "unit": "元"},
        ]
        _CACHE["summary"] = {"available": True, "kpis": kpis}
    except Exception:  # noqa: BLE001
        logger.exception("analytics summary query failed")
        _CACHE["summary"] = {"available": False, "reason": "query failed"}
    return _CACHE["summary"]


@router.get("/analytics/trends")
def analytics_trends() -> dict[str, Any]:
    """Monthly sales trend, top suppliers, top customers, order mix."""
    if "trends" in _CACHE:
        return _CACHE["trends"]
    if not _dataset_available():
        return {"available": False, "reason": "DuckDB dataset not found"}

    try:
        monthly = _run_sql(
            "SELECT strftime(i.InvoiceDate, '%Y-%m') AS month, "
            "SUM(l.ExtendedPrice) AS total "
            "FROM Sales_InvoiceLines l "
            "JOIN Sales_Invoices i ON l.InvoiceID = i.InvoiceID "
            "GROUP BY 1 ORDER BY 1"
        )
        suppliers = _run_sql(
            "SELECT s.SupplierName, SUM(l.ExtendedPrice) AS total "
            "FROM Sales_InvoiceLines l "
            "JOIN Warehouse_StockItems st ON l.StockItemID = st.StockItemID "
            "JOIN Purchasing_Suppliers s ON st.SupplierID = s.SupplierID "
            "GROUP BY 1 ORDER BY 2 DESC LIMIT 10"
        )
        customers = _run_sql(
            "SELECT c.CustomerName, SUM(l.ExtendedPrice) AS total "
            "FROM Sales_InvoiceLines l "
            "JOIN Sales_Invoices i ON l.InvoiceID = i.InvoiceID "
            "JOIN Sales_Customers c ON i.CustomerID = c.CustomerID "
            "GROUP BY 1 ORDER BY 2 DESC LIMIT 10"
        )
        invoices = _run_sql(
            "SELECT COUNT(*) AS n, "
            "SUM(CASE WHEN IsCreditNote THEN 1 ELSE 0 END) AS credit_notes, "
            "MIN(InvoiceDate) AS first_date, MAX(InvoiceDate) AS last_date "
            "FROM Sales_Invoices"
        )
        r = invoices[0]
        _CACHE["trends"] = {
            "available": True,
            "monthly_sales": [
                {"month": m["month"], "total": round(_f(m["total"]), 2)} for m in monthly
            ],
            "top_suppliers": [
                {"name": s["SupplierName"], "total": round(_f(s["total"]), 2)}
                for s in suppliers
            ],
            "top_customers": [
                {"name": c["CustomerName"], "total": round(_f(c["total"]), 2)}
                for c in customers
            ],
            "order_stats": {
                "total": int(r["n"]),
                "credit_notes": int(r["credit_notes"] or 0),
                "first_invoice": str(r["first_date"]),
                "last_invoice": str(r["last_date"]),
            },
        }
    except Exception:  # noqa: BLE001
        logger.exception("analytics trends query failed")
        _CACHE["trends"] = {"available": False, "reason": "query failed"}
    return _CACHE["trends"]


__all__ = ["router"]
