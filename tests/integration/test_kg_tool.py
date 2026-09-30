"""Integration test for the KG tool against a live Neo4j.

Skipped unless ``NEO4J_INTEGRATION=1`` (with ``NEO4J_PASSWORD`` set).  Every
query result is cross-checked against DuckDB SQL so the test verifies both
graph correctness and graph/SQL consistency.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pytest

from src.tools.kg_tool import KGTool

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("NEO4J_INTEGRATION") != "1",
        reason="set NEO4J_INTEGRATION=1 (with a running Neo4j + NEO4J_PASSWORD) to enable",
    ),
]


@pytest.fixture(scope="module")
def db():
    con = duckdb.connect(str(_PROJECT_ROOT / "data" / "runtime" / "wwi.duckdb"), read_only=True)
    yield con
    con.close()


def _label_key(label: str) -> str:
    return {
        "Customer": "customer_id", "Order": "order_id", "Invoice": "invoice_id",
        "StockItem": "stock_item_id", "Supplier": "supplier_id",
        "BuyingGroup": "buying_group_id", "City": "city_id",
    }[label]


class TestKGToolQueries:
    def test_customer_orders_vs_sql(self, db: duckdb.DuckDBPyConnection) -> None:
        customer_id, _ = db.execute(
            "SELECT c.CustomerID, COUNT(*) AS c "
            "FROM Sales_Customers c JOIN Sales_Orders o ON c.CustomerID = o.CustomerID "
            "GROUP BY 1 ORDER BY c DESC, c.CustomerID LIMIT 1"
        ).fetchone()
        expected = {r[0] for r in db.execute(
            "SELECT OrderID FROM Sales_Orders WHERE CustomerID = ?", [customer_id]
        ).fetchall()}

        tool = KGTool()
        try:
            result = tool.find_customer_orders(customer_id)
        finally:
            tool.close()
        assert result["success"] is True
        graph_ids = {row["order"]["Order"]["order_id"] for row in result["data"]}
        assert graph_ids == expected

    def test_order_invoices_vs_sql(self, db: duckdb.DuckDBPyConnection) -> None:
        order_id, = db.execute(
            "SELECT i.OrderID FROM Sales_Invoices i WHERE i.OrderID IS NOT NULL LIMIT 1"
        ).fetchone()
        expected = {r[0] for r in db.execute(
            "SELECT InvoiceID FROM Sales_Invoices WHERE OrderID = ?", [order_id]
        ).fetchall()}

        tool = KGTool()
        try:
            result = tool.find_order_invoices(order_id)
        finally:
            tool.close()
        assert result["success"] is True
        graph_ids = {row["inv"]["Invoice"]["invoice_id"] for row in result["data"]}
        assert graph_ids == expected

    def test_item_suppliers_vs_sql(self, db: duckdb.DuckDBPyConnection) -> None:
        item_id, sup_id = db.execute(
            "SELECT StockItemID, SupplierID FROM Warehouse_StockItems WHERE SupplierID IS NOT NULL LIMIT 1"
        ).fetchone()
        tool = KGTool()
        try:
            result = tool.find_item_suppliers(item_id)
        finally:
            tool.close()
        assert result["success"] is True
        assert {row["sup"]["Supplier"]["supplier_id"] for row in result["data"]} == {sup_id}

    def test_count_orders_aggregation_matches_sql(self, db: duckdb.DuckDBPyConnection) -> None:
        customer_id, expected_count = db.execute(
            "SELECT c.CustomerID, COUNT(*) "
            "FROM Sales_Customers c JOIN Sales_Orders o ON c.CustomerID = o.CustomerID "
            "GROUP BY 1 ORDER BY 2 DESC LIMIT 1"
        ).fetchone()
        tool = KGTool()
        try:
            result = tool.count_orders_by_customer(customer_id)
        finally:
            tool.close()
        assert result["success"] is True
        assert result["data"][0]["order_count"] == expected_count

    def test_two_hop_path_non_empty(self, db: duckdb.DuckDBPyConnection) -> None:
        customer_id, = db.execute(
            "SELECT c.CustomerID FROM Sales_Customers c "
            "WHERE c.CustomerID IN (SELECT CustomerID FROM Sales_Orders) LIMIT 1"
        ).fetchone()
        tool = KGTool()
        try:
            result = tool.find_customer_order_item_paths(customer_id)
        finally:
            tool.close()
        assert result["success"] is True
        assert result["rows"] >= 1

    def test_relation_existence(self, db: duckdb.DuckDBPyConnection) -> None:
        customer_id, order_id = db.execute(
            "SELECT CustomerID, OrderID FROM Sales_Orders LIMIT 1"
        ).fetchone()
        tool = KGTool()
        try:
            yes = tool.query_relation_exists(
                from_label="Customer", from_key=_label_key("Customer"), from_value=customer_id,
                rel_type="PLACED", to_label="Order", to_key=_label_key("Order"), to_value=order_id,
            )
            no = tool.query_relation_exists(
                from_label="Customer", from_key=_label_key("Customer"), from_value=customer_id,
                rel_type="PLACED", to_label="Order", to_key=_label_key("Order"), to_value=999999,
            )
        finally:
            tool.close()
        assert yes["success"] is True and yes["data"][0]["exists"] is True
        assert no["success"] is True and no["data"][0]["exists"] is False

    def test_failure_payloads_are_clean(self) -> None:
        tool = KGTool()
        try:
            unknown = tool.run(template_id="does_not_exist", parameters={})
            missing = tool.run(template_id="customer_orders", parameters={})
        finally:
            tool.close()
        for payload in (unknown, missing):
            assert payload["success"] is False
            assert "bolt" not in str(payload)
            assert "password" not in str(payload).lower()
            assert "NEO4J" not in str(payload).upper() or payload["error"]["code"] in ("unknown_template", "missing_parameter")
