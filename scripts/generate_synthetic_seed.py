"""生成业务大表合成种子数据（Phase 1）。

官方 WideWorldImporters 的 Sales_Customers / Purchasing_Suppliers /
Warehouse_StockItems / Sales_Orders / Sales_Invoices 等大表在官方 T-SQL
部署脚本中由存储过程（随机生成器）动态生成，无法在 DuckDB 中直接执行。
本脚本生成 **合成** 业务数据补齐这些表，使 Text-to-SQL 全链路
（few-shot、e2e）可在真实表结构上跑通。

红线（docs/DATASET.md）：
- 全部数据为合成（synthetic），脚本头部与输出均显式标注；
- 不伪造"真实"业务指标——任何统计结论必须来自这些可复现的合成行；
- 列名严格对齐 DDL（scripts/import_wwi.py 生成的 database_schema.yaml），
  主键/外键与已入库维度表保持引用一致，保证 JOIN 语义有效。

用法（在项目根目录，先运行 scripts/import_wwi.py）::

    python scripts/generate_synthetic_seed.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "runtime" / "wwi.duckdb"


def _log(msg: str) -> None:
    print(f"[synthetic-seed] {msg}", flush=True)


def _fetch_dim(connection: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """读取已入库维度表，供合成数据引用（保证外键有效）。"""
    dim: dict[str, Any] = {}

    def _ids(table: str, col: str, limit: int | None = None) -> list[int]:
        q = f"SELECT {col} FROM {table} ORDER BY {col}"
        if limit:
            q += f" LIMIT {limit}"
        return [r[0] for r in connection.execute(q).fetchall()]

    dim["person_ids"] = _ids("Application_People", "PersonID", 50) or [1]
    dim["city_ids"] = _ids("Application_Cities", "CityID", 200) or [1]
    dim["customer_category_ids"] = _ids("Sales_CustomerCategories", "CustomerCategoryID") or [1]
    dim["buying_group_ids"] = _ids("Sales_BuyingGroups", "BuyingGroupID") or [1]
    dim["delivery_method_ids"] = _ids("Application_DeliveryMethods", "DeliveryMethodID") or [1]
    dim["color_ids"] = _ids("Warehouse_Colors", "ColorID") or [1]
    dim["package_type_ids"] = _ids("Warehouse_PackageTypes", "PackageTypeID") or [1]
    dim["supplier_category_ids"] = _ids("Purchasing_SupplierCategories", "SupplierCategoryID") or [1]

    def _max(table: str, col: str) -> int:
        return int(connection.execute(
            f"SELECT COALESCE(MAX({col}), 0) FROM {table}").fetchone()[0])

    dim["last_customer_id"] = _max("Sales_Customers", "CustomerID")
    dim["last_supplier_id"] = _max("Purchasing_Suppliers", "SupplierID")
    dim["last_stock_item_id"] = _max("Warehouse_StockItems", "StockItemID")
    dim["last_order_id"] = _max("Sales_Orders", "OrderID")
    dim["last_invoice_id"] = _max("Sales_Invoices", "InvoiceID")
    dim["last_invoice_line_id"] = _max("Sales_InvoiceLines", "InvoiceLineID")
    dim["last_stock_item_stock_group_id"] = _max("Warehouse_StockItemStockGroups", "StockItemStockGroupID")
    dim["stock_group_ids"] = _ids("Warehouse_StockGroups", "StockGroupID") or [1]
    return dim


# ---------------------------------------------------------------------------
# 合成数据生成器（确定性：纯函数 + 固定输入，可复现）
# ---------------------------------------------------------------------------

_COMPANY_SUFFIXES = ["Ltd", "Inc", "Co", "Group", "Partners", "LLC", "Corp"]
_COMPANY_FIRST = ["Acme", "Globex", "Initech", "Umbrella", "Stark", "Wayne",
                  "Hooli", "Vandelay", "Soylent", "Pied Piper", "Dunder", "Massive"]


def _pick(lst: list[Any], i: int) -> Any:
    return lst[i % len(lst)]


def _company_name(seed: int, idx: int) -> str:
    first = _COMPANY_FIRST[(seed + idx * 7) % len(_COMPANY_FIRST)]
    suffix = _COMPANY_SUFFIXES[(seed + idx * 13) % len(_COMPANY_SUFFIXES)]
    return f"{first} {suffix}"


def generate_customers(dim: dict[str, Any], n: int = 500) -> list[dict[str, Any]]:
    """Sales_Customers 列（严格对齐 DDL）。"""
    rows = []
    for i in range(1, n + 1):
        rows.append({
            "CustomerID": dim["last_customer_id"] + i,
            "CustomerName": _company_name(1, i),
            "BillToCustomerID": dim["last_customer_id"] + 1,
            "CustomerCategoryID": _pick(dim["customer_category_ids"], i),
            "BuyingGroupID": _pick(dim["buying_group_ids"], i),
            "PrimaryContactPersonID": _pick(dim["person_ids"], i),
            "AlternateContactPersonID": _pick(dim["person_ids"], i + 1),
            "DeliveryMethodID": _pick(dim["delivery_method_ids"], i),
            "DeliveryCityID": _pick(dim["city_ids"], i),
            "PostalCityID": _pick(dim["city_ids"], i + 3),
            "CreditLimit": round(10000 + i * 50, 2),
            "AccountOpenedDate": "2013-10-04",
            "StandardDiscountPercentage": round(i % 7 * 0.5, 1),
            "IsStatementSent": i % 2 == 0,
            "IsOnCreditHold": i % 25 == 0,
            "PaymentDays": 1 + i % 20,
            "PhoneNumber": f"555-{i % 10000:04d}",
            "FaxNumber": f"555-{i % 10000:04d}F",
            "DeliveryRun": 0,
            "RunPosition": 0,
            "WebsiteURL": f"https://example.com/cust/{i}",
            "DeliveryAddressLine1": f"{i} Synthetic Street",
            "DeliveryAddressLine2": None,
            "DeliveryPostalCode": f"SYN{i % 100:02d}",
            "DeliveryLocation": None,
            "PostalAddressLine1": f"{i} Synthetic Street",
            "PostalAddressLine2": None,
            "PostalPostalCode": f"SYN{i % 100:02d}",
            "LastEditedBy": _pick(dim["person_ids"], i + 2),
            "ValidFrom": "2020-01-01",
            "ValidTo": "9999-12-31",
        })
    return rows


def generate_suppliers(dim: dict[str, Any], n: int = 60) -> list[dict[str, Any]]:
    """Purchasing_Suppliers 列（严格对齐 DDL）。"""
    rows = []
    for i in range(1, n + 1):
        rows.append({
            "SupplierID": dim["last_supplier_id"] + i,
            "SupplierName": f"Synthetic Supplier {i}",
            "SupplierCategoryID": _pick(dim.get("supplier_category_ids", [1]), i),
            "PrimaryContactPersonID": _pick(dim["person_ids"], i),
            "AlternateContactPersonID": _pick(dim["person_ids"], i + 1),
            "DeliveryMethodID": _pick(dim["delivery_method_ids"], i),
            "DeliveryCityID": _pick(dim["city_ids"], i),
            "PostalCityID": _pick(dim["city_ids"], i + 2),
            "SupplierReference": f"SUP-{i:04d}",
            "BankAccountName": None,
            "BankAccountBranch": None,
            "BankAccountCode": None,
            "BankAccountNumber": None,
            "BankInternationalCode": None,
            "PaymentDays": 30,
            "InternalComments": "synthetic",
            "PhoneNumber": f"555-{i + 5000:04d}",
            "FaxNumber": f"555-{i + 5000:04d}F",
            "WebsiteURL": f"https://example.com/sup/{i}",
            "DeliveryAddressLine1": f"{i} Supplier Road",
            "DeliveryAddressLine2": None,
            "DeliveryPostalCode": f"SPS{i % 100:02d}",
            "DeliveryLocation": None,
            "PostalAddressLine1": f"{i} Supplier Road",
            "PostalAddressLine2": None,
            "PostalPostalCode": f"SPS{i % 100:02d}",
            "LastEditedBy": _pick(dim["person_ids"], i + 2),
            "ValidFrom": "2020-01-01",
            "ValidTo": "9999-12-31",
        })
    return rows


def generate_stock_items(dim: dict[str, Any], n: int = 120) -> list[dict[str, Any]]:
    """Warehouse_StockItems 列（严格对齐 DDL）。"""
    supplier_ids = list(range(dim["last_supplier_id"] + 1, dim["last_supplier_id"] + 61))
    rows = []
    for i in range(1, n + 1):
        rows.append({
            "StockItemID": dim["last_stock_item_id"] + i,
            "StockItemName": f"Synthetic Item {i}",
            "SupplierID": _pick(supplier_ids, i),
            "ColorID": _pick(dim["color_ids"], i),
            "UnitPackageID": _pick(dim["package_type_ids"], i),
            "OuterPackageID": _pick(dim["package_type_ids"], i + 1),
            "Brand": None,
            "Size": None,
            "LeadTimeDays": 7 + i % 20,
            "QuantityPerOuter": 10 + i % 40,
            "IsChillerStock": False,
            "Barcode": f"SYN-{i:08d}",
            "TaxRate": 0.0,
            "UnitPrice": round(1.5 + (i % 50) * 0.37, 2),
            "RecommendedRetailPrice": round(5.0 + (i % 50) * 0.5, 2),
            "TypicalWeightPerUnit": round(0.1 + (i % 10) * 0.2, 2),
            "MarketingComments": None,
            "InternalComments": "synthetic",
            "Photo": None,
            "CustomFields": None,
            "Tags": None,
            "SearchDetails": None,
            "LastEditedBy": _pick(dim["person_ids"], i),
            "ValidFrom": "2020-01-01",
            "ValidTo": "9999-12-31",
        })
    return rows


def generate_orders(dim: dict[str, Any], n: int = 800) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """销售链路：Sales_Orders + Sales_Invoices + Sales_InvoiceLines。

    金额 = 数量 × 单价（确定性可复现）；外键引用刚生成的客户与库存商品。
    """
    customer_ids = list(range(dim["last_customer_id"] + 1, dim["last_customer_id"] + 501))
    stock_items = dim.get("new_stock_items") or []
    stock_ids = [r["StockItemID"] for r in stock_items] or [dim["last_stock_item_id"] + 1]
    prices = {r["StockItemID"]: r["UnitPrice"] for r in stock_items} or {stock_ids[0]: 10.0}

    orders, invoices, invoice_lines = [], [], []
    for i in range(1, n + 1):
        cust = _pick(customer_ids, i)
        order_id = dim["last_order_id"] + i
        order_date = date(2024, 1, 1) + timedelta(days=(i * 5) % 730)
        orders.append({
            "OrderID": order_id,
            "CustomerID": cust,
            "SalespersonPersonID": _pick(dim["person_ids"], i),
            "PickedByPersonID": None,
            "ContactPersonID": _pick(dim["person_ids"], i + 1),
            "BackorderOrderID": 0,
            "OrderDate": str(order_date),
            "ExpectedDeliveryDate": str(order_date + timedelta(days=5)),
            "CustomerPurchaseOrderNumber": f"CUSTPO-{i:06d}",
            "IsUndersupplyBackordered": False,
            "Comments": "synthetic order",
            "DeliveryInstructions": "Front desk",
            "InternalComments": None,
            "PickingCompletedWhen": None,
            "LastEditedBy": _pick(dim["person_ids"], i + 2),
            "LastEditedWhen": str(order_date),
        })

        inv_id = dim["last_invoice_id"] + i
        inv_date = str(order_date + timedelta(days=3))
        qty = 1 + i % 20
        sid = _pick(stock_ids, i)
        unit_price = prices.get(sid, 10.0)
        tax = round(qty * unit_price * 0.15, 2)
        subtotal = round(qty * unit_price, 2)
        invoice_lines.append({
            "InvoiceLineID": dim["last_invoice_line_id"] + i,
            "InvoiceID": inv_id,
            "StockItemID": sid,
            "Description": f"Synthetic item line {i}",
            "PackageTypeID": _pick(dim["package_type_ids"], i),
            "Quantity": qty,
            "UnitPrice": unit_price,
            "TaxRate": 0.15,
            "TaxAmount": tax,
            "LineProfit": round(qty * unit_price * 0.3, 2),
            "ExtendedPrice": subtotal,
            "LastEditedBy": _pick(dim["person_ids"], i + 3),
            "LastEditedWhen": inv_date,
        })
        invoices.append({
            "InvoiceID": inv_id,
            "CustomerID": cust,
            "BillToCustomerID": cust,
            "OrderID": order_id,
            "DeliveryMethodID": _pick(dim["delivery_method_ids"], i),
            "ContactPersonID": _pick(dim["person_ids"], i),
            "AccountsPersonID": _pick(dim["person_ids"], i + 2),
            "SalespersonPersonID": _pick(dim["person_ids"], i + 3),
            "PackedByPersonID": _pick(dim["person_ids"], i + 4),
            "InvoiceDate": inv_date,
            "CustomerPurchaseOrderNumber": f"CUSTPO-{i:06d}",
            "IsCreditNote": False,
            "CreditNoteReason": None,
            "Comments": "synthetic",
            "DeliveryInstructions": "Front desk",
            "InternalComments": None,
            "TotalDryItems": 0,
            "TotalChillerItems": 0,
            "DeliveryRun": 0,
            "RunPosition": 0,
            "ReturnedDeliveryData": None,
            "ConfirmedDeliveryTime": None,
            "ConfirmedReceivedBy": None,
            "LastEditedBy": _pick(dim["person_ids"], i + 5),
            "LastEditedWhen": inv_date,
        })
    return orders, invoices, invoice_lines


def _stock_item_group_mapping(dim: dict[str, Any]) -> list[dict[str, Any]]:
    """Warehouse_StockItemStockGroups：每个商品映射到一个商品类别。"""
    stock_items = dim.get("new_stock_items") or []
    group_ids = dim.get("stock_group_ids") or [1]
    next_id = dim["last_stock_item_stock_group_id"]
    rows = []
    for i, item in enumerate(stock_items):
        next_id += 1
        rows.append({
            "StockItemStockGroupID": next_id,
            "StockItemID": item["StockItemID"],
            "StockGroupID": group_ids[i % len(group_ids)],
            "LastEditedBy": _pick(dim["person_ids"], i),
            "LastEditedWhen": "2024-01-01",
        })
    return rows


def _insert_dicts(connection: duckdb.DuckDBPyConnection, table: str, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    cols = list(rows[0].keys())
    col_list = ", ".join(f'"{c}"' for c in cols)
    ph = ", ".join("?" for _ in cols)
    sql = f"INSERT INTO {table} ({col_list}) VALUES ({ph})"
    param_rows = [tuple(r[c] for c in cols) for r in rows]
    connection.executemany(sql, param_rows)
    return len(param_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成业务大表合成种子数据")
    parser.add_argument("--customers", type=int, default=500)
    parser.add_argument("--suppliers", type=int, default=60)
    parser.add_argument("--stock-items", type=int, default=120)
    parser.add_argument("--orders", type=int, default=800)
    args = parser.parse_args()

    if not DB_PATH.exists():
        _log(f"运行时数据库不存在: {DB_PATH} - 先运行 scripts/import_wwi.py")
        return 1

    connection = duckdb.connect(str(DB_PATH))
    total = 0
    try:
        # 幂等：先清空目标表（官方 T-SQL 种子生成器无法在 DuckDB 执行，这些表由
        # 本脚本合成补齐；清空保证可重复执行、主键/行数不漂移）
        _log("清空目标表（合成数据可重复生成）")
        for table in [
            "Sales_Customers", "Purchasing_Suppliers", "Warehouse_StockItems",
            "Sales_Orders", "Sales_Invoices", "Sales_InvoiceLines",
            "Warehouse_StockItemStockGroups",
        ]:
            connection.execute(f"DELETE FROM {table}")

        dim = _fetch_dim(connection)
        _log(f"维度表就绪：person={len(dim['person_ids'])} city={len(dim['city_ids'])}")

        customers = generate_customers(dim, args.customers)
        suppliers = generate_suppliers(dim, args.suppliers)
        stock_items = generate_stock_items(dim, args.stock_items)
        dim["new_stock_items"] = stock_items
        orders, invoices, invoice_lines = generate_orders(dim, args.orders)

        for table, rows in [
            ("Sales_Customers", customers),
            ("Purchasing_Suppliers", suppliers),
            ("Warehouse_StockItems", stock_items),
            ("Sales_Orders", orders),
            ("Sales_Invoices", invoices),
            ("Sales_InvoiceLines", invoice_lines),
        ]:
            n = _insert_dicts(connection, table, rows)
            total += n
            _log(f"  {table:22s} {n:6d} 行（synthetic）")

        # 商品->类别映射（供"销售额按商品类别"类 few-shot / 业务问题可查）
        n = _insert_dicts(connection, "Warehouse_StockItemStockGroups", _stock_item_group_mapping(dim))
        total += n
        _log(f"  {'Warehouse_StockItemStockGroups':22s} {n:6d} 行（synthetic）")

        connection.execute("CHECKPOINT")
        _log(f"完成：{total} 行合成业务数据写入 {DB_PATH}")
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
