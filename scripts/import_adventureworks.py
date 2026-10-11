"""Import the official Microsoft AdventureWorks OLTP dataset into the runtime DuckDB.

The existing database was seeded with synthetic WideWorldImporters-style data, which
left the core sales fact tables empty (``Sales_OrderLines`` had 0 rows). This script
replaces the sales / customer / product / people tables with **real** AdventureWorks
records while keeping the WideWorldImporters table and column names, so the
auto-generated data dictionary, tool schemas and SQL few-shots keep working.

Source: AdventureWorks-oltp-install-script.zip from
https://github.com/microsoft/sql-server-samples/releases
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path

import duckdb
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
AW_DIR = ROOT / "data" / "downloads" / "aw-oltp"
DB_PATH = ROOT / "data" / "runtime" / "wwi.duckdb"
BACKUP_PATH = ROOT / "data" / "runtime" / "wwi_before_aw.duckdb"
SCHEMAS_DIR = ROOT / "data" / "schemas"

KEEP_TABLES = ("finance_expenses",)

CSV_SPECS: dict[str, tuple[list[str], list[str]]] = {
    "SalesOrderHeader": (
        ["SalesOrderID", "RevisionNumber", "OrderDate", "DueDate", "ShipDate", "Status",
         "OnlineOrderFlag", "SalesOrderNumber", "PurchaseOrderNumber", "AccountNumber",
         "CustomerID", "SalesPersonID", "TerritoryID", "BillToAddressID", "ShipToAddressID",
         "ShipMethodID", "CreditCardID", "CreditCardApprovalCode", "CurrencyRateID",
         "SubTotal", "TaxAmt", "Freight", "TotalDue", "Comment", "rowguid", "ModifiedDate"],
        ["INTEGER", "UTINYINT", "TIMESTAMP", "TIMESTAMP", "TIMESTAMP", "UTINYINT",
         "BOOLEAN", "VARCHAR", "VARCHAR", "VARCHAR",
         "INTEGER", "INTEGER", "INTEGER", "INTEGER", "INTEGER",
         "INTEGER", "INTEGER", "VARCHAR", "INTEGER",
         "DOUBLE", "DOUBLE", "DOUBLE", "DOUBLE", "VARCHAR", "VARCHAR", "TIMESTAMP"],
    ),
    "SalesOrderDetail": (
        ["SalesOrderID", "SalesOrderDetailID", "CarrierTrackingNumber", "OrderQty",
         "ProductID", "SpecialOfferID", "UnitPrice", "UnitPriceDiscount", "LineTotal",
         "rowguid", "ModifiedDate"],
        ["INTEGER", "INTEGER", "VARCHAR", "SMALLINT",
         "INTEGER", "INTEGER", "DOUBLE", "DOUBLE", "DOUBLE", "VARCHAR", "TIMESTAMP"],
    ),
    "Customer": (
        ["CustomerID", "PersonID", "StoreID", "TerritoryID", "AccountNumber",
         "rowguid", "ModifiedDate"],
        ["INTEGER", "INTEGER", "INTEGER", "INTEGER", "VARCHAR", "VARCHAR", "TIMESTAMP"],
    ),
    "Product": (
        ["ProductID", "Name", "ProductNumber", "MakeFlag", "FinishedGoodsFlag", "Color",
         "SafetyStockLevel", "ReorderPoint", "StandardCost", "ListPrice", "Size",
         "SizeUnitMeasureCode", "WeightUnitMeasureCode", "Weight", "DaysToManufacture",
         "ProductLine", "Class", "Style", "ProductSubcategoryID", "ProductModelID",
         "SellStartDate", "SellEndDate", "DiscontinuedDate", "rowguid", "ModifiedDate"],
        ["INTEGER", "VARCHAR", "VARCHAR", "BOOLEAN", "BOOLEAN", "VARCHAR",
         "SMALLINT", "SMALLINT", "DOUBLE", "DOUBLE", "VARCHAR",
         "VARCHAR", "VARCHAR", "DOUBLE", "INTEGER",
         "VARCHAR", "VARCHAR", "VARCHAR", "INTEGER", "INTEGER",
         "TIMESTAMP", "TIMESTAMP", "TIMESTAMP", "VARCHAR", "TIMESTAMP"],
    ),
    "ProductCategory": (
        ["ProductCategoryID", "Name", "rowguid", "ModifiedDate"],
        ["INTEGER", "VARCHAR", "VARCHAR", "TIMESTAMP"],
    ),
    "ProductSubcategory": (
        ["ProductSubcategoryID", "ProductCategoryID", "Name", "rowguid", "ModifiedDate"],
        ["INTEGER", "INTEGER", "VARCHAR", "VARCHAR", "TIMESTAMP"],
    ),
    # Purchasing.Vendor / Purchasing.ProductVendor.  Only BusinessEntityID and
    # ProductID are consumed downstream; the remaining columns are typed loosely
    # (strict_mode=false turns a mismatch into NULL) so a differing dump order
    # cannot abort the import.
    "Vendor": (
        ["BusinessEntityID", "AccountNumber", "Name", "CreditRating",
         "PreferredVendorFlag", "ActiveFlag", "PurchasingWebServiceURL", "ModifiedDate"],
        ["INTEGER", "VARCHAR", "VARCHAR", "SMALLINT", "BOOLEAN", "BOOLEAN",
         "VARCHAR", "TIMESTAMP"],
    ),
    "ProductVendor": (
        ["ProductID", "BusinessEntityID", "AverageLeadTime", "LastReceiptCost",
         "UnitPrice", "LastReceiptDate", "OrderQty", "OnOrderQty", "StandardQty",
         "UnitMeasureCode", "ModifiedDate"],
        ["INTEGER", "INTEGER", "INTEGER", "DOUBLE", "DOUBLE", "TIMESTAMP",
         "SMALLINT", "SMALLINT", "SMALLINT", "VARCHAR", "TIMESTAMP"],
    ),
    "Address": (
        ["AddressID", "AddressLine1", "AddressLine2", "City", "StateProvinceID",
         "PostalCode", "SpatialLocation", "rowguid", "ModifiedDate"],
        ["INTEGER", "VARCHAR", "VARCHAR", "VARCHAR", "INTEGER", "VARCHAR",
         "VARCHAR", "VARCHAR", "TIMESTAMP"],
    ),
    "StateProvince": (
        ["StateProvinceID", "StateProvinceCode", "CountryRegionCode",
         "IsOnlyStateProvinceFlag", "StateProvinceName", "TerritoryID",
         "rowguid", "ModifiedDate"],
        ["INTEGER", "VARCHAR", "VARCHAR", "BOOLEAN", "VARCHAR", "SMALLINT",
         "VARCHAR", "TIMESTAMP"],
    ),
    "SalesPerson": (
        ["BusinessEntityID", "TerritoryID", "SalesQuota", "Bonus", "CommissionPct",
         "SalesYTD", "SalesLastYear", "rowguid", "ModifiedDate"],
        ["INTEGER", "INTEGER", "DOUBLE", "DOUBLE", "DOUBLE",
         "DOUBLE", "DOUBLE", "VARCHAR", "TIMESTAMP"],
    ),
}

PERSON_COLUMNS = [
    "BusinessEntityID", "PersonType", "NameStyle", "Title", "FirstName", "MiddleName",
    "LastName", "Suffix", "EmailPromotion", "AdditionalContactInfo", "Demographics",
    "rowguid", "ModifiedDate",
]

FAR_FUTURE = "TIMESTAMP '9999-12-31 23:59:59.997'"

# AdventureWorks' address dimension is disjoint from the WideWorldImporters
# one still sitting in Application_Cities / Application_StateProvinces.  Rather
# than replace those tables (build_enterprise_tasks anchors on low ids such as
# ``CityID = 1``), the AW rows are APPENDED above the existing maximum so every
# pre-existing id keeps its meaning.
AW_STATE_PROVINCE_ID_BASE = 1000
AW_CITY_ID_BASE = 40000


def _log(message: str) -> None:
    print(message, flush=True)


def load_tab_csv(con: duckdb.DuckDBPyConnection, name: str) -> int:
    cols, types = CSV_SPECS[name]
    spec = ", ".join(f'"{c}": "{t}"' for c, t in zip(cols, types, strict=True))
    path = str(AW_DIR / f"{name}.csv").replace("\\", "/")
    con.execute(
        f'CREATE OR REPLACE TABLE "_aw_{name}" AS '
        f"SELECT * FROM read_csv('{path}', delim='\t', header=false, "
        f"nullstr='', strict_mode=false, columns={{{spec}}})"
    )
    return int(con.execute(f'SELECT COUNT(*) FROM "_aw_{name}"').fetchone()[0])


def load_person(con: duckdb.DuckDBPyConnection) -> int:
    """Person.csv uses '+|' as the delimiter and a trailing '&|' row marker."""
    rows: list[tuple] = []
    with (AW_DIR / "Person.csv").open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if line.endswith("&|"):
                line = line[:-2]
            if not line:
                continue
            parts = line.split("+|")
            if len(parts) != len(PERSON_COLUMNS):
                continue
            rows.append((
                int(parts[0]), parts[1], parts[2] == "1", parts[3] or None,
                parts[4], parts[5] or None, parts[6], parts[7] or None,
                int(parts[8]), parts[9] or None, parts[10] or None, parts[11],
                dt.datetime.fromisoformat(parts[12]),
            ))
    con.execute('DROP TABLE IF EXISTS "_aw_person"')
    con.execute(
        'CREATE TABLE "_aw_person" ('
        '"BusinessEntityID" INTEGER, "PersonType" VARCHAR, "NameStyle" BOOLEAN, '
        '"Title" VARCHAR, "FirstName" VARCHAR, "MiddleName" VARCHAR, '
        '"LastName" VARCHAR, "Suffix" VARCHAR, "EmailPromotion" INTEGER, '
        '"AdditionalContactInfo" VARCHAR, "Demographics" VARCHAR, '
        '"rowguid" VARCHAR, "ModifiedDate" TIMESTAMP)'
    )
    con.executemany('INSERT INTO "_aw_person" VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', rows)
    return len(rows)


BEA_COLUMNS = ["BusinessEntityID", "AddressID", "AddressTypeID", "rowguid", "ModifiedDate"]


def load_business_entity_address(con: duckdb.DuckDBPyConnection) -> int:
    """BusinessEntityAddress.csv uses the same '+|' / trailing '&|' shape as Person."""
    rows: list[tuple] = []
    with (AW_DIR / "BusinessEntityAddress.csv").open(
        "r", encoding="utf-8-sig", errors="replace"
    ) as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if line.endswith("&|"):
                line = line[:-2]
            if not line:
                continue
            parts = line.split("+|")
            if len(parts) != len(BEA_COLUMNS):
                continue
            rows.append((
                int(parts[0]), int(parts[1]), int(parts[2]), parts[3],
                dt.datetime.fromisoformat(parts[4]),
            ))
    con.execute('DROP TABLE IF EXISTS "_aw_BusinessEntityAddress"')
    con.execute(
        'CREATE TABLE "_aw_BusinessEntityAddress" ('
        '"BusinessEntityID" INTEGER, "AddressID" INTEGER, "AddressTypeID" INTEGER, '
        '"rowguid" VARCHAR, "ModifiedDate" TIMESTAMP)'
    )
    con.executemany(
        'INSERT INTO "_aw_BusinessEntityAddress" VALUES (?,?,?,?,?)', rows
    )
    return len(rows)


def extend_address_dimension(con: duckdb.DuckDBPyConnection) -> None:
    """Append AdventureWorks states/cities above the existing id ranges.

    WideWorldImporters' city catalogue stays untouched (its low ids are
    referenced by several benchmark anchors), and AW rows are added in a
    disjoint range.  ``_aw_city_map`` then translates an AW ``AddressID`` into
    the ``CityID`` we just allocated.
    """
    con.execute(f"""
        INSERT INTO Application_StateProvinces (
            StateProvinceID, StateProvinceCode, CountryID, StateProvinceName,
            SalesTerritory, LatestRecordedPopulation, LastEditedBy,
            ValidFrom, ValidTo
        )
        SELECT sp.StateProvinceID::INTEGER + {AW_STATE_PROVINCE_ID_BASE},
            TRIM(sp.StateProvinceCode),
            239,
            sp.StateProvinceName,
            sp.TerritoryID::VARCHAR,
            NULL,
            1,
            sp.ModifiedDate::TIMESTAMP,
            {FAR_FUTURE}
        FROM "_aw_StateProvince" sp
        WHERE sp.StateProvinceID::INTEGER + {AW_STATE_PROVINCE_ID_BASE}
              NOT IN (SELECT StateProvinceID FROM Application_StateProvinces)
        ORDER BY sp.StateProvinceID
    """)

    # One allocated CityID per distinct AW city, carrying the payload needed
    # for the Application_Cities insert so no second join has to rediscover it.
    # The ids come from a fixed base rather than MAX()+1 so a re-run reallocates
    # the same ids and the NOT IN below de-duplicates instead of duplicating.
    con.execute(f"""
        CREATE OR REPLACE TABLE "_aw_city_map" AS
        SELECT
            {AW_CITY_ID_BASE} + ROW_NUMBER() OVER (ORDER BY d.City, d.StateProvinceID) - 1
                AS CityID,
            d.City AS CityName,
            d.StateProvinceID::INTEGER + {AW_STATE_PROVINCE_ID_BASE} AS StateProvinceID
        FROM (
            SELECT DISTINCT City, StateProvinceID
            FROM "_aw_Address"
            WHERE City IS NOT NULL AND City <> ''
        ) d
    """)

    con.execute(f"""
        INSERT INTO Application_Cities (
            CityID, CityName, StateProvinceID, LatestRecordedPopulation,
            LastEditedBy, ValidFrom, ValidTo
        )
        SELECT m.CityID, m.CityName, m.StateProvinceID,
               NULL, 1, TIMESTAMP '2019-01-01', {FAR_FUTURE}
        FROM "_aw_city_map" m
        WHERE m.CityID NOT IN (SELECT CityID FROM Application_Cities)
        ORDER BY m.CityID
    """)

    # Translate AW AddressID -> allocated CityID for the supplier edges below.
    con.execute(f"""
        CREATE OR REPLACE TABLE "_aw_address_city" AS
        SELECT a.AddressID::INTEGER AS AddressID, m.CityID
        FROM "_aw_Address" a
        JOIN "_aw_city_map" m
          ON m.CityName = a.City
         AND m.StateProvinceID = a.StateProvinceID::INTEGER
                              + {AW_STATE_PROVINCE_ID_BASE}
    """)


def rebuild_tables(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Recreate the WideWorldImporters-named tables from real AdventureWorks rows."""
    done: list[str] = []

    con.execute("""
        CREATE OR REPLACE TABLE Sales_Orders AS
        SELECT h.SalesOrderID::INTEGER AS OrderID, h.CustomerID::INTEGER AS CustomerID,
            h.SalesPersonID::INTEGER AS SalespersonPersonID, NULL::INTEGER AS PickedByPersonID,
            h.CustomerID::INTEGER AS ContactPersonID, NULL::INTEGER AS BackorderOrderID,
            h.OrderDate::DATE AS OrderDate, h.DueDate::DATE AS ExpectedDeliveryDate,
            h.PurchaseOrderNumber AS CustomerPurchaseOrderNumber,
            FALSE AS IsUndersupplyBackordered, NULL::VARCHAR AS Comments,
            NULL::VARCHAR AS DeliveryInstructions, NULL::VARCHAR AS InternalComments,
            h.ShipDate::TIMESTAMP AS PickingCompletedWhen, 1 AS LastEditedBy,
            h.ModifiedDate::TIMESTAMP AS LastEditedWhen
        FROM "_aw_SalesOrderHeader" h
    """)
    done.append("Sales_Orders")

    con.execute("""
        CREATE OR REPLACE TABLE Sales_OrderLines AS
        SELECT d.SalesOrderDetailID::INTEGER AS OrderLineID, d.SalesOrderID::INTEGER AS OrderID,
            d.ProductID::INTEGER AS StockItemID, p.Name AS Description, 1 AS PackageTypeID,
            d.OrderQty::INTEGER AS Quantity, d.UnitPrice::DECIMAL(18,2) AS UnitPrice,
            0.000::DECIMAL(18,3) AS TaxRate, d.OrderQty::INTEGER AS PickedQuantity,
            h.ShipDate::TIMESTAMP AS PickingCompletedWhen, 1 AS LastEditedBy,
            d.ModifiedDate::TIMESTAMP AS LastEditedWhen
        FROM "_aw_SalesOrderDetail" d
        JOIN "_aw_Product" p ON p.ProductID = d.ProductID
        JOIN "_aw_SalesOrderHeader" h ON h.SalesOrderID = d.SalesOrderID
    """)
    done.append("Sales_OrderLines")

    con.execute("""
        CREATE OR REPLACE TABLE Sales_Invoices AS
        SELECT h.SalesOrderID::INTEGER AS InvoiceID, h.CustomerID::INTEGER AS CustomerID,
            h.CustomerID::INTEGER AS BillToCustomerID, h.SalesOrderID::INTEGER AS OrderID,
            h.ShipMethodID::INTEGER AS DeliveryMethodID, NULL::INTEGER AS ContactPersonID,
            NULL::INTEGER AS AccountsPersonID, h.SalesPersonID::INTEGER AS SalespersonPersonID,
            NULL::INTEGER AS PackedByPersonID,
            COALESCE(h.ShipDate, h.OrderDate)::DATE AS InvoiceDate,
            h.PurchaseOrderNumber AS CustomerPurchaseOrderNumber,
            FALSE AS IsCreditNote, NULL::VARCHAR AS CreditNoteReason,
            NULL::VARCHAR AS Comments, NULL::VARCHAR AS DeliveryInstructions,
            NULL::VARCHAR AS InternalComments, 0 AS TotalDryItems, 0 AS TotalChillerItems,
            NULL::VARCHAR AS DeliveryRun, NULL::VARCHAR AS RunPosition,
            NULL::VARCHAR AS ReturnedDeliveryData, NULL::VARCHAR AS ConfirmedDeliveryTime,
            NULL::VARCHAR AS ConfirmedReceivedBy, 1 AS LastEditedBy,
            h.ModifiedDate::TIMESTAMP AS LastEditedWhen
        FROM "_aw_SalesOrderHeader" h
    """)
    done.append("Sales_Invoices")

    con.execute("""
        CREATE OR REPLACE TABLE Sales_InvoiceLines AS
        SELECT d.SalesOrderDetailID::INTEGER AS InvoiceLineID,
            d.SalesOrderID::INTEGER AS InvoiceID, d.ProductID::INTEGER AS StockItemID,
            p.Name AS Description, 1 AS PackageTypeID, d.OrderQty::INTEGER AS Quantity,
            d.UnitPrice::DECIMAL(18,2) AS UnitPrice, 0.000::DECIMAL(18,3) AS TaxRate,
            0.00::DECIMAL(18,2) AS TaxAmount,
            (d.LineTotal - p.StandardCost * d.OrderQty)::DECIMAL(18,2) AS LineProfit,
            d.LineTotal::DECIMAL(18,2) AS ExtendedPrice, 1 AS LastEditedBy,
            d.ModifiedDate::TIMESTAMP AS LastEditedWhen
        FROM "_aw_SalesOrderDetail" d
        JOIN "_aw_Product" p ON p.ProductID = d.ProductID
    """)
    done.append("Sales_InvoiceLines")

    con.execute("""
        CREATE OR REPLACE TABLE Sales_CustomerTransactions AS
        SELECT ROW_NUMBER() OVER (ORDER BY h.SalesOrderID)::INTEGER AS CustomerTransactionID,
            h.CustomerID::INTEGER AS CustomerID, 1 AS TransactionTypeID,
            h.SalesOrderID::INTEGER AS InvoiceID, 1 AS PaymentMethodID,
            COALESCE(h.ShipDate, h.OrderDate)::DATE AS TransactionDate,
            h.SubTotal::DECIMAL(18,2) AS AmountExcludingTax,
            h.TaxAmt::DECIMAL(18,2) AS TaxAmount,
            h.TotalDue::DECIMAL(18,2) AS TransactionAmount,
            0.00::DECIMAL(18,2) AS OutstandingBalance,
            COALESCE(h.ShipDate, h.OrderDate)::DATE AS FinalizationDate,
            'Y' AS IsFinalized, 1 AS LastEditedBy,
            h.ModifiedDate::TIMESTAMP AS LastEditedWhen
        FROM "_aw_SalesOrderHeader" h
    """)
    done.append("Sales_CustomerTransactions")

    con.execute(f"""
        CREATE OR REPLACE TABLE Sales_Customers AS
        SELECT c.CustomerID::INTEGER AS CustomerID,
            COALESCE(NULLIF(TRIM(CONCAT_WS(' ', p.FirstName, p.MiddleName, p.LastName)), ''),
                     'Store ' || c.CustomerID::VARCHAR) AS CustomerName,
            c.CustomerID::INTEGER AS BillToCustomerID,
            CASE WHEN c.StoreID IS NOT NULL THEN 2 ELSE 1 END AS CustomerCategoryID,
            NULL::INTEGER AS BuyingGroupID, c.PersonID::INTEGER AS PrimaryContactPersonID,
            NULL::INTEGER AS AlternateContactPersonID, 1 AS DeliveryMethodID,
            NULL::INTEGER AS DeliveryCityID, NULL::INTEGER AS PostalCityID,
            NULL::DECIMAL(18,2) AS CreditLimit, NULL::DATE AS AccountOpenedDate,
            0.000::DECIMAL(18,3) AS StandardDiscountPercentage,
            FALSE AS IsStatementSent, FALSE AS IsOnCreditHold, 30 AS PaymentDays,
            NULL::VARCHAR AS PhoneNumber, NULL::VARCHAR AS FaxNumber,
            NULL::VARCHAR AS DeliveryRun, NULL::VARCHAR AS RunPosition,
            NULL::VARCHAR AS WebsiteURL, NULL::VARCHAR AS DeliveryAddressLine1,
            NULL::VARCHAR AS DeliveryAddressLine2, NULL::VARCHAR AS DeliveryPostalCode,
            NULL::VARCHAR AS DeliveryLocation, NULL::VARCHAR AS PostalAddressLine1,
            NULL::VARCHAR AS PostalAddressLine2, NULL::VARCHAR AS PostalPostalCode,
            1 AS LastEditedBy, c.ModifiedDate::TIMESTAMP AS ValidFrom,
            {FAR_FUTURE} AS ValidTo
        FROM "_aw_Customer" c
        LEFT JOIN "_aw_person" p ON p.BusinessEntityID = c.PersonID
    """)
    done.append("Sales_Customers")

    con.execute(f"""
        CREATE OR REPLACE TABLE Warehouse_StockItems AS
        SELECT p.ProductID::INTEGER AS StockItemID, p.Name AS StockItemName,
            -- AdventureWorks carries no supplier FK on Production.Product; the
            -- link lives in Purchasing.ProductVendor.  A product can have several
            -- vendors, so take the lowest BusinessEntityID deterministically.
            v.BusinessEntityID::INTEGER AS SupplierID,
            NULL::INTEGER AS ColorID,
            1 AS UnitPackageID, 1 AS OuterPackageID, p.Color AS Brand, p.Size AS Size,
            0 AS LeadTimeDays, 1 AS QuantityPerOuter, FALSE AS IsChillerStock,
            NULL::VARCHAR AS Barcode, 0.000::DECIMAL(18,3) AS TaxRate,
            p.ListPrice::DECIMAL(18,2) AS UnitPrice,
            p.ListPrice::DECIMAL(18,2) AS RecommendedRetailPrice,
            p.Weight::DECIMAL(18,3) AS TypicalWeightPerUnit,
            NULL::VARCHAR AS MarketingComments, NULL::VARCHAR AS InternalComments,
            NULL::BLOB AS Photo,
            json_object('Color', p.Color, 'Class', p.Class, 'Style', p.Style,
                        'ProductLine', p.ProductLine, 'StandardCost', p.StandardCost,
                        'SafetyStockLevel', p.SafetyStockLevel, 'ReorderPoint', p.ReorderPoint,
                        'ProductNumber', p.ProductNumber) AS CustomFields,
            p.Name AS SearchDetails, 1 AS LastEditedBy,
            p.SellStartDate::TIMESTAMP AS ValidFrom,
            COALESCE(p.SellEndDate::TIMESTAMP, {FAR_FUTURE}) AS ValidTo
        FROM "_aw_Product" p
        LEFT JOIN (
            SELECT pv.ProductID, pv.BusinessEntityID
            FROM (
                SELECT ProductID, BusinessEntityID,
                       ROW_NUMBER() OVER (PARTITION BY ProductID
                                          ORDER BY BusinessEntityID) AS rn
                FROM "_aw_ProductVendor"
            ) pv
            WHERE pv.rn = 1
        ) v ON v.ProductID = p.ProductID
    """)
    done.append("Warehouse_StockItems")

    con.execute("""
        CREATE OR REPLACE TABLE Warehouse_StockGroups AS
        SELECT ProductCategoryID::INTEGER AS StockGroupID, Name AS StockGroupName,
            1 AS LastEditedBy, TIMESTAMP '2019-01-01' AS LastEditedWhen
        FROM "_aw_ProductCategory"
    """)
    done.append("Warehouse_StockGroups")

    con.execute("""
        CREATE OR REPLACE TABLE Warehouse_StockItemStockGroups AS
        SELECT ROW_NUMBER() OVER (ORDER BY p.ProductID)::INTEGER AS StockItemStockGroupID,
            p.ProductID::INTEGER AS StockItemID,
            s.ProductCategoryID::INTEGER AS StockGroupID,
            1 AS LastEditedBy, TIMESTAMP '2019-01-01' AS LastEditedWhen
        FROM "_aw_Product" p
        JOIN "_aw_ProductSubcategory" s ON s.ProductSubcategoryID = p.ProductSubcategoryID
        WHERE p.ProductSubcategoryID IS NOT NULL
    """)
    done.append("Warehouse_StockItemStockGroups")

    # The stock Supplied_by graph must point at real vendors.  The placeholder
    # "Synthetic Supplier N" rows inherited from the WideWorldImporters sample
    # are replaced by AdventureWorks' own Purchasing.Vendor rows, keyed on
    # BusinessEntityID so they line up with Warehouse_StockItems.SupplierID.
    con.execute(f"""
        CREATE OR REPLACE TABLE Purchasing_Suppliers AS
        SELECT v.BusinessEntityID::INTEGER AS SupplierID,
            v.Name AS SupplierName,
            1 AS SupplierCategoryID,
            NULL::INTEGER AS PrimaryContactPersonID,
            NULL::INTEGER AS AlternateContactPersonID,
            1 AS DeliveryMethodID,
            -- Vendor -> BusinessEntityAddress -> Address -> allocated CityID.
            -- This keeps SUPPLIER_IN_CITY populated with real geography instead
            -- of the arbitrary cities the old synthetic suppliers pointed at.
            ac.CityID::INTEGER AS DeliveryCityID,
            NULL::INTEGER AS PostalCityID,
            v.AccountNumber AS SupplierReference,
            NULL::VARCHAR AS BankAccountName, NULL::VARCHAR AS BankAccountBranch,
            NULL::VARCHAR AS BankAccountCode, NULL::VARCHAR AS BankAccountNumber,
            NULL::VARCHAR AS BankInternationalCode,
            30 AS PaymentDays,
            NULL::VARCHAR AS InternalComments,
            NULL::VARCHAR AS PhoneNumber, NULL::VARCHAR AS FaxNumber,
            v.PurchasingWebServiceURL AS WebsiteURL,
            NULL::VARCHAR AS DeliveryAddressLine1, NULL::VARCHAR AS DeliveryAddressLine2,
            NULL::VARCHAR AS DeliveryPostalCode, NULL::VARCHAR AS DeliveryLocation,
            NULL::VARCHAR AS PostalAddressLine1, NULL::VARCHAR AS PostalAddressLine2,
            NULL::VARCHAR AS PostalPostalCode,
            1 AS LastEditedBy, v.ModifiedDate::TIMESTAMP AS ValidFrom,
            {FAR_FUTURE} AS ValidTo
        FROM "_aw_Vendor" v
        LEFT JOIN (
            SELECT bea.BusinessEntityID, MIN(ac.CityID) AS CityID
            FROM "_aw_BusinessEntityAddress" bea
            JOIN "_aw_address_city" ac ON ac.AddressID = bea.AddressID
            GROUP BY bea.BusinessEntityID
        ) ac ON ac.BusinessEntityID = v.BusinessEntityID
        ORDER BY v.BusinessEntityID
    """)
    done.append("Purchasing_Suppliers")

    con.execute(f"""
        CREATE OR REPLACE TABLE Application_People AS
        SELECT p.BusinessEntityID::INTEGER AS PersonID,
            TRIM(CONCAT_WS(' ', p.FirstName, p.MiddleName, p.LastName)) AS FullName,
            p.FirstName AS PreferredName,
            UPPER(TRIM(CONCAT_WS(' ', p.FirstName, p.MiddleName, p.LastName))) AS SearchName,
            TRUE AS IsPermittedToLogon, NULL::VARCHAR AS LogonName,
            FALSE AS IsExternalLogonProvider, NULL::BLOB AS HashedPassword,
            FALSE AS IsSystemUser, (p.PersonType IN ('EM','SC','VC')) AS IsEmployee,
            (p.BusinessEntityID IN (SELECT BusinessEntityID FROM "_aw_SalesPerson"))
                AS IsSalesperson,
            NULL::VARCHAR AS UserPreferences, NULL::VARCHAR AS PhoneNumber,
            NULL::VARCHAR AS FaxNumber, NULL::VARCHAR AS EmailAddress,
            NULL::BLOB AS Photo, NULL::VARCHAR AS CustomFields,
            NULL::VARCHAR AS OtherLanguages, 1 AS LastEditedBy,
            p.ModifiedDate::TIMESTAMP AS ValidFrom, {FAR_FUTURE} AS ValidTo
        FROM "_aw_person" p
    """)
    done.append("Application_People")

    con.execute(f"""
        CREATE OR REPLACE TABLE Sales_CustomerCategories AS
        SELECT 1 AS CustomerCategoryID, 'Individual' AS CustomerCategoryName,
            1 AS LastEditedBy, TIMESTAMP '2019-01-01' AS ValidFrom, {FAR_FUTURE} AS ValidTo
        UNION ALL
        SELECT 2, 'Store', 1, TIMESTAMP '2019-01-01', {FAR_FUTURE}
        UNION ALL
        SELECT 3, 'Online', 1, TIMESTAMP '2019-01-01', {FAR_FUTURE}
    """)
    done.append("Sales_CustomerCategories")

    con.execute("""
        CREATE OR REPLACE TABLE Sales_SpecialDeals AS
        SELECT d.ProductID::INTEGER AS SpecialDealID,
            d.ProductID::INTEGER AS StockItemID,
            d.UnitPriceDiscount::DECIMAL(18,3) AS DiscountPercentage,
            d.ModifiedDate::DATE AS StartDate,
            d.ModifiedDate::DATE AS EndDate,
            'Online promotion' AS Description, 1 AS LastEditedBy,
            d.ModifiedDate::TIMESTAMP AS LastEditedWhen
        FROM "_aw_SalesOrderDetail" d
        WHERE d.UnitPriceDiscount > 0
        QUALIFY ROW_NUMBER() OVER (PARTITION BY d.ProductID ORDER BY d.SalesOrderDetailID) = 1
    """)
    done.append("Sales_SpecialDeals")

    return done


def drop_aw_temp(con: duckdb.DuckDBPyConnection) -> None:
    for (t,) in con.execute("SHOW TABLES").fetchall():
        if t.startswith("_aw_"):
            con.execute(f'DROP TABLE "{t}"')


def _row_count(con: duckdb.DuckDBPyConnection, table: str) -> int:
    try:
        return int(con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
    except duckdb.Error:
        return -1


def generate_metadata(con: duckdb.DuckDBPyConnection) -> None:
    """Regenerate database_schema.yaml + data_dictionary.yaml from the live database."""
    from import_wwi import _describe_table, _list_tables, _render_schema_yaml

    info = {t: _describe_table(con, t) for t in _list_tables(con)}

    SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)
    (SCHEMAS_DIR / "database_schema.yaml").write_text(
        _render_schema_yaml(info), encoding="utf-8")

    doc = {
        "version": "v0.3",
        "status": "verified",
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_database": "AdventureWorks OLTP (sales facts) + WideWorldImporters (schema)",
        "source_note": (
            "Sales / customer / product / people tables are populated from Microsoft's "
            "official AdventureWorks OLTP sample dataset. The table and column names keep "
            "the WideWorldImporters spelling so existing tool schemas stay valid."
        ),
        "coverage": {
            "order_date": "2022-05-30 .. 2025-06-29",
            "invoice_date": "2022-06-06 .. 2025-07-06",
            "note": (
                "数据截止 2025 年中；遇到“最近三个月/最近一年”等相对时间问题时，"
                "请以 MAX(OrderDate) 为锚点向前推算，而不是用今天的日历日期，"
                "否则会查出空结果。涉及趋势、环比、同比的问题，"
                "请用 strftime(OrderDate, '%Y-%m') 按月分组取数，"
                "不要只返回单个聚合值，否则无法计算环比。"
            ),
        },
        "tables": {
            name: {
                "primary_key": table["primary_key"],
                "columns": [{"name": c["name"], "type": c["type"]} for c in table["columns"]],
            }
            for name, table in sorted(info.items())
        },
        "metrics": {
            "total_revenue": {
                "chinese": "总营收",
                "definition": "订单行金额合计 = Quantity * UnitPrice（发票口径见 ExtendedPrice）",
                "formula": "SUM(Quantity * UnitPrice) FROM Sales_OrderLines",
            },
            "invoiced_revenue": {
                "chinese": "已开票营收",
                "definition": "发票行的扩展金额合计，已扣折扣",
                "formula": "SUM(ExtendedPrice) FROM Sales_InvoiceLines",
            },
            "gross_profit": {
                "chinese": "毛利润",
                "definition": "发票行毛利 = ExtendedPrice - 标准成本 * 数量",
                "formula": "SUM(LineProfit) FROM Sales_InvoiceLines",
            },
            "order_count": {
                "chinese": "订单量",
                "definition": "订单头数量",
                "formula": "COUNT(OrderID) FROM Sales_Orders",
            },
            "active_customer": {
                "chinese": "活跃客户",
                "definition": "统计周期内至少产生 1 笔订单的客户",
                "formula": "COUNT(DISTINCT CustomerID) FROM Sales_Orders",
            },
            "average_order_value": {
                "chinese": "客单价",
                "definition": "订单金额合计 / 订单数",
                "formula": "SUM(Quantity * UnitPrice) / COUNT(DISTINCT OrderID)",
            },
        },
        "notes": [
            "本字典由 scripts/import_adventureworks.py 从 DuckDB 内省结果自动生成。",
            "销售事实表（Sales_Orders / Sales_OrderLines / Sales_Invoices / "
            "Sales_InvoiceLines / Sales_CustomerTransactions）与客户、商品、人员维度表"
            "均为 Microsoft AdventureWorks 官方示例数据，属公开教学数据集，"
            "不得声称属于任何真实企业。",
            "Sales_CustomerCategories / Warehouse_StockGroups 由商品类别映射生成；"
            "Sales_Customers 的地理字段（DeliveryCityID 等）为 NULL，"
            "跨地区分析请改用 Sales_SpecialDeals 或商品类别维度。",
            "WWI 维度表主键为 varchar dimension id（如 'WWI1'），非自增 int。",
        ],
    }
    (SCHEMAS_DIR / "data_dictionary.yaml").write_text(
        yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8")
    _log(f"regenerated schema metadata for {len(info)} tables -> {SCHEMAS_DIR}")


FEWSHOT_EXAMPLES = [
    # Subquery anchor rather than a comma cross join: the LLM regularly drops
    # the ", mx" half and then DuckDB rejects the statement with
    # `Referenced table "mx" not found`.  A scalar subquery has no alias to
    # forget, so it survives generation far more often.
    ("最近三个月的月度销售额分别是多少（按月拆分，便于计算总额与环比）",
     "SELECT strftime(o.OrderDate, '%Y-%m') AS month, "
     "SUM(ol.Quantity * ol.UnitPrice) AS revenue "
     "FROM Sales_Orders o JOIN Sales_OrderLines ol ON ol.OrderID = o.OrderID "
     "WHERE o.OrderDate >= (SELECT MAX(OrderDate) - INTERVAL 3 MONTH FROM Sales_Orders) "
     "GROUP BY 1 ORDER BY 1"),
    ("订单数量最多的前 10 个客户",
     "SELECT c.CustomerName, COUNT(o.OrderID) AS order_count "
     "FROM Sales_Orders o JOIN Sales_Customers c ON c.CustomerID = o.CustomerID "
     "GROUP BY c.CustomerName ORDER BY order_count DESC LIMIT 10"),
    ("销售额最高的 5 个商品类别",
     "SELECT sg.StockGroupName, SUM(ol.Quantity * ol.UnitPrice) AS revenue "
     "FROM Sales_OrderLines ol "
     "JOIN Warehouse_StockItemStockGroups sgk ON sgk.StockItemID = ol.StockItemID "
     "JOIN Warehouse_StockGroups sg ON sg.StockGroupID = sgk.StockGroupID "
     "GROUP BY sg.StockGroupName ORDER BY revenue DESC LIMIT 5"),
    ("每个销售员的销售额排名",
     "SELECT p.FullName, SUM(ol.Quantity * ol.UnitPrice) AS revenue "
     "FROM Sales_OrderLines ol "
     "JOIN Sales_Orders o ON o.OrderID = ol.OrderID "
     "JOIN Application_People p ON p.PersonID = o.SalespersonPersonID "
     "GROUP BY p.FullName ORDER BY revenue DESC"),
    ("2024 年每月订单金额趋势",
     "SELECT strftime(o.OrderDate, '%Y-%m') AS month, "
     "SUM(ol.Quantity * ol.UnitPrice) AS revenue "
     "FROM Sales_Orders o JOIN Sales_OrderLines ol ON ol.OrderID = o.OrderID "
     "WHERE o.OrderDate >= DATE '2024-01-01' AND o.OrderDate < DATE '2025-01-01' "
     "GROUP BY 1 ORDER BY 1"),
    ("客单价最高的 5 个客户",
     "SELECT c.CustomerName, SUM(ol.Quantity * ol.UnitPrice) / COUNT(DISTINCT o.OrderID) "
     "AS avg_order_value "
     "FROM Sales_Orders o "
     "JOIN Sales_OrderLines ol ON ol.OrderID = o.OrderID "
     "JOIN Sales_Customers c ON c.CustomerID = o.CustomerID "
     "GROUP BY c.CustomerName HAVING COUNT(DISTINCT o.OrderID) >= 5 "
     "ORDER BY avg_order_value DESC LIMIT 5"),
    ("毛利润最高的 10 个商品",
     "SELECT s.StockItemName, SUM(il.LineProfit) AS gross_profit "
     "FROM Sales_InvoiceLines il "
     "JOIN Warehouse_StockItems s ON s.StockItemID = il.StockItemID "
     "GROUP BY s.StockItemName ORDER BY gross_profit DESC LIMIT 10"),
]


def generate_fewshots(con: duckdb.DuckDBPyConnection) -> None:
    """Write SQL few-shot examples and verify every one against the live database."""
    verified: list[dict[str, object]] = []
    for i, (question, sql) in enumerate(FEWSHOT_EXAMPLES, 1):
        try:
            rows = con.execute(sql).fetchall()
            verified.append({
                "id": f"fs-{i:03d}",
                "question": question,
                "sql": sql,
                "rows_returned": len(rows),
                "difficulty": "medium",
                "status": "verified",
            })
            _log(f"  few-shot {i} verified ({len(rows)} rows)")
        except duckdb.Error as exc:
            _log(f"  few-shot {i} FAILED: {exc}")

    doc = {
        "version": "v0.3",
        "dialect": "duckdb",
        "target_database": "AdventureWorks (WideWorldImporters schema)",
        "status": "verified",
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "few_shots": verified,
        "constraints": [
            "Sales_OrderLines.Quantity * Sales_OrderLines.UnitPrice 是订单金额口径。",
            "Sales_InvoiceLines.ExtendedPrice 是已开票金额口径，LineProfit 是毛利。",
            "数据覆盖 2022-05-30 至 2025-06-29，相对日期请锚定 MAX(OrderDate)。",
            "Sales_Customers 的地理字段为 NULL，不要做跨省/跨区域聚合。",
        ],
    }
    SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)
    (SCHEMAS_DIR / "sql_fewshots.yaml").write_text(
        yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8")
    _log(f"wrote {len(verified)} verified few-shots -> "
         f"{SCHEMAS_DIR / 'sql_fewshots.yaml'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-backup", action="store_true", help="skip the wwi.duckdb backup")
    args = parser.parse_args()

    if not (AW_DIR / "SalesOrderHeader.csv").exists():
        print(f"ERROR: AdventureWorks CSVs not found under {AW_DIR}", file=sys.stderr)
        return 1

    if not args.no_backup and DB_PATH.exists():
        shutil.copy2(DB_PATH, BACKUP_PATH)
        _log(f"backed up -> {BACKUP_PATH}")

    con = duckdb.connect(str(DB_PATH))
    try:
        _log("loading AdventureWorks CSVs ...")
        for name in CSV_SPECS:
            n = load_tab_csv(con, name)
            _log(f"  _aw_{name:<18} {n:>7} rows")
        n = load_person(con)
        _log(f"  _aw_person            {n:>7} rows")
        n = load_business_entity_address(con)
        _log(f"  _aw_BusinessEntityAddress {n:>7} rows")

        _log("extending address dimension with AdventureWorks states/cities ...")
        extend_address_dimension(con)
        n = _row_count(con, "Application_Cities")
        _log(f"  Application_Cities    {n:>7} rows")
        n = _row_count(con, "Application_StateProvinces")
        _log(f"  Application_StateProvinces {n:>7} rows")

        _log("rebuilding WideWorldImporters-named tables ...")
        for table in rebuild_tables(con):
            n = _row_count(con, table)
            _log(f"  {table:<32} {n:>7} rows")

        drop_aw_temp(con)
        _log("dropped temporary _aw_* tables")

        _log("\nfinal table inventory:")
        for t, in sorted(con.execute("SHOW TABLES").fetchall()):
            _log(f"  {t:<36} {_row_count(con, t):>7} rows")

        _log(f"\nkept synthetic tables: {', '.join(KEEP_TABLES)}")

        _log("\nregenerating schema metadata + few-shots ...")
        generate_metadata(con)
        generate_fewshots(con)
    finally:
        con.close()

    _log("\ndone. restart the backend to pick up the new data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
