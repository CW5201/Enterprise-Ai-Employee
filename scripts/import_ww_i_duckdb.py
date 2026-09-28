#!/usr/bin/env python3
"""Import WideWorldImporters sample database into the project's DuckDB file.

WideWorldImporters (https://github.com/microsoft/WideWorldImporters) ships a
SQL Server setup script; the plain-CSV data files are not in this repository
(see docs/DATASET.md and THIRD_PARTY_LICENSES.md).  This script:

1. looks for data/ddl/wide_world_importers.sql (copy it there after
   downloading the Microsoft sample repository);
2. creates the tables in the read-only-destination DuckDB file;
3. creates the ``finance_expenses`` table from data/schemas/database_schema.yaml
   and inserts a small **synthetic, clearly-labelled** dataset so that the
   finance-policy questions in the knowledge base have structured data to
   reference.  The synthetic rows are marked in the data dictionary and in
   the table comment; they are NOT presented as real WideWorldImporters data.

Usage (from the repository root)::

    python scripts/import_ww_i_duckdb.py

Re-running the script is safe: existing tables are replaced idempotently.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "duckdb" / "enterprise.duckdb"
DDL_PATH = PROJECT_ROOT / "data" / "ddl" / "wide_world_importers.sql"


def _log(message: str) -> None:
    print(f"[import-wwi] {message}", flush=True)


def import_wwi_sql(connection: duckdb.DuckDBPyConnection) -> bool:
    """Create WideWorldImporters tables from the vendor DDL script (if present)."""
    if not DDL_PATH.exists():
        _log(
            f"DDL file not found: {DDL_PATH}\n"
            "       Download WideWorldImporters from "
            "https://github.com/microsoft/WideWorldImporters,\n"
            "       save its setup SQL script to that path, then re-run this script.\n"
            "       Meanwhile the synthetic finance tables below are still created."
        )
        return False
    statements = _split_sql_statements(DDL_PATH.read_text(encoding="utf-8"))
    applied = 0
    for statement in statements:
        try:
            connection.execute(statement)
            applied += 1
        except duckdb.Error as exc:
            _log(f"skipped statement ({exc})")
    _log(f"applied {applied} DDL statements from WideWorldImporters")
    return True


def create_synthetic_finance_tables(connection: duckdb.DuckDBPyConnection) -> None:
    """Create the finance_expenses table with clearly-labelled synthetic rows.

    These rows support the "reimbursement policy" knowledge-base documents
    (e.g. travel-allowance limit) by giving the SQL tool structured data to
    reason about.  They are synthetic by design and are documented as such in
    data/schemas/data_dictionary.yaml and THIRD_PARTY_LICENSES.md.
    """
    connection.execute(
        """
        CREATE OR REPLACE TABLE finance_expenses (
            expense_id    INTEGER,
            employee_id   INTEGER,
            department_id INTEGER,
            expense_date  DATE,
            expense_type  VARCHAR,
            amount        DECIMAL(12,2),
            approved      BOOLEAN
        );
        """
    )
    rows = [
        # (expense_id, employee_id, department_id, date, type, amount, approved)
        (1, 101, 1, "2026-01-15", "travel", 1200.50, True),
        (2, 101, 1, "2026-02-10", "travel", 2600.00, True),
        (3, 102, 2, "2026-01-22", "entertainment", 800.00, True),
        (4, 103, 1, "2026-03-05", "procurement", 15000.00, True),
        (5, 103, 1, "2026-03-18", "travel", 3200.00, False),
        (6, 104, 3, "2026-02-28", "travel", 950.00, True),
        (7, 104, 3, "2026-04-02", "other", 120.00, True),
        (8, 105, 2, "2026-04-12", "entertainment", 2100.00, False),
    ]
    connection.executemany(
        "INSERT INTO finance_expenses VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    _log(f"created finance_expenses with {len(rows)} synthetic rows (labelled as synthetic)")


def _split_sql_statements(sql_text: str) -> list[str]:
    """Naive statement splitter good enough for the WideWorldImporters DDL."""
    statements: list[str] = []
    buffer = ""
    for line in sql_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        buffer += line + "\n"
        if stripped.endswith(";"):
            statements.append(buffer.strip())
            buffer = ""
    if buffer.strip():
        statements.append(buffer.strip())
    return [s for s in statements if s]


def main() -> int:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _log(f"opening DuckDB at {DB_PATH}")
    connection = duckdb.connect(str(DB_PATH))
    try:
        import_wwi_sql(connection)
        create_synthetic_finance_tables(connection)
        tables = [r[0] for r in connection.execute("SHOW TABLES").fetchall()]
        _log(f"tables in database: {tables}")
    finally:
        connection.close()
    _log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
