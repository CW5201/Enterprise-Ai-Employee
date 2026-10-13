"""De-duplicate Application_StateProvinces / Application_Cities.

``import_adventureworks.extend_address_dimension`` used to append every AW
state above the existing id range without checking whether a row with the same
code was already there.  The WideWorldImporters tables already held the 53 US
states, so the database ended up with two ``California`` rows:

* ``WHERE StateProvinceCode='CA'`` returned two rows (38332521 and NULL as the
  population), so a "加州人口是多少" question had no single answer;
* the state's cities were split across both rows (119 under id 5, 96 under
  id 1009), so anything grouping by ``StateProvinceID`` saw two Californias;
* 39 cities existed twice once the two halves were merged.

This is a one-off repair of already-loaded data — the import now reuses rows
by natural key, but it cannot retract rows inserted by earlier runs.  It is
idempotent: a second run reports nothing to do.

    python scripts/dedupe_address_dimensions.py            # dry run
    python scripts/dedupe_address_dimensions.py --apply    # write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "runtime" / "wwi.duckdb"

# For each duplicated (CountryID, StateProvinceCode) group, keep the row that
# carries a population — the WideWorldImporters rows do, the appended AW rows
# do not — falling back to the lowest id.  Same rule the import's resolver
# uses, so a later re-import lands on the same canonical row.
_CANON = """
WITH canon AS (
    SELECT CountryID, StateProvinceCode,
           COALESCE(
               MIN(StateProvinceID) FILTER (WHERE LatestRecordedPopulation IS NOT NULL),
               MIN(StateProvinceID)
           ) AS keep_id
    FROM Application_StateProvinces
    GROUP BY 1, 2
)
"""

# Rows to remove: everything in a group that is not the canonical row.
_PLAN = _CANON + """
, plan AS (
    SELECT s.StateProvinceID AS src, c.keep_id AS dst
    FROM Application_StateProvinces s
    JOIN canon c ON c.CountryID = s.CountryID
                AND c.StateProvinceCode = s.StateProvinceCode
    WHERE s.StateProvinceID <> c.keep_id
)
"""

DUP_SP_SQL = _PLAN + "SELECT src, dst FROM plan"

# Cities that sit on a soon-to-be-removed state row.
REPARENT_COUNT_SQL = _PLAN + """
SELECT COUNT(*)
FROM Application_Cities ci
JOIN plan p ON p.src = ci.StateProvinceID
"""

# City duplicates that only become visible once both halves point at one state.
DUP_CITY_SQL = _PLAN + """
, moved AS (
    SELECT ci.CityID, ci.CityName, COALESCE(p.dst, ci.StateProvinceID) AS sp
    FROM Application_Cities ci
    LEFT JOIN plan p ON p.src = ci.StateProvinceID
), keys AS (
    SELECT CityName, sp, MIN(CityID) AS keep_id
    FROM moved
    GROUP BY 1, 2
    HAVING COUNT(*) > 1
)
SELECT m.CityID
FROM moved m
JOIN keys k ON k.CityName = m.CityName AND k.sp = m.sp
WHERE m.CityID <> k.keep_id
"""

# Names that differ only by decoration one copy carries:
# "MassachusettsE" vs "Massachusetts", "Puerto Rico (US Territory)" vs
# "Puerto Rico".  When one is a prefix of the other the shorter is the real
# one, so adopt it on the surviving row.
NAME_FIX_SQL = _CANON + """
, dup AS (
    SELECT s.StateProvinceID AS src, c.keep_id AS dst,
           s.StateProvinceName AS src_name, k.StateProvinceName AS dst_name
    FROM Application_StateProvinces s
    JOIN canon c ON c.CountryID = s.CountryID
                AND c.StateProvinceCode = s.StateProvinceCode
    JOIN Application_StateProvinces k ON k.StateProvinceID = c.keep_id
    WHERE s.StateProvinceID <> c.keep_id
)
SELECT DISTINCT dst, dst_name, src_name
FROM dup
WHERE dst_name <> src_name
  AND (src_name LIKE dst_name || '%' OR dst_name LIKE src_name || '%')
  AND length(dst_name) <> length(src_name)
"""


def plan(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    """Everything the repair would change, without touching the database."""
    return {
        "duplicate_state_provinces": con.execute(DUP_SP_SQL).fetchall(),
        "cities_to_reparent": int(con.execute(REPARENT_COUNT_SQL).fetchone()[0]),
        "duplicate_cities": con.execute(DUP_CITY_SQL).fetchall(),
        "name_fixes": con.execute(NAME_FIX_SQL).fetchall(),
    }


def apply(con: duckdb.DuckDBPyConnection, p: dict[str, object]) -> None:
    con.execute("BEGIN TRANSACTION")
    try:
        # 1. normalise names while both rows still exist.
        for dst, dst_name, src_name in p["name_fixes"]:
            short = dst_name if len(dst_name) < len(src_name) else src_name
            con.execute(
                "UPDATE Application_StateProvinces SET StateProvinceName = ? "
                "WHERE StateProvinceID = ?",
                [short, dst],
            )
        # 2. reparent cities onto the surviving state row.
        con.execute(
            _PLAN
            + "UPDATE Application_Cities c SET StateProvinceID = p.dst "
              "FROM plan p WHERE p.src = c.StateProvinceID"
        )
        # 3. drop city duplicates (nothing else in the schema references CityID).
        con.execute(
            "WITH keep AS ("
            "SELECT CityName, StateProvinceID, MIN(CityID) AS keep_id "
            "FROM Application_Cities GROUP BY 1, 2 HAVING COUNT(*) > 1) "
            "DELETE FROM Application_Cities c USING keep k "
            "WHERE k.CityName = c.CityName "
            "AND k.StateProvinceID = c.StateProvinceID "
            "AND c.CityID <> k.keep_id"
        )
        # 4. drop the duplicate state rows.
        con.execute(
            _PLAN
            + "DELETE FROM Application_StateProvinces s USING plan p "
              "WHERE p.src = s.StateProvinceID"
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write the changes")
    ap.add_argument("--db", default=str(DB_PATH), help="path to wwi.duckdb")
    args = ap.parse_args(argv)

    con = duckdb.connect(args.db, read_only=not args.apply)
    p = plan(con)
    dup_sp = p["duplicate_state_provinces"]
    dup_city = p["duplicate_cities"]
    name_fixes = p["name_fixes"]
    n_sp = len(dup_sp)                      # type: ignore[arg-type]
    n_re = int(p["cities_to_reparent"])     # type: ignore[arg-type]
    n_city = len(dup_city)                  # type: ignore[arg-type]
    n_name = len(name_fixes)                # type: ignore[arg-type]

    print(f"state provinces to delete : {n_sp}")
    print(f"cities to reparent        : {n_re}")
    print(f"cities to delete (dupes)  : {n_city}")
    print(f"state names to normalise  : {n_name}")
    for dst, dst_name, src_name in name_fixes:  # type: ignore[union-attr]
        short = dst_name if len(dst_name) < len(src_name) else src_name
        print(f"    id={dst}: {dst_name!r} / {src_name!r} -> {short!r}")

    if not args.apply:
        print("\ndry run - pass --apply to write")
        return 0
    if not (n_sp or n_re or n_city or n_name):
        print("\nnothing to do")
        return 0

    apply(con, p)
    print("\napplied")

    bad_sp = con.execute(
        "SELECT COUNT(*) FROM (SELECT StateProvinceCode FROM "
        "Application_StateProvinces GROUP BY 1 HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    bad_city = con.execute(
        "SELECT COUNT(*) FROM (SELECT CityName, StateProvinceID FROM "
        "Application_Cities GROUP BY 1, 2 HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    print(f"remaining duplicate state codes : {bad_sp}")
    print(f"remaining duplicate cities      : {bad_city}")
    return 0 if bad_sp == 0 and bad_city == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
