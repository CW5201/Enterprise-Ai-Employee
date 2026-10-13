"""Tests for scripts/dedupe_address_dimensions.py.

The load path now reuses rows by natural key, but it cannot retract rows that
earlier runs already appended, so the repair script has to be correct and —
because it is meant to be safe to re-run — idempotent.
"""

from __future__ import annotations

import duckdb
import pytest
from scripts.dedupe_address_dimensions import apply, plan

_COLS_SP = (
    "StateProvinceID INTEGER, StateProvinceCode VARCHAR, CountryID INTEGER, "
    "StateProvinceName VARCHAR, SalesTerritory VARCHAR, "
    "LatestRecordedPopulation BIGINT, LastEditedBy INTEGER, "
    "ValidFrom TIMESTAMP, ValidTo TIMESTAMP"
)
_COLS_CITY = (
    "CityID INTEGER, CityName VARCHAR, StateProvinceID INTEGER, "
    "LatestRecordedPopulation BIGINT, LastEditedBy INTEGER, "
    "ValidFrom TIMESTAMP, ValidTo TIMESTAMP"
)


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    """California split across two state rows, one city on each, plus a city
    that becomes a duplicate once both halves point at the same state."""
    c = duckdb.connect(":memory:")
    c.execute(f"CREATE TABLE Application_StateProvinces ({_COLS_SP})")
    c.execute(f"CREATE TABLE Application_Cities ({_COLS_CITY})")
    c.execute(
        "INSERT INTO Application_StateProvinces VALUES "
        "(5, 'CA', 239, 'California', 'Far West', 38332521, 1, "
        " TIMESTAMP '2020-01-01', TIMESTAMP '9999-12-31'), "
        "(1009, 'CA', 239, 'California', '4', NULL, 1, "
        " TIMESTAMP '2025-02-07', TIMESTAMP '9999-12-31'), "
        "(22, 'MA', 239, 'MassachusettsE', 'New England', 6692824, 1, "
        " TIMESTAMP '2020-01-01', TIMESTAMP '9999-12-31'), "
        "(1030, 'MA', 239, 'Massachusetts', '4', NULL, 1, "
        " TIMESTAMP '2025-02-07', TIMESTAMP '9999-12-31')"
    )
    c.execute(
        "INSERT INTO Application_Cities VALUES "
        "(51, 'Acampo', 5, NULL, 1, TIMESTAMP '2019-01-01', TIMESTAMP '9999-12-31'), "
        "(40004, 'Alhambra', 1009, NULL, 1, TIMESTAMP '2019-01-01', TIMESTAMP '9999-12-31'), "
        "(40005, 'Daly City', 1009, NULL, 1, TIMESTAMP '2019-01-01', TIMESTAMP '9999-12-31'), "
        "(60, 'Daly City', 5, NULL, 1, TIMESTAMP '2019-01-01', TIMESTAMP '9999-12-31')"
    )
    return c


@pytest.mark.unit
class TestDedupeAddressDimensions:
    def test_plan_reports_the_duplicate_rows_without_writing(self, con) -> None:
        p = plan(con)
        assert sorted(src for src, _ in p["duplicate_state_provinces"]) == [1009, 1030]  # type: ignore[arg-type]
        assert p["cities_to_reparent"] == 2
        assert len(p["duplicate_cities"]) == 1  # Daly City, after reparenting
        assert len(p["name_fixes"]) == 1  # MassachusettsE

    def test_apply_makes_natural_keys_unique(self, con) -> None:
        apply(con, plan(con))
        assert con.execute(
            "SELECT COUNT(*) FROM (SELECT StateProvinceCode FROM "
            "Application_StateProvinces GROUP BY 1 HAVING COUNT(*) > 1)"
        ).fetchone()[0] == 0
        assert con.execute(
            "SELECT COUNT(*) FROM (SELECT CityName, StateProvinceID FROM "
            "Application_Cities GROUP BY 1, 2 HAVING COUNT(*) > 1)"
        ).fetchone()[0] == 0

    def test_cities_move_onto_the_surviving_state_row(self, con) -> None:
        apply(con, plan(con))
        rows = con.execute(
            "SELECT CityName, StateProvinceID FROM Application_Cities ORDER BY CityName"
        ).fetchall()
        assert rows == [
            ("Acampo", 5),
            ("Alhambra", 5),
            ("Daly City", 5),  # the 40005 duplicate was dropped
        ]
        assert con.execute(
            "SELECT COUNT(*) FROM Application_Cities c LEFT JOIN "
            "Application_StateProvinces s ON s.StateProvinceID = c.StateProvinceID "
            "WHERE s.StateProvinceID IS NULL"
        ).fetchone()[0] == 0

    def test_the_population_bearing_row_survives(self, con) -> None:
        apply(con, plan(con))
        row = con.execute(
            "SELECT StateProvinceID, StateProvinceName, LatestRecordedPopulation "
            "FROM Application_StateProvinces WHERE StateProvinceCode = 'CA'"
        ).fetchone()
        assert row == (5, "California", 38332521)

    def test_decoration_is_normalised_to_the_shorter_name(self, con) -> None:
        apply(con, plan(con))
        assert con.execute(
            "SELECT StateProvinceName FROM Application_StateProvinces "
            "WHERE StateProvinceCode = 'MA'"
        ).fetchone() == ("Massachusetts",)

    def test_second_run_is_a_no_op(self, con) -> None:
        apply(con, plan(con))
        before = con.execute(
            "SELECT COUNT(*) FROM Application_StateProvinces"
        ).fetchone()[0], con.execute("SELECT COUNT(*) FROM Application_Cities").fetchone()[0]
        p2 = plan(con)
        assert p2["duplicate_state_provinces"] == []
        assert p2["cities_to_reparent"] == 0
        assert p2["duplicate_cities"] == []
        assert p2["name_fixes"] == []
        apply(con, p2)
        after = con.execute(
            "SELECT COUNT(*) FROM Application_StateProvinces"
        ).fetchone()[0], con.execute("SELECT COUNT(*) FROM Application_Cities").fetchone()[0]
        assert before == after


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
