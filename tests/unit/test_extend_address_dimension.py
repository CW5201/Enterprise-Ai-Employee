"""Regression tests for ``import_adventureworks.extend_address_dimension``.

The function used to append *every* AW state above the existing id range with
only an ``id NOT IN`` guard, never checking whether a row with the same state
code already existed.  The WideWorldImporters tables already held the 53 US
states, so the load produced a second ``California``:

* ``WHERE StateProvinceCode='CA'`` returned two rows (one with a population,
  one NULL) — no single answer to "加州人口是多少";
* the state's cities were split across both rows (119 under id 5, 96 under
  id 1009), so anything grouping by ``StateProvinceID`` saw two Californias.

The repair lives in ``scripts/dedupe_address_dimensions.py``; these tests pin
the behaviour that stops it happening again — reuse by natural key, allocate
only what is genuinely new, and be a no-op on a second run.
"""

from __future__ import annotations

import duckdb
import pytest
from scripts.import_adventureworks import (
    AW_CITY_ID_BASE,
    AW_STATE_PROVINCE_ID_BASE,
    extend_address_dimension,
)

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
    c = duckdb.connect(":memory:")
    c.execute(f"CREATE TABLE Application_StateProvinces ({_COLS_SP})")
    c.execute(f"CREATE TABLE Application_Cities ({_COLS_CITY})")
    # The WideWorldImporters rows that were already there: California with a
    # population and two of its cities.
    c.execute(
        "INSERT INTO Application_StateProvinces VALUES "
        "(5, 'CA', 239, 'California', 'Far West', 38332521, 1, "
        " TIMESTAMP '2020-01-01', TIMESTAMP '9999-12-31')"
    )
    c.execute(
        "INSERT INTO Application_Cities VALUES "
        "(51, 'Acampo', 5, NULL, 1, TIMESTAMP '2019-01-01', TIMESTAMP '9999-12-31'), "
        "(52, 'Acton', 5, NULL, 1, TIMESTAMP '2019-01-01', TIMESTAMP '9999-12-31')"
    )
    # AW staging: the same California (must be REUSED), a brand-new state, and
    # addresses covering a reused city plus a new city in each state.
    c.execute(
        'CREATE TABLE "_aw_StateProvince" AS SELECT * FROM (VALUES '
        "(5, 'CA', 4, 'California', TIMESTAMP '2020-01-01'), "
        "(99, 'XX', 4, 'Newland', TIMESTAMP '2020-01-01') "
        ") t(StateProvinceID, StateProvinceCode, TerritoryID, StateProvinceName, ModifiedDate)"
    )
    c.execute(
        'CREATE TABLE "_aw_Address" AS SELECT * FROM (VALUES '
        "(1, 'Acampo', 5), "
        "(2, 'Springfield', 5), "
        "(3, 'Springfield', 99) "
        ") t(AddressID, City, StateProvinceID)"
    )
    return c


@pytest.mark.unit
class TestExtendAddressDimension:
    def test_reuses_an_existing_state_instead_of_duplicating_it(self, con) -> None:
        extend_address_dimension(con)
        rows = con.execute(
            "SELECT StateProvinceID, StateProvinceName FROM Application_StateProvinces "
            "WHERE StateProvinceCode = 'CA'"
        ).fetchall()
        assert rows == [(5, "California")], "a second California was appended"

    def test_state_codes_stay_unique(self, con) -> None:
        extend_address_dimension(con)
        dupes = con.execute(
            "SELECT COUNT(*) FROM (SELECT StateProvinceCode FROM "
            "Application_StateProvinces GROUP BY 1 HAVING COUNT(*) > 1)"
        ).fetchone()[0]
        assert dupes == 0

    def test_a_genuinely_new_state_is_still_allocated_above_the_range(self, con) -> None:
        extend_address_dimension(con)
        row = con.execute(
            "SELECT StateProvinceID, StateProvinceName FROM Application_StateProvinces "
            "WHERE StateProvinceCode = 'XX'"
        ).fetchone()
        assert row == (99 + AW_STATE_PROVINCE_ID_BASE, "Newland")

    def test_an_existing_city_keeps_its_id(self, con) -> None:
        extend_address_dimension(con)
        addr = con.execute(
            "SELECT CityID FROM \"_aw_address_city\" WHERE AddressID = 1"
        ).fetchone()
        assert addr == (51,), "Acampo must resolve to the CityID already in the table"
        city = con.execute(
            "SELECT COUNT(*) FROM Application_Cities WHERE CityName = 'Acampo'"
        ).fetchone()[0]
        assert city == 1

    def test_new_cities_are_allocated_above_the_existing_range(self, con) -> None:
        extend_address_dimension(con)
        ids = [
            r[0]
            for r in con.execute(
                "SELECT CityID FROM Application_Cities WHERE CityName = 'Springfield'"
            ).fetchall()
        ]
        assert len(ids) == 2, "one per state"
        assert all(i >= AW_CITY_ID_BASE for i in ids)
        assert con.execute(
            "SELECT COUNT(*) FROM (SELECT CityName, StateProvinceID FROM "
            "Application_Cities GROUP BY 1, 2 HAVING COUNT(*) > 1)"
        ).fetchone()[0] == 0

    def test_address_map_resolves_both_states(self, con) -> None:
        extend_address_dimension(con)
        ca_id = con.execute(
            "SELECT StateProvinceID FROM Application_StateProvinces WHERE StateProvinceCode='CA'"
        ).fetchone()[0]
        xx_id = con.execute(
            "SELECT StateProvinceID FROM Application_StateProvinces WHERE StateProvinceCode='XX'"
        ).fetchone()[0]
        got = dict(con.execute("SELECT AddressID, CityID FROM \"_aw_address_city\"").fetchall())
        assert set(got) == {1, 2, 3}
        sp_of = dict(
            con.execute("SELECT CityID, StateProvinceID FROM Application_Cities").fetchall()
        )
        assert sp_of[got[2]] == ca_id
        assert sp_of[got[3]] == xx_id

    def test_second_run_is_a_no_op(self, con) -> None:
        extend_address_dimension(con)
        before = con.execute(
            "SELECT COUNT(*) FROM Application_StateProvinces"
        ).fetchone()[0], con.execute("SELECT COUNT(*) FROM Application_Cities").fetchone()[0]
        extend_address_dimension(con)
        after = con.execute(
            "SELECT COUNT(*) FROM Application_StateProvinces"
        ).fetchone()[0], con.execute("SELECT COUNT(*) FROM Application_Cities").fetchone()[0]
        assert before == after, "rerunning the import duplicated rows"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
