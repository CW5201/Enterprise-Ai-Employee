"""Unit tests for src/core/cypher_validator.py (no Neo4j required)."""

from __future__ import annotations

import pytest

from src.core.cypher_validator import (
    CypherValidationError,
    ValidatedQuery,
    validate,
)

# ---------------------------------------------------------------------------
# Legal statements
# ---------------------------------------------------------------------------


class TestValidStatements:
    @pytest.mark.parametrize(
        "cypher",
        [
            "MATCH (c:Customer) RETURN c",
            "MATCH (c:Customer {customer_id: $id})-[o:PLACED]->(order:Order) RETURN order",
            "MATCH (o:Order) WHERE o.order_id = $id RETURN o ORDER BY o.order_date LIMIT 10",
            "MATCH (a:Customer)-[:PLACED]->(b:Order)-[:HAS_LINE]->(c:StockItem) RETURN c",
            "OPTIONAL MATCH (s:Supplier)-[r:SUPPLIED_BY]->(i:StockItem) RETURN s, r",
            "MATCH (n) RETURN count(n) AS c, collect(n) AS nodes",
            "MATCH (n:Customer) WITH n WHERE n.credit_limit > $limit RETURN n SKIP 0 LIMIT 5",
            "MATCH (i:StockItem) RETURN avg(i.unit_price) AS avg_price, min(i.unit_price), max(i.unit_price)",
            "MATCH (o:Order)-[h:HAS_LINE]->(i:StockItem) RETURN i, sum(h.quantity) AS total_qty",
            "RETURN 1 AS one",
            "WITH 1 AS x WHERE x = 1 RETURN x",
        ],
    )
    def test_accepted(self, cypher: str) -> None:
        result = validate(cypher)
        assert isinstance(result, ValidatedQuery)
        assert result.cypher == cypher.strip().rstrip(";").strip()

    def test_trailing_semicolon_stripped(self) -> None:
        result = validate("MATCH (n) RETURN n;")
        assert result.cypher.endswith("RETURN n")

    def test_parameters_passthrough(self) -> None:
        result = validate("MATCH (n) RETURN n", {"a": 1, "b": 2})
        assert result.parameters == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# Illegal statements
# ---------------------------------------------------------------------------


class TestInvalidStatements:
    @pytest.mark.parametrize(
        "cypher",
        [
            "CREATE (n:Customer {name: 'x'})",
            "MERGE (n:Order {order_id: 1})",
            "MATCH (n) DELETE n",
            "MATCH (n) DETACH DELETE n",
            "MATCH (n) SET n.x = 1",
            "MATCH (n:Customer) REMOVE n.name",
            "DROP CONSTRAINT customer_key",
            "CALL dbms.killQuery(0)",
            "LOAD CSV FROM 'http://x' AS line RETURN line",
            "FOREACH (x IN [1] | CREATE (n {v: x}))",
            "MATCH (n) CREATE (n)-[:X]->(m)",
            "APPLY something",
            "SCHEMA { CREATE INDEX ON (n:Customer.name) }",
            "CREATE INDEX idx FOR (n:Customer) ON (n.name)",
        ],
    )
    def test_rejected(self, cypher: str) -> None:
        with pytest.raises(CypherValidationError):
            validate(cypher)

    def test_must_start_with_allowed_keyword(self) -> None:
        with pytest.raises(CypherValidationError) as exc_info:
            validate("UPDATE customers SET x = 1")
        assert "must begin with" in exc_info.value.message

    def test_empty_statement(self) -> None:
        with pytest.raises(CypherValidationError):
            validate("   ")
        with pytest.raises(CypherValidationError):
            validate("")

    def test_write_keyword_inside_return_body_rejected(self) -> None:
        # A MATCH that sneaks CREATE into the RETURN clause must fail.
        with pytest.raises(CypherValidationError) as exc_info:
            validate("MATCH (n) RETURN n, 'create' AS x")
        # 'create' inside a string literal is conservatively rejected (word-boundary match).
        assert "CREATE" in str(exc_info.value.details)

    def test_unbalanced_parentheses_rejected(self) -> None:
        with pytest.raises(CypherValidationError):
            validate("MATCH (n:Customer WHERE n.x = 1 RETURN n")

    def test_unbalanced_brackets_rejected(self) -> None:
        with pytest.raises(CypherValidationError):
            validate("MATCH (n)-[:X {a: [1, 2]->() RETURN n")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_case_insensitive_keywords(self) -> None:
        with pytest.raises(CypherValidationError):
            validate("match (n) return n  // MERGE later")

    def test_keyword_as_bare_word_rejected(self) -> None:
        # A bare ``SET`` keyword anywhere is rejected; ``settings`` as a
        # property name (substring) is NOT, because the regex uses word
        # boundaries.  This is the intended, documented behaviour.
        with pytest.raises(CypherValidationError):
            validate("MATCH (n) WHERE n.x = 1 SET n.y = 2 RETURN n")
        # property names containing 'set' as a substring pass:
        validate("MATCH (n) WHERE n.settings = 1 RETURN n")

    def test_none_input(self) -> None:
        with pytest.raises(CypherValidationError):
            validate(None)  # type: ignore[arg-type]
