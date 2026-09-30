"""Unit tests for src/tools/kg_tool.py (no Neo4j required — mocked client)."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.tools.kg_tool import _TEMPLATES, KGTool, _validate_templates_at_import


class TestTemplateCatalogue:
    def test_all_static_templates_are_valid_cypher(self) -> None:
        # Re-run the import-time check; it must not raise.
        _validate_templates_at_import()
        assert len(_TEMPLATES) >= 10

    def test_template_params_declared(self) -> None:
        for spec in _TEMPLATES.values():
            assert spec["params"], f"template {spec['id']} must declare params"

    def test_list_templates_shape(self) -> None:
        entries = KGTool.list_templates()
        assert all("id" in e and "parameters" in e for e in entries)


class TestToolFailures:
    def _tool_with_mock_client(self) -> tuple[KGTool, MagicMock]:
        client = MagicMock()
        tool = KGTool(client=client)
        return tool, client

    def test_unknown_template(self) -> None:
        tool, _ = self._tool_with_mock_client()
        result = tool.run(template_id="not_a_template", parameters={})
        assert result["success"] is False
        assert result["error"]["code"] == "unknown_template"
        # No stack trace, no URI, no password in the payload.
        assert "bolt" not in str(result)
        assert "password" not in str(result).lower()

    def test_missing_parameter(self) -> None:
        tool, client = self._tool_with_mock_client()
        result = tool.run(template_id="customer_orders", parameters={})
        assert result["success"] is False
        assert result["error"]["code"] == "missing_parameter"
        client.execute_readonly.assert_not_called()

    def test_neo4j_error_does_not_leak_credentials(self) -> None:
        from src.core.neo4j_client import Neo4jUnavailableError

        tool, client = self._tool_with_mock_client()
        client.execute_readonly.side_effect = Neo4jUnavailableError(
            "Neo4j unavailable",
            details={"kind": "connection", "message": "connection refused"},
        )
        result = tool.find_customer_orders(1)
        assert result["success"] is False
        assert result["error"]["code"] == "neo4j_error"
        assert "bolt" not in str(result)
        # No credential-looking token may leak into the returned payload.
        assert "secret" not in str(result).lower()

    def test_success_envelope_shape(self) -> None:
        from src.core.neo4j_client import Neo4jResult

        tool, client = self._tool_with_mock_client()
        client.execute_readonly.return_value = Neo4jResult(
            cypher="MATCH ...", records=[{"order_count": 3}], latency_ms=12.5
        )
        result = tool.count_orders_by_customer(1)
        assert result["success"] is True
        assert result["rows"] == 1
        assert result["data"] == [{"order_count": 3}]
        assert result["source"] == "neo4j"
        assert result["query_type"] == "count_orders_by_customer"
        assert result["latency_ms"] == 12.5

    def test_node_serialisation(self) -> None:
        from src.core.neo4j_client import Neo4jResult

        class _FakeNode(dict):
            labels = {"Customer"}

        tool, client = self._tool_with_mock_client()
        client.execute_readonly.return_value = Neo4jResult(
            cypher="...", records=[{"c": _FakeNode({"customer_id": 1, "name": "A"})}], latency_ms=1.0
        )
        result = tool.run(template_id="customer_orders", parameters={"customer_id": 1})
        assert result["data"][0]["c"]["Customer"]["customer_id"] == 1
        assert result["data"][0]["c"]["Customer"]["name"] == "A"
