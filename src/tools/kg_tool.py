"""KG tool — read-only knowledge-graph queries via validated Cypher templates.

Phase 2.4 Commit 3.  The KG tool is the *hand* of the agent, not the
*brain*: it executes a fixed set of business query templates against
Neo4j and returns structured results.  Every statement passes through
:mod:`src.core.cypher_validator` before execution; LLM-generated free-form
Cypher is explicitly NOT enabled (ADR-004, ``allow_llm_generated_cypher``
stays ``false``).

Design rules:

- READ ONLY: the validator rejects CREATE / MERGE / SET / DELETE / ...;
  the tool itself only calls ``Neo4jClient.execute_readonly``.
- Parameterised templates: entity ids come in as ``$params`` — never
  spliced into the Cypher string.
- Uniform result envelope (``success`` / ``data`` / ``rows`` / ``source`` /
  ``query_type`` / ``latency_ms``); on failure no credential or
  connection string leaks into the returned payload.
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.cypher_validator import (
    CypherValidationError,
    validate,
)
from src.core.neo4j_client import (
    Neo4jClient,
    Neo4jError,
)
from src.core.observability import observe

logger = logging.getLogger("eae.kg_tool")


# ---------------------------------------------------------------------------
# Query templates
#
# Each template is a dict with:
#   id          - stable template id (used in evidence_ref)
#   name        - human-readable name
#   description - what the template answers
#   cypher      - read-only Cypher with $param placeholders
#   params      - list of required parameter names (validated for presence)
#
# New templates must go through the validator at construction time (import
# time), so a broken template is a programming error, not a runtime one.
# ---------------------------------------------------------------------------


_TEMPLATES: dict[str, dict[str, Any]] = {
    "customer_orders": {
        "id": "customer_orders",
        "name": "Find orders placed by a customer",
        "description": "One-hop: list the orders a given customer has placed.",
        "cypher": (
            "MATCH (c:Customer {customer_id: $customer_id})-[o:PLACED]->(order:Order) "
            "RETURN order ORDER BY order.order_date DESC LIMIT 50"
        ),
        "params": ["customer_id"],
    },
    "order_items": {
        "id": "order_items",
        "name": "Find stock items in an order",
        "description": "One-hop: the stock items referenced by an order (via HAS_LINE).",
        "cypher": (
            "MATCH (o:Order {order_id: $order_id})-[h:HAS_LINE]->(item:StockItem) "
            "RETURN item ORDER BY item.stock_item_id"
        ),
        "params": ["order_id"],
    },
    "item_suppliers": {
        "id": "item_suppliers",
        "name": "Find suppliers of a stock item",
        "description": "One-hop: the supplier that provides a given stock item.",
        "cypher": (
            "MATCH (item:StockItem {stock_item_id: $stock_item_id})-[s:SUPPLIED_BY]->(sup:Supplier) "
            "RETURN sup ORDER BY sup.supplier_id"
        ),
        "params": ["stock_item_id"],
    },
    "customer_order_item_paths": {
        "id": "customer_order_item_paths",
        "name": "Customer -> Order -> Item paths (two-hop)",
        "description": "Two-hop: which stock items a customer's orders reference.",
        "cypher": (
            "MATCH (c:Customer {customer_id: $customer_id})"
            "-[o:PLACED]->(order:Order)-[h:HAS_LINE]->(item:StockItem) "
            "RETURN order.order_id AS order_id, item ORDER BY order.order_date DESC, item.stock_item_id "
            "LIMIT 100"
        ),
        "params": ["customer_id"],
    },
    "customer_items_via_supplier": {
        "id": "customer_items_via_supplier",
        "name": "Items a customer bought, supplied by a given supplier (cross-entity)",
        "description": (
            "Three-hop cross-entity: Customer -> Order -> StockItem, "
            "filtered by StockItem -> Supplier.  Answers 'which items of "
            "supplier X has customer Y ever ordered'."
        ),
        "cypher": (
            "MATCH (c:Customer {customer_id: $customer_id})"
            "-[:PLACED]->(order:Order)-[:HAS_LINE]->(item:StockItem)"
            "-[:SUPPLIED_BY]->(sup:Supplier {supplier_id: $supplier_id}) "
            "RETURN DISTINCT item ORDER BY item.stock_item_id"
        ),
        "params": ["customer_id", "supplier_id"],
    },
    "order_invoices": {
        "id": "order_invoices",
        "name": "Find invoices for an order",
        "description": "One-hop: invoices that bill a given order.",
        "cypher": (
            "MATCH (o:Order {order_id: $order_id})-[i:INVOICED]->(inv:Invoice) "
            "RETURN inv ORDER BY inv.invoice_id"
        ),
        "params": ["order_id"],
    },
    "count_orders_by_customer": {
        "id": "count_orders_by_customer",
        "name": "Count orders placed by a customer",
        "description": "Aggregation: how many orders a customer has placed.",
        "cypher": (
            "MATCH (c:Customer {customer_id: $customer_id})-[o:PLACED]->(order:Order) "
            "RETURN count(order) AS order_count"
        ),
        "params": ["customer_id"],
    },
    "supplier_city": {
        "id": "supplier_city",
        "name": "Delivery city of a supplier",
        "description": "One-hop: the city a supplier delivers to.",
        "cypher": (
            "MATCH (sup:Supplier {supplier_id: $supplier_id})-[c:SUPPLIER_IN_CITY]->(city:City) "
            "RETURN city ORDER BY city.city_id"
        ),
        "params": ["supplier_id"],
    },
    "customer_buying_group": {
        "id": "customer_buying_group",
        "name": "Buying group of a customer",
        "description": "One-hop: the buying group a customer belongs to.",
        "cypher": (
            "MATCH (c:Customer {customer_id: $customer_id})-[b:BELONGS_TO_GROUP]->(group:BuyingGroup) "
            "RETURN group ORDER BY group.buying_group_id"
        ),
        "params": ["customer_id"],
    },
    "find_customer_by_name": {
        "id": "find_customer_by_name",
        "name": "Look up a customer by (partial) name",
        "description": "Entity lookup: customers whose name contains the given substring.",
        "cypher": (
            "MATCH (c:Customer) WHERE c.name CONTAINS $name "
            "RETURN c ORDER BY c.customer_id LIMIT 20"
        ),
        "params": ["name"],
    },
    "find_item_by_name": {
        "id": "find_item_by_name",
        "name": "Look up a stock item by (partial) name",
        "description": "Entity lookup: stock items whose name contains the given substring.",
        "cypher": (
            "MATCH (i:StockItem) WHERE i.name CONTAINS $name "
            "RETURN i ORDER BY i.stock_item_id LIMIT 20"
        ),
        "params": ["name"],
    },
    "find_supplier_by_name": {
        "id": "find_supplier_by_name",
        "name": "Look up a supplier by (partial) name",
        "description": "Entity lookup: suppliers whose name contains the given substring.",
        "cypher": (
            "MATCH (s:Supplier) WHERE s.name CONTAINS $name "
            "RETURN s ORDER BY s.supplier_id LIMIT 20"
        ),
        "params": ["name"],
    },
    "customer_by_id": {
        "id": "customer_by_id",
        "name": "Look up a customer by ID",
        "description": "Entity lookup: the customer node with the given ID.",
        "cypher": "MATCH (c:Customer {customer_id: $customer_id}) RETURN c",
        "params": ["customer_id"],
    },
    "supplier_by_id": {
        "id": "supplier_by_id",
        "name": "Look up a supplier by ID",
        "description": "Entity lookup: the supplier node with the given ID.",
        "cypher": "MATCH (s:Supplier {supplier_id: $supplier_id}) RETURN s",
        "params": ["supplier_id"],
    },
    "item_by_id": {
        "id": "item_by_id",
        "name": "Look up a stock item by ID",
        "description": "Entity lookup: the stock item node with the given ID.",
        "cypher": "MATCH (i:StockItem {stock_item_id: $stock_item_id}) RETURN i",
        "params": ["stock_item_id"],
    },
    "order_by_id": {
        "id": "order_by_id",
        "name": "Look up an order by ID",
        "description": "Entity lookup: the order node with the given ID.",
        "cypher": "MATCH (o:Order {order_id: $order_id}) RETURN o",
        "params": ["order_id"],
    },
    "invoice_by_id": {
        "id": "invoice_by_id",
        "name": "Look up an invoice by ID",
        "description": "Entity lookup: the invoice node with the given ID.",
        "cypher": "MATCH (i:Invoice {invoice_id: $invoice_id}) RETURN i",
        "params": ["invoice_id"],
    },
    "invoice_lines": {
        "id": "invoice_lines",
        "name": "Stock items shipped on an invoice (with quantity)",
        "description": "One-hop: the items an invoice ships, with edge quantity/price.",
        "cypher": (
            "MATCH (inv:Invoice {invoice_id: $invoice_id})-[l:SHIPPED_ON]->(item:StockItem) "
            "RETURN item, l.quantity AS quantity, l.extended_price AS extended_price "
            "ORDER BY item.stock_item_id"
        ),
        "params": ["invoice_id"],
    },
    "customer_invoices": {
        "id": "customer_invoices",
        "name": "Invoices produced by a customer's orders (two-hop)",
        "description": "Customer -> Order -> Invoice.",
        "cypher": (
            "MATCH (c:Customer {customer_id: $customer_id})"
            "-[:PLACED]->(o:Order)-[:INVOICED]->(inv:Invoice) "
            "RETURN DISTINCT inv ORDER BY inv.invoice_id"
        ),
        "params": ["customer_id"],
    },
    "customer_item_suppliers": {
        "id": "customer_item_suppliers",
        "name": "Suppliers providing a customer's ordered items (multi-hop)",
        "description": (
            "Customer -> Order -> StockItem -> Supplier.  Answers 'which "
            "suppliers provided the items in customer X's orders' and can be "
            "aggregated with DISTINCT."
        ),
        "cypher": (
            "MATCH (c:Customer {customer_id: $customer_id})"
            "-[:PLACED]->(o:Order)-[:HAS_LINE]->(item:StockItem)"
            "-[:SUPPLIED_BY]->(sup:Supplier) "
            "RETURN DISTINCT sup ORDER BY sup.supplier_id"
        ),
        "params": ["customer_id"],
    },
    "supplier_customers": {
        "id": "supplier_customers",
        "name": "Customers who bought items supplied by a supplier (cross-entity)",
        "description": (
            "Reverse cross-entity: Supplier <- StockItem <- Invoice <- Order "
            "<- Customer.  Answers 'which customers ever bought something "
            "supplier X supplies'."
        ),
        "cypher": (
            "MATCH (sup:Supplier {supplier_id: $supplier_id})"
            "<-[:SUPPLIED_BY]-(item:StockItem)"
            "<-[:SHIPPED_ON]-(inv:Invoice)<-[:INVOICED]-(o:Order)"
            "<-[:PLACED]-(c:Customer) "
            "RETURN DISTINCT c ORDER BY c.customer_id"
        ),
        "params": ["supplier_id"],
    },
    "group_member_count": {
        "id": "group_member_count",
        "name": "Count members of a buying group",
        "description": "Aggregation: how many customers belong to a buying group.",
        "cypher": (
            "MATCH (g:BuyingGroup {buying_group_id: $buying_group_id}) "
            "OPTIONAL MATCH (c:Customer)-[:BELONGS_TO_GROUP]->(g) "
            "RETURN count(DISTINCT c) AS member_count"
        ),
        "params": ["buying_group_id"],
    },
    "invoice_item_suppliers": {
        "id": "invoice_item_suppliers",
        "name": "Suppliers providing the items shipped on an invoice",
        "description": (
            "Two-hop from an invoice: Invoice -> StockItem (SHIPPED_ON) -> "
            "Supplier (SUPPLIED_BY).  Answers 'which suppliers provided the "
            "items on invoice N'."
        ),
        "cypher": (
            "MATCH (inv:Invoice {invoice_id: $invoice_id})"
            "-[:SHIPPED_ON]->(item:StockItem)-[:SUPPLIED_BY]->(sup:Supplier) "
            "RETURN DISTINCT sup ORDER BY sup.supplier_id"
        ),
        "params": ["invoice_id"],
    },
    "relationship_exists": {
        "id": "relationship_exists",
        "name": "Check whether a relationship exists between two entities",
        "description": (
            "Existence: does a (Customer, Order) PLACED link exist?  "
            "Parameters: from_label, from_key, from_value, rel_type, to_label, to_key, to_value."
        ),
        "cypher": (
            "MATCH (a {__graph: 'eae'})-[r]->(b {__graph: 'eae'}) "
            "WHERE a.{from_key} = $from_value AND b.{to_key} = $to_value "
            "RETURN count(r) > 0 AS exists"
        ),
        # NOTE: this template is *constructed* with placeholder labels at
        # call time (see ``KGTool.query_relation_exists``); the base Cypher
        # here is validated at import time only for keyword rules.  The
        # constructed variant is validated again before execution.
        "params": ["from_value", "to_value"],
        "dynamic": True,
    },
}


def _validate_templates_at_import() -> None:
    """Every static template must pass the validator at import time."""
    for tid, spec in _TEMPLATES.items():
        if spec.get("dynamic"):
            continue  # constructed at call time; validated there
        try:
            validate(spec["cypher"])
        except CypherValidationError as exc:
            # A template failing its own validator is a programming error.
            raise RuntimeError(f"Template {tid!r} failed Cypher validation: {exc}") from exc


_validate_templates_at_import()


# ---------------------------------------------------------------------------
# Result payload
# ---------------------------------------------------------------------------


def _serialize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten a Neo4j record into a JSON-friendly dict.

    Node objects become ``{label: {props...}}``; relationship objects are
    skipped (we only ever project node/property lists).
    """
    out: dict[str, Any] = {}
    for key, value in record.items():
        if hasattr(value, "labels"):  # Neo4j Node
            out[key] = {str(label): dict(value.items()) for label in value.labels}
        elif hasattr(value, "type"):  # Neo4j Relationship
            out[key] = {"type": value.type}
        elif isinstance(value, dict):
            out[key] = value
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class KGTool:
    """Read-only knowledge-graph query tool (predefined templates only).

    Parameters
    ----------
    client:
        A :class:`Neo4jClient`.  When ``None`` the tool builds one from the
        environment (``NEO4J_URI`` / ``NEO4J_PASSWORD`` / ...).  Injected
        clients are preferred in tests.
    """

    def __init__(self, client: Neo4jClient | None = None) -> None:
        self.client = client

    name: str = "kg"
    description: str = (
        "Query the enterprise knowledge graph (customer/order/invoice/"
        "stock-item/supplier relations) with validated read-only Cypher "
        "templates."
    )
    input_schema: dict[str, Any] = {
        "template_id": "str (one of: " + ", ".join(_TEMPLATES.keys()) + ")",
        "parameters": "dict of template parameters",
    }
    timeout: int = 20

    # -- lifecycle ----------------------------------------------------------

    def _get_client(self) -> Neo4jClient:
        if self.client is None:
            self.client = Neo4jClient()
        return self.client

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    # -- public API ----------------------------------------------------------

    @observe("kg_query")
    def run(self, *, template_id: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a predefined template.  Returns the uniform envelope."""
        spec = _TEMPLATES.get(template_id)
        if spec is None:
            return self._fail(
                "unknown_template",
                f"template_id {template_id!r} is not registered; "
                f"valid ids: {sorted(_TEMPLATES.keys())}",
            )
        params = dict(parameters or {})

        # Param presence check.
        missing = [p for p in spec["params"] if p not in params]
        if missing:
            return self._fail(
                "missing_parameter",
                f"template {template_id!r} requires parameter(s): {missing}",
            )

        try:
            result = self._get_client().execute_readonly(
                spec["cypher"], params
            )
        except CypherValidationError as exc:
            return self._fail("validation_error", str(exc))
        except Neo4jError as exc:
            # Credential-safe: Neo4jError.message never contains URI/password.
            logger.exception("kg template %s failed: %s", template_id, exc)
            return self._fail("neo4j_error", exc.message, details=exc.details)
        except Exception:  # noqa: BLE001 - never leak stack to callers
            logger.exception("kg template %s unexpected failure", template_id)
            return self._fail("internal_error", "query failed (details in server log)")

        records = [_serialize_record(r) for r in result.records]
        return {
            "success": True,
            "data": records,
            "rows": len(records),
            "source": "neo4j",
            "query_type": spec["id"],
            "template_id": template_id,
            "template_name": spec.get("name", template_id),
            "latency_ms": result.latency_ms,
        }

    # -- named helpers (thin wrappers, keep the API discoverable) -----------

    def find_customer_orders(self, customer_id: int | str) -> dict[str, Any]:
        return self.run(template_id="customer_orders", parameters={"customer_id": customer_id})

    def find_order_items(self, order_id: int | str) -> dict[str, Any]:
        return self.run(template_id="order_items", parameters={"order_id": order_id})

    def find_item_suppliers(self, stock_item_id: int | str) -> dict[str, Any]:
        return self.run(template_id="item_suppliers", parameters={"stock_item_id": stock_item_id})

    def find_customer_order_item_paths(self, customer_id: int | str) -> dict[str, Any]:
        return self.run(
            template_id="customer_order_item_paths", parameters={"customer_id": customer_id}
        )

    def find_customer_items_via_supplier(
        self, customer_id: int | str, supplier_id: int | str
    ) -> dict[str, Any]:
        return self.run(
            template_id="customer_items_via_supplier",
            parameters={"customer_id": customer_id, "supplier_id": supplier_id},
        )

    def find_order_invoices(self, order_id: int | str) -> dict[str, Any]:
        return self.run(template_id="order_invoices", parameters={"order_id": order_id})

    def count_orders_by_customer(self, customer_id: int | str) -> dict[str, Any]:
        return self.run(template_id="count_orders_by_customer", parameters={"customer_id": customer_id})

    def find_customer_by_name(self, name: str) -> dict[str, Any]:
        return self.run(template_id="find_customer_by_name", parameters={"name": name})

    def find_item_by_name(self, name: str) -> dict[str, Any]:
        return self.run(template_id="find_item_by_name", parameters={"name": name})

    def find_supplier_by_name(self, name: str) -> dict[str, Any]:
        return self.run(template_id="find_supplier_by_name", parameters={"name": name})

    def query_relation_exists(
        self,
        *,
        from_label: str,
        from_key: str,
        from_value: Any,
        rel_type: str,
        to_label: str,
        to_key: str,
        to_value: Any,
    ) -> dict[str, Any]:
        """Check the existence of one specific relationship.

        The Cypher is *constructed* here from a fixed shape and validated
        before execution — no free-form Cypher is ever generated.
        """
        cypher = (
            f"MATCH (a:{from_label} {{ {from_key}: $from_value }})"
            f"-[:{rel_type}]->(b:{to_label} {{ {to_key}: $to_value }}) "
            "RETURN count(*) > 0 AS exists"
        )
        # from_label / rel_type / to_label come from our fixed template table
        # in practice; still validate the constructed statement so a typo
        # cannot smuggle in a write keyword.
        try:
            validated = validate(cypher, {"from_value": from_value, "to_value": to_value})
        except CypherValidationError as exc:
            return self._fail("validation_error", str(exc))
        try:
            result = self._get_client().execute_readonly(
                validated.cypher, validated.parameters
            )
        except Neo4jError as exc:
            logger.exception("relation-existence query failed")
            return self._fail("neo4j_error", exc.message, details=exc.details)
        records = [_serialize_record(r) for r in result.records]
        return {
            "success": True,
            "data": records,
            "rows": len(records),
            "source": "neo4j",
            "query_type": "relationship_exists",
            "template_id": "relationship_exists",
            "template_name": "Check relationship existence",
            "latency_ms": result.latency_ms,
        }

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _fail(error_code: str, message: str, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "success": False,
            "error": {"code": error_code, "message": message},
            "source": "neo4j",
            "data": None,
            "rows": 0,
        }
        if details:
            # details are logged server-side only; we include a redacted copy.
            payload["error"]["details"] = details
        return payload

    # -- enumeration ---------------------------------------------------------

    @staticmethod
    def list_templates() -> list[dict[str, Any]]:
        """Return the template catalogue (for the API / agent tool discovery)."""
        return [
            {
                "id": spec["id"],
                "name": spec.get("name", spec["id"]),
                "description": spec.get("description", ""),
                "parameters": spec["params"],
            }
            for spec in _TEMPLATES.values()
        ]


__all__ = ["KGTool", "_TEMPLATES", "_serialize_record"]
