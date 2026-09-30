#!/usr/bin/env python
"""Build the WWI business knowledge graph in Neo4j from the DuckDB runtime.

Phase 2.4 Commit 2.  Driven by ``data/schemas/kg_mapping.yaml`` so that the
node/relationship model stays declarative and re-runnable.

Usage::

    python scripts/build_kg.py            # incremental MERGE (idempotent)
    python scripts/build_kg.py --reset     # wipe only the eae-managed graph first
    python scripts/build_kg.py --dry-run   # count what would be imported, write nothing
    python scripts/build_kg.py --limit 50  # only import first N rows per node table
    python scripts/build_kg.py --verbose   # per-stage logging

Credentials come from the environment (``NEO4J_PASSWORD`` etc.); the DuckDB
path defaults to ``data/runtime/wwi.duckdb`` and can be overridden with
``DUCKDB_PATH``.  The script writes a build summary to
``artifacts/phase2.4/kg_build_summary.json`` (gitignored).

Idempotency: every write uses MERGE on the stable business key, so repeated
runs converge to the same graph.  ``--reset`` deletes only nodes and
relationships carrying ``__graph='eae'`` and never touches the rest of the
database.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.neo4j_client import Neo4jClient  # noqa: E402
from src.core.neo4j_schema import apply_constraints_and_indexes  # noqa: E402

logger = logging.getLogger("eae.build_kg")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAPPING_FILE = _PROJECT_ROOT / "data" / "schemas" / "kg_mapping.yaml"
ARTIFACT_FILE = _PROJECT_ROOT / "artifacts" / "phase2.4" / "kg_build_summary.json"


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


def _cypher_value(v: Any) -> Any:
    """Convert a DuckDB value into a value usable as a Cypher parameter."""
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, bytes):
        return None  # BLOB columns (Photo etc.) are not imported
    return v


@dataclass
class BuildReport:
    nodes_by_label: dict[str, int] = field(default_factory=dict)
    relationships_by_type: dict[str, int] = field(default_factory=dict)
    duplicates_merged: int = 0
    skipped_records: int = 0
    failed_records: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "nodes_by_label": self.nodes_by_label,
            "relationships_by_type": self.relationships_by_type,
            "total_nodes": sum(self.nodes_by_label.values()),
            "total_relationships": sum(self.relationships_by_type.values()),
            "duplicates_merged": self.duplicates_merged,
            "skipped_records": self.skipped_records,
            "failed_records": self.failed_records,
            "errors": self.errors[:50],
        }


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------

_MAPPING_CACHE: dict[str, Any] | None = None


def load_mapping(path: Path | None = None) -> dict[str, Any]:
    global _MAPPING_CACHE
    if _MAPPING_CACHE is None:
        with (path or MAPPING_FILE).open(encoding="utf-8") as fh:
            _MAPPING_CACHE = yaml.safe_load(fh)
        if not isinstance(_MAPPING_CACHE, dict):
            raise ValueError("kg_mapping.yaml did not parse to a mapping")
    return _MAPPING_CACHE


def _key_prop(label: str) -> str:
    """The unique-constraint property of a label, as declared in
    ``config/neo4j_schema.yaml``.  The naming is not derivable by simple
    lower-casing (StockItem -> stock_item_id, BuyingGroup -> buying_group_id),
    so it is fixed here and matched against the schema constraints.
    """
    key_map = {
        "Customer": "customer_id",
        "Order": "order_id",
        "Invoice": "invoice_id",
        "StockItem": "stock_item_id",
        "Supplier": "supplier_id",
        "BuyingGroup": "buying_group_id",
        "City": "city_id",
    }
    if label not in key_map:
        raise ValueError(f"Unknown graph label {label!r}; add it to _key_prop")
    return key_map[label]


# ---------------------------------------------------------------------------
# DuckDB access
# ---------------------------------------------------------------------------


def _resolve_duckdb_path() -> Path:
    env = os.environ.get("DUCKDB_PATH")
    if env:
        p = Path(env)
        return p if p.is_absolute() else _PROJECT_ROOT / p
    return _PROJECT_ROOT / "data" / "runtime" / "wwi.duckdb"


def _read_table(con: duckdb.DuckDBPyConnection, table: str, limit: int | None) -> list[dict[str, Any]]:
    sql = f'SELECT * FROM "{table}"'
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    result = con.execute(sql)
    cols = [d[0] for d in result.description]
    return [{c: _cypher_value(r[i]) for i, c in enumerate(cols)} for r in result.fetchall()]


# ---------------------------------------------------------------------------
# Neo4j writes
# ---------------------------------------------------------------------------


def _merge_node(
    client: Neo4jClient,
    label: str,
    row: dict[str, Any],
    props: dict[str, Any],
    report: BuildReport,
    dry_run: bool,
) -> None:
    """MERGE one node by its business key (ON CREATE SET keeps repeated runs
    idempotent without clobbering newer property values).

    ``row`` is keyed by *source* column names; the business key's source
    column is looked up from the ``props`` mapping (source -> target).
    """
    key_prop = _key_prop(label)
    key_source_col = next((src for src, tgt in props.items() if tgt == key_prop), None)
    key_value = _cypher_value(row.get(key_source_col)) if key_source_col else None
    if key_value is None:
        report.skipped_records += 1
        return
    node_props = {p: _cypher_value(row.get(src)) for src, p in props.items()}
    node_props = {k: v for k, v in node_props.items() if v is not None}
    node_props["__source"] = f"wwi_duckdb:{label}"
    node_props["__graph"] = "eae"

    if dry_run:
        report.nodes_by_label[label] = report.nodes_by_label.get(label, 0) + 1
        return

    cypher = (
        f"MERGE (n:{label} {{ {key_prop}: $key }}) "
        f"ON CREATE SET n += $props"
    )
    try:
        client.run_write(cypher, {"key": key_value, "props": node_props})
        report.nodes_by_label[label] = report.nodes_by_label.get(label, 0) + 1
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        report.failed_records += 1
        report.errors.append(f"node {label} key={key_value}: {exc}")
        logger.warning("node write failed (%s key=%s): %s", label, key_value, exc)


def _merge_relationship_batch(
    client: Neo4jClient,
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
    report: BuildReport,
    dry_run: bool,
) -> None:
    """MERGE edges for one relationship type from pre-computed join rows.

    Each row must carry ``_from_key`` / ``_to_key`` (the business keys of the
    two endpoints) plus any declared edge properties.
    """
    rel_type = spec["relationship"]
    from_label = spec["from_label"]
    to_label = spec["to_label"]
    edge_props: dict[str, str] = spec.get("edge_properties", {})
    source_tag = spec.get("source_table", rel_type)

    for row in rows:
        from_val = row.get("_from_key")
        to_val = row.get("_to_key")
        if from_val is None or to_val is None:
            report.skipped_records += 1
            continue
        if dry_run:
            report.relationships_by_type[rel_type] = report.relationships_by_type.get(rel_type, 0) + 1
            continue

        params: dict[str, Any] = {
            "fk": from_val,
            "tk": to_val,
            "src": f"wwi_duckdb:{source_tag}",
            "graph": "eae",
        }
        edge_set: list[str] = ["r.__source = $src", "r.__graph = $graph"]
        for i, (prop_name, _src_col) in enumerate(edge_props.items()):
            # edge_props maps {edge_property_name: source_column}; the row
            # already carries the property under its edge-property name.
            param = f"ep{i}"
            params[param] = _cypher_value(row.get(prop_name))
            edge_set.append(f"r.{prop_name} = ${param}")

        cypher = (
            f"MATCH (f:{from_label} {{ {_key_prop(from_label)}: $fk }}) "
            f"MATCH (t:{to_label} {{ {_key_prop(to_label)}: $tk }}) "
            f"MERGE (f)-[r:{rel_type}]->(t) "
            f"SET {', '.join(edge_set)}"
        )
        try:
            client.run_write(cypher, params)
            report.relationships_by_type[rel_type] = (
                report.relationships_by_type.get(rel_type, 0) + 1
            )
        except Exception as exc:  # noqa: BLE001
            report.failed_records += 1
            report.errors.append(f"rel {rel_type} fk={from_val} tk={to_val}: {exc}")
            logger.warning("relationship write failed (%s %s->%s): %s", rel_type, from_val, to_val, exc)


def _relationship_rows(con: duckdb.DuckDBPyConnection, spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Run the DuckDB join that backs one relationship type.

    - direct edge: read ``source_table`` and project ``from_key`` / ``to_key``
      (plus any declared edge properties).  Rows with a NULL FK stay in the
      result; the merge step skips them and counts them as skipped.
    - derived edge (``via: [invoices_table, invoice_lines_table]``): join the
      two tables so each row is (OrderID, StockItemID).
    """
    via = spec.get("via")
    edge_props: dict[str, str] = spec.get("edge_properties", {})

    if via:
        sql = (
            f"SELECT i.OrderID AS _from_key, l.StockItemID AS _to_key, "
            f"l.Quantity, l.ExtendedPrice "
            f"FROM \"{via[0]}\" i "
            f"JOIN \"{via[1]}\" l ON i.InvoiceID = l.InvoiceID"
        )
        result = con.execute(sql)
        cols = [d[0] for d in result.description]
        rows: list[dict[str, Any]] = [
            {c: _cypher_value(r[i]) for i, c in enumerate(cols)}
            for r in result.fetchall()
        ]
        for row in rows:
            # edge_properties maps {edge_property_name: source_column}
            for prop_name, src_col in edge_props.items():
                row[prop_name] = _cypher_value(row.get(src_col))
        return rows

    table_rows = _read_table(con, spec["source_table"], None)
    rows = []
    for r in table_rows:
        row = {
            "_from_key": _cypher_value(r.get(spec["from_key"])),
            "_to_key": _cypher_value(r.get(spec["to_key"])),
        }
        for prop_name, src_col in edge_props.items():
            row[prop_name] = _cypher_value(r.get(src_col))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Reset (scoped to the eae namespace only — never wipes the whole database)
# ---------------------------------------------------------------------------


def reset_graph(client: Neo4jClient, namespace: str = "eae") -> dict[str, int]:
    """Delete only nodes/relationships marked with ``__graph=namespace``."""
    rels = client.run_write(
        f"MATCH ()-[r {{__graph: '{namespace}'}}]->() DELETE r RETURN count(*) AS c"
    )
    nodes = client.run_write(
        f"MATCH (n {{__graph: '{namespace}'}}) DETACH DELETE n RETURN count(*) AS c"
    )
    deleted = {
        "relationships": rels.records[0]["c"] if rels.records else 0,
        "nodes": nodes.records[0]["c"] if nodes.records else 0,
    }
    logger.info("graph reset (namespace=%s): %s", namespace, deleted)
    return deleted


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------


def build_graph(
    client: Neo4jClient,
    *,
    reset: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
) -> BuildReport:
    report = BuildReport()
    mapping = load_mapping()
    con = duckdb.connect(str(_resolve_duckdb_path()), read_only=True)

    try:
        if reset:
            if dry_run:
                logger.info("[dry-run] would reset namespace 'eae'")
            else:
                reset_graph(client, mapping.get("reset", {}).get("namespace", "eae"))

        if not dry_run:
            schema_res = apply_constraints_and_indexes(client)
            if not schema_res.ok:
                report.errors.extend(f"schema: {e}" for e in schema_res.errors)

        # 1. Nodes.
        for node_spec in mapping.get("nodes", []):
            label = node_spec["target_label"]
            table = node_spec["source_table"]
            props = node_spec.get("properties", {})
            logger.info("importing %s <- %s%s", label, table, f" (limit {limit})" if limit else "")
            rows = _read_table(con, table, limit)
            for row in rows:
                _merge_node(client, label, row, props, report, dry_run)
            if not dry_run:
                created = client.execute_readonly(
                    f"MATCH (n:{label} {{__graph: 'eae'}}) RETURN count(n) AS c"
                ).records[0]["c"]
                logger.info("%s: %d rows read, %d nodes in graph", label, len(rows), created)

        # 2. Relationships.
        for rel_spec in mapping.get("relationships", []):
            rel_type = rel_spec["relationship"]
            logger.info("building %s (%s -> %s)", rel_type, rel_spec["from_label"], rel_spec["to_label"])
            rows = _relationship_rows(con, rel_spec)
            _merge_relationship_batch(client, rel_spec, rows, report, dry_run)

        # 3. Sanity checks.
        if not dry_run:
            sanity = _run_sanity_checks(client)
            report.errors.extend(f"sanity: {s}" for s in sanity)
    finally:
        con.close()

    return report


def _run_sanity_checks(client: Neo4jClient) -> list[str]:
    """Return human-readable warnings (empty = all checks passed)."""
    checks: list[str] = []
    for cypher, label in [
        ("MATCH (c:Customer)-[:PLACED]->(o:Order) RETURN count(*) AS n", "Customer->Order"),
        ("MATCH (o:Order)-[:HAS_LINE]->(s:StockItem) RETURN count(*) AS n", "Order->StockItem"),
        ("MATCH (s:StockItem)-[:SUPPLIED_BY]->(p:Supplier) RETURN count(*) AS n", "StockItem->Supplier"),
        ("MATCH (o:Order)-[:INVOICED]->(i:Invoice) RETURN count(*) AS n", "Order->Invoice"),
        ("MATCH (i:Invoice)-[:SHIPPED_ON]->(s:StockItem) RETURN count(*) AS n", "Invoice->StockItem"),
    ]:
        try:
            res = client.execute_readonly(cypher)
            n = int(res.records[0]["n"])
            if n == 0:
                checks.append(f"no edges found for {label}")
            else:
                logger.info("sanity: %s -> %d edges", label, n)
        except Exception as exc:  # noqa: BLE001
            checks.append(f"sanity check failed for {label}: {exc}")

    try:
        orphans = client.execute_readonly(
            "MATCH (n {__graph: 'eae'}) WHERE NOT (n)--() RETURN count(n) AS c"
        ).records[0]["c"]
        total = client.execute_readonly(
            "MATCH (n {__graph: 'eae'}) RETURN count(n) AS c"
        ).records[0]["c"]
        if total > 0:
            ratio = orphans / total
            logger.info("orphan node ratio: %.2f%% (%d/%d)", ratio * 100, orphans, total)
            if ratio > 0.5:
                # Distinguish structural orphans (unused reference-dimension
                # nodes, e.g. cities no supplier delivers to) from core-chain
                # orphans (Customer/Order/Invoice/StockItem/Supplier with no
                # edges — would indicate a real import problem).
                core_orphans = client.execute_readonly(
                    "MATCH (n {__graph: 'eae'}) "
                    "WHERE (n:Customer OR n:Order OR n:Invoice "
                    "       OR n:StockItem OR n:Supplier OR n:BuyingGroup) "
                    "AND NOT (n)--() "
                    "RETURN count(n) AS c"
                ).records[0]["c"]
                if core_orphans > 0:
                    checks.append(
                        f"orphan node ratio {ratio:.2%}; {core_orphans} in core "
                        f"business chain (import problem?)"
                    )
                else:
                    logger.info(
                        "orphans are all in reference dimensions (%d non-core / %d total)",
                        orphans, total,
                    )
    except Exception as exc:  # noqa: BLE001
        checks.append(f"orphan check failed: {exc}")
    return checks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the WWI knowledge graph in Neo4j")
    parser.add_argument("--reset", action="store_true", help="wipe eae-managed graph before building")
    parser.add_argument("--dry-run", action="store_true", help="report only, do not write to Neo4j")
    parser.add_argument("--limit", type=int, default=None, help="max rows per source node table")
    parser.add_argument("--verbose", action="store_true", help="per-stage logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    client = Neo4jClient()
    try:
        if not args.dry_run:
            health = client.health_check()
            logger.info("neo4j health: %s", health)
        report = build_graph(
            client, reset=args.reset, dry_run=args.dry_run, limit=args.limit
        )
        summary = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "mode": "dry-run" if args.dry_run else "write",
            "reset": args.reset,
            "limit": args.limit,
            **report.as_dict(),
        }
        if not args.dry_run:
            ARTIFACT_FILE.parent.mkdir(parents=True, exist_ok=True)
            ARTIFACT_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("summary written to %s", ARTIFACT_FILE)
        else:
            print(json.dumps(summary, indent=2, ensure_ascii=False))

        failed = report.failed_records + len(report.errors)
        print(
            f"Build finished ({summary['mode']}): {summary['total_nodes']} nodes, "
            f"{summary['total_relationships']} relationships, "
            f"{report.skipped_records} skipped, {report.failed_records} failed, "
            f"{len(report.errors)} warnings."
        )
        return 1 if failed else 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
