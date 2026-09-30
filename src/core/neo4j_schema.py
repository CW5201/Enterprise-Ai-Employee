"""Neo4j constraint / index management (idempotent DDL runner).

Phase 2.4 Commit 1.  Reads ``config/neo4j_schema.yaml`` and applies the
declared constraints and indexes to a live Neo4j instance.  Every DDL
statement uses ``IF NOT EXISTS`` so the runner is safe to re-execute.

This module is the only place that creates constraints/indexes.  Data
population (MERGE of nodes and relationships) lives in
``scripts/build_kg.py`` (Commit 2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.core.neo4j_client import Neo4jClient

logger = logging.getLogger("eae.neo4j.schema")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCHEMA_FILE = _PROJECT_ROOT / "config" / "neo4j_schema.yaml"


@dataclass
class SchemaApplyResult:
    constraints_created: int = 0
    indexes_created: int = 0
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []

    @property
    def ok(self) -> bool:
        return not self.errors


def load_schema(path: Path | None = None) -> dict[str, Any]:
    """Load and return the raw schema mapping from ``neo4j_schema.yaml``."""
    path = path or SCHEMA_FILE
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Schema file {path} did not parse to a mapping")
    return data


def apply_constraints_and_indexes(client: Neo4jClient, path: Path | None = None) -> SchemaApplyResult:
    """Apply all constraints and indexes declared in the schema file.

    Idempotent: every statement uses ``IF NOT EXISTS``.  On failure of an
    individual statement the error is recorded and the runner continues so a
    partial schema can still be inspected; the aggregate :attr:`ok` flag
    reflects whether *any* statement failed.
    """
    schema = load_schema(path)
    result = SchemaApplyResult()

    for stmt in schema.get("constraints", []):
        _run_ddl(client, stmt, result, kind="constraint")

    for stmt in schema.get("indexes", []):
        _run_ddl(client, stmt, result, kind="index")

    logger.info(
        "neo4j schema applied: %d constraints, %d indexes, %d errors",
        result.constraints_created, result.indexes_created, len(result.errors),
    )
    return result


def _run_ddl(
    client: Neo4jClient,
    statement: str,
    result: SchemaApplyResult,
    *,
    kind: str,
) -> None:
    try:
        client.run_write(statement.strip())
        if kind == "constraint":
            result.constraints_created += 1
        else:
            result.indexes_created += 1
    except Exception as exc:  # noqa: BLE001 - mapped to error list for reporting
        result.errors.append(f"[{kind}] {statement[:80]}: {exc}")
        logger.warning("failed to apply %s: %s", kind, exc)


__all__ = ["SchemaApplyResult", "apply_constraints_and_indexes", "load_schema"]
