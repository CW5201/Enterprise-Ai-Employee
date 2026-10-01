"""API route: graph - Neo4j entity / relation queries.

Phase 6.5: read-only queries against the enterprise knowledge graph.
Neo4j access goes through the existing Neo4jClient (src/core/neo4j_client.py);
if Neo4j is unavailable (no NEO4J_PASSWORD, no server, ...) the endpoints
return empty lists rather than errors, so the frontend can show a correct
empty state.  Credentials never leave the process; errors are logged with
credential-safe details only.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from fastapi import APIRouter, Query

router = APIRouter(tags=["graph"])
logger = logging.getLogger("eae.api.graph")

# One shared Neo4jClient (and therefore one underlying driver + connection
# pool) for the whole process, guarded by a lock so concurrent requests do
# not race on first construction.  Creating a fresh GraphDatabase.driver()
# per request would leak connection pools without bound.
_shared_client: Any | None = None
_shared_client_lock = threading.Lock()


def _neo4j_client() -> Any | None:
    """Return the process-wide Neo4jClient, or None when offline.

    ``Neo4jClient()`` is lazy — constructing it never fails.  When
    ``NEO4J_PASSWORD`` is unset (the normal local-dev / CI case) or the
    server is unreachable, connecting raises :class:`Neo4jUnavailableError`;
    in that case the endpoints return empty lists instead of crashing.

    The client is memoised per process so the neo4j driver's connection pool
    is created exactly once and shared by all requests (no per-request leak).
    """
    global _shared_client
    with _shared_client_lock:
        if _shared_client is None:
            try:
                from src.core.neo4j_client import Neo4jClient, Neo4jUnavailableError

                candidate = Neo4jClient()
                candidate.get_driver()  # force connect so offline shows up here
                _shared_client = candidate
                logger.info("shared Neo4j client initialised for %s", candidate.uri)
            except Neo4jUnavailableError as exc:
                logger.warning("Neo4j offline (%s); returning empty result", exc.message)
                return None
            except Exception:  # noqa: BLE001
                logger.warning("Neo4j client unavailable; returning empty result")
                return None
    return _shared_client


def _query_entities(client: Any, search: str | None, limit: int) -> list[dict[str, Any]]:
    if client is None:
        return []
    if search:
        cypher = (
            "MATCH (n) WHERE n.name CONTAINS $search "
            "RETURN n LIMIT $limit"
        )
        params: dict[str, Any] = {"search": search, "limit": limit}
    else:
        cypher = "MATCH (n) RETURN n LIMIT $limit"
        params = {"limit": limit}

    try:
        result = client.execute_readonly(cypher, parameters=params)
    except Exception:  # noqa: BLE001
        logger.exception("graph entity query failed")
        return []

    entities: list[dict[str, Any]] = []
    for row in result.records:
        node = row.get("n")
        if node is None:
            continue
        try:
            name = node.get("name", "")
            label = next(iter(node.labels), "")
        except AttributeError:  # driver not installed / non-Node value
            continue
        entities.append({
            "id": str(node.get("id", "")),
            "name": name,
            "type": label,
            "properties": {k: v for k, v in dict(node).items() if k not in ("id", "name")},
        })
    return entities


def _query_relations(client: Any, entity_name: str | None, limit: int) -> list[dict[str, Any]]:
    if client is None:
        return []
    if entity_name:
        cypher = (
            "MATCH (a {name: $name})-[r]->(b) "
            "RETURN a.name AS source_name, type(r) AS rel_type, b.name AS target_name "
            "LIMIT $limit"
        )
        params: dict[str, Any] = {"name": entity_name, "limit": limit}
    else:
        cypher = (
            "MATCH (a)-[r]->(b) "
            "RETURN a.name AS source_name, type(r) AS rel_type, b.name AS target_name "
            "LIMIT $limit"
        )
        params = {"limit": limit}

    try:
        result = client.execute_readonly(cypher, parameters=params)
    except Exception:  # noqa: BLE001
        logger.exception("graph relation query failed")
        return []

    relations: list[dict[str, Any]] = []
    for row in result.records:
        relations.append({
            "source": row.get("source_name", ""),
            "type": row.get("rel_type", ""),
            "target": row.get("target_name", ""),
        })
    return relations


@router.get("/graph/entities")
def list_entities(
    search: str | None = Query(default=None),
    limit: int = Query(default=50, le=200),
) -> list[dict[str, Any]]:
    client = _neo4j_client()
    return _query_entities(client, search, limit)


@router.get("/graph/relations")
def list_relations(
    entity: str | None = Query(default=None, description="Filter relations by entity name"),
    limit: int = Query(default=50, le=200),
) -> list[dict[str, Any]]:
    client = _neo4j_client()
    return _query_relations(client, entity, limit)


__all__ = ["router"]
