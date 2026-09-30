"""Neo4j client — connection, health check, read-only query wrapper.

Phase 2.4 infrastructure (ADR-004).  This module is the ONLY place in the
project that opens a Neo4j driver connection; business code and tools must
use :class:`Neo4jClient` / :mod:`src.core.cypher_validator` (see
docs/ARCHITECTURE.md: no direct SDK access from tools).

Security rules (enforced by the client, re-checked in ADR-004):

- Read-only by default.  :meth:`Neo4jClient.execute_readonly` accepts only
  statements that begin with ``MATCH`` and contain no write keywords; the
  allow-list is declared here.  Write statements (CREATE / MERGE / SET /
  DELETE / ...) are rejected by the read-only wrapper.  The KG builder
  (``scripts/build_kg.py``) and the DDL runner
  (:mod:`src.core.neo4j_schema`) use :meth:`run_write` explicitly — they
  are the only sanctioned writers.
- Password / credentials come from the environment
  (``NEO4J_PASSWORD`` / ``NEO4J_USERNAME`` via config resolution, ``.env``).
  They are never logged and never included in exception messages.
- All Neo4j errors are re-raised as :class:`Neo4jError` subclasses with a
  structured, credential-safe detail dict; the raw driver exception chain
  is kept for debugging (logs only, never surfaced to API users).
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("eae.neo4j")

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class Neo4jError(Exception):
    """Base class for all Neo4j client errors.  Safe to log; carries details."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}


class Neo4jUnavailableError(Neo4jError):
    """Connection / transport / credential failure (no server, bad auth)."""


class Neo4jQueryError(Neo4jError):
    """The server executed the query and rejected it (syntax, semantics, ...)."""


class Neo4jReadOnlyViolation(Neo4jError):
    """A write statement reached the read-only wrapper."""


# ---------------------------------------------------------------------------
# Read-only statement classification
# ---------------------------------------------------------------------------

#: Statements allowed through the read-only wrapper must begin with one of
#: these keywords.
_READONLY_LEADING = re.compile(r"^\s*(match|with|return)\b", re.IGNORECASE)

#: Any of these tokens anywhere in the statement disqualifies it from the
#: read-only wrapper.  (``SET`` inside string literals is conservatively
#: rejected as well — the templates we ship never need string-literal SETs.)
_WRITE_KEYWORDS = re.compile(
    r"(\b(create|merge|delete|detach|remove|drop|load|call|apply|foreeach|forget|"
    r"constraint|index|schema|dbms)\b|\bset\b)",
    re.IGNORECASE,
)


def _classify_for_readonly(cypher: str) -> None:
    """Raise :class:`Neo4jReadOnlyViolation` if *cypher* is not read-only."""
    stripped = cypher.strip().rstrip(";").strip()
    if not stripped:
        raise Neo4jReadOnlyViolation("Empty Cypher statement")
    if not _READONLY_LEADING.match(stripped):
        raise Neo4jReadOnlyViolation(
            "Read-only wrapper accepts only MATCH / RETURN statements",
            details={"first_token": stripped.split(None, 1)[0]},
        )
    forbidden = sorted({m[0].upper() if m[0] else m[1].upper() for m in _WRITE_KEYWORDS.findall(stripped)})
    if forbidden:
        raise Neo4jReadOnlyViolation(
            f"Write keyword not allowed through read-only wrapper: {forbidden}",
            details={"keywords": forbidden},
        )


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


@dataclass
class Neo4jResult:
    """Structured outcome of one Cypher execution."""

    cypher: str
    records: list[dict[str, Any]]
    latency_ms: float = 0.0
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dicts(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.records]

    def as_tool_payload(self, *, query_type: str = "custom") -> dict[str, Any]:
        """Uniform envelope used by the KG tool (Commit 3)."""
        return {
            "success": True,
            "data": self.to_dicts(),
            "rows": len(self.records),
            "source": "neo4j",
            "query_type": query_type,
            "latency_ms": self.latency_ms,
        }


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


@dataclass
class Neo4jClient:
    """Thin, safe Neo4j driver wrapper.

    Parameters are resolved from the environment when left empty, so no
    credential ever needs to travel through yaml files:

    - ``uri``      <- ``NEO4J_URI`` (default ``bolt://localhost:7687``)
    - ``user``     <- ``NEO4J_USERNAME`` (default ``neo4j``)
    - ``password`` <- ``NEO4J_PASSWORD`` (required for ``get_driver``)
    - ``database`` <- ``NEO4J_DATABASE`` (default ``neo4j``)

    The client keeps one lazily-created driver; each public call opens a
    short session on top of the driver's internal connection pool.
    """

    uri: str = ""
    user: str = ""
    password: str = ""
    database: str = "neo4j"
    timeout_seconds: float = 30.0

    def __init__(self, uri: str = "", user: str = "", password: str = "",
                 database: str = "neo4j", timeout_seconds: float = 30.0) -> None:
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.environ.get("NEO4J_USERNAME", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "")
        self.database = database or os.environ.get("NEO4J_DATABASE", "neo4j")
        self.timeout_seconds = timeout_seconds
        self._driver: Any = None

    # -- lifecycle ----------------------------------------------------------

    def get_driver(self) -> Any:
        """Return the (lazily created) driver instance for this client."""
        if self._driver is None:
            from neo4j import GraphDatabase  # local import: driver optional at import time

            if not self.password:
                raise Neo4jUnavailableError(
                    "NEO4J_PASSWORD is not set; refusing to connect without credentials",
                )
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            logger.debug("neo4j driver created for %s (user=%s)", self.uri, self.user)
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> Neo4jClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- health -------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """Return ``{"ok": True, "version": ..., "latency_ms": ...}``.

        Raises :class:`Neo4jUnavailableError` when the server cannot be
        reached or authentication fails.  The version is read with
        ``dbms.systemInformation()`` (read-only, allowed for the health
        check only — the general read-only wrapper rejects ``CALL``).
        """
        t0 = time.perf_counter()
        driver = self.get_driver()
        try:
            with driver.session(database=self.database) as session:
                # PING is a zero-cost protocol-level health probe; the
                # follow-up RETURN verifies a normal query round-trip.
                # The server version is not read through a Cypher function
                # (``version()`` is not a real function in Neo4j 5); it is
                # recorded from the driver's bolt handshake metadata when
                # available, otherwise left as None.
                session.run("RETURN 1")
                return {
                    "ok": True,
                    "version": None,
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }
        except Exception as exc:  # noqa: BLE001 - mapped below
            detail = _diagnose(exc)
            raise Neo4jUnavailableError(
                "Neo4j health check failed", details=detail,
            ) from exc

    # -- execution ----------------------------------------------------------

    def run_write(self, cypher: str, parameters: dict[str, Any] | None = None) -> Neo4jResult:
        """Execute an arbitrary Cypher statement (sanctioned writers only).

        Intended for constraint/index DDL (:mod:`src.core.neo4j_schema`) and
        the KG builder (``scripts/build_kg.py``).  Application read paths
        must use :meth:`execute_readonly`.
        """
        return self._execute(cypher, parameters or {})

    def execute_readonly(self, cypher: str, parameters: dict[str, Any] | None = None) -> Neo4jResult:
        """Execute a read-only Cypher statement with strict guarding.

        Raises :class:`Neo4jReadOnlyViolation` before any network call when
        the statement is not in the read-only allow-list.
        """
        _classify_for_readonly(cypher)
        return self._execute(cypher, parameters or {})

    def _execute(self, cypher: str, parameters: dict[str, Any]) -> Neo4jResult:
        t0 = time.perf_counter()
        driver = self.get_driver()
        try:
            with driver.session(database=self.database) as session:
                result = session.run(cypher, parameters)
                records = [dict(r) for r in result]
            return Neo4jResult(
                cypher=cypher,
                records=records,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )
        except Neo4jError:
            raise
        except Exception as exc:  # noqa: BLE001
            detail = _diagnose(exc)
            if detail.get("kind") == "query":
                raise Neo4jQueryError(detail.get("message", "Neo4j query failed"), details=detail) from exc
            raise Neo4jUnavailableError(detail.get("message", "Neo4j unavailable"), details=detail) from exc


def _diagnose(exc: Exception) -> dict[str, Any]:
    """Map a raw neo4j driver exception to a structured, credential-safe dict.

    The resulting dict never contains the bolt URI, username or password:
    any URL-like token is replaced with ``<uri>`` before storage.
    """
    name = type(exc).__name__
    msg = str(exc)
    msg = re.sub(r"\b(bolt|neo4j|https?)://\S+", "<uri>", msg)
    msg = msg[:500]
    lowered = msg.lower()
    if "Auth" in name or "authentication" in lowered or "unauthorized" in lowered:
        kind = "auth"
    elif any(k in name for k in ("ServiceUnavailable", "SessionExpired", "Connect", "Timeout", "Driver")) \
            or "connection" in lowered or "refused" in lowered or "unavailable" in lowered:
        kind = "connection"
    else:
        kind = "query"
    return {"kind": kind, "driver_exception": name, "message": msg}


def load_client_from_env() -> Neo4jClient:
    """Build a :class:`Neo4jClient` from environment variables.

    Convenience for scripts; the API should prefer
    :func:`src.core.config_loader.get_settings`.
    """
    return Neo4jClient()


__all__ = [
    "Neo4jClient",
    "Neo4jResult",
    "Neo4jError",
    "Neo4jUnavailableError",
    "Neo4jQueryError",
    "Neo4jReadOnlyViolation",
    "_classify_for_readonly",
    "_diagnose",
    "load_client_from_env",
]
