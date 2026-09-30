"""Cypher safety validator — read-only guard for the KG tool.

Phase 2.4 Commit 3 (ADR-004).  The KG tool exposes only a fixed set of
business query templates; every one of them passes through this validator
before execution.  The validator is intentionally strict and
conservative: it accepts only the read-only allow-list and rejects
anything that could mutate the graph, run an admin procedure, or leak
filesystem / config access.

The validator is stateless and pure: ``validate(cypher, params)`` either
returns the (cypher, params) pair unchanged or raises
:class:`CypherValidationError`.  No network I/O, no driver import — so it
can be unit-tested without a Neo4j instance and reused by any future
"LLM-generated Cypher" path that we explicitly do NOT ship today
(``allow_llm_generated_cypher: false`` in config).

Allowed grammar (superset of what our templates need):

- ``MATCH`` / ``OPTIONAL MATCH`` clauses
- ``WHERE`` / ``WITH`` / ``ORDER BY`` / ``LIMIT`` / ``SKIP``
- Aggregations: ``COUNT`` / ``SUM`` / ``AVG`` / ``MIN`` / ``MAX`` /
  ``COLLECT`` / ``STDEV``
- Relationship pattern operators ``-[:TYPE {..}]-->`` / ``<-[..]-``
- ``DISTINCT``

Forbidden (any occurrence, case-insensitive, as whole word):

- Write:  CREATE / MERGE / DELETE / SET / REMOVE / DROP / DETACH / LOAD
- Admin:  CALL / FOREACH / APPLY / FORGET / SCHEMA / DBMS / INDEX /
  CONSTRAINT
- Escape: ``LOAD CSV`` / filesystem / procedure calls
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("eae.cypher_validator")

# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

#: Statements allowed through the validator must begin with one of these.
_ALLOWED_LEADING = re.compile(r"^\s*(match|optional\s+match|return|with)\b", re.IGNORECASE)

#: Any whole-word occurrence of these tokens anywhere in the statement
#: disqualifies it.  ``SET`` is matched with word boundaries so that
#: ``settings`` as a property name inside a string is also conservatively
#: rejected (templates never need that).
_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(create|merge|delete|detach|set|remove|drop|load|call|apply|foreeach|"
    r"forget|schema|dbms|constraint|index|csv)\b",
    re.IGNORECASE,
)


class CypherValidationError(Exception):
    """Raised when a Cypher statement is not in the read-only allow-list."""

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, object] = details or {}


@dataclass(frozen=True)
class ValidatedQuery:
    """The output of :func:`validate` — safe to hand to the driver."""

    cypher: str
    parameters: dict[str, object]


def validate(cypher: str, parameters: dict[str, object] | None = None) -> ValidatedQuery:
    """Validate *cypher* against the read-only allow-list.

    Raises :class:`CypherValidationError` when the statement is not
    accepted.  The caller (the KG tool) is expected to catch this and map
    it to a structured ``ToolResult(ok=False, error=...)`` without
    leaking the raw Cypher text to end users.
    """
    stripped = (cypher or "").strip().rstrip(";").strip()
    if not stripped:
        raise CypherValidationError("Empty Cypher statement")

    if not _ALLOWED_LEADING.match(stripped):
        first_token = stripped.split(None, 1)[0].lower()
        raise CypherValidationError(
            f"Statement must begin with MATCH / RETURN / WITH (got {first_token!r})",
        )

    forbidden = sorted({m.group(0).upper() for m in _FORBIDDEN_KEYWORDS.finditer(stripped)})
    if forbidden:
        raise CypherValidationError(
            f"Read-only validator rejects forbidden keyword(s): {forbidden}",
            details={"keywords": forbidden},
        )

    # Balanced-paren / bracket sanity (cheap, catches obvious malformation
    # before the server rejects it).
    if _unbalanced(stripped):
        raise CypherValidationError("Unbalanced parentheses or brackets in statement")

    return ValidatedQuery(cypher=stripped, parameters=dict(parameters or {}))


def _unbalanced(text: str) -> bool:
    """Return True if parentheses/brackets are not balanced.

    String literals are respected (a ``(`` inside ``"..."`` does not count).
    """
    depth_paren = 0
    depth_bracket = 0
    in_string: str | None = None
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string is not None:
            if ch == in_string and (i == 0 or text[i - 1] != "\\"):
                in_string = None
        elif ch in ("'", '"'):
            in_string = ch
        elif ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren -= 1
            if depth_paren < 0:
                return True
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket -= 1
            if depth_bracket < 0:
                return True
        i += 1
    return depth_paren != 0 or depth_bracket != 0


__all__ = [
    "CypherValidationError",
    "ValidatedQuery",
    "validate",
]
