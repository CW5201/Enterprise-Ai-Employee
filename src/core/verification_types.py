"""Claim / Evidence / Verification data models (Phase 4, RQ2).

This module is the *single source of truth* for the claim-evidence
verification layer (Innovation 3).  It defines three Pydantic models:

- :class:`Claim` — one checkable statement extracted from a generated
  answer, with a typed ``claim_type`` and importance.
- :class:`Evidence` — one evidence record *from a real source*
  (RAG / SQL / KG / Analysis), with full provenance so a claim can
  always be traced back to *which* query / document / template
  produced it.  Evidence is never LLM-generated content.
- :class:`VerificationResult` — the verdict for one claim against a
  candidate-evidence set.

Design rules (docs/RESEARCH.md RQ2, ADR discipline carried from Phase 3):

- Evidence ``source_ref`` / ``provenance`` are stable and *inspectable* —
  SQL evidence keeps the executed query, KG evidence keeps the template +
  parameters, RAG keeps doc/chunk ids, analysis keeps the input evidence.
- No secret material (passwords, URIs with credentials, API keys) may ever
  be stored in these models; :func:`assert_no_secrets` enforces a static
  guard on ``model_dump`` boundaries.
- ``VerificationResult`` is self-describing (``verifier_type`` +
  ``reason``) so the evaluation harness and ablation can inspect exactly
  *which* layer produced a verdict.

The AgentState channels ``claims`` / ``evidence_v4`` /
``verification_results`` (see :mod:`src.core.state`) use append reducers
so nodes add items without clobbering earlier ones.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Claim model
# ---------------------------------------------------------------------------

ClaimType = Literal[
    "factual",          # a single factual assertion ("客户 A 的名字是…")
    "numerical",        # a numeric value or count ("本月销售额为 120000")
    "relational",       # an entity relationship ("客户 A 购买过商品 B")
    "rule_based",       # a policy / rule statement ("差旅报销上限是 800 元")
    "derived",          # computed from other values ("环比增长 20%")
    "opinion_or_summary",  # summarising / evaluative statement (low checkability)
]

CLAIM_TYPES: tuple[str, ...] = ("factual", "numerical", "relational", "rule_based", "derived", "opinion_or_summary")

Importance = Literal["critical", "normal", "minor"]

IMPORTANCE_LEVELS: tuple[str, ...] = ("critical", "normal", "minor")


class Claim(BaseModel):
    """One checkable statement extracted from a generated answer."""

    claim_id: str
    text: str = Field(min_length=1)
    claim_type: ClaimType = "factual"
    importance: Importance = "normal"
    # source_refs: explicit evidence ids / refs this claim was written
    # against when extracted (may be empty -> resolved by linking later).
    source_refs: list[str] = Field(default_factory=list)
    # For ``derived`` claims: the numeric inputs and the formula, so the
    # verifier can *recompute* instead of trusting the LLM.
    derived_from: list[str] = Field(default_factory=list)
    formula: str = ""
    # Entities / values the claim mentions (for linking + exact checks).
    entities: list[str] = Field(default_factory=list)
    value: Any = None  # the asserted numeric / scalar value, if parseable

    @field_validator("claim_type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        if v not in CLAIM_TYPES:
            raise ValueError(f"unknown claim_type {v!r}; allowed: {CLAIM_TYPES}")
        return v

    @field_validator("importance")
    @classmethod
    def _check_importance(cls, v: str) -> str:
        if v not in IMPORTANCE_LEVELS:
            raise ValueError(f"unknown importance {v!r}; allowed: {IMPORTANCE_LEVELS}")
        return v

    def is_checkable(self) -> bool:
        """opinion_or_summary claims are tracked but not hard-checked."""
        return self.claim_type != "opinion_or_summary"


# ---------------------------------------------------------------------------
# Evidence model
# ---------------------------------------------------------------------------

EvidenceSourceType = Literal["rag", "sql", "kg", "analysis"]

EVIDENCE_SOURCE_TYPES: tuple[str, ...] = ("rag", "sql", "kg", "analysis")

# Legacy source_type strings used by pre-Phase-4 EvidenceItem (state.py)
# are normalised onto the Phase 4 vocabulary when evidence is collected.
_LEGACY_SOURCE_MAP = {
    "duckdb": "sql",
    "milvus": "rag",
    "fake": "rag",
    "neo4j": "kg",
}


class Evidence(BaseModel):
    """One evidence record from a *real* tool source (never LLM text)."""

    evidence_id: str
    source_type: EvidenceSourceType
    source_ref: str = Field(description="Stable, inspectable reference (see provenance rules)")
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Provenance: *how* this value was produced.  Per source_type:
    #   rag      -> {"doc_id": ..., "chunk_id": ..., "title": ...}
    #   sql      -> {"database": ..., "table": ..., "query": ...}
    #   kg       -> {"entities": [...], "relation": ..., "template_id": ...}
    #   analysis -> {"operation": ..., "input_evidence_ids": [...]}
    provenance: dict[str, Any] = Field(default_factory=dict)
    retrieved_score: float | None = Field(default=None, description="Retrieval score when applicable (RAG)")
    structured_value: Any = Field(default=None, description="The scalar/tabular value the evidence asserts")

    @field_validator("source_type")
    @classmethod
    def _check_source_type(cls, v: str) -> str:
        if v not in EVIDENCE_SOURCE_TYPES:
            raise ValueError(f"unknown source_type {v!r}; allowed: {EVIDENCE_SOURCE_TYPES}")
        return v

    def normalize_legacy_source_type(self, raw_type: str) -> str:
        """Map pre-Phase-4 source_type strings (duckdb/milvus/fake/neo4j)."""
        return _LEGACY_SOURCE_MAP.get(raw_type, raw_type)


def normalize_source_type(raw_type: str) -> str:
    """Public helper: legacy source_type -> Phase 4 vocabulary."""
    return _LEGACY_SOURCE_MAP.get(raw_type, raw_type)


# ---------------------------------------------------------------------------
# Verification result model
# ---------------------------------------------------------------------------

SupportStatus = Literal["supported", "unsupported", "conflict"]

SUPPORT_STATUSES: tuple[str, ...] = ("supported", "unsupported", "conflict")

VerifierType = Literal["exact", "rule", "semantic", "derived", "none"]

VERIFIER_TYPES: tuple[str, ...] = ("exact", "rule", "semantic", "derived", "none")


class VerificationResult(BaseModel):
    """Verdict for one claim against its candidate evidence set."""

    claim_id: str
    claim_type: ClaimType = "factual"
    supported: bool
    support_score: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    conflict: bool = False
    status: SupportStatus = "unsupported"
    verifier_type: VerifierType = "none"
    reason: str = ""
    # For derived claims: the recomputed value vs. the asserted value.
    computed_value: Any = None
    claimed_value: Any = None

    @model_validator(mode="after")
    def _check_consistency(self) -> VerificationResult:
        if self.conflict and (self.status != "conflict" or self.supported):
                raise ValueError("conflict=True requires status='conflict' and supported=False")
        if self.supported and self.status != "supported":
            raise ValueError("supported=True requires status='supported'")
        if not self.supported and self.status == "supported":
            raise ValueError("supported=False requires status in {unsupported, conflict}")
        return self


# ---------------------------------------------------------------------------
# Secret guard — these models must never carry credentials
# ---------------------------------------------------------------------------

_SECRET_PATTERNS = [
    re.compile(r"(?i)password\s*[:=]\s*['\"]?(\S{6,})"),
    re.compile(r"(?i)(api[_-]?key|token|secret)\s*[:=]\s*['\"]?([A-Za-z0-9_\-]{16,})"),
    re.compile(r"(?i)(mysql|postgres|mongodb|neo4j|bolt|redis)://[^\s:@/]+:[^\s/@]+@"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
]


def assert_no_secrets(payload: Any) -> None:
    """Raise ValueError if ``payload`` looks like it embeds credentials.

    Call at every boundary where Claim / Evidence / VerificationResult
    are serialised (API response, state dump, eval artifact).
    """
    text = str(payload)
    for pat in _SECRET_PATTERNS:
        if pat.search(text):
            raise ValueError(
                "verification payload appears to contain secret material "
                f"(pattern {pat.pattern!r}); refusing to serialise"
            )


__all__ = [
    "Claim",
    "ClaimType",
    "CLAIM_TYPES",
    "Importance",
    "IMPORTANCE_LEVELS",
    "Evidence",
    "EvidenceSourceType",
    "EVIDENCE_SOURCE_TYPES",
    "VerificationResult",
    "SupportStatus",
    "SUPPORT_STATUSES",
    "VerifierType",
    "VERIFIER_TYPES",
    "normalize_source_type",
    "assert_no_secrets",
]
