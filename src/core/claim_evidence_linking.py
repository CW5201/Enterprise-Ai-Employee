"""Claim -> candidate-evidence linking (Phase 4, first version).

A simple, *explainable* mechanism — deliberately NOT graph reasoning
(spec: "第一版不要做复杂知识图谱推理").  Each claim is matched to a
candidate-evidence set by the first rule that fires, in order:

1. **explicit source_refs** — the claim names evidence ids / refs that
   exist in the pool; those are the candidates.
2. **entity / value overlap** — claim ``entities`` + the asserted
   ``value`` appear in an evidence record's content / provenance /
   structured_value; source-type compatibility is checked.
3. **lexical / semantic similarity** — token-overlap (Jaccard) between
   claim text and evidence content, with the top matches returned;
   the LLM semantic verifier (Layer 3) does the final judgment on the
   candidates returned here.
4. **source-type compatibility** — as a tie-breaker, evidence whose
   ``source_type`` matches the claim's natural source wins.

The output is ``{claim_id: [evidence_id, ...]}`` plus a per-claim
``match_reason`` so the verifier's audit trail explains *why* each
candidate was chosen.
"""

from __future__ import annotations

import re
from typing import Any

from src.core.verification_types import Evidence

__all__ = ["link_claims_to_evidence", "candidate_evidence"]


_TOKEN_RE = re.compile(r"[A-Za-z0-9_一-鿿]+")
# claim types that naturally come from a structured source
_TYPE_SOURCE_PREF: dict[str, tuple[str, ...]] = {
    "numerical": ("sql", "analysis"),
    "derived": ("analysis", "sql"),
    "relational": ("kg", "sql"),
    "rule_based": ("rag",),
    "factual": ("sql", "rag", "kg"),
    "opinion_or_summary": ("rag",),
}


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _evidence_text(ev: Evidence) -> str:
    """Concatenate everything searchable on an evidence record."""
    parts = [ev.content or ""]
    prov = ev.provenance or {}
    for v in prov.values():
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, (int, float)):
            parts.append(str(v))
    if ev.structured_value is not None:
        parts.append(str(ev.structured_value))
    return " ".join(parts)


def _score_explicit_refs(claim: dict[str, Any], pool: list[Evidence]) -> tuple[list[str], str]:
    refs = list(claim.get("source_refs") or [])
    if not refs:
        return [], ""
    by_ref = {e.evidence_id: e for e in pool}
    by_ref.update({e.source_ref: e for e in pool})
    matched = [by_ref[r].evidence_id for r in refs if r in by_ref]
    return matched, "explicit_source_refs"


def _score_entity_overlap(
    claim: dict[str, Any], ev: Evidence
) -> tuple[bool, str]:
    entities = [str(e) for e in (claim.get("entities") or [])]
    value = claim.get("value")
    needles: list[str] = []
    needles += entities
    if value is not None:
        needles.append(str(value))
    text = _evidence_text(ev)
    hits = [n for n in needles if n and n in text]
    if hits:
        return True, f"entity_overlap:{'+'.join(hits)}"
    return False, ""


def _score_similarity(claim: dict[str, Any], ev: Evidence) -> float:
    return _jaccard(_tokens(str(claim.get("text", ""))), _tokens(_evidence_text(ev)))


def _source_compatible(claim_type: str, ev: Evidence) -> bool:
    prefs = _TYPE_SOURCE_PREF.get(claim_type)
    if not prefs:
        return True
    return ev.source_type in prefs


def candidate_evidence(
    claim: dict[str, Any],
    pool: list[Evidence],
    *,
    max_candidates: int = 5,
    min_similarity: float = 0.1,
) -> tuple[list[str], str]:
    """Return (evidence_ids, match_reason) for one claim."""
    if not pool:
        return [], "no_evidence"

    explicit, reason = _score_explicit_refs(claim, pool)
    if explicit:
        return explicit[:max_candidates], reason

    claim_type = str(claim.get("claim_type", "factual"))

    # entity/value overlap, restricted to compatible sources when possible
    overlap_ids: list[str] = []
    for ev in pool:
        ok, _ = _score_entity_overlap(claim, ev)
        if ok and _source_compatible(claim_type, ev):
            overlap_ids.append(ev.evidence_id)
    if overlap_ids:
        return overlap_ids[:max_candidates], "entity_overlap"

    # lexical / semantic similarity, source-compatible first
    scored = sorted(
        ((_score_similarity(claim, ev), ev.evidence_id, _source_compatible(claim_type, ev)) for ev in pool),
        key=lambda t: (-t[2], -t[0], t[1]),
    )
    hits = [eid for sim, eid, compat in scored if sim >= min_similarity][:max_candidates]
    if hits:
        return hits, "lexical_similarity"
    return [], "below_similarity_threshold"


def link_claims_to_evidence(
    claims: list[dict[str, Any]],
    pool: list[Evidence],
    *,
    max_candidates: int = 5,
) -> dict[str, dict[str, Any]]:
    """Link every claim to its candidate evidence.  Returns
    ``{claim_id: {"evidence_ids": [...], "match_reason": str}}``."""
    out: dict[str, dict[str, Any]] = {}
    for claim in claims:
        ids, reason = candidate_evidence(
            claim, pool, max_candidates=max_candidates
        )
        out[str(claim.get("claim_id", ""))] = {
            "evidence_ids": ids,
            "match_reason": reason,
        }
    return out
