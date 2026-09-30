"""Claim -> candidate-evidence linking (Phase 4, first version).

A simple, *explainable* mechanism — deliberately NOT graph reasoning
(spec: "第一版不要做复杂知识图谱推理").  Each claim is matched to a
candidate-evidence set by the first decisive rule, in order:

1. **explicit source_refs** — the claim names evidence ids / refs that
   exist in the pool; those are the candidates.
2. **entity / value overlap** — claim ``entities`` + the asserted
   ``value`` appear in an evidence record's content / provenance /
   structured_value; restricted to source-type-compatible records.
3. **lexical / semantic similarity** — token-overlap (Jaccard) between
   claim text and evidence content, source-compatible first; the LLM
   semantic verifier (Layer 3) does the final judgment on these.
4. **source-type pool** (final fallback): numeric / derived claims get
   the full structured-source (sql / analysis) pool so the verifier can
   surface *conflicts* across sources; every other claim type gets the
   source-compatible pool.

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
    """Legacy helper kept for direct use; ``candidate_evidence`` no longer
    calls it (the rules are inlined in that function now)."""
    refs = list(claim.get("source_refs") or [])
    if refs:
        by_ref = {e.evidence_id: e for e in pool}
        by_ref.update({e.source_ref: e for e in pool})
        matched = [by_ref[r].evidence_id for r in refs if r in by_ref]
        if matched:
            return matched[:10], "explicit_source_refs"
    return [], "no_explicit_refs"


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
    """Return (evidence_ids, match_reason) for one claim.

    Order of rules (first decisive rule wins):
    1. explicit ``source_refs`` (when they match ids in the pool)
    2. entity / value overlap, restricted to source-compatible records
    3. lexical / semantic similarity, source-compatible first
    4. source-type-compatible pool (numeric/derived claims get the full
       structured-source pool so the verifier can check conflicts)
    """
    if not pool:
        return [], "no_evidence"

    claim_type = str(claim.get("claim_type", "factual"))

    # 1. explicit source refs — when the refs actually resolve against
    #    the pool, they are authoritative: the linker does NOT mix in
    #    other candidates (that is the whole point of explicit refs).
    refs = list(claim.get("source_refs") or [])
    if refs:
        by_ref: dict[str, Evidence] = {e.evidence_id: e for e in pool}
        by_ref.update({e.source_ref: e for e in pool})
        matched = [by_ref[r].evidence_id for r in refs if r in by_ref]
        if matched:
            return matched[:max_candidates], "explicit_source_refs"
        # refs present but none matched: the claim cites evidence that
        # does not exist in this run.  A numeric/derived claim still
        # falls back to the structured pool so a real conflict (if any)
        # is surfaced rather than silently unsupported.
        if claim_type in ("numerical", "derived"):
            out = [e.evidence_id for e in pool if e.source_type in ("sql", "analysis")]
            if out:
                return out[:max_candidates], "type_compat_all"
            return [e.evidence_id for e in pool][:max_candidates], "no_strict_match"
        return [], "no_explicit_refs_matched"

    # 2. entity / value overlap on compatible sources.  For numeric /
    #    derived claims a single overlapping candidate is not enough to
    #    *decide*: other structured sources may contradict it.  So we
    #    return the overlap hit AND the full structured pool (deduped)
    #    and let the exact layer flag a cross-source conflict.
    overlap_ids: list[str] = []
    for ev in pool:
        ok, _ = _score_entity_overlap(claim, ev)
        if ok and _source_compatible(claim_type, ev):
            overlap_ids.append(ev.evidence_id)
    if overlap_ids:
        if claim_type in ("numerical", "derived"):
            structured = [e.evidence_id for e in pool if e.source_type in ("sql", "analysis")]
            combined = list(dict.fromkeys(overlap_ids + structured))
            if structured:
                return combined[:max_candidates], "entity_overlap_plus_structured"
        return overlap_ids[:max_candidates], "entity_overlap"

    # 3. lexical / semantic similarity, source-compatible first
    scored = sorted(
        ((_score_similarity(claim, ev), ev.evidence_id, _source_compatible(claim_type, ev)) for ev in pool),
        key=lambda t: (-t[2], -t[0], t[1]),
    )
    hits = [eid for sim, eid, compat in scored if sim >= min_similarity][:max_candidates]
    if hits:
        return hits, "lexical_similarity"

    # 4. source-type-compatible pool (final fallback).  For numeric /
    #    derived claims we surface the full structured-source pool so the
    #    verifier can detect cross-source conflicts; for others we keep
    #    only source-compatible records.
    if claim_type in ("numerical", "derived"):
        out = [e.evidence_id for e in pool if e.source_type in ("sql", "analysis")]
        if out:
            return out[:max_candidates], "type_compat_all"
        # no structured source in the pool at all — hand over the whole
        # pool so the semantic layer can judge (a numeric claim linked to
        # nothing can never be supported, so we must not claim it is).
        return [e.evidence_id for e in pool][:max_candidates], "no_strict_match"
    out = [e.evidence_id for e in pool if _source_compatible(claim_type, e)]
    if out:
        return out[:max_candidates], "type_compat_all"
    # Nothing source-compatible in the pool: an empty candidate set would
    # let the claim fall through to a "no candidate evidence" verdict
    # before the verifier ever reads the content.  Hand the full pool
    # over so the semantic layer can judge on similarity.
    return [e.evidence_id for e in pool][:max_candidates], "no_strict_match"


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
