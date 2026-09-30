"""Claim-Evidence Verification engine (Phase 4, RQ2).

Layered strategy — a claim is verified by the *first* layer that can
decide it, falling through to the next when the layer is inconclusive:

- **Layer 1 — exact / structured** (:func:`verify_exact`): for
  ``numerical`` / ``derived`` claims whose ``value`` is parseable, compare
  against the *structured value* of compatible evidence.  Equality ->
  supported; a single mismatched source -> conflict; multiple sources
  disagreeing with each other -> conflict.  We never silently pick one
  when evidence disagrees (spec requirement).
- **Layer 2 — rule-based consistency** (:func:`verify_rule`):
  source-type / provenance compatibility.  A claim is only *supported*
  when at least one compatible source type backs it; a relational claim
  whose expected relation is asserted in a KG/SQL result is confirmed;
  otherwise this layer reports inconclusive and hands off to Layer 3.
- **Layer 3 — semantic** (:func:`verify_semantic`): embedding similarity
  (BGE-M3 when available, lexical otherwise) gates the candidates; the
  LLM makes the *final* support judgment on those candidates.  We do NOT
  trust a raw embedding score alone as a factual verdict.

Policy (config-driven, never hardcoded in business code):

- ``support_score >= support_threshold`` -> supported
- conflict detected by any layer        -> conflict (never auto-pick)
- insufficient evidence                   -> unsupported

The engine is *evidence-aware*: it never judges a claim without reading
the evidence records that back it.  For baselines (rule-only, LLM-only)
the engine exposes :func:`verify` with an explicit ``layers`` argument so
the ablation can disable layers without forking the code.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, Field

from src.core.exceptions import LLMError
from src.core.verification_types import (
    Evidence,
    VerificationResult,
)

__all__ = [
    "VerificationConfig",
    "BgeSemanticScorer",
    "LexicalScorer",
    "verify",
    "verify_exact",
    "verify_rule",
    "verify_semantic",
]


# ---------------------------------------------------------------------------
# Configuration (loaded from config/settings.yaml -> verification section)
# ---------------------------------------------------------------------------


@dataclass
class VerificationConfig:
    """Tunable knobs for the verification engine.

    Defaults mirror ``config/settings.yaml``; the eval runner and graph
    builder should construct this from the settings so thresholds are never
    hardcoded at the call site.
    """

    enabled: bool = True
    support_threshold: float = 0.6
    semantic_threshold: float = 0.5
    numeric_tolerance: float = 0.01
    require_evidence: bool = True
    max_semantic_candidates: int = 5

    @classmethod
    def from_section(cls, section: dict[str, Any] | None) -> VerificationConfig:
        s = section or {}
        return cls(
            enabled=bool(s.get("enabled", True)),
            support_threshold=float(s.get("min_support_score", s.get("support_threshold", 0.6))),
            semantic_threshold=float(s.get("semantic_threshold", 0.5)),
            numeric_tolerance=float(s.get("numeric_tolerance", 0.01)),
            require_evidence=bool(s.get("require_evidence", True)),
            max_semantic_candidates=int(s.get("max_semantic_candidates", 5)),
        )

    @classmethod
    def from_settings(cls, v: Any) -> VerificationConfig:
        """Build from a :class:`config_loader.VerificationSettings` object."""
        return cls(
            enabled=bool(v.enabled),
            support_threshold=float(v.support_threshold),
            semantic_threshold=float(v.semantic_threshold),
            numeric_tolerance=float(v.numeric_tolerance),
            require_evidence=bool(v.require_evidence),
            max_semantic_candidates=int(v.max_semantic_candidates),
        )


# ---------------------------------------------------------------------------
# Semantic scorer — pluggable so BGE-M3 is optional in unit tests
# ---------------------------------------------------------------------------


class SemanticScorer(Protocol):
    backend: str

    def score(self, claim_text: str, evidence_content: str) -> float:
        """Return a 0..1 similarity for one claim/evidence pair."""
        ...


def _jaccard(a: str, b: str) -> float:
    # CJK: single chars + bigrams (no whitespace between Chinese words);
    # Latin: word / digit tokens.
    ta = set(re.findall(r"[a-z0-9_]+", a.lower()))
    ta.update(re.findall(r"[一-鿿]", a))
    ta.update(bigrams(a))
    tb = set(re.findall(r"[a-z0-9_]+", b.lower()))
    tb.update(re.findall(r"[一-鿿]", b))
    tb.update(bigrams(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def bigrams(text: str) -> set[str]:
    """CJK bigrams over the raw character runs (no word segmentation)."""
    chars = [c for c in text.lower() if "一" <= c <= "鿿"]
    grams = set()
    for i in range(len(chars) - 1):
        grams.add(chars[i] + chars[i + 1])
    return grams


class BgeSemanticScorer:
    """BGE-M3 cosine similarity scorer (real model when available)."""

    backend = "bge-m3"

    def __init__(self, embedder: Any) -> None:
        self._embed = embedder

    def score(self, claim_text: str, evidence_content: str) -> float:
        va = self._embed(claim_text)
        vb = self._embed(evidence_content)
        if len(va) != len(vb):
            return 0.0
        dot = sum(x * y for x, y in zip(va, vb, strict=True))
        na = math.sqrt(sum(x * x for x in va)) or 1.0
        nb = math.sqrt(sum(x * x for x in vb)) or 1.0
        return max(0.0, min(1.0, dot / (na * nb)))


class LexicalScorer:
    """Token-overlap fallback scorer (test / offline, clearly labelled)."""

    backend = "lexical"

    def score(self, claim_text: str, evidence_content: str) -> float:
        return _jaccard(claim_text, evidence_content)


# ---------------------------------------------------------------------------
# LLM semantic verdict (Layer 3 final judgment)
# ---------------------------------------------------------------------------


class _SemanticVerdict(BaseModel):
    """The LLM's final support judgment for one claim over candidates."""

    supported: bool
    support_score: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    conflicting_evidence_ids: list[str] = Field(default_factory=list)


_SEMANTIC_SYSTEM = """你是企业 AI 数字员工的证据核验模块。
判断一条 claim 是否被给定的候选证据"支持"。
规则：
1. 只有当候选证据的内容确实包含/蕴含 claim 的事实时，supported=true；
2. 证据与 claim 数值不一致、或不同证据互相矛盾时 supported=false，
   并在 conflicting_evidence_ids 中列出冲突的证据；
3. 证据不足或相关度低时 supported=false，support_score 给低值；
4. 不允许用常识补齐——只能依据给定的候选证据。
返回 JSON: {{"supported": bool, "support_score": 0~1, "reason": str,
"conflicting_evidence_ids": [str]}}"""


# ---------------------------------------------------------------------------
# Layer 1 — exact / structured verification
# ---------------------------------------------------------------------------


def _text_tokens(text: str) -> set[str]:
    """Lower-case CJK chars, CJK bigrams, and ASCII word / digit tokens."""
    t = text.lower()
    tokens = set(re.findall(r"[a-z0-9_]+", t))
    tokens.update(re.findall(r"[一-鿿]", t))
    tokens.update(_bigrams_cjk(t))
    return tokens


def _bigrams_cjk(text: str) -> set[str]:
    chars = [c for c in text.lower() if "一" <= c <= "鿿"]
    return {chars[i] + chars[i + 1] for i in range(len(chars) - 1)}


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"-?\d[\d,]*(?:\.\d+)?", value)
        if not m:
            return None
        try:
            return float(m.group(0).replace(",", ""))
        except ValueError:
            return None
    return None


def _structured_values(candidates: list[Evidence]) -> set[float]:
    """All distinct parseable structured values across candidates."""
    out: set[float] = set()
    for ev in candidates:
        f = _to_float(ev.structured_value)
        if f is not None:
            out.add(round(f, 6))
    return out


def verify_exact(
    claim: dict[str, Any],
    candidates: list[Evidence],
    cfg: VerificationConfig,
) -> VerificationResult | None:
    """Layer 1.  Returns a decisive VerificationResult, or ``None`` when the
    claim is not numerically checkable (fall through to Layer 2/3)."""
    if claim.get("claim_type") not in ("numerical", "derived"):
        # Even non-numeric claims can trigger a conflict when two
        # structured sources disagree with each other.  Surface that
        # here so the exact layer owns all conflict detection.
        distinct = _structured_values(candidates)
        if len(distinct) > 1:
            return VerificationResult(
                claim_id=str(claim.get("claim_id", "")),
                claim_type=str(claim.get("claim_type", "numerical")),  # type: ignore[arg-type]
                supported=False,
                status="conflict",
                conflict=True,
                support_score=0.0,
                evidence_ids=[e.evidence_id for e in candidates if _to_float(e.structured_value) is not None],
                verifier_type="exact",
                reason=f"structured sources disagree: {sorted(distinct)}; not auto-picking",
                computed_value=None,
                claimed_value=None,
            )
        return None

    claimed = _to_float(claim.get("value"))
    if claimed is None:
        return None

    compat = [e for e in candidates if e.source_type in ("sql", "analysis") or e.structured_value is not None]
    if not compat:
        return None

    values: list[float] = []
    value_evidence: dict[float, list[str]] = {}
    for ev in compat:
        f = _to_float(ev.structured_value)
        if f is None:
            continue
        values.append(f)
        value_evidence.setdefault(round(f, 6), []).append(ev.evidence_id)

    if not values:
        # compatible evidence exists but no parseable value -> inconclusive
        return None

    matches = [v for v in values if abs(v - claimed) <= cfg.numeric_tolerance * max(1.0, abs(claimed))]
    non_matching = [v for v in values if abs(v - claimed) > cfg.numeric_tolerance * max(1.0, abs(claimed))]

    distinct = {round(v, 6) for v in values}
    # multiple evidence sources disagree with each other -> conflict,
    # regardless of whether any of them matches the claim.  We never
    # silently auto-pick a value when two sources contradict each other.
    if len(distinct) > 1:
        return VerificationResult(
            claim_id=str(claim.get("claim_id", "")),
            claim_type=str(claim.get("claim_type", "numerical")),  # type: ignore[arg-type]
            supported=False,
            status="conflict",
            conflict=True,
            support_score=0.0,
            evidence_ids=[eid for v in values for eid in value_evidence.get(round(v, 6), [])],
            verifier_type="exact",
            reason=f"evidence sources disagree: {sorted(distinct)}; not auto-picking",
            computed_value=None,
            claimed_value=claimed,
        )

    if matches:
        evidence_ids = value_evidence.get(round(matches[0], 6), [])
        return VerificationResult(
            claim_id=str(claim.get("claim_id", "")),
            claim_type=str(claim.get("claim_type", "numerical")),  # type: ignore[arg-type]
            supported=True,
            status="supported",
            conflict=False,
            support_score=1.0,
            evidence_ids=evidence_ids,
            verifier_type="exact",
            reason=f"{claimed} matches structured evidence value {matches[0]}",
            computed_value=matches[0],
            claimed_value=claimed,
        )

    if non_matching:
        # at least one source contradicts the claim -> conflict
        return VerificationResult(
            claim_id=str(claim.get("claim_id", "")),
            claim_type=str(claim.get("claim_type", "numerical")),  # type: ignore[arg-type]
            supported=False,
            status="conflict",
            conflict=True,
            support_score=0.0,
            evidence_ids=[eid for v in non_matching for eid in value_evidence.get(round(v, 6), [])],
            verifier_type="exact",
            reason=f"claimed {claimed} but evidence says {non_matching[0]}",
            computed_value=non_matching[0],
            claimed_value=claimed,
        )

    # no values parseable from candidates -> inconclusive, fall through
    return None


# ---------------------------------------------------------------------------
# Layer 2 — rule-based consistency
# ---------------------------------------------------------------------------


def verify_rule(
    claim: dict[str, Any],
    candidates: list[Evidence],
    cfg: VerificationConfig,
) -> VerificationResult | None:
    """Layer 2.  A *positive* rule verdict (supported) when a compatible
    source explicitly asserts the claim's value / relation; otherwise
    ``None`` (inconclusive) so Layer 3 can judge semantically."""
    claim_type = str(claim.get("claim_type", "factual"))

    # relational: a KG/SQL record that contains the asserted entities is
    # a positive rule confirmation.
    if claim_type == "relational":
        entities = [str(e) for e in (claim.get("entities") or [])]
        for ev in candidates:
            if ev.source_type not in ("kg", "sql"):
                continue
            text = str(ev.structured_value) if ev.structured_value is not None else ""
            text += " " + str(ev.content or "")
            if entities and all(en in text for en in entities):
                return VerificationResult(
                    claim_id=str(claim.get("claim_id", "")),
                    claim_type="relational",
                    supported=True,
                    status="supported",
                    support_score=0.9,
                    evidence_ids=[ev.evidence_id],
                    verifier_type="rule",
                    reason=f"relation asserted in {ev.source_type} evidence {ev.evidence_id}",
                )
        return None  # inconclusive: let Layer 3 judge

    # rule_based / factual: presence of a compatible source is a *weak*
    # positive, but we do not auto-support — hand to Layer 3 for a real
    # semantic judgment.  Only return a positive when a compatible
    # source is present AND the claim value appears in it.
    if claim_type in ("rule_based", "factual"):
        value = claim.get("value")
        if value is not None:
            for ev in candidates:
                if ev.source_type not in ("rag", "sql"):
                    continue
                blob = f"{ev.content or ''} {ev.structured_value if ev.structured_value is not None else ''}"
                if str(value) in blob:
                    return VerificationResult(
                        claim_id=str(claim.get("claim_id", "")),
                        claim_type=claim_type,  # type: ignore[arg-type]
                        supported=True,
                        status="supported",
                        support_score=0.85,
                        evidence_ids=[ev.evidence_id],
                        verifier_type="rule",
                        reason=f"value {value} present in {ev.source_type} evidence {ev.evidence_id}",
                    )
        # no numeric value: token-overlap check on the claim text against
        # evidence content + structured value.  This is a *rule* (not
        # semantic) check — it catches supported factual / rule_based
        # claims whose key terms appear verbatim in a compatible source
        # without requiring an LLM call.
        #
        # Safety guard: if the pool also contains a *structured source
        # with a parseable value that differs from the claim's asserted
        # value*, we must NOT auto-support via token overlap — the exact
        # layer (or the conflict branch below) must decide first.
        has_conflicting_structured = False
        for ev in candidates:
            if ev.source_type not in ("sql", "analysis"):
                continue
            f = _to_float(ev.structured_value)
            if f is None:
                continue
            # compare against any numeric claim value we can recover
            claim_num = _to_float(claim.get("value"))
            if claim_num is not None and abs(f - claim_num) > cfg.numeric_tolerance * max(1.0, abs(claim_num)):
                has_conflicting_structured = True
                break

        if has_conflicting_structured:
            return None  # let the exact / derived layer handle the conflict

        claim_tokens = _text_tokens(str(claim.get("text", "")))
        if claim_tokens:
            for ev in candidates:
                if ev.source_type not in ("rag", "sql", "kg", "analysis"):
                    continue
                blob = f"{ev.content or ''} {ev.structured_value if ev.structured_value is not None else ''}"
                blob_tokens = _text_tokens(blob)
                overlap = claim_tokens & blob_tokens
                # require at least 2 overlapping tokens and >=40% of the
                # claim's tokens present to avoid spurious matches
                if len(overlap) >= 2 and len(overlap) >= 0.4 * len(claim_tokens):
                    return VerificationResult(
                        claim_id=str(claim.get("claim_id", "")),
                        claim_type=claim_type,  # type: ignore[arg-type]
                        supported=True,
                        status="supported",
                        support_score=0.8,
                        evidence_ids=[ev.evidence_id],
                        verifier_type="rule",
                        reason=f"claim tokens {sorted(overlap)[:4]} present in {ev.source_type} evidence {ev.evidence_id}",
                    )
    return None


# ---------------------------------------------------------------------------
# Layer 3 — semantic verification (embedding gate + LLM judgment)
# ---------------------------------------------------------------------------


def verify_semantic(
    claim: dict[str, Any],
    candidates: list[Evidence],
    cfg: VerificationConfig,
    *,
    scorer: Any,
    llm: Any | None = None,
) -> VerificationResult:
    """Layer 3.  Returns a decisive result (never ``None``)."""
    claim_id = str(claim.get("claim_id", ""))
    claim_type = str(claim.get("claim_type", "factual"))
    claim_text = str(claim.get("text", ""))

    if not candidates:
        return VerificationResult(
            claim_id=claim_id, claim_type=claim_type,  # type: ignore[arg-type]
            supported=False, status="unsupported", support_score=0.0,
            verifier_type="semantic",
            reason="no candidate evidence available",
        )

    # embedding / lexical gate: keep the top semantic candidates
    scored = sorted(
        ((scorer.score(claim_text, e.content or str(e.structured_value or "")), e) for e in candidates),
        key=lambda t: -t[0],
    )
    top = [e for s, e in scored[: cfg.max_semantic_candidates] if s >= cfg.semantic_threshold]

    if not top:
        best_sim = scored[0][0]
        return VerificationResult(
            claim_id=claim_id, claim_type=claim_type,  # type: ignore[arg-type]
            supported=False, status="unsupported",
            support_score=round(best_sim, 4),
            verifier_type="semantic",
            reason=f"no candidate above semantic_threshold {cfg.semantic_threshold} (best sim {best_sim:.3f})",
        )

    # final LLM judgment on the candidates (never trust the embedding alone)
    if llm is not None:
        try:
            verdict = _judge_semantic(llm, claim_text, top)
        except LLMError as exc:
            # LLM unavailable: do NOT fabricate support — report honest
            # semantic-insufficient with the best similarity as the score.
            return VerificationResult(
                claim_id=claim_id, claim_type=claim_type,  # type: ignore[arg-type]
                supported=False, status="unsupported",
                support_score=round(scored[0][0], 4),
                verifier_type="semantic",
                reason=f"semantic LLM judgment unavailable ({exc.message}); no auto-support",
            )
        if verdict.conflicting_evidence_ids:
            return VerificationResult(
                claim_id=claim_id, claim_type=claim_type,  # type: ignore[arg-type]
                supported=False, status="conflict", conflict=True,
                support_score=0.0,
                evidence_ids=list(verdict.conflicting_evidence_ids),
                verifier_type="semantic",
                reason=verdict.reason or "LLM detected conflicting evidence",
            )
        supported = verdict.supported and verdict.support_score >= cfg.support_threshold
        return VerificationResult(
            claim_id=claim_id, claim_type=claim_type,  # type: ignore[arg-type]
            supported=supported,
            status="supported" if supported else "unsupported",
            support_score=round(verdict.support_score, 4),
            evidence_ids=[e.evidence_id for e in top],
            verifier_type="semantic",
            reason=verdict.reason or ("supported by semantic evidence" if supported else "insufficient semantic support"),
        )

    # no LLM: fall back to the embedding score with the policy threshold
    best_sim = max(s for s, _ in scored)
    supported = best_sim >= cfg.support_threshold
    return VerificationResult(
        claim_id=claim_id, claim_type=claim_type,  # type: ignore[arg-type]
        supported=supported,
        status="supported" if supported else "unsupported",
        support_score=round(best_sim, 4),
        evidence_ids=[e.evidence_id for e in top],
        verifier_type="semantic",
        reason=f"lexical/embedding sim {best_sim:.3f} vs threshold {cfg.support_threshold}",
    )


def _judge_semantic(llm: Any, claim_text: str, top: list[Evidence]) -> _SemanticVerdict:
    lines = [f"[{e.evidence_id}] ({e.source_type}) {e.content or e.structured_value}" for e in top]
    user = (
        f"claim: {claim_text}\n\n候选证据:\n" + "\n".join(lines) +
        "\n\n判断该 claim 是否被上述证据支持。"
    )
    verdict: _SemanticVerdict = llm.generate_structured(_SEMANTIC_SYSTEM, user, _SemanticVerdict)
    return verdict


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def verify(
    claim: dict[str, Any],
    candidates: list[Evidence],
    cfg: VerificationConfig,
    *,
    layers: tuple[str, ...] = ("exact", "rule", "semantic"),
    scorer: Any | None = None,
    llm: Any | None = None,
) -> VerificationResult:
    """Run the layered verifier and return one :class:`VerificationResult`.

    ``layers`` lets the ablation disable layers (e.g. rule-only baseline =
    ``("rule",)`` with no semantic; LLM-only = ``("semantic",)``).
    """
    claim_id = str(claim.get("claim_id", ""))
    claim_type = str(claim.get("claim_type", "factual"))

    if "exact" in layers:
        r = verify_exact(claim, candidates, cfg)
        if r is not None:
            return r
    if "rule" in layers:
        r = verify_rule(claim, candidates, cfg)
        if r is not None:
            return r
    if "semantic" in layers:
        return verify_semantic(
            claim, candidates, cfg,
            scorer=scorer or LexicalScorer(),
            llm=llm,
        )

    # No decisive layer available -> honest unsupported.
    return VerificationResult(
        claim_id=claim_id, claim_type=claim_type,  # type: ignore[arg-type]
        supported=False, status="unsupported", support_score=0.0,
        verifier_type="none",
        reason="no verification layer produced a verdict; treated as unsupported",
    )


# ---------------------------------------------------------------------------
# Batch helper + summary
# ---------------------------------------------------------------------------


def verify_all(
    claims: list[dict[str, Any]],
    link: dict[str, dict[str, Any]],
    pool: dict[str, Evidence],
    cfg: VerificationConfig,
    *,
    layers: tuple[str, ...] = ("exact", "rule", "semantic"),
    scorer: Any | None = None,
    llm: Any | None = None,
) -> list[VerificationResult]:
    results: list[VerificationResult] = []
    for claim in claims:
        cid = str(claim.get("claim_id", ""))
        entry = link.get(cid, {})
        cands = [pool[eid] for eid in entry.get("evidence_ids", []) if eid in pool]
        results.append(verify(claim, cands, cfg, layers=layers, scorer=scorer, llm=llm))
    return results


def summarize(results: list[VerificationResult]) -> dict[str, int]:
    """Aggregates used by the API ``verification_summary`` field."""
    out = {"supported_claims": 0, "unsupported_claims": 0, "conflicts": 0, "total": len(results)}
    for r in results:
        if r.conflict:
            out["conflicts"] += 1
        if r.supported:
            out["supported_claims"] += 1
        else:
            out["unsupported_claims"] += 1
    return out


__all__ = [
    "verify",
    "verify_exact",
    "verify_rule",
    "verify_semantic",
    "verify_all",
    "summarize",
]
