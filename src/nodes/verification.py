"""Node: Verification — run the layered verifier over the answer's claims.

Phase 4 (replaces the Phase-0 placeholder).  Reads the claims produced by
:mod:`src.nodes.claim_extraction` and the evidence produced by
:mod:`src.nodes.evidence_collection`, links each claim to candidate
evidence, runs :func:`src.core.claim_verifier.verify`, and writes one
:class:`VerificationResult` per claim to ``state["verification_results"]``.

Wiring contract (docs/ARCHITECTURE.md):

    claim_extraction  -> claims
    evidence_collection-> evidence_v4
    verification      -> verification_results

The node only orchestrates; all verdicts come from the engine.  It reads
the ``verification`` config section for thresholds (never hardcoded).
"""

from __future__ import annotations

from typing import Any

from src.core.claim_evidence_linking import link_claims_to_evidence
from src.core.claim_verifier import (
    BgeSemanticScorer,
    LexicalScorer,
    VerificationConfig,
    summarize,
    verify_all,
)
from src.core.config_loader import get_settings
from src.core.observability import observe
from src.core.state import AgentState
from src.core.verification_types import Evidence

__all__ = ["VerificationNode"]


class VerificationNode:
    """Run the layered claim-evidence verifier over the current state.

    Args:
        llm: optional LLM client for the Layer-3 semantic judgment.  When
            ``None`` the engine falls back to a pure similarity gate (no
            fabricated support).
        embedder: optional embedder for BGE-M3 semantic scoring.  When
            ``None`` the lexical scorer is used (test / offline mode).
        scorer: an explicit semantic scorer (wins over ``embedder``).
        llm_verifier_enabled: force the Layer-3 LLM judgment on/off
            independent of ``llm`` (used by the LLM-only baseline D).
    """

    def __init__(
        self,
        llm: Any | None = None,
        embedder: Any | None = None,
        scorer: Any | None = None,
        llm_verifier_enabled: bool = True,
    ) -> None:
        self._llm = llm
        self._embedder = embedder
        self._scorer = scorer
        self._llm_verifier_enabled = llm_verifier_enabled

    def _make_scorer(self) -> Any:
        if self._scorer is not None:
            return self._scorer
        if self._embedder is not None:
            return BgeSemanticScorer(self._embedder)
        return LexicalScorer()

    @observe("verification")
    def run(self, state: AgentState) -> AgentState:
        settings = get_settings()
        cfg = VerificationConfig.from_settings(settings.verification)
        claims: list[dict[str, Any]] = list(state.get("claims") or [])
        evidence_dicts: list[dict[str, Any]] = list(state.get("evidence_v4") or [])

        if not claims:
            # Nothing extracted -> honest empty result (guard handles it).
            return {"verification_results": [], "verification_summary": summarize([])}

        pool: dict[str, Evidence] = {}
        for ed in evidence_dicts:
            ev = Evidence.model_validate(ed)
            pool[ev.evidence_id] = ev
        evidence_list = list(pool.values())

        link = link_claims_to_evidence(claims, evidence_list)
        results = verify_all(
            claims, link, pool, cfg,
            scorer=self._make_scorer(),
            llm=self._llm if self._llm_verifier_enabled else None,
        )
        return {
            "verification_results": [r.model_dump() for r in results],
            "verification_summary": summarize(results),
        }
