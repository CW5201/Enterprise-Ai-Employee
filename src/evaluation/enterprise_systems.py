"""System / ablation matrix for the Phase 5 enterprise benchmark.

This module defines the **runnable configurations** that the end-to-end
runner instantiates.  It does NOT re-implement any Phase 1–4 component —
it only re-parameterises the existing graph builder / tools so a single
codebase produces every system in the comparison:

Systems (the spec's main lines):

- **A — Full Enterprise AI Employee**: Task-Adaptive Dynamic Routing
  (``phase3=True``) + hybrid retrieval + the Claim-Evidence Verification
  tail (``phase4=True``).  The complete system.
- **B — Full minus Verification**: same routing + tools, but the
  verification tail is off (``phase4=False``).  Answers have no
  claim-evidence guard.
- **C — Static / Rule Routing, no Verification**: the Phase 1
  SupervisorRouter (``phase3=False``) with static RAG / SQL routing and no
  verification.  The composite no-LLM-router, no-verification baseline.

Baselines (single-component, built directly against the tools — no graph):

- **B1 LLM-only**: one structured LLM call answered against *no* evidence.
  Measures how far an ungrounded model gets (hallucination floor).
- **B2 Static RAG**: always run the hybrid RAG tool, answer from its chunks.
- **B3 Static SQL**: always run the SQL tool, answer from its rows.
- **B4 Rule-based Routing**: deterministic keyword rules -> capability flags
  -> the same rule engine the dynamic router uses (no LLM for the profile).

Component ablations of the full system (each toggles one module off, keeping
everything else at its Phase-A setting):

- **w/o Dynamic Routing**   : ``phase3=False`` (SupervisorRouter) + phase4 on.
- **w/o Verification**      : same as System B.
- **w/o KG**                : full system but the KG tool is disabled, so
  ``kg`` steps are honestly recorded as skipped (no silent RAG fallback).
- **w/o Hybrid Retrieval**  : full system but RAG runs dense-only.
- **w/o Reranker**          : full system but RAG runs hybrid without rerank.

Every configuration is a dataclass so a runner can enumerate them
deterministically, and so the "cost" of an expensive ablation can be
*recorded as a reason for skipping* rather than faked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RetrievalMode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]
RouterMode = Literal["dynamic", "supervisor", "rule"]


@dataclass(frozen=True)
class SystemConfig:
    """One runnable configuration of the Enterprise AI Employee.

    Attributes
    ----------
    name:
        Stable identifier written into the results (e.g. ``A_full``,
        ``B1_llm_only``).
    kind:
        ``system`` (graph-based) | ``baseline`` (tool-direct) | ``ablation``.
    description:
        Human summary of what this configuration does.
    router:
        ``dynamic`` -> TaskRouterNode; ``supervisor`` -> Phase 1 static
        router; ``rule`` -> deterministic keyword profile (no LLM).
    retrieval_mode:
        Which retrieval channel the RAG tool uses (relevant when the system
        actually runs RAG).
    enable_verification:
        Whether the Phase 4 claim-evidence tail runs.
    enable_kg:
        Whether the KG tool is available to the orchestrator.  When False,
        ``kg`` plan steps are recorded as ``skipped_unsupported`` — honest
        failure, never a silent fallback.
    llm_only:
        Baseline flag: answer from a single LLM call with no tool evidence.
    """

    name: str
    kind: str
    description: str
    router: RouterMode = "dynamic"
    retrieval_mode: RetrievalMode = "hybrid"
    enable_verification: bool = True
    enable_kg: bool = True
    llm_only: bool = False
    # Why an expensive component was intentionally not run (record, not fake).
    skip_reasons: dict[str, str] = field(default_factory=dict, compare=False)

    @property
    def phase3(self) -> bool:
        """Dynamic routing on the graph (rule/llm-adaptive both use it)."""
        return self.router in ("dynamic", "rule")

    @property
    def phase4(self) -> bool:
        return self.enable_verification and not self.llm_only

    def build_kwargs(self) -> dict[str, Any]:
        """Graph-builder kwargs for this config (graph-based systems)."""
        return {
            "phase3": self.phase3,
            "phase4": self.phase4,
            "retrieval_mode": self.retrieval_mode,
            "enable_kg": self.enable_kg,
        }


# ---------------------------------------------------------------------------
# The canonical matrix
# ---------------------------------------------------------------------------

SYSTEMS: tuple[SystemConfig, ...] = (
    SystemConfig(
        name="A_full", kind="system",
        description="Full Enterprise AI Employee: dynamic routing + hybrid "
                    "retrieval + claim-evidence verification.",
        router="dynamic", retrieval_mode="hybrid",
        enable_verification=True, enable_kg=True,
    ),
    SystemConfig(
        name="B_no_verification", kind="system",
        description="Dynamic routing + tools, verification off (System B).",
        router="dynamic", retrieval_mode="hybrid",
        enable_verification=False, enable_kg=True,
    ),
    SystemConfig(
        name="C_static_no_verification", kind="system",
        description="Phase 1 static router (SupervisorRouter), no verification "
                    "(System C composite baseline).",
        router="supervisor", retrieval_mode="hybrid",
        enable_verification=False, enable_kg=False,
    ),
    # -- baselines ---------------------------------------------------------
    SystemConfig(
        name="B1_llm_only", kind="baseline",
        description="One grounded-claim LLM call, no tool evidence (floor).",
        llm_only=True,
    ),
    SystemConfig(
        name="B2_static_rag", kind="baseline",
        description="Always run the hybrid RAG tool and answer from its chunks.",
        router="supervisor", retrieval_mode="hybrid",
        enable_verification=False, enable_kg=False,
    ),
    SystemConfig(
        name="B3_static_sql", kind="baseline",
        description="Always run the SQL tool and answer from its rows.",
        router="supervisor", retrieval_mode="hybrid",
        enable_verification=False, enable_kg=False,
    ),
    SystemConfig(
        name="B4_rule_routing", kind="baseline",
        description="Deterministic keyword capability rules -> rule engine "
                    "(no LLM task profile).",
        router="rule", retrieval_mode="hybrid",
        enable_verification=False, enable_kg=True,
    ),
    # -- component ablations (full system minus one module) ----------------
    SystemConfig(
        name="abl_w_dynamic_routing", kind="ablation",
        description="Full system but with the Phase 1 static router.",
        router="supervisor", retrieval_mode="hybrid",
        enable_verification=True, enable_kg=True,
    ),
    SystemConfig(
        name="abl_w_kg", kind="ablation",
        description="Full system but the KG tool is disabled (kg steps skip).",
        router="dynamic", retrieval_mode="hybrid",
        enable_verification=True, enable_kg=False,
    ),
    SystemConfig(
        name="abl_w_hybrid_retrieval", kind="ablation",
        description="Full system but RAG runs dense-only (no BM25/RRF).",
        router="dynamic", retrieval_mode="dense",
        enable_verification=True, enable_kg=True,
    ),
    SystemConfig(
        name="abl_w_reranker", kind="ablation",
        description="Full system but RAG runs hybrid without the reranker.",
        router="dynamic", retrieval_mode="hybrid",
        enable_verification=True, enable_kg=True,
    ),
    SystemConfig(
        name="abl_w_verification", kind="ablation",
        description="Full system minus verification (== System B).",
        router="dynamic", retrieval_mode="hybrid",
        enable_verification=False, enable_kg=True,
    ),
)

_SYSTEMS_BY_NAME = {s.name: s for s in SYSTEMS}


def get_system(name: str) -> SystemConfig:
    if name not in _SYSTEMS_BY_NAME:
        raise KeyError(f"unknown system {name!r}; known: {sorted(_SYSTEMS_BY_NAME)}")
    return _SYSTEMS_BY_NAME[name]


def all_systems() -> list[SystemConfig]:
    return list(SYSTEMS)


# Systems that run through the graph (vs. tool-direct baselines).
def graph_systems() -> list[SystemConfig]:
    return [s for s in SYSTEMS if not s.llm_only and s.kind in ("system", "ablation")]


__all__ = [
    "SystemConfig",
    "SYSTEMS",
    "get_system",
    "all_systems",
    "graph_systems",
]
