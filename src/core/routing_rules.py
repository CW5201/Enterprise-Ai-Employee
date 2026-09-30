"""Routing rule engine — capability signals -> candidate tools -> decision.

Phase 3 Commit 1.  This module is the *deterministic* heart of
Task-Adaptive Routing: given a :class:`~src.core.routing_types.TaskProfile`
it produces candidate tools (from the ``capability_rules`` policy in
``config/routing_rules.yaml``) and, combined with the LLM's confidence,
builds a :class:`~src.core.routing_types.RoutingDecision`.

The LLM proposes; the rules dispose.  An LLM's free-text or malformed
output never becomes the decision — it is validated through the Pydantic
models and this rule engine, and unknown tools / routes are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.core.routing_types import (
    ROUTE_TYPES,
    ExecutionStep,
    RoutingDecision,
    TaskProfile,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ROUTING_RULES_FILE = _PROJECT_ROOT / "config" / "routing_rules.yaml"


class RoutingPolicyError(ValueError):
    """The routing policy file is malformed or inconsistent."""


@dataclass
class Candidate:
    """A tool the rule engine thinks might serve the task."""

    tool: str
    weight: float
    reason: str


@dataclass
class RoutingPolicy:
    """Parsed view over the routing policy (capability rules + thresholds)."""

    capability_rules: list[dict[str, Any]] = field(default_factory=list)
    multi_tool_trigger_flags: list[str] = field(default_factory=list)
    clarification_max_ambiguity: float = 0.6
    clarification_auto_types: list[str] = field(default_factory=list)
    require_at_least_one_flag: bool = True
    multi_tool_penalty: float = 0.15
    ambiguity_penalty_factor: float = 0.5
    strong_single_tool_weight: float = 1.0
    known_tools: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None = None) -> RoutingPolicy:
        path = path or _ROUTING_RULES_FILE
        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        policy = cls()
        policy.capability_rules = raw.get("capability_rules", []) or []
        policy.multi_tool_trigger_flags = raw.get("multi_tool_trigger_flags", []) or []
        clar = raw.get("clarification", {}) or {}
        policy.clarification_max_ambiguity = float(clar.get("max_ambiguity", 0.6))
        policy.clarification_auto_types = clar.get("auto_clarify_task_types", []) or []
        policy.require_at_least_one_flag = bool(clar.get("require_at_least_one_flag", True))
        conf = raw.get("confidence", {}) or {}
        policy.multi_tool_penalty = float(conf.get("multi_tool_penalty", 0.15))
        policy.ambiguity_penalty_factor = float(conf.get("ambiguity_penalty_factor", 0.5))
        policy.strong_single_tool_weight = float(conf.get("strong_single_tool_weight", 1.0))
        policy.known_tools = raw.get("route_types", []) or []
        # known_tools here is the route vocabulary; the *tool* whitelist is
        # KNOWN_TOOLS from routing_types (synced with tool_registry.yaml).
        policy.known_tools = [
            "rag", "sql", "kg", "analysis", "chart", "report",
        ]
        if not policy.capability_rules:
            raise RoutingPolicyError("capability_rules must be non-empty")
        return policy


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def _capability_flag(profile: TaskProfile, flag: str) -> bool:
    return bool(getattr(profile, flag, False))


def generate_candidates(
    profile: TaskProfile,
    policy: RoutingPolicy | None = None,
) -> list[Candidate]:
    """Return the candidate tools for *profile*, ordered by descending weight.

    A tool becomes a candidate when a capability rule fires for one of the
    profile's set capability flags.  Weights are summed when several flags
    map to the same tool (they don't in this policy, but the engine is
    generic).  The result order is the *candidate* order, not yet the final
    execution plan.
    """
    policy = policy or RoutingPolicy.load()
    acc: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}

    for rule in policy.capability_rules:
        flag = str(rule["flag"])
        tool = str(rule["tool"])
        weight = float(rule.get("weight", 1.0))
        if _capability_flag(profile, flag):
            acc[tool] = acc.get(tool, 0.0) + weight
            reasons.setdefault(tool, []).append(f"flag {flag!r}")

    # chart is a companion: only keep it if a real source tool is present
    candidates = [
        Candidate(tool=t, weight=w, reason="; ".join(reasons[t]))
        for t, w in acc.items()
    ]
    # order: descending weight, then registry order for ties
    registry_order = {t: i for i, t in enumerate(policy.known_tools)}
    candidates.sort(key=lambda c: (-c.weight, registry_order.get(c.tool, 99)))
    return candidates


# ---------------------------------------------------------------------------
# Decision construction
# ---------------------------------------------------------------------------


def decide(profile: TaskProfile, policy: RoutingPolicy | None = None) -> RoutingDecision:
    """Build a validated :class:`RoutingDecision` from a *profile*.

    This is the pure capability-driven path (used by the rule-based
    baseline and as the deterministic core of the dynamic router).
    """
    policy = policy or RoutingPolicy.load()
    candidates = generate_candidates(profile, policy)

    # 1. Under-specified task -> clarification (no guessing).
    if profile.task_type in policy.clarification_auto_types:
        return _clarification_decision(profile, "task type is inherently ambiguous")
    if profile.ambiguity >= policy.clarification_max_ambiguity:
        return _clarification_decision(profile, f"ambiguity {profile.ambiguity} >= {policy.clarification_max_ambiguity}")
    if policy.require_at_least_one_flag and not candidates:
        return _clarification_decision(profile, "no capability signal matched")

    # 2. Pick the route: single candidate => single-tool route; multiple
    #    primary-source candidates => multi_tool.
    # chart/report are non-source companions and never define the route alone.
    primary = [c for c in candidates if c.tool in ("rag", "sql", "kg", "analysis")]
    if len(primary) == 1:
        return _single_tool_decision(profile, primary[0], candidates, policy)
    if len(primary) >= 2 or profile.requires_multi_source:
        return _multi_tool_decision(profile, candidates, policy)
    # only companion tools matched (chart/report) — nothing to run
    return _clarification_decision(profile, "only companion tools matched; no data source")


def _clarification_decision(profile: TaskProfile, reason: str) -> RoutingDecision:
    return RoutingDecision(
        route_type="clarification",
        tools=[],
        execution_plan=[],
        reason=f"clarification: {reason}",
        confidence=round(min(profile.ambiguity, 0.5), 2),
        requires_clarification=True,
    )


def _confidence(profile: TaskProfile, is_multi: bool, policy: RoutingPolicy, top_weight: float) -> float:
    base = 0.5 + 0.4 * min(top_weight / max(policy.strong_single_tool_weight, 1e-6), 1.0)
    if is_multi:
        base = max(0.3, base - policy.multi_tool_penalty)
    base *= 1.0 - profile.ambiguity * policy.ambiguity_penalty_factor
    return round(max(0.0, min(1.0, base)), 2)


def _single_tool_decision(
    profile: TaskProfile,
    top: Candidate,
    all_candidates: list[Candidate],
    policy: RoutingPolicy,
) -> RoutingDecision:
    route = top.tool  # rag / sql / kg / analysis
    if route not in ROUTE_TYPES:
        raise RoutingPolicyError(f"tool {route!r} is not a valid single-tool route")
    plan = [
        ExecutionStep(step=1, tool=top.tool, purpose=f"{top.tool}: {top.reason}")
    ]
    conf = _confidence(profile, is_multi=False, policy=policy, top_weight=top.weight)
    return RoutingDecision(
        route_type=route,
        tools=[top.tool],
        execution_plan=plan,
        reason=f"single tool {top.tool!r} (candidates: {', '.join(c.tool for c in all_candidates)})",
        confidence=conf,
    )


def _multi_tool_decision(
    profile: TaskProfile,
    candidates: list[Candidate],
    policy: RoutingPolicy,
) -> RoutingDecision:
    # Keep primary-source tools first, in candidate (weight) order; drop
    # duplicate companions so the plan stays minimal.
    order = {t: i for i, t in enumerate(policy.known_tools)}
    tools: list[str] = []
    for c in candidates:
        if c.tool in ("rag", "sql", "kg", "analysis") and c.tool not in tools:
            tools.append(c.tool)
    if not tools:
        return _clarification_decision(profile, "multi_tool but no primary source")
    tools.sort(key=lambda t: order[t])

    # Data-dependency: statistical analysis and report consume upstream data.
    plan: list[ExecutionStep] = []
    for i, t in enumerate(tools, start=1):
        step_id = i
        depends_on: list[int] = []
        if t == "analysis" and step_id > 1:
            depends_on = [s.step for s in plan if s.tool in ("sql", "kg")]
        elif t == "report" and step_id > 1:
            depends_on = [s.step for s in plan]
        purpose = {
            "sql": "retrieve structured business data",
            "rag": "retrieve unstructured enterprise knowledge",
            "kg": "traverse entity relationships",
            "analysis": "compute derived statistics / trends over upstream results",
            "report": "synthesize all results into the final report",
        }.get(t, t)
        plan.append(ExecutionStep(step=step_id, tool=t, purpose=purpose, depends_on=depends_on))

    conf = _confidence(profile, is_multi=True, policy=policy, top_weight=candidates[0].weight)
    return RoutingDecision(
        route_type="multi_tool",
        tools=tools,
        execution_plan=plan,
        reason=f"multi_tool over {tools} (requires_multi_source={profile.requires_multi_source})",
        confidence=conf,
    )


__all__ = [
    "RoutingPolicy",
    "RoutingPolicyError",
    "Candidate",
    "decide",
    "generate_candidates",
]
