"""Node: Task-Adaptive Dynamic Router (Phase 3).

Pipeline (Innovation 1, RQ1)::

    user query + intent
        │
        ▼
    LLM structured TaskProfile  (capability flags + ambiguity + difficulty)
        │
        ▼
    capability extraction      (TaskProfile.capability_vector)
        │
        ▼
    candidate tools            (config/routing_rules.yaml capability_rules)
        │
        ▼
    routing rules + guardrails (single vs multi_tool vs clarification)
        │
        ▼
    RoutingDecision.model_validate  (strict — malformed output rejected)

Two lanes (mirrors Phase 1 intent node):

1. **Rule lane** — when the Phase 1 intent is unambiguous and its
   capability mapping is decisive, build the TaskProfile deterministically
   (no LLM call, cheap, fast, fully reproducible baseline-C behaviour).
2. **LLM lane** — ambiguous intents request a structured TaskProfile from
   the model.  The model output is *proposed*, not trusted: it is
   Pydantic-validated and re-checked by the rule engine; an unknown tool
   or route raises :class:`RoutingDecisionError` rather than silently
   falling back to RAG.

The node writes the structured decision and a full audit trace to
AgentState: ``routing_decision``, ``routing_trace`` (task profile,
candidates, selected tools, reason, timestamp, latency — never secrets).
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.core.exceptions import LLMError
from src.core.llm_client import LLMClient
from src.core.observability import observe
from src.core.routing_rules import RoutingPolicy, decide as _decide_rule, generate_candidates
from src.core.routing_types import RoutingDecision, TaskProfile
from src.core.state import AgentState, ErrorRecord


class RoutingDecisionError(Exception):
    """Raised when a proposed routing decision fails validation.

    No silent fallback: the caller (graph / orchestrator) records the
    error and routes to clarification.
    """

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}


# ---------------------------------------------------------------------------
# Structured LLM schema — the model proposes a TaskProfile, not free text
# ---------------------------------------------------------------------------


class TaskProfileProposal(BaseModel):
    """The LLM's structured proposal for a task's capability profile."""

    task_type: Literal[
        "knowledge_lookup", "structured_lookup", "aggregation",
        "relationship_query", "statistical_analysis", "trend_analysis",
        "comparison", "report_generation", "multi_source_analysis",
        "ambiguous_task",
    ] = "ambiguous_task"
    requires_structured_data: bool = False
    requires_unstructured_knowledge: bool = False
    requires_relationship_reasoning: bool = False
    requires_statistical_analysis: bool = False
    requires_visualization: bool = False
    requires_multi_source: bool = False
    ambiguity: float = Field(default=0.0, ge=0.0, le=1.0)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    reason: str = ""

    def to_task_profile(self, task_id: str) -> TaskProfile:
        return TaskProfile(
            task_id=task_id,
            task_type=self.task_type,  # type: ignore[arg-type]
            requires_structured_data=self.requires_structured_data,
            requires_unstructured_knowledge=self.requires_unstructured_knowledge,
            requires_relationship_reasoning=self.requires_relationship_reasoning,
            requires_statistical_analysis=self.requires_statistical_analysis,
            requires_visualization=self.requires_visualization,
            requires_multi_source=self.requires_multi_source,
            ambiguity=self.ambiguity,
            difficulty=self.difficulty,
        )


_TASK_PROFILE_SYSTEM = """你是企业 AI 数字员工的任务特征分析模块。
把用户任务转成结构化的 TaskProfile（能力需求标志 + 歧义度 + 难度）。
判断标准：
- requires_structured_data: 需要精确业务数据/统计（订单、金额、数量、排名…）
- requires_unstructured_knowledge: 需要政策/制度/流程/规范等文档知识
- requires_relationship_reasoning: 需要实体关系或多跳（客户-订单-商品-供应商、组织汇报）
- requires_statistical_analysis: 需要计算派生指标（增长率、占比、均值、趋势）
- requires_visualization: 明确要求图表/可视化
- requires_multi_source: 真正需要 ≥2 类知识源协同
- ambiguity: 0~1，关键信息缺失越多越高；"最近/大概/帮我看看"这类模糊措辞提高
- difficulty: easy / medium / hard
返回 JSON: {{"task_type": str, "requires_structured_data": bool, ... , "ambiguity": 0~1, "difficulty": str, "reason": str}}"""


# Intent -> deterministic capability flags (rule lane).
# Only unambiguous Phase 1 intents map directly; everything else goes LLM.
_RULE_LANE: dict[str, dict[str, Any]] = {
    "knowledge_query": {"task_type": "knowledge_lookup", "requires_unstructured_knowledge": True},
    "data_query": {"task_type": "structured_lookup", "requires_structured_data": True},
    "complex_analysis": {
        "task_type": "multi_source_analysis",
        "requires_structured_data": True,
        "requires_unstructured_knowledge": True,
        "requires_statistical_analysis": True,
        "requires_multi_source": True,
    },
}


class TaskRouterNode:
    """Dynamic Task-Adaptive Router (replaces Phase 1 SupervisorRouterNode
    when constructed via ``build_graph(phase3=True)``)."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        policy: RoutingPolicy | None = None,
        min_confidence: float | None = None,
    ) -> None:
        self._llm = llm or LLMClient()
        self._policy = policy or RoutingPolicy.load()
        self._min_confidence = (
            min_confidence if min_confidence is not None
            else float(self._policy.__dict__.get("clarification_max_ambiguity", 0.6))
        )

    # -- public -------------------------------------------------------------

    @observe("routing")
    def run(self, state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        query = state.get("user_query", "")
        intent = str(state.get("intent") or "")
        intent_conf = float(state.get("confidence") or 0.0)
        task_id = str(state.get("request_id") or "task")
        errors: list[ErrorRecord] = []

        # 1. Task profile: rule lane first, LLM lane for the rest.
        profile = self._build_profile(task_id, query, intent, intent_conf, errors)

        # 2. Candidates + decision (rule engine; validated models only).
        candidates = generate_candidates(profile, self._policy)
        try:
            decision = self._decide(profile, errors)
        except RoutingDecisionError:
            # Rejected decision: route to clarification explicitly.  No
            # silent fallback to RAG — the trace records the rejection.
            decision = RoutingDecision(
                route_type="clarification",
                tools=[],
                execution_plan=[],
                reason="routing decision rejected by validation -> clarification",
                confidence=0.0,
                requires_clarification=True,
            )

        # 3. Confidence gate: a low-confidence *non-clarification* decision
        #    must be demoted to clarification — no silent guessing.
        if decision.route_type != "clarification" and decision.confidence < self._min_confidence:
            decision = RoutingDecision(
                route_type="clarification",
                tools=[],
                execution_plan=[],
                reason=f"low routing confidence {decision.confidence:.2f} < {self._min_confidence}",
                confidence=decision.confidence,
                requires_clarification=True,
            )
            errors.append(ErrorRecord(
                stage="routing", message="confidence gate -> clarification",
                details={"confidence": decision.confidence},
            ))

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        trace = {
            "task_profile": profile.model_dump(),
            "candidates": [
                {"tool": c.tool, "weight": c.weight, "reason": c.reason} for c in candidates
            ],
            "selected_tools": list(decision.tools),
            "route_type": decision.route_type,
            "reason": decision.reason,
            "confidence": decision.confidence,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "latency_ms": latency_ms,
        }

        return {
            "route": decision.route_type,
            "route_confidence": decision.confidence,
            "plan": _plan_to_text(decision),
            "routing_decision": decision.model_dump(),
            "routing_trace": trace,
            "tools_selected": list(decision.tools),
            "errors": errors,
        }

    # -- lanes --------------------------------------------------------------

    def _build_profile(
        self,
        task_id: str,
        query: str,
        intent: str,
        intent_conf: float,
        errors: list[ErrorRecord],
    ) -> TaskProfile:
        if intent in _RULE_LANE and intent_conf >= 0.8:
            flags: dict[str, Any] = {"task_id": task_id, **_RULE_LANE[intent]}
            return TaskProfile(**flags)

        try:
            proposal: TaskProfileProposal = self._llm.generate_structured(
                _TASK_PROFILE_SYSTEM,
                f"用户任务: {query}\n已识别意图: {intent or 'unknown'}",
                TaskProfileProposal,
            )
            return proposal.to_task_profile(task_id)
        except LLMError as exc:
            errors.append(ErrorRecord(stage="routing", message=f"task profile LLM failed: {exc.message}"))
            # Honest degradation: mark the task ambiguous — the rule engine
            # will route to clarification.  NEVER guess a capability set.
            return TaskProfile(task_id=task_id, task_type="ambiguous_task", ambiguity=1.0)
        except Exception as exc:  # noqa: BLE001 — malformed structured output
            errors.append(ErrorRecord(stage="routing", message=f"task profile malformed: {exc}"))
            return TaskProfile(task_id=task_id, task_type="ambiguous_task", ambiguity=1.0)

    def _decide(self, profile: TaskProfile, errors: list[ErrorRecord]) -> RoutingDecision:
        try:
            return _decide_validated(profile, self._policy)
        except Exception as exc:  # noqa: BLE001
            errors.append(ErrorRecord(
                stage="routing", message=f"routing decision rejected: {exc}",
                details={"task_profile": profile.model_dump()},
            ))
            raise RoutingDecisionError(str(exc), details={"task_profile": profile.model_dump()}) from exc


def _decide_validated(profile: TaskProfile, policy: RoutingPolicy) -> RoutingDecision:
    """Rule-engine decision, re-validated through the strict Pydantic model.

    The policy engine is imported lazily as ``_decide_rule`` at module
    level so tests can monkeypatch ``tr._decide_rule`` to inject broken
    decisions and assert the rejection path.
    """
    decision = _decide_rule(profile, policy)
    # Re-validate through the model so any policy bug that produced an
    # inconsistent decision surfaces here, not at execution time.
    return RoutingDecision.model_validate(decision.model_dump())


def _plan_to_text(decision: RoutingDecision) -> str:
    if decision.route_type == "clarification":
        return "clarification"
    return " -> ".join(f"{s.step}:{s.tool}" for s in decision.execution_plan) or decision.route_type


__all__ = ["TaskRouterNode", "RoutingDecisionError", "TaskProfileProposal"]
