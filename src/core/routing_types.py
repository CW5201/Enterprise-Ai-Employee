"""Routing schema — structured task profile and routing decision (Phase 3).

Task-Adaptive Routing (RQ1, Innovation 1).  These Pydantic models are the
single source of truth for everything that moves between the router,
the execution orchestrator and the evaluation harness.

Design rules (docs/RESEARCH.md RQ1, ADR-009 discipline):

- The router outputs a :class:`RoutingDecision`, validated with
  ``model_validate`` — free-text LLM output can never reach downstream
  nodes unstructured.
- ``execution_plan`` is a strict list of :class:`ExecutionStep`; arbitrary
  Python is not expressible here by construction.
- Every decision is self-describing (``reason``, ``confidence``,
  ``requires_clarification``) so evaluation and ablation can inspect it.

``confidence`` is the **routing-decision confidence**, NOT the predicted
answer correctness.  It reflects how certain the router is that the chosen
tools/plan are the right ones for this task profile.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Task taxonomy (task_type values) — see config/routing_rules.yaml
# ---------------------------------------------------------------------------

TaskType = Literal[
    "knowledge_lookup",      # document / policy / procedure question
    "structured_lookup",     # exact fact from a business table
    "aggregation",           # COUNT / SUM / AVG / GROUP BY over data
    "relationship_query",    # entity traversal / multi-hop relations
    "statistical_analysis",  # compute a derived statistic
    "trend_analysis",        # time-series change / growth
    "comparison",           # side-by-side / ranking
    "report_generation",     # formatted multi-part output
    "multi_source_analysis", # combines >= 2 distinct knowledge sources
    "ambiguous_task",        # cannot be resolved without clarification
]

TASK_TYPES: tuple[str, ...] = tuple(TaskType.__args__)  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Tool vocabulary (registry whitelist) — see config/tool_registry.yaml
# ---------------------------------------------------------------------------

ToolName = Literal["rag", "sql", "kg", "analysis", "chart", "report"]

KNOWN_TOOLS: tuple[str, ...] = ("rag", "sql", "kg", "analysis", "chart", "report")

# ---------------------------------------------------------------------------
# Route vocabulary
# ---------------------------------------------------------------------------

RouteType = Literal[
    "rag",
    "sql",
    "kg",
    "analysis",
    "multi_tool",
    "clarification",
]

ROUTE_TYPES: tuple[str, ...] = ("rag", "sql", "kg", "analysis", "multi_tool", "clarification")


# ---------------------------------------------------------------------------
# TaskProfile — what the task *is* (capability requirements)
# ---------------------------------------------------------------------------


class TaskProfile(BaseModel):
    """Standardised description of one enterprise task.

    Produced by the task-understanding stage (LLM structured output +
    rule signals).  The fields are *capability requirements*, not raw
    intent labels — the router maps these onto tools.
    """

    task_id: str
    domain: str = "enterprise"
    task_type: TaskType = "ambiguous_task"

    # Capability flags (7 signals the router reasons over)
    requires_structured_data: bool = False
    requires_unstructured_knowledge: bool = False
    requires_relationship_reasoning: bool = False
    requires_statistical_analysis: bool = False
    requires_visualization: bool = False
    requires_multi_source: bool = False
    ambiguity: float = 0.0  # 0..1, 1 = fully ambiguous
    difficulty: Literal["easy", "medium", "hard"] = "medium"

    @field_validator("ambiguity")
    @classmethod
    def _clamp_ambiguity(cls, v: float) -> float:
        return min(max(v, 0.0), 1.0)

    def capability_vector(self) -> dict[str, bool]:
        """The 7 capability flags as a plain dict (for logging / eval)."""
        return {
            "requires_structured_data": self.requires_structured_data,
            "requires_unstructured_knowledge": self.requires_unstructured_knowledge,
            "requires_relationship_reasoning": self.requires_relationship_reasoning,
            "requires_statistical_analysis": self.requires_statistical_analysis,
            "requires_visualization": self.requires_visualization,
            "requires_multi_source": self.requires_multi_source,
        }


# ---------------------------------------------------------------------------
# ExecutionStep — one element of an execution plan (no arbitrary code)
# ---------------------------------------------------------------------------


class ExecutionStep(BaseModel):
    """A single tool invocation in the plan.

    ``depends_on`` names the steps (by index) whose outputs feed this
    step — this is how multi-tool flows (e.g. SQL -> Analysis) express
    data dependencies without any code.
    """

    step: int
    tool: ToolName
    purpose: str = ""
    depends_on: list[int] = Field(default_factory=list)

    @field_validator("tool")
    @classmethod
    def _check_tool_known(cls, v: str) -> str:
        if v not in KNOWN_TOOLS:
            raise ValueError(f"unknown tool {v!r}; allowed: {KNOWN_TOOLS}")
        return v

    @field_validator("step")
    @classmethod
    def _check_step_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("step index must be >= 1")
        return v

    @model_validator(mode="after")
    def _check_dep_refs(self) -> ExecutionStep:
        step_ids = {self.step}
        for d in self.depends_on:
            if d in step_ids or d < 1:
                raise ValueError(f"depends_on must reference other valid steps, got {self.depends_on}")
        return self


# ---------------------------------------------------------------------------
# RoutingDecision — the router's structured output
# ---------------------------------------------------------------------------


class RoutingDecision(BaseModel):
    """What the router decides to do, in a fully structured form."""

    route_type: RouteType
    tools: list[ToolName] = Field(default_factory=list)
    execution_plan: list[ExecutionStep] = Field(default_factory=list)
    reason: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    requires_clarification: bool = False

    @field_validator("route_type")
    @classmethod
    def _check_route(cls, v: str) -> str:
        if v not in ROUTE_TYPES:
            raise ValueError(f"unknown route_type {v!r}; allowed: {ROUTE_TYPES}")
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> RoutingDecision:
        # clarification routes carry no tools / plan
        if self.route_type == "clarification":
            if self.tools or self.execution_plan:
                raise ValueError("clarification route must have empty tools/execution_plan")
        else:
            if not self.tools:
                raise ValueError(f"route_type={self.route_type!r} requires a non-empty tools list")
            if self.route_type != "multi_tool" and len(self.tools) > 1:
                raise ValueError(
                    f"route_type={self.route_type!r} is single-tool; "
                    f"got {len(self.tools)} tools — use 'multi_tool'"
                )
        # every plan step must reference a known tool and a declared tool
        declared = set(self.tools)
        for step in self.execution_plan:
            if step.tool not in declared:
                raise ValueError(
                    f"execution_plan step {step.step} uses undeclared tool {step.tool!r}"
                )
        return self

    def tool_set(self) -> set[str]:
        return set(self.tools)

    def to_state(self) -> dict[str, Any]:
        """Serialise into the AgentState keys the graph reads."""
        return {
            "route": self.route_type,
            "route_confidence": self.confidence,
            "routing_decision": self.model_dump(),
            "tools_selected": list(self.tools),
        }


__all__ = [
    "TaskProfile",
    "RoutingDecision",
    "ExecutionStep",
    "TaskType",
    "RouteType",
    "ToolName",
    "TASK_TYPES",
    "ROUTE_TYPES",
    "KNOWN_TOOLS",
]
