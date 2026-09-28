"""Node: Clarification - ask the user when a task is under-specified.

Phase 1 keeps this minimal: when the router routes to ``clarify`` (low
intent confidence, or out-of-scope intent), the node produces a short,
honest clarifying message instead of guessing.  In later phases this node
gains the ambiguity-register logic from ``data/schemas/data_dictionary.yaml``.
"""

from __future__ import annotations

from src.core.observability import observe
from src.core.state import AgentState

_SCOPE_NOTICE = (
    "该任务不属于企业数据分析 / 知识问答范围（意图: {intent}）。"
    "请提出与业务数据、制度文档相关的具体任务。"
)


class ClarificationNode:
    @observe("clarify")
    def run(self, state: AgentState) -> AgentState:
        intent = state.get("intent", "unknown")
        confidence = float(state.get("intent_confidence", 0.0))
        if intent in ("chitchat_or_out_of_scope", ""):
            return {
                "answer": _SCOPE_NOTICE.format(intent=intent or "unknown"),
                "citations": [],
                "status": "clarified_out_of_scope",
                "next_action": "stop",
            }
        missing = _missing_slots(state)
        prompt = "请补充以下信息后重新提交任务：" + "；".join(missing)
        return {
            "answer": f"{prompt}\n（当前意图: {intent}, 置信度: {confidence:.2f}）",
            "citations": [],
            "status": "clarify_needed",
            "next_action": "stop",
        }


def _missing_slots(state: AgentState) -> list[str]:
    slots = state.get("slots") or {}
    constraints = state.get("constraints") or {}
    missing: list[str] = []
    if not slots.get("time_range"):
        missing.append("统计时间范围（如：本季度 / 近 12 个月）")
    if not constraints.get("department") and not slots.get("department"):
        missing.append("部门 / 区域（如：指定部门时请说明）")
    return missing


__all__ = ["ClarificationNode"]
