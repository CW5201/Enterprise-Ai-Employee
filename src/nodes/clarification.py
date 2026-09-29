"""Node: Clarification — ask the user when the task is under-specified.

Reached when the router routes to ``clarification`` (unknown/ambiguous
intent, or confidence below the gate).  Produces a short, honest
clarifying message instead of guessing, and stops the loop.
"""

from __future__ import annotations

from src.core.observability import observe
from src.core.state import AgentState

_SCOPE_NOTICE = (
    "该任务意图不明确或不属于「企业数据分析 / 企业知识问答」范围。"
    "请提出与业务数据（订单、客户、金额…）或制度文档（政策、流程、标准…）相关的具体问题。"
)


class ClarificationNode:
    @observe("clarify")
    def run(self, state: AgentState) -> AgentState:
        intent = state.get("intent", "") or "unknown"
        confidence = float(state.get("confidence") or 0.0)
        if intent in ("clarification", "") or confidence < 0.6:
            answer = _SCOPE_NOTICE
            status = "clarified_out_of_scope"
        else:
            answer = (
                f"我需要再确认一下您的意图（当前判断为 {intent}，置信度 {confidence:.2f}）。"
                f"原问题：「{state.get('user_query', '')}」。"
                "请说明：您是想查询业务数据，还是想了解某项公司制度/流程？"
            )
            status = "clarify_needed"
        return {"answer": answer, "status": status}


__all__ = ["ClarificationNode"]
