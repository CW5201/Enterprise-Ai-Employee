"""Node: Intent Understanding - parse the user task into a typed intent.

Uses the ``intent_understanding`` prompt template (``config/prompt_templates.yaml``)
and the LLM client (``src/core/llm_client.py``).  The intent vocabulary comes
from ``config/routing_rules.yaml`` so routing and understanding share ONE taxonomy.

Output written to AgentState:
    intent, slots, constraints, intent_confidence, intent_reason
"""

from __future__ import annotations

from typing import Any

from src.core.config_loader import RoutingRules
from src.core.exceptions import LLMError
from src.core.llm_client import LLMClient
from src.core.observability import observe
from src.core.state import AgentState


def _error_record(stage: str, message: str) -> dict[str, Any]:
    return {"stage": stage, "message": message}


def _append_error(state: AgentState, stage: str, message: str) -> list[dict[str, Any]]:
    errors = list(state.get("errors") or [])
    errors.append(_error_record(stage, message))
    return errors

_SYSTEM_PROMPT = """你是企业 AI 数字员工的意图理解模块。
将用户任务分类到给定的意图体系，抽取槽位与约束。
意图体系: {intent_taxonomy}
返回 JSON: {{"intent": str, "slots": object, "constraints": object,
            "confidence": 0~1, "reason": str}}
若任务模糊或超范围，将 confidence 调低并在 reason 说明。"""

_USER_TEMPLATE = "用户任务: {user_task}\n对话上下文: {context}"

# fallback intent if the LLM returns something outside the taxonomy
_DEFAULT_INTENT = "chitchat_or_out_of_scope"


class IntentUnderstandingNode:
    def __init__(self, rules: RoutingRules, llm: LLMClient | None = None) -> None:
        self.rules = rules
        self.llm = llm or LLMClient()

    @observe("intent")
    def run(self, state: AgentState) -> AgentState:
        user_task = state.get("user_task", "")
        context = json_context(state.get("conversation") or [])
        taxonomy = dict(self.rules.intent_descriptions.items())
        system = _SYSTEM_PROMPT.format(intent_taxonomy=_format_taxonomy(taxonomy))
        user = _USER_TEMPLATE.format(user_task=user_task, context=context or "(none)")
        try:
            parsed = self.llm.chat_json(system, user)
        except LLMError as exc:
            # do not crash the pipeline on a single LLM failure: record and default
            return {
                "intent": _DEFAULT_INTENT,
                "slots": {},
                "constraints": {},
                "intent_confidence": 0.0,
                "intent_reason": f"LLM unavailable: {exc.message}",
                "errors": _append_error(state, "intent", str(exc)),
            }

        intent = str(parsed.get("intent", _DEFAULT_INTENT))
        if intent not in self.rules.intent_tools:
            intent = _DEFAULT_INTENT
        confidence = float(parsed.get("confidence", 0.5) or 0.5)
        confidence = max(0.0, min(1.0, confidence))
        return {
            "intent": intent,
            "slots": _as_dict(parsed.get("slots")),
            "constraints": _as_dict(parsed.get("constraints")),
            "intent_confidence": confidence,
            "intent_reason": str(parsed.get("reason", "")),
        }


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _format_taxonomy(taxonomy: dict[str, str]) -> str:
    if not taxonomy:
        return "(none declared)"
    return "\n".join(f"- {name}: {desc}" for name, desc in taxonomy.items())


def json_context(conversation: list[dict[str, str]]) -> str:
    import json

    if not conversation:
        return ""
    return json.dumps(conversation[-4:], ensure_ascii=False)


__all__ = ["IntentUnderstandingNode"]
