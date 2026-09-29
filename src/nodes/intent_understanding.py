"""Node: Intent Understanding — classify the user query into a typed intent.

Two lanes (docs/ROADMAP.md Phase 1):

1. **Rule fast lane** — deterministic keyword signals decide
   ``data_query`` / ``knowledge_query`` / ``complex_analysis`` immediately
   (cheap, fast, no model call).  Ambiguous queries fall through to the LLM.
2. **LLM lane** — :class:`IntentResult` is requested through the unified
   LLM client and validated by Pydantic; free-text model output can never
   reach downstream nodes unstructured.

Outputs written to AgentState: ``intent``, ``confidence``, ``entities``,
``plan`` (the intent-level interpretation).
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from src.core.exceptions import LLMError
from src.core.llm_client import LLMClient
from src.core.observability import observe
from src.core.state import AgentState, ErrorRecord

# ---------------------------------------------------------------------------
# Structured schema
# ---------------------------------------------------------------------------

IntentLabel = Literal["data_query", "knowledge_query", "complex_analysis", "clarification"]

INTENTS: tuple[str, ...] = ("data_query", "knowledge_query", "complex_analysis", "clarification")


class IntentResult(BaseModel):
    """Pydantic schema for the intent LLM call — output is always validated."""

    intent: IntentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    entities: list[str] = Field(default_factory=list)
    reason: str = ""

    @field_validator("intent")
    @classmethod
    def _check_intent(cls, value: str) -> str:
        if value not in INTENTS:
            raise ValueError(f"intent {value!r} is not in the Phase 1 taxonomy {INTENTS}")
        return value


# ---------------------------------------------------------------------------
# Rule fast lane
# ---------------------------------------------------------------------------

_DATA_KEYWORDS = [
    "查询", "统计", "多少", "数量", "金额", "总额", "订单", "销售额", "营收",
    "利润", "客户", "产品", "库存", "前10", "top", "排名", "合计", "平均",
    "占比", "趋势", "环比", "同比", "最大", "最少", "最多",
]
_KNOWLEDGE_KEYWORDS = [
    "政策", "制度", "规定", "标准", "报销", "流程", "手册", "规范",
    "要求", "说明", "是什么", "什么意思", "怎么", "如何",
]
_COMPLEX_MARKERS = [
    "并结合", "以及", "综合", "同时", "交叉", "对比", "结合相关", "解释结果",
]


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(k in text.lower() for k in keywords)


def classify_by_rules(query: str) -> tuple[str, float, list[str], str] | None:
    """Return (intent, confidence, entities, reason) when a rule is decisive.

    A rule fires only when the signal is strong enough to skip the LLM:
    - complex markers + data terms  -> complex_analysis
    - data terms only               -> data_query
    - policy terms only             -> knowledge_query
    Anything ambiguous returns None (LLM lane).
    """
    has_complex = _contains_any(query, _COMPLEX_MARKERS)
    has_data = _contains_any(query, _DATA_KEYWORDS)
    has_knowledge = _contains_any(query, _KNOWLEDGE_KEYWORDS)

    if has_complex and has_data:
        return "complex_analysis", 0.95, _extract_data_entities(query), "rule: 数据查询 + 结合文档解释"
    if has_data and not has_knowledge:
        return "data_query", 0.95, _extract_data_entities(query), "rule: 数据查询关键词"
    if has_knowledge and not has_data:
        return "knowledge_query", 0.95, [], "rule: 政策/制度/文档关键词"
    return None


def _extract_data_entities(query: str) -> list[str]:
    entities: list[str] = []
    top_match = re.search(r"前\s*(\d+)", query)
    if top_match:
        entities.append(f"top_n:{top_match.group(1)}")
    for name in re.findall(r"['‘’]([^’‘']{1,40})['‘’]", query):
        entities.append(name)
    return entities


# ---------------------------------------------------------------------------
# LLM lane
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """你是企业 AI 数字员工的意图理解模块。
将用户问题分类到以下意图之一，并抽取关键实体：
- data_query: 针对结构化业务数据的查询/统计（订单、金额、客户、库存…）
- knowledge_query: 企业制度、政策、流程、规范类文档问答
- complex_analysis: 需要同时使用数据查询和业务文档解释的复合任务
- clarification: 问题模糊、缺关键信息或超出企业任务范围

返回 JSON: {{"intent": str, "confidence": 0~1, "entities": [str], "reason": str}}
意图必须是上述四个取值之一，不得返回其他值。"""


def _llm_classify(query: str, llm: LLMClient) -> IntentResult:
    return llm.generate_structured(
        _SYSTEM_PROMPT,
        f"用户问题: {query}",
        IntentResult,
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class IntentUnderstandingNode:
    """Intent node: rule fast lane first, LLM lane for ambiguous queries."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    @observe("intent")
    def run(self, state: AgentState) -> AgentState:
        query = state.get("user_query", "")
        errors: list[ErrorRecord] = []
        entities: list[dict[str, Any]] = []

        rule = classify_by_rules(query)
        if rule is not None:
            intent, confidence, raw_entities, reason = rule
        else:
            try:
                parsed = _llm_classify(query, self._llm)
                intent, confidence = parsed.intent, parsed.confidence
                raw_entities, reason = parsed.entities, parsed.reason
            except LLMError as exc:
                errors.append(ErrorRecord(stage="intent", message=str(exc.message)))
                intent, confidence, raw_entities, reason = "clarification", 0.0, [], f"LLM 不可用: {exc.message}"

        entities = [{"type": "raw", "value": e} for e in raw_entities]
        return {
            "intent": intent,
            "confidence": confidence,
            "entities": entities,
            "plan": f"intent={intent} ({reason})",
            "errors": errors,
        }


__all__ = ["IntentResult", "INTENTS", "IntentUnderstandingNode", "classify_by_rules"]
