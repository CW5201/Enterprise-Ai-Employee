"""Node: SQL execution - Text-to-SQL generation + safe execution (Phase 1).

Phase 1 flow:

    natural-language question + DuckDB schema hint
      -> LLM generates SQL (``text_to_sql`` prompt template)
      -> SQL guard (SELECT/WITH only) via src.core.sql_executor
      -> execute on DuckDB, shape into an Evidence item

On LLM failure the node records the error in ``state.errors`` and returns
an empty result so the pipeline can still produce an honest answer.
"""

from __future__ import annotations

import json
from typing import Any

from src.core.exceptions import LLMError
from src.core.llm_client import LLMClient
from src.core.observability import observe
from src.core.state import AgentState, Evidence
from src.tools.base import ToolRegistry, ToolResult

_SYSTEM_PROMPT = """你是企业数据 Text-to-SQL 生成器，目标数据库为 DuckDB（只读）。
可用表结构:
{schema}
规则:
- 只生成单条只读 SELECT 或 WITH...SELECT 语句;
- 禁止 INSERT/UPDATE/DELETE/DROP/CREATE/ATTACH 等;
- 数值口径参考数据字典: {data_dictionary}
返回 JSON: {{"sql": str, "rationale": str}}"""

_USER_TEMPLATE = "用户问题: {question}\n意图: {intent}\n槽位: {slots}"


class SQLExecutionNode:
    def __init__(self, registry: ToolRegistry, llm: LLMClient | None = None,
                 schema_hint: str = "", data_dictionary: str = "") -> None:
        self._registry = registry
        self._llm = llm or LLMClient()
        self._schema_hint = schema_hint
        self._data_dictionary = data_dictionary

    @observe("sql_node")
    def run(self, state: AgentState) -> AgentState:
        question = state.get("user_task", "")
        intent = state.get("intent", "")
        slots = state.get("slots") or {}

        try:
            parsed = self._llm.chat_json(
                _SYSTEM_PROMPT.format(
                    schema=self._schema_hint or "(未配置 schema 提示)",
                    data_dictionary=self._data_dictionary or "(未配置数据字典)",
                ),
                _USER_TEMPLATE.format(question=question, intent=intent, slots=json.dumps(slots, ensure_ascii=False)),
            )
        except LLMError as exc:
            errors = list(state.get("errors") or [])
            errors.append({"stage": "sql_generation", "message": str(exc)})
            return {"errors": errors}

        sql = str(parsed.get("sql", "")).strip()
        if not sql:
            errors = list(state.get("errors") or [])
            errors.append({"stage": "sql_generation", "message": "LLM produced empty SQL"})
            return {"errors": errors}

        result: ToolResult = self._registry.call("sql", sql=sql, question=question)
        evidence: list[Evidence] = []
        if result.ok:
            payload = result.data or {}
            evidence.append(
                Evidence(
                    evidence_id=f"ev-{state.get('task_id', 'task')}-sql",
                    source_type="duckdb",
                    source_ref=result.evidence_ref or f"sql:{sql}",
                    content=_summarise(payload, sql, parsed.get("rationale", "")),
                    payload=payload,
                    metadata={"intent": intent, "rationale": str(parsed.get("rationale", ""))},
                )
            )
        else:
            errors = list(state.get("errors") or [])
            errors.append({"stage": "sql_execution", "message": result.error or "sql tool failed"})
            return {"errors": errors}

        tool_calls = list(state.get("tool_calls") or [])
        tool_calls.append(result.to_call_record())
        return {
            "evidence": evidence,
            "tool_calls": tool_calls,
        }


def _summarise(payload: dict[str, Any], sql: str, rationale: str) -> str:
    columns = payload.get("columns", [])
    rows = payload.get("rows", [])
    head = " | ".join(str(c) for c in columns)
    preview = "\n".join(" | ".join(str(v) for v in row) for row in rows[:5])
    return (
        f"SQL: {sql}\n"
        f"Rationale: {rationale}\n"
        f"Columns: {head}\n"
        f"Rows ({len(rows)} shown): \n{preview}"
    )


__all__ = ["SQLExecutionNode"]
