"""Node: Answer generation — final answer from fused SQL + RAG evidence.

Contract (Phase 1, docs/ROADMAP.md):
- NEVER fabricate data: SQL conclusions come strictly from ``sql_result``;
  RAG conclusions keep their source; data findings and knowledge findings
  are kept visibly distinct in the answer.
- No full Claim-Evidence Verification in Phase 1, but every fact is written
  back into the unified ``evidence`` list so Phase 4 can verify it.

Input: user_query, intent, route, sql_result, retrieved_context, evidence.
Output: answer (data and knowledge sections kept distinct + citations),
status.
"""

from __future__ import annotations

import json
from typing import Any

from src.core.exceptions import LLMError
from src.core.llm_client import LLMClient
from src.core.observability import observe
from src.core.state import AgentState, ErrorRecord


class AnswerGenerationNode:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    @observe("answer_generation")
    def run(self, state: AgentState) -> AgentState:
        question = state.get("user_query", "")
        intent = str(state.get("intent", ""))
        route = str(state.get("route", ""))
        sql_result = state.get("sql_result")
        retrieved: list[Any] = list(state.get("retrieved_context") or [])
        errors: list[ErrorRecord] = list(state.get("errors") or [])

        has_sql = sql_result is not None
        has_rag = bool(retrieved)
        if not has_sql and not has_rag:
            # honest empty-evidence answer (no guessing)
            return {
                "answer": _no_evidence_answer(question, route, errors),
                "status": "completed_no_evidence",
                "errors": errors,
            }

        system = _SYSTEM_PROMPT
        user = (
            f"用户问题: {question}\n意图: {intent}\n路由: {route}\n\n"
            f"== 数据查询结果 (SQL) ==\n{_render_sql_block(sql_result)}\n\n"
            f"== 知识库检索结果 (RAG) ==\n{_render_rag_block(retrieved)}\n\n"
            "请按系统要求作答。"
        )
        try:
            answer = self._llm.generate(system, user)
            answer = _plain_answer(answer)
            status = "completed"
        except LLMError as exc:
            # Deterministic fallback: report what we actually have, never invent.
            answer = _fallback_answer(question, intent, route, sql_result, retrieved)
            status = "completed_fallback"
            errors.append(ErrorRecord(stage="answer_generation", message=exc.message))

        return {"answer": answer, "status": status, "errors": errors}


# ---------------------------------------------------------------------------
# Rendering / fallback helpers
# ---------------------------------------------------------------------------


def _render_sql_block(sql_result: Any) -> str:
    if sql_result is None:
        return "(无 SQL 结果)"
    columns = sql_result.columns
    rows = sql_result.rows
    if not rows:
        return "(SQL 执行成功但返回 0 行)"
    head = " | ".join(str(c) for c in columns)
    preview = "\n".join(" | ".join(str(v) for v in row) for row in rows[:10])
    extra = f" …(共 {sql_result.row_count} 行)" if sql_result.row_count > 10 else ""
    return f"{head}\n{preview}{extra}"


def _render_rag_block(chunks: list[Any]) -> str:
    if not chunks:
        return "(无知识库检索结果)"
    lines: list[str] = []
    for c in chunks:
        meta = c.metadata if isinstance(c, dict) else getattr(c, "metadata", {})
        source = (meta or {}).get("title", c.source if hasattr(c, "source") else c.get("source"))
        chunk_id = c.chunk_id if hasattr(c, "chunk_id") else c.get("chunk_id")
        lines.append(f"[{chunk_id}] 来源: {source} (score={c.score:.3f})\n{c.text}")
    return "\n\n".join(lines)


def _plain_answer(raw: str) -> str:
    """The offline backend returns JSON; extract a readable answer."""
    stripped = raw.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
            if isinstance(data, dict) and data.get("text"):
                return str(data["text"])
        except json.JSONDecodeError:
            pass
    return stripped


def _no_evidence_answer(question: str, route: str, errors: list[ErrorRecord]) -> str:
    note = ""
    if errors:
        note = "（执行过程错误: " + "; ".join(e.message for e in errors) + "）"
    return f"无法回答任务「{question}」：路由 {route or '(unknown)'} 没有产生任何数据或文档证据。{note}"


def _fallback_answer(
    question: str, intent: str, route: str, sql_result: Any, chunks: list[Any]
) -> str:
    parts: list[str] = ["（LLM 不可用，以下为基于真实证据的结构化结果，非模型生成）"]
    if sql_result is not None:
        parts.append("## 数据结论（来自 DuckDB 真实查询）\n" + _render_sql_block(sql_result))
    if chunks:
        parts.append("## 知识结论（来自知识库，含来源）\n" + _render_rag_block(chunks))
    if sql_result is None and not chunks:
        parts.append("本任务未能产生数据或知识证据，无法给出有依据的结论。")
    return "\n\n".join(parts)


_SYSTEM_PROMPT = """你是企业 AI 数字员工的答案生成模块。
规则：
1. 只基于提供的【数据查询结果】与【知识库检索结果】作答，绝不编造数据或文档。
2. 数据结论只能来自 SQL 结果；知识结论必须保留来源（doc_id/部门/标题）。
3. 将"数据"与"知识"两类结论分小节展示，清晰区分。
4. 若证据不足以回答，明确说明缺少什么，不要猜测。
5. 用简洁中文作答。"""


__all__ = ["AnswerGenerationNode"]
