"""Node: Answer generation - final answer from fused multi-source evidence.

Phase 1 behaviour:

- consumes the unified ``evidence`` list (duckdb + milvus items);
- asks the LLM to answer using ONLY the provided evidence, with inline
  citations to ``evidence_id``;
- on LLM failure, falls back to a template answer that lists the raw
  evidence, so the pipeline never silently fails.

Claim-Evidence Verification (the research contribution, Innovation 3) is
added in Phase 4 on top of this node.
"""

from __future__ import annotations

import json
from typing import Any

from src.core.exceptions import LLMError
from src.core.llm_client import LLMClient
from src.core.observability import observe
from src.core.state import AgentState, Evidence

_SYSTEM_PROMPT = """你是企业 AI 数字员工的答案生成模块。
只使用提供的证据回答用户任务，不得使用证据之外的知识。
每条事实性结论必须用 [引用: <evidence_id>] 的形式标注来源。
若证据不足以回答，明确说明缺少什么，不要猜测。

证据列表:
{evidence_block}

用户任务: {user_task}"""


class AnswerGenerationNode:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    @observe("answer_generation")
    def run(self, state: AgentState) -> AgentState:
        question = state.get("user_task", "")
        evidence: list[Evidence] = list(state.get("evidence") or [])
        citations = [ev.evidence_id for ev in evidence]

        if not evidence:
            # honest empty-evidence answer (no guessing)
            return {
                "answer": _no_evidence_answer(question, state.get("errors") or []),
                "citations": [],
                "status": "completed_no_evidence",
            }

        block = json.dumps(
            [
                {
                    "id": ev.evidence_id,
                    "source_type": ev.source_type,
                    "content": ev.content,
                    "score": ev.score,
                }
                for ev in evidence
            ],
            ensure_ascii=False,
            indent=1,
        )
        try:
            text = self.llm.chat_text(
                _SYSTEM_PROMPT.format(evidence_block=block, user_task=question),
                "请按要求作答。",
            )
            return {"answer": text, "citations": citations, "status": "completed"}
        except LLMError as exc:
            # deterministic fallback: list the evidence, never fabricate
            return {
                "answer": _fallback_answer(question, evidence, str(exc)),
                "citations": citations,
                "status": "completed_fallback",
            }


def _no_evidence_answer(question: str, errors: list[dict[str, Any]]) -> str:
    note = ""
    if errors:
        note = f"（执行过程中出现的错误: {json.dumps(errors, ensure_ascii=False)}）"
    return f"无法回答任务「{question}」：没有任何可用的数据或文档证据。{note}"


def _fallback_answer(question: str, evidence: list[Evidence], reason: str) -> str:
    lines = [f"LLM 不可用（{reason}），以下为原始证据列表：", f"任务：{question}", ""]
    for ev in evidence:
        lines.append(f"- [{ev.evidence_id}] ({ev.source_type}) {ev.content}")
    return "\n".join(lines)


__all__ = ["AnswerGenerationNode"]
