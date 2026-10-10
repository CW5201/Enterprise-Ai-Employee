"""API route: reports - report assembly, listing, retrieval and download.

Phase 6.7: every finished run (``POST /api/chat`` and ``POST /api/chat/stream``)
assembles a Markdown report through :class:`src.tools.report_tool.ReportTool`
and stores it in an in-memory store, mirroring the task store in
``routes/tasks.py`` (no new DB dependency).  Nothing is synthesised here: a
report exists because a run produced it, otherwise the list is empty.

Disclosure rule: a stored report never exposes more than the API response of
the run that produced it did - the chat route stores the same evidence it
returned, the streaming route stores the same sanitised evidence it streamed.

Routes:
- GET /api/reports                - report listing (newest first)
- GET /api/reports/{id}           - one report, incl. the Markdown body
- GET /api/reports/{id}/download  - the same body as a ``.md`` attachment
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

logger = logging.getLogger("eae.api.reports")

router = APIRouter(tags=["reports"])

# In-memory report store.  Capped so a long-lived workbench process cannot
# grow without bound: the oldest report is evicted first.
_reports: dict[str, dict[str, Any]] = {}
_reports_seq = 0
MAX_REPORTS = 200

_TITLE_MAX = 40


class ReportOut(BaseModel):
    """Report summary (no Markdown body) for the listing table."""

    id: str
    title: str
    question: str
    request_id: str
    created_at: str
    format: str = "markdown"
    content_chars: int = 0
    evidence_count: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    conflicts: int = 0


class ReportDetail(ReportOut):
    """Report with its assembled Markdown body."""

    markdown: str = ""


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _title(question: str) -> str:
    """A short, human-readable subject for the listing table."""
    text = " ".join(question.split())
    return text if len(text) <= _TITLE_MAX else text[: _TITLE_MAX - 1] + "…"


def _summary(record: dict[str, Any]) -> dict[str, Any]:
    """Strip the Markdown body (and the ordering key) from a stored record."""
    return {k: v for k, v in record.items() if k not in ("markdown", "seq")}


# Terminal statuses that mean "this run produced an answer worth exporting".
#
# The Phase 4 tail overwrites ``status`` as it runs (answer_generation ->
# claim_extraction -> answer_guard), so the terminal value is a claim / guard
# status, never the answer node's plain "completed".  Gating on
# ``status == "completed"`` therefore disabled the report feature for every
# real run in the app's configuration (phase4=True) — see FH-SSE-004.
_REPORTABLE_STATUSES: frozenset[str] = frozenset({
    "completed",              # answer_generation
    "completed_fallback",     # answer_generation, LLM unavailable
    "executed",               # multi-tool orchestrator (no phase-4 tail)
    "claims_extracted",       # claim_extraction
    "claims_empty",           # claim_extraction: no claims to verify
    "verified",               # answer_guard: every claim supported
    "guarded",                # answer_guard: unsupported / conflicting claims
    "guard_blocked",          # answer_guard: critical claim unsupported
})


def should_record_report(*, status: str, answer: str) -> bool:
    """Whether a finished run should become a report.

    Runs that produced an answer are exported, including ones the guard
    annotated or blocked (the report carries the claim verdicts, which is
    exactly what a reader needs to judge the answer).  Clarification replies
    and failed runs are not: no task was executed, so there is nothing to
    export.
    """
    return bool(answer.strip()) and status in _REPORTABLE_STATUSES


def record_run_report(
    *,
    request_id: str,
    question: str,
    answer: str,
    evidence: list[dict[str, Any]] | None = None,
    verification_summary: dict[str, Any] | None = None,
    charts: list[dict[str, Any]] | None = None,
) -> ReportOut:
    """Assemble a report from one finished run and store it.

    Assembly failures are logged and stored as an empty body rather than
    raised: a broken report must never turn a successful task into a failed
    HTTP response.
    """
    global _reports_seq

    from src.tools.report_tool import ReportTool

    items = list(evidence or [])
    summary = verification_summary or {}
    result = ReportTool().run(
        answer=answer,
        evidence=items,
        charts=charts,
        question=question,
        verification_summary=summary,
    )
    if not result.get("ok"):
        logger.warning("report assembly failed for request %s: %s", request_id, result.get("error"))

    markdown = (result.get("data") or {}).get("report_markdown") or ""
    _reports_seq += 1
    report_id = str(uuid.uuid4())
    record: dict[str, Any] = {
        "id": report_id,
        "seq": _reports_seq,
        "title": _title(question),
        "question": question,
        "request_id": request_id,
        "created_at": _now_iso(),
        "format": "markdown",
        "markdown": markdown,
        "content_chars": len(markdown),
        "evidence_count": len(items),
        "supported_claims": int(summary.get("supported_claims") or 0),
        "unsupported_claims": int(summary.get("unsupported_claims") or 0),
        "conflicts": int(summary.get("conflicts") or 0),
    }
    _reports[report_id] = record

    if len(_reports) > MAX_REPORTS:
        oldest = min(_reports.values(), key=lambda r: r["seq"])
        _reports.pop(oldest["id"], None)

    return ReportOut(**_summary(record))


def _get(report_id: str) -> dict[str, Any]:
    record = _reports.get(report_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"report {report_id!r} not found")
    return record


@router.get("/reports", response_model=list[ReportOut])
def list_reports() -> list[ReportOut]:
    """Return generated reports, newest first."""
    ordered = sorted(_reports.values(), key=lambda r: r["seq"], reverse=True)
    return [ReportOut(**_summary(r)) for r in ordered]


@router.get("/reports/{report_id}", response_model=ReportDetail)
def get_report(report_id: str) -> ReportDetail:
    """Return one report including its assembled Markdown body."""
    record = _get(report_id)
    return ReportDetail(**_summary(record), markdown=record["markdown"])


@router.get("/reports/{report_id}/download")
def download_report(report_id: str) -> PlainTextResponse:
    """Return the report as a downloadable Markdown file."""
    record = _get(report_id)
    return PlainTextResponse(
        record["markdown"],
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="report-{report_id[:8]}.md"',
        },
    )


__all__ = [
    "MAX_REPORTS",
    "ReportDetail",
    "ReportOut",
    "record_run_report",
    "router",
    "should_record_report",
]
