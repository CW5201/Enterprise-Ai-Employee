"""Unit tests for ReportTool - report assembly used by the workbench export.

Phase 1 covered the answer / evidence / chart sections; Phase 6.7 added the
task subject and the Phase 4 verification summary, and the API layer turns
the result into a downloadable Markdown file (see routes/reports.py).
"""

from __future__ import annotations

from src.tools.report_tool import ReportTool


def test_run_assembles_task_answer_evidence_and_verification() -> None:
    out = ReportTool().run(
        answer="最近三个月销售额合计 12345。",
        evidence=[
            {"source_type": "duckdb", "source_ref": "fact_sale", "content": "x" * 300},
        ],
        question="统计最近三个月销售额",
        verification_summary={"supported_claims": 2, "unsupported_claims": 1, "conflicts": 0},
    )

    assert out["ok"] is True
    md = out["data"]["report_markdown"]
    assert "## 任务" in md
    assert "统计最近三个月销售额" in md
    assert "## 答案" in md
    assert "最近三个月销售额合计 12345。" in md
    assert "## 证据来源" in md
    assert "fact_sale" in md
    assert "## 校验结论" in md
    assert out["metadata"]["supported_claims"] == 2
    assert out["metadata"]["unsupported_claims"] == 1
    assert out["metadata"]["evidence_count"] == 1
    # Long evidence excerpts stay truncated; a report is not a data dump.
    assert "x" * 300 not in md


def test_run_without_optional_context_stays_minimal() -> None:
    out = ReportTool().run(answer="无证据回答", evidence=[])

    md = out["data"]["report_markdown"]
    assert "## 答案" in md
    assert "未提供证据" in md
    assert "## 任务" not in md
    assert "## 校验结论" not in md
    assert out["metadata"]["chart_count"] == 0


def test_run_reports_failure_instead_of_raising() -> None:
    out = ReportTool().run(answer=None, evidence=[])  # type: ignore[arg-type]

    assert out["ok"] is False
    assert out["data"] is None
    assert "Report assembly failed" in out["error"]
