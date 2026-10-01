"""Tool: Report - assemble the final report

Status
------
Phase 1: Basic implementation for assembling the final report with evidence and charts.

See docs/ROADMAP.md for the phase in which this module is implemented,
docs/ARCHITECTURE.md for its role in the task pipeline, and ADR.md for the
design decisions behind it.
"""

from __future__ import annotations

from typing import Any


class ReportTool:
    """Tool for assembling the final structured report with evidence and charts.

    This tool combines the answer, evidence, and optional charts into a
    coherent report format.
    """

    name: str = "report"
    description: str = "Assemble the final structured report with evidence and charts."
    timeout: int = 30

    def __init__(self):
        """Initialize the report tool."""
        pass

    def run(self, *, answer: str, evidence: list[dict[str, Any]], charts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """
        Assemble the final report.

        Args:
            answer: The final answer text
            evidence: List of evidence items from tool executions
            charts: Optional list of chart specifications

        Returns:
            Dictionary containing the report in markdown format and metadata
        """
        try:
            report_md = self._assemble_report(answer, evidence, charts or [])

            return {
                "tool": self.name,
                "ok": True,
                "data": {
                    "report_markdown": report_md,
                    "format": "markdown"
                },
                "evidence_ref": f"report:{len(evidence)}_evidence",
                "metadata": {
                    "evidence_count": len(evidence),
                    "chart_count": len(charts or []),
                    "answer_length": len(answer)
                }
            }
        except Exception as e:
            return {
                "tool": self.name,
                "ok": False,
                "error": f"Report assembly failed: {str(e)}",
                "data": None,
                "evidence_ref": None,
                "metadata": {}
            }

    def _assemble_report(self, answer: str, evidence: list[dict[str, Any]], charts: list[dict[str, Any]]) -> str:
        """Assemble the report in markdown format."""
        lines = []

        # Title
        lines.append("# 企业AI员工报告\n")

        # Answer section
        lines.append("## 答案\n")
        lines.append(answer + "\n\n")

        # Evidence section
        if evidence:
            lines.append("## 证据来源\n")
            for i, ev in enumerate(evidence, 1):
                source_type = ev.get("source_type", "unknown")
                source_ref = ev.get("source_ref", "unknown")
                content = ev.get("content", "")
                # Truncate content if too long
                if len(content) > 200:
                    content = content[:200] + "..."
                lines.append(f"{i}. **来源**: {source_type} ({source_ref})\n")
                lines.append(f"   内容: {content}\n\n")
        else:
            lines.append("## 证据来源\n")
            lines.append("未提供证据。\n\n")

        # Charts section
        if charts:
            lines.append("## 图表\n")
            for i, chart in enumerate(charts, 1):
                chart_type = chart.get("chart_type", "unknown")
                lines.append(f"### 图表 {i}: {chart_type}\n")
                # If the chart contains data that can be displayed, we might show it
                # For now, we'll just note that a chart was generated
                lines.append(f"*图表数据已生成（类型：{chart_type}）*\n\n")
        else:
            # Only add charts section if there are charts
            pass

        # Footer
        lines.append("---\n")
        lines.append(f"*报告生成时间：{self._get_timestamp()}*\n")

        return "".join(lines)

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


__all__: list[str] = ["ReportTool"]
