"""API route: reports - report retrieval.

Phase 6.5: report listing.  Report generation is not yet implemented
(see src/tools/report_tool.py); this endpoint returns an empty list so
the frontend can show a correct empty state.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["reports"])


@router.get("/reports")
def list_reports() -> list[dict]:
    """Return generated reports. Empty until report generation lands."""
    return []


@router.get("/reports/{report_id}")
def get_report(report_id: str) -> dict:
    """Return one report by id."""
    raise HTTPException(status_code=404, detail=f"report {report_id!r} not found")


__all__ = ["router"]
