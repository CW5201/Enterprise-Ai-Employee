"""API route: evaluation - read the Phase 5 evaluation artifacts.

Phase 6.6: expose the baseline / ablation summary JSON so the Vue3
evaluation dashboard can render real numbers.  No hardcoding in the
frontend; all data comes from the artifact files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter

router = APIRouter(tags=["evaluation"])
logger = logging.getLogger("eae.api.evaluation")

# Project root = 3 levels up from this file: routes -> api -> src -> root
_ROOT = Path(__file__).resolve().parents[3]
_PHASE5 = _ROOT / "artifacts" / "phase5" / "baseline_ablation_summary.json"
_PHASE_REPORT = _ROOT / "docs" / "phase5" / "EXPERIMENT_REPORT.md"
_PHASE4 = _ROOT / "artifacts" / "phase4" / "verification_summary.json"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        logger.exception("failed to load %s", path)
        return None


def _md_summary(text: str, max_chars: int = 6000) -> str:
    """First N chars of the report, capped so the API stays cheap."""
    return text[:max_chars]


@router.get("/evaluation/summary")
def evaluation_summary() -> dict[str, Any]:
    """Return the Phase 5 baseline/ablation + Phase 4 verification data."""
    result: dict[str, Any] = {
        "baselines": {},
        "ablations": {},
        "verification": {},
        "available": False,
        "report_path": str(_PHASE_REPORT.relative_to(_ROOT)) if _PHASE_REPORT.is_file() else None,
        "report_excerpt": None,
    }

    p5 = _load_json(_PHASE5)
    if p5:
        systems: dict[str, Any] = p5.get("systems", {})
        for name, data in systems.items():
            summary = data.get("summary", {}).get("overall", {})
            config = data.get("config", {})
            kind = config.get("kind", "system")
            if kind == "ablation":
                result["ablations"][name] = {"metrics": summary, "config": config}
            else:
                # kind "system" (full) and "baseline" both belong to the
                # baseline group from the dashboard's perspective.
                result["baselines"][name] = {"metrics": summary, "config": config}
        result["available"] = True

    p4 = _load_json(_PHASE4)
    if p4:
        result["verification"] = p4

    if _PHASE_REPORT.is_file():
        try:
            result["report_excerpt"] = _md_summary(_PHASE_REPORT.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            logger.exception("failed to read experiment report")

    return result


__all__ = ["router"]
