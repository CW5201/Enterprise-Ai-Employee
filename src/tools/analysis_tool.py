"""Tool: Analysis — controlled statistical operations over result tables.

Phase 3 Commit 3.  The analysis tool is a *scheduler* over a fixed set of
whitelisted, parameterised statistics — **no arbitrary Python** is
executable (``analysis.allow_arbitrary_python`` stays ``false`` in
``config/settings.yaml``, and this module has no ``eval``/``exec`` by
construction).

Supported operations (see ``OPERATIONS``):

- ``growth_rate``  — period-over-period relative change of a value column
- ``mean`` / ``median`` / ``std`` / ``min`` / ``max`` / ``count``
- ``top_n``        — the N largest rows by a value column
- ``group_mean``   — mean of a value column grouped by a category column
- ``ratio``        — value column as a percentage of the total

Input is the *previous tool's result table* (a list of row-dicts, e.g.
the SQL tool's ``rows`` re-mapped onto ``columns``) plus an operation
spec; output is a uniform :class:`src.tools.base.ToolResult` whose
``data`` carries ``{operation, result, input_rows, latency_ms}`` so the
answer node and Phase 5 evaluation can inspect it.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

from src.core.observability import observe
from src.tools.base import ToolResult

logger = logging.getLogger("eae.analysis_tool")


# ---------------------------------------------------------------------------
# Operation implementations (pure functions over a DataFrame)
# ---------------------------------------------------------------------------


def _to_frame(rows: list[dict[str, Any]] | list[Any] | Any, columns: list[str] | None = None) -> pd.DataFrame:
    """Accept row-dicts or a (columns, rows) pair from the SQL tool."""
    if isinstance(rows, pd.DataFrame):
        return rows
    if columns and isinstance(rows, list) and rows and not isinstance(rows[0], dict):
        return pd.DataFrame(rows, columns=columns[: len(rows[0])] if isinstance(rows[0], (list, tuple)) else columns)
    return pd.DataFrame(rows or [])


def _op_growth_rate(df: pd.DataFrame, value: str) -> dict[str, Any]:
    if df.empty or len(df) < 2:
        return {"growth_rate": None, "note": "需要至少 2 个周期的数据"}
    vals = pd.to_numeric(df[value], errors="coerce").dropna()
    if len(vals) < 2:
        return {"growth_rate": None, "note": "有效数值不足 2 个"}
    first, last = float(vals.iloc[0]), float(vals.iloc[-1])
    if first == 0:
        return {"growth_rate": None, "note": "首期为 0，增长率无定义"}
    return {"growth_rate": round((last - first) / first, 4), "first": first, "last": last}


def _op_stat(df: pd.DataFrame, value: str, stat: str) -> dict[str, Any]:
    s = pd.to_numeric(df[value], errors="coerce")
    if s.dropna().empty:
        return {stat: None}
    return {stat: float(getattr(s, stat)())}


def _op_top_n(df: pd.DataFrame, value: str, n: int) -> dict[str, Any]:
    s = df.assign(**{value: pd.to_numeric(df[value], errors="coerce")}).dropna(subset=[value])
    top = s.nlargest(min(n, len(s)), value)
    return {"top_n": n, "rows": top.to_dict(orient="records")}


def _op_group_mean(df: pd.DataFrame, value: str, group: str) -> dict[str, Any]:
    s = df.assign(**{value: pd.to_numeric(df[value], errors="coerce")}).dropna(subset=[value])
    return {"group_mean": s.groupby(group)[value].mean().round(4).to_dict()}


def _op_ratio(df: pd.DataFrame, value: str) -> dict[str, Any]:
    s = pd.to_numeric(df[value], errors="coerce")
    total = float(s.sum())
    if total == 0:
        return {"ratios": {}}
    return {"ratios": {str(i): round(float(v) / total, 4) for i, v in s.items()}}


OPERATIONS: dict[str, str] = {
    "growth_rate": "period-over-period relative change of the value column",
    "mean": "arithmetic mean of the value column",
    "median": "median of the value column",
    "std": "sample standard deviation of the value column",
    "min": "minimum of the value column",
    "max": "maximum of the value column",
    "count": "row count",
    "top_n": "N largest rows by the value column (params.n)",
    "group_mean": "mean of value grouped by the category column (params.group_by)",
    "ratio": "share of each row's value in the column total",
}


class AnalysisTool:
    """Whitelisted statistical operations over tabular tool results."""

    name: str = "analysis"
    description: str = (
        "Run controlled statistics (growth rate / mean / top-N / group mean / "
        "ratio …) over a previous tool's result table. Arbitrary Python is "
        "not allowed."
    )
    input_schema: dict[str, Any] = {
        "operation": "str (one of: " + ", ".join(OPERATIONS) + ")",
        "rows": "list of row-dicts (or the previous tool's result data)",
        "columns": "list of column names (optional, for positional rows)",
        "params": "dict (value, n, group_by — per operation)",
    }
    timeout: int = 30

    @observe("analysis_exec")
    def run(
        self,
        *,
        operation: str,
        rows: Any = None,
        columns: list[str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> ToolResult:
        params = params or {}
        t0 = time.perf_counter()
        if operation not in OPERATIONS:
            return ToolResult(
                tool_name=self.name, ok=False,
                error=f"unknown operation {operation!r}; allowed: {sorted(OPERATIONS)}",
            )
        if rows is None:
            return ToolResult(tool_name=self.name, ok=False, error="no input rows; run a data tool first")

        df = _to_frame(rows, columns)
        value = str(params.get("value", ""))
        try:
            if operation == "growth_rate":
                result = _op_growth_rate(df, value or df.columns[0])
            elif operation in ("mean", "median", "std", "min", "max"):
                result = _op_stat(df, value or df.columns[0], operation)
            elif operation == "count":
                result = {"count": int(len(df))}
            elif operation == "top_n":
                result = _op_top_n(df, value or df.columns[0], int(params.get("n", 10)))
            elif operation == "group_mean":
                result = _op_group_mean(
                    df, value or df.columns[0], str(params.get("group_by", df.columns[0]))
                )
            elif operation == "ratio":
                result = _op_ratio(df, value or df.columns[0])
            else:  # pragma: no cover — guarded by the whitelist above
                return ToolResult(tool_name=self.name, ok=False, error=f"unsupported {operation!r}")
        except Exception as exc:  # noqa: BLE001 — structured, no traceback to caller
            logger.exception("analysis op %s failed", operation)
            return ToolResult(
                tool_name=self.name, ok=False,
                error=f"analysis failed: {exc}",
                metadata={"operation": operation},
            )

        latency = round((time.perf_counter() - t0) * 1000.0, 2)
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data={
                "operation": operation,
                "result": result,
                "input_rows": int(len(df)),
                "latency_ms": latency,
            },
            evidence_ref=f"analysis:{operation}",
            metadata={"operation": operation},
        )

    @staticmethod
    def available_operations() -> dict[str, str]:
        return dict(OPERATIONS)


__all__ = ["AnalysisTool", "OPERATIONS"]
