"""Observability - structured logging, tracing and latency metrics.

Phase 1 provides the minimum the pipeline needs:

- :func:`observe` - a context manager / decorator that times a named stage
  and stores the latency in the current task's latency table.
- :func:`get_task_latencies` / :func:`set_task_latencies` - the per-task
  latency store (a ``contextvars``-based registry so each graph run has
  its own table).

Stage names used by Phase 1: ``intent``, ``routing``, ``sql_exec``,
``rag_retrieval``, ``answer_generation``, ``llm_chat``.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import time
from typing import Any

logger = logging.getLogger("eae.observability")

_latencies_var: contextvars.ContextVar[dict[str, float] | None] = contextvars.ContextVar(
    "eae_task_latencies", default=None
)


def set_task_latencies(table: dict[str, float]) -> None:
    """Bind a latency table to the current task/trace (call at graph entry)."""
    _latencies_var.set(table)


@contextlib.contextmanager
def observe(stage: str) -> Any:
    """Time ``with observe('intent') as t: ...`` and record it.

    Yields the elapsed seconds; the stage is recorded under the bound
    latency table if one exists.
    """
    start = time.perf_counter()
    logger.debug("stage=%s start", stage)
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        table = _latencies_var.get()
        if table is not None:
            table[stage] = table.get(stage, 0.0) + elapsed_ms
            logger.debug("stage=%s elapsed_ms=%.2f", stage, elapsed_ms)


def get_task_latencies() -> dict[str, float]:
    table = _latencies_var.get()
    return dict(table) if table is not None else {}


def observe_stage(stage: str, func: Any, *args: Any, **kwargs: Any) -> Any:
    """Functional helper: ``result = observe_stage('sql_exec', tool.run, sql)``."""
    with observe(stage):
        return func(*args, **kwargs)


__all__ = [
    "get_task_latencies",
    "observe",
    "observe_stage",
    "set_task_latencies",
]
