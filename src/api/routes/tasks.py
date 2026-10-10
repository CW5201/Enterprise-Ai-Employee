"""API route: tasks - task submission and history.

Phase 6.2: in-memory task store (no new DB dependency).

Phase 6.7: the store is written by *both* entry points.  ``POST /api/tasks``
(a blocking run) and ``POST /api/chat/stream`` (the workbench's default mode)
each record the run, so 任务历史 reflects what the user actually did — before
this, streamed runs were invisible there while their reports existed
(FH-TASK-001).
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(tags=["tasks"])

# In-memory task store.  FastAPI sync handlers run in a thread pool,
# so a plain dict is safe under CPython GIL.  Capped (oldest evicted first)
# so a long-lived workbench process cannot grow without bound.
_tasks: dict[str, dict[str, Any]] = {}
_tasks_seq = 0
MAX_TASKS = 200


class TaskOut(BaseModel):
    id: str
    question: str
    status: str
    created_at: str
    completed_at: str | None = None
    latency_ms: float | None = None
    route: str = ""
    answer: str = ""


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _next_seq() -> int:
    """Monotonic ordering key.

    ``created_at`` has one-second resolution, so two runs in the same second
    would order non-deterministically; the counter keeps "newest first" true.
    """
    global _tasks_seq
    _tasks_seq += 1
    return _tasks_seq


def _to_out(record: dict[str, Any]) -> TaskOut:
    """Project a stored record onto the response model (drops ``seq``)."""
    return TaskOut(
        id=record["id"],
        question=record["question"],
        status=record["status"],
        created_at=record["created_at"],
        completed_at=record.get("completed_at"),
        latency_ms=record.get("latency_ms"),
        route=record.get("route", ""),
        answer=record.get("answer", ""),
    )


def _evict_oldest() -> None:
    if len(_tasks) > MAX_TASKS:
        oldest = min(_tasks.values(), key=lambda t: t["seq"])
        _tasks.pop(oldest["id"], None)


def record_task(
    *,
    question: str,
    status: str,
    route: str = "",
    answer: str = "",
    latency_ms: float | None = None,
    task_id: str | None = None,
) -> TaskOut:
    """Store one finished run in the task history.

    Called by POST /api/tasks (via :func:`create_task`) and by the SSE route.
    """
    record: dict[str, Any] = {
        "id": task_id or str(uuid.uuid4()),
        "seq": _next_seq(),
        "question": question,
        "status": status,
        "created_at": _now_iso(),
        "completed_at": _now_iso(),
        "latency_ms": latency_ms,
        "route": route,
        "answer": answer,
    }
    _tasks[record["id"]] = record
    _evict_oldest()
    return _to_out(record)


def task_status_for_run(*, status: str, answer: str) -> str:
    """Task-history status for one finished pipeline run.

    A run that produced an answer is ``completed`` (the same predicate the
    report store uses); a clarification reply keeps its own label so the
    history does not misreport it as a failure; everything else is ``failed``.
    """
    from src.api.routes.reports import should_record_report

    if should_record_report(status=status, answer=answer):
        return "completed"
    if status.startswith("clarif"):
        return "clarification"
    return "failed"


@router.post("/tasks", response_model=TaskOut, status_code=201)
def create_task(body: dict[str, Any], request: Request) -> TaskOut:
    """Submit an enterprise task, run it, store the result."""
    from src.api.routes.chat import ChatRequest
    from src.api.routes.chat import chat as _run_chat

    question = (body.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="`question` must be non-empty")

    task_id = str(uuid.uuid4())
    now = _now_iso()

    record: dict[str, Any] = {
        "id": task_id,
        "seq": _next_seq(),
        "question": question,
        "status": "running",
        "created_at": now,
        "completed_at": None,
        "latency_ms": None,
        "route": "",
        "answer": "",
    }
    _tasks[task_id] = record
    _evict_oldest()

    try:
        result = _run_chat(request, ChatRequest(message=question, request_id=task_id))
        record["status"] = "completed"
        record["route"] = result.get("route", "")
        record["answer"] = result.get("answer", "")
        record["latency_ms"] = result.get("latency", {}).get("total_ms", 0.0)
    except Exception:  # noqa: BLE001
        record["status"] = "failed"
    finally:
        record["completed_at"] = _now_iso()

    return _to_out(record)


@router.get("/tasks", response_model=list[TaskOut])
def list_tasks() -> list[TaskOut]:
    """Return all submitted tasks, newest first."""
    items = sorted(_tasks.values(), key=lambda t: t["seq"], reverse=True)
    return [_to_out(t) for t in items]


@router.get("/tasks/{task_id}", response_model=TaskOut)
def get_task(task_id: str) -> TaskOut:
    """Return one task by id."""
    t = _tasks.get(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail=f"task {task_id} not found")
    return _to_out(t)


__all__ = ["MAX_TASKS", "TaskOut", "record_task", "router", "task_status_for_run"]
