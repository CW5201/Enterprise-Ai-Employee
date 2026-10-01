"""API route: tasks - task submission and history.

Phase 6.2: in-memory task store (no new DB dependency).
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(tags=["tasks"])

# In-memory task store.  FastAPI sync handlers run in a thread pool,
# so a plain dict is safe under CPython GIL.
_tasks: dict[str, dict[str, Any]] = {}


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
        "question": question,
        "status": "running",
        "created_at": now,
        "completed_at": None,
        "latency_ms": None,
        "route": "",
        "answer": "",
    }
    _tasks[task_id] = record

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

    return TaskOut(**record)


@router.get("/tasks", response_model=list[TaskOut])
def list_tasks() -> list[TaskOut]:
    """Return all submitted tasks, newest first."""
    items = sorted(_tasks.values(), key=lambda t: t["created_at"], reverse=True)
    return [TaskOut(**t) for t in items]


@router.get("/tasks/{task_id}", response_model=TaskOut)
def get_task(task_id: str) -> TaskOut:
    """Return one task by id."""
    t = _tasks.get(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail=f"task {task_id} not found")
    return TaskOut(**t)


__all__ = ["router", "TaskOut"]
