"""Phase 6.3 SSE streaming endpoint test.

The LangGraph pipeline is not executed here (no LLM / Milvus in CI);
instead ``employee`` is monkeypatched with a fake whose ``_graph.stream``
yields the same per-node update dicts the real pipeline produces, so the
test asserts the SSE event contract the frontend relies on:
- event sequence (task_started -> ... -> completed)
- structured payload fields
- no secrets / stack traces in payloads
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient


class _FakeGraph:
    def __init__(self, updates: list[dict[str, dict[str, Any]]]) -> None:
        self._updates = updates

    def stream(self, state: dict[str, Any], stream_mode: str = "updates") -> Iterator[dict[str, dict[str, Any]]]:
        assert stream_mode == "updates"
        # Faithfully mimic LangGraph: the caller's dict is NOT mutated —
        # LangGraph accumulates internally.  The caller must fold the
        # deltas themselves to obtain the final state.  This is exactly
        # the semantics the SSE route has to rely on; a fake that
        # mutates `state` in place would mask that contract.
        yield from self._updates


class _FakeEmployee:
    def __init__(self, updates: list[dict[str, dict[str, Any]]]) -> None:
        self._graph = _FakeGraph(updates)


def _sse_events(text: str) -> list[dict[str, Any]]:
    """Parse SSE `data:` frames from a StreamingResponse body string."""
    out: list[dict[str, Any]] = []
    for block in text.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data:"):
                out.append(json.loads(line[5:].strip()))
    return out


@pytest.fixture()
def client_with_fake_employee() -> tuple[TestClient, list[dict[str, dict[str, Any]]]]:
    os.environ["EAE_RAG_BACKEND"] = "fake"
    import src.api.main as api_main

    updates = [
        {"intent": {"intent": "sales_query", "confidence": 0.9, "plan": "query sales"}},
        {"route": {"route": "sql", "route_confidence": 0.95, "tools_selected": ["sql"],
                    "routing_decision": {}, "routing_trace": {}}},
        {"sql": {"sql": "SELECT 1", "sql_result": {"success": True, "data": [[1]]},
                  "tool_calls": [{"tool": "sql", "ok": True, "input": {}, "duration_ms": 10, "audit_id": "a1"}],
                  "evidence": [], "errors": []}},
        {"answer": {"answer": "1 行结果", "status": "completed",
                     "latency": {"total_ms": 42.0}}},
    ]
    fake = _FakeEmployee(updates)

    app = api_main.create_app()
    app.state.employee = fake
    return TestClient(app, raise_server_exceptions=False), updates


def test_sse_event_sequence(client_with_fake_employee) -> None:
    client, _ = client_with_fake_employee
    r = client.post("/api/chat/stream", json={"message": "统计销售额"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    events = _sse_events(r.text)
    types = [e["type"] for e in events]
    assert types[0] == "task_started"
    assert types[-1] == "completed"
    assert "routing" in types
    assert "tool_started" in types
    assert "tool_completed" in types
    # every event is structured
    for e in events:
        assert {"type", "timestamp", "step", "status", "payload"} <= set(e)
        # no secrets or stack traces leak
        blob = json.dumps(e["payload"])
        for bad in ("password", "stack", "Traceback"):
            assert bad not in blob


def test_sse_completed_payload(client_with_fake_employee) -> None:
    client, _ = client_with_fake_employee
    events = _sse_events(client.post("/api/chat/stream", json={"message": "x"}).text)
    completed = next(e for e in events if e["type"] == "completed")
    p = completed["payload"]
    assert p["answer"] == "1 行结果"
    assert p["status"] == "completed"
    assert p["route"] == "sql"
    assert p["latency"]["total_ms"] == 42.0


def test_sse_rejects_empty_message(client_with_fake_employee) -> None:
    client, _ = client_with_fake_employee
    assert client.post("/api/chat/stream", json={"message": "   "}).status_code == 400
