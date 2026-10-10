"""Real-pipeline SSE + report tests (FH-SSE-001 / FH-SSE-002 / FH-SSE-003).

``test_phase6_sse.py`` drives the route with hand-written *dict* deltas.  That
fake hides the two defects these tests exist for:

1. the real nodes write Pydantic models (``EvidenceItem`` / ``ToolCallRecord``)
   into the state, and the SSE encoder used to call ``.items()`` on them —
   pydantic v2 models have no ``.items()``, so every evidence-producing run
   raised inside the generator and the client got an ``error`` event instead
   of ``completed`` (FH-SSE-001);
2. the same models went out through ``json.dumps(default=str)``, so payload
   fields arrived at the browser as stringified reprs instead of objects
   (FH-SSE-002);
3. the route folded node deltas with ``final_state[k] = v``, which *overwrites*
   the append channels instead of merging them, so the stored report kept only
   the last writer's evidence (FH-SSE-003).

The tests therefore drive the real graph (fake RAG backend + offline LLM) and
assert on model-shaped data all the way to the report store.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

# A knowledge question: routes to RAG (fake backend), produces real
# EvidenceItem models, and contains no ambiguity-register phrasing.
_QUESTION = "远程办公有哪些要求？"


@pytest.fixture(scope="module")
def client() -> TestClient:
    os.environ["EAE_RAG_BACKEND"] = "fake"
    os.environ["LLM_FORCE_OFFLINE"] = "1"
    os.environ.pop("NEO4J_PASSWORD", None)
    from src.api.main import create_app

    return TestClient(create_app())


@pytest.fixture(autouse=True)
def _clean_stores() -> Iterator[None]:
    """The report / task stores are process-global; keep tests independent."""
    from src.api.routes import reports as reports_route
    from src.api.routes import tasks as tasks_route

    reports_route._reports.clear()
    tasks_route._tasks.clear()
    yield
    reports_route._reports.clear()
    tasks_route._tasks.clear()


def _events(text: str) -> list[dict]:
    out: list[dict] = []
    for block in text.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data:"):
                out.append(json.loads(line[5:].strip()))
    return out


def _stream(client: TestClient, question: str = _QUESTION) -> list[dict]:
    r = client.post("/api/chat/stream", json={"message": question})
    assert r.status_code == 200
    return _events(r.text)


@pytest.mark.unit
def test_append_channels_cover_the_state_reducers() -> None:
    """FH-SSE-003 guard: the fold helper must know the append channels.

    ``AgentState`` uses ``from __future__ import annotations``, so a naive
    ``__annotations__`` scan yields an *empty* set and every list channel would
    silently go back to being overwritten.  Pinning the derived set here keeps
    that from returning unnoticed.
    """
    from src.core.state import APPEND_CHANNELS

    for name in (
        "conversation",
        "entities",
        "tools_selected",
        "tool_calls",
        "errors",
        "retrieved_context",
        "evidence",
        "claims",
        "evidence_v4",
        "verification_results",
        "tool_results",
    ):
        assert name in APPEND_CHANNELS, f"{name} is an append channel"

    for name in ("answer", "status", "sql", "intent", "verification_summary", "guard_decision"):
        assert name not in APPEND_CHANNELS, f"{name} is last-write-wins"


@pytest.mark.unit
def test_streamed_run_reaches_completed_with_evidence(client: TestClient) -> None:
    """FH-SSE-001: real EvidenceItem models must not break the stream."""
    events = _stream(client)
    errors = [e["payload"].get("message") for e in events if e["type"] == "error"]
    assert not errors, f"stream failed on a real run: {errors}"

    completed = next((e for e in events if e["type"] == "completed"), None)
    assert completed is not None, f"no completed event (types={[e['type'] for e in events]})"
    payload = completed["payload"]

    evidence = payload.get("evidence")
    assert isinstance(evidence, list) and evidence, "completed payload lost the run's evidence"
    assert isinstance(evidence[0], dict), "evidence must be a JSON object, not a stringified model"
    assert evidence[0].get("source_type") and evidence[0].get("evidence_id")

    tool_calls = payload.get("tool_calls")
    assert isinstance(tool_calls, list) and tool_calls
    assert isinstance(tool_calls[0], dict), "tool_calls must be JSON objects"
    assert tool_calls[0].get("tool")
    # raw row data / chunk text / payload stay out of the browser projection
    assert "payload" not in evidence[0] and "content" not in evidence[0]


@pytest.mark.unit
def test_streamed_run_records_a_report_with_all_its_evidence(client: TestClient) -> None:
    """FH-SSE-003 + report gate: a finished streamed run is exportable."""
    events = _stream(client)
    payload = next(e for e in events if e["type"] == "completed")["payload"]

    listing = client.get("/api/reports").json()
    assert listing, "a completed streamed run must be stored as a report"
    report = listing[0]
    assert report["question"] == _QUESTION
    assert report["evidence_count"] == len(payload["evidence"]), (
        "the report must carry every evidence item the run produced "
        f"(report={report['evidence_count']}, stream={len(payload['evidence'])})"
    )

    detail = client.get(f"/api/reports/{report['id']}").json()
    assert detail["markdown"].strip()
    assert payload["answer"] in detail["markdown"]


@pytest.mark.unit
def test_streamed_run_lands_in_task_history(client: TestClient) -> None:
    """FH-TASK-001: the default (streaming) mode must fill 任务历史 too.

    Only ``POST /api/tasks`` used to write the history, so a workbench user
    running tasks in the default mode saw an empty 任务历史 while the report
    centre listed the very same runs.
    """
    events = _stream(client)
    payload = next(e for e in events if e["type"] == "completed")["payload"]

    history = client.get("/api/tasks").json()
    assert history, "a streamed run must appear in the task history"
    assert history[0]["question"] == _QUESTION
    assert history[0]["status"] == "completed"
    assert history[0]["answer"] == payload["answer"]


@pytest.mark.unit
def test_chat_run_records_a_report(client: TestClient) -> None:
    """The same report gate applies to POST /api/chat."""
    body = client.post("/api/chat", json={"message": _QUESTION}).json()
    assert body["answer"]

    listing = client.get("/api/reports").json()
    assert listing, f"a finished run (status={body['status']!r}) must be stored as a report"
    assert listing[0]["evidence_count"] == len(body["evidence"])
