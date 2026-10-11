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


# --- latency contract (FH-LAT-001) -----------------------------------------
#
# The graph writes per-stage latencies into ``state["latency"]`` via the
# ``@observe`` decorators, and the run's end-to-end time under the key the
# API, the task history and the frontend all read: ``total_ms``.  Hand-written
# dict deltas (``test_phase6_sse.py``) fake that table, so they can never
# catch a route that forgets to bind it — which is exactly what the streaming
# path used to do: it called ``employee._graph.stream()`` directly, outside
# ``DigitalEmployee.run()``, so no table was bound and the browser always
# rendered 耗时 "—".
#
# Two defects are pinned here:
#   1. the key name (``total_ms``, never ``total``) — the eval aggregator reads
#      the same key off ``record["latency_ms"]``;
#   2. the binding must survive every SSE ``__next__()``: Starlette drives the
#      body with ``iterate_in_threadpool()``, which copies the request task's
#      contextvars on *each* call, so a ``set_task_latencies()`` performed
#      inside the generator only reached the first node.


@pytest.mark.unit
def test_streamed_run_records_every_stage_and_total_ms(client: TestClient) -> None:
    """SSE: the completed payload carries the real latency table."""
    events = _stream(client)
    payload = next(e for e in events if e["type"] == "completed")["payload"]
    latency = payload.get("latency")
    assert isinstance(latency, dict) and latency, "streamed run produced no latency table"

    # the end-to-end key the frontend reads (fmtMs(result.latency.total_ms))
    assert "total_ms" in latency, f"latency table lacks total_ms: {sorted(latency)}"
    assert "total" not in latency, (
        "the run wrote 'total' but the API/frontend read 'total_ms'; "
        "keep the two in sync (see src/graph/builder.py)"
    )
    assert float(latency["total_ms"]) > 0

    # every @observe-decorated node that ran must have left a stage entry,
    # proving the contextvar binding survived each __next__() of the stream
    for stage in ("intent", "routing"):
        assert stage in latency, f"stage {stage!r} not recorded (table={sorted(latency)})"
    assert len([k for k in latency if k != "total_ms"]) >= 2, (
        "only the first node's latency was recorded — the table binding is "
        "lost between SSE __next__() calls"
    )


@pytest.mark.unit
def test_chat_run_records_total_ms(client: TestClient) -> None:
    """Blocking path: POST /api/chat exposes the same end-to-end key."""
    body = client.post("/api/chat", json={"message": _QUESTION}).json()
    latency = body.get("latency")
    assert isinstance(latency, dict) and latency
    assert "total_ms" in latency, f"latency table lacks total_ms: {sorted(latency)}"
    assert float(latency["total_ms"]) > 0


@pytest.mark.unit
def test_task_history_records_a_real_latency(client: TestClient) -> None:
    """任务历史 latency_ms must be the run's real end-to-end time, not 0."""
    client.post("/api/chat", json={"message": _QUESTION})

    history = client.get("/api/tasks").json()
    assert history
    recorded = history[0].get("latency_ms")
    assert recorded, f"task history recorded latency_ms={recorded!r} (expected > 0)"


# --- phase-3 evidence plumbing (FH-EXEC-001 / FH-EXEC-002 / FH-SRC-001) ------
#
# MultiToolExecutionNode.run() used to return only tool_results / tool_calls /
# errors / evidence.  AnswerGenerationNode does not look inside tool_results —
# it reads the ``sql_result`` and ``retrieved_context`` channels directly — so
# with the production wiring (phase3=True, phase4=True) EVERY run reached the
# answer node with no evidence and answered 「无法回答…没有产生任何数据或文档证据」
# even though the tool had succeeded.  The tool's output therefore has to be
# lifted back into those channels by the executor.
#
# A second defect is pinned alongside it: ``evidence`` / ``errors`` are append
# channels, and answer_generation echoed back the full list it had just read
# from state, so every RAG hit and every error was stored twice.

_STATE_QUESTION = "统计订单总数"


def _employee() -> object:
    from src.graph.builder import DigitalEmployee

    return DigitalEmployee(backend="fake", retrieval_mode="hybrid", phase3=True, phase4=True)


@pytest.mark.unit
def test_answer_node_receives_the_executed_tool_output() -> None:
    """The executor must surface tool output via the channels the answer node reads."""
    emp = _employee()

    sql_run = emp.run(_STATE_QUESTION)
    assert sql_run.get("sql_result") is not None, (
        "the sql step succeeded but never reached state['sql_result']; "
        "AnswerGenerationNode only reads that channel, so the run answers "
        "'no evidence' despite a successful query"
    )
    assert "无法回答" not in str(sql_run.get("answer", ""))

    rag_run = emp.run(_QUESTION)
    assert rag_run.get("retrieved_context"), (
        "the rag step returned chunks but they never reached "
        "state['retrieved_context']"
    )
    assert "无法回答" not in str(rag_run.get("answer", ""))


@pytest.mark.unit
def test_evidence_is_not_double_appended() -> None:
    """Append channels must not be echoed back by a node that just read them."""
    state = _employee().run(_QUESTION)

    items = list(state.get("evidence") or [])
    ids = [getattr(e, "evidence_id", None) for e in items]
    assert len(ids) == len(set(ids)), (
        f"the evidence channel holds duplicates ({ids}); a node returned the "
        "append-channel list it had just read from state"
    )


@pytest.mark.unit
def test_answer_sources_survives_the_graph() -> None:
    """``answer_sources`` must be declared in AgentState, else LangGraph drops it."""
    state = _employee().run(_QUESTION)

    sources = state.get("answer_sources")
    assert isinstance(sources, list) and sources, (
        "answer_sources came back empty; it is not declared in AgentState, "
        "so LangGraph silently discards the node's return value and the API "
        "always reports an empty `sources` field"
    )


# --- POST /api/tasks response fidelity (FH-TASK-002) -------------------------
#
# create_task() pre-creates the store entry, then chat() records the finished
# run via record_task(), which REPLATES that dict with a new one.  create_task
# kept writing its result onto the old (now orphaned) object and built the
# response from it, so a clarification run was reported to the client as
# "completed" while the task history correctly said "clarification".


@pytest.mark.unit
def test_create_task_response_matches_the_stored_history(client: TestClient) -> None:
    r = client.post("/api/tasks", json={"question": "asdf"})
    assert r.status_code == 201
    body = r.json()

    stored = client.get(f"/api/tasks/{body['id']}").json()
    assert body["status"] == stored["status"], (
        f"POST /api/tasks reported {body['status']!r} while the task history "
        f"holds {stored['status']!r}; create_task wrote onto a dict that "
        "record_task had already replaced"
    )
