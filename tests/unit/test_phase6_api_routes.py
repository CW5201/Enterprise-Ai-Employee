"""Phase 6 API route tests (workbench backend surface).

Covers the minimal routes added for the Vue3 workbench:
- POST/GET /api/tasks        (in-memory task store)
- GET  /api/knowledge/*      (Markdown KB scan)
- GET  /api/graph/*         (Neo4j off -> empty lists, no crash)
- GET  /api/reports          (reports assembled from completed runs)
- GET  /api/evaluation/*    (artifact-backed summary, available flag)

The app is created with ``EAE_RAG_BACKEND=fake`` so Milvus is not
required; the Neo4j client is monkeypatched to None for the graph
endpoints (unconfigured environment).
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client() -> TestClient:
    os.environ["EAE_RAG_BACKEND"] = "fake"
    # Make sure no leftover credentials push the graph route to a real server
    os.environ.pop("NEO4J_PASSWORD", None)
    from src.api.main import create_app

    return TestClient(create_app())


@pytest.fixture(autouse=True)
def _clean_report_store() -> Iterator[None]:
    """The report store is process-global; keep each test independent."""
    from src.api.routes import reports as reports_route

    reports_route._reports.clear()
    yield
    reports_route._reports.clear()


def test_tasks_endpoint_roundtrip(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /api/tasks stores a record; GET returns it; GET /{id} works."""
    # Keep the test hermetic: stub the pipeline so no LLM / Milvus is called.
    import src.api.routes.chat as chat_route

    monkeypatch.setattr(
        chat_route, "chat",
        lambda request, body: {
            "answer": "ok", "route": "sql", "intent": "query",
            "latency": {"total_ms": 12.3},
        },
        raising=False,
    )

    r = client.post("/api/tasks", json={"question": "最近三个月销售额"})
    assert r.status_code == 201, r.text
    tid = r.json()["id"]
    assert r.json()["status"] == "completed"

    listing = client.get("/api/tasks").json()
    assert any(t["id"] == tid for t in listing)

    detail = client.get(f"/api/tasks/{tid}").json()
    assert detail["question"] == "最近三个月销售额"

    assert client.get("/api/tasks/nope").status_code == 404


def test_tasks_rejects_empty_question(client: TestClient) -> None:
    assert client.post("/api/tasks", json={"question": "  "}).status_code == 400


def test_knowledge_documents_lists_kb(client: TestClient) -> None:
    docs = client.get("/api/knowledge/documents").json()
    assert docs, "knowledge base should list the Markdown documents"
    first = docs[0]
    assert set(first) >= {"id", "title", "department", "source", "effective_date", "synthetic"}
    # filter by department
    fin = client.get("/api/knowledge/documents", params={"department": "finance"}).json()
    assert fin and all(d["department"] == "finance" for d in fin)


def test_graph_endpoints_empty_when_neo4j_off(client: TestClient) -> None:
    import src.api.routes.graph as graph_route

    old = graph_route._neo4j_client
    graph_route._neo4j_client = lambda: None
    try:
        assert client.get("/api/graph/entities").json() == []
        assert client.get("/api/graph/relations").json() == []
    finally:
        graph_route._neo4j_client = old


def test_reports_empty_state(client: TestClient) -> None:
    """No run yet -> an empty list (no placeholder rows), 404 for unknown ids."""
    assert client.get("/api/reports").json() == []
    assert client.get("/api/reports/missing").status_code == 404
    assert client.get("/api/reports/missing/download").status_code == 404


def test_reports_listed_newest_first(client: TestClient) -> None:
    """The listing returns the runs that produced reports, newest first."""
    from src.api.routes.reports import record_run_report

    first = record_run_report(request_id="req-1", question="第一个任务", answer="答案 A")
    second = record_run_report(request_id="req-2", question="第二个任务", answer="答案 B")

    listing = client.get("/api/reports").json()
    assert [r["id"] for r in listing] == [second.id, first.id]
    assert "markdown" not in listing[0], "the listing must not ship report bodies"


def test_report_detail_and_download(client: TestClient) -> None:
    """Detail returns the assembled Markdown; download returns it as a file."""
    from src.api.routes.reports import record_run_report

    summary = record_run_report(
        request_id="req-3",
        question="统计最近三个月销售额并分析增长趋势",
        answer="最近三个月销售额合计 12345，环比增长 8%。",
        evidence=[{"source_type": "duckdb", "source_ref": "fact_sale", "content": "行数 92"}],
        verification_summary={"supported_claims": 1, "unsupported_claims": 0, "conflicts": 0},
    )
    assert summary.evidence_count == 1
    assert summary.supported_claims == 1

    detail = client.get(f"/api/reports/{summary.id}").json()
    assert detail["question"] == "统计最近三个月销售额并分析增长趋势"
    assert "## 任务" in detail["markdown"]
    assert "## 答案" in detail["markdown"]
    assert "## 证据来源" in detail["markdown"]
    assert "## 校验结论" in detail["markdown"]
    assert "fact_sale" in detail["markdown"]

    dl = client.get(f"/api/reports/{summary.id}/download")
    assert dl.status_code == 200
    assert dl.headers["content-type"].startswith("text/markdown")
    assert "attachment" in dl.headers["content-disposition"]
    assert "## 答案" in dl.text


def test_chat_route_records_report_for_completed_run(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /api/chat assembles a report out of the run it just executed."""
    import src.graph.builder as builder

    def fake_run(self: object, text: str, request_id: str | None = None) -> dict:
        return {
            "request_id": request_id or "req-chat", "answer": f"答案：{text}",
            "intent": "query", "route": "sql", "sql": "SELECT 1",
            "evidence": [], "tool_calls": [], "status": "completed",
            "latency": {"total_ms": 5.0}, "answer_sources": [],
            "verification_results": [],
            "verification_summary": {"supported_claims": 0, "unsupported_claims": 0,
                                     "conflicts": 0},
            "guard_decision": {},
        }

    monkeypatch.setattr(builder.DigitalEmployee, "run", fake_run, raising=False)
    assert client.post("/api/chat", json={"message": "查一下最近销售额"}).status_code == 200

    reports = client.get("/api/reports").json()
    assert len(reports) == 1
    assert reports[0]["question"] == "查一下最近销售额"
    assert reports[0]["request_id"] == "req-chat"

    # A failed run produced no usable answer, so it is not exported.
    monkeypatch.setattr(
        builder.DigitalEmployee, "run",
        lambda self, text, request_id=None: {
            "request_id": request_id or "req-fail", "answer": "", "status": "failed",
            "evidence": [], "tool_calls": [], "latency": {},
        },
        raising=False,
    )
    assert client.post("/api/chat", json={"message": "再查一次"}).status_code == 200
    assert len(client.get("/api/reports").json()) == 1


def test_evaluation_summary_available_flag(client: TestClient) -> None:
    body = client.get("/api/evaluation/summary").json()
    assert {"baselines", "ablations", "verification", "available",
            "report_path", "report_excerpt"} <= set(body)
    # The Phase 5 summary artifact exists in this repo checkout.
    assert body["available"] is True
    assert "A_full" in body["baselines"]
    assert "abl_w_kg" in body["ablations"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "status",
    [
        "completed",           # answer_generation
        "completed_fallback",  # answer_generation, LLM unavailable
        # An honest "no evidence to cite" still produced an answer.  Leaving
        # it out made the task history record the run as ``failed`` (red)
        # while the result panel rendered the same run as success.
        "completed_no_evidence",
        "executed",            # multi-tool orchestrator (no phase-4 tail)
        "claims_extracted",    # claim_extraction
        "claims_empty",        # claim_extraction, nothing to verify
        "verified",            # answer_guard
        "guarded",             # answer_guard, annotated
        "guard_blocked",       # answer_guard, critical claim unsupported
    ],
)
def test_answer_producing_statuses_are_reportable(status: str) -> None:
    """FH-SSE-004: the report gate must cover every terminal answer status.

    Gating on ``status == "completed"`` alone meant no report was ever stored
    under the app's real configuration (phase4=True), because the verification
    tail overwrites the status with a claim / guard value.
    """
    from src.api.routes.reports import should_record_report

    assert should_record_report(status=status, answer="结论：销售额 12345。")


@pytest.mark.unit
@pytest.mark.parametrize(
    "status",
    [
        "",
        "clarified_out_of_scope",    # clarification reply, no task executed
        "clarify_needed",
        "execution_all_failed",
        "execution_no_plan",
        "claims_extraction_failed",
        "failed",
    ],
)
def test_non_execution_statuses_are_not_reportable(status: str) -> None:
    from src.api.routes.reports import should_record_report

    assert not should_record_report(status=status, answer="我们需要再确认一下您的意图。")


@pytest.mark.unit
def test_empty_answer_is_never_reportable() -> None:
    from src.api.routes.reports import should_record_report

    assert not should_record_report(status="verified", answer="")
    assert not should_record_report(status="verified", answer="   \n ")
