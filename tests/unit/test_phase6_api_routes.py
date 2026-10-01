"""Phase 6 API route tests (workbench backend surface).

Covers the minimal routes added for the Vue3 workbench:
- POST/GET /api/tasks        (in-memory task store)
- GET  /api/knowledge/*      (Markdown KB scan)
- GET  /api/graph/*         (Neo4j off -> empty lists, no crash)
- GET  /api/reports          (empty until report generation lands)
- GET  /api/evaluation/*    (artifact-backed summary, available flag)

The app is created with ``EAE_RAG_BACKEND=fake`` so Milvus is not
required; the Neo4j client is monkeypatched to None for the graph
endpoints (unconfigured environment).
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client() -> TestClient:
    os.environ["EAE_RAG_BACKEND"] = "fake"
    # Make sure no leftover credentials push the graph route to a real server
    os.environ.pop("NEO4J_PASSWORD", None)
    from src.api.main import create_app

    return TestClient(create_app())


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
    assert client.get("/api/reports").json() == []
    assert client.get("/api/reports/missing").status_code == 404


def test_evaluation_summary_available_flag(client: TestClient) -> None:
    body = client.get("/api/evaluation/summary").json()
    assert {"baselines", "ablations", "verification", "available"} <= set(body)
    # The Phase 5 summary artifact exists in this repo checkout.
    assert body["available"] is True
    assert "A_full" in body["baselines"] or body["baselines"], "at least one baseline entry"
