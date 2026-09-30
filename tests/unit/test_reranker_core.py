"""Unit tests for the BGE-Reranker-v2-M3 core (Phase 2.3, Commit 2).

Covers the reranker abstraction and its contract:
- score / rerank / top_k / dedup / ordering / determinism
- empty + invalid inputs
- model-unavailable behaviour (no silent fallback)
- batch scoring

All tests use the deterministic :class:`FakeScorer` (test backend, explicitly
labelled) or a hand-rolled scorer so they run offline — the real
BGE-Reranker-v2-M3 is validated separately by the integration / smoke test.
"""

from __future__ import annotations

import os

import pytest

os.environ["EMBEDDING_FORCE_OFFLINE"] = "1"

from src.core.exceptions import RerankerUnavailableError  # noqa: E402
from src.core.reranker import FakeScorer, Reranker, RerankResult  # noqa: E402

pytestmark = pytest.mark.unit


def _candidates():
    return [
        {"chunk_id": "kb-a", "text": "员工报销单 提交 财务", "source": "finance/finance-0001",
         "title": "报销", "fusion_score": 0.02, "dense_rank": 1, "metadata": {"d": 1}},
        {"chunk_id": "kb-b", "text": "库存 盘点 仓储", "source": "ops/ops-0001",
         "title": "库存", "fusion_score": 0.01, "dense_rank": 3, "metadata": {"d": 2}},
        {"chunk_id": "kb-c", "text": "远程办公 居家 规定", "source": "hr/hr-0001",
         "title": "远程", "fusion_score": 0.03, "dense_rank": 2, "metadata": {"d": 3}},
        {"chunk_id": "kb-a", "text": "员工报销单 提交 财务 审批", "source": "finance/finance-0001",
         "title": "报销", "fusion_score": 0.02, "dense_rank": 1, "metadata": {"d": 1}},
    ]


# ---------------------------------------------------------------------------
# 1. model / interface
# ---------------------------------------------------------------------------


def test_fake_scorer_backend_is_labelled() -> None:
    assert FakeScorer().backend == "fake"
    r = Reranker(FakeScorer())
    assert r.backend == "fake"
    assert r.health_check()["loaded"] is True


def test_rerank_result_payload_fields() -> None:
    res = RerankResult(chunk_id="kb-x", rerank_score=0.5, rerank_rank=1)
    p = res.to_payload()
    for key in ("chunk_id", "rerank_score", "rerank_rank", "dense_score",
                "dense_rank", "bm25_score", "bm25_rank", "fusion_score",
                "text", "source", "title", "metadata"):
        assert key in p


# ---------------------------------------------------------------------------
# 2. score
# ---------------------------------------------------------------------------


def test_score_returns_one_per_doc_in_order() -> None:
    r = Reranker(FakeScorer())
    docs = ["员工报销 财务", "库存 盘点", "远程办公"]
    scores = r.score("员工报销", docs)
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)
    # first doc matches the query, so it should score higher than a disjoint one
    assert scores[0] > scores[2]


def test_score_empty_docs_returns_empty() -> None:
    r = Reranker(FakeScorer())
    assert r.score("任意查询", []) == []


# ---------------------------------------------------------------------------
# 3. rerank
# ---------------------------------------------------------------------------


def test_rerank_orders_by_score_desc() -> None:
    r = Reranker(FakeScorer())
    out = r.rerank("员工报销 财务", _candidates())
    assert [h.rerank_score for h in out] == sorted((h.rerank_score for h in out), reverse=True)
    # ranks are 1-based and strictly increasing in list order
    assert [h.rerank_rank for h in out] == list(range(1, len(out) + 1))


def test_rerank_preserves_context() -> None:
    r = Reranker(FakeScorer())
    out = r.rerank("员工报销 财务", _candidates())
    top = out[0]
    assert top.source == "finance/finance-0001"
    assert top.metadata.get("d") == 1
    # fusion_score is carried through, never overwritten
    assert top.fusion_score == 0.02


# ---------------------------------------------------------------------------
# 4. top_k
# ---------------------------------------------------------------------------


def test_rerank_top_k_truncates() -> None:
    r = Reranker(FakeScorer())
    out = r.rerank("员工报销 财务 库存 盘点 远程", _candidates(), top_k=2)
    assert len(out) == 2
    assert [h.rerank_rank for h in out] == [1, 2]


def test_rerank_top_k_zero_keeps_all() -> None:
    r = Reranker(FakeScorer())
    out = r.rerank("员工报销 财务", _candidates(), top_k=0)
    # 4 candidates -> dedup kb-a -> 3 unique
    assert len(out) == 3


# ---------------------------------------------------------------------------
# 5. empty documents
# ---------------------------------------------------------------------------


def test_rerank_empty_candidates() -> None:
    r = Reranker(FakeScorer())
    assert r.rerank("任意", []) == []


# ---------------------------------------------------------------------------
# 6. duplicate chunks
# ---------------------------------------------------------------------------


def test_rerank_dedupes_duplicate_chunk_ids() -> None:
    r = Reranker(FakeScorer())
    out = r.rerank("员工报销 财务 审批", _candidates())
    ids = [h.chunk_id for h in out]
    assert ids.count("kb-a") == 1
    assert len(ids) == 3  # kb-a (deduped), kb-b, kb-c


# ---------------------------------------------------------------------------
# 7 / 8. ordering + determinism
# ---------------------------------------------------------------------------


def test_rerank_deterministic_for_same_input() -> None:
    r = Reranker(FakeScorer())
    a = r.rerank("员工报销", _candidates())
    b = r.rerank("员工报销", _candidates())
    assert [x.rerank_rank for x in a] == [x.rerank_rank for x in b]
    assert [x.rerank_score for x in a] == [x.rerank_score for x in b]
    assert [x.chunk_id for x in a] == [x.chunk_id for x in b]


def test_score_ordering_matches_input_order() -> None:
    r = Reranker(FakeScorer())
    docs = ["员工报销 财务", "库存", "员工报销"]
    scores = r.score("员工报销", docs)
    assert scores[0] >= scores[2] > scores[1]


# ---------------------------------------------------------------------------
# 9. invalid input
# ---------------------------------------------------------------------------


def test_score_rejects_non_str_query() -> None:
    r = Reranker(FakeScorer())
    with pytest.raises(RerankerUnavailableError):
        r.score(123, ["x"])  # type: ignore[arg-type]


def test_score_rejects_bare_string_documents() -> None:
    r = Reranker(FakeScorer())
    with pytest.raises(RerankerUnavailableError):
        r.score("查询", "员工报销")  # type: ignore[arg-type]


def test_unloaded_reranker_raises_on_score() -> None:
    # A Reranker built with no scorer must NOT auto-fallback to a fake.
    r = Reranker()
    with pytest.raises(RerankerUnavailableError):
        r.score("查询", ["文档"])


# ---------------------------------------------------------------------------
# 10. model unavailable
# ---------------------------------------------------------------------------


def test_load_default_unavailable_model_raises() -> None:
    # Point at a path that cannot possibly load; must raise, not fallback.
    with pytest.raises(RerankerUnavailableError):
        Reranker.load_default(model_source="definitely/does/not/exist", device="cpu")


def test_fake_scorer_batch_records_latency() -> None:
    scorer = FakeScorer()
    _ = scorer("查询", ["文档一", "文档二", "文档三"])
    assert hasattr(scorer, "rerank_latency_ms")


def test_batch_scoring_single_model_call() -> None:
    # A custom scorer that counts invocations proves batch (one call, not a
    # per-document loop).
    calls: list[int] = []

    def scoring(query: str, docs: list[str]) -> list[float]:
        calls.append(len(docs))
        return [1.0 / (i + 1) for i in range(len(docs))]

    r = Reranker(scoring)  # type: ignore[arg-type]
    r.rerank("查询", _candidates()[:3])
    assert calls == [3]  # a single batched call over all 3 docs
