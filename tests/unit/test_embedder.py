"""Unit tests for the BGE-M3 embedder (Phase 2.1).

The formal ``Embedder`` backend loads a real model; its "cannot load" path
is tested with an invalid model source.  The hashing embedder
(``HashEmbedder``) is the test-mode fake, verified here for dimension,
normalisation and determinism — it must never be mistaken for BGE-M3.
"""

from __future__ import annotations

import math
import os

import pytest

# Force fake backend so the tests never touch the real model in CI.
os.environ["EMBEDDING_FORCE_OFFLINE"] = "1"

from src.core.embedder import BGE_M3_DIM, Embedder, HashEmbedder, create_embedder  # noqa: E402
from src.core.exceptions import EmbeddingUnavailableError  # noqa: E402


@pytest.mark.unit
def test_fake_backend_labelled() -> None:
    embedder = create_embedder(force_offline=True)
    assert embedder.backend == "fake"
    assert isinstance(embedder, HashEmbedder)


@pytest.mark.unit
def test_fake_embedding_dimension_and_norm() -> None:
    embedder = HashEmbedder()
    vec = embedder("员工差旅费用报销需要哪些审批？")
    assert len(vec) == BGE_M3_DIM
    norm = math.sqrt(sum(v * v for v in vec))
    assert abs(norm - 1.0) < 1e-9


@pytest.mark.unit
def test_fake_embedding_is_deterministic() -> None:
    embedder = HashEmbedder()
    a = embedder("差旅报销")
    b = embedder("差旅报销")
    assert a == b


@pytest.mark.unit
def test_fake_batch_matches_single() -> None:
    embedder = HashEmbedder()
    texts = ["差旅报销", "采购流程", "信息安全"]
    batch = embedder.embed_batch(texts)
    for t, vec in zip(texts, batch, strict=True):
        assert vec == embedder(t)


@pytest.mark.unit
def test_fake_batch_dimension() -> None:
    embedder = HashEmbedder()
    batch = embedder.embed_batch(["a", "b", "c"])
    assert all(len(v) == BGE_M3_DIM for v in batch)


@pytest.mark.unit
def test_embedder_class_requires_model() -> None:
    """Embedder (the formal backend) must raise when it cannot load a model.

    A non-existent local path must raise EmbeddingUnavailableError — it must
    NEVER silently fall back to hashing.  OFFLINE mode makes the failure
    deterministic and fast (no network retries).
    """
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        with pytest.raises(EmbeddingUnavailableError):
            Embedder(model_source="nonexistent-model-dir-xyz")
    finally:
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.pop("HF_HUB_OFFLINE", None)


@pytest.mark.unit
def test_create_embedder_offline_flag_yields_fake() -> None:
    embedder = create_embedder(force_offline=True)
    vec = embedder("任意文本")
    assert len(vec) == BGE_M3_DIM


@pytest.mark.unit
def test_embedding_result_format() -> None:
    embedder = HashEmbedder()
    vec = embedder("报销制度")
    assert isinstance(vec, list)
    assert all(isinstance(v, float) for v in vec)
    assert len(vec) == BGE_M3_DIM


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
