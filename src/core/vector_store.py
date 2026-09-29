"""Vector store — unified interface with two explicitly separated backends.

Phase 2.1 target backend is Milvus (pymilvus ``MilvusClient``):

- :class:`MilvusVectorStore` — formal backend (``backend = "milvus"``).
  Connects to a running Milvus server (URI/token from
  ``config/settings.yaml`` resolved by :mod:`src.core.config_loader`),
  creates the collection schema on first use, and performs real
  dense retrieval.
- :class:`FakeVectorStore` — local in-memory fallback
  (``backend = "fake"``).  **Unit tests / test mode ONLY.**  It is
  selected explicitly (``create_vector_store(backend="fake")`` or
  ``EMBEDDING_FORCE_OFFLINE=1`` in test contexts) and must never be
  silently substituted when a formal Milvus connection fails: a
  formal backend failure raises :class:`RetrievalError`, it does not
  downgrade.

The Milvus collection stores per chunk::

    chunk_id (VARCHAR, PK), document_id, title, source, category,
    text, metadata_json (VARCHAR), vector (FLOAT_VECTOR, dim measured
    from the real BGE-M3 model, never hardcoded)

Search results keep ``source`` / ``title`` / ``category`` so downstream
answer generation can cite them.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.core.embedder import EmbedFn
from src.core.exceptions import RetrievalError

logger = logging.getLogger("eae.vector_store")

_OUTPUT_FIELDS: list[str] = ["chunk_id", "document_id", "title", "source", "category", "text", "metadata_json"]


# ---------------------------------------------------------------------------
# Search hit
# ---------------------------------------------------------------------------


@dataclass
class SearchHit:
    """One retrieved chunk, source always preserved."""

    chunk_id: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    document_id: str = ""
    title: str = ""
    category: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "title": self.title,
            "source": self.source,
            "category": self.category,
            "metadata": self.metadata,
            "score": self.score,
        }


class VectorStore(Protocol):
    """Common retrieval interface shared by the Milvus and fake backends."""

    backend: str

    def connect(self) -> None: ...

    def health_check(self) -> dict[str, Any]: ...

    def create_collection(self) -> None: ...

    def insert(self, records: list[dict[str, Any]]) -> int: ...

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,  # noqa: A002 - mirrors Milvus expr API
    ) -> list[SearchHit]: ...

    def delete(self, chunk_ids: list[str]) -> int: ...

    def size(self) -> int: ...

    def drop(self) -> None: ...


# ---------------------------------------------------------------------------
# Fake backend — UNIT TEST / TEST MODE ONLY
# ---------------------------------------------------------------------------


class FakeVectorStore:
    """In-process cosine index for tests (pre-computed embeddings stored).

    This is an explicit, labelled fake: ``backend = "fake"``.  It exists so
    unit tests stay fast and deterministic; it must NEVER be used as the
    formal retrieval backend.
    """

    backend = "fake"

    def __init__(self, collection: str, dim: int, embed_fn: EmbedFn | None = None) -> None:
        self.collection = collection
        self.dim = dim
        self._embed_fn = embed_fn
        self._records: dict[str, dict[str, Any]] = {}
        self._vectors: dict[str, list[float]] = {}
        self._connected = False

    def connect(self) -> None:
        self._connected = True

    def health_check(self) -> dict[str, Any]:
        return {"backend": "fake", "collection": self.collection, "entity_count": self.size()}

    def create_collection(self) -> None:
        return None

    def insert(self, records: list[dict[str, Any]]) -> int:
        for record in records:
            chunk_id = str(record["chunk_id"])
            self._records[chunk_id] = {k: v for k, v in record.items() if k != "vector"}
            self._vectors[chunk_id] = list(record.get("vector") or [])
        return len(records)

    def search(self, query: str, top_k: int = 5, filter: dict[str, Any] | None = None) -> list[SearchHit]:
        if not self._records:
            return []
        if self._embed_fn is None:
            raise RetrievalError("FakeVectorStore.search requires an embed_fn to vectorise the query")
        qv = self._embed_fn(query)
        hits: list[SearchHit] = []
        for chunk_id, record in self._records.items():
            if filter and any(record.get(k) != v for k, v in filter.items()):
                continue
            rv = self._vectors.get(chunk_id)
            if not rv:
                continue
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    text=str(record.get("text", "")),
                    source=str(record.get("source", "")),
                    metadata=record.get("metadata", {}),
                    score=_cosine(qv, rv),
                    document_id=str(record.get("document_id", "")),
                    title=str(record.get("title", "")),
                    category=str(record.get("category", "")),
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def delete(self, chunk_ids: list[str]) -> int:
        removed = 0
        for chunk_id in chunk_ids:
            if chunk_id in self._records:
                del self._records[chunk_id]
                self._vectors.pop(chunk_id, None)
                removed += 1
        return removed

    def size(self) -> int:
        return len(self._records)

    def drop(self) -> None:
        self._records.clear()
        self._vectors.clear()


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Milvus backend — Phase 2.1 formal retrieval
# ---------------------------------------------------------------------------


class MilvusVectorStore:
    """Milvus-backed store (pymilvus ``MilvusClient``) — the formal backend.

    The dimension is **measured from the real BGE-M3 model**, not hardcoded;
    a schema dimension that disagrees with the model output raises
    :class:`RetrievalError` immediately instead of corrupting retrieval.
    """

    backend = "milvus"

    def __init__(
        self,
        collection: str,
        dim: int,
        *,
        uri: str,
        user: str = "",
        password: str = "",
        metric_type: str = "IP",
        index_type: str = "HNSW",
        index_params: dict[str, int] | None = None,
        embed_fn: EmbedFn | None = None,
    ) -> None:
        from src.core.embedder import create_embedder

        self.collection = collection
        self.dim = dim
        self._uri = uri
        self._user = user
        self._password = password
        self._metric_type = metric_type
        self._index_type = index_type
        self._index_params = index_params or {"M": 16, "efConstruction": 200}
        self._embed_fn = embed_fn or create_embedder()
        self._client: Any = None
        self._collection_ready = False

    # -- lifecycle ----------------------------------------------------------

    def connect(self) -> None:
        if self._client is not None:
            return
        from pymilvus import MilvusClient

        token = f"{self._user}:{self._password}" if self._user or self._password else ""
        try:
            self._client = MilvusClient(uri=self._uri, token=token)
        except Exception as exc:  # noqa: BLE001 - surface as unified retrieval error
            raise RetrievalError(
                f"Milvus connection failed at {self._uri}: {exc}",
                details={"uri": self._uri, "collection": self.collection},
            ) from exc

    def health_check(self) -> dict[str, Any]:
        self.connect()
        if not self._client.has_collection(self.collection):
            return {
                "backend": "milvus",
                "collection": self.collection,
                "uri": self._uri,
                "dimension": self.dim,
                "metric_type": self._metric_type,
                "index_type": self._index_type,
                "entity_count": 0,
                "exists": False,
            }
        self._ensure()
        entity_count = self.size()
        return {
            "backend": "milvus",
            "collection": self.collection,
            "uri": self._uri,
            "dimension": self.dim,
            "metric_type": self._metric_type,
            "index_type": self._index_type,
            "entity_count": entity_count,
        }

    def create_collection(self) -> None:
        """Create the collection + HNSW index if it does not exist yet."""
        self.connect()
        if self._client.has_collection(self.collection):
            self._collection_ready = True
            return
        from pymilvus import CollectionSchema, DataType, FieldSchema
        from pymilvus.milvus_client import IndexParams

        fields: list[dict[str, Any]] = [
            {"name": "chunk_id", "dtype": DataType.VARCHAR, "is_primary": True, "max_length": 128},
            {"name": "document_id", "dtype": DataType.VARCHAR, "max_length": 64},
            {"name": "title", "dtype": DataType.VARCHAR, "max_length": 256},
            {"name": "source", "dtype": DataType.VARCHAR, "max_length": 512},
            {"name": "category", "dtype": DataType.VARCHAR, "max_length": 64},
            {"name": "text", "dtype": DataType.VARCHAR, "max_length": 8192},
            {"name": "metadata_json", "dtype": DataType.VARCHAR, "max_length": 8192},
            {"name": "vector", "dtype": DataType.FLOAT_VECTOR, "dim": self.dim},
        ]
        schema = CollectionSchema(fields=[FieldSchema(**f) for f in fields])
        index_params = IndexParams()
        index_params.add_index(
            "vector",
            index_type=self._index_type,
            metric_type=self._metric_type,
            **self._index_params,
        )
        self._client.create_collection(self.collection, schema=schema, index_params=index_params)
        self._client.load_collection(self.collection)
        self._collection_ready = True

    def _ensure(self) -> None:
        """Make sure the collection exists and is loaded for search."""
        if not self._client.has_collection(self.collection):
            self.create_collection()
        elif not self._collection_ready:
            self._client.load_collection(self.collection)
            self._collection_ready = True

    def insert(self, records: list[dict[str, Any]]) -> int:
        self._ensure()
        if not records:
            return 0
        data: list[dict[str, Any]] = []
        for record in records:
            vector = record.get("vector") or []
            if len(vector) != self.dim:
                logger.error(
                    "Embedding dim %d != collection dim %d (%s); refusing to insert mismatched vectors",
                    len(vector), self.dim, self.collection,
                )
                raise RetrievalError(
                    f"Vector dimension mismatch: got {len(vector)}, collection '{self.collection}' "
                    f"is {self.dim}",
                    details={"record_chunk_id": record.get("chunk_id"), "expected": self.dim, "got": len(vector)},
                )
            data.append(
                {
                    "chunk_id": str(record["chunk_id"]),
                    "document_id": str(record.get("document_id", "")),
                    "title": str(record.get("title", "")),
                    "source": str(record.get("source", "")),
                    "category": str(record.get("category", "")),
                    "text": str(record.get("text", "")),
                    "metadata_json": json.dumps(record.get("metadata", {}), ensure_ascii=False),
                    "vector": [float(v) for v in vector],
                }
            )
        self._client.insert(self.collection, data)
        self._client.flush(self.collection)
        return len(data)

    def search(self, query: str, top_k: int = 5, filter: dict[str, Any] | None = None) -> list[SearchHit]:
        self._ensure()
        if top_k < 1:
            raise RetrievalError(f"top_k must be >= 1, got {top_k}")
        vector = [float(v) for v in self._embed_fn(query)]
        if len(vector) != self.dim:
            raise RetrievalError(
                f"Query vector dimension {len(vector)} != collection dimension {self.dim}",
                details={"collection": self.collection},
            )
        expr = _build_filter_expr(filter)
        results = self._client.search(
            self.collection,
            data=[vector],
            limit=int(top_k),
            filter=expr,
            output_fields=_OUTPUT_FIELDS,
            search_params={"metric_type": self._metric_type},
        )
        hits: list[SearchHit] = []
        for row in results[0]:
            ent = row.get("entity", row)
            meta: dict[str, Any] = {}
            try:
                meta = json.loads(ent.get("metadata_json") or "{}")
            except json.JSONDecodeError:
                meta = {}
            hits.append(
                SearchHit(
                    chunk_id=str(ent.get("chunk_id", row.get("id", ""))),
                    document_id=str(ent.get("document_id", "")),
                    text=str(ent.get("text", "")),
                    title=str(ent.get("title", "")),
                    source=str(ent.get("source", "")),
                    category=str(ent.get("category", "")),
                    metadata=meta,
                    score=float(row.get("distance", 0.0)),
                )
            )
        return hits

    def delete(self, chunk_ids: list[str]) -> int:
        self._ensure()
        if not chunk_ids:
            return 0
        self._client.delete(self.collection, ids=chunk_ids)
        self._client.flush(self.collection)
        return len(chunk_ids)

    def size(self) -> int:
        self._ensure()
        stats = self._client.get_collection_stats(self.collection)
        return int(stats.get("row_count", 0))

    def drop(self) -> None:
        self.connect()
        if self._client.has_collection(self.collection):
            self._client.drop_collection(self.collection)
        self._collection_ready = False


def _build_filter_expr(filter: dict[str, Any] | None) -> str | None:
    """Translate a simple equality filter dict into a Milvus expr."""
    if not filter:
        return None
    parts: list[str] = []
    for key, value in filter.items():
        if key == "vector":
            continue
        parts.append(f'{key} == "{value}"')
    return " and ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_vector_store(
    backend: str | None = None,
    *,
    dim: int | None = None,
    collection: str | None = None,
) -> MilvusVectorStore | FakeVectorStore:
    """Build a vector store for an explicitly chosen backend.

    - ``backend="milvus"`` (default): formal backend.  Connection or
      schema problems RAISE :class:`RetrievalError` — a formal failure
      never silently downgrades to the fake backend.
    - ``backend="fake"``: test-mode only in-memory store.

    ``dim`` is read from the real embedder when not given, so the Milvus
    schema dimension always matches BGE-M3's actual output.
    """
    from src.core.config_loader import get_settings

    settings = get_settings()
    milvus_cfg = settings.raw.get("milvus", {})
    collection_name = collection or str(milvus_cfg.get("collection", "enterprise_knowledge"))
    chosen = backend or "milvus"
    if chosen not in ("milvus", "fake"):
        raise RetrievalError(f"Unknown vector store backend: {chosen!r} (expected 'milvus' or 'fake')")

    if chosen == "fake":
        fake_dim = dim
        if fake_dim is None:
            from src.core.embedder import HashEmbedder

            fake_dim = HashEmbedder().dim
        return FakeVectorStore(collection_name, int(fake_dim))

    from src.core.embedder import create_embedder

    embed_fn = create_embedder()
    measured_dim = dim
    if measured_dim is None:
        measured_dim = embed_fn.dim
        # Probe the real model once to detect the actual output dimension.
        probe = embed_fn("dimension probe")
        if len(probe):
            measured_dim = len(probe)
    host = str(milvus_cfg.get("host", "localhost"))
    port = str(milvus_cfg.get("port", "19530"))
    # host may already carry a scheme when the whole URI is given in config
    uri = host if host.startswith(("http://", "https://")) else f"http://{host}:{port}"
    store = MilvusVectorStore(
        collection_name,
        int(measured_dim),
        uri=uri,
        user=str(milvus_cfg.get("user", "")),
        password=str(milvus_cfg.get("password", "")),
        metric_type=str(milvus_cfg.get("metric_type", "IP")),
        index_type=str(milvus_cfg.get("index_type", "HNSW")),
        index_params=dict(milvus_cfg.get("index_params") or {}),
        embed_fn=embed_fn,
    )
    # Formal backend: connect eagerly so misconfiguration fails now,
    # never at first retrieval.
    store.connect()
    return store


__all__ = [
    "FakeVectorStore",
    "MilvusVectorStore",
    "SearchHit",
    "VectorStore",
    "create_vector_store",
]
