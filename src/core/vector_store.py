"""Vector store — unified Milvus interface with an in-process fallback.

Phase 1 target is Milvus (pymilvus).  Real deployments connect to a running
Milvus server (``MILVUS_HOST``).  To keep the minimum loop runnable and
testable on a machine without a Milvus server, this module also provides:

- :class:`MilvusVectorStore` — the pymilvus-backed store (used when a server
  is reachable / configured).
- :class:`LocalVectorStore` — an in-process cosine index with the *same
  search API*, persisted to ``data/runtime/wwi.kb.json`` so the RAG path
  still runs end-to-end offline.

The two are selected by :func:`create_vector_store` based on
configuration and reachability; callers only see one search/insert API so
swapping in the real Milvus server later is a config change, not a code
change.  Extension points (BM25 / reranker / metadata filtering) keep their
interfaces here for Phase 2.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from src.core.embedder import EmbedFn, create_embedder
from src.core.exceptions import RetrievalError

_RUNTIME_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "runtime"
_LOCAL_INDEX_FILE = _RUNTIME_DIR / "wwi.kb.json"


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

    def to_payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "metadata": self.metadata,
            "score": self.score,
        }


class VectorStore(Protocol):
    """Common retrieval interface shared by the Milvus and local backends."""

    backend: str

    def insert(self, records: list[dict[str, Any]]) -> int: ...

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,  # noqa: A002 - mirrors Milvus expr API
    ) -> list[SearchHit]: ...

    def size(self) -> int: ...

    def drop(self) -> None: ...


# ---------------------------------------------------------------------------
# In-process local store (offline fallback, same API as Milvus)
# ---------------------------------------------------------------------------


class LocalVectorStore:
    """Cosine similarity index persisted to a JSON sidecar file.

    Stores pre-computed embeddings (from the injected ``embed_fn``) so
    search does not need a model at query time.
    """

    backend = "local"

    def __init__(self, embed_fn: EmbedFn, path: Path | None = None) -> None:
        self._embed_fn = embed_fn
        self._path = path or _LOCAL_INDEX_FILE
        self._records: dict[str, dict[str, Any]] = {}
        self._vectors: dict[str, list[float]] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                for rec in data.get("records", []):
                    self._records[rec["chunk_id"]] = rec
                    self._vectors[rec["chunk_id"]] = rec.get("vector", [])
            except (json.JSONDecodeError, KeyError) as exc:
                raise RetrievalError(f"Local index corrupt: {exc}") from exc

    def _save(self) -> None:
        records = [
            {**r, "vector": self._vectors[r["chunk_id"]]}
            for r in self._records.values()
            if self._vectors.get(r["chunk_id"])
        ]
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({"records": records}, ensure_ascii=False),
            encoding="utf-8",
        )

    def insert(self, records: list[dict[str, Any]]) -> int:
        for record in records:
            chunk_id = record["chunk_id"]
            self._records[chunk_id] = {k: v for k, v in record.items() if k != "vector"}
            self._vectors[chunk_id] = record.get("vector") or self._embed_fn(record.get("text", ""))
        self._save()
        return len(records)

    def search(self, query: str, top_k: int = 5, filter: dict[str, Any] | None = None) -> list[SearchHit]:
        if not self._records:
            return []
        qv = self._embed_fn(query)
        hits: list[SearchHit] = []
        for chunk_id, record in self._records.items():
            if filter and any(record.get(k) != v for k, v in filter.items()):
                continue
            rv = self._vectors.get(chunk_id)
            if not rv:
                continue
            score = _cosine(qv, rv)
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    text=record.get("text", ""),
                    source=str(record.get("source", "")),
                    metadata=record.get("metadata", {}),
                    score=score,
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def size(self) -> int:
        return len(self._records)

    def drop(self) -> None:
        self._records.clear()
        self._vectors.clear()
        if self._path.exists():
            self._path.unlink()


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Milvus-backed store (pymilvus)
# ---------------------------------------------------------------------------


class MilvusVectorStore:
    """pymilvus-backed store — the Phase 2 primary backend.

    Created only when a Milvus server is actually reachable; otherwise the
    factory falls back to :class:`LocalVectorStore` so the pipeline still runs.
    """

    backend = "milvus"

    def __init__(self, collection: str, dim: int, uri: str = "http://localhost:19530") -> None:
        from pymilvus import connections, utility  # type: ignore[import-not-found]

        self._connections = connections
        self._utility = utility
        self._collection_name = collection
        self._dim = dim
        self._uri = uri
        self._collection = None

    # -- lifecycle ----------------------------------------------------------

    def _connect(self) -> None:
        if self._collection is None:
            import pymilvus  # type: ignore[import-not-found]

            handle = self._connections.connect("default", uri=self._uri)
            if handle is None:
                raise RetrievalError(f"Milvus unreachable at {self._uri}")
            self._prepare_collection(pymilvus)

    def _prepare_collection(self, pymilvus: Any) -> None:
        if self._utility.has_collection(self._collection_name):
            self._collection = pymilvus.Collection(self._collection_name)
        else:
            schema = pymilvus.CollectionSchema(fields=[
                {"name": "chunk_id", "datatype": pymilvus.DataType.VARCHAR, "is_primary": True, "max_length": 128},
                {"name": "text", "datatype": pymilvus.DataType.VARCHAR, "max_length": 8192},
                {"name": "source", "datatype": pymilvus.DataType.VARCHAR, "max_length": 512},
                {"name": "metadata_json", "datatype": pymilvus.DataType.VARCHAR, "max_length": 8192},
                {"name": "vector", "datatype": pymilvus.DataType.FLOAT_VECTOR, "dim": self._dim},
            ])
            self._collection = pymilvus.Collection(name=self._collection_name, schema=schema)
            index_params = {"metric_type": "IP", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}}
            self._collection.create_index("vector", index_type="HNSW", **index_params)
        self._collection.load()

    # -- VectorStore interface --------------------------------------------

    def insert(self, records: list[dict[str, Any]]) -> int:

        self._connect()
        if not records:
            return 0
        data = [
            [r["chunk_id"] for r in records],
            [r.get("text", "") for r in records],
            [str(r.get("source", "")) for r in records],
            [json.dumps(r.get("metadata", {}), ensure_ascii=False) for r in records],
            [r.get("vector") or [] for r in records],
        ]
        self._collection.insert(data)
        self._collection.flush()
        return len(records)

    def search(self, query: str, top_k: int = 5, filter: dict[str, Any] | None = None) -> list[SearchHit]:
        self._connect()

        from src.core.embedder import create_embedder

        embedder = create_embedder()
        vector = embedder(query)
        expr = None
        if filter:
            expr = " and ".join(f'metadata_json like "%{k}:{v}"%' for k, v in filter.items())
        res = self._collection.search(
            [vector],
            limit=top_k,
            output_fields=["text", "source", "metadata_json"],
            expr=expr,
        )
        hits: list[SearchHit] = []
        for row in res[0]:
            meta = {}
            try:
                meta = json.loads(row.entity.get("metadata_json") or "{}")
            except json.JSONDecodeError:
                meta = {}
            hits.append(
                SearchHit(
                    chunk_id=row.id,
                    text=row.entity.get("text", ""),
                    source=str(row.entity.get("source", "")),
                    metadata=meta,
                    score=float(row.distance),
                )
            )
        return hits

    def size(self) -> int:
        self._connect()
        return int(self._collection.num_entities)

    def drop(self) -> None:
        from pymilvus import utility  # type: ignore[import-not-found]

        if utility.has_collection(self._collection_name):
            utility.drop_collection(self._collection_name)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_vector_store() -> VectorStore:
    """Pick the best available backend.

    Returns a Milvus store only when a server is reachable; otherwise a
    persistent local store so the RAG pipeline always runs.
    """
    try:
        import socket

        host = "localhost"
        port = 19530
        with socket.create_connection((host, port), timeout=0.5):
            pass
    except OSError:
        # No live Milvus server — fall back to the persistent local store.
        return LocalVectorStore(create_embedder())
    try:
        store: VectorStore = MilvusVectorStore(collection="enterprise_knowledge", dim=1024, uri="http://localhost:19530")
        store.size()  # forces connect + prepare
        return store
    except RetrievalError:
        return LocalVectorStore(create_embedder())


__all__ = [
    "LocalVectorStore",
    "MilvusVectorStore",
    "SearchHit",
    "VectorStore",
    "create_vector_store",
]
