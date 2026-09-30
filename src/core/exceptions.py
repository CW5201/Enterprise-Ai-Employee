"""Unified exception hierarchy for the whole system.

Every custom exception inherits from :class:`AppError`, which carries a
machine-readable ``code`` (used by the API layer for structured error
responses) and ``details`` (free-form context for debugging).

Exception taxonomy (see FAILURE_HANDBOOK.md for the failure modes each
maps to):

- :class:`AppError`                base
- :class:`ConfigError`             settings.yaml / environment problems
- :class:`LLMError`                LLM service failure (retryable or not)
- :class:`SQLGuardError`           SQL safety guard rejected the statement
- :class:`SQLExecutionError`       DuckDB execution failure
- :class:`KnowledgeBaseError`      KB document access failure
- :class:`RetrievalError`          RAG retrieval failure
- :class:`EmbeddingUnavailableError` BGE-M3 model cannot be loaded
- :class:`RerankerUnavailableError` BGE-Reranker-v2-M3 cannot be loaded
- :class:`ValidationError`         claim-evidence verification rejection
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for all project-specific exceptions."""

    code = "app_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class ConfigError(AppError):
    """Configuration is missing or invalid."""

    code = "config_error"


class LLMError(AppError):
    """The LLM service could not complete the request."""

    code = "llm_error"

    def __init__(self, message: str, *, retryable: bool = True, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.retryable = retryable


class SQLGuardError(AppError):
    """The SQL safety guard rejected a statement."""

    code = "sql_guard_error"


class SQLExecutionError(AppError):
    """A statement was rejected by or failed on DuckDB."""

    code = "sql_execution_error"


class KnowledgeBaseError(AppError):
    """The knowledge base could not provide the requested documents."""

    code = "knowledge_base_error"


class RetrievalError(AppError):
    """Retrieval failed (index empty, embedding failure, ...)."""

    code = "retrieval_error"


class EmbeddingUnavailableError(RetrievalError):
    """The BGE-M3 embedding model could not be loaded.

    Raised ONLY when the formal (non-test) mode cannot produce a real
    BGE-M3 vector.  The hashing fallback is test-mode only and never
    used as a substitute for the model in production paths.
    """

    code = "embedding_unavailable"


class RerankerUnavailableError(RetrievalError):
    """The BGE-Reranker-v2-M3 model could not be loaded.

    Raised ONLY when the formal (non-test) mode cannot produce a real
    cross-encoder relevance score.  A fake / hash / random scorer is
    test-mode only and must never be a silent substitute for the model.
    """

    code = "reranker_unavailable"


class ValidationError(AppError):
    """Claim-evidence verification rejected the answer."""

    code = "verification_error"


__all__ = [
    "AppError",
    "ConfigError",
    "LLMError",
    "SQLGuardError",
    "SQLExecutionError",
    "KnowledgeBaseError",
    "RetrievalError",
    "EmbeddingUnavailableError",
    "RerankerUnavailableError",
    "ValidationError",
]
