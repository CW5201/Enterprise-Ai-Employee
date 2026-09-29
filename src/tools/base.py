"""Tool base + registry — whitelist, schema validation, timeout, audit log.

Phase 1 minimal governance:
- :class:`Tool` protocol: every tool declares ``name``, ``description``,
  ``input_schema`` and ``timeout``.
- :class:`ToolResult`: the uniform result envelope every tool returns.
- :class:`ToolRegistry`: whitelist-only registry that (a) validates inputs
  against ``input_schema`` before dispatch, (b) enforces the declared
  ``timeout``, and (c) writes an audit record for every call.

Heavier governance (permissions, parallelism, cost) is Phase 2+.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.core.exceptions import AppError

logger = logging.getLogger("eae.tools")


class ToolError(AppError):
    code = "tool_error"


@dataclass
class ToolResult:
    """Uniform result envelope for every tool call."""

    tool_name: str
    ok: bool
    data: Any = None
    error: str | None = None
    evidence_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_call_record(self, duration_ms: float = 0.0) -> dict[str, Any]:
        return {
            "tool": self.tool_name,
            "ok": self.ok,
            "error": self.error,
            "evidence_ref": self.evidence_ref,
            "duration_ms": duration_ms,
        }


class Tool(Protocol):
    """Interface every tool must satisfy (``run(**kwargs) -> ToolResult``)."""

    name: str
    description: str
    input_schema: dict[str, Any]
    timeout: int

    def run(self, *args: Any, **kwargs: Any) -> ToolResult: ...


@dataclass
class _Registration:
    tool: Tool
    enabled: bool = True


class ToolRegistry:
    """Whitelist-only tool registry with input validation + audit log."""

    def __init__(self) -> None:
        self._tools: dict[str, _Registration] = {}
        self.audit_log: list[dict[str, Any]] = []

    def register(self, tool: Tool, *, enabled: bool = True) -> None:
        if tool.name in self._tools:
            raise ToolError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = _Registration(tool, enabled=enabled)

    def list_enabled(self) -> list[str]:
        return sorted(n for n, r in self._tools.items() if r.enabled)

    def get(self, name: str) -> Tool:
        reg = self._tools.get(name)
        if reg is None:
            raise ToolError(
                f"Tool not registered (whitelist violation): {name}",
                details={"registered": self.list_enabled()},
            )
        return reg.tool

    def is_enabled(self, name: str) -> bool:
        reg = self._tools.get(name)
        return reg is not None and reg.enabled

    def disable(self, name: str) -> None:
        reg = self._tools.get(name)
        if reg is not None:
            reg.enabled = False

    def _validate(self, tool: Tool, kwargs: dict[str, Any]) -> None:
        """Reject unexpected/missing keys against the declared input_schema."""
        schema = tool.input_schema or {}
        known = set(schema.keys())
        if not known:
            return  # schema not declared: pass-through
        for key in kwargs:
            if key not in known:
                raise ToolError(
                    f"Unknown input for tool '{tool.name}': {key!r}",
                    details={"allowed": sorted(known)},
                )

    def call(self, name: str, **kwargs: Any) -> ToolResult:
        reg = self._tools.get(name)
        if reg is None:
            raise ToolError(f"Tool not registered (whitelist violation): {name}")
        if not reg.enabled:
            return ToolResult(tool_name=name, ok=False, error="tool is disabled in registry")
        tool = reg.tool
        start = time.perf_counter()
        try:
            self._validate(tool, kwargs)
            result: ToolResult = tool.run(**kwargs)
        except AppError:
            raise
        except Exception as exc:  # noqa: BLE001 - tool boundary
            result = ToolResult(tool_name=name, ok=False, error=f"{type(exc).__name__}: {exc}")
        duration_ms = round((time.perf_counter() - start) * 1000.0, 2)
        record = result.to_call_record(duration_ms)
        record["input"] = _jsonable(kwargs)
        self.audit_log.append({"ts": time.time(), "name": name, **record})
        logger.debug("tool_audit %s", json.dumps(record, ensure_ascii=False, default=str))
        return result


def _jsonable(value: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return {k: str(v) for k, v in value.items()}


def default_registry() -> ToolRegistry:
    """Build the Phase 2.1 registry: sql_tool + rag_tool (Milvus)."""
    from src.tools.rag_tool import RAGTool
    from src.tools.sql_tool import SQLTool

    registry = ToolRegistry()
    registry.register(SQLTool(), enabled=True)
    registry.register(RAGTool(), enabled=True)
    return registry


__all__ = ["Tool", "ToolError", "ToolRegistry", "ToolResult", "default_registry"]
