"""Tool base - shared protocol and metadata for all tools.

A tool is the "hand" of the digital employee (see docs/ARCHITECTURE.md,
Tool Layer).  Nodes never call tools directly through ad-hoc code - they
go through :class:`ToolRegistry`, which enforces the whitelist and
permissions declared in ``config/tool_registry.yaml``.

Phase 1 implements the base protocol + registry with the two active tools
(``sql`` and ``rag``).  The remaining tools (kg / analysis / chart /
report) keep their placeholder files until their phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.core.exceptions import AppError


@dataclass
class ToolResult:
    """Uniform result envelope for every tool call."""

    tool_name: str
    ok: bool
    data: Any = None
    error: str | None = None
    evidence_ref: str | None = None  # e.g. "sql:SELECT ..." or "milvus:chunk-0001"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_call_record(self) -> dict[str, Any]:
        return {
            "tool": self.tool_name,
            "ok": self.ok,
            "error": self.error,
            "evidence_ref": self.evidence_ref,
        }


class Tool(Protocol):
    """Interface every tool must satisfy.

    Tools declare keyword-only arguments specific to their domain
    (e.g. ``SQLTool.run(sql=...)``, ``RAGTool.run(query=...)``).
    The registry calls them via ``run(**kwargs)``.
    """

    name: str

    def run(self, *args: Any, **kwargs: Any) -> ToolResult: ...


class ToolError(AppError):
    code = "tool_error"


@dataclass
class _ToolRegistration:
    tool: Tool
    enabled: bool = True
    timeout: int = 30
    permissions: dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Whitelist-only tool registry (ADR-005 / governance)."""

    def __init__(self) -> None:
        self._tools: dict[str, _ToolRegistration] = {}

    def register(self, tool: Tool, *, enabled: bool = True, timeout: int = 30,
                 permissions: dict[str, Any] | None = None) -> None:
        if tool.name in self._tools:
            raise ToolError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = _ToolRegistration(
            tool, enabled=enabled, timeout=timeout, permissions=permissions or {}
        )

    def list_enabled(self) -> list[str]:
        return sorted(name for name, reg in self._tools.items() if reg.enabled)

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(
                f"Tool not registered (whitelist violation): {name}",
                details={"registered": self.list_enabled()},
            )
        return self._tools[name].tool

    def is_enabled(self, name: str) -> bool:
        return name in self._tools and self._tools[name].enabled

    def call(self, name: str, **kwargs: Any) -> ToolResult:
        """Invoke a tool with whitelist + enabled checks."""
        reg = self._tools.get(name)
        if reg is None:
            raise ToolError(f"Tool not registered (whitelist violation): {name}")
        if not reg.enabled:
            return ToolResult(tool_name=name, ok=False, error="tool is disabled in registry")
        try:
            return reg.tool.run(**kwargs)
        except AppError:
            raise
        except Exception as exc:  # noqa: BLE001 - tool layer boundary
            return ToolResult(tool_name=name, ok=False, error=f"{type(exc).__name__}: {exc}")


def default_registry() -> ToolRegistry:
    """Build the Phase 1 registry: sql + rag (others added in their phase)."""
    from src.tools.rag_tool import RAGTool
    from src.tools.sql_tool import SQLTool

    registry = ToolRegistry()
    registry.register(RAGTool(), enabled=True, timeout=20,
                      permissions={"read_only": True})
    registry.register(SQLTool(), enabled=True, timeout=30,
                      permissions={"read_only": True, "statement_type": ["SELECT", "WITH"]})
    return registry


__all__ = ["Tool", "ToolError", "ToolRegistry", "ToolResult", "default_registry"]
