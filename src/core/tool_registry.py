"""Core tool registry — loads the whitelist from config/tool_registry.yaml.

Phase 1: bridge between the YAML whitelist and the runtime
:class:`src.tools.base.ToolRegistry`.  The YAML declares which tools exist,
their input schemas, timeouts and permissions; this module instantiates the
enabled tools and registers them, so the *whitelist is the source of truth*
(a tool not in the YAML can never be called).
"""

from __future__ import annotations

from pathlib import Path

from src.tools.base import ToolRegistry, default_registry

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "tool_registry.yaml"


def _load_whitelist() -> dict[str, dict[str, object]]:
    try:
        import yaml

        with _CONFIG_PATH.open(encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        return {name: cfg for name, cfg in (doc.get("tool_name") or {}).items() if isinstance(cfg, dict)}
    except (FileNotFoundError, ModuleNotFoundError):
        return {}


def build_registry() -> ToolRegistry:
    """Create the Phase 1 registry, filtered by the YAML whitelist.

    The YAML declares rag + sql (plus placeholders for kg/analysis/chart/
    report that are not implemented yet).  Only tools whose registry entry is
    ``enabled: true`` AND that have a real implementation are registered.
    """
    registry = default_registry()  # sql_tool + rag_tool
    whitelist = _load_whitelist()
    for name in list(registry.list_enabled()):
        entry = whitelist.get(name)
        if entry is not None and entry.get("enabled") is False:
            # keep it registered but disabled so whitelist-off is respected
            registry.disable(name)
    return registry


__all__ = ["build_registry"]
