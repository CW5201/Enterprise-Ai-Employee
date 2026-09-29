"""Config loader - reads config/*.yaml into typed settings objects.

Phase 1 keeps this deliberately small: it loads ``settings.yaml`` and
``routing_rules.yaml`` into plain dataclasses with environment-variable
overrides.  Anything that touches secrets reads them from ``os.environ``
(never from the yaml files - the yaml uses ``${VAR}`` placeholders that
are resolved here).

Usage::

    from src.core.config_loader import get_settings
    settings = get_settings()
    print(settings.duckdb.path)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml

from src.core.exceptions import ConfigError

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"

_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z0-9_]+)(?::-(.*?))?\}")


def _resolve_env(value: Any) -> Any:
    """Recursively replace ``${VAR}`` / ``${VAR:-default}`` with env values."""
    if isinstance(value, str):

        def _sub(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            found = os.environ.get(name)
            if found is not None:
                return found
            if default is not None:
                return default
            return ""

        return _ENV_VAR_RE.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(
            f"Config file not found: {path}",
            details={"path": str(path)},
        )
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"Config file {path} did not parse to a mapping",
            details={"path": str(path)},
        )
    return cast(dict[str, Any], _resolve_env(data))


# ---------------------------------------------------------------------------
# Typed settings (Phase 1 only reads the sections it needs; the rest of the
# yaml is exposed via ``.raw`` for forward compatibility)
# ---------------------------------------------------------------------------


@dataclass
class DuckDBSettings:
    path: str = "data/duckdb/enterprise.duckdb"
    read_only: bool = True
    max_rows: int = 1000
    timeout_seconds: int = 30
    allow_ddl: bool = False
    allow_dml: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DuckDBSettings:
        return cls(
            path=str(d.get("path", cls.path)),
            read_only=bool(d.get("read_only", cls.read_only)),
            max_rows=int(d.get("max_rows", cls.max_rows)),
            timeout_seconds=int(d.get("timeout_seconds", cls.timeout_seconds)),
        )


@dataclass
class LLMSettings:
    provider: str = "qwen"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout_seconds: int = 60
    max_retries: int = 2

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LLMSettings:
        return cls(
            provider=str(d.get("provider", cls.provider)),
            base_url=str(d.get("base_url", "")),
            api_key=str(d.get("api_key", "")),
            model=str(d.get("model", "")),
            temperature=float(d.get("temperature", cls.temperature)),
            max_tokens=int(d.get("max_tokens", cls.max_tokens)),
            timeout_seconds=int(d.get("timeout_seconds", cls.timeout_seconds)),
            max_retries=int(d.get("max_retries", cls.max_retries)),
        )


@dataclass
class MilvusSettings:
    host: str = "localhost"
    port: int = 19530
    user: str = ""
    password: str = ""
    collection: str = "enterprise_knowledge"
    metric_type: str = "IP"
    index_type: str = "HNSW"
    index_dir: str = "data/kb_index"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MilvusSettings:
        return cls(
            host=str(d.get("host", cls.host)),
            port=int(d.get("port", cls.port)),
            user=str(d.get("user", "")),
            password=str(d.get("password", "")),
            collection=str(d.get("collection", cls.collection)),
            metric_type=str(d.get("metric_type", cls.metric_type)),
            index_type=str(d.get("index_type", cls.index_type)),
        )


@dataclass
class RagSettings:
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RagSettings:
        return cls(
            chunk_size=int(d.get("chunk_size", cls.chunk_size)),
            chunk_overlap=int(d.get("chunk_overlap", cls.chunk_overlap)),
            top_k=int(d.get("top_k", cls.top_k)),
        )


@dataclass
class Settings:
    """Typed view over config/settings.yaml + routing_rules.yaml."""

    app: dict[str, Any] = field(default_factory=dict)
    llm: LLMSettings = field(default_factory=LLMSettings)
    milvus: MilvusSettings = field(default_factory=MilvusSettings)
    rag: RagSettings = field(default_factory=RagSettings)
    duckdb: DuckDBSettings = field(default_factory=DuckDBSettings)
    routing: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def db_path(self) -> Path:
        """Resolve DuckDB path relative to the project root."""
        p = Path(self.duckdb.path)
        return p if p.is_absolute() else _PROJECT_ROOT / p


def get_settings(config_dir: Path | None = None) -> Settings:
    """Load settings.yaml and routing_rules.yaml once per process."""
    config_dir = config_dir or _CONFIG_DIR
    settings_raw = _load_yaml(config_dir / "settings.yaml")
    routing_raw = _load_yaml(config_dir / "routing_rules.yaml")

    def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
        value = raw.get(key, {})
        if not isinstance(value, dict):
            raise ConfigError(
                f"Section '{key}' in settings.yaml is not a mapping",
                details={"section": key},
            )
        return cast(dict[str, Any], value)

    return Settings(
        app=_section(settings_raw, "app"),
        llm=LLMSettings.from_dict(_section(settings_raw, "llm")),
        milvus=MilvusSettings.from_dict(_section(settings_raw, "milvus")),
        rag=RagSettings.from_dict(_section(settings_raw, "rag")),
        duckdb=DuckDBSettings.from_dict(_section(settings_raw, "duckdb")),
        routing=routing_raw,
        raw=settings_raw,
    )


@dataclass
class RoutingRules:
    """Typed view over config/routing_rules.yaml (Phase 1 subset)."""

    default_strategy: str = "clarify"
    min_confidence: float = 0.6
    intent_tools: dict[str, list[str]] = field(default_factory=dict)
    intent_descriptions: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_rules(cls, rules: dict[str, object]) -> RoutingRules:
        intent_entries = rules.get("intent") or []
        intent_tools: dict[str, list[str]] = {}
        intent_desc: dict[str, str] = {}
        if isinstance(intent_entries, list):
            for entry in intent_entries:
                if isinstance(entry, dict) and "name" in entry:
                    name = str(entry["name"])
                    tools = entry.get("tools") or []
                    intent_tools[name] = [str(t) for t in tools] if isinstance(tools, list) else []
                    intent_desc[name] = str(entry.get("description", ""))
        route = rules.get("route") or {}
        route = route if isinstance(route, dict) else {}
        return cls(
            default_strategy=str(route.get("default_strategy", cls.default_strategy)),
            min_confidence=float(route.get("min_confidence", cls.min_confidence)),
            intent_tools=intent_tools,
            intent_descriptions=intent_desc,
        )


def get_routing_rules(settings: Settings | None = None) -> RoutingRules:
    """Load routing rules; reuses an already-loaded Settings if provided."""
    settings = settings or get_settings()
    return RoutingRules.from_rules(settings.routing)


__all__ = [
    "DuckDBSettings",
    "LLMSettings",
    "MilvusSettings",
    "RagSettings",
    "RoutingRules",
    "Settings",
    "get_routing_rules",
    "get_settings",
]
