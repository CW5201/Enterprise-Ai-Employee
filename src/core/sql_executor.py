"""SQL executor — safe, read-only DuckDB execution (Phase 1).

Guarantees:
1. **Read-only**: a ``read_only`` executor never falls back to a read-write
   connection.  If the database file is missing it fails loudly instead of
   silently creating an empty database (which would also poison every other
   read-only connection in the process with a configuration clash).
2. **Single statement**: statements are split on ``;``; anything other than
   exactly one non-empty statement is rejected.  DuckDB's ``execute`` runs
   every statement and returns the last one's result, so without this a
   ``SELECT 1; SET threads=1`` payload would pass a keyword check and still
   mutate process-wide connection configuration.
3. **Statement whitelist**: only ``SELECT`` / ``WITH ... SELECT`` pass;
   INSERT / UPDATE / DELETE / DROP / ALTER / CREATE / SET / ... are rejected
   with :class:`SQLGuardError` *before* execution.
4. **No file/table-function escape hatches**: ``read_csv_auto`` / ``read_text``
   / ``read_parquet`` / ``glob`` / ``sqlite_scan`` and friends turn a
   "read-only analytics" query into arbitrary local file disclosure, so they
   are rejected by name.
5. **Bounded execution**: a real timeout backed by ``Connection.interrupt()``
   plus a max-rows cap keep any LLM-generated statement from exhausting the
   process.  (A ``ThreadPoolExecutor`` timeout is *not* real: its ``__exit__``
   calls ``shutdown(wait=True)``, so the caller still blocks until the
   runaway query finishes.)
6. **Serialized access**: one DuckDB connection is not safe for concurrent
   use from several FastAPI worker threads, so every entry point takes the
   executor lock.
7. **Structured errors**: failures raise :class:`SQLExecutionError`
   (subclass of :class:`AppError`) with machine-readable details.  This
   includes connection failures — a raw ``duckdb.IOException`` escaping would
   surface as a 502 instead of being recorded in ``state["errors"]``.
8. **Timing**: every execution records ``execution_time_ms`` in the
   returned payload.
"""

from __future__ import annotations

import contextlib
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb

from src.core.exceptions import SQLExecutionError, SQLGuardError

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_ALLOWED_LEADING = re.compile(r"^\s*(with|select)\b", re.IGNORECASE)
_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|pragma|call|export|"
    r"install|load|vacuum|backup|restore|set|reset|checkpoint|import|use)\b",
    re.IGNORECASE,
)

# Table functions that read from (or write to) the local filesystem or a
# foreign database.  Allowing them defeats the read-only guarantee: a SELECT
# wrapping ``read_csv_auto('C:/Windows/win.ini')`` returns real file contents.
_FILE_FUNCTIONS = (
    "read_csv", "read_csv_auto", "read_parquet", "read_json", "read_json_auto",
    "read_ndjson", "read_ndjson_auto", "read_text", "read_blob", "read_xlsx",
    "read_arrow", "read_openssl", "glob", "sqlite_scan", "postgres_scan",
    "mysql_scan", "postgres_scan_pushdown", "iceberg_scan", "delta_scan",
    "shape", "st_read", "sniff_csv",
)
_FILE_FUNCTION_RE = re.compile(
    r"\b(" + "|".join(sorted(_FILE_FUNCTIONS, key=len, reverse=True)) + r")\s*\(",
    re.IGNORECASE,
)

# A trailing lone ";" is harmless punctuation; anything else means several
# statements were submitted.
_STATEMENT_SPLIT_RE = re.compile(r";")

# Replaces a quoted literal / comment body so structural checks only ever look
# at SQL skeleton.  Without this the keyword blacklist fires on legitimate
# business text: `WHERE note = 'please update this row'` was rejected.
_MASK = "''"


def _mask_literals(sql: str) -> str:
    """Blank out string literals, dollar-quoted bodies and comments.

    The result keeps every identifier, keyword, paren and ``;`` in place, so
    statement counting and keyword scanning stay exact while user text inside
    quotes can no longer trip the guard.  Escaped quotes (``''``) and nested
    ``/* ... */`` comments are handled.
    """
    out: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch == "-" and sql.startswith("--", i):          # line comment
            j = sql.find("\n", i)
            i = n if j == -1 else j
            out.append(" ")
        elif ch == "/" and sql.startswith("/*", i):        # block comment
            j = sql.find("*/", i + 2)
            i = n if j == -1 else j + 2
            out.append(" ")
        elif ch == "$":                                     # $tag$ ... $tag$
            m = re.match(r"\$[A-Za-z_0-9]*\$", sql[i:])
            if m:
                tag = m.group(0)
                j = sql.find(tag, i + len(tag))
                i = n if j == -1 else j + len(tag)
                out.append(_MASK)
            else:
                out.append(ch)
                i += 1
        elif ch == "'":                                     # 'literal', '' escape
            j = i + 1
            while j < n:
                if sql[j] == "'":
                    if j + 1 < n and sql[j + 1] == "'":
                        j += 2
                        continue
                    break
                j += 1
            i = min(j + 1, n)
            out.append(_MASK)
        elif ch == '"':                                     # "quoted identifier"
            j = sql.find('"', i + 1)
            i = n if j == -1 else j + 1
            out.append(_MASK)
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def default_database_path() -> Path:
    """The runtime database produced by scripts/import_wwi.py."""
    return _PROJECT_ROOT / "data" / "runtime" / "wwi.duckdb"


def guard_sql(sql: str) -> str:
    """Reject anything that is not a single plain read-only SELECT/WITH."""
    raw = sql.strip()
    if not raw:
        raise SQLGuardError("Empty SQL statement")

    # DuckDB executes every statement in the string and returns the result of
    # the last one, so "one SELECT plus one SET" is a config-tampering vector
    # that a leading-keyword check alone cannot catch.  Count on the masked
    # text so a ";" inside a string literal is not mistaken for a separator.
    masked = _mask_literals(raw)
    statements = [s for s in _STATEMENT_SPLIT_RE.split(masked) if s.strip()]
    if len(statements) != 1:
        raise SQLGuardError(
            f"Exactly one statement is allowed (got {len(statements)})",
            details={"statement_count": len(statements)},
        )
    stripped = raw.rstrip(";").strip()

    if not _ALLOWED_LEADING.match(masked):
        first_word = masked.split(None, 1)[0].lower()
        raise SQLGuardError(
            f"Statement must start with SELECT or WITH (got {first_word!r})",
        )
    # Scan the masked text: a keyword or function name that only appears inside
    # a quoted literal is user data, not SQL structure.
    forbidden = sorted({k.upper() for k in _FORBIDDEN_KEYWORDS.findall(masked)})
    if forbidden:
        raise SQLGuardError(
            "Statement contains forbidden keyword(s)",
            details={"keywords": forbidden},
        )
    file_funcs = sorted({m.group(1) for m in _FILE_FUNCTION_RE.finditer(masked)})
    if file_funcs:
        raise SQLGuardError(
            "Statement contains forbidden table function(s)",
            details={"functions": file_funcs},
        )
    return stripped


@dataclass
class SQLExecutor:
    """Thin safe wrapper around one DuckDB connection (read-only when the
    database file exists)."""

    db_path: Path | None = None
    read_only: bool = True
    max_rows: int = 1000
    timeout_seconds: int = 30
    _connection: duckdb.DuckDBPyConnection | None = field(default=None, repr=False, init=False)
    # Serializes every entry point: a single DuckDB connection is not safe to
    # use from the several worker threads FastAPI runs sync routes on, and the
    # lazy first-connect is racy without a lock (two threads can each build a
    # connection and drop one on the floor, or open the same file with
    # different read_only settings).
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, init=False)

    # -- lifecycle ----------------------------------------------------------

    def connect(self) -> None:
        path = self.db_path or default_database_path()
        read_only = self.read_only
        if read_only and not path.exists():
            # Do NOT degrade to read_write: that would create an empty
            # database file, hold the write lock (blocking the import scripts),
            # and make every later read_only open in this process fail with
            # "Can't open a connection to same database file with a different
            # configuration".
            raise SQLExecutionError(
                f"Database file not found: {path}. Run the data import first.",
                details={"db_path": str(path)},
            )
        try:
            self._connection = duckdb.connect(str(path), read_only=read_only)
        except duckdb.Error as exc:
            # A raw IOException here would escape every caller's except
            # clauses and turn into a 500/502 for the whole request.
            raise SQLExecutionError(
                f"Could not open DuckDB database: {exc}",
                details={"db_path": str(path), "duckdb_error": str(exc)},
            ) from exc

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                with contextlib.suppress(duckdb.Error):
                    self._connection.close()
                self._connection = None

    def __enter__(self) -> SQLExecutor:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- execution ----------------------------------------------------------

    def execute(self, sql: str) -> dict[str, Any]:
        """Guard + execute + shape.  Returns the structured result payload.

        Raises SQLGuardError (rejected by the guard) or SQLExecutionError
        (DuckDB failure / timeout) — both carry structured details.
        """
        # Guard first: it is pure and needs no connection, so a rejected
        # statement never touches the database at all.
        safe_sql = guard_sql(sql)

        with self._lock:
            if self._connection is None:
                self.connect()
            assert self._connection is not None

            start = time.perf_counter()
            try:
                payload = self._execute_with_timeout(safe_sql)
            except SQLExecutionError as exc:
                # Refresh the details with the statement that failed so the
                # caller can report *what* was attempted, not just why.
                if "sql" not in exc.details:
                    exc.details["sql"] = safe_sql
                raise
            except duckdb.Error as exc:
                # LLM-generated SQL routinely hits catalog / parser errors.
                # The contract (see the module docstring) is that every
                # failure is an SQLExecutionError carrying details, and every
                # caller (SQLExecutionNode, SQLTool) only catches that type —
                # letting a raw DuckDB exception escape would propagate out of
                # the graph as a 500 instead of being recorded in
                # state["errors"].
                raise SQLExecutionError(
                    f"DuckDB rejected the statement: {exc}",
                    details={"sql": safe_sql, "duckdb_error": str(exc)},
                ) from exc

        payload["execution_time_ms"] = round((time.perf_counter() - start) * 1000.0, 2)
        return payload

    def _execute_with_timeout(self, sql: str) -> dict[str, Any]:
        """Run ``sql`` on a worker thread with a *real* timeout.

        ``future.result(timeout=...)`` only stops waiting; the enclosing
        ``with ThreadPoolExecutor`` then calls ``shutdown(wait=True)`` and
        the caller blocks until the runaway query finishes anyway.  A daemon
        thread plus ``Connection.interrupt()`` actually aborts the statement.
        """
        conn = self._connection
        assert conn is not None

        box: dict[str, Any] = {}
        failure: list[BaseException] = []

        def _run() -> None:
            try:
                cursor = conn.execute(sql)
                columns = [d[0] for d in (cursor.description or [])]
                rows = cursor.fetchmany(self.max_rows + 1)
                truncated = len(rows) > self.max_rows
                if truncated:
                    rows = rows[: self.max_rows]
                box.update({
                    "sql": sql,
                    "columns": columns,
                    "rows": [tuple(r) for r in rows],
                    "row_count": len(rows),
                    "truncated": truncated,
                    "execution_time_ms": 0.0,  # filled in by the caller
                })
            except BaseException as exc:  # noqa: BLE001 - re-raised on join
                failure.append(exc)

        worker = threading.Thread(target=_run, daemon=True, name="duckdb-query")
        worker.start()
        worker.join(self.timeout_seconds)
        if worker.is_alive():
            # Interrupt the running statement so the thread can unwind.
            with contextlib.suppress(duckdb.Error):
                conn.interrupt()
            # Give it a moment to surface its own error, then report a timeout
            # either way — the caller must not wait on a stuck query.
            worker.join(5)
            raise SQLExecutionError(
                f"SQL query timed out after {self.timeout_seconds}s",
                details={"sql": sql, "timeout_seconds": self.timeout_seconds},
            )
        if failure:
            raise failure[0]
        return box

    def list_tables(self) -> list[str]:
        """User table names (schema introspection / tests)."""
        with self._lock:
            if self._connection is None:
                self.connect()
            assert self._connection is not None
            rows = self._connection.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' AND table_type = 'BASE TABLE' ORDER BY table_name"
            ).fetchall()
        return [r[0] for r in rows]


__all__ = ["SQLExecutor", "default_database_path", "guard_sql"]
