"""Retry-policy tests for the Qwen backend in src/core/llm_client.py.

The shipped behaviour was a linear ``0.5 * (attempt + 1)`` sleep with three
total attempts — 1.5s of cumulative backoff.  Real rate-limit windows are
seconds to tens of seconds long, so every attempt landed inside the same
window and the pipeline failed with "Qwen request failed after 3 attempts:
429 Too Many Requests" (docs/phase3/EXPERIMENT_REPORT.md).
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import pytest

from src.core import llm_client as lc
from src.core.exceptions import LLMError


def _backend(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> lc._QwenOpenAIBackend:
    """A backend with a stubbed config and no real network access."""
    monkeypatch.setattr(lc, "_llm_config", lambda: {"llm": {
        "base_url": "http://llm.invalid/v1",
        "model": "test-model",
        "timeout_seconds": 5,
        "max_retries": 2,
        **overrides,
    }})
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    return lc._QwenOpenAIBackend()


def _status_error(status: int, headers: dict[str, str] | None = None) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://llm.invalid/v1/chat/completions")
    resp = httpx.Response(status, headers=headers or {}, request=req)
    return httpx.HTTPStatusError(f"{status}", request=req, response=resp)


class _ScriptedClient:
    """Stands in for httpx.Client; replays a scripted list of outcomes."""

    def __init__(self, script: list[Exception | dict[str, Any]]) -> None:
        self.script = list(script)
        self.calls = 0

    def __enter__(self) -> _ScriptedClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def post(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        # Repeat the last scripted outcome once exhausted, so a test that
        # under-scripts still exercises "keeps failing" rather than blowing
        # up on an IndexError inside the mock.
        outcome = self.script[min(self.calls - 1, len(self.script) - 1)]
        if isinstance(outcome, Exception):
            raise outcome

        class _Resp:
            status_code = 200

            @staticmethod
            def raise_for_status() -> None:
                return None

            @staticmethod
            def json() -> dict[str, Any]:
                return outcome

        return _Resp()


def _script_client(monkeypatch: pytest.MonkeyPatch,
                   script: list[Exception | dict[str, Any]]) -> _ScriptedClient:
    holder = _ScriptedClient(script)
    monkeypatch.setattr(lc.httpx, "Client", lambda **kw: holder)
    return holder


@pytest.mark.unit
class TestRetryPolicy:
    def test_rate_limit_gets_a_longer_budget_than_configured_retries(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 429 must outlive the normal attempt count — that is the whole
        point of a rate-limit-specific budget."""
        backend = _backend(monkeypatch)
        base_attempts = backend.max_retries + 1  # 3
        # Script one more than the extended budget so the loop is bounded by
        # max_attempts, not by the mock running dry.
        client = _script_client(
            monkeypatch, [_status_error(429)] * (base_attempts + lc._RATE_LIMIT_EXTRA_RETRIES)
        )
        monkeypatch.setattr(time, "sleep", lambda s: None)

        with pytest.raises(LLMError) as e:
            backend._post({"model": "m", "messages": []})

        assert client.calls > base_attempts, (
            f"429 used only {client.calls} attempts; expected more than {base_attempts}"
        )
        assert client.calls == base_attempts + lc._RATE_LIMIT_EXTRA_RETRIES
        assert "429" in str(e.value)

    def test_rate_limit_then_success_recovers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = _backend(monkeypatch)
        client = _script_client(
            monkeypatch, [_status_error(429), _status_error(429), {"ok": True}]
        )
        monkeypatch.setattr(time, "sleep", lambda s: None)

        assert backend._post({"model": "m"}) == {"ok": True}
        assert client.calls == 3

    def test_permanent_client_error_fails_fast_without_burning_budget(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        """401/400 will never succeed on retry; burning quota on them hides
        the real cause behind "failed after N attempts"."""
        backend = _backend(monkeypatch)
        client = _script_client(monkeypatch, [_status_error(401)])
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", sleeps.append)

        with pytest.raises(LLMError) as e:
            backend._post({"model": "m"})

        assert client.calls == 1
        assert sleeps == [], "a non-retryable status must not sleep"
        assert e.value.retryable is False
        assert "401" in str(e.value)

    def test_server_error_stops_at_the_configured_attempt_count(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = _backend(monkeypatch)
        client = _script_client(monkeypatch, [_status_error(500)] * 10)
        monkeypatch.setattr(time, "sleep", lambda s: None)

        with pytest.raises(LLMError):
            backend._post({"model": "m"})

        assert client.calls == backend.max_retries + 1

    def test_backoff_is_exponential_not_linear(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = _backend(monkeypatch)
        _script_client(monkeypatch, [_status_error(500)] * 4)
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", sleeps.append)
        # Neutralise jitter so the doubling is exact and the assertion is
        # not flaky.
        monkeypatch.setattr(lc.random, "uniform", lambda a, b: 0.0)

        with pytest.raises(LLMError):
            backend._post({"model": "m"})

        # Three attempts sleep twice, between them: 1s then 2s.
        assert sleeps == [1.0, 2.0], f"expected exponential backoff, got {sleeps}"

    def test_retry_after_header_is_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = _backend(monkeypatch)
        client = _script_client(
            monkeypatch,
            [_status_error(429, headers={"Retry-After": "45"}), {"ok": True}],
        )
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", sleeps.append)

        assert backend._post({"model": "m"}) == {"ok": True}
        assert client.calls == 2
        assert sleeps[0] >= 45.0, f"Retry-After not honored: slept {sleeps[0]}s"

    def test_backoff_is_capped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = _backend(monkeypatch, max_retries=20)
        _script_client(monkeypatch, [_status_error(500)] * 20)
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", sleeps.append)

        with pytest.raises(LLMError):
            backend._post({"model": "m"})

        assert max(sleeps) <= lc._BACKOFF_CAP_SECONDS + lc._JITTER_SECONDS


@pytest.mark.unit
class TestParseRetryAfter:
    @pytest.mark.parametrize("value,expected", [
        ("7", 7.0),
        (" 12 ", 12.0),
        ("-3", 0.0),
        (None, None),
        ("", None),
        ("not-a-date", None),
    ])
    def test_delta_seconds(self, value: str | None, expected: float | None) -> None:
        assert lc._parse_retry_after(value) == expected

    def test_http_date_in_the_future(self) -> None:
        from datetime import UTC, datetime, timedelta

        future = (datetime.now(UTC) + timedelta(seconds=30))
        header = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
        parsed = lc._parse_retry_after(header)
        assert parsed is not None
        assert 0 < parsed <= 31

    def test_http_date_in_the_past_clamps_to_zero(self) -> None:
        assert lc._parse_retry_after("Mon, 01 Jan 2001 00:00:00 GMT") == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
