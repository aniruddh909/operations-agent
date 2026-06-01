"""Tests for the transient-retry tier (call_with_retry)."""

from __future__ import annotations

import pytest

from operations_agent.errors import SemanticError, TransientError
from operations_agent.retry import call_with_retry


def _no_sleep(_):
    pass


def test_retries_transient_then_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientError(503, "service unavailable")
        return "ok"

    result = call_with_retry(flaky, max_retries=3, sleep=_no_sleep)
    assert result == "ok"
    assert calls["n"] == 3


def test_gives_up_after_max_retries():
    def always_503():
        raise TransientError(503, "down")

    with pytest.raises(TransientError):
        call_with_retry(always_503, max_retries=2, sleep=_no_sleep)


def test_semantic_error_is_not_retried():
    calls = {"n": 0}

    def semantic():
        calls["n"] += 1
        raise SemanticError(409, "duplicate")

    with pytest.raises(SemanticError):
        call_with_retry(semantic, max_retries=5, sleep=_no_sleep)
    assert calls["n"] == 1  # tried once, never retried


def test_on_retry_hook_is_called_per_attempt():
    seen = []

    def flaky():
        if len(seen) < 2:
            raise TransientError(429, "rate limited")
        return "done"

    call_with_retry(
        flaky,
        max_retries=5,
        sleep=_no_sleep,
        on_retry=lambda attempt, err, delay: seen.append(attempt),
    )
    assert seen == [1, 2]
