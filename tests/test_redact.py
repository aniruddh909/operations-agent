"""Tests for trace redaction."""

from __future__ import annotations

import json

from operations_agent.models import BugReport, EventType, RunStatus, Trace
from operations_agent.redact import redact_trace


def _trace_with_secrets() -> Trace:
    t = Trace(bug_report=BugReport(raw_text="bug from jane@example.com"))
    t.record(
        EventType.STEP_COMPLETED, "filed",
        tool="create_ticket",
        result={"key": "OPS-1", "url": "https://x.atlassian.net/browse/OPS-1"},
        note="auth header Bearer sk-ant-abc123XYZ used",
    )
    t.finish(RunStatus.COMPLETED, ticket={"key": "OPS-1"})
    return t


def test_redacts_secret_shaped_strings():
    red = redact_trace(_trace_with_secrets())
    blob = json.dumps(red)
    assert "sk-ant-abc123XYZ" not in blob
    assert "jane@example.com" not in blob
    assert "[REDACTED]" in blob
    # Non-secret content is preserved.
    assert "OPS-1" in blob


def test_scrub_text_removes_free_text_fields():
    red = redact_trace(_trace_with_secrets(), scrub_text=True)
    # raw_text was scrubbed wholesale.
    assert red["bug_report"]["raw_text"] == "[REDACTED]"


def test_redaction_output_is_json_serializable():
    red = redact_trace(_trace_with_secrets())
    json.dumps(red)  # must not raise
