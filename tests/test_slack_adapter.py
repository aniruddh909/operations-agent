"""Tests for the Slack entry-point adapter.

The adapter must produce the same BugReport the CLI does, so the loop is
unchanged. We assert on the constructed BugReport, and that it can drive
run_triage exactly like a CLI-sourced report.
"""

from __future__ import annotations

import pytest

from operations_agent.agent import run_triage
from operations_agent.clients import FakeJiraClient, FakeSlackClient, ModelResponse
from operations_agent.models import BugSource, EventType, RunStatus
from operations_agent.slack_adapter import (
    bug_report_from_event,
    bug_report_from_slash_command,
)


def test_slash_command_builds_slack_sourced_bug_report():
    payload = {"text": "checkout returns 500 on pay", "user_name": "aniruddh"}
    bug = bug_report_from_slash_command(payload)
    assert bug.raw_text == "checkout returns 500 on pay"
    assert bug.source is BugSource.SLACK
    assert bug.reporter == "aniruddh"


def test_event_payload_builds_bug_report():
    payload = {"event": {"text": "dashboard is slow", "user": "U123"}}
    bug = bug_report_from_event(payload)
    assert bug.raw_text == "dashboard is slow"
    assert bug.source is BugSource.SLACK
    assert bug.reporter == "U123"


def test_empty_text_is_rejected():
    with pytest.raises(ValueError):
        bug_report_from_slash_command({"text": "  "})


def test_slack_bug_report_drives_the_same_loop():
    # The whole point: a Slack-sourced BugReport runs through run_triage unchanged.
    bug = bug_report_from_slash_command(
        {"text": "login crashes on submit, repro every time", "user_name": "x"}
    )

    class M:
        def call(self, *, system, messages, tools):
            t = tools[0]["name"]
            if t == "submit_evidence":
                return ModelResponse(tool_name=t, tool_input={
                    "info_sufficiency": {"level": "high", "justification": "x"},
                    "severity_clarity": {"level": "high", "justification": "x"},
                    "component_clarity": {"level": "high", "justification": "x"}})
            return ModelResponse(tool_name=t, tool_input={
                "rationale": "file", "steps": [{"tool": "create_ticket",
                "intent": "file", "args": {"summary": "s", "description": "d",
                "priority": "P1", "component": "auth"}}]})

    trace = run_triage(bug, model=M(), jira=FakeJiraClient(), slack=FakeSlackClient())
    assert trace.status is RunStatus.COMPLETED
    assert trace.bug_report.source is BugSource.SLACK
    assert any(e.type is EventType.RUN_COMPLETED for e in trace.events)
