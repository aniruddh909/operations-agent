"""Behavior tests for the evidence -> gate -> ask_human flow via run_triage.

A small dispatching fake model returns the right tool input depending on which
tool is being requested (evidence vs. plan), so we can drive the whole flow
deterministically and assert on the Trace.
"""

from __future__ import annotations

from operations_agent.agent import run_triage
from operations_agent.clients import (
    FakeJiraClient,
    ModelResponse,
    ScriptedHumanClient,
)
from operations_agent.models import BugReport, EventType, RunStatus


class DispatchModel:
    """Returns evidence or a plan depending on the offered tool.

    ``evidence`` is a dict of {dimension: level}; the plan is fixed. This lets a
    single fake serve both model calls run_triage makes (no duplicates here).
    """

    def __init__(self, evidence_levels: dict):
        self._evidence_levels = evidence_levels
        self.evidence_calls = 0
        self.plan_calls = 0

    def call(self, *, system, messages, tools):
        tool = tools[0]["name"]
        if tool == "submit_evidence":
            self.evidence_calls += 1
            return ModelResponse(
                tool_name="submit_evidence",
                tool_input={
                    dim: {"level": lvl, "justification": "because"}
                    for dim, lvl in self._evidence_levels.items()
                },
            )
        if tool == "submit_plan":
            self.plan_calls += 1
            return ModelResponse(
                tool_name="submit_plan",
                tool_input={
                    "rationale": "File it.",
                    "steps": [
                        {
                            "tool": "create_ticket",
                            "intent": "file",
                            "args": {
                                "summary": "Bug",
                                "description": "desc",
                                "priority": "P2",
                                "component": "auth",
                            },
                        }
                    ],
                },
            )
        raise AssertionError(f"unexpected tool {tool}")


_ALL_HIGH = {
    "info_sufficiency": "high",
    "severity_clarity": "high",
    "component_clarity": "high",
}
_LOW_INFO = {**_ALL_HIGH, "info_sufficiency": "low"}


def test_clear_report_proceeds_without_asking():
    model = DispatchModel(_ALL_HIGH)
    jira = FakeJiraClient()
    human = ScriptedHumanClient()  # no answers needed

    trace = run_triage(
        BugReport(raw_text="detailed, clear bug report"),
        model=model, jira=jira, human=human,
    )

    assert trace.status is RunStatus.COMPLETED
    assert human.asked == []  # never asked
    assert len(jira.created) == 1
    types = [e.type for e in trace.events]
    assert EventType.EVIDENCE_SUBMITTED in types
    assert EventType.GATE_DECISION in types
    assert EventType.HUMAN_ASKED not in types


def test_vague_report_asks_then_resumes_and_files():
    model = DispatchModel(_LOW_INFO)
    jira = FakeJiraClient()
    human = ScriptedHumanClient("Repro: open app, click X. Expected Y, got Z.")

    trace = run_triage(
        BugReport(raw_text="it's broken"),
        model=model, jira=jira, human=human,
    )

    assert trace.status is RunStatus.COMPLETED
    assert len(human.asked) == 1  # asked exactly once
    assert len(jira.created) == 1  # then resumed and filed
    types = [e.type for e in trace.events]
    assert EventType.HUMAN_ASKED in types
    assert EventType.HUMAN_ANSWERED in types
    # the clarifying answer reached planning
    answered = [e for e in trace.events if e.type is EventType.HUMAN_ANSWERED][0]
    assert "Repro" in answered.message


def test_ask_needed_but_no_human_fails_gracefully():
    model = DispatchModel(_LOW_INFO)
    jira = FakeJiraClient()

    trace = run_triage(
        BugReport(raw_text="it's broken"),
        model=model, jira=jira, human=None,
    )

    assert trace.status is RunStatus.FAILED
    assert jira.created == []  # never guessed a ticket
    assert any(e.type is EventType.RUN_FAILED for e in trace.events)
