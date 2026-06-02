"""Behavior tests for generate_triage_summary + notify_slack.

The planner chooses a 3-step plan (create_ticket -> generate_triage_summary ->
notify_slack). A dispatching fake model serves the evidence call, the plan, and
the summary text. We assert via the Trace and the fake Slack client that the
summary is grounded in the ticket and was posted.
"""

from __future__ import annotations

from operations_agent.agent import run_triage
from operations_agent.clients import (
    FakeJiraClient,
    FakeSlackClient,
    ModelResponse,
)
from operations_agent.models import BugReport, EventType, RunStatus

_EVIDENCE_HIGH = {
    "info_sufficiency": {"level": "high", "justification": "x"},
    "severity_clarity": {"level": "high", "justification": "x"},
    "component_clarity": {"level": "high", "justification": "x"},
}

_THREE_STEP_PLAN = {
    "rationale": "File, summarize, notify.",
    "steps": [
        {"tool": "create_ticket", "intent": "file the bug",
         "args": {"summary": "Login 500 on empty password",
                  "description": "500 on blank password submit",
                  "priority": "P1", "component": "auth"}},
        {"tool": "generate_triage_summary", "intent": "summarize", "args": {}},
        {"tool": "notify_slack", "intent": "notify the team", "args": {}},
    ],
}


class PlannerModel:
    """Serves evidence, then the plan, then the summary text."""

    def __init__(self, summary_text):
        self._summary = summary_text
        self.calls = []

    def call(self, *, system, messages, tools):
        tool = tools[0]["name"]
        self.calls.append(tool)
        if tool == "submit_evidence":
            return ModelResponse(tool_name="submit_evidence", tool_input=_EVIDENCE_HIGH)
        if tool == "submit_plan":
            return ModelResponse(tool_name="submit_plan", tool_input=_THREE_STEP_PLAN)
        if tool == "submit_summary":
            return ModelResponse(
                tool_name="submit_summary", tool_input={"summary": self._summary}
            )
        raise AssertionError(tool)


def test_planner_composes_summary_and_slack_post():
    summary = "Filed OPS-1 (P1, auth): login returns 500 on empty password submit."
    model = PlannerModel(summary)
    jira = FakeJiraClient()
    slack = FakeSlackClient()

    trace = run_triage(
        BugReport(raw_text="login 500 on empty password"),
        model=model, jira=jira, slack=slack,
    )

    assert trace.status is RunStatus.COMPLETED
    # The plan the model chose includes all three tools (not hardcoded).
    tools = [s.tool for s in trace.plan.steps]
    assert tools == ["create_ticket", "generate_triage_summary", "notify_slack"]
    # Slack received the generated summary.
    assert slack.posted == [summary]
    # All three actions are recorded as completed steps in the Trace.
    completed = [
        e.data.get("tool")
        for e in trace.events
        if e.type is EventType.STEP_COMPLETED
    ]
    assert completed == ["create_ticket", "generate_triage_summary", "notify_slack"]


def test_summary_is_grounded_in_the_filed_ticket():
    # The summarizer is handed the ticket facts; assert it sees the real key.
    seen = {}

    class CapturingModel(PlannerModel):
        def call(self, *, system, messages, tools):
            if tools[0]["name"] == "submit_summary":
                seen["facts"] = messages[0]["content"]
            return super().call(system=system, messages=messages, tools=tools)

    model = CapturingModel("summary text")
    run_triage(
        BugReport(raw_text="bug"), model=model,
        jira=FakeJiraClient(), slack=FakeSlackClient(),
    )
    assert "OPS-1" in seen["facts"]  # the filed ticket key reached the summarizer
    assert "auth" in seen["facts"]   # and its component


def test_notify_slack_without_client_records_intent_not_failure():
    model = PlannerModel("s")
    jira = FakeJiraClient()

    trace = run_triage(
        BugReport(raw_text="bug"), model=model, jira=jira, slack=None,
    )

    assert trace.status is RunStatus.COMPLETED  # didn't fail with no Slack
    slack_step = [s for s in trace.plan.steps if s.tool == "notify_slack"][0]
    assert slack_step.result["posted"] is False
