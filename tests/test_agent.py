"""Behavior tests for the walking skeleton.

These assert on the returned ``Trace`` — the agent's external behavior — rather
than on the order of internal calls. The model is injected as a scripted fake so
runs are deterministic without touching the network.
"""

from __future__ import annotations

from operations_agent.agent import run_triage
from operations_agent.clients import FakeJiraClient, ModelResponse
from operations_agent.models import BugReport, EventType, RunStatus, StepStatus


class ScriptedModel:
    """A ModelClient that returns canned tool inputs, one per ``call``.

    Lets a test stage a malformed plan followed by a valid one to exercise the
    repair-retry path.
    """

    def __init__(self, *plan_inputs: dict) -> None:
        self._queue = list(plan_inputs)
        self.calls = 0  # counts plan calls (repair-retry assertions rely on this)

    def call(self, *, system, messages, tools) -> ModelResponse:
        tool = tools[0]["name"]
        if tool == "submit_evidence":
            # All-high so the confidence gate proceeds straight to planning;
            # these tests are about plan validation, not the gate.
            return ModelResponse(
                tool_name="submit_evidence",
                tool_input={
                    "info_sufficiency": {"level": "high", "justification": "x"},
                    "severity_clarity": {"level": "high", "justification": "x"},
                    "component_clarity": {"level": "high", "justification": "x"},
                },
            )
        self.calls += 1
        payload = self._queue.pop(0)
        return ModelResponse(tool_name="submit_plan", tool_input=payload)


def _valid_plan(priority: str = "P2") -> dict:
    return {
        "rationale": "File the reported crash.",
        "steps": [
            {
                "tool": "create_ticket",
                "intent": "File the bug",
                "args": {
                    "summary": "App crashes on login",
                    "description": "User reports a crash when logging in.",
                    "priority": priority,
                    "component": "auth",
                },
            }
        ],
    }


def test_happy_path_files_a_ticket_and_completes():
    model = ScriptedModel(_valid_plan())
    jira = FakeJiraClient()

    trace = run_triage(
        BugReport(raw_text="app crashes on login"), model=model, jira=jira
    )

    assert trace.status is RunStatus.COMPLETED
    assert trace.plan is not None and len(trace.plan.steps) == 1
    assert trace.plan.steps[0].status is StepStatus.DONE
    assert trace.outcome["ticket"]["key"] == "OPS-1"
    assert len(jira.created) == 1
    assert jira.created[0]["priority"] == "P2"
    # The completion is recorded as an event in the trace.
    assert any(e.type is EventType.RUN_COMPLETED for e in trace.events)


def test_invalid_plan_triggers_exactly_one_repair_then_succeeds():
    bad = {"steps": []}  # missing required `rationale`
    model = ScriptedModel(bad, _valid_plan())
    jira = FakeJiraClient()

    trace = run_triage(BugReport(raw_text="bug"), model=model, jira=jira)

    assert model.calls == 2  # original + one repair
    assert trace.status is RunStatus.COMPLETED
    assert any(e.type is EventType.PLAN_REPAIR for e in trace.events)


def test_unrepairable_plan_fails_gracefully_without_filing():
    bad = {"steps": []}  # missing `rationale`, and stays bad
    model = ScriptedModel(bad, bad)
    jira = FakeJiraClient()

    trace = run_triage(BugReport(raw_text="bug"), model=model, jira=jira)

    assert model.calls == 2  # original + one repair, then gives up
    assert trace.status is RunStatus.FAILED
    assert trace.plan is None
    assert jira.created == []
    assert any(e.type is EventType.RUN_FAILED for e in trace.events)
