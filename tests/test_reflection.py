"""Behavior tests for two-tier error handling in the agent loop.

- A semantic 409 from Jira surfaces to the model, which reflects and revises the
  remaining plan (here: stop, the ticket already exists).
- A transient 429 is retried below the loop and never becomes a reasoning
  failure; the step ultimately succeeds.

Both assert on the Trace. The model is a small dispatcher returning evidence,
the plan, and (for the 409 case) the revised plan.
"""

from __future__ import annotations

from operations_agent.agent import RetryConfig, run_triage
from operations_agent.clients import ModelResponse
from operations_agent.errors import SemanticError, TransientError
from operations_agent.models import BugReport, EventType, RunStatus


_EVIDENCE_HIGH = {
    "info_sufficiency": {"level": "high", "justification": "x"},
    "severity_clarity": {"level": "high", "justification": "x"},
    "component_clarity": {"level": "high", "justification": "x"},
}


def _plan(steps):
    return {"rationale": "r", "steps": steps}


_FILE_STEP = {
    "tool": "create_ticket",
    "intent": "file",
    "args": {
        "summary": "Bug",
        "description": "desc",
        "priority": "P2",
        "component": "auth",
    },
}


class FlowModel:
    """Returns evidence, then a plan, then (optionally) a revised plan."""

    def __init__(self, *, revised_steps=None):
        self._revised_steps = revised_steps
        self.calls = []

    def call(self, *, system, messages, tools):
        tool = tools[0]["name"]
        self.calls.append(tool)
        if tool == "submit_evidence":
            return ModelResponse(tool_name="submit_evidence", tool_input=_EVIDENCE_HIGH)
        if tool == "submit_plan":
            # First submit_plan call is the initial plan; a later one is the
            # reflection revision (distinguished by the revise system prompt).
            if "failed" in system.lower() and self._revised_steps is not None:
                return ModelResponse(
                    tool_name="submit_plan",
                    tool_input=_plan(self._revised_steps),
                )
            return ModelResponse(
                tool_name="submit_plan", tool_input=_plan([_FILE_STEP])
            )
        raise AssertionError(tool)


class RaisingJira:
    """Jira fake whose create_ticket raises a scripted error, then succeeds."""

    def __init__(self, *, raises):
        self._raises = list(raises)  # exceptions to raise, in order
        self.calls = 0

    def create_ticket(self, **kwargs):
        self.calls += 1
        if self._raises:
            err = self._raises.pop(0)
            if err is not None:
                raise err
        return {"key": "OPS-1", "summary": kwargs.get("summary", ""),
                "description": kwargs.get("description", ""),
                "priority": kwargs.get("priority")}


def test_semantic_409_triggers_reflection_and_revised_plan():
    # File fails with 409 (already exists); reflection revises to an empty plan
    # (the right move: stop, don't re-file).
    jira = RaisingJira(raises=[SemanticError(409, "duplicate exists")])
    model = FlowModel(revised_steps=[])  # reflection says: nothing more to do

    trace = run_triage(
        BugReport(raw_text="some bug"), model=model, jira=jira,
    )

    types = [e.type for e in trace.events]
    assert EventType.STEP_FAILED in types
    assert EventType.REFLECTION in types
    assert EventType.PLAN_REVISED in types
    assert trace.status is RunStatus.COMPLETED  # recovered, didn't crash
    assert jira.calls == 1  # never re-filed the duplicate


def test_transient_429_is_retried_then_succeeds():
    # First create_ticket raises 429, second succeeds. Retried below the loop;
    # the agent never sees a reasoning failure.
    jira = RaisingJira(raises=[TransientError(429, "rate limited"), None])
    model = FlowModel()
    fast_retry = RetryConfig(sleep=lambda _: None)  # no real waiting

    trace = run_triage(
        BugReport(raw_text="some bug"), model=model, jira=jira,
        retry_config=fast_retry,
    )

    types = [e.type for e in trace.events]
    assert EventType.STEP_RETRY in types
    assert EventType.STEP_FAILED not in types  # transient never a failure
    assert trace.status is RunStatus.COMPLETED
    assert jira.calls == 2  # failed once, retried, succeeded
