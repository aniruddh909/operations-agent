"""Tests for the Rich trace renderer.

The renderer is a pure function of the Trace, so we render a constructed Trace
to a string (via Rich's capture) and assert the key facts surface. We don't
assert exact layout — only that the important signals are legible.
"""

from __future__ import annotations

from rich.console import Console

from operations_agent.models import (
    BugReport,
    EventType,
    GateAction,
    Plan,
    PlanStep,
    RunStatus,
    StepStatus,
    Trace,
)
from operations_agent.render import render_trace


def _capture(trace: Trace) -> str:
    console = Console(width=100, force_terminal=False)
    with console.capture() as cap:
        console.print(render_trace(trace))
    return cap.get()


def test_renders_evidence_gate_clarify_plan_and_outcome():
    trace = Trace(bug_report=BugReport(raw_text="login is broken"))
    trace.record(
        EventType.DUPLICATE_CHECK, "dup",
        classification="ambiguous",
        candidates=[{"key": "KAN-3", "summary": "login crash", "score": 0.72}],
    )
    trace.record(
        EventType.EVIDENCE_SUBMITTED, "ev",
        info_sufficiency={"level": "low", "justification": "no repro"},
        severity_clarity={"level": "medium", "justification": "unclear"},
        component_clarity={"level": "high", "justification": "auth"},
        duplicate_ambiguity={"level": "low", "justification": "close to KAN-3"},
    )
    trace.record(
        EventType.GATE_DECISION, "gate",
        action=GateAction.ASK_HUMAN.value,
        triggered=["info_sufficiency", "duplicate_ambiguity"],
    )
    trace.record(EventType.HUMAN_ASKED, "What are the exact repro steps?")
    trace.record(EventType.HUMAN_ANSWERED, "Blank password -> 500.")

    step = PlanStep(tool="create_ticket", intent="file the bug")
    step.status = StepStatus.DONE
    step.result = {"key": "KAN-20"}
    trace.plan = Plan(rationale="file it", steps=[step])
    trace.finish(RunStatus.COMPLETED, ticket={"key": "KAN-20", "url": "http://x/KAN-20"})

    out = _capture(trace)

    assert "login is broken" in out
    assert "ambiguous" in out
    assert "info_sufficiency" in out  # evidence table
    assert "ASK HUMAN" in out  # gate highlighted
    assert "repro steps" in out  # clarifying question
    assert "Blank password" in out  # the answer
    assert "create_ticket" in out  # plan step
    assert "KAN-20" in out  # outcome


def test_renders_reflection_and_failure_recovery():
    trace = Trace(bug_report=BugReport(raw_text="some bug"))
    step = PlanStep(tool="create_ticket", intent="file")
    step.status = StepStatus.FAILED
    trace.plan = Plan(rationale="r", steps=[step])
    trace.record(EventType.STEP_FAILED, "Semantic failure (409): duplicate")
    trace.record(EventType.REFLECTION, "Reflecting on 409 failure.")
    trace.record(EventType.PLAN_REVISED, "Revised plan: 0 remaining step(s).")
    trace.finish(RunStatus.COMPLETED, ticket={})

    out = _capture(trace)
    assert "Reflect" in out
    assert "Revised plan" in out


def test_renders_duplicate_short_circuit_outcome():
    trace = Trace(bug_report=BugReport(raw_text="password reset emails missing"))
    trace.record(EventType.DUPLICATE_CHECK, "dup", classification="clear",
                 candidates=[{"key": "KAN-7", "summary": "x", "score": 0.85}])
    trace.finish(RunStatus.COMPLETED, duplicate_of="KAN-7", filed=False)

    out = _capture(trace)
    assert "duplicate of KAN-7" in out
