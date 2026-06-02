"""Tests for the evaluation harness, scorer, judge, and report."""

from __future__ import annotations

from operations_agent.clients import FakeJiraClient, ModelResponse
from operations_agent.embeddings import FakeEmbeddingClient
from operations_agent.evaluation.cases import GoldenCase, load_golden_set
from operations_agent.evaluation.harness import compare, run_eval
from operations_agent.evaluation.judge import judge_trace
from operations_agent.evaluation.report import comparison_to_markdown, to_markdown
from operations_agent.evaluation.scorer import (
    aggregate,
    extract_decisions,
    score_case,
)
from operations_agent.models import (
    BugReport,
    EventType,
    RunStatus,
    Trace,
)


# --------------------------------------------------------------------------- #
# Golden set
# --------------------------------------------------------------------------- #


def test_golden_set_loads_and_covers_six_dimensions():
    cases = load_golden_set()
    assert len(cases) >= 18
    dims = {c.dimension for c in cases}
    assert {
        "clear-dup",
        "ambiguous-dup",
        "novel",
        "insufficient-info",
        "clear-P1",
        "ambiguous-priority",
    } <= dims


# --------------------------------------------------------------------------- #
# Scorer (pure)
# --------------------------------------------------------------------------- #


def _trace_with(gate=None, dup=None, ticket=None, filed_false=False) -> Trace:
    t = Trace(bug_report=BugReport(raw_text="x"))
    if dup is not None:
        t.record(EventType.DUPLICATE_CHECK, "d", classification=dup)
    if gate is not None:
        t.record(EventType.GATE_DECISION, "g", action=gate)
    if ticket is not None:
        t.outcome = {"ticket": ticket}
    elif filed_false:
        t.outcome = {"filed": False}
    t.status = RunStatus.COMPLETED
    return t


def test_extract_decisions_reads_gate_dup_and_ticket():
    t = _trace_with(gate="proceed", dup="none",
                    ticket={"key": "OPS-1", "priority": "P1", "component": "auth"})
    d = extract_decisions(t)
    assert d["gate"] == "proceed"
    assert d["duplicate"] == "none"
    assert d["filed"] is True
    assert d["priority"] == "P1"
    assert d["component"] == "auth"


def test_score_case_marks_correct_and_incorrect():
    case = GoldenCase(
        id="c", raw_text="x", dimension="novel",
        expect_gate="proceed", expect_priority="P1",
        expect_component="auth", expect_duplicate="none", expect_filed=True,
    )
    t = _trace_with(gate="proceed", dup="none",
                    ticket={"key": "OPS-1", "priority": "P1", "component": "Auth service"})
    score = score_case(case, t)
    assert score.passed
    assert score.results == {
        "gate": True, "duplicate": True, "filed": True,
        "priority": True, "component": True,
    }


def test_score_case_component_is_substring_and_priority_exact():
    case = GoldenCase(id="c", raw_text="x", dimension="novel",
                      expect_priority="P0", expect_component="checkout")
    t = _trace_with(ticket={"key": "OPS-1", "priority": "P1", "component": "checkout-flow"})
    score = score_case(case, t)
    assert score.results["component"] is True   # substring match
    assert score.results["priority"] is False   # P1 != P0


def test_aggregate_counts_per_metric_and_overall():
    case = GoldenCase(id="c", raw_text="x", dimension="d", expect_gate="proceed")
    good = score_case(case, _trace_with(gate="proceed"))
    bad = score_case(case, _trace_with(gate="ask_human"))
    totals = aggregate([good, bad])
    assert totals["gate"] == {"correct": 1, "total": 2}
    assert totals["overall"] == {"correct": 1, "total": 2}


# --------------------------------------------------------------------------- #
# Harness (fake model + fake embedder, no network)
# --------------------------------------------------------------------------- #


class _ProceedModel:
    """Always: novel duplicate verdict, all-high evidence, a simple file plan."""

    def call(self, *, system, messages, tools):
        tool = tools[0]["name"]
        if tool == "submit_duplicate_verdict":
            return ModelResponse(tool_name=tool, tool_input={
                "is_duplicate": False, "matched_key": None, "reasoning": "novel"})
        if tool == "submit_evidence":
            return ModelResponse(tool_name=tool, tool_input={
                "info_sufficiency": {"level": "high", "justification": "x"},
                "severity_clarity": {"level": "high", "justification": "x"},
                "component_clarity": {"level": "high", "justification": "x"}})
        if tool == "submit_plan":
            return ModelResponse(tool_name=tool, tool_input={
                "rationale": "file", "steps": [{"tool": "create_ticket",
                "intent": "file", "args": {"summary": "s", "description": "d",
                "priority": "P2", "component": "auth"}}]})
        raise AssertionError(tool)


def test_run_eval_produces_report_with_totals():
    # Score only a couple of novel cases the fake model will get "right".
    cases = [
        GoldenCase(id="n1", raw_text="a novel detailed bug", dimension="novel",
                   expect_gate="proceed", expect_filed=True),
    ]
    report = run_eval(
        model=_ProceedModel(), embedder=FakeEmbeddingClient(),
        clear_band=0.95, ambiguous_band=0.9, label="fake", cases=cases,
    )
    assert report.label == "fake"
    assert report.totals["overall"]["total"] >= 1
    assert 0.0 <= report.accuracy("overall") <= 1.0


def test_compare_two_reports_has_delta():
    cases = [GoldenCase(id="n1", raw_text="bug", dimension="novel",
                        expect_gate="proceed")]
    a = run_eval(model=_ProceedModel(), embedder=FakeEmbeddingClient(),
                 clear_band=0.95, ambiguous_band=0.9, label="a", cases=cases)
    b = run_eval(model=_ProceedModel(), embedder=FakeEmbeddingClient(),
                 clear_band=0.95, ambiguous_band=0.9, label="b", cases=cases)
    diff = compare(a, b)
    assert "overall" in diff
    assert diff["overall"]["delta"] == 0.0  # identical models


# --------------------------------------------------------------------------- #
# Report rendering + judge
# --------------------------------------------------------------------------- #


def test_to_markdown_includes_gate_metric():
    cases = [GoldenCase(id="n1", raw_text="bug", dimension="novel",
                        expect_gate="proceed")]
    report = run_eval(model=_ProceedModel(), embedder=FakeEmbeddingClient(),
                      clear_band=0.95, ambiguous_band=0.9, cases=cases)
    md = to_markdown(report)
    assert "Confidence-gate accuracy" in md
    assert "non-negotiable" in md
    assert "| Metric |" in md


def test_judge_returns_rubric_scores():
    t = Trace(bug_report=BugReport(raw_text="bug"))
    t.record(EventType.EVIDENCE_SUBMITTED, "ev",
             info_sufficiency={"level": "high", "justification": "clear repro"},
             severity_clarity={"level": "high", "justification": "blocks all"},
             component_clarity={"level": "high", "justification": "auth"},
             duplicate_ambiguity={"level": "high", "justification": "novel"})

    class JudgeModel:
        def call(self, *, system, messages, tools):
            return ModelResponse(tool_name="submit_grades", tool_input={
                "reasoning_quality": 4, "summary_quality": 3, "note": "solid"})

    result = judge_trace(t, judge_model=JudgeModel())
    assert result.reasoning_quality == 4
    assert result.summary_quality == 3
