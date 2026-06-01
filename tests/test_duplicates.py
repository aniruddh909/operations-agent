"""Behavior tests for duplicate detection.

Uses the deterministic FakeEmbeddingClient and a real SQLite index (in a tmp
file), plus a scripted model for adjudication — no network, no torch. Asserts on
the DuplicateVerdict and on the Trace produced by run_triage.
"""

from __future__ import annotations

from operations_agent.agent import DuplicateChecker, run_triage
from operations_agent.clients import FakeJiraClient, ModelResponse
from operations_agent.duplicates import find_duplicate
from operations_agent.embeddings import FakeEmbeddingClient
from operations_agent.index import IndexedTicket, TicketIndex
from operations_agent.models import (
    BugReport,
    DuplicateClassification,
    EventType,
)


def _index(tmp_path) -> TicketIndex:
    idx = TicketIndex(tmp_path / "idx.db", FakeEmbeddingClient())
    idx.add(IndexedTicket("KAN-1", "App crashes when logging in",
                          "Crashes to white screen after submitting login."))
    idx.add(IndexedTicket("KAN-2", "Export to CSV produces empty file",
                          "CSV export downloads only headers, no rows."))
    return idx


class _Model:
    """Scripted adjudicator: always returns the staged verdict."""

    def __init__(self, **verdict):
        self._verdict = verdict
        self.calls = 0

    def call(self, *, system, messages, tools):
        self.calls += 1
        return ModelResponse(
            tool_name="submit_duplicate_verdict", tool_input=self._verdict
        )


def test_paraphrased_duplicate_is_caught(tmp_path):
    idx = _index(tmp_path)
    model = _Model(is_duplicate=True, matched_key="KAN-1",
                   reasoning="Same login crash, reworded.")

    verdict = find_duplicate(
        "Login screen freezes and the app dies when I sign in",
        index=idx, model=model, clear_band=0.3, ambiguous_band=0.15,
    )

    assert verdict.classification is DuplicateClassification.CLEAR
    assert verdict.matched_key == "KAN-1"
    assert model.calls == 1


def test_novel_bug_is_not_flagged_and_skips_model(tmp_path):
    idx = _index(tmp_path)
    model = _Model(is_duplicate=False, matched_key=None, reasoning="n/a")

    # A clearly unrelated bug; with high bands it falls below ambiguous and
    # should short-circuit WITHOUT calling the model.
    verdict = find_duplicate(
        "Calendar timezone is wrong for recurring events",
        index=idx, model=model, clear_band=0.9, ambiguous_band=0.85,
    )

    assert verdict.classification is DuplicateClassification.NONE
    assert verdict.matched_key is None
    assert model.calls == 0  # cheap path: no adjudication needed


def test_clear_duplicate_short_circuits_run_without_filing(tmp_path):
    idx = _index(tmp_path)
    jira = FakeJiraClient()
    model = _Model(is_duplicate=True, matched_key="KAN-1", reasoning="same bug")
    dup = DuplicateChecker(index=idx, clear_band=0.3, ambiguous_band=0.15)

    trace = run_triage(
        BugReport(raw_text="the app crashes on login every time"),
        model=model, jira=jira, duplicates=dup,
    )

    # No new ticket filed; run completed flagged as a duplicate.
    assert jira.created == []
    assert trace.outcome.get("filed") is False
    assert trace.outcome.get("duplicate_of") == "KAN-1"
    assert any(e.type is EventType.DUPLICATE_CHECK for e in trace.events)


def test_novel_bug_runs_planning_and_embeds_on_ingest(tmp_path):
    idx = _index(tmp_path)
    jira = FakeJiraClient()
    dup = DuplicateChecker(index=idx, clear_band=0.9, ambiguous_band=0.85)

    # High bands => novel (no adjudication call). The model is then asked for
    # evidence (all-high so the gate proceeds), then the plan.
    class TwoStep:
        def __init__(self):
            self.calls = 0

        def call(self, *, system, messages, tools):
            self.calls += 1
            tool = tools[0]["name"]
            if tool == "submit_evidence":
                return ModelResponse(tool_name="submit_evidence", tool_input={
                    "info_sufficiency": {"level": "high", "justification": "x"},
                    "severity_clarity": {"level": "high", "justification": "x"},
                    "component_clarity": {"level": "high", "justification": "x"},
                })
            return ModelResponse(tool_name="submit_plan", tool_input={
                "rationale": "Novel bug, file it.",
                "steps": [{"tool": "create_ticket", "intent": "file",
                           "args": {"summary": "Avatar PNG upload rejected",
                                    "description": "PNG avatars rejected as unsupported.",
                                    "priority": "P2", "component": "profile"}}],
            })

    before = idx.count()
    trace = run_triage(
        BugReport(raw_text="uploading a png profile picture is rejected"),
        model=TwoStep(), jira=jira, duplicates=dup,
    )

    assert len(jira.created) == 1
    assert idx.count() == before + 1  # embed-on-ingest added the new ticket
    assert trace.status.value == "completed"
