"""Score a Trace against a GoldenCase — pure, deterministic, no model calls.

We pull the agent's actual decisions out of the Trace (gate action, duplicate
verdict, filed priority/component, whether a ticket was filed) and compare them
to the case's expected labels. Only labels present on the case are scored, so
each case contributes only to the dimensions it's designed to test.

The headline metric is gate accuracy: did the agent correctly decide to ask a
human vs. proceed?
"""

from __future__ import annotations

from pydantic import BaseModel

from ..models import EventType, Trace
from .cases import GoldenCase


class CaseScore(BaseModel):
    """Per-case results: each scored dimension maps to a pass/fail (or None)."""

    case_id: str
    dimension: str
    results: dict[str, bool]  # metric name -> correct?
    actual: dict[str, object]  # what the agent actually decided (for debugging)

    @property
    def passed(self) -> bool:
        return all(self.results.values())


# --------------------------------------------------------------------------- #
# Decision extraction from a Trace
# --------------------------------------------------------------------------- #


def extract_decisions(trace: Trace) -> dict:
    """Pull the agent's actual decisions out of a Trace."""
    decisions: dict[str, object] = {
        "gate": None,
        "duplicate": None,
        "filed": None,
        "priority": None,
        "component": None,
    }

    for ev in trace.events:
        if ev.type is EventType.GATE_DECISION:
            decisions["gate"] = ev.data.get("action")
        elif ev.type is EventType.DUPLICATE_CHECK:
            decisions["duplicate"] = ev.data.get("classification")

    ticket = (trace.outcome or {}).get("ticket") or {}
    if ticket.get("key"):
        decisions["filed"] = True
        decisions["priority"] = ticket.get("priority")
        decisions["component"] = ticket.get("component")
    elif (trace.outcome or {}).get("filed") is False:
        decisions["filed"] = False

    return decisions


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def score_case(case: GoldenCase, trace: Trace) -> CaseScore:
    actual = extract_decisions(trace)
    results: dict[str, bool] = {}

    if case.expect_gate is not None:
        results["gate"] = actual["gate"] == case.expect_gate

    if case.expect_duplicate is not None:
        results["duplicate"] = actual["duplicate"] == case.expect_duplicate

    if case.expect_filed is not None:
        results["filed"] = actual["filed"] == case.expect_filed

    if case.expect_priority is not None:
        results["priority"] = actual["priority"] == case.expect_priority

    if case.expect_component is not None:
        comp = (actual["component"] or "").lower()
        results["component"] = case.expect_component.lower() in comp

    return CaseScore(
        case_id=case.id, dimension=case.dimension, results=results, actual=actual
    )


def aggregate(scores: list[CaseScore]) -> dict[str, dict[str, int]]:
    """Aggregate per-metric correct/total counts across all cases."""
    totals: dict[str, dict[str, int]] = {}
    for score in scores:
        for metric, ok in score.results.items():
            bucket = totals.setdefault(metric, {"correct": 0, "total": 0})
            bucket["total"] += 1
            bucket["correct"] += int(ok)
    # Overall (every scored metric on every case).
    overall = {"correct": 0, "total": 0}
    for b in totals.values():
        overall["correct"] += b["correct"]
        overall["total"] += b["total"]
    totals["overall"] = overall
    return totals
