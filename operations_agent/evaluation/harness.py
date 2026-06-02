"""Run the agent over the golden set and report scores.

The harness scores the agent's *decisions*, so it runs against a FakeJiraClient
and a freshly-seeded in-memory index (the same fixtures the demo uses) — no real
Jira writes. Only the model is real. Ask-human cases get an auto-answering human
so the run completes; what we score is that the gate *decided* to ask.

``run_eval`` returns an EvalReport. ``compare`` diffs two reports (e.g. two
prompt or model versions) on the same set.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import BaseModel

from ..agent import DuplicateChecker, run_triage
from ..clients import FakeJiraClient, ModelClient, ScriptedHumanClient
from ..embeddings import EmbeddingClient
from ..index import IndexedTicket, TicketIndex
from ..models import BugReport
from ..seed import seed
from .cases import GoldenCase, load_golden_set
from .scorer import CaseScore, aggregate, score_case


class EvalReport(BaseModel):
    label: str
    scores: list[CaseScore]
    totals: dict[str, dict[str, int]]

    def accuracy(self, metric: str = "overall") -> float:
        b = self.totals.get(metric)
        if not b or b["total"] == 0:
            return 0.0
        return b["correct"] / b["total"]


def _fresh_index(embedder: EmbeddingClient) -> TicketIndex:
    """Seed a temporary index with the demo fixtures (for dup detection)."""
    tmp = Path(tempfile.mkdtemp()) / "eval_index.db"
    index = TicketIndex(tmp, embedder)
    seed(jira=FakeJiraClient(), index=index)
    return index


def run_eval(
    *,
    model: ModelClient,
    embedder: EmbeddingClient,
    clear_band: float,
    ambiguous_band: float,
    label: str = "current",
    cases: list[GoldenCase] | None = None,
) -> EvalReport:
    """Run the agent over every case and score the resulting Traces."""
    cases = cases if cases is not None else load_golden_set()
    index = _fresh_index(embedder)
    scores: list[CaseScore] = []

    for case in cases:
        dup = DuplicateChecker(
            index=index, clear_band=clear_band, ambiguous_band=ambiguous_band
        )
        # Auto-answer clarifications so ask-human runs complete; we score the
        # gate decision, not the answer.
        human = ScriptedHumanClient(
            "Here are the details: repro steps included, treat as a new bug."
        )
        trace = run_triage(
            BugReport(raw_text=case.raw_text),
            model=model,
            jira=FakeJiraClient(),
            duplicates=dup,
            human=human,
        )
        scores.append(score_case(case, trace))

    return EvalReport(label=label, scores=scores, totals=aggregate(scores))


def compare(a: EvalReport, b: EvalReport) -> dict[str, dict[str, float]]:
    """Diff two reports per metric: accuracy of each plus the delta."""
    metrics = sorted(set(a.totals) | set(b.totals))
    out: dict[str, dict[str, float]] = {}
    for m in metrics:
        av, bv = a.accuracy(m), b.accuracy(m)
        out[m] = {a.label: av, b.label: bv, "delta": bv - av}
    return out
