"""Foundational schemas for the triage agent.

These three models are the contract every later slice builds on:

- ``BugReport`` — the normalized input, whatever the entry point (CLI now,
  Slack later). Adapters construct this; the loop only ever sees this.
- ``Plan`` / ``PlanStep`` — the agent's explicit, printable intent. The plan is
  a first-class object so it can be rendered, logged, and (later) revised by a
  reflection step.
- ``Trace`` / ``TraceEvent`` — the single source of truth for one run. It is
  what gets rendered live (Slice 6), saved to disk, and scored by the eval
  harness (Slice 8). Nothing about a run should live *only* in memory or logs;
  if it matters, it goes in the Trace.

Keep these lean but extensible — fields added later (evidence checks,
reflections) should slot in without reshaping what exists here.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #


class BugSource(str, Enum):
    """Where a bug report entered the system."""

    CLI = "cli"
    SLACK = "slack"


class BugReport(BaseModel):
    """A normalized inbound bug report.

    Entry-point adapters (CLI, and later Slack) are responsible for building
    this; the agent loop is written against ``BugReport`` alone and never
    touches a raw Slack event or argv.
    """

    id: str = Field(default_factory=lambda: _new_id("bug"))
    raw_text: str
    source: BugSource = BugSource.CLI
    reporter: Optional[str] = None
    created_at: datetime = Field(default_factory=_utcnow)


# --------------------------------------------------------------------------- #
# Plan
# --------------------------------------------------------------------------- #


class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanStep(BaseModel):
    """One intended action in the plan.

    ``tool`` names a tool the executor knows how to run; ``args`` are the
    arguments the model proposed for it. ``status`` and ``result`` are filled in
    by the executor as the step runs — the model only supplies ``tool``,
    ``intent``, and ``args``.
    """

    id: str = Field(default_factory=lambda: _new_id("step"))
    tool: str
    intent: str = Field(description="One line on why this step exists.")
    args: dict[str, Any] = Field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    result: Optional[dict[str, Any]] = None


class Plan(BaseModel):
    """The agent's ordered, explicit plan for triaging one bug."""

    rationale: str = Field(description="Brief reasoning for the overall plan.")
    steps: list[PlanStep] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Duplicate detection
# --------------------------------------------------------------------------- #


class DuplicateClassification(str, Enum):
    """How confident we are that the incoming bug duplicates an existing one.

    Derived from the retrieval cosine score against the configured bands, then
    confirmed by the model. In Slice 4 ``AMBIGUOUS`` is what drives the
    confidence gate to ask a human.
    """

    CLEAR = "clear"  # high similarity + model agrees: it's a duplicate
    AMBIGUOUS = "ambiguous"  # mid similarity: unsure, worth a human's eye
    NONE = "none"  # low similarity / model says different: novel bug


class DuplicateCandidate(BaseModel):
    """One retrieved existing ticket considered as a possible duplicate."""

    key: str
    summary: str
    score: float = Field(description="Cosine similarity to the incoming bug.")


class DuplicateVerdict(BaseModel):
    """The outcome of duplicate detection for one incoming bug."""

    classification: DuplicateClassification
    candidates: list[DuplicateCandidate] = Field(default_factory=list)
    matched_key: Optional[str] = Field(
        default=None, description="The duplicate ticket key, if one was found."
    )
    reasoning: str = Field(
        default="", description="The model's justification for the verdict."
    )


# --------------------------------------------------------------------------- #
# Evidence & the confidence gate
# --------------------------------------------------------------------------- #


class EvidenceLevel(str, Enum):
    """How strong/clear the signal is for one evidence dimension.

    ``LOW`` means the agent is NOT confident on that dimension — the gate treats
    any LOW as a reason to ask a human rather than guess.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvidenceCheck(BaseModel):
    """One named evidence dimension: a level plus a one-line justification."""

    level: EvidenceLevel
    justification: str = Field(description="One line on why this level.")


class EvidenceChecks(BaseModel):
    """The structured evidence behind a triage decision.

    The model supplies ``info_sufficiency``, ``severity_clarity`` and
    ``component_clarity`` by reading the report. ``duplicate_ambiguity`` is NOT
    asked of the model — it is derived in our code from the Slice 3 cosine
    verdict, so confidence is grounded in a concrete signal, not a vibe.
    """

    info_sufficiency: EvidenceCheck
    severity_clarity: EvidenceCheck
    component_clarity: EvidenceCheck
    duplicate_ambiguity: EvidenceCheck


class GateAction(str, Enum):
    PROCEED = "proceed"
    ASK_HUMAN = "ask_human"


class GateDecision(BaseModel):
    """The output of the (pure, code-side) confidence-gating policy."""

    action: GateAction
    triggered: list[str] = Field(
        default_factory=list,
        description="Names of the checks that forced an ask_human.",
    )
    question: Optional[str] = Field(
        default=None, description="The clarifying question to put to the human."
    )


# --------------------------------------------------------------------------- #
# Trace
# --------------------------------------------------------------------------- #


class EventType(str, Enum):
    """The kinds of things that happen during a run, in order."""

    DUPLICATE_CHECK = "duplicate_check"
    EVIDENCE_SUBMITTED = "evidence_submitted"
    GATE_DECISION = "gate_decision"
    HUMAN_ASKED = "human_asked"
    HUMAN_ANSWERED = "human_answered"
    PLAN_PROPOSED = "plan_proposed"
    PLAN_REPAIR = "plan_repair"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"


class TraceEvent(BaseModel):
    """A single timestamped entry in a run's history.

    ``data`` is deliberately a free-form dict so new event kinds (evidence
    checks, reflections, retries) can be recorded in later slices without a
    schema change here.
    """

    type: EventType
    at: datetime = Field(default_factory=_utcnow)
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class RunStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


class Trace(BaseModel):
    """Everything that happened during one triage run — the source of truth."""

    id: str = Field(default_factory=lambda: _new_id("trace"))
    bug_report: BugReport
    plan: Optional[Plan] = None
    events: list[TraceEvent] = Field(default_factory=list)
    status: Optional[RunStatus] = None
    outcome: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: Optional[datetime] = None

    # -- recording helpers (keep mutation in one place) -- #

    def record(
        self,
        type: EventType,
        message: str = "",
        **data: Any,
    ) -> TraceEvent:
        event = TraceEvent(type=type, message=message, data=data)
        self.events.append(event)
        return event

    def finish(self, status: RunStatus, **outcome: Any) -> None:
        self.status = status
        self.outcome = outcome
        self.finished_at = _utcnow()
