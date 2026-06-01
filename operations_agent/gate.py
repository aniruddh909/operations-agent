"""The confidence gate — a pure policy over evidence.

This is deliberately plain, deterministic code (no model call). The *judgment*
lives in the evidence levels the model produced; the *decision* of whether to
proceed or ask a human lives here, where a reviewer can read it and a unit test
can pin it exactly. That separation is the whole point of the slice: calibrated
behaviour you can trace and measure.

Policy: if any evidence dimension is LOW, the agent is not confident enough to
triage responsibly, so it asks a human a single clarifying question targeted at
the weakest dimension(s). Otherwise it proceeds.
"""

from __future__ import annotations

from .models import (
    EvidenceCheck,
    EvidenceChecks,
    EvidenceLevel,
    GateAction,
    GateDecision,
)

# How each dimension is phrased when it's the reason we have to ask.
_QUESTION_FOR = {
    "info_sufficiency": "The report doesn't have enough detail to triage "
    "confidently. What are the exact steps to reproduce, and what did you "
    "expect to happen instead?",
    "severity_clarity": "How severe is this for users — is it blocking work, "
    "or a minor annoyance? Roughly how many people are affected?",
    "component_clarity": "Which part of the product does this affect (e.g. "
    "auth, dashboard, notifications)?",
    "duplicate_ambiguity": "This looks similar to an existing ticket but I'm "
    "not sure it's the same bug. Is this a duplicate of an existing issue?",
}


def gate(evidence: EvidenceChecks) -> GateDecision:
    """Decide whether to proceed or ask a human, from the evidence alone."""
    checks: dict[str, EvidenceCheck] = {
        "info_sufficiency": evidence.info_sufficiency,
        "severity_clarity": evidence.severity_clarity,
        "component_clarity": evidence.component_clarity,
        "duplicate_ambiguity": evidence.duplicate_ambiguity,
    }

    triggered = [
        name
        for name, check in checks.items()
        if check.level is EvidenceLevel.LOW
    ]

    if not triggered:
        return GateDecision(action=GateAction.PROCEED)

    return GateDecision(
        action=GateAction.ASK_HUMAN,
        triggered=triggered,
        question=_build_question(triggered),
    )


def _build_question(triggered: list[str]) -> str:
    # Ask about the weakest dimensions, in a stable priority order so the
    # question is predictable (info first — it's the most common blocker).
    order = [
        "info_sufficiency",
        "duplicate_ambiguity",
        "severity_clarity",
        "component_clarity",
    ]
    parts = [_QUESTION_FOR[name] for name in order if name in triggered]
    return " ".join(parts)
