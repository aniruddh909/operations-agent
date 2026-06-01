"""Collect the structured evidence behind a triage decision.

Three of the four dimensions (info sufficiency, severity clarity, component
clarity) are assessed by the model reading the report. The fourth,
``duplicate_ambiguity``, is NOT left to the model — it is derived in code from
the Slice 3 duplicate verdict, so that signal stays grounded in the concrete
cosine result rather than the model's self-report.
"""

from __future__ import annotations

from .clients import ModelClient
from .models import (
    DuplicateClassification,
    DuplicateVerdict,
    EvidenceCheck,
    EvidenceChecks,
    EvidenceLevel,
)

EVIDENCE_SYSTEM = """You assess how confidently a bug report can be triaged. \
For each dimension, give a level (high/medium/low) and a one-line justification.

Dimensions:
- info_sufficiency: is there enough detail (repro steps, expected vs actual) to \
file a useful ticket? low = too vague to act on.
- severity_clarity: is the severity/impact clear enough to set a priority? \
low = no signal about how bad it is.
- component_clarity: is it clear which part of the product is affected? \
low = can't tell which component.

Call `submit_evidence` exactly once."""


def _evidence_tool() -> dict:
    level = {"type": "string", "enum": ["high", "medium", "low"]}
    check = {
        "type": "object",
        "properties": {"level": level, "justification": {"type": "string"}},
        "required": ["level", "justification"],
    }
    return {
        "name": "submit_evidence",
        "description": "Submit the triage evidence assessment.",
        "input_schema": {
            "type": "object",
            "properties": {
                "info_sufficiency": check,
                "severity_clarity": check,
                "component_clarity": check,
            },
            "required": [
                "info_sufficiency",
                "severity_clarity",
                "component_clarity",
            ],
        },
    }


def collect_evidence(
    bug_text: str,
    *,
    model: ModelClient,
    duplicate: DuplicateVerdict | None,
) -> EvidenceChecks:
    """Assess info/severity/component via the model; derive duplicate_ambiguity."""
    response = model.call(
        system=EVIDENCE_SYSTEM,
        messages=[{"role": "user", "content": f"Bug report:\n{bug_text}"}],
        tools=[_evidence_tool()],
    )
    raw = response.tool_input

    return EvidenceChecks(
        info_sufficiency=_check(raw["info_sufficiency"]),
        severity_clarity=_check(raw["severity_clarity"]),
        component_clarity=_check(raw["component_clarity"]),
        duplicate_ambiguity=_duplicate_check(duplicate),
    )


def _check(raw: dict) -> EvidenceCheck:
    return EvidenceCheck(
        level=EvidenceLevel(raw["level"]),
        justification=raw.get("justification", ""),
    )


def _duplicate_check(duplicate: DuplicateVerdict | None) -> EvidenceCheck:
    """Map the Slice 3 duplicate verdict onto an evidence level.

    AMBIGUOUS duplicate => LOW confidence (ask a human). A CLEAR duplicate never
    reaches here (it short-circuits the run before evidence is collected); NONE
    means we're confident it's novel.
    """
    if duplicate is None:
        return EvidenceCheck(
            level=EvidenceLevel.HIGH,
            justification="Duplicate detection not run.",
        )
    if duplicate.classification is DuplicateClassification.AMBIGUOUS:
        top = duplicate.candidates[0] if duplicate.candidates else None
        near = f" (closest: {top.key} @ {top.score})" if top else ""
        return EvidenceCheck(
            level=EvidenceLevel.LOW,
            justification=f"Similar to an existing ticket but unconfirmed{near}.",
        )
    return EvidenceCheck(
        level=EvidenceLevel.HIGH,
        justification="No ambiguous duplicate found; treat as novel.",
    )
