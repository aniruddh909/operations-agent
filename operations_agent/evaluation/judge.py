"""LLM-as-judge pass for the soft qualities exact-match can't score.

A cheap model (Haiku) grades two subjective dimensions on a 1-5 rubric:
- reasoning quality: are the evidence justifications sound and specific?
- summary quality: is the generated Slack summary clear and grounded?

This complements the exact-match scorer: the scorer says *whether* a decision
was right; the judge says *how good* the reasoning behind it was. Kept thin and
optional so a full eval run doesn't require it.
"""

from __future__ import annotations

from pydantic import BaseModel

from ..clients import ModelClient
from ..models import EventType, Trace

JUDGE_SYSTEM = """You grade an AI triage agent's work on two dimensions, each \
1-5 (5 best). Be a strict but fair reviewer.
- reasoning_quality: are the evidence justifications specific and sound (not \
generic filler)?
- summary_quality: if a summary was produced, is it clear, accurate to the \
ticket, and useful? If no summary exists, score 3 (neutral).
Call `submit_grades` exactly once with integer scores and a one-line note."""


def _judge_tool() -> dict:
    score = {"type": "integer", "minimum": 1, "maximum": 5}
    return {
        "name": "submit_grades",
        "description": "Submit rubric grades.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning_quality": score,
                "summary_quality": score,
                "note": {"type": "string"},
            },
            "required": ["reasoning_quality", "summary_quality", "note"],
        },
    }


class JudgeResult(BaseModel):
    reasoning_quality: int
    summary_quality: int
    note: str


def judge_trace(trace: Trace, *, judge_model: ModelClient) -> JudgeResult:
    """Grade one Trace's reasoning and summary quality with the cheap model."""
    evidence = _evidence_text(trace)
    summary = _summary_text(trace)
    context = (
        f"Bug: {trace.bug_report.raw_text}\n\n"
        f"Evidence justifications:\n{evidence}\n\n"
        f"Generated summary:\n{summary or '(none)'}"
    )
    response = judge_model.call(
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": context}],
        tools=[_judge_tool()],
    )
    return JudgeResult.model_validate(response.tool_input)


def _evidence_text(trace: Trace) -> str:
    for ev in trace.events:
        if ev.type is EventType.EVIDENCE_SUBMITTED:
            return "\n".join(
                f"- {dim}: {ev.data.get(dim, {}).get('level')} - "
                f"{ev.data.get(dim, {}).get('justification')}"
                for dim in (
                    "info_sufficiency",
                    "severity_clarity",
                    "component_clarity",
                    "duplicate_ambiguity",
                )
            )
    return "(none)"


def _summary_text(trace: Trace) -> str | None:
    for ev in trace.events:
        if ev.type is EventType.STEP_COMPLETED and ev.data.get("tool") == "generate_triage_summary":
            return (ev.data.get("result") or {}).get("summary")
    return None
