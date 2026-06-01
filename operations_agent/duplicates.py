"""Duplicate detection: retrieve, then let the model adjudicate.

The flow, in one place:

1. Retrieve the top-K most similar existing tickets from the index (cosine).
2. If the best score is below the ambiguous band, it's a novel bug — done, no
   model call (cheap path, and most bugs are novel).
3. Otherwise ask the model to adjudicate: is the incoming bug the *same* bug as
   any candidate? It returns a same/different verdict with a justification.
4. Combine the cosine score and the model's verdict into a classification:
   CLEAR (high score + model agrees), AMBIGUOUS (mid score, or model unsure), or
   NONE. The cosine bands live in config so they're tunable.

The CLEAR/AMBIGUOUS/NONE classification is what Slice 4's confidence gate keys
off — AMBIGUOUS is the signal to ask a human.
"""

from __future__ import annotations

from .clients import ModelClient
from .index import TicketIndex
from .models import (
    DuplicateCandidate,
    DuplicateClassification,
    DuplicateVerdict,
)

ADJUDICATE_SYSTEM = """You judge whether an incoming bug report describes the \
SAME underlying bug as any of the candidate tickets — even if worded very \
differently. Two reports are the same bug if fixing one would fix the other.

Call `submit_duplicate_verdict` exactly once with:
- is_duplicate (bool): true if the incoming bug matches one candidate
- matched_key (str|null): the candidate key it matches, or null
- reasoning (str): one or two sentences justifying the call."""


def _verdict_tool() -> dict:
    return {
        "name": "submit_duplicate_verdict",
        "description": "Submit the duplicate adjudication.",
        "input_schema": {
            "type": "object",
            "properties": {
                "is_duplicate": {"type": "boolean"},
                "matched_key": {"type": ["string", "null"]},
                "reasoning": {"type": "string"},
            },
            "required": ["is_duplicate", "matched_key", "reasoning"],
        },
    }


def find_duplicate(
    bug_text: str,
    *,
    index: TicketIndex,
    model: ModelClient,
    clear_band: float,
    ambiguous_band: float,
    top_k: int = 5,
) -> DuplicateVerdict:
    """Detect whether ``bug_text`` duplicates an indexed ticket.

    ``clear_band`` >= ``ambiguous_band``. Scores at/above clear_band are strong
    matches; between the bands are ambiguous; below ambiguous_band are novel.
    """
    hits = index.search(bug_text, top_k=top_k)
    candidates = [
        DuplicateCandidate(key=h.key, summary=h.summary, score=round(h.score, 4))
        for h in hits
    ]

    if not hits or hits[0].score < ambiguous_band:
        return DuplicateVerdict(
            classification=DuplicateClassification.NONE,
            candidates=candidates,
            reasoning="No candidate exceeded the ambiguous similarity band.",
        )

    # Worth a model call: adjudicate against the retrieved candidates.
    verdict = _adjudicate(bug_text, hits, model=model)
    top_score = hits[0].score

    if verdict["is_duplicate"]:
        classification = (
            DuplicateClassification.CLEAR
            if top_score >= clear_band
            else DuplicateClassification.AMBIGUOUS
        )
        matched_key = verdict.get("matched_key") or hits[0].key
    else:
        # Model says different. If similarity is very high we still flag it as
        # ambiguous (worth a human glance); otherwise treat as novel.
        classification = (
            DuplicateClassification.AMBIGUOUS
            if top_score >= clear_band
            else DuplicateClassification.NONE
        )
        matched_key = None

    return DuplicateVerdict(
        classification=classification,
        candidates=candidates,
        matched_key=matched_key,
        reasoning=verdict.get("reasoning", ""),
    )


def _adjudicate(bug_text: str, hits, *, model: ModelClient) -> dict:
    candidate_block = "\n\n".join(
        f"[{h.key}] {h.summary}\n{h.text}" for h in hits
    )
    messages = [
        {
            "role": "user",
            "content": (
                f"Incoming bug:\n{bug_text}\n\n"
                f"Candidate tickets:\n{candidate_block}"
            ),
        }
    ]
    response = model.call(
        system=ADJUDICATE_SYSTEM, messages=messages, tools=[_verdict_tool()]
    )
    return response.tool_input
