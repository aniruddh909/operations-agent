"""Golden-set case schema and loader.

Each case is a bug report plus the *expected* triage decisions. Fields are
optional where a dimension doesn't apply (e.g. a report we expect to be filed
won't have an expected duplicate key). The harness only scores the labels that
are present, so cases stay focused on what they're testing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

DEFAULT_GOLDEN_PATH = Path(__file__).with_name("golden_set.json")


class GoldenCase(BaseModel):
    """One labeled evaluation case."""

    id: str
    raw_text: str
    dimension: str = Field(
        description="Which scenario this case exercises (for grouping/reporting)."
    )

    # Expected decisions (any may be omitted if not applicable to the case).
    expect_gate: Optional[str] = Field(
        default=None, description="'ask_human' or 'proceed' — the gate decision."
    )
    expect_priority: Optional[str] = Field(
        default=None, description="Expected P0..P3 when the agent files."
    )
    expect_component: Optional[str] = Field(
        default=None, description="Expected component keyword (substring match)."
    )
    expect_duplicate: Optional[str] = Field(
        default=None,
        description="'clear' | 'ambiguous' | 'none' — expected duplicate verdict.",
    )
    expect_filed: Optional[bool] = Field(
        default=None, description="Whether a new ticket should be filed."
    )


def load_golden_set(path: str | Path = DEFAULT_GOLDEN_PATH) -> list[GoldenCase]:
    data = json.loads(Path(path).read_text())
    return [GoldenCase.model_validate(c) for c in data]
