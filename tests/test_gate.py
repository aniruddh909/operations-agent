"""Unit tests for the pure confidence-gating policy.

The gate is deterministic code, so it's tested directly (no model). These pin
the policy: any LOW dimension forces an ask; all-clear proceeds.
"""

from __future__ import annotations

from operations_agent.gate import gate
from operations_agent.models import (
    EvidenceCheck,
    EvidenceChecks,
    EvidenceLevel,
    GateAction,
)


def _checks(info="high", severity="high", component="high", dup="high"):
    def c(level):
        return EvidenceCheck(level=EvidenceLevel(level), justification="x")

    return EvidenceChecks(
        info_sufficiency=c(info),
        severity_clarity=c(severity),
        component_clarity=c(component),
        duplicate_ambiguity=c(dup),
    )


def test_all_high_proceeds():
    decision = gate(_checks())
    assert decision.action is GateAction.PROCEED
    assert decision.triggered == []
    assert decision.question is None


def test_medium_levels_still_proceed():
    decision = gate(_checks(info="medium", severity="medium"))
    assert decision.action is GateAction.PROCEED


def test_low_info_triggers_ask_with_question():
    decision = gate(_checks(info="low"))
    assert decision.action is GateAction.ASK_HUMAN
    assert "info_sufficiency" in decision.triggered
    assert decision.question  # non-empty clarifying question


def test_low_duplicate_ambiguity_triggers_ask():
    decision = gate(_checks(dup="low"))
    assert decision.action is GateAction.ASK_HUMAN
    assert "duplicate_ambiguity" in decision.triggered


def test_multiple_low_dimensions_all_reported():
    decision = gate(_checks(info="low", component="low"))
    assert decision.action is GateAction.ASK_HUMAN
    assert set(decision.triggered) == {"info_sufficiency", "component_clarity"}
