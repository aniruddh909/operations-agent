"""Tests for record/replay of model responses."""

from __future__ import annotations

from operations_agent.clients import ModelResponse
from operations_agent.replay import RecordingModelClient, ReplayModelClient


class StubModel:
    def __init__(self, *responses):
        self._responses = list(responses)
        self._i = 0

    def call(self, *, system, messages, tools):
        r = self._responses[self._i]
        self._i += 1
        return r


def test_record_then_replay_reproduces_responses(tmp_path):
    cassette = tmp_path / "c.json"
    inner = StubModel(
        ModelResponse(tool_name="submit_evidence", tool_input={"a": 1}),
        ModelResponse(tool_name="submit_plan", tool_input={"b": 2}),
    )
    rec = RecordingModelClient(inner, cassette)
    r1 = rec.call(system="s", messages=[], tools=[{"name": "submit_evidence"}])
    r2 = rec.call(system="s", messages=[], tools=[{"name": "submit_plan"}])

    # Replay returns the same sequence, no inner client involved.
    replay = ReplayModelClient(cassette)
    p1 = replay.call(system="x", messages=[], tools=[{"name": "submit_evidence"}])
    p2 = replay.call(system="x", messages=[], tools=[{"name": "submit_plan"}])

    assert (p1.tool_name, p1.tool_input) == (r1.tool_name, r1.tool_input)
    assert (p2.tool_name, p2.tool_input) == (r2.tool_name, r2.tool_input)


def test_replay_exhaustion_raises(tmp_path):
    cassette = tmp_path / "c.json"
    RecordingModelClient(
        StubModel(ModelResponse(tool_name="submit_plan", tool_input={})), cassette
    ).call(system="s", messages=[], tools=[{"name": "submit_plan"}])

    replay = ReplayModelClient(cassette)
    replay.call(system="x", messages=[], tools=[{"name": "submit_plan"}])
    try:
        replay.call(system="x", messages=[], tools=[{"name": "submit_plan"}])
        assert False, "expected exhaustion error"
    except RuntimeError as e:
        assert "exhausted" in str(e)
