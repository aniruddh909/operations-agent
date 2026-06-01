"""Record/replay for the model client — deterministic, network-free runs.

Two wrappers around a ``ModelClient``:

- ``RecordingModelClient`` wraps a real client and writes every response to a
  cassette file (JSON list of tool_name/tool_input pairs) as it runs.
- ``ReplayModelClient`` reads a cassette and returns those responses in order,
  making no network calls. This powers the dry-run mode and lets the demo (and
  tests) replay a known-good run with zero dependence on live conditions.

Cassettes are keyed only by call order, which is enough here because a triage
run makes a fixed, deterministic sequence of model calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .clients import ModelClient, ModelResponse


class RecordingModelClient:
    """Wraps a real model client and records each response to a cassette."""

    def __init__(self, inner: ModelClient, cassette_path: str | Path) -> None:
        self._inner = inner
        self._path = Path(cassette_path)
        self._recorded: list[dict[str, Any]] = []

    def call(self, *, system, messages, tools) -> ModelResponse:
        response = self._inner.call(
            system=system, messages=messages, tools=tools
        )
        self._recorded.append(
            {
                "tool_name": response.tool_name,
                "tool_input": response.tool_input,
            }
        )
        self._flush()
        return response

    def _flush(self) -> None:
        self._path.write_text(json.dumps(self._recorded, indent=2))


class ReplayModelClient:
    """Replays recorded responses in order; makes no network calls."""

    def __init__(self, cassette_path: str | Path) -> None:
        data = json.loads(Path(cassette_path).read_text())
        self._queue = list(data)
        self._i = 0

    def call(self, *, system, messages, tools) -> ModelResponse:
        if self._i >= len(self._queue):
            raise RuntimeError(
                "Replay cassette exhausted: more model calls than recorded. "
                "Re-record the cassette."
            )
        entry = self._queue[self._i]
        self._i += 1
        return ModelResponse(
            tool_name=entry["tool_name"], tool_input=entry["tool_input"]
        )
