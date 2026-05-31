"""Real ``ModelClient`` backed by the Anthropic SDK.

Kept apart from the agent loop so the loop has no hard dependency on the SDK and
stays trivially testable. This wrapper does one thing: turn a tool-use request
into a single ``ModelResponse`` (the chosen tool + its raw input). All
validation and repair logic stays in the agent.
"""

from __future__ import annotations

import os
from typing import Any

from .clients import ModelResponse

DEFAULT_MODEL = "claude-sonnet-4-6"


class AnthropicModelClient:
    """Adapts ``anthropic.Anthropic`` to the ``ModelClient`` protocol."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 2048,
        api_key: str | None = None,
    ) -> None:
        # Imported lazily so tests and `--help` don't require the SDK/key.
        from anthropic import Anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to your environment or "
                ".env file before running live triage."
            )
        self._client = Anthropic(api_key=key)
        self._model = model
        self._max_tokens = max_tokens

    def call(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        # Force the model to answer via the (single) provided tool.
        tool_choice = {"type": "tool", "name": tools[0]["name"]}
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                return ModelResponse(
                    tool_name=block.name, tool_input=dict(block.input)
                )
        raise RuntimeError("Model returned no tool_use block.")
