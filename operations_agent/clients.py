"""Client seams: the model and the tools the agent can call.

Everything the agent talks to is behind a Protocol so the loop can be driven
with fakes in tests and recorded responses in replay mode (Slice 5). The real
implementations wrap the Anthropic SDK and (later) Jira/Slack.

Slice 1 keeps the tool surface to a single fake ``create_ticket`` — enough to
prove the loop end-to-end. Real Jira arrives in Slice 2.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


# --------------------------------------------------------------------------- #
# Model client
# --------------------------------------------------------------------------- #


class ModelResponse(BaseModel):
    """A single tool-use result from the model.

    The agent constrains the model to respond by calling exactly one tool, so a
    response is just the chosen tool name plus its raw (unvalidated) input. The
    agent is responsible for validating ``tool_input`` against the matching
    Pydantic schema and, on failure, issuing one repair-retry.
    """

    tool_name: str
    tool_input: dict[str, Any]


@runtime_checkable
class ModelClient(Protocol):
    """Thin wrapper over a tool-use chat completion.

    Deliberately low-level: the validate-then-repair loop lives in the agent
    (the part worth showing), not hidden inside this client. ``messages`` and
    ``tools`` follow the Anthropic Messages API shape.
    """

    def call(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse: ...


# --------------------------------------------------------------------------- #
# Tool clients
# --------------------------------------------------------------------------- #


@runtime_checkable
class JiraClient(Protocol):
    """The ticket-store seam. Real implementation lands in Slice 2."""

    def create_ticket(
        self,
        *,
        summary: str,
        description: str,
        priority: str,
        component: str | None = None,
    ) -> dict[str, Any]:
        """Create a ticket and return at least ``{"key": <id>, ...}``."""
        ...


class FakeJiraClient:
    """In-memory ticket store for the walking skeleton and tests."""

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def create_ticket(
        self,
        *,
        summary: str,
        description: str,
        priority: str,
        component: str | None = None,
    ) -> dict[str, Any]:
        key = f"OPS-{len(self.created) + 1}"
        ticket = {
            "key": key,
            "summary": summary,
            "description": description,
            "priority": priority,
            "component": component,
        }
        self.created.append(ticket)
        return ticket


# --------------------------------------------------------------------------- #
# Human-in-the-loop
# --------------------------------------------------------------------------- #


@runtime_checkable
class HumanClient(Protocol):
    """The seam for asking a human a clarifying question and getting an answer."""

    def ask(self, question: str) -> str:
        """Put the question to the human and return their answer."""
        ...


class CliHumanClient:
    """Asks via stdin/stdout — the interactive path for the CLI."""

    def ask(self, question: str) -> str:
        # Prompt to stderr so it never corrupts the JSON Trace on stdout.
        import sys

        print(f"\n[clarification needed] {question}", file=sys.stderr)
        print("> ", end="", file=sys.stderr, flush=True)
        try:
            return input().strip()
        except EOFError:
            return ""


class ScriptedHumanClient:
    """Returns canned answers in order — for tests and replay."""

    def __init__(self, *answers: str) -> None:
        self._queue = list(answers)
        self.asked: list[str] = []

    def ask(self, question: str) -> str:
        self.asked.append(question)
        return self._queue.pop(0) if self._queue else ""


# --------------------------------------------------------------------------- #
# Slack notifications
# --------------------------------------------------------------------------- #


@runtime_checkable
class SlackClient(Protocol):
    """The seam for posting a message to Slack."""

    def post(self, text: str) -> dict[str, Any]:
        """Post ``text`` and return at least ``{"ok": bool}``."""
        ...


class FakeSlackClient:
    """In-memory Slack for tests and offline runs; records posted messages."""

    def __init__(self) -> None:
        self.posted: list[str] = []

    def post(self, text: str) -> dict[str, Any]:
        self.posted.append(text)
        return {"ok": True, "text": text}
