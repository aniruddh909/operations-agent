"""Two-tier error taxonomy for tool calls.

The whole point of the two tiers is to route failures to the right handler:

- ``TransientError`` — a temporary, technical hiccup (timeout, rate limit, a 5xx).
  The *code* handles these by retrying with backoff below the loop. The agent
  never sees them; they are not reasoning failures.

- ``SemanticError`` — a meaningful, business-level failure the *agent* should
  reason about (409 duplicate already exists, 400 validation, 403 permission).
  These are surfaced to the agent as a tool result so reflection can rewrite the
  plan instead of crashing.

Anything else stays a generic error and aborts the run (we don't pretend to know
how to recover from the unknown).
"""

from __future__ import annotations


class ToolError(RuntimeError):
    """Base class for tool-call failures carrying an HTTP-ish status."""

    def __init__(self, status: int | None, message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(message)


class TransientError(ToolError):
    """A temporary failure that should be retried below the loop."""


class SemanticError(ToolError):
    """A meaningful failure the agent should reflect on and re-plan around."""


# Status codes we treat as transient (retry) vs. semantic (reflect).
TRANSIENT_STATUSES = {429, 500, 502, 503, 504}
SEMANTIC_STATUSES = {400, 403, 409, 422}


def classify_status(status: int, message: str) -> ToolError:
    """Map an HTTP status to the appropriate error tier."""
    if status in TRANSIENT_STATUSES:
        return TransientError(status, message)
    if status in SEMANTIC_STATUSES:
        return SemanticError(status, message)
    # Unknown 4xx/other: treat as semantic so the agent at least gets a chance to
    # react, rather than silently retrying something that will never succeed.
    return SemanticError(status, message)
