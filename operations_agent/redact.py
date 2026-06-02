"""Redact a Trace before saving it as a shareable artifact.

Saved traces are portfolio evidence and may be committed, so they must not leak
secrets. This strips anything token/auth-shaped and, optionally, scrubs the
free-text bug/ticket content (which could contain customer data). The redaction
operates on the serialized dict so it covers every nested field uniformly.
"""

from __future__ import annotations

import re
from typing import Any

from .models import Trace

# Patterns for secret-shaped values anywhere in the trace.
_SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9\-_]+"),          # Anthropic keys
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]+"),         # Slack tokens
    re.compile(r"https://hooks\.slack\.com/\S+"),     # Slack webhooks
    re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),  # emails
    re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*"),    # bearer tokens
]
_REDACTED = "[REDACTED]"

# Keys whose values are scrubbed wholesale when scrub_text=True.
_TEXT_KEYS = {"raw_text", "description", "summary", "text", "message", "justification"}


def redact_trace(trace: Trace, *, scrub_text: bool = False) -> dict:
    """Return a redacted serializable dict of the Trace.

    - Always: replace secret-shaped substrings (keys, tokens, webhooks, emails).
    - ``scrub_text``: additionally replace free-text content fields wholesale.
    """
    data = trace.model_dump(mode="json")
    return _walk(data, scrub_text=scrub_text)


def _walk(node: Any, *, scrub_text: bool, key: str | None = None) -> Any:
    if isinstance(node, dict):
        return {k: _walk(v, scrub_text=scrub_text, key=k) for k, v in node.items()}
    if isinstance(node, list):
        return [_walk(v, scrub_text=scrub_text) for v in node]
    if isinstance(node, str):
        if scrub_text and key in _TEXT_KEYS and node:
            return _REDACTED
        return _redact_secrets(node)
    return node


def _redact_secrets(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(_REDACTED, text)
    return text
