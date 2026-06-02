"""Real Slack client implementing the ``SlackClient`` protocol.

Posts to a Slack Incoming Webhook URL (the simplest auth-free-ish path: one
secret URL, no OAuth dance). Kept separate from the loop and lazily constructed
so tests and ``--help`` need no network or webhook.

Webhook errors are classified into the same two-tier taxonomy as Jira, so the
agent's retry/reflection machinery treats a flaky Slack call consistently.
"""

from __future__ import annotations

from typing import Any

from .errors import TransientError, classify_status


class SlackWebhookClient:
    """Posts messages via a Slack Incoming Webhook."""

    def __init__(self, webhook_url: str, *, timeout: float = 15.0) -> None:
        import httpx

        self._url = webhook_url
        self._client = httpx.Client(timeout=timeout)

    def post(self, text: str) -> dict[str, Any]:
        try:
            resp = self._client.post(self._url, json={"text": text})
        except Exception as err:  # noqa: BLE001 - network -> transient
            raise TransientError(None, f"slack network error: {err}") from err
        if resp.status_code >= 400:
            raise classify_status(resp.status_code, resp.text)
        return {"ok": True, "text": text}

    def close(self) -> None:
        self._client.close()
