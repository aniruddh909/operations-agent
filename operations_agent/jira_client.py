"""Real Jira client implementing the ``JiraClient`` protocol.

Talks to the Jira Cloud REST API v3 over basic auth (email + API token). Kept
separate from the agent loop and lazily constructed by the CLI so that tests and
``--help`` never need network or credentials.

Priority strings (P0..P3) are mapped to whatever names the target Jira instance
uses; the default map matches a common scheme and can be overridden.
"""

from __future__ import annotations

from typing import Any

# Maps our internal priority vocabulary to Jira priority names.
DEFAULT_PRIORITY_MAP = {
    "P0": "Highest",
    "P1": "High",
    "P2": "Medium",
    "P3": "Low",
}


class JiraError(RuntimeError):
    """Raised when Jira returns an error response.

    Slice 5 will distinguish transient (retry) from semantic (re-plan) errors;
    for now this surfaces a readable message and the HTTP status.
    """

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        super().__init__(f"Jira {status}: {message}")


class JiraCloudClient:
    """Creates issues via the Jira Cloud REST API."""

    def __init__(
        self,
        *,
        base_url: str,
        email: str,
        api_token: str,
        project_key: str,
        priority_map: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        import httpx

        self._project_key = project_key
        self._priority_map = priority_map or DEFAULT_PRIORITY_MAP
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            auth=(email, api_token),
            timeout=timeout,
            headers={"Accept": "application/json"},
        )

    def create_ticket(
        self,
        *,
        summary: str,
        description: str,
        priority: str,
        component: str | None = None,
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "project": {"key": self._project_key},
            "summary": summary,
            "issuetype": {"name": "Bug"},
            "description": _to_adf(description),
            "priority": {"name": self._priority_map.get(priority, "Medium")},
        }
        if component:
            fields["components"] = [{"name": component}]

        resp = self._client.post("/rest/api/3/issue", json={"fields": fields})
        if resp.status_code >= 400:
            raise JiraError(resp.status_code, resp.text)

        data = resp.json()
        key = data.get("key")
        return {
            "key": key,
            "summary": summary,
            "description": description,
            "priority": priority,
            "component": component,
            "url": f"{self._client.base_url}/browse/{key}",
        }

    def close(self) -> None:
        self._client.close()


def _to_adf(text: str) -> dict[str, Any]:
    """Wrap plain text in Atlassian Document Format (required by REST v3)."""
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": text or " "}],
            }
        ],
    }
