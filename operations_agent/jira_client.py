"""Real Jira client implementing the ``JiraClient`` protocol.

Talks to the Jira Cloud REST API v3 over basic auth (email + API token). Kept
separate from the agent loop and lazily constructed by the CLI so that tests and
``--help`` never need network or credentials.

Priority strings (P0..P3) are mapped to whatever names the target Jira instance
uses; the default map matches a common scheme and can be overridden.
"""

from __future__ import annotations

from typing import Any

from .errors import SemanticError, ToolError, TransientError, classify_status

# Maps our internal priority vocabulary to Jira priority names.
DEFAULT_PRIORITY_MAP = {
    "P0": "Highest",
    "P1": "High",
    "P2": "Medium",
    "P3": "Low",
}


# Back-compat alias: existing call sites/tests refer to JiraError. It is now the
# generic ToolError base; the client raises the more specific Transient/Semantic
# subclasses, which are themselves ToolErrors.
JiraError = ToolError


class JiraCloudClient:
    """Creates issues via the Jira Cloud REST API."""

    def __init__(
        self,
        *,
        base_url: str,
        email: str,
        api_token: str,
        project_key: str,
        issue_type: str = "Bug",
        priority_map: dict[str, str] | None = None,
        use_native_priority: bool = False,
        timeout: float = 30.0,
    ) -> None:
        import httpx

        self._project_key = project_key
        self._issue_type = issue_type
        self._priority_map = priority_map or DEFAULT_PRIORITY_MAP
        # Team-managed (next-gen) projects often lack a native Priority field.
        # When that's the case we encode priority/component as labels so the
        # agent's decision is still visible in Jira without a 400.
        self._use_native_priority = use_native_priority
        self._client = httpx.Client(
            base_url=_normalize_base_url(base_url),
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
            "issuetype": {"name": self._issue_type},
            "description": _to_adf(description),
        }

        if self._use_native_priority:
            fields["priority"] = {
                "name": self._priority_map.get(priority, "Medium")
            }
            if component:
                fields["components"] = [{"name": component}]
        else:
            # Encode the agent's triage decision as labels (portable everywhere).
            labels = [f"priority-{priority}"]
            if component:
                labels.append(f"component-{_slug(component)}")
            fields["labels"] = labels

        resp = self._request("POST", "/rest/api/3/issue", json={"fields": fields})
        if resp.status_code >= 400:
            raise classify_status(resp.status_code, resp.text)

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

    def fetch_all(self, *, page_size: int = 100) -> list[dict[str, Any]]:
        """Return all issues in the project as {key, summary, description}.

        Used by ``reindex`` to rebuild the local vector index from Jira's
        current state. Uses the token-paginated ``/search/jql`` endpoint (the
        old ``/search`` was removed by Atlassian). Descriptions come back as ADF,
        which we flatten to text.
        """
        results: list[dict[str, Any]] = []
        jql = f"project = {self._project_key} ORDER BY created ASC"
        next_token: str | None = None
        while True:
            params: dict[str, Any] = {
                "jql": jql,
                "maxResults": page_size,
                "fields": "summary,description",
            }
            if next_token:
                params["nextPageToken"] = next_token
            resp = self._request("GET", "/rest/api/3/search/jql", params=params)
            if resp.status_code >= 400:
                raise classify_status(resp.status_code, resp.text)
            data = resp.json()
            for issue in data.get("issues", []):
                fields = issue.get("fields", {})
                results.append(
                    {
                        "key": issue["key"],
                        "summary": fields.get("summary", ""),
                        "description": _adf_to_text(fields.get("description")),
                    }
                )
            next_token = data.get("nextPageToken")
            if not next_token or data.get("isLast", True):
                break
        return results

    def _request(self, method: str, url: str, **kwargs):
        """Issue a request, mapping network-level failures to TransientError.

        Timeouts and connection errors are transient by nature, so they become
        TransientError (retryable below the loop) rather than crashing the run.
        """
        import httpx

        try:
            return self._client.request(method, url, **kwargs)
        except (httpx.TimeoutException, httpx.TransportError) as err:
            raise TransientError(None, f"network error: {err}") from err

    def close(self) -> None:
        self._client.close()


def fetch_all_tickets(settings) -> list:
    """Build a client from settings and return tickets as ``IndexedTicket``s."""
    from .index import IndexedTicket

    client = JiraCloudClient(
        base_url=settings.jira_base_url,
        email=settings.jira_email,
        api_token=settings.jira_api_token,
        project_key=settings.jira_project_key,
    )
    try:
        return [
            IndexedTicket(key=t["key"], summary=t["summary"], text=t["description"])
            for t in client.fetch_all()
        ]
    finally:
        client.close()


def _adf_to_text(adf: Any) -> str:
    """Flatten an Atlassian Document Format value to plain text (best effort)."""
    if not isinstance(adf, dict):
        return ""
    out: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "text" and isinstance(node.get("text"), str):
                out.append(node["text"])
            for child in node.get("content", []) or []:
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(adf)
    return " ".join(out).strip()


def _normalize_base_url(base_url: str) -> str:
    """Tolerate a base URL with no scheme (e.g. ``site.atlassian.net``)."""
    url = base_url.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def _slug(text: str) -> str:
    return "-".join(text.lower().split())


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
