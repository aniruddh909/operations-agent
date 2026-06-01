"""Tests for the real Jira client's request/response handling.

The HTTP boundary is faked with a httpx MockTransport so we exercise the actual
request construction (ADF body, priority mapping, project key) and error
handling without a live Jira. This is the prior art for tool-client tests:
assert on what crosses the boundary, not on internals.
"""

from __future__ import annotations

import httpx
import pytest

from operations_agent.jira_client import JiraCloudClient, JiraError


def _client_with(handler, **kwargs) -> JiraCloudClient:
    client = JiraCloudClient(
        base_url="https://example.atlassian.net",
        email="a@b.com",
        api_token="tok",
        project_key="OPS",
        **kwargs,
    )
    client._client = httpx.Client(
        base_url="https://example.atlassian.net",
        transport=httpx.MockTransport(handler),
    )
    return client


def test_create_ticket_default_encodes_triage_as_labels():
    # Default (portable) path: priority + component become labels, so it works
    # on team-managed projects that lack a native Priority field.
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json={"key": "OPS-42"})

    client = _client_with(handler)
    result = client.create_ticket(
        summary="Login crash",
        description="crashes on login",
        priority="P1",
        component="auth service",
    )

    assert result["key"] == "OPS-42"
    assert result["priority"] == "P1"
    fields = captured["body"]["fields"]
    assert fields["project"]["key"] == "OPS"
    assert fields["issuetype"]["name"] == "Bug"
    assert "priority" not in fields  # not sent natively
    assert "priority-P1" in fields["labels"]
    assert "component-auth-service" in fields["labels"]
    assert fields["description"]["type"] == "doc"  # ADF-wrapped


def test_create_ticket_native_priority_when_enabled():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json={"key": "OPS-7"})

    client = _client_with(handler, use_native_priority=True)
    client.create_ticket(
        summary="x", description="y", priority="P1", component="auth"
    )

    fields = captured["body"]["fields"]
    assert fields["priority"]["name"] == "High"  # P1 -> High
    assert fields["components"] == [{"name": "auth"}]


def test_base_url_without_scheme_is_normalized():
    c = JiraCloudClient(
        base_url="site.atlassian.net",
        email="a@b.com",
        api_token="tok",
        project_key="OPS",
    )
    assert str(c._client.base_url).startswith("https://site.atlassian.net")
    c.close()


def test_fetch_all_uses_jql_search_endpoint_and_paginates():
    # Regression: Atlassian removed /rest/api/3/search (410). fetch_all must use
    # the token-paginated /search/jql endpoint.
    seen_paths = []
    pages = {
        None: {
            "issues": [
                {"key": "KAN-1", "fields": {"summary": "a", "description": None}}
            ],
            "nextPageToken": "tok2",
            "isLast": False,
        },
        "tok2": {
            "issues": [
                {"key": "KAN-2", "fields": {"summary": "b", "description": None}}
            ],
            "isLast": True,
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        token = request.url.params.get("nextPageToken")
        return httpx.Response(200, json=pages[token])

    client = _client_with(handler)
    results = client.fetch_all()

    assert all(p == "/rest/api/3/search/jql" for p in seen_paths)
    assert [r["key"] for r in results] == ["KAN-1", "KAN-2"]


def test_create_ticket_raises_on_error_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad request")

    client = _client_with(handler)
    with pytest.raises(JiraError) as exc:
        client.create_ticket(
            summary="x", description="y", priority="P2"
        )
    assert exc.value.status == 400
