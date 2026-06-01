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


def _client_with(handler) -> JiraCloudClient:
    client = JiraCloudClient(
        base_url="https://example.atlassian.net",
        email="a@b.com",
        api_token="tok",
        project_key="OPS",
    )
    client._client = httpx.Client(
        base_url="https://example.atlassian.net",
        transport=httpx.MockTransport(handler),
    )
    return client


def test_create_ticket_builds_request_and_returns_key():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json={"key": "OPS-42"})

    client = _client_with(handler)
    result = client.create_ticket(
        summary="Login crash",
        description="crashes on login",
        priority="P1",
        component="auth",
    )

    assert result["key"] == "OPS-42"
    assert result["priority"] == "P1"
    assert "OPS-42" in result["url"]
    fields = captured["body"]["fields"]
    assert fields["project"]["key"] == "OPS"
    assert fields["priority"]["name"] == "High"  # P1 -> High
    assert fields["components"] == [{"name": "auth"}]
    # description is wrapped in ADF
    assert fields["description"]["type"] == "doc"


def test_create_ticket_raises_on_error_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad request")

    client = _client_with(handler)
    with pytest.raises(JiraError) as exc:
        client.create_ticket(
            summary="x", description="y", priority="P2"
        )
    assert exc.value.status == 400
