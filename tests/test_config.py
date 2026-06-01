"""Tests for typed config + fail-fast behavior.

These don't touch the network: they construct ``Settings`` directly and assert
the fail-fast contract. Env-file loading is bypassed by passing values in.
"""

from __future__ import annotations

import pytest

from operations_agent.config import MissingConfigError, Settings


def _settings(**overrides) -> Settings:
    # _env_file=None so a developer's real .env doesn't leak into the test.
    return Settings(_env_file=None, **overrides)


def test_require_live_passes_when_all_secrets_present():
    s = _settings(
        anthropic_api_key="sk-ant-x",
        jira_base_url="https://x.atlassian.net",
        jira_email="a@b.com",
        jira_api_token="tok",
    )
    s.require_live()  # should not raise


def test_require_live_names_every_missing_var():
    s = _settings(anthropic_api_key="sk-ant-x")  # Jira ones missing

    with pytest.raises(MissingConfigError) as exc:
        s.require_live()

    msg = str(exc.value)
    assert "JIRA_BASE_URL" in msg
    assert "JIRA_EMAIL" in msg
    assert "JIRA_API_TOKEN" in msg
    assert "ANTHROPIC_API_KEY" not in msg  # this one was provided


def test_tunable_defaults_present():
    s = _settings()
    assert s.jira_project_key == "OPS"
    assert s.planning_model
    assert s.judge_model
