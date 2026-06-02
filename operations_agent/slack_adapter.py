"""Slack entry-point adapter.

Proves the entry-point interface boundary: a Slack event or slash-command
payload is parsed into the *same* ``BugReport`` the CLI produces, which feeds the
*same* ``run_triage`` loop. The loop is unchanged - it never sees a Slack
payload, only a BugReport. Adding Slack as a trigger was a thin adapter, not a
rewrite, which is the point of having abstracted the entry point in Slice 1.

Two common Slack shapes are supported:
- a slash command (``application/x-www-form-urlencoded`` -> dict with ``text``
  and ``user_name``)
- an Events API message event (JSON with ``event.text`` / ``event.user``)
"""

from __future__ import annotations

from .models import BugReport, BugSource


def bug_report_from_slash_command(payload: dict) -> BugReport:
    """Build a BugReport from a Slack slash-command payload.

    Slack posts slash commands as form fields; ``text`` is the command argument
    and ``user_name`` (or ``user_id``) identifies the reporter.
    """
    text = (payload.get("text") or "").strip()
    if not text:
        raise ValueError("Slash command had no bug text.")
    reporter = payload.get("user_name") or payload.get("user_id")
    return BugReport(raw_text=text, source=BugSource.SLACK, reporter=reporter)


def bug_report_from_event(payload: dict) -> BugReport:
    """Build a BugReport from a Slack Events API message payload."""
    event = payload.get("event", payload)
    text = (event.get("text") or "").strip()
    if not text:
        raise ValueError("Slack event had no message text.")
    reporter = event.get("user")
    return BugReport(raw_text=text, source=BugSource.SLACK, reporter=reporter)
