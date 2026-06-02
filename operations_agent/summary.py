"""Generate a short, readable triage summary grounded in the filed ticket.

This is the one *generative* (non-CRUD) capability the agent can compose into
its plan. The model is given the filed ticket's real fields and asked for a
2-3 sentence Slack-ready summary — so the summary is grounded in what was
actually filed, not the raw report.
"""

from __future__ import annotations

from .clients import ModelClient

SUMMARY_SYSTEM = """You write a short Slack notification announcing a bug ticket \
that was just triaged and filed. Use ONLY the ticket facts provided. Keep it to \
2-3 sentences: what the bug is, its priority and component, and the ticket key. \
Plain, professional, no emojis. Call `submit_summary` exactly once."""


def _summary_tool() -> dict:
    return {
        "name": "submit_summary",
        "description": "Submit the triage summary text.",
        "input_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    }


def generate_summary(ticket: dict, *, model: ModelClient) -> str:
    """Produce a Slack-ready summary string from a filed ticket's fields."""
    facts = (
        f"key: {ticket.get('key')}\n"
        f"summary: {ticket.get('summary')}\n"
        f"priority: {ticket.get('priority')}\n"
        f"component: {ticket.get('component')}\n"
        f"url: {ticket.get('url', '')}"
    )
    response = model.call(
        system=SUMMARY_SYSTEM,
        messages=[{"role": "user", "content": f"Ticket facts:\n{facts}"}],
        tools=[_summary_tool()],
    )
    return response.tool_input.get("summary", "").strip()
