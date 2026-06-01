"""Seeding and reindexing helpers.

``seed`` files the fixture backlog into Jira and indexes each filed ticket.
``reindex`` rebuilds the local vector index from a list of tickets (the
consistency-recovery path the ``reindex`` CLI command uses). Both are thin glue
over the Jira client and the index so they're easy to test with fakes.
"""

from __future__ import annotations

from .clients import JiraClient
from .fixtures import SEED_FIXTURES, FixtureBug
from .index import IndexedTicket, TicketIndex


def seed(*, jira: JiraClient, index: TicketIndex, fixtures: list[FixtureBug] | None = None) -> list[str]:
    """File each fixture into Jira and index it. Returns the created keys.

    Priority is left to ``P3`` for seed data — these are backlog tickets, not
    freshly triaged ones; the agent's job is to triage *new* bugs against them.
    """
    fixtures = fixtures if fixtures is not None else SEED_FIXTURES
    keys: list[str] = []
    for bug in fixtures:
        ticket = jira.create_ticket(
            summary=bug.summary,
            description=bug.description,
            priority="P3",
            component=bug.component,
        )
        key = ticket["key"]
        index.add(
            IndexedTicket(key=key, summary=bug.summary, text=bug.description)
        )
        keys.append(key)
    return keys


def reindex(*, index: TicketIndex, tickets: list[IndexedTicket]) -> int:
    """Rebuild the index from the given tickets. Returns the count indexed."""
    return index.rebuild(tickets)
