"""CLI entry points for seeding and reindexing the ticket index.

- ``ops-seed`` files the fixture backlog into the configured Jira project and
  builds the local index from it. Run once against a fresh sandbox.
- ``ops-reindex`` rebuilds the local index from the tickets currently in Jira,
  the recovery path when the index drifts from Jira.

Both build their clients from typed ``Settings`` and the local embedding model,
so they share the same configuration surface as the agent.
"""

from __future__ import annotations

import sys

from .config import MissingConfigError, Settings
from .embeddings import LocalEmbeddingClient
from .index import IndexedTicket, TicketIndex


def _build_index(settings: Settings) -> TicketIndex:
    return TicketIndex(
        settings.index_path, LocalEmbeddingClient(settings.embedding_model)
    )


def seed_main(argv: list[str] | None = None) -> int:
    settings = Settings()
    try:
        settings.require_live()
    except MissingConfigError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    from .jira_client import JiraCloudClient
    from .seed import seed

    jira = JiraCloudClient(
        base_url=settings.jira_base_url,  # type: ignore[arg-type]
        email=settings.jira_email,  # type: ignore[arg-type]
        api_token=settings.jira_api_token,  # type: ignore[arg-type]
        project_key=settings.jira_project_key,
    )
    index = _build_index(settings)
    keys = seed(jira=jira, index=index)
    print(f"Seeded {len(keys)} tickets: {', '.join(keys)}")
    return 0


def reindex_main(argv: list[str] | None = None) -> int:
    settings = Settings()
    try:
        settings.require_live()
    except MissingConfigError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    from .jira_client import fetch_all_tickets
    from .seed import reindex

    tickets: list[IndexedTicket] = fetch_all_tickets(settings)
    index = _build_index(settings)
    count = reindex(index=index, tickets=tickets)
    print(f"Reindexed {count} tickets from Jira project {settings.jira_project_key}.")
    return 0


if __name__ == "__main__":
    sys.exit(seed_main())
