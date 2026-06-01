"""``triage "<text>"`` — the CLI entry point.

Builds a ``BugReport`` from the command line and runs the loop. By default it
runs *live*: real Anthropic model client + real Jira client, both configured
from typed ``Settings`` (env / ``.env``), failing fast if anything required is
missing. Pass ``--offline`` to run the loop without external accounts using the
in-memory fake Jira client (handy for trying the agent out; the offline path
grows into the replay mode in Slice 5).

Prints the resulting ``Trace`` as JSON — the same object that later drives the
live Rich view and the eval harness.
"""

from __future__ import annotations

import argparse
import sys

from .agent import run_triage
from .clients import FakeJiraClient, JiraClient
from .config import MissingConfigError, Settings
from .models import BugReport, BugSource


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="triage", description="Triage a bug report into a ticket."
    )
    parser.add_argument("text", help="The raw bug report text.")
    parser.add_argument(
        "--reporter", default=None, help="Who reported the bug (optional)."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run without external accounts (fake Jira, no live calls).",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Skip duplicate detection (no embedding model / index needed).",
    )
    args = parser.parse_args(argv)

    settings = Settings()
    bug = BugReport(
        raw_text=args.text, source=BugSource.CLI, reporter=args.reporter
    )

    try:
        model, jira = _build_clients(settings, offline=args.offline)
    except MissingConfigError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    duplicates = None
    if not args.no_dedup:
        duplicates = _build_duplicate_checker(settings)

    trace = run_triage(bug, model=model, jira=jira, duplicates=duplicates)
    print(trace.model_dump_json(indent=2))
    return 0 if trace.status and trace.status.value == "completed" else 1


def _build_clients(settings: Settings, *, offline: bool):
    """Construct the model + Jira clients, or fail fast naming missing config."""
    if offline:
        # Offline still needs the model to plan; only Jira is faked.
        if not settings.anthropic_api_key:
            raise MissingConfigError(
                "Missing required configuration for a live run: "
                "ANTHROPIC_API_KEY. Copy .env.example to .env and fill it in."
            )
        jira: JiraClient = FakeJiraClient()
    else:
        settings.require_live()
        from .jira_client import JiraCloudClient

        jira = JiraCloudClient(
            base_url=settings.jira_base_url,  # type: ignore[arg-type]
            email=settings.jira_email,  # type: ignore[arg-type]
            api_token=settings.jira_api_token,  # type: ignore[arg-type]
            project_key=settings.jira_project_key,
        )

    from .anthropic_client import AnthropicModelClient

    model = AnthropicModelClient(
        model=settings.planning_model, api_key=settings.anthropic_api_key
    )
    return model, jira


def _build_duplicate_checker(settings: Settings):
    """Construct the duplicate checker from the local index + config bands."""
    from .agent import DuplicateChecker
    from .embeddings import LocalEmbeddingClient
    from .index import TicketIndex

    index = TicketIndex(
        settings.index_path, LocalEmbeddingClient(settings.embedding_model)
    )
    return DuplicateChecker(
        index=index,
        clear_band=settings.dup_clear_band,
        ambiguous_band=settings.dup_ambiguous_band,
    )


if __name__ == "__main__":
    sys.exit(main())
