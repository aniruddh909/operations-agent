"""``triage "<text>"`` — the CLI entry point.

Builds a ``BugReport`` from the command line and runs the loop against the real
Anthropic model client and (for this slice) the in-memory fake Jira client.
Prints the resulting ``Trace`` as JSON — the same object that will later drive
the live Rich view and the eval harness.
"""

from __future__ import annotations

import argparse
import sys

from .agent import run_triage
from .clients import FakeJiraClient
from .models import BugReport, BugSource


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="triage", description="Triage a bug report into a ticket."
    )
    parser.add_argument("text", help="The raw bug report text.")
    parser.add_argument(
        "--reporter", default=None, help="Who reported the bug (optional)."
    )
    args = parser.parse_args(argv)

    bug = BugReport(
        raw_text=args.text, source=BugSource.CLI, reporter=args.reporter
    )

    # Lazy import so `triage --help` works without the SDK/key installed.
    from .anthropic_client import AnthropicModelClient

    model = AnthropicModelClient()
    jira = FakeJiraClient()

    trace = run_triage(bug, model=model, jira=jira)
    print(trace.model_dump_json(indent=2))
    return 0 if trace.status and trace.status.value == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
