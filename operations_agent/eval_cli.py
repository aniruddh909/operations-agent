"""``ops-eval`` — run the golden-set evaluation and print a results table.

Builds the tiered clients from config (strong planning model, local embeddings),
runs the agent over the golden set, scores the Traces, and prints a Markdown
table suitable for pasting into the README. ``--out`` writes the table to a file.
"""

from __future__ import annotations

import argparse
import sys

from .config import MissingConfigError, Settings


def eval_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ops-eval", description="Evaluate the triage agent on the golden set."
    )
    parser.add_argument("--label", default="current", help="Label for this run.")
    parser.add_argument("--out", default=None, help="Write the Markdown table here.")
    parser.add_argument(
        "--model", default=None, help="Override the planning model ID."
    )
    args = parser.parse_args(argv)

    settings = Settings()
    if not settings.anthropic_api_key:
        print(
            "error: ANTHROPIC_API_KEY is required to run the eval.",
            file=sys.stderr,
        )
        return 2

    from .anthropic_client import AnthropicModelClient
    from .embeddings import LocalEmbeddingClient
    from .evaluation.harness import run_eval
    from .evaluation.report import to_markdown

    model = AnthropicModelClient(
        model=args.model or settings.planning_model,
        api_key=settings.anthropic_api_key,
    )
    embedder = LocalEmbeddingClient(settings.embedding_model)

    report = run_eval(
        model=model,
        embedder=embedder,
        clear_band=settings.dup_clear_band,
        ambiguous_band=settings.dup_ambiguous_band,
        label=args.label,
    )

    table = to_markdown(report)
    print(table)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(table + "\n")
        print(f"\nWrote table to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(eval_main())
