"""Live terminal rendering of a Trace with Rich.

The renderer is a pure function of the Trace: every time the Trace mutates it
re-renders the whole view from scratch. There is no separate rendering state —
the Trace remains the single source of truth (the same object that's saved to
JSON and scored by the eval harness). This file only knows how to *display* it.

Usage::

    with live_render(trace):
        run_triage(...)   # the Trace's observer drives updates

Rendering goes to stderr so stdout stays clean JSON (for piping / replay / eval).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .models import EventType, RunStatus, StepStatus, Trace

_STATUS_ICON = {
    StepStatus.PENDING: ("[ ]", "grey50"),
    StepStatus.IN_PROGRESS: ("[~]", "yellow"),
    StepStatus.DONE: ("[x]", "green"),
    StepStatus.FAILED: ("[!]", "red"),
    StepStatus.SKIPPED: ("[-]", "grey50"),
}

_LEVEL_STYLE = {"high": "green", "medium": "yellow", "low": "red"}


def render_trace(trace: Trace) -> Group:
    """Build the full renderable view of the current Trace state."""
    parts = [_header(trace)]

    evidence = _latest(trace, EventType.EVIDENCE_SUBMITTED)
    if evidence:
        parts.append(_evidence_table(evidence.data))

    gate = _latest(trace, EventType.GATE_DECISION)
    if gate:
        parts.append(_gate_panel(gate.data))

    asked = _latest(trace, EventType.HUMAN_ASKED)
    answered = _latest(trace, EventType.HUMAN_ANSWERED)
    if asked:
        parts.append(_clarify_panel(asked.message, answered.message if answered else None))

    if trace.plan:
        parts.append(_plan_panel(trace))

    for ev in trace.events:
        if ev.type in (EventType.REFLECTION, EventType.PLAN_REVISED, EventType.STEP_RETRY):
            parts.append(_event_line(ev))

    if trace.status is not None:
        parts.append(_outcome(trace))

    return Group(*parts)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


def _header(trace: Trace) -> Panel:
    dup = _latest(trace, EventType.DUPLICATE_CHECK)
    dup_line = ""
    if dup:
        cls = dup.data.get("classification", "?")
        cands = dup.data.get("candidates", [])
        top = f" - closest {cands[0]['key']} @ {cands[0]['score']}" if cands else ""
        color = {"clear": "red", "ambiguous": "yellow", "none": "green"}.get(cls, "white")
        dup_line = Text.assemble(
            "\nDuplicate check: ", (cls, f"bold {color}"), top
        )
    body = Text(trace.bug_report.raw_text.strip())
    if dup_line:
        body = Text.assemble(body, dup_line)
    return Panel(body, title="Bug report", border_style="cyan", expand=True)


def _evidence_table(data: dict) -> Table:
    table = Table(title="Evidence", title_style="bold", expand=True)
    table.add_column("Dimension")
    table.add_column("Level")
    table.add_column("Why", overflow="fold")
    for dim in ("info_sufficiency", "severity_clarity", "component_clarity", "duplicate_ambiguity"):
        check = data.get(dim) or {}
        level = check.get("level", "?")
        table.add_row(
            dim,
            Text(level, style=_LEVEL_STYLE.get(level, "white")),
            check.get("justification", ""),
        )
    return table


def _gate_panel(data: dict) -> Panel:
    action = data.get("action", "?")
    if action == "ask_human":
        triggered = ", ".join(data.get("triggered", []))
        return Panel(
            Text(f"ASK HUMAN - low confidence on: {triggered}", style="bold yellow"),
            border_style="yellow",
            title="Confidence gate",
        )
    return Panel(
        Text("PROCEED - confident enough to triage", style="bold green"),
        border_style="green",
        title="Confidence gate",
    )


def _clarify_panel(question: str, answer: str | None) -> Panel:
    body = Text.assemble(("Q: ", "bold"), question)
    if answer:
        body = Text.assemble(body, "\n", ("A: ", "bold green"), answer)
    return Panel(body, border_style="magenta", title="Clarification")


def _plan_panel(trace: Trace) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column(width=3)
    table.add_column()
    for step in trace.plan.steps:
        icon, color = _STATUS_ICON.get(step.status, ("?", "white"))
        label = f"{step.tool} - {step.intent}"
        if step.result and step.result.get("key"):
            label += f"  ->  {step.result['key']}"
        table.add_row(Text(icon, style=color), Text(label, style=color))
    return Panel(table, title="Plan", border_style="blue")


def _event_line(ev) -> Text:
    color = {
        EventType.REFLECTION: "yellow",
        EventType.PLAN_REVISED: "cyan",
        EventType.STEP_RETRY: "orange3",
    }.get(ev.type, "white")
    icon = {
        EventType.REFLECTION: "[reflect]",
        EventType.PLAN_REVISED: "[revised]",
        EventType.STEP_RETRY: "[retry]",
    }.get(ev.type, "-")
    return Text(f"  {icon} {ev.message}", style=color)


def _outcome(trace: Trace) -> Panel:
    if trace.status is RunStatus.COMPLETED:
        o = trace.outcome
        if o.get("filed") is False and o.get("duplicate_of"):
            msg = Text(f"Done - skipped as duplicate of {o['duplicate_of']}", style="bold green")
        else:
            ticket = o.get("ticket") or {}
            key = ticket.get("key", "-")
            url = ticket.get("url", "")
            msg = Text.assemble(("Done - filed ", "bold green"), (key, "bold green"))
            if url:
                msg = Text.assemble(msg, "\n", (url, "blue underline"))
        return Panel(msg, border_style="green", title="Outcome")
    return Panel(
        Text(f"Failed - {trace.outcome.get('reason') or trace.outcome.get('error', '')}", style="bold red"),
        border_style="red",
        title="Outcome",
    )


# --------------------------------------------------------------------------- #
# Helpers + live driver
# --------------------------------------------------------------------------- #


def _latest(trace: Trace, event_type: EventType):
    for ev in reversed(trace.events):
        if ev.type is event_type:
            return ev
    return None


class LiveRenderer:
    """Drives a Rich Live view from whatever Trace the observer is handed.

    The renderer owns the terminal while active, so when the agent needs to ask
    the human a question it must pause the Live display, read input normally,
    then resume — otherwise the prompt and the live view fight over the screen.
    ``ask`` exposes that pause/resume so a HumanClient can defer to it.

    It is also itself a HumanClient (has ``ask``), so the CLI can pass it
    directly as both observer and human.
    """

    def __init__(self) -> None:
        self._console = Console(stderr=True)
        self._live = Live(console=self._console, refresh_per_second=12)

    # -- observer: called after every Trace mutation -- #
    def __call__(self, trace: Trace) -> None:
        self._live.update(render_trace(trace))

    # -- HumanClient -- #
    def ask(self, question: str) -> str:
        self._live.stop()  # release the terminal for the prompt
        self._console.print(f"\n[bold magenta]?[/] {question}")
        try:
            answer = self._console.input("[bold]> [/]").strip()
        except EOFError:
            answer = ""
        self._live.start(refresh=True)
        return answer


@contextmanager
def live_render():
    """Yield a LiveRenderer usable as both the run's observer and human client."""
    renderer = LiveRenderer()
    renderer._live.start(refresh=True)
    try:
        yield renderer
    finally:
        renderer._live.stop()
