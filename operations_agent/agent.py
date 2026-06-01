"""The plan-then-execute agent loop.

Slice 1 establishes the primary behavioral seam — ``run_triage`` — and the
shape of the loop:

    1. Ask the model for an explicit Plan (via the ``submit_plan`` tool).
       Validate it against Pydantic; on failure, issue exactly one repair-retry
       feeding the validation error back, then give up gracefully.
    2. Execute each step against the injected tool clients.
    3. Record everything in a Trace and return it.

Reflection/re-plan, evidence checks, and the confidence gate are deliberately
NOT here yet — they slot into this same loop in later slices.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

import time
from typing import Callable

from .clients import HumanClient, JiraClient, ModelClient
from .duplicates import find_duplicate
from .errors import SemanticError, TransientError
from .evidence import collect_evidence
from .gate import gate
from .index import IndexedTicket, TicketIndex
from .models import (
    BugReport,
    DuplicateClassification,
    DuplicateVerdict,
    EventType,
    GateAction,
    Plan,
    PlanStep,
    RunStatus,
    StepStatus,
    Trace,
)
from .retry import call_with_retry

MAX_REFLECTIONS = 2


@dataclass
class RetryConfig:
    """Tuning for the transient-retry tier. ``sleep`` is injectable for tests."""

    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    sleep: Callable[[float], None] = time.sleep


@dataclass
class DuplicateChecker:
    """Bundles the index + tuning bands for the duplicate-detection step.

    Optional: pass ``None`` to ``run_triage`` to skip duplicate detection (e.g.
    the Slice 1 walking-skeleton path and unit tests that don't care about it).
    """

    index: TicketIndex
    clear_band: float
    ambiguous_band: float
    top_k: int = 5

MAX_PLAN_REPAIRS = 1

SYSTEM_PROMPT = """You are a product-operations triage agent. Given a raw bug \
report, produce an explicit plan to triage it and file a ticket.

Respond by calling the `submit_plan` tool exactly once. The plan must contain an \
ordered list of steps. For this slice the only tool a step may use is \
`create_ticket`, whose args are: summary (str), description (str), \
priority (one of P0/P1/P2/P3), and component (str, optional).

Keep the plan minimal: usually a single `create_ticket` step. Choose a sensible \
priority and component from the report."""


def _submit_plan_tool() -> dict:
    """Tool schema the model fills to deliver a Plan.

    Derived from the Pydantic model so the schema and the validator never drift.
    """
    schema = Plan.model_json_schema()
    return {
        "name": "submit_plan",
        "description": "Submit the ordered triage plan.",
        "input_schema": schema,
    }


class UnknownToolError(Exception):
    """Raised when a plan step names a tool the executor doesn't know."""


def run_triage(
    bug_report: BugReport,
    *,
    model: ModelClient,
    jira: JiraClient,
    duplicates: DuplicateChecker | None = None,
    human: HumanClient | None = None,
    retry_config: RetryConfig | None = None,
) -> Trace:
    """Triage one bug report end-to-end, returning the full Trace.

    This is the seam tests drive: inject a fake/recorded ``model`` and fake tool
    clients, feed a ``BugReport``, and assert on the returned ``Trace``.

    Flow: duplicate check (Slice 3) -> evidence assessment + confidence gate
    (Slice 4) -> optional clarifying question to a human -> plan -> execute. A
    clear duplicate short-circuits before any filing; newly filed tickets are
    embedded into the index (embed-on-ingest).
    """
    trace = Trace(bug_report=bug_report)

    verdict: DuplicateVerdict | None = None
    if duplicates is not None:
        verdict = _check_duplicate(bug_report, model=model, dup=duplicates)
        trace.record(
            EventType.DUPLICATE_CHECK,
            f"Duplicate check: {verdict.classification.value}.",
            **verdict.model_dump(),
        )
        if verdict.classification is DuplicateClassification.CLEAR:
            trace.finish(
                RunStatus.COMPLETED,
                duplicate_of=verdict.matched_key,
                filed=False,
            )
            trace.record(
                EventType.RUN_COMPLETED,
                f"Skipped filing — clear duplicate of {verdict.matched_key}.",
                duplicate_of=verdict.matched_key,
            )
            return trace

    # Evidence assessment + the (pure, code-side) confidence gate.
    clarification: str | None = None
    evidence = collect_evidence(
        bug_report.raw_text, model=model, duplicate=verdict
    )
    trace.record(
        EventType.EVIDENCE_SUBMITTED,
        "Evidence assessed.",
        **evidence.model_dump(),
    )
    decision = gate(evidence)
    trace.record(
        EventType.GATE_DECISION,
        f"Gate: {decision.action.value}.",
        **decision.model_dump(),
    )

    if decision.action is GateAction.ASK_HUMAN:
        if human is None:
            # No way to ask — fail rather than guess what we said we wouldn't.
            trace.finish(
                RunStatus.FAILED,
                reason="clarification needed but no human available",
                triggered=decision.triggered,
            )
            trace.record(
                EventType.RUN_FAILED,
                "Needed clarification but no human client was provided.",
            )
            return trace
        trace.record(
            EventType.HUMAN_ASKED, decision.question or "", triggered=decision.triggered
        )
        clarification = human.ask(decision.question or "")
        trace.record(EventType.HUMAN_ANSWERED, clarification)

    plan = _propose_plan(
        bug_report, model=model, trace=trace, clarification=clarification
    )
    if plan is None:
        trace.finish(RunStatus.FAILED, reason="could not produce a valid plan")
        trace.record(EventType.RUN_FAILED, "No valid plan after repair-retry.")
        return trace

    trace.plan = plan
    _execute_plan(
        plan,
        model=model,
        jira=jira,
        trace=trace,
        duplicates=duplicates,
        retry_config=retry_config,
    )
    return trace


def _check_duplicate(
    bug_report: BugReport, *, model: ModelClient, dup: DuplicateChecker
) -> DuplicateVerdict:
    return find_duplicate(
        bug_report.raw_text,
        index=dup.index,
        model=model,
        clear_band=dup.clear_band,
        ambiguous_band=dup.ambiguous_band,
        top_k=dup.top_k,
    )


# --------------------------------------------------------------------------- #
# Planning (with one bounded repair-retry)
# --------------------------------------------------------------------------- #


def _propose_plan(
    bug_report: BugReport,
    *,
    model: ModelClient,
    trace: Trace,
    clarification: str | None = None,
) -> Plan | None:
    tools = [_submit_plan_tool()]
    content = f"Bug report:\n\n{bug_report.raw_text}"
    if clarification:
        # Fold the human's answer into the planning context so the plan
        # reflects it (this is why we asked).
        content += f"\n\nClarification from the reporter:\n{clarification}"
    messages: list[dict] = [{"role": "user", "content": content}]

    for attempt in range(MAX_PLAN_REPAIRS + 1):
        response = model.call(
            system=SYSTEM_PROMPT, messages=messages, tools=tools
        )
        try:
            plan = Plan.model_validate(response.tool_input)
        except ValidationError as err:
            if attempt >= MAX_PLAN_REPAIRS:
                trace.record(
                    EventType.PLAN_REPAIR,
                    "Plan invalid after final attempt; giving up.",
                    errors=err.errors(include_url=False),
                )
                return None
            trace.record(
                EventType.PLAN_REPAIR,
                "Plan failed validation; issuing one repair-retry.",
                errors=err.errors(include_url=False),
            )
            # Feed the model its own bad output plus the error, ask for a fix.
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": response.tool_name,
                            "input": response.tool_input,
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That plan failed schema validation with these errors:\n"
                        f"{err.errors(include_url=False)}\n"
                        "Call submit_plan again with a corrected plan."
                    ),
                }
            )
            continue

        trace.record(
            EventType.PLAN_PROPOSED,
            f"Plan with {len(plan.steps)} step(s).",
            rationale=plan.rationale,
        )
        return plan

    return None


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


def _execute_plan(
    plan: Plan,
    *,
    model: ModelClient,
    jira: JiraClient,
    trace: Trace,
    duplicates: DuplicateChecker | None = None,
    retry_config: RetryConfig | None = None,
) -> None:
    """Execute the plan step by step, with two-tier error handling.

    - Transient failures (timeouts, 429, 5xx) are retried with backoff *inside*
      the step; the agent never sees them.
    - Semantic failures (409/400/403) are surfaced to the model, which reflects
      and rewrites the remaining steps. Capped at ``MAX_REFLECTIONS`` rounds.
    """
    retry_config = retry_config or RetryConfig()
    reflections = 0
    i = 0
    while i < len(plan.steps):
        step = plan.steps[i]
        step.status = StepStatus.IN_PROGRESS
        trace.record(
            EventType.STEP_STARTED, step.intent, step_id=step.id, tool=step.tool
        )

        try:
            result = _run_step_with_retry(
                step, jira=jira, trace=trace, retry_config=retry_config
            )
        except SemanticError as err:
            # Reality diverged in a way the agent should reason about.
            step.status = StepStatus.FAILED
            step.result = {"error": str(err), "status": err.status}
            trace.record(
                EventType.STEP_FAILED,
                f"Semantic failure ({err.status}): {err.message}",
                step_id=step.id,
                tool=step.tool,
                status=err.status,
            )
            if reflections >= MAX_REFLECTIONS:
                trace.finish(
                    RunStatus.FAILED, failed_step=step.id, error=str(err)
                )
                trace.record(
                    EventType.RUN_FAILED,
                    "Reflection limit reached; aborting.",
                )
                return
            reflections += 1
            revised = _reflect_and_revise(
                plan, failed_index=i, error=err, model=model, trace=trace
            )
            if revised is None:
                trace.finish(
                    RunStatus.FAILED, failed_step=step.id, error=str(err)
                )
                trace.record(
                    EventType.RUN_FAILED, "Reflection produced no usable plan."
                )
                return
            # Replace remaining steps with the revised ones and continue.
            plan.steps = plan.steps[: i + 1] + revised
            i += 1
            continue
        except Exception as err:  # noqa: BLE001 - unknown failure, abort cleanly
            step.status = StepStatus.FAILED
            step.result = {"error": str(err)}
            trace.record(
                EventType.STEP_FAILED,
                f"Step failed: {err}",
                step_id=step.id,
                tool=step.tool,
            )
            trace.finish(RunStatus.FAILED, failed_step=step.id, error=str(err))
            trace.record(EventType.RUN_FAILED, "Run aborted on step failure.")
            return

        step.status = StepStatus.DONE
        step.result = result
        trace.record(
            EventType.STEP_COMPLETED,
            f"{step.tool} → {result}",
            step_id=step.id,
            tool=step.tool,
            result=result,
        )

        # Embed-on-ingest: a newly filed ticket joins the index so future bugs
        # can be matched against it.
        if duplicates is not None and step.tool == "create_ticket" and result.get("key"):
            duplicates.index.add(
                IndexedTicket(
                    key=result["key"],
                    summary=result.get("summary", ""),
                    text=result.get("description", ""),
                )
            )
        i += 1

    done = [s for s in plan.steps if s.status is StepStatus.DONE]
    last = done[-1].result if done else {}
    trace.finish(RunStatus.COMPLETED, ticket=last)
    trace.record(EventType.RUN_COMPLETED, "Triage complete.", ticket=last)


def _run_step_with_retry(
    step: PlanStep,
    *,
    jira: JiraClient,
    trace: Trace,
    retry_config: RetryConfig,
) -> dict:
    """Run one step, retrying transient failures below the loop."""

    def on_retry(attempt: int, err: TransientError, delay: float) -> None:
        trace.record(
            EventType.STEP_RETRY,
            f"Transient failure ({err.status}); retry {attempt} after {delay}s.",
            step_id=step.id,
            attempt=attempt,
            status=err.status,
        )

    return call_with_retry(
        lambda: _run_step(step, jira=jira),
        max_retries=retry_config.max_retries,
        base_delay=retry_config.base_delay,
        max_delay=retry_config.max_delay,
        sleep=retry_config.sleep,
        on_retry=on_retry,
    )


def _reflect_and_revise(
    plan: Plan,
    *,
    failed_index: int,
    error: SemanticError,
    model: ModelClient,
    trace: Trace,
) -> list[PlanStep] | None:
    """Ask the model to rewrite the remaining steps given the failure.

    The failed step's outcome is fed back as context. The model returns a new
    list of steps to run *after* the failed one (which may be empty if the right
    move is to stop — e.g. the ticket already exists).
    """
    trace.record(
        EventType.REFLECTION,
        f"Reflecting on {error.status} failure at step {failed_index}.",
        status=error.status,
    )
    remaining = plan.steps[failed_index + 1 :]
    revise_system = (
        "A step in your plan failed with a meaningful error. Decide how to "
        "proceed: submit a revised list of the REMAINING steps to run. If the "
        "error means no further action is needed (e.g. the ticket already "
        "exists), submit an empty steps list. Only use the create_ticket tool."
    )
    context = (
        f"Failed step: {plan.steps[failed_index].tool} "
        f"({plan.steps[failed_index].intent})\n"
        f"Error {error.status}: {error.message}\n"
        f"Originally-remaining steps: {[s.tool for s in remaining]}"
    )
    response = model.call(
        system=revise_system,
        messages=[{"role": "user", "content": context}],
        tools=[_submit_plan_tool()],
    )
    try:
        revised_plan = Plan.model_validate(response.tool_input)
    except ValidationError:
        return None

    trace.record(
        EventType.PLAN_REVISED,
        f"Revised plan: {len(revised_plan.steps)} remaining step(s).",
        rationale=revised_plan.rationale,
    )
    return revised_plan.steps


def _run_step(step: PlanStep, *, jira: JiraClient) -> dict:
    if step.tool == "create_ticket":
        return jira.create_ticket(
            summary=step.args.get("summary", ""),
            description=step.args.get("description", ""),
            priority=step.args.get("priority", "P3"),
            component=step.args.get("component"),
        )
    raise UnknownToolError(f"Unknown tool: {step.tool}")
