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

from pydantic import ValidationError

from .clients import JiraClient, ModelClient
from .models import (
    BugReport,
    EventType,
    Plan,
    PlanStep,
    RunStatus,
    StepStatus,
    Trace,
)

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
) -> Trace:
    """Triage one bug report end-to-end, returning the full Trace.

    This is the seam tests drive: inject a fake/recorded ``model`` and fake tool
    clients, feed a ``BugReport``, and assert on the returned ``Trace``.
    """
    trace = Trace(bug_report=bug_report)

    plan = _propose_plan(bug_report, model=model, trace=trace)
    if plan is None:
        trace.finish(RunStatus.FAILED, reason="could not produce a valid plan")
        trace.record(EventType.RUN_FAILED, "No valid plan after repair-retry.")
        return trace

    trace.plan = plan
    _execute_plan(plan, jira=jira, trace=trace)
    return trace


# --------------------------------------------------------------------------- #
# Planning (with one bounded repair-retry)
# --------------------------------------------------------------------------- #


def _propose_plan(
    bug_report: BugReport,
    *,
    model: ModelClient,
    trace: Trace,
) -> Plan | None:
    tools = [_submit_plan_tool()]
    messages: list[dict] = [
        {"role": "user", "content": f"Bug report:\n\n{bug_report.raw_text}"}
    ]

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


def _execute_plan(plan: Plan, *, jira: JiraClient, trace: Trace) -> None:
    for step in plan.steps:
        step.status = StepStatus.IN_PROGRESS
        trace.record(
            EventType.STEP_STARTED, step.intent, step_id=step.id, tool=step.tool
        )
        try:
            result = _run_step(step, jira=jira)
        except Exception as err:  # noqa: BLE001 - recorded, run continues to fail
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

    last = plan.steps[-1].result if plan.steps else {}
    trace.finish(RunStatus.COMPLETED, ticket=last)
    trace.record(EventType.RUN_COMPLETED, "Triage complete.", ticket=last)


def _run_step(step: PlanStep, *, jira: JiraClient) -> dict:
    if step.tool == "create_ticket":
        return jira.create_ticket(
            summary=step.args.get("summary", ""),
            description=step.args.get("description", ""),
            priority=step.args.get("priority", "P3"),
            component=step.args.get("component"),
        )
    raise UnknownToolError(f"Unknown tool: {step.tool}")
