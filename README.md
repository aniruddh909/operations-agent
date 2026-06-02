# AI Product Operations Agent

A reasoning agent that triages incoming bug reports: it checks for duplicates,
assesses its own confidence, asks a human when it is unsure rather than guessing,
files a properly-prioritized Jira ticket, and notifies the team on Slack.

The agent loop is built from scratch on the Anthropic API (no agent framework),
and every run's decisions are measured by an evaluation harness.

## What it does

Given a raw bug report, the agent runs this loop:

1. **Duplicate check** - embeds the report, retrieves the most similar existing
   tickets (cosine similarity over a local index), and has the model adjudicate
   whether it is the same bug. A clear duplicate short-circuits (no second
   ticket); an ambiguous one feeds the confidence gate.
2. **Evidence assessment** - the model rates four dimensions with one-line
   justifications: info sufficiency, severity clarity, component clarity, and
   duplicate ambiguity (the last derived in code from the cosine result).
3. **Confidence gate** - deterministic code (not the model) decides: if any
   dimension is low, ask the human one targeted question; otherwise proceed.
4. **Plan and execute** - the model writes an explicit plan (file ticket ->
   summarize -> notify) and the agent runs it, with two-tier error handling
   (transient errors retried below the loop; semantic errors like a 409 surfaced
   to the model, which reflects and re-plans).

Everything a run does is recorded in one `Trace` object, which drives the live
terminal view, is saved as an artifact, and is scored by the eval harness.

## Evaluation

The agent's decisions are scored against a labeled golden set covering six
scenarios (clear/ambiguous duplicate, novel, insufficient-info, clear-P1,
ambiguous-priority). The headline metric is **confidence-gate accuracy**: did the
agent correctly decide to ask a human vs. proceed?

Results below are from a live run on one representative case per dimension
(6-case subset) against Claude Sonnet + local embeddings. Run the full set with
`ops-eval`.

### Eval results (subset-6, live)

| Metric | Accuracy | Correct / Total |
| --- | --- | --- |
| Confidence-gate accuracy (non-negotiable) | 100% | 5/5 |
| Duplicate verdict | 100% | 4/4 |
| File / skip decision | 100% | 3/3 |
| Priority correctness | 100% | 1/1 |
| Component correctness | 100% | 2/2 |
| Overall | 100% | 15/15 |

The harness can also compare two model/prompt versions on the same set to catch
regressions (`compare()` in `operations_agent/evaluation/harness.py`).

## Architecture

- **From-scratch agent loop** on the raw Anthropic SDK - plan-then-execute with a
  reflection step; structured output via tool-use + Pydantic validation + one
  bounded repair-retry.
- **The `Trace`** is the single source of truth - it drives live rendering, saved
  artifacts, and the eval harness from one backbone.
- **Entry-point boundary** - triggers (CLI, Slack) construct the same `BugReport`
  that feeds the same loop. Adding Slack was a thin adapter, not a loop change.

### Why each model

| Role | Model | Why |
| --- | --- | --- |
| Planning, reasoning, duplicate adjudication | Strong (Claude Sonnet) | The hard judgment calls - worth the better model |
| LLM-as-judge (eval) | Cheap (Claude Haiku) | Grading is a simpler, high-volume task; keep eval runs cheap |
| Embeddings (duplicate retrieval) | Local sentence-transformers | Free, offline, and recruiter-reproducible; no per-call cost |

Model IDs are config values, so the eval harness can compare two versions on the
same golden set.

### Two-tier error handling

| Error | Tier | Handling |
| --- | --- | --- |
| Timeout, 429, 5xx | Transient | Retried with exponential backoff below the loop; never a reasoning failure |
| 409, 400, 403, 422 | Semantic | Surfaced to the model as a tool result; reflection rewrites the remaining plan |

## Tech stack

Python, Pydantic, the Anthropic API (tool-use), sentence-transformers (local
embeddings), SQLite (vector index), httpx (Jira + Slack), Rich (terminal view),
pytest.

## Running it

There are two run modes.

**Full live** (bring your own accounts - real triage end to end):

```bash
pip install -e ".[embeddings]"      # local embeddings for duplicate detection
cp .env.example .env                # fill in API keys (see below)
ops-seed                            # seed the fixture backlog into Jira + index (once)
triage "your bug report text"       # triage a bug, live
triage "..." --live                 # with the live reasoning view
triage "..." --save-trace out.json  # save a redacted Trace artifact
ops-eval                            # run the golden-set evaluation
```

Required config (`.env`): `ANTHROPIC_API_KEY`, plus `JIRA_BASE_URL`,
`JIRA_EMAIL`, `JIRA_API_TOKEN`, `JIRA_PROJECT_KEY` for live filing. Optional:
`SLACK_WEBHOOK_URL`. Missing required vars fail fast with a clear message.

**Replay / offline** (no external accounts, ~60s): record a run once, then replay
it deterministically against fake clients - no model API calls, nothing to set
up. Useful for demos and for seeing the agent work without keys.

```bash
triage "..." --record run.cassette  # record model responses during a live run
triage "..." --replay run.cassette --offline   # replay later with fakes, no network
```

A committed sample redacted Trace lives at `traces/sample-redacted.json`.

A bug can also enter via the **Slack adapter** (`operations_agent/slack_adapter.py`),
which turns a slash-command or Events-API payload into the same `BugReport`.

Tuning knobs (in config): confidence behavior, the two cosine bands
(clear-dup / ambiguous), and the tiered model IDs.

## Tests

```bash
pytest        # 56 tests, no network (fake model + fake clients)
```

Tests assert on the `Trace` (the agent's external behavior), not on internal call
order, and run deterministically with an injected scripted model.

## Future work

- Natural-language analytics ("what's our churn this week?") over the same agent
  loop.
- Document workflows (auto-generated PRDs / status docs kept in sync with Jira).
- A Slack entry point so a `/bug` message triggers triage directly.
