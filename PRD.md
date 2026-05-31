# PRD: AI Product Operations Agent — Bug-Triage Reasoning Agent

> Labels: `ready-for-agent`
> Source: synthesized from the grill-me design session (2026-05-31). Full design rationale in `context.md`.

## Problem Statement

Product teams lose hours to the manual coordination work around incoming bug
reports: reading a vague report, deciding its priority and component, checking
whether it's already been filed, deciding whether there's even enough information
to act on, filing it in Jira, and telling the right people in Slack. This is
repetitive, judgment-light-until-it-isn't work that sits between tools and people.

As the *author* of this project, the problem is narrower and sharper: I need a
portfolio piece that proves to AI/ML recruiters that I understand how to build a
reasoning agent — one that plans, acts, observes, knows the boundary of its own
competence, recovers from failure, and whose quality I can *measure*. Most
portfolio agents are undifferentiated "it calls an LLM and an API" demos with no
evaluation; that reads as a toy to the audience I care about.

## Solution

A from-scratch Python reasoning agent that performs **bug-report → Jira triage**
with *visible, measured* intelligence. Given a bug report (via CLI/endpoint first,
a thin Slack adapter later), the agent:

1. Writes an explicit, printable **plan**.
2. Executes it step by step against real tools (Jira, Slack, an embedding-backed
   duplicate index).
3. Emits **structured evidence checks** about the bug (info sufficiency,
   duplicate ambiguity, severity clarity, component clarity).
4. Applies an explicit **confidence-gating policy** — when confidence is low it
   inserts an "ask the human" step rather than guessing.
5. **Reflects** after steps and re-plans when reality diverges (e.g. a duplicate
   already exists).
6. Produces a **structured trace** that drives a live Rich terminal view, a saved
   artifact, and an **eval harness** that scores the agent's decisions against a
   golden set.

The "wow" is the reasoning loop and the eval harness, not integration breadth.

## User Stories

1. As a product ops engineer, I want to hand the agent a raw bug report, so that a well-formed, correctly-triaged Jira ticket is created without my manual effort.
2. As a product ops engineer, I want the agent to write out an explicit plan before acting, so that I can see and trust what it intends to do.
3. As a product ops engineer, I want the agent to detect when an incoming bug duplicates an existing ticket, so that the backlog doesn't fill with redundant issues.
4. As a product ops engineer, I want the agent to recognize a *paraphrased* duplicate (different words, same bug), so that near-duplicates are caught, not just keyword matches.
5. As a product ops engineer, I want the agent to ask me a clarifying question when it lacks the information to triage responsibly, so that it doesn't silently guess a priority.
6. As a product ops engineer, I want the agent to set a sensible priority and component when the report is clear, so that I only get interrupted for genuinely ambiguous cases.
7. As a product ops engineer, I want the agent to post a triage summary to Slack after filing, so that the relevant people are notified without me writing the message.
8. As a product ops engineer, I want the agent to recover when a step fails (e.g. the ticket already exists), so that a single hiccup doesn't crash the whole run.
9. As an eng lead, I want to see the agent's confidence reasoning broken into named evidence checks, so that I can judge *why* it made a call, not just *what* it decided.
10. As an eng lead, I want the agent to never set a priority it isn't confident about, so that triage stays trustworthy.
11. As a PM, I want a readable triage summary generated from the ticket data, so that I get the gist without reading the raw report.
12. As the project author, I want a live terminal view that renders the plan as a checklist with evidence tables and highlighted gate/re-plan events, so that the agent's reasoning is legible in a demo video.
13. As the project author, I want every run to emit a structured trace to disk, so that I have portfolio artifacts and eval inputs from one backbone.
14. As the project author, I want a golden set of labeled bug reports scored automatically, so that I can prove the agent works and catch regressions after prompt changes.
15. As the project author, I want to score whether the agent *correctly decided to ask vs. proceed*, so that I can measure the calibration that is the project's core thesis.
16. As the project author, I want an LLM-as-judge pass on reasoning and summary quality, so that I can measure the soft dimensions exact-match can't.
17. As the project author, I want to compare two model/prompt versions on the same golden set, so that I can demonstrate evaluation-driven iteration.
18. As a recruiter, I want to run the project offline in ~60 seconds with zero accounts, so that I can actually see it work without setting up Jira/Slack/Anthropic keys.
19. As a recruiter with credentials, I want a full live mode (BYO keys + seed script), so that I can verify the integrations are real.
20. As the project author, I want secrets kept out of the repo and stripped from saved traces, so that publishing publicly doesn't leak credentials or ticket data.
21. As the project author, I want the agent to fail fast with a clear message when a required env var is missing, so that first-run setup is frictionless.
22. As the project author, I want the tuning knobs (confidence threshold, cosine bands, project key, model IDs) in one typed config, so that what's tunable is obvious and changeable without code edits.
23. As the project author, I want a seed script that loads ~30–50 fixture tickets with deliberate near-duplicate clusters, so that I can reliably stage both the clear-dupe and ambiguous-dupe demo beats.
24. As the project author, I want a `reindex` command that rebuilds the vector index from Jira, so that the index can be re-synced if it drifts.
25. As the project author, I want the entry point abstracted behind a `BugReport` interface, so that I can iterate on reasoning via CLI first and add a Slack adapter later without touching the loop.
26. As the project author, I want transient API errors retried with backoff below the loop, so that network blips don't surface as reasoning failures.
27. As the project author, I want semantic API errors (409/400/403) surfaced to the agent as tool results, so that reflection can re-plan against real-world divergence.
28. As the project author, I want a replay/dry-run mode that runs over recorded interactions, so that the demo never depends on live network conditions.
29. As the project author, I want structured output enforced via tool-use + Pydantic with one repair-retry, so that malformed model output is handled without a heavyweight library.
30. As the project author, I want tiered models (strong for planning, cheap for judging/routine, dedicated for embeddings), so that the project demonstrates cost-awareness.
31. As a future maintainer, I want analytics and doc-sync described as scoped future work, so that the extension path is clear without bloating the current build.

## Implementation Decisions

**Agent core**
- Hand-rolled agent loop on the raw Anthropic SDK — no agent framework.
- Explicit **plan-then-execute with reflection**: the plan is a first-class data
  structure (ordered steps with intended tools), printable and revisable.
  Reflection runs after steps and may rewrite the remaining plan.
- "Ask the human" is a plan step the agent can *insert*, not a fixed wrapper.

**Confidence model**
- The model emits a structured `EvidenceChecks` object: named checks
  (`info_sufficiency`, `duplicate_ambiguity`, `severity_clarity`,
  `component_clarity`), each a value + one-line justification.
- A **pure gating function in our code** maps evidence → decision
  (proceed vs. ask-human). The judgment is the model's; the gate is deterministic
  code so it is traceable and testable. Cosine similarity from duplicate
  retrieval is a concrete input to `duplicate_ambiguity` (high sim + "same" =
  confident dupe; medium sim = ambiguous → ask).

**Structured output & models**
- Structured outputs via **tool-use** (schemas for `submit_plan`,
  `submit_evidence`, and the action tools) + **Pydantic** validation +
  **one bounded repair-retry** (feed the validation error back once, then fail
  gracefully into the trace).
- Tiered models, IDs in config: Sonnet 4.6 (plan / reason / duplicate
  adjudication), Haiku 4.5 (LLM-as-judge / routine classification), a dedicated
  embedding model (retrieval).

**Tool surface**
- `search_tickets` (semantic + JQL), `create_ticket`, `update_ticket`,
  `ask_human`, `notify_slack`, `generate_triage_summary`.
- `generate_triage_summary` is the one adjacent capability proving the planner
  can compose generation, not just CRUD.

**Integrations & data**
- Entry point behind a `BugReport` interface boundary. CLI/endpoint trigger built
  first; a thin Slack adapter built second constructs the same `BugReport` and
  feeds the same loop.
- Duplicate detection: embed the incoming bug → retrieve top-K existing tickets
  by cosine → hand candidates to the model to adjudicate same/different with
  justification.
- Corpus: a **seed script** loads ~30–50 fixture tickets with deliberate
  near-duplicate clusters. A **persisted local index** (SQLite / in-process
  vector store) is populated **embed-on-ingest** (filing a ticket also indexes
  it). A **`reindex` command** rebuilds vectors from Jira; documented as the
  known consistency boundary.

**Trace & observability**
- A single **`Trace`** object per run is the source of truth: plan, steps,
  tool I/O, evidence checks, gate decisions, reflections. It drives the live Rich
  rendering, is saved to `traces/*.json`, and is the eval harness input.

**Error handling**
- Two tiers. *Below the loop:* a thin client wrapper retries transient errors
  (timeout, 429, 5xx) with exponential backoff and a max-retry cap, degrading
  gracefully into the trace. *Above the loop:* semantic errors (409 duplicate,
  400 validation, 403 permission) are returned to the agent as tool results so
  reflection can re-plan.
- A **replay/dry-run mode** runs the loop over recorded model/API interactions.

**Config & secrets**
- `.env.example` committed; real `.env` gitignored; fail-fast on missing required
  vars with a message naming the missing key.
- Trace redaction strips tokens/auth (and optionally scrubs ticket text); a
  committed sample redacted trace serves as portfolio evidence.
- One typed config (Pydantic Settings / `config.yaml`) exposes the tuning knobs:
  confidence threshold, the two cosine bands (clear-dup / ambiguous), Jira
  project key, model IDs.
- Dual run modes: **full live** (BYO keys + seed + live triage) and
  **replay/offline** (no accounts; runs over committed sample traces).

## Testing Decisions

A good test here asserts on **external behavior via the `Trace`**, not on
internal call sequences or prompt strings. Given the model is non-deterministic,
deterministic tests run against **recorded model/API interactions** (the same
replay mechanism that powers dry-run mode), so assertions are about *what the
agent decided*, not *which tokens it emitted*.

Seams to test (all new — greenfield — proposed at the highest points):

- **`run_triage(BugReport, clients) -> Trace`** — the primary behavioral seam.
  Inject fake tool clients and a recorded model client; feed representative
  `BugReport`s; assert on the resulting `Trace`: plan shape, evidence checks,
  gate decision, final action. This is where most behavior tests live.
- **`gate(evidence) -> Decision`** — pure function; deterministic unit tests pin
  the confidence-gating policy directly. High priority since confidence-gate
  accuracy is the non-negotiable metric.
- **Tool-client interfaces (Jira/Slack/embeddings)** — in-memory fakes. Fakes
  that raise 409/429/403 exercise the two-tier error handling: transient errors
  should be retried below the loop and never reach the trace as failures;
  semantic errors should appear as tool results that trigger reflection/re-plan.
- **Model boundary** — injected Anthropic client; recorded responses give
  deterministic loop tests and validate the repair-retry path (a deliberately
  malformed recorded response should trigger exactly one repair attempt).
- **Eval harness over saved `Trace` JSON** — the golden set (~20 labeled reports
  across clear-dup, ambiguous-dup, novel, insufficient-info, clear-P1,
  ambiguous-priority) is the highest-level behavioral validation. It scores:
  priority correctness, component correctness, duplicate verdict, and
  **confidence-gate accuracy (did it correctly ask vs. proceed)**. An
  LLM-as-judge pass scores reasoning and summary quality. Results render as a
  README table; the harness supports comparing two model/prompt versions.

No prior art exists in this repo (greenfield); these seams *are* the prior art
for future features.

## Out of Scope

- The other two original domains — **NL analytics dashboards** and
  **doc-sync / PRD generation** — are explicitly documented as future work, not
  built. (`generate_triage_summary` is the single adjacent capability that ships.)
- Hosting / multi-tenant deployment. The deliverable is a local repo + README +
  demo video, not a hosted service.
- A web UI. A crisp Rich terminal trace is the chosen surface; a browser view is
  out.
- Real-data corpus import (e.g. scraping a public issue tracker). A controlled
  seed script is used so demo collisions are guaranteed.
- Production-grade vector infrastructure. A local SQLite / in-process index is
  sufficient; running a managed vector DB is out.
- Email/webhook ingestion from real bug sources beyond the CLI/endpoint and the
  thin Slack adapter.

## Further Notes

- Two staged demo beats anchor the video: (1) **failure recovery** — plan → step
  hits a duplicate (409 / clear cosine match) → reflection re-plans instead of
  crashing; (2) **calibrated intelligence** — vague bug → evidence flags low info
  / ambiguous dup → agent inserts a clarifying question → resumes.
- The structured `Trace` is deliberately the shared backbone for three otherwise-
  separate efforts (visible reasoning, eval, saved artifacts) — build it early.
- README should carry: the architecture sketch, the eval results table, the
  "error type → handler" two-tier table, the why-each-model note, and the two
  run-mode instructions.
- Full decision rationale (16 resolved decisions with the alternatives weighed)
  lives in `context.md`.
