# AI Product Operations Agent — Design & PRD

> Single source of truth for this project: the locked design (from a grill-me
> session, 2026-05-31) followed by the PRD synthesized from it. Greenfield — no
> code yet.

---

# Part 1 — Locked Design

## Thesis

A from-scratch reasoning agent that does **bug-report → Jira triage** with
*visible, measured* intelligence. Portfolio piece for **AI/ML recruiters**.
The wow is the **reasoning loop + the eval harness** — NOT integration breadth.

## Deliverable

- Local **Python** repo, real APIs, strong README + demo video. Not hosted.
- Audience: recruiters at tech companies, specifically AI/ML roles.

## The reasoning core

- **Hand-rolled agent loop** on the raw Anthropic SDK — no framework.
- **Explicit plan-then-execute with reflection** — the plan is a first-class,
  printable artifact; reflection revises the remaining steps on divergence.
- **Confidence-gated clarifying questions** — "ask human" is an *insertable plan
  step*, not a fixed prompt.
- **Confidence = structured evidence checks** emitted by the model
  (info sufficiency, duplicate ambiguity, severity clarity, component clarity),
  each with a one-line justification. The **explicit gating policy lives in your
  own code**, not in the model's raw score.
- **Structured output via tool-use + Pydantic + one bounded repair-retry.** On a
  validation miss, feed the error back once, then fail gracefully into the trace.
- **Tiered models** (IDs in config, not hardcoded):
  - Sonnet 4.6 — planning / reasoning / duplicate adjudication
  - Haiku 4.5 — LLM-as-judge / routine classification
  - Dedicated embedding model — retrieval

## Domain & tools

- **Hero vertical:** bug-triage.
- **Tool surface:** `search_tickets` (semantic + JQL), `create_ticket` /
  `update_ticket`, `ask_human`, `notify_slack`, `generate_triage_summary` (the
  one adjacent capability proving the planner can compose generation, not just
  CRUD).
- **Analytics + doc-sync:** documented as **future work**, not built.

## Integrations & data

- **Entry point:** a single `BugReport` interface boundary. CLI/endpoint trigger
  first; thin **Slack adapter as the demo skin**, built second, constructing the
  same `BugReport` object and feeding the same loop.
- **Duplicate detection:** embeddings retrieval (top-K by cosine) → LLM
  adjudication of the candidates with justification. Cosine score feeds the
  `duplicate_ambiguity` evidence check (high sim + "same" = confident dupe;
  medium sim = ambiguous → ask human).
- **Corpus / seeding:**
  - **Seed script**: ~30–50 realistic fixture tickets with **deliberate
    near-duplicate clusters** (e.g. three differently-worded "login crash"
    reports) so both a clear-dupe and an ambiguous-dupe can be demoed on command.
  - **Persisted local index** (SQLite / in-process vector store),
    **embed-on-ingest** (filing a ticket also indexes it).
  - **`reindex` command** rebuilds vectors from Jira to cover drift; documented
    as a known consistency boundary.

## Trust & legibility

- **Structured trace = single backbone.** One trace object per run (plan, steps,
  tool I/O, evidence checks, gate decisions, reflections) drives: live **Rich**
  terminal rendering (plan as live checklist ✓/✗/⏳, evidence as a table,
  confidence gate + re-plan as highlighted events), the **eval harness** input,
  and saved `traces/*.json` portfolio artifacts.
- **Eval harness:**
  - Golden set of ~20 labeled bug reports across evidence dimensions
    (clear-dup, ambiguous-dup, novel, insufficient-info, clear-P1,
    ambiguous-priority).
  - **Confidence-gate accuracy is NON-NEGOTIABLE** — score "did it correctly
    decide to ask vs. proceed?" alongside priority/component/dup verdicts.
  - Building toward: + LLM-as-judge for reasoning quality + regression tracking +
    a results table in the README + cross-model/version comparison.
- **Two-tier error handling:**
  - *Below the loop (infra):* thin client wrapper with retry + exponential
    backoff for transient errors (timeout, 429, 5xx), max-retry cap, graceful
    degradation into the trace.
  - *Above the loop (semantic):* meaningful failures (409 duplicate, 400
    validation, 403 permission) are surfaced to the agent as tool *results* so
    reflection can re-plan. Document as a one-line "error type → handler" table.
  - **Replay / dry-run mode** so the demo never depends on live network weather.
- **Reproducibility contract:**
  1. `.env.example` committed, real `.env` gitignored, fail-fast on missing vars;
     **trace redaction** strips tokens/auth (and optionally scrubs ticket text);
     keep a committed sample redacted trace as evidence.
  2. Single typed config (Pydantic Settings / `config.yaml`) exposing the tuning
     knobs: confidence threshold, the two cosine bands (clear-dup / ambiguous),
     Jira project key, model IDs — documented inline.
  3. Dual run modes: **(a) full live** (BYO keys + seed script + live triage) and
     **(b) replay/offline** (no accounts needed; runs over committed sample
     traces so a recruiter sees it work in ~60s with zero setup).

## Two staged demo beats

1. **Failure recovery:** plan → step hits a duplicate (409 / clear cosine match)
   → reflection re-plans instead of crashing.
2. **Calibrated intelligence:** vague bug → evidence check flags low info /
   ambiguous dup → agent inserts a clarifying question → resumes triage.

## Decision index

| # | Decision | Choice |
|---|----------|--------|
| Q1 | Deliverable | Local repo + real APIs + README/video, not hosted |
| Q2 | Hero capability / vertical | Reasoning agent; bug-triage |
| Q3 | Loop | From scratch on Anthropic SDK (no framework) |
| Q4 | Control architecture | Explicit plan-then-execute + reflection |
| Q5 | Ambiguity stance | Confidence-gated clarifying questions |
| Q6 | Confidence source | Structured evidence checks + code-side gating policy |
| Q7 | Entry point | `BugReport` interface; CLI first, Slack adapter as skin |
| Q8 | Duplicate detection | Embeddings retrieval + LLM adjudication |
| Q9 | Corpus | Seed script + persisted index + reindex command |
| Q10 | Language | Python (AI/ML roles) |
| Q11 | Eval | Golden set; confidence-gate accuracy non-negotiable; → (D) |
| Q12 | Tool surface / scope | Triage tools + `generate_triage_summary`; rest = future work |
| Q13 | Reasoning visibility | Structured trace backbone → Rich + eval + artifacts |
| Q14 | Structured output / models | Tool-use + Pydantic + 1 repair-retry; Sonnet/Haiku/embeddings |
| Q15 | Error handling | Two-tier (infra retry / semantic re-plan) + replay mode |
| Q16 | Secrets/config/repro | `.env` hygiene + redaction; typed config; live/offline modes |

---

# Part 2 — PRD

> Status: drafted locally (no issue tracker configured yet). Equivalent to a
> `ready-for-agent` PRD.

## Problem Statement

Product teams lose hours to the manual coordination work around incoming bug
reports: reading a vague report, deciding priority and component, checking for
duplicates, deciding whether there's even enough information to act, filing it in
Jira, and notifying the right people in Slack. Repetitive, judgment-light-until-
it-isn't work that sits between tools and people.

As the *author*, the problem is sharper: I need a portfolio piece that proves to
AI/ML recruiters that I understand how to build a reasoning agent — one that
plans, acts, observes, knows the boundary of its own competence, recovers from
failure, and whose quality I can *measure*. Most portfolio agents are
undifferentiated "it calls an LLM and an API" demos with no evaluation; that
reads as a toy to the audience I care about.

## Solution

A from-scratch Python reasoning agent that performs **bug-report → Jira triage**
with *visible, measured* intelligence. Given a bug report (CLI/endpoint first, a
thin Slack adapter later), the agent: writes an explicit printable **plan**;
executes it step by step against real tools (Jira, Slack, an embedding-backed
duplicate index); emits **structured evidence checks**; applies an explicit
**confidence-gating policy** (asking a human rather than guessing when unsure);
**reflects** and re-plans when reality diverges; and produces a **structured
trace** that drives a live Rich view, a saved artifact, and an **eval harness**.
The wow is the reasoning loop and the eval harness, not integration breadth.

## User Stories

1. As a product ops engineer, I want to hand the agent a raw bug report, so that a well-formed, correctly-triaged Jira ticket is created without my manual effort.
2. As a product ops engineer, I want the agent to write out an explicit plan before acting, so that I can see and trust what it intends to do.
3. As a product ops engineer, I want the agent to detect when an incoming bug duplicates an existing ticket, so that the backlog doesn't fill with redundant issues.
4. As a product ops engineer, I want the agent to recognize a *paraphrased* duplicate (different words, same bug), so that near-duplicates are caught, not just keyword matches.
5. As a product ops engineer, I want the agent to ask me a clarifying question when it lacks the information to triage responsibly, so that it doesn't silently guess a priority.
6. As a product ops engineer, I want the agent to set a sensible priority and component when the report is clear, so that I'm only interrupted for genuinely ambiguous cases.
7. As a product ops engineer, I want the agent to post a triage summary to Slack after filing, so that the relevant people are notified without me writing the message.
8. As a product ops engineer, I want the agent to recover when a step fails (e.g. the ticket already exists), so that a single hiccup doesn't crash the whole run.
9. As an eng lead, I want the agent's confidence broken into named evidence checks, so that I can judge *why* it made a call, not just *what* it decided.
10. As an eng lead, I want the agent to never set a priority it isn't confident about, so that triage stays trustworthy.
11. As a PM, I want a readable triage summary generated from the ticket data, so that I get the gist without reading the raw report.
12. As the author, I want a live terminal view rendering the plan as a checklist with evidence tables and highlighted gate/re-plan events, so that the reasoning is legible in a demo video.

---

# Part 2 — PRD

The PRD synthesized from this design lives in **`PRD.md`** (also published as a
GitHub issue labeled `ready-for-agent`). To slice it into implementation tickets,
run `/to-issues`.
