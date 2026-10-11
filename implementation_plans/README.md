# Implementation plans

Step-by-step plans that an implementing model (or a person) follows to build the next piece of the project. Each plan is self-contained: it says which files to read first, exactly what to write, how to test it, and when it is done.

The project plan of record is [GAME_PLAN.md](../GAME_PLAN.md). These plans turn its roadmap items into executable work.

## Naming scheme

```
NN-<area>-p<phase>-<slug>.md
```

| Part | Meaning | Example |
|---|---|---|
| `NN` | **Plan number.** Two digits, global across the whole project, assigned in execution order. Never reused or renumbered, even if a plan is abandoned. | `03` |
| `<area>` | **Area** of the system the plan belongs to. One lowercase word from the table below. | `agent` |
| `p<phase>` | **Phase within that area**, starting at 1. Phases in one area run in order. | `p2` |
| `<slug>` | A few lowercase words, hyphenated, saying what the plan builds. | `agent-loop` |

So `02-agent-p2-agent-loop.md` is plan 2 overall and phase 2 of the agent area.

**How to refer to a plan.** All of these name the same file, and an implementing model should accept any of them:

- "plan 2", "implementation plan 2", "plan 02"
- "agent phase 2", "agent p2"
- "the agent loop plan"

To find a plan from a reference: list this directory and match the number (`02-*`) or the area + phase (`*-agent-p2-*`).

### Areas

| Area | Covers | GAME_PLAN reference |
|---|---|---|
| `agent` | Agentic retrieval: research tools, the tool-use loop, its evaluation | Phase 2 |
| `verifier` | Post-answer citation check (does each cited passage support its claim?) | Phase 2 |
| `worksheet` | Eligibility worksheet answer format (facts used, requirement rows) | Phase 2, §3.1 |
| `api` | FastAPI service, streaming, PII check, feedback and cost logging | Phases 2–3 |
| `frontend` | Next.js web app | Phase 3 |
| `infra` | Docker Compose, Terraform, GCP | Phases 0, 4 |
| `ci` | GitHub Actions: lint, type check, tests, eval gate | Phases 0, 5 |
| `docs` | README, GAME_PLAN, portfolio write-ups | Phase 6 |

Add a new area only when a plan doesn't fit any of these. Add it to this table in the same commit.

## Index

| # | Plan | Area / phase | Status | Depends on |
|---|---|---|---|---|
| 01 | [Research tools](01-agent-p1-research-tools.md) | agent p1 | done | none |
| 02 | [Agent loop](02-agent-p2-agent-loop.md) | agent p2 | in progress | 01 |
| 03 | [Agent evaluation](03-agent-p3-evaluation.md) | agent p3 | not started | 02 |

Expected next plans (not written yet, numbers not reserved): verifier p1, worksheet p1, api p1, frontend p1, infra p1, ci p1. Write them after plan 03's results are in, because those results decide whether the agent becomes the production pipeline.

### Status values

`not started` → `in progress` → `done` (or `blocked: <reason>`, `abandoned: <reason>`, `superseded by NN`).

Every plan has the same status in two places: the `Status` line in its own header and the Index table above. Update both in the commit that finishes the plan.

## Rules for the implementing model

1. **Read the whole plan before changing anything**, then read every file listed in its "Read first" section. Plans quote existing code, but the repository is the source of truth. If the plan and the code disagree, stop and tell the user rather than guessing.
2. **Do the steps in order.** Each step ends with a check. Don't start the next step until the check passes.
3. **Stay inside the plan's scope.** Don't refactor unrelated code, rename things, add dependencies, or "improve" files the plan doesn't mention. Each plan lists its non-goals.
4. **Ask before spending money.** Any command that calls the Anthropic API costs real money. Plans mark these steps with 💲 and give an estimated cost. Unit tests never call the API.
5. **Match the existing style:** Python 3.12, type hints everywhere, short module docstrings that explain *why*, comments only where the code isn't obvious, no frameworks (no LangChain). Line length is 100.
6. **Finish with the acceptance checklist.** All of it must pass: `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, and `uv run pyright src`.
7. **Commit once per plan** (or once per step if the plan says so) on a branch named `plan-NN-<slug>`, with a message that names the plan, for example `Add agent research tools (implementation plan 01)`. Don't push or open a PR unless the user asks.
8. **Report back** with what was built, the test results, any deviation from the plan and why, and the measured numbers if the plan has an evaluation step.

## Writing a new plan

Copy this skeleton:

```markdown
# Plan NN: <title>

| | |
|---|---|
| **Plan** | NN · <area> phase <k> |
| **Status** | not started |
| **Depends on** | <plans, or "none"> |
| **Estimated size** | <files touched, rough lines> |
| **API cost** | <$0, or an estimate> |

## Goal
## Background (why this exists)
## Read first
## Non-goals
## Steps
### Step 1: ...   (each step ends with a **Check:**)
## Acceptance checklist
## Commit
## Gotchas
```

Keep each plan small enough that a model can finish it in one session: roughly 1–6 new or changed files and fewer than ~600 lines of new code. Split anything bigger into phases.
