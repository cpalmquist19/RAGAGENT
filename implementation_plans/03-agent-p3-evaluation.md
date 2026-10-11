# Plan 03: Agent evaluation

| | |
|---|---|
| **Plan** | 03 · agent phase 3 of 3 |
| **Status** | not started |
| **Depends on** | 02 (agent loop) must be `done` |
| **Estimated size** | 4 small edits to existing files (~150 lines), test additions (~80 lines), doc updates |
| **API cost** | $0 for steps 1–5. 💲 About $2.50–4.00 in total for the eval runs in step 6, each run approved by the user first. |

## Goal

Measure the agent pipeline against the RAG pipeline and the long-context benchmark, and publish the numbers. Concretely:

1. Add a **context recall** metric: the share of the answer key's sections that were in front of the answering model. It is computed from `context_ids`, needs no LLM, and is directly comparable with the retrieval ablation's Recall@20 (69% for RAG).
2. Record the agent's **trace, research turns, and context size** in each eval record and summarize them in reports.
3. Add a free `licn-eval context` command that computes context recall for existing runs, including the RAG runs already in `evals/runs/`.
4. 💲 Run the evals (sample first, then all 220 questions) and the calculator ablation.
5. Update README.md and GAME_PLAN.md with the results, honestly, whatever they are.

## Background

Current results (from [README.md](../README.md), 220 questions, Haiku 5.5 answering and judging):

| | Long-context benchmark | RAG k=20 (`rag`) |
|---|---|---|
| Correct (judge) | 56% | 37% |
| Key-point coverage | 81% | 69% |
| Retrieval Recall@20 (sections) | n/a | 69% |
| Cost / median latency | $0.0018 / 12s | $0.0024 / 11s |

The agent exists to close the gap between those two columns **without** putting the whole corpus in the prompt. So the questions this plan answers are:

- Does the agent get more answers correct than RAG? (judge `correct`, key-point coverage)
- Does it put more of the right sections in front of the model? (new: context recall)
- How much policy text does it use to get there? (new: context words; the full corpus is about 17,000 words)
- What does it cost per question, and how slow is it? (target from GAME_PLAN §7: under $0.01 per question)
- Does the calculator help? (`agent` vs `agent_nocalc` on renewal math and eligibility questions)

## Read first

1. [src/licn_rag/evals/runner.py](../src/licn_rag/evals/runner.py): `METRICS`, `score_one`, `aggregate`, `print_report`, `cmd_run`, `cmd_retrieval`, `main`.
2. [src/licn_rag/evals/score.py](../src/licn_rag/evals/score.py): `section_of`, `ratio`, `deterministic`.
3. [src/licn_rag/pipelines/base.py](../src/licn_rag/pipelines/base.py) (`PipelineResult` now has `trace`, `research`, `context_words`), [src/licn_rag/pipelines/agent.py](../src/licn_rag/pipelines/agent.py), [src/licn_rag/pipelines/rag.py](../src/licn_rag/pipelines/rag.py), [src/licn_rag/pipelines/long_context.py](../src/licn_rag/pipelines/long_context.py).
4. [tests/test_score.py](../tests/test_score.py).
5. One existing run, to see the record format: `evals/runs/20261010-164558-rag/` (`config.json`, the first line of `results.jsonl`, `summary.json`).
6. README.md sections "Results so far" and "Roadmap", and GAME_PLAN.md §0 and §7.

The eval data lives in `evals/data/draft_questions.jsonl` and `evals/data/silver_answers.jsonl` (committed). If they're missing, stop and tell the user.

## Non-goals

- Don't change the answer prompt, the answer schema, the judge, or the existing metrics' definitions.
- Don't change the default pipeline of `licn-ask` (it stays `rag`). Whether the agent becomes the production pipeline is the user's decision after seeing the results; put a recommendation in your report.
- No CI, no verifier, no web app.
- Don't regenerate answer keys.
- Don't re-run the long-context or RAG full evals. Their runs already exist; compare against them.

## Steps

### Step 1: The context recall metric (`src/licn_rag/evals/score.py`)

Add two functions after `ratio`:

```python
def key_sections(key: dict[str, Any]) -> set[str]:
    """Sections the answer key cites, in its expected citations and its key points."""
    cites = set(key["expected_citations"]) | {c for kp in key["key_points"] for c in kp["citations"]}
    return {section_of(c) for c in cites}


def context_recall(context_ids: list[str] | None, key: dict[str, Any]) -> float | None:
    """Share of the key's sections that were in the answering model's context.

    None when the whole corpus was in context (context_ids is None) or the question is
    out of scope. Same definition as `licn-eval retrieval`'s Recall@k, so the numbers are
    comparable.
    """
    if context_ids is None or key["answerable"] == "no":
        return None
    return ratio(len(key_sections(key) & {section_of(c) for c in context_ids}), len(key_sections(key)))
```

Don't change `deterministic()`.

Add tests to `tests/test_score.py` (it already has a `key()` helper whose citations are `LICN-030 §1(a)`, `LICN-001 §2(g)`, and key point `LICN-030 §1(a)(i)`, i.e. sections `LICN-030 §1` and `LICN-001 §2`):

```python
def test_context_recall() -> None:
    assert context_recall(["LICN-030 §1(b)", "LICN-019 §2"], key()) == 0.5
    assert context_recall(["LICN-030 §1", "LICN-001 §2(w)"], key()) == 1.0
    assert context_recall([], key()) == 0.0
    assert context_recall(None, key()) is None                   # full corpus in context
    assert context_recall(["LICN-030 §1"], key(answerable="no")) is None
```

**Check:** `uv run pytest tests/test_score.py -q` passes.

---

### Step 2: Context size for the other pipelines

So reports can compare context size across all pipelines:

- In `RagPipeline.answer` ([pipelines/rag.py](../src/licn_rag/pipelines/rag.py)): after `generate(...)` returns, set `result.context_words = sum(len(c.text.split()) for c in chunks)` and return `result`. (Change `return generate(...)` into `result = generate(...)`, set the field, `return result`.)
- In `LongContextPipeline`: compute `self.context_words = len(corpus.render().split())` in `__init__`, and set `result.context_words = self.context_words` in `answer` the same way.

The agent pipeline already sets `context_words` (plan 02).

**Check:** `uv run pytest -q` passes; `uv run pyright src` reports 0 errors.

---

### Step 3: Record and summarize the new fields (`src/licn_rag/evals/runner.py`)

**3a. `METRICS`**: insert a row directly before `section_recall`:

```python
    ("context_recall", "Key sections in the model's context", "rate"),
```

**3b. `score_one`**: after `record |= {"model": ..., "context_ids": ...}`, add the new fields when present:

```python
    for field in ("context_words", "research", "trace"):
        if (value := getattr(result, field)) is not None:
            record[field] = value
```

and after `metrics = deterministic(...)`:

```python
    metrics["context_recall"] = context_recall(result.context_ids, key)
```

(Import `context_recall` from `licn_rag.evals.score`.)

**3c. `aggregate`**: add these keys to the returned dict. Every one must be `None` when no record has the underlying field, so old-style runs still aggregate.

```python
words = [r["context_words"] for r in scored if r.get("context_words") is not None]
traced = [r for r in scored if "trace" in r]
...
"context_words_p50": round(statistics.median(words)) if words else None,
"tool_calls_mean": (
    round(statistics.mean(
        sum(t["tool"] != "seed_search" for t in r["trace"]) for r in traced
    ), 2) if traced else None
),
"research_turns_mean": (
    round(statistics.mean(r["research"]["turns"] for r in traced), 2) if traced else None
),
"stopped_at_turn_limit": (
    sum(r["research"]["stop"] == "max_turns" for r in traced) if traced else None
),
"tool_errors": (
    sum(t["error"] for r in traced for t in r["trace"]) if traced else None
),
```

Records with a `trace` always have `research` (plan 02 sets both), so `r["research"]` is safe inside `traced`.

**3d. `print_report`**: after the `latency p50 / p95` row, add rows that read with `.get` and print `–` when the value is `None` or missing (old summaries don't have these keys):

| Row label | Value |
|---|---|
| `context words (median)` | `context_words_p50`, formatted `f"{v:,}"` |
| `tool calls / question (agent)` | `tool_calls_mean` |
| `research turns / question (agent)` | `research_turns_mean` |
| `stopped at turn limit (agent)` | `stopped_at_turn_limit` |

A small helper keeps this tidy:

```python
def cell(s: dict[str, Any], key: str, fmt: str = "{}") -> str:
    v = s.get(key)
    return "–" if v is None else fmt.format(v)
```

Also add `"context_recall"` to the per-type line printed for single runs: change the tuple `("correct", "key_point_coverage", "verdict_match", "section_recall")` to include `"context_recall"` at the end, and update the header string to `(correct · key-point coverage · verdict match · section recall · context recall)`.

**3e. `cmd_run`'s `config.json`**: record the pipeline's configuration for reproducibility. After the pipeline is built:

```python
pipeline_config = getattr(pipeline, "config", None)
...
config = {
    ...existing keys...,
    "pipeline_config": (
        dataclasses.asdict(pipeline_config) if dataclasses.is_dataclass(pipeline_config) else None
    ),
}
```

(`dataclasses` is already imported in `runner.py`. `RagPipeline` and `AgentPipeline` have a dataclass `config`; `LongContextPipeline` has none, which gives `None`.) If pyright objects that `asdict` may receive a dataclass *type*, add `and not isinstance(pipeline_config, type)` to the condition.

**3f. Tests.** Create `tests/test_runner.py`:

```python
"""Eval aggregation and reporting (no API calls)."""

from typing import Any

from licn_rag.evals.runner import aggregate, print_report


def record(**extra: Any) -> dict[str, Any]:
    base = {"id": "q1", "type": "eligibility", "cost_usd": 0.004, "latency_s": 12.0,
            "metrics": {"correct": True, "context_recall": 0.5}}
    return base | extra
```

Tests:

- `test_aggregate_old_style_records`: `aggregate([record()])` has `context_words_p50`, `tool_calls_mean`, `research_turns_mean`, `stopped_at_turn_limit` all `None`, and `overall["context_recall"] == 0.5`.
- `test_aggregate_agent_records`: two records, each with `context_words` (1000 and 3000), `research={"turns": 3, "stop": "end_turn", "calculations": []}` (make the second one `"max_turns"`), and a `trace` of `[{"tool": "seed_search", "error": False, ...}, {"tool": "get_section", "error": True, ...}]` (include `input` and `added_words` keys). Assert `context_words_p50 == 2000`, `tool_calls_mean == 1.0` (seed excluded), `research_turns_mean == 3.0`, `stopped_at_turn_limit == 1`, `tool_errors == 2`.
- `test_print_report_handles_old_summaries`: call `print_report([("old", aggregate([record()])), ("new", aggregate([...agent records...]))])` and assert it doesn't raise. Use pytest's `capsys` fixture to check that the output contains `"Key sections in the model's context"` and `"–"`.

**Check:** `uv run pytest -q` passes.

---

### Step 4: `licn-eval context` (free, for existing runs)

Add a subcommand that recomputes context recall from saved `results.jsonl` files, so the existing RAG runs get the new number without re-running anything.

```python
def cmd_context(args: argparse.Namespace) -> None:
    """Context recall of saved runs, recomputed from their context_ids (no LLM calls)."""
    keys = {r["id"]: r["answer"] for r in read_jsonl(ANSWER_KEYS) if "answer" in r}
    print(f"{'run':40}{'context recall':>16}{'n':>6}{'context words p50':>20}")
    for d in map(Path, args.runs):
        rows = read_jsonl(d / "results.jsonl")
        recalls = [
            v for r in rows
            if r["id"] in keys
            and (v := context_recall(r.get("context_ids"), keys[r["id"]])) is not None
        ]
        words = [r["context_words"] for r in rows if r.get("context_words") is not None]
        recall = f"{statistics.mean(recalls):.0%}" if recalls else "–"
        median_words = f"{statistics.median(words):,.0f}" if words else "–"
        print(f"{d.name:40}{recall:>16}{len(recalls):>6}{median_words:>20}")
```

Register it in `main()` next to `report`:

```python
c = sub.add_parser("context", help="context recall of saved runs (free)")
c.add_argument("runs", nargs="+", help="run directories")
```

and add `"context": cmd_context` to the dispatch dict.

**Check (free):**

```bash
uv run licn-eval context evals/runs/20261010-164558-rag evals/runs/20261010-163742-rag_k12 evals/runs/20261010-153106-long_context
```

Expected: `rag` about 69% (it should be close to Recall@20 in the README's ablation table, because it's the same definition over the same 191 answerable questions), `rag_k12` about 56%, and `long_context` shows `–` (its `context_ids` are `null`). If `rag` is far from 69%, stop and investigate before continuing: something in step 1 or 4 is wrong.

---

### Step 5: Full local check before spending money

- [ ] `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright src` all pass.
- [ ] Commit steps 1–5 now (see Commit below, first commit), so the code that produced the runs is in history before the runs.

---

### Step 6: 💲 Run the evals

**Ask the user before each run below**, giving the estimated cost. Estimates assume about $0.006 per agent question plus about $0.002 per question for the judge; the real numbers print at the end of each run. Run them in this order and stop at any decision point that fails.

All runs use the defaults: Haiku 5.5 for answering (`--model claude-haiku-5-5`) and judging (`--judge-model claude-haiku-5-5`), 8 workers.

#### 6a. Sample run, agent vs RAG on the same 30 questions (about $0.40)

`--sample 30` picks the same stratified questions every time, so these two runs are directly comparable:

```bash
uv run licn-eval run --pipeline agent --sample 30
uv run licn-eval run --pipeline rag --sample 30
uv run licn-eval report evals/runs/<agent-sample-dir> evals/runs/<rag-sample-dir>
```

**Decision point A.** Look at the agent run and check:

| Check | Healthy | If not |
|---|---|---|
| errors | 0–1 | Read the `error` fields in `results.jsonl`. Fix real bugs (that's in scope), then re-run 6a. |
| stopped at turn limit | ≤ 3 of 30 | The agent isn't finishing. See "Tuning" below. |
| tool calls / question | 1–6 | 0 means it isn't using tools; >6 means it's flailing. See "Tuning". |
| context recall | higher than RAG's on the same sample | If not higher, the agent isn't finding more of the right sections. See "Tuning". |
| cost / question | < $0.01 | Report it; ask the user whether to continue. |

Also read 3–4 traces in `results.jsonl` for questions the agent got wrong and RAG got right (or both wrong). Note what went wrong in your report: missed policy, read the right section but answered wrong, budget hit, and so on.

#### Tuning (only if decision point A fails, at most one round, user approval required)

You may change **only** these, and only one or two at a time:

- `AgentConfig` defaults in `pipelines/agent.py`: `seed_k` (try 20), `max_turns`, `max_evidence_words`, `research_effort` (`low` or `high`);
- the wording of `RESEARCH_INSTRUCTIONS` in `agent/prompt.py` (keep it general; never add facts about licensure rules or anything copied from an answer key);
- tool descriptions in `agent/tools.py`.

Re-run only `uv run licn-eval run --pipeline agent --sample 30` (about $0.25) and compare with `report`. Keep the change only if it helps. If one round doesn't fix it, stop and report to the user with the traces; don't keep iterating.

#### 6b. Full run (about $1.50–2.50)

```bash
uv run licn-eval run --pipeline agent
uv run licn-eval report evals/runs/20261010-153106-long_context evals/runs/20261010-164558-rag evals/runs/<agent-full-dir>
uv run licn-eval context evals/runs/20261010-164558-rag evals/runs/<agent-full-dir>
```

If the run has more than 5 errors, inspect them; for transient API errors (rate limits, overloaded) re-run only those questions with `--ids <id> <id> ...` into a separate run and report both. Don't edit results files by hand.

#### 6c. Calculator ablation (about $0.80, optional, ask the user)

```bash
uv run licn-eval run --pipeline agent_nocalc --types renewal_math eligibility
```

Compare against the agent full run **restricted to the same types**: the per-type lines in each run's printed report (`by question type`) give `renewal_math` and `eligibility` separately. Report correct and key-point coverage for both types, both pipelines.

---

### Step 7: Update the docs

Use the real numbers from step 6. If the agent did worse on some metric, say so plainly; the project's README already reports RAG's gap honestly, and this is the same standard.

**README.md:**

1. "Results so far" table: add a column `Agent (agent)` after `RAG, 20 chunks`. Fill every existing row from the full agent run's report. Add three rows at the bottom:
   - `Key sections in the model's context`: long-context `100%` (the whole corpus is in context), RAG k=12 and k=20 from `licn-eval context`, agent from its run.
   - `Policy text shown to the model (median words)`: long-context = its `context_words_p50` if you have it from a new run, otherwise compute `len(Corpus.load().render().split())` and write that; RAG and agent from `context_words_p50` (for the old RAG runs, which predate `context_words`, compute it the same way as step 2: words in the 20 or 12 chunks; or write "~3,000" / "~1,800" and say it's approximate).
   - `Tool calls per question`: `–`, `–`, `–`, agent's `tool_calls_mean`.
2. Replace the "What this shows" paragraph's last two sentences ("Closing this gap is the next phase…") with 2–4 sentences on what the agent run showed: correct rate vs RAG and the benchmark, context recall, cost and latency, and the most common remaining failure you saw in the traces.
3. Add the ablation result (`agent` vs `agent_nocalc` on renewal math and eligibility) as a sentence or a small table under the results, if 6c was run.
4. "How it works" section 3 (Retrieval): add a short paragraph describing agentic retrieval: seed search, then a bounded tool loop (name the six tools), then the same answer step over every passage the tools showed.
5. "Quickstart": add `uv run licn-ask --pipeline agent --trace "..."`.
6. "Roadmap": tick `Agentic retrieval` (`[x]`). The calculator is included; if 6c showed it didn't help, say so in the results instead of unticking.
7. The status blockquote at the top: update "Agentic retrieval is next" to reflect the result.
8. "Repository layout": add `agent/          research tools for agentic retrieval: search, outline, sections, references, calculator`.

Don't add any mention of human (gold) review of the answer keys to README.md.

**GAME_PLAN.md:**

1. §0 status table: change the `Agentic retrieval` row from `⏭ next` to `✅` with the headline numbers (correct %, context recall, cost per question). Add bullets under "What the measurements changed" for anything the runs taught (for example, whether the calculator helped, typical tool-call count, whether the turn limit was ever hit).
2. Update the revision line at the top (Revision 5, today's date, one clause on what changed).
3. §7 ablation paragraph: the "Still to come" list `+ agent tools → + calculator → + verifier` becomes done/remaining accordingly.
4. §8 Phase 2 checklist: tick the tools item and the "close the gap" item **only if** the agent's correct rate moved meaningfully toward the benchmark; otherwise leave it unticked and add a sub-bullet with the measured result.

**Check:** README tables render (look at them in a Markdown preview); every number you wrote can be traced to a `summary.json` or `licn-eval context` output you can name in your report.

## Acceptance checklist

- [ ] `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright src` all pass.
- [ ] `licn-eval context` reproduces about 69% for the existing `rag` run.
- [ ] The agent's sample and full run directories (and the ablation, if run) exist under `evals/runs/` with `config.json` (including `pipeline_config`), `results.jsonl`, `summary.json`.
- [ ] README.md and GAME_PLAN.md updated with the real numbers.
- [ ] Your report to the user contains: the comparison table (long-context / RAG / agent), context recall, cost and latency, 3–4 trace-based observations about failures, the calculator ablation result, total money spent, and a one-paragraph recommendation on whether `agent` should replace `rag` as the default pipeline.

## Commit

Branch `plan-03-agent-evaluation`. Two commits:

1. After step 5 (code only):

   ```
   Add context recall and agent trace metrics to the eval harness (implementation plan 03)
   ```

2. After step 7 (runs + docs):

   ```
   Add agentic retrieval eval results (implementation plan 03)

   <one line with the headline: agent X% correct vs RAG 37% and benchmark 56%,
   context recall Y%, $Z per question>
   ```

   Include the new `evals/runs/*` directories (the existing runs are committed the same way). Set this plan's **Status** to `done` here and in [README.md](README.md) in this commit.

## Gotchas

- **Old runs lack the new fields.** `aggregate`, `print_report`, and `cmd_context` must all tolerate records and summaries without `context_words`, `trace`, `research`, or the new summary keys. Use `.get` and print `–`.
- **The judge is also Haiku**, so absolute numbers carry the self-grading caveat already described in GAME_PLAN §7. Comparisons between pipelines are fair because every pipeline faces the same keys and judge. Don't switch the judge model without asking.
- **Don't compare a sample run with a full run.** Sample results go against sample results (6a), full against full (6b).
- **Rate limits.** The agent makes about 4–6 API calls per question instead of 1, so 8 workers put more load on the API. The client already retries up to 6 times. If you see many 429 errors, re-run with `--workers 4` and say so in the report.
- **Windows paths.** Run directories are named like `evals/runs/20261011-101500-agent`. Use forward slashes in commands; both shells accept them.
