# LICN RAG: Citation-Grounded Answers on NC Educator Licensure Policy

A retrieval-augmented generation (RAG) assistant for North Carolina educator licensure rules. A licensure analyst describes an anonymized educator ("BS in biology, year 2 of a Residency License, hasn't passed the content test") and gets an answer grounded in the State Board of Education's **LICN policy series**. Every claim cites the exact policy subsection it comes from, the assistant asks for missing facts instead of guessing, and it says when the policies don't cover a question.

> **Status:** in active development. The corpus pipeline, evaluation harness, long-context benchmark, hybrid RAG, and agentic retrieval pipelines are built and evaluated on all 220 questions. Since that evaluation, the LLM judge was rebuilt and the answer prompt was tuned on a held-out dev split; a full re-test of every pipeline on the untouched test split is next. See [Roadmap](#roadmap).

## Why this is harder than "chat with your PDFs"

Published evaluations of commercial legal research tools found incorrect or unsupported answers on roughly **17–33%** of queries ([Magesh et al.](https://arxiv.org/abs/2405.20362)). The usual failure isn't an invented source. It's a **real citation that doesn't support the claim**, or a rule applied without its exception. This project treats that as the core problem:

- **Citations follow the policies' own numbering.** Answers cite `LICN-030 §1(a)(ii)`, the same way the policies refer to themselves ("Subsection 2(a)"), so an analyst can check any claim in one step.
- **Every citation is checked.** The evaluation verifies that each cited passage exists and actually supports the claim attached to it.
- **Missing facts produce questions, not assumptions.** Eligibility questions often hinge on a fact the analyst didn't mention (grade band, years of experience, test results). The assistant is evaluated on whether it asks.
- **Out-of-scope questions are declined.** Salary schedules, HR rules, and processing times aren't in licensure policy, and the evaluation checks that the assistant says so.
- **Arithmetic and unit conversions run in code.** Renewal questions turn on CEU math, and models reliably convert units in the wrong direction (see [Results](#results-so-far)).

## How it works

```
Saved policy pages ──► HTML parser ──► structured corpus (24 policies, ~15k words, 623 citable passages)
                                              │
                         ┌────────────────────┴───────────────────┐
                         ▼                                        ▼
              Answering pipelines                         Evaluation set
              • long-context benchmark                    • 220 analyst-style questions
              • hybrid RAG                                • answer keys generated from the full corpus
              • agentic retrieval (tool use)              • dev (54) / test (166) split
                         │                                        │
                         └──────────► structured answer ◄─────────┘
                                      verdict · cited claims ·        scored by
                                      questions for the analyst ·     deterministic checks
                                      premise corrections ·           + an LLM judge
                                      not-covered parts
```

### 1. Corpus: from messy web pages to citable passages
The State Board's policy manual is hosted behind bot protection, so the pages are saved from a normal browser and parsed offline (`licn-ingest`). The source HTML varies a lot. Some policies are Word exports with every word in its own `<span>`, and some split one numbered list into separate fragments. The parser handles:
- nested numbering (`1` → `(a)` → `(i)`, Roman-numeral sections, alphabetic lists past `(z)` → `(aa)`)
- lists the source splits in two, which are reattached to the right section
- tables with `rowspan`/`colspan` (for example, the 96-row licensure exam table)
- two different page header formats
- flags for pending revisions and errors in the source itself, such as a duplicated section number

Each policy becomes structured JSON plus readable Markdown ([example](corpus/licn/LICN-030.md)), with effective dates, statutory and administrative-code citations, and a content hash for change detection.

### 2. A shared answer contract
Every pipeline returns the same structured answer (`AppAnswer`): a verdict (`yes`, `no`, `conditional`, `needs_more_info`, `informational`, `not_covered`), a bottom-line summary, claims that each carry citation IDs, questions for the analyst, premise corrections, and the parts the policies don't cover. Pipelines share one answer prompt and differ only in **which passages they show the model**, so evaluations compare retrieval strategies, not prompts. When the prompt changes, it changes for every pipeline, and they are all re-run.

### 3. Retrieval
Chunks follow the policy structure: a section and its sub-items stay together up to about 300 words, oversized sections split along their numbering, and the 96-row exam table splits by license group with its header repeated (113 chunks, every passage covered). Each chunk gets an LLM-written context note (Anthropic's *contextual retrieval*) before indexing. Retrieval is hybrid: **Qwen3-Embedding-0.6B** vectors in **sqlite-vec** and **FTS5 BM25**, merged with reciprocal rank fusion, all in a single SQLite file. Both models run locally on CPU, in float32, since bfloat16 was about 6x slower on CPUs without native bf16 support.

**Agentic retrieval.** The `agent` pipeline starts with the same hybrid search on the question (12 chunks), then lets Claude Haiku 5.5 research for up to 8 tool-use turns with seven tools: `search` (optionally inside one policy), `get_policy_outline`, `get_section`, `follow_references` (reads the sections a passage points at and lists passages elsewhere that cite its policy), `list_license_types`, `calculate` (exact arithmetic), and `convert` (CEUs ↔ semester credit hours ↔ clock hours, using the CEU definition in LICN-001 §2(g)). A cap on the policy text collected is enforced in code. The answer step is then the same structured-answer call the RAG pipeline makes, over every passage the tools showed, so the comparison with RAG changes only which passages the model sees. `licn-ask --trace` prints each tool call.

**Why a `convert` tool and not just a calculator.** On seven conversion-heavy questions, Haiku 5.5 wrote 4 conversions backwards and Sonnet 5.5 wrote 2 (6 semester hours as `6 × 2/3 = 4` CEUs; it is `6 ÷ 2/3 = 9`), often *inside* the calculator, which then reported the wrong number as exact. The calculator's own example in its tool schema had the direction backwards too. `convert` takes named units, owns the factor and its direction, and adds the defining passage to the evidence so the answer can cite it. `calculate` refuses hand-written `2/3` factors and points to `convert`.

### 4. Evaluation
- **Questions:** 220 analyst-style questions in six categories: lookup, eligibility, renewal math, missing facts, unanswerable, and temporal (time-dependent). A fixed, stratified **dev split (54)** is used for tuning; the **test split (166)** is held out for reporting.
- **Answer keys:** generated by a model that reads the *entire* corpus, so the keys don't depend on any retrieval step. Each key holds the expected verdict, cited key points, false premises in the question, facts to ask for, and out-of-scope parts. Every key citation is validated against the corpus.
- **Deterministic scoring (`licn-eval`):** verdict match, section- and policy-level citation recall and precision, invalid citations, correct abstention, asking for missing facts, **context recall** (share of the key's sections the answering model actually saw), and a **unit-conversion check** that parses conversion statements in answers ("6 semester hours equal 9 CEUs", "11.5 × 2/3 = 7.67 CEUs") and checks them against LICN-001 §2(g).
- **LLM judge:** Claude Haiku 5.5 grades each answer against its key in small, checkable parts: coverage of each key point (and whether that point is *central* to the bottom line), whether the bottom line agrees with the key, material errors (wrong *and* would change what the analyst does) kept separate from minor issues, and whether each cited passage supports its claim. The judge sees each cited list item together with the lead-in text of the sections above it. *Correct / partially correct / incorrect* is computed from those parts in code rather than asked for as one holistic label.
- **Experiment tooling:** `--split dev|test`; `--replay <run>` re-runs only the answer step on the passages a saved agent run collected (isolates prompt changes from retrieval variance, at about a third of the cost); `licn-eval rejudge <run>` grades saved answers again; every run records a fingerprint of the prompts it used.

## Results so far

All runs: 220 questions, Claude Haiku 5.5 answering and judging. The **long-context benchmark** puts the entire corpus in the prompt. It's the "perfect recall" bar the retrieval pipelines are measured against, not a production design.

| Metric | Long-context benchmark | RAG, 12 chunks | RAG, 20 chunks | **Agent** |
|---|---|---|---|---|
| Correct (judge v1) | **56%** | 27% | 37% | 47% |
| Correct or partially correct (judge v1) | 100% | 97% | 99% | 100% |
| Key-point coverage | **81%** | 63% | 69% | 77% |
| Claims supported by the cited text | 90% | 87% | 87% | 89% |
| Answers contradicting the answer key | 3% | 7% | 7% | **2%** |
| Verdict matches the answer key | 80% | 80% | 82% | 81% |
| Key sections in the model's context | 100% (whole corpus) | 56% | 69% | **80%** |
| Answers with a backwards unit conversion | 2 | 2 | 1 | **0** |
| Citations to non-existent passages | 0% | 0% | 1% | 0% |
| Declines out-of-scope questions | 100% | 100% | 100% | 100% |
| Asks for missing facts when needed | 100% | 99% | 100% | 100% |
| Cost / median latency per question | $0.0018 / 12s | $0.0019 / 10s | $0.0024 / 11s | $0.0047 / 17s |

The agent averaged 5.3 tool calls and 3.3 research turns per question and never hit the turn limit.

**What this shows**
- **Single-shot RAG gets the bottom line right but misses supporting policies.** Licensure answers are multi-hop: the median answer key draws on **5 different policies** (out-of-state rules, the license types they lead to, definitions, exam tables), and one search on the question text finds only the main one.
- **Agentic retrieval closes about half the gap to the benchmark** on correctness ((47−37)/(56−37)) and reads more of the right sections than either RAG setting (80% vs 69%) while showing the model about the same amount of text (median 3,400 words vs 3,200 for RAG). In a paired comparison on an earlier run the agent beat RAG (48 vs 28 wins, p = 0.03) and still trailed the benchmark (29 vs 50, p = 0.02). It costs about twice as much and takes 1.6× as long.
- **Unit conversions are a reasoning problem, not a retrieval problem.** Even the benchmark, with the whole corpus in view, wrote conversions backwards. The `convert` tool took backwards conversions to zero across all 220 answers.

**Read these numbers with care.** The "correct" rows come from the first judge (v1), which had two bugs found later: it cut cited passages off at 6,000 characters (the exam table is longer, so 87 of the agent's 220 answers had claims checked against truncated text), and it couldn't see the lead-in of a cited list item. Its single holistic label also flipped on 17% of identical answers between two grading passes. The rebuilt judge (v2, described above) agrees with a hand-graded 24-answer reference set on 20 of 24 answers, versus 11–14 for v1, and defines "correct" differently (right bottom line, central points, no material error), so **v1 and v2 "correct" rates are not comparable**. The rows that don't depend on the judge (verdict match, context recall, citation validity, conversions) stand as measured.

### Prompt tuning (dev split, judge v2)

The answer prompt was tuned against the failure modes in the judge's rationales: overstating the cited text ("at least" for "more than"), mapping out-of-state credentials to NC areas as if settled, unsupported effective dates, hedging when the policies do settle a question, uncorrected false premises, and leaving out related rules the analyst needs next. Each version was tested by replaying the answer step over the same saved passages for the 54 dev questions. A noise test came first: two identical runs differed on 17 of 54 per-question "correct" calls, so decisions were made on aggregate key-point coverage (stable within about ±1 point) with verdict match and unsupported claims as guardrails.

| Answer prompt | Key-point coverage | Verdict match | Answers with an unsupported claim | Claims supported |
|---|---|---|---|---|
| v0 (original, 2 runs) | 73–74% | 44–45 / 54 | 4–5 | 92% |
| v3 (current) | **78%** | 44 / 54 | 4 | **95%** |

Intermediate versions raised coverage but shifted verdict labels (a "give the outcome when every reading leads to it" rule turned `needs_more_info` into `no`) and added uncited related-rule claims; v3 keeps the coverage gain without either. A research-prompt change aimed at the same gap didn't move context recall (79% vs 78%) and was reverted. These are dev-split results; the test split is untouched until the full re-test.

### Retrieval ablation

Retrieval scored alone (no LLM, free): section-level recall of the answer key's citations within the top-k chunks, over the 191 answerable questions.

| Retrieval | Recall@8 | Recall@12 | Recall@20 | MRR | Search time |
|---|---|---|---|---|---|
| BM25 only | 36% | 45% | 56% | 0.70 | <1 ms |
| Dense only (Qwen3-Embedding-0.6B) | 45% | 55% | 68% | 0.82 | 233 ms |
| Hybrid (RRF), no context notes | 45% | 53% | 66% | 0.80 | 236 ms |
| **Hybrid + contextual chunks (production)** | **46%** | **56%** | **69%** | 0.81 | 241 ms |
| + cross-encoder reranker (mxbai-rerank-xsmall, 57-question sample) | 42% | 53% | – | 0.75 | 7,244 ms |

The reranker was dropped. On this domain the small general-purpose cross-encoder **lowered** recall and added about 7s per query on CPU. Recall@k is a conservative measure here, because the answer keys deliberately cite every relevant passage.

## Quickstart

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and an Anthropic API key.

```bash
uv sync
cp .env.example .env            # then add your ANTHROPIC_API_KEY

# Build the search index (first run downloads Qwen3-Embedding-0.6B, ~1.2 GB; context
# notes come from corpus/chunk_contexts.json, so no API calls unless chunks changed)
uv run licn-index build
uv run licn-index search "CEUs to renew a K-5 CPL"

# Ask a question (RAG by default; --pipeline long_context for the benchmark)
uv run licn-ask "Teacher has 6.5 CEUs this cycle on a CPL and teaches 4th grade. Enough to renew?"
uv run licn-ask --sources "..."   # also print the text of every cited passage

# Agentic retrieval: the model researches with tools first (--trace prints each tool call)
uv run licn-ask --pipeline agent --trace "..."

# Evaluate
uv run licn-eval retrieval                                   # retrieval-only ablation (free)
uv run licn-eval run --pipeline agent --split dev            # tune on dev; report on --split test
uv run licn-eval run --replay evals/runs/<agent-run> --split dev --label "what changed"
uv run licn-eval rejudge evals/runs/<run>                    # grade saved answers again
uv run licn-eval report evals/runs/<run-a> evals/runs/<run-b>
uv run licn-eval context evals/runs/<run>                    # context recall (free)
uv run licn-eval conversions -v evals/runs/<run>             # unit-conversion errors (free)

# Rebuild the corpus from saved policy pages, then the index
uv run licn-ingest path/to/saved-pages --out corpus/licn

# Tests, lint, type check
uv run pytest && uv run ruff check . && uv run pyright src
```

## Repository layout

```
src/licn_rag/
  ingest/        saved-page parser → structured corpus (licn-ingest)
  corpus.py      corpus loading, citation IDs, citation normalization and lookup
  answer/        shared answer schema and prompt
  rag/           chunking, contextual notes, SQLite vector + BM25 index (licn-index)
  agent/         research tools: search, outline, sections, references, calculator, unit conversion
  pipelines/     answering pipelines: RAG variants, long-context benchmark, agent, answer-step replay
  evals/         answer-key generation (licn-silver), scoring, judge, conversion check, runner (licn-eval)
  ask.py         command-line Q&A (licn-ask)
corpus/licn/     parsed LICN policies (JSON + Markdown) and manifest
evals/data/      eval questions, answer keys, dev/test split, judge reference labels
evals/runs/      saved evaluation runs (config records prompt fingerprint, split, label)
tests/           parser, scoring, judge, agent, and tooling tests
implementation_plans/  step-by-step build plans
GAME_PLAN.md     research notes, design rationale, and the full project plan
```

## Design decisions

- **Respect the source's access controls.** Rather than defeating the site's bot protection, policy pages are saved by hand and everything after that is automated. Re-ingesting reports which policies changed.
- **Benchmark against "perfect recall."** The corpus is small enough (~47k tokens) to fit in one prompt. That makes a long-context pipeline a useful upper-bound benchmark: the retrieval pipelines have to match its accuracy while using a fraction of the context.
- **Answer keys from a different context than the system under test.** Keys are written from the full corpus, independent of retrieval, so retrieval failures show up as lower scores instead of being baked into the keys.
- **Code owns arithmetic and units.** The model decides *what* to compute; `calculate` and `convert` decide the result. A calculator alone wasn't enough, because the model chose the wrong formula.
- **Measure the measuring instrument.** The judge was tested for repeatability (same answers graded twice) and against hand-graded references before it was trusted, and rebuilt when it failed both. Its overall label is computed from smaller judgments instead of asked for directly.
- **Tune on dev, report on test.** Prompts are tuned against a fixed dev split with cheap answer-step replays; reported numbers come from the held-out test split.
- **Measure, don't assume.** Every retrieval technique was ablated. The reranker was removed because it made results worse, and gaps to the benchmark are reported rather than hidden.
- **Cost-conscious by default.** Claude Haiku 5.5 with prompt caching and local CPU models for embedding. A full 220-question evaluation, including the judge, costs about $1–1.40.
- **Thin code over frameworks.** The pipelines, prompt, and scoring are a few hundred readable lines of Python rather than an orchestration framework, which makes them easy to inspect and change.

## Roadmap

- [x] Corpus ingestion with policy-native citation IDs
- [x] Evaluation set: 220 questions with validated answer keys
- [x] Evaluation harness: deterministic metrics + LLM judge, run comparison
- [x] Long-context benchmark pipeline
- [x] Hybrid RAG: structure-aware chunks with contextual notes, dense + BM25 retrieval, reciprocal rank fusion
- [x] Retrieval ablation and end-to-end comparison with the benchmark
- [x] **Agentic retrieval:** iterative search, exact policy lookup, cross-reference following, calculator and unit conversion
  - [x] Research tools and tool-use loop (`agent`, plus `agent_nocalc` without the math tools)
  - [x] Smoke test and full evaluation against RAG and the benchmark
- [x] Evaluation hardening: dev/test split, answer-step replay, re-judging, rebuilt judge validated against reference labels, unit-conversion check
- [x] First answer-prompt tuning round on the dev split
- [ ] **Full re-test** of long-context, RAG, and agent with answer prompt v3 and judge v2 on the test split, then decide whether the agent becomes the production pipeline
- [ ] Citation verification pass before answers are returned
- [ ] Web app (Next.js + FastAPI) with click-through citations
- [ ] Deployment on Google Cloud Run (Terraform), with CI running the evaluation suite

### Next steps

1. **Check the judge against a stronger model** once: grade the 24 reference answers with Sonnet 5.5 (about $0.50) to see whether Haiku goes easy on Haiku's answers.
2. **Full re-test on the test split** (166 questions) with the current prompts and judge: long-context, RAG (k=20), and agent, about $2.50 in total. These become the headline numbers.
3. **Decide from the numbers** whether the agent replaces RAG as the production pipeline, then write the follow-on plans: citation verifier, eligibility worksheet format, API, web app, deployment, CI.

The plans live in [implementation_plans/](implementation_plans/README.md).

## Disclaimer

This is an independent research and portfolio project, not an official tool of the NC State Board of Education or the Department of Public Instruction. Answers are a research aid, not licensure determinations. Confirm decisions with DPI Licensure. The policy text is public and belongs to the NC State Board of Education.
