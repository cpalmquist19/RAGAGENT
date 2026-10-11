# LICN RAG: Citation-Grounded Answers on NC Educator Licensure Policy

A retrieval-augmented generation (RAG) assistant for North Carolina educator licensure rules. A licensure analyst describes an anonymized educator ("BS in biology, year 2 of a Residency License, hasn't passed the content test") and gets an answer grounded in the State Board of Education's **LICN policy series**. Every claim cites the exact policy subsection it comes from, the assistant asks for missing facts instead of guessing, and it says when the policies don't cover a question.

> **Status:** in active development. The corpus pipeline, evaluation set and harness, long-context benchmark, and hybrid RAG pipeline are complete and evaluated. The agentic retrieval pipeline is built and unit-tested; its first real-API smoke test and evaluation are next. See [Roadmap](#roadmap).

## Why this is harder than "chat with your PDFs"

Published evaluations of commercial legal research tools found incorrect or unsupported answers on roughly **17–33%** of queries ([Magesh et al.](https://arxiv.org/abs/2405.20362)). The usual failure isn't an invented source. It's a **real citation that doesn't support the claim**, or a rule applied without its exception. This project treats that as the core problem:

- **Citations follow the policies' own numbering.** Answers cite `LICN-030 §1(a)(ii)`, the same way the policies refer to themselves ("Subsection 2(a)"), so an analyst can check any claim in one step.
- **Every citation is checked.** The evaluation verifies that each cited passage exists and actually supports the claim attached to it.
- **Missing facts produce questions, not assumptions.** Eligibility questions often hinge on a fact the analyst didn't mention (grade band, years of experience, test results). The assistant is evaluated on whether it asks.
- **Out-of-scope questions are declined.** Salary schedules, HR rules, and processing times aren't in licensure policy, and the evaluation checks that the assistant says so.

## How it works

```
Saved policy pages ──► HTML parser ──► structured corpus (24 policies, ~15k words, 623 citable passages)
                                              │
                         ┌────────────────────┴───────────────────┐
                         ▼                                        ▼
              Answering pipelines                         Evaluation set
              • long-context benchmark                    • 220 analyst-style questions
              • hybrid RAG (production path)              • answer keys generated from the full corpus
              • agentic retrieval (not yet measured)
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
Every pipeline returns the same structured answer (`AppAnswer`): a verdict (`yes`, `no`, `conditional`, `needs_more_info`, `informational`, `not_covered`), a bottom-line summary, claims that each carry citation IDs, questions for the analyst, premise corrections, and the parts the policies don't cover. Pipelines share one prompt and differ only in **which passages they show the model**, so evaluations compare retrieval strategies, not prompt wording.

### 3. Retrieval
Chunks follow the policy structure: a section and its sub-items stay together up to about 300 words, oversized sections split along their numbering, and the 96-row exam table splits by license group with its header repeated (113 chunks, every passage covered). Each chunk gets an LLM-written context note (Anthropic's *contextual retrieval*) before indexing. Retrieval is hybrid: **Qwen3-Embedding-0.6B** vectors in **sqlite-vec** and **FTS5 BM25**, merged with reciprocal rank fusion, all in a single SQLite file. Both models run locally on CPU, in float32, since bfloat16 was about 6x slower on CPUs without native bf16 support.

**Agentic retrieval** (built, not yet measured). The `agent` pipeline starts with the same hybrid search on the question (12 chunks), then lets Claude Haiku 5.5 research for up to 8 tool-use turns with six tools: `search` (optionally inside one policy), `get_policy_outline`, `get_section`, `follow_references` (reads the sections a passage points at and lists passages elsewhere that cite its policy), `list_license_types`, and `calculate` (exact arithmetic for CEU and credit conversions). A cap on the policy text collected is enforced in code. The answer step is then the same structured-answer call the RAG pipeline makes, over every passage the tools showed, so the comparison with RAG changes only which passages the model sees. `licn-ask --trace` prints each tool call.

### 4. Evaluation
- **Questions:** 220 analyst-style questions in six categories: lookup, eligibility, renewal math, missing facts, unanswerable, and temporal (time-dependent).
- **Answer keys:** generated by a model that reads the *entire* corpus, so the keys don't depend on any retrieval step. Each key holds the expected verdict, cited key points, false premises in the question, facts to ask for, and out-of-scope parts. Every key citation is validated against the corpus.
- **Scoring (`licn-eval`):**
  - deterministic: verdict match, section- and policy-level citation recall and precision, invalid citations, correct abstention, asking for missing facts
  - LLM judge: key-point coverage, whether each cited passage supports its claim, overall correctness

  Runs are saved and can be compared side by side.

## Results so far

All runs: 220 questions, Claude Haiku 5.5 for answering and judging. The **long-context benchmark** puts the entire corpus in the prompt. It's the "perfect recall" bar the RAG pipeline is measured against, not a production design.

| Metric | Long-context benchmark | RAG, 12 chunks (`rag_k12`) | RAG, 20 chunks (`rag`, production) |
|---|---|---|---|
| Correct (judge) | **56%** | 27% | 37% |
| Correct or partially correct | 100% | 97% | 99% |
| Key-point coverage | 81% | 63% | 69% |
| Claims supported by the cited text | 90% | 87% | 87% |
| Answers contradicting the answer key | 3% | 7% | 7% |
| Verdict matches the answer key | 80% | 80% | 82% |
| Citations to non-existent passages | 0% | 0% | 1% |
| Declines out-of-scope questions | 100% | 100% | 100% |
| Asks for missing facts when needed | 100% | 99% | 100% |
| Cost / median latency per question | $0.0018 / 12s | $0.0019 / 10s | $0.0024 / 11s |

**What this shows:** single-shot RAG usually gets the bottom line right (97–99% correct-or-partial) but misses supporting details. Licensure answers are multi-hop. The median answer key draws on **5 different policies** (for example, out-of-state rules, the license types they lead to, definitions, and exam tables), and one search on the question text reliably finds only the main one. Closing this gap is the goal of **agentic retrieval**, where the model searches iteratively, looks up specific policies, and follows cross-references. That pipeline is built but not yet evaluated, so no agent numbers are reported here yet.

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

The most common answer errors overall are **unit conversions** (for example, semester hours to CEUs), **declaring a question "not covered" when the policies do answer it**, and **time-dependent questions**. Planned fixes are a calculator tool, tighter abstention rules, and clearer handling of effective dates.

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
uv run licn-eval retrieval                        # retrieval-only ablation (free)
uv run licn-eval run --pipeline rag --sample 30   # end-to-end, scored by the judge
uv run licn-eval report evals/runs/<run-a> evals/runs/<run-b>

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
  agent/         research tools for agentic retrieval: search, outline, sections, references, calculator
  pipelines/     answering pipelines: RAG variants, the long-context benchmark, and the agent
  evals/         answer-key generation (licn-silver), scoring, and runner (licn-eval)
  ask.py         command-line Q&A (licn-ask)
corpus/licn/     parsed LICN policies (JSON + Markdown) and manifest
evals/data/      eval questions and answer keys
evals/runs/      saved evaluation runs
tests/           parser, scoring, and agent tests
implementation_plans/  step-by-step build plans for the agentic retrieval work
GAME_PLAN.md     research notes and full project plan
```

## Design decisions

- **Respect the source's access controls.** Rather than defeating the site's bot protection, policy pages are saved by hand and everything after that is automated. Re-ingesting reports which policies changed.
- **Benchmark against "perfect recall."** The corpus is small enough (~47k tokens) to fit in one prompt. That makes a long-context pipeline a useful upper-bound benchmark: the production RAG pipeline has to match its accuracy while using a fraction of the context.
- **Answer keys from a different context than the system under test.** Keys are written from the full corpus, independent of retrieval, so retrieval failures show up as lower scores instead of being baked into the keys.
- **Measure, don't assume.** Every retrieval technique was ablated. The reranker was removed because it made results worse, and RAG's gap to the benchmark is reported rather than hidden.
- **Cost-conscious by default.** Claude Haiku 5.5 with prompt caching and local CPU models for embedding. A full 220-question evaluation, including the judge, costs about $1.
- **Thin code over frameworks.** The pipelines, prompt, and scoring are a few hundred readable lines of Python rather than an orchestration framework, which makes them easy to inspect and change.

## Roadmap

- [x] Corpus ingestion with policy-native citation IDs
- [x] Evaluation set: 220 questions with validated answer keys
- [x] Evaluation harness: deterministic metrics + LLM judge, run comparison
- [x] Long-context benchmark pipeline
- [x] Hybrid RAG: structure-aware chunks with contextual notes, dense + BM25 retrieval, reciprocal rank fusion
- [x] Retrieval ablation and end-to-end comparison with the benchmark
- [ ] **Agentic retrieval:** iterative search, exact policy lookup, cross-reference following, a calculator for credit math
  - [x] Research tools: search, outline, section reads, cross-references, license types, calculator
  - [x] Tool-use loop pipeline (`agent`, plus `agent_nocalc` for the calculator ablation)
  - [ ] Smoke test against the real API
  - [ ] Evaluation against RAG and the long-context benchmark, then decide whether it becomes the production pipeline
- [ ] Citation verification pass before answers are returned
- [ ] Web app (Next.js + FastAPI) with click-through citations
- [ ] Deployment on Google Cloud Run (Terraform), with CI running the evaluation suite

### Next steps

1. **Smoke test the agent** on two real questions (about $0.01–0.03). Check that it calls tools, stops on its own, and cites sensibly.
2. **Evaluate it** ([plan 03](implementation_plans/03-agent-p3-evaluation.md)): add a free context-recall metric, run the agent against RAG on a 30-question sample, then on all 220 questions, and run the calculator ablation. Estimated total is about $2.50–4.00, and each run is approved before it starts.
3. **Decide from the numbers** whether the agent replaces RAG as the production pipeline, then write the follow-on plans: citation verifier, eligibility worksheet format, API, web app, deployment, CI.

The plans live in [implementation_plans/](implementation_plans/README.md).

## Disclaimer

This is an independent research and portfolio project, not an official tool of the NC State Board of Education or the Department of Public Instruction. Answers are a research aid, not licensure determinations. Confirm decisions with DPI Licensure. The policy text is public and belongs to the NC State Board of Education.
