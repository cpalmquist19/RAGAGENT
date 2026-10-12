# Licensure RAG Assistant: Research Summary & Game Plan

*Prepared October 10, 2026 · Revision 6 (October 11, evening): the agent is evaluated on all 220 questions, a `convert` tool fixed backwards unit conversions, the LLM judge was rebuilt after a review, and the answer prompt was tuned on a new dev split. Revision 5: agent built and unit-tested (plans 01–02). Revision 4: corpus, answer keys, benchmark, hybrid RAG. See [§0 Status](#0-status-as-of-october-11-2026).*

*How this file and the README split the work: [README.md](README.md) is the public project page and the single source of truth for status and measured results. This file is the internal plan: goals and audiences, research notes, design rationale, budget, cloud architecture, phases, and risks. Numbers here are only the ones that drove a decision.*

**Decisions locked in:**
- **Hosting:** GCP Cloud Run on the always-free tier, managed with Terraform (not built yet)
- **Frontend:** Next.js (not built yet)
- **Answering model:** Claude Haiku 5.5
- **Corpus collection:** pages are saved from a normal browser and parsed by `licn-ingest`. Automated scraping was dropped because the site's bot protection blocks it (§2.2).
- **Search index:** SQLite + sqlite-vec + FTS5 in one file (currently `data/index/`, later in Cloud Storage)
- **Retrieval:** contextual chunks, hybrid dense + BM25 with reciprocal rank fusion, 20 chunks, **no reranker** (it lowered recall, §0)
- **Citations:** structured output with policy-native citation IDs (`LICN-030 §1(a)(ii)`), checked against the corpus
- **Arithmetic and units in code:** `calculate` for arithmetic, `convert` for CEU / semester-hour / clock-hour conversions (§3.1)
- **Evaluation discipline:** tune prompts on the dev split (54 questions), report on the held-out test split (166); judge v2 (decomposed rubric, §7)

**Goal:** Build an assistant that answers NC educator-licensure questions using only the **State Board of Education Licensure (LICN) policies**. Most questions will be hypothetical scenarios: *"Someone has X degree and Y license with Z years. What do they qualify for? Do they need more renewal credits?"* Each answer must cite the exact policy and section it relies on, show the user what facts it assumed, and say so plainly when the policies don't settle the question.

**Two audiences, one design:**
1. **Your wife (the real user):** she needs answers she can trust and check in one click.
2. **Hiring managers (the resume):** they want to see a modern architecture, measured quality (evals with numbers), and good engineering habits (tests, CI, observability, a clean deployment).

**Constraint: as cheap as possible.** The target is under $5/month in total. §4 shows how.

---

## 0. Status (as of October 11, 2026)

Measured results live in the [README](README.md#results-so-far); this section records state and the decisions the numbers drove.

| Area | State |
|---|---|
| Corpus | ✅ 24 LICN policies → `corpus/licn/` (623 citable passages). LICN-014 is cited by other policies but not in the corpus. |
| Eval set | ✅ 220 questions + silver answer keys; fixed dev (54) / test (166) split in `evals/data/splits.json`. Keys not yet human-reviewed. |
| Eval harness | ✅ `licn-eval`: deterministic metrics, context recall, unit-conversion check, judge v2, replay and re-judge tooling |
| Long-context benchmark | ✅ evaluated (220) |
| Hybrid RAG | ✅ evaluated (220, k=12 and k=20) |
| Agentic retrieval | ✅ evaluated (220): closes about half the RAG-to-benchmark gap at about 2× RAG's cost. Seven tools including `convert`. |
| Judge | ✅ rebuilt (v2) after review; agrees with a 24-answer hand-graded reference on 20/24. Sonnet cross-check pending. |
| Answer prompt | ✅ v3, tuned on dev (coverage 73–74% → 78%, no verdict or support regressions) |
| Full re-test on the test split | ⏭ next: all three pipelines with prompt v3 and judge v2 (~$2.50) |
| Web UI, API, cloud, CI | not started |

**Next steps** (build plans are in [implementation_plans/](implementation_plans/README.md)):
1. Sonnet 5.5 grades the 24 reference answers once (~$0.50), to check for self-preference in the Haiku judge.
2. Re-test long-context, RAG (k=20), and agent on the test split with prompt v3 and judge v2. These become the headline numbers.
3. Decide whether the agent becomes the production pipeline, then write the follow-on plans: citation verifier, eligibility worksheet format, API, frontend, infra, CI.

**What the measurements changed:**
- **The reranker was dropped.** `mxbai-rerank-xsmall-v1` lowered Recall@8 from 48% to 42% and added about 7s per query on CPU.
- **Context notes help a little** (+1–3 points of recall). Hybrid beats dense-only by about 1 point. BM25 alone is clearly worse.
- **20 retrieved chunks beat 12** end to end.
- **Embedding models must load in float32.** Qwen3-Embedding loads in bfloat16 by default, which was about 6x slower on a CPU without native bf16. That matters for Cloud Run too.
- **The long-context benchmark beats single-shot RAG** because answers are multi-hop (median key spans 5 policies). **The agent reads more of the right sections** (80% of key sections in context vs 69% for RAG) and recovers about half the gap.
- **A calculator isn't enough for unit math.** Models picked the wrong conversion direction, even inside the calculator (whose own schema example was backwards). Moving the direction into code (`convert`) took backwards conversions from 3 to 0 in 220 answers. Even the long-context benchmark makes this error, so it's reasoning, not retrieval.
- **The first judge wasn't trustworthy.** It truncated cited passages (87 of 220 agent answers affected), couldn't see a list item's lead-in, and its single label flipped on 17% of identical answers. Rebuilt as smaller judgments with the label computed in code. **"Correct" rates from judge v1 and v2 are not comparable.**
- **Per-question "correct" is too noisy to tune on.** Two identical runs differed on 17 of 54 questions. Tuning decisions use aggregate key-point coverage (stable to about ±1 point) with verdict match and unsupported claims as guardrails.
- **Prompt tuning is part of the experiments, not frozen.** The earlier rule against tuning existed to keep pipeline comparisons fair and avoid tuning on the reported questions. Both are now handled differently: the shared answer prompt changes for every pipeline at once, and tuning uses only the dev split.

---

## 1. State of RAG in late 2026: what the research says

### 1.1 The consensus stack
Across 2026 papers and practitioner write-ups, the "serious" RAG baseline has settled into:

| Layer | 2026 standard practice | How it applies here |
|---|---|---|
| **Parsing** | Layout-aware parsing into structured Markdown/JSON that keeps headings, lists, and tables | Our source is **HTML**, not PDF, so we parse the DOM directly and keep heading and table structure. |
| **Chunking** | Structure-aware chunking that follows the document's own hierarchy, not fixed token windows | Studies report that structure- and metadata-aware chunking matters *more* than which parser you pick. |
| **Contextual retrieval** | Before embedding a chunk, prepend a short LLM-generated note on where it sits in the document | A paragraph that says "the individual must complete…" means nothing on its own. Prepending "LICN-001 General Licensure Requirements › Renewal" fixes that. |
| **Hybrid search** | Dense vectors + BM25 keyword search, fused with Reciprocal Rank Fusion (RRF) | Questions mix concepts ("can she teach outside her area?") with exact terms ("CEU", "Residency License", "LICN-005"). You need both. |
| **Reranking** | A cross-encoder reranker over the top ~50 fused results, keeping the best ~8 | Often cited as a large gain, but **on this corpus a small local reranker made recall worse** and cost ~7s/query on CPU, so it was dropped (§0). A stronger reranker could be re-tested later. |
| **Agentic retrieval** | The LLM drives retrieval as tools in a bounded loop: search, read the full section, follow cross-references | LICN policies refer to each other, to statutes, and to NCAC rules. A single retrieve-then-answer pass can't follow that. |
| **Grounded generation** | Native citation APIs with a strict citation contract: every claim is tied to a source span, and the model abstains when evidence is missing | This is the main hallucination control. |
| **Verification** | An automated post-check that each cited span actually supports the sentence citing it | Linked citations are not the same as grounded citations (see 1.2). |
| **Evaluation** | Golden question sets + retrieval metrics (Recall@k, MRR) + generation metrics (faithfulness, citation accuracy, abstention) run in CI | Without this you can't claim the system is reliable, and you can't show it on a resume. |

### 1.2 Sobering findings specific to legal RAG
- **Commercial legal RAG still fails often.** The Stanford study of Lexis+ AI and Westlaw AI-Assisted Research found that they gave incorrect or unsupported answers on roughly **17–33%** of queries. ([Magesh et al.](https://arxiv.org/pdf/2405.20362))
- **Statutory questions are especially hard.** A 2026 CSLAW benchmark found low scores for LLM+RAG systems on statutory questions, and frequent citations to sources that didn't support the claim. ([CSLAW '26](https://dho.stanford.edu/wp-content/uploads/Stat_Surveys.pdf))
- **"A linked citation is not a grounded one."** Showing a citation proves nothing unless the cited text supports the claim. ([SSRN 2026](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6945200))
- **The failure mode has changed.** Modern systems cite *real* documents that are **outdated, superseded, or missing context**. ([arXiv 2606.09724](https://arxiv.org/pdf/2606.09724))

**Our biggest risks, in order:**
1. **Bad eligibility reasoning:** the model applies the right policy to the scenario incorrectly. For example, it skips a requirement, miscounts renewal credits, or assumes a fact the user never gave.
2. **Stale text:** it cites a pending revision or a superseded version as current.
3. **Missing context:** it misses an exception or definition elsewhere in the policy.
4. **Ungrounded citations:** it cites a real section that doesn't say what the answer claims.

The architecture below targets each of these directly.

### 1.3 What we're deliberately *not* building
- **Full GraphRAG:** not worth it for this corpus. Every LICN policy header already lists its statutory and NCAC citations, which gives us a citation graph for free.
- **Fine-tuning:** not justified until evals show a specific gap.
- **Multi-agent systems:** one well-tooled agent is enough.
- **Long-context-only in production:** the whole LICN series fits in one prompt (about 47k tokens). It's built as a **benchmark** for "perfect recall," and currently it beats single-shot RAG (§0). Production still uses retrieval, which scales beyond this corpus and is the system being demonstrated. The goal of Phase 2 is to close the gap.

---

## 2. The corpus: NC SBE Policy Manual, LICN series

**Source:** the NC State Board of Education Policy Manual, hosted on Simbli (eBoard Solutions):
`https://simbli.eboardsolutions.com/Policy/PolicyListing.aspx?S=10399&ptid=muNUlKiR2jsXcslsh28JpBkiw==`

**Scope:** only policies whose code starts with `LICN-`. Everything else is ignored.

### 2.1 What a policy looks like (from LICN-001)
Each policy has a structured header followed by the body text:

| Field | Example (LICN-001) | Use in the system |
|---|---|---|
| Code / Title | LICN-001, *General Licensure Requirements* | Primary citation key and exact-lookup target |
| Status | ADOPTED | Filters out non-adopted text |
| Original Adopted / Last Revised / Last Reviewed | 05/13/2021 · 07/01/2026 · 07/01/2026 | Detects changes, shows the "as of" date |
| Effective Date | July 1, 2026 | Shown on every citation |
| Statutory Citation(s) | G.S. 115C-12(9); 115C-270.1; 115C-270.20; … | Free citation graph, so the answer can say "this rests on G.S. 115C-270.20" |
| NCAC Citation(s) | 16 NCAC 06C .0301; .0302 | Same |

### 2.2 Getting the documents (decided: browser save + automated ingest)
- Simbli is behind **Imperva bot protection**, which blocks automated browsers. We don't try to get around it. Instead, a person saves each LICN policy page from a normal browser into `nc-licensure-papers/`. That folder is git-ignored: the pages are about 148 MB with site assets, and they contain session tokens.
- `uv run licn-ingest nc-licensure-papers` parses each saved page into `corpus/licn/<CODE>.json` (structured) and `<CODE>.md` (readable). It also writes a `manifest.json` with each policy's dates, source URL, and a content hash, and it reports any warnings.
- The parser keeps the policies' own numbering as block paths (`LICN-030 §2(b)(iv)`), so citations match how the policies cite themselves. It handles:
  - Word-export markup that wraps every word in its own element
  - merged table cells (rowspan/colspan)
  - lists that the source splits into two
  - pending revisions (`isPndg=true`)
- **Freshness:** re-save the pages after the SBE's July updates (or whenever a revision is announced), then re-run the ingester. Changed hashes show what to re-index.
- **Option B is still open:** asking SBE/DPI or eBoard Solutions for a bulk export or permission to automate. If that's granted, an automated crawler (for example Playwright with an honest user agent) could replace the manual save.
- **Actual size (October 10, 2026):** **24 policies, about 15,200 words, about 21k tokens.** That's far smaller than the original estimate.
  - LICN-006 (salary credit) and LICN-008 (fees, dated 2018) use an older page format, which the parser also handles.
  - LICN-014 (Provisional License) is cited by LICN-001, -010, and -011 but is not in the corpus. It may have been repealed. Answers note it as a gap.
  - Known errors in the source pages: LICN-006 numbers two sections "IV"; LICN-003 cites "GS 270.15" and is missing the PLT score.

> **Design implication:** the whole corpus fits in a single prompt for about $0.002–0.003 with Haiku, or less with prompt caching. The long-context baseline (§1.3) is now a serious contender, not a straw man. The evals will decide whether retrieval beats it on citation precision, latency, and cost. Whichever wins, the ablation table will show the comparison honestly.

### 2.3 Chunking (built: `src/licn_rag/rag/chunking.py`)
- **A chunk is a section subtree:** an item plus its sub-items, up to about 300 words. Oversized sections split along their numbering. A split section's title opens its first child chunk rather than becoming a tiny chunk of its own. Small neighboring sections are packed together. Result: **113 chunks**, median about 146 words, covering all 623 citable passages.
- **The 96-row exam table in LICN-003** splits by license group, with the header row repeated in each piece.
- **Each chunk carries:**
  - a breadcrumb (policy › section › parent lead-in)
  - its passages, each prefixed with its citation ID
  - the list of citation IDs it covers
  - an LLM-written context note, cached in `corpus/chunk_contexts.json`
- **Deep links (planned for the UI):** each citation opens the official Simbli page, using a URL text fragment (`#:~:text=…`) to highlight the passage. The cached passage is also shown inline, so it works even if the site changes.

---

## 3. Her actual use case: scenario-based eligibility questions

Her questions are mostly **"apply the rules to this person"**, not "what does the policy say". That changes the design: retrieval alone isn't enough. The system has to **reason over requirements**, and the reasoning has to be just as checkable as the citations.

### 3.1 The "eligibility worksheet" answer format
For scenario questions, the agent answers in a fixed structure instead of free prose:

```
Facts I used (from your question):
  • Bachelor's in Biology  • Holds an Initial Professional License (IPL)  • 3 years teaching  • 4.5 CEUs this cycle
Facts I'm assuming or missing:  ⚠ Has the EPP recommendation been made? (affects requirement 3)

License: Continuing Professional License (CPL)                       Verdict: NOT YET, 1 item outstanding
 #  Requirement                                     Status        Source
 1  3 years of teaching experience                  ✅ Met        LICN-00X §… [link]
 2  Completed the Beginning Teacher Support Program  ✅ Met        LICN-00X §… [link]
 3  …                                               ❓ Unknown    LICN-00X §… [link]
 4  8.0 renewal credits incl. X in literacy          ❌ 3.5 short  LICN-00X §… [link]

Bottom line: … (every sentence cited)
Not legal advice. Confirm with DPI Licensure or district HR.
```

*(The requirements above are made up to illustrate the format. Real values come only from retrieved policy text.)*

**Why this format matters for reliability:**
- **Fact extraction is separated from rule application.** The model first lists the facts it pulled from her question. Missing facts become **❓ Unknown plus a follow-up question**, never a silent assumption. This targets risk #1.
- **Every requirement row carries its own citation**, so she can check any single row in one click.
- **Arithmetic is done in code, not by the LLM.** Credit hours, years, and deadlines go through a small `calculate` tool, and unit conversions through `convert`, which owns the conversion direction (the model only names the units). Measured: models get the direction wrong often enough that a calculator alone didn't fix it (§0).
- **The answer is a structured output** (a Pydantic schema), so the UI renders it consistently and the evals can grade each row separately.

### 3.2 Privacy by design
She already plans to describe people generically ("this person has…"), which is the right approach. The app reinforces it:
- A banner says not to include names, employee IDs, or SSNs.
- A lightweight **PII detector** (regex plus a name heuristic) runs before anything is sent to the LLM. If it finds anything, it warns her and offers to redact.
- **No conversation storage by default.** Traces in the observability tool are scrubbed.
- The corpus is entirely public policy, and no state or district data is ingested.
- Until the district's AI policy is known, treat the tool as a **personal research aid** that points to official sources, not an official determination.

---

## 4. Budget & hosting: "as cheap as possible", in the cloud

### 4.1 Why GCP over AWS
| | **GCP (recommended)** | AWS |
|---|---|---|
| Free tier model | **Always Free never expires.** Cloud Run: about 2M requests, 180k vCPU-seconds, and 360k GiB-seconds per month. Firestore: 1 GiB. Cloud Storage: 5 GB (US regions). | Accounts created after July 15, 2025 get a **credit-based Free Plan ($100–200) that ends after 6 months**. Lambda, DynamoDB, and some other services keep Always Free allowances. |
| Fit for this app | Cloud Run runs ordinary **Docker containers** (FastAPI, Next.js) and **scales to zero**. This is the simplest serverless path, and the same containers run locally. | Doable with Lambda container images + DynamoDB + S3 + CloudFront, but with more glue (API Gateway, Lambda size limits, cold starts with ML models). |
| Resume value | Cloud Run, Terraform, Workload Identity Federation, Secret Manager, IAM service-to-service auth | Equivalent, and AWS appears in more job postings |

**Recommendation: GCP**, because its free tier doesn't expire and Cloud Run matches our container-based design. Everything is in Terraform and Docker, so an AWS port later is a realistic stretch goal, and "deployed on both" is a good resume line.

> The free-tier figures above come from third-party 2026 guides. Confirm them at cloud.google.com/free when setting up. **Set a billing budget alert at $1 and $5 on day one.**

### 4.2 Cost by component
| Component | Choice | Cost |
|---|---|---|
| Web frontend | **Next.js** on Cloud Run (standalone output, scales to zero) | **$0** (free tier) |
| API | **FastAPI** on Cloud Run (scales to zero, 2 GiB memory) | **$0** (free tier) |
| Search index | **SQLite + `sqlite-vec` (vectors) + FTS5 (keyword/BM25)** in a single file. It is built by the ingestion job, uploaded to Cloud Storage, and loaded by the API at startup. The corpus is small and read-only between crawls, so a database server would be wasted money. *(Cloud SQL Postgres is not free.)* | **$0** |
| Query embeddings | **Qwen3-Embedding-0.6B** via sentence-transformers in **float32** (about 0.2s per query on CPU); ONNX int8 is an option for a smaller container. Corpus embeddings are computed when the index is built. | **$0** |
| Reranking | **None in production.** A local cross-encoder was tested and lowered recall (§0). | **$0** |
| Bulk LLM work (chunk context notes, eval judging) | **Claude Haiku 5.5**: $0.10 / $0.50 per M tokens (input/output). Context notes for all 113 chunks cost $0.03. A full 220-question eval with judging costs about $0.80 (long-context) to $1.40 (agent), run directly rather than through the Batch API (batch turnaround was unpredictable). Prompt experiments use dev-split replays at about $0.25 each. | **~$1 per full eval** |
| Answering | **Claude Haiku 5.5** (decided). Measured at **$0.0024/question** for RAG (20 chunks), $0.0047 for the agent, and $0.0018 for the cached long-context benchmark. | **~$0.002–0.005/question** |
| Re-ingest | Manual: re-save pages, then run `licn-ingest` and `licn-index build`. A scheduled job would need an approved automated source (§2.2). | **$0** |
| Feedback & usage logs | **Firestore** (thumbs up/down, flagged citations, per-question cost). No question text stored by default. | **$0** |
| Secrets | **Secret Manager** (Anthropic API key) | **$0** (free tier) |
| Observability | **OpenTelemetry → Cloud Trace + Cloud Logging** (free allotments). Langfuse Cloud's free hobby tier is optional for LLM-specific trace views. | **$0** |
| Container images | **Artifact Registry**. The API image includes PyTorch and the embedding model, which exceeds the 0.5 GB free allowance. | **~$0.10–0.30/month** |
| Local-model mode | LM Studio adapter, for dev runs and eval comparisons only | **$0** |

**Realistic monthly cost:** about $0.30 for image storage + under $1 in Haiku calls + about $0.10–0.50 per full eval run. **Total: about $1–3/month.**

Prompt caching helps a lot here: the system prompt, tool definitions, and frequently retrieved policies are identical across questions, so cache reads cost a fraction of normal input.

**Cold starts:** with scale-to-zero, the first question after an idle period may take about 5–15 seconds while the container loads the embedding model and index. Later questions are fast. If this bothers her, setting `min-instances=1` on the API removes the delay, but that probably costs a few dollars a month. We'll measure before deciding.

---

## 5. Target architecture

```
  Legend: ✅ built and measured   🔨 built, not yet measured   ⏭ next (Phase 2)   ○ later

  INGESTION (manual trigger, offline)
  ✅ pages saved from a browser ──► licn-ingest: parse ──► corpus/licn/*.json (+ .md, manifest with hashes)
        │
        ▼
  ✅ licn-index build: section-subtree chunks (≤ ~300 words, big tables split by group)
        ├─► ✅ context note per chunk (Haiku, prompt-cached per policy, notes cached on disk)
        ├─► ✅ Qwen3-Embedding vectors (sqlite-vec) + FTS5 BM25, in one SQLite file
        ├─► ○ cross-reference graph: policy ↔ policy, policy ↔ G.S./NCAC
        └─► ○ change log between corpus versions

  QUERY (online)
  ✅ today:   question ──► hybrid search (dense + BM25 → RRF → top 20) ──► Haiku, structured answer
  ✅ `agent` pipeline: question ──► seed search (hybrid, 12 chunks) ──► research loop
        (Haiku 5.5, max 8 turns, tools stop reading at 6,000 words of policy text)
        ├─ search(query, policy?)               the hybrid search above, optionally inside one policy
        ├─ get_policy_outline(code)             a policy's sections with citation IDs and word counts
        ├─ get_section(id)                      read a whole section with its sub-items (a bare code reads a short policy)
        ├─ follow_references(id)                read what a passage points at; list passages that cite its policy
        ├─ list_license_types()                 the license catalog, so it reasons over the right options
        ├─ calculate(expression)                credits, years, deadlines computed in code
        └─ convert(value, from_unit, to_unit)   CEU ↔ semester hour ↔ clock hour, direction owned by code
        │
        ▼  every passage the tools showed, in document order
  ✅ Structured answer (same call and prompt as RAG): verdict • cited claims • questions for the analyst • not covered
  ⏭ Phase 2, still to build: PII check before the loop • worksheet format (facts used, requirement rows)
        │
        ▼
  Verifier (Haiku): does each cited passage support its row/sentence? If not: flag, or regenerate once
        │
        ▼
  Web UI: worksheet view │ click a citation → passage shown inline + "open on Simbli" deep link
          "Policies current as of <date>" │ "pending revision exists" badge │ follow-up questions for missing facts
```

### Cloud deployment (GCP, all in Terraform)
```
   Her phone / laptop
          │ HTTPS
          ▼
  ┌──────────────────────┐   IAM-authenticated call    ┌───────────────────────────────┐
  │ Cloud Run: web       │ ─────────────────────────►  │ Cloud Run: api (private)      │
  │ Next.js + Auth.js    │  (only web's service account │ FastAPI + agent loop          │
  │ Google sign-in,      │   may invoke the API)        │ Qwen3 embedder + index.sqlite │
  │ email allowlist      │                              │ (loaded from GCS at startup)  │
  └──────────────────────┘                              └──────┬──────────────┬─────────┘
                                                               │              │
                                     Secret Manager (API key) ─┘              ├─► Anthropic API (Haiku 5.5)
                                                                              └─► Firestore (feedback, cost logs)
  Re-ingest (manual trigger; scheduled only if an approved automated source is found)
          │
          ▼
  Cloud Run Job: ingest ──► saved pages from GCS ──► parse/chunk/embed ──► run retrieval eval smoke test
                                                                     │  (a new index is published only if the eval passes)
                                                                     ▼
                                                 GCS: index-<date>.sqlite + raw HTML archive + change log
  GitHub Actions ──(Workload Identity Federation, no stored keys)──► Artifact Registry ──► Cloud Run deploy
```

**Security notes:**
- The API is **not publicly invocable**. Only the web service's identity can call it.
- Only allowlisted Google accounts (hers and yours) can sign in.
- No long-lived cloud keys exist anywhere, because CI authenticates through Workload Identity Federation.
- **Eval-gated index publishing:** a crawl that produces a broken index never goes live.

These points make a good security section in the README.

### Key design decisions
1. **The chunk is the policy section**, not a token window. Tables stay whole.
2. **Exact lookup before semantic search.** "LICN-005" or "Residency License" go to a direct lookup or keyword hit first.
3. **Retrieve small, read big.** Search over sections, then let the agent read the full surrounding section so exceptions aren't missed.
4. **Citations through structured output, not native `search_result` citations.** Claude's native citations can't be combined with structured output (`output_config.format`), and the structured answer (verdict, cited claims, questions for the analyst) is what the evals and UI need. So every passage is shown with its citation ID, the model cites IDs in each claim, and code validates every ID against the corpus. Native citations remain an option for a free-text mode.
5. **A strict citation contract** in the system prompt:
   - Use only the provided policy text.
   - Every requirement and claim needs a citation.
   - If a fact is missing, ask for it rather than assume it.
   - If the policies don't answer the question, say so.
6. **Verification pass** to catch citations that don't support their claims.
7. **Time-awareness.** Answers use current adopted text by default, pending revisions are flagged rather than mixed in, and the "as of" date is always shown.
8. **Model-agnostic core.** Retrieval, chunking, evals, and the answer schema don't depend on Claude. A local-model adapter (for example LM Studio) could be added and compared with the same evals.
9. **Research and answer are separate calls.** The agent's tool loop only gathers passages; the answer is then the same structured-output call RAG makes. That keeps the comparison with RAG about retrieval alone, and keeps tool use and structured output in separate requests.

---

## 6. Tech stack

| Concern | Choice |
|---|---|
| Language | Python 3.12, `uv`, ruff, pyright, pytest |
| Corpus collection | Browser "Save page as" + `licn-ingest` (BeautifulSoup/lxml parser) |
| API | FastAPI with streaming answers; Pydantic v2 models at every boundary |
| Storage | **SQLite + sqlite-vec + FTS5**: a single index file, versioned in Cloud Storage. It sits behind a `Retriever` interface, so Postgres + pgvector can be swapped in if the corpus grows. |
| ML | `sentence-transformers` + CPU PyTorch, models pinned to float32 (Qwen3-Embedding-0.6B). ONNX Runtime is an option for a smaller container. |
| LLM | `anthropic` SDK (1.x): Messages API, structured outputs, prompt caching, tool use (agent pipeline) |
| Frontend | **Next.js (App Router) + TypeScript + Tailwind**, mobile-first; **Auth.js** with Google sign-in and an email allowlist; streams answers from the API |
| Agent loop | Thin custom code (no LangChain). A few hundred readable lines, easier to debug and to explain in interviews. |
| Evals | Custom harness (`licn-eval`): deterministic metrics + Haiku LLM judge; runs stored under `evals/runs/` |
| Observability | OpenTelemetry → Cloud Trace / Cloud Logging (optional: Langfuse Cloud free tier) |
| Infrastructure | **Terraform** for all GCP resources (Cloud Run services and job, Scheduler, GCS, Firestore, Secret Manager, Artifact Registry, IAM, budget alerts) |
| Local dev | Docker Compose that mirrors production (web + api + a local GCS emulator or plain filesystem) |
| CI/CD | GitHub Actions: lint, type check, unit tests, and a **fast eval subset that fails the build if citation accuracy or Recall@8 drops**. On merge to `main`, build images and deploy to Cloud Run through Workload Identity Federation. |

---

## 7. Evaluation plan (the resume centerpiece)

**Eval set (built).** 220 questions, in three tiers so work isn't blocked on her time:

1. **Draft questions (done):** Haiku generated questions with [evals/prompts/generate_questions.md](evals/prompts/generate_questions.md), in batches by category. These are **questions only, with no answers**. Without seeing the policies, an LLM would invent requirements. Merged into `evals/data/draft_questions.jsonl`.
2. **Silver answer keys (done):** `licn-silver` gives each question plus the **entire** corpus (about 47k tokens, prompt-cached) to a model that writes:
   - the expected verdict and a reference answer
   - cited key points
   - false premises in the question and facts to ask for
   - the parts the policies don't cover, and a re-labeled question type

   Every citation is validated against the corpus. 6 keys came from Sonnet 5.5 (pilot) and 214 from Haiku 5.5, chosen for cost. Total cost was $0.94. All are marked `review_status: "silver"` in `evals/data/silver_answers.jsonl`.
3. **Gold (her review, later):** she spot-checks silver items and marks each `gold`, `fixed`, or `rejected`, starting with the 18 low-confidence keys. Headline metrics will then be reported on gold items.

**Dev/test split.** `evals/data/splits.json` fixes 54 dev and 166 test questions, stratified by question type (hash of the id, so it's deterministic). Prompts and tools are tuned against dev only; reported numbers come from test. Cheap tuning loop: `licn-eval run --replay <agent run> --split dev` re-runs only the answer step on the passages a saved run collected (no retrieval variance, about a third of the cost); `licn-eval rejudge <run>` grades saved answers again. Every run records a fingerprint of the prompts, the split, and a label.

**Judge (v2, rebuilt October 11).** A review found the first judge unreliable: cited passages were cut at 6,000 characters (the exam table is longer), list items were shown without their lead-in, the policy header (effective date) was hidden, and its single holistic label flipped on 17% of identical answers. v2 shows the full passage with the lead-in of each parent section and the policy header, and asks for smaller judgments: each key point's coverage and whether it is *central* to the bottom line, whether the bottom line agrees with the key, material errors (wrong and would change what the analyst does) separate from minor issues, and per-claim support. Correct / partial / incorrect is computed in code. Validation on 54 dev answers graded twice: agreement with a 24-answer hand-graded reference set (`evals/data/judge_reference.jsonl`, graded by Claude Opus 5.5) rose from 11–14/24 to 20/24; aggregate "correct" was identical across passes, but about 15% of per-question labels still flip (mostly agrees vs partly-agrees on hedged answers), so **compare aggregates, never single questions**.

**Known limitations:** Haiku writes most keys, answers the questions, and judges the answers, which risks rating its own mistakes as correct, and the reference set was graded by another Claude model. Comparisons between pipelines are still fair, since every pipeline faces the same keys and judge. Before the headline re-test: grade the 24 references with Sonnet once (`licn-eval rejudge <run> --judge-model claude-sonnet-5-5 --ids ...`, about $0.50) to check for self-preference, and have her spot-check the reference labels and the low-confidence keys.

Planned mix (actual after re-labeling: eligibility 82, unanswerable 50, missing facts 36, lookup 33, renewal math 13, temporal 6):

| Type | Share | Example |
|---|---|---|
| `lookup` | 20% | "How long is a Residency License valid?" |
| `eligibility` | **40%** | "Has a BA in English, a passing Praxis score, and 2 years on an RL. Can they move to an IPL?" → expected verdict + expected requirement rows + citations |
| `renewal_math` | 15% | "Has 5.5 CEUs, 1.0 of them in literacy. Are they done?" → expected shortfall |
| `missing_facts` | 10% | Scenarios that leave out a deciding fact → the system must ask, not guess |
| `unanswerable` | 10% | Questions outside the LICN policies → the system must abstain |
| `temporal` | 5% | Questions where a pending revision exists → it must not be cited as current |

**Metrics:**
| Layer | Metric | Target (v1) |
|---|---|---|
| Retrieval | Recall@8 of expected policy sections; MRR | ≥ 0.90 / ≥ 0.75 |
| Citations | Citation precision (cited text supports the claim); citation coverage | ≥ 0.95 / ≥ 0.95 |
| Eligibility | **Verdict accuracy**; **requirement-row accuracy**; correct ❓ on missing facts | ≥ 0.90 / ≥ 0.90 / ≥ 0.90 |
| Safety | Abstention accuracy; pending or superseded text cited as current | ≥ 0.90 / 0 occurrences |
| Ops | p95 latency; cost per question | < 10s; < $0.01 |

**Ablation table** (in the README). Run the same eval set through each pipeline and report the gains. Done so far: BM25 → dense → hybrid → + context notes → ± reranker → k=12 vs k=20, plus the long-context benchmark. Still to come:

`+ agent tools` (measured: README) → `+ calculator/convert` → `+ verifier`. The `agent_nocalc` ablation was deliberately not run: the calculator's problem was that the model chose the wrong formula, which an ablation can't diagnose. Conversion errors are measured directly instead (`licn-eval conversions`), and `convert` took them to zero.

Also run **Haiku vs. Sonnet vs. a local model** on the same evals. "I measured each technique and model on a real policy corpus and chose the cheapest configuration that met the bar" is a strong interview story.

---

## 8. Phased roadmap

**Local first, then cloud.** Everything runs on your PC through Docker Compose until the app is useful (end of Phase 3). Then Phase 4 moves it to GCP. To make that move a configuration change rather than a rewrite, cloud-specific pieces sit behind small interfaces from day one:

| Concern | Local implementation | Cloud implementation |
|---|---|---|
| Blob storage (index, raw HTML) | `./data/` folder | GCS bucket |
| Feedback & cost logs | SQLite table | Firestore |
| Secrets | `.env` file (git-ignored) | Secret Manager |
| Scheduled crawl | `make ingest` / manual run | Cloud Scheduler → Cloud Run Job |
| Auth | Disabled, or a dev login | Auth.js Google sign-in + allowlist |
| Tracing | Console / local Jaeger | Cloud Trace |

### Phase 0: Foundations & local skeleton
- [ ] **Sit down with her for an hour:** collect real (anonymized) scenario questions, plus her answers to them
- [x] Git repo ([GitHub](https://github.com/cpalmquist19/RAGAGENT)) with a Python package `src/licn_rag/` (uv, ruff, pyright, pytest). The planned `web/` and `infra/` folders come with Phases 3–4.
- [ ] Docker Compose: `web` (Next.js) + `api` (FastAPI), with a **"hello world" page calling `/health`** running locally
- [ ] Storage / feedback / secrets interfaces with local implementations (table above)
- [ ] GitHub Actions CI (lint, type check, tests). No deploy yet.
- [x] Generate draft eval questions with Haiku (220 questions)

### Phase 1: Corpus, silver eval set & retrieval ✅
- [x] ~~Playwright crawler~~ → pages saved from a browser + `licn-ingest` parser (bot protection; §2.2)
- [x] Silver answer keys for all 220 questions, citations validated (§7). Per-section question generation was skipped for now.
- [x] HTML parser → structured corpus with policy-native citation IDs; structure-aware chunks with tables split by group
- [x] Context notes (Haiku); Qwen3 embeddings; SQLite index (sqlite-vec + FTS5); RRF. Reranker tested and dropped.
- [x] Retrieval evals (Recall@k, MRR) and the first rows of the ablation table
- [x] Measure total corpus tokens (about 47k) and build the long-context benchmark
- [x] End-to-end evals: benchmark vs RAG (k=12, k=20)
- [ ] Header citations → `citation_graph` table. In-text cross-references between LICN passages are now extracted in memory (`agent/references.py`, plan 01); the statutory/NCAC header citations are still not a table.
- [ ] Package ingestion as a **Cloud Run Job** that publishes `index-<date>.sqlite` to GCS (Phase 4)

### Phase 2: Agent & eligibility reasoning ⏭ next
- [ ] **Goal: close the gap between RAG (37% correct) and the long-context benchmark (56%)** without giving up retrieval
- [x] Tools (plan 01): `search`, `get_policy_outline`, `get_section` (a bare policy code reads a short policy, which replaces a separate `lookup_policy`), `follow_references`, `list_license_types`, `calculate`
- [x] Tool-use loop pipeline `agent` plus `agent_nocalc` ablation, and `licn-ask --trace` (plan 02), smoke-tested against the real API.
- [x] Evaluate the agent against RAG and the benchmark: context recall, correctness, cost, latency (plan 03). The agent recovers about half the gap; the goal above is not met yet.
- [x] `convert` tool for unit conversions; conversion check in the harness
- [x] Judge review and rebuild (v2); dev/test split; replay and re-judge tooling
- [x] Answer prompt tuning round 1 on dev (v3)
- [ ] Sonnet cross-check of the judge; full re-test of all pipelines on the test split
- [ ] Worksheet output schema + strict citation contract + missing-fact questions + abstention
- [ ] Verifier pass; eligibility, citation, and abstention metrics
- [ ] PII check; OpenTelemetry tracing; per-question cost logging to Firestore; a CLI for quick testing

### Phase 3: Production UI (weeks 6–7)
- [ ] Streaming API endpoint; Next.js mobile-first UI: worksheet view, inline cited passages, Simbli deep links, "as of" and "pending revision" badges, follow-up prompts for missing facts
- [ ] 👍/👎 and "wrong citation" feedback → Firestore → goes back into the golden set
- [ ] **She starts using it for real** (on your home network). Iterate weekly on what she flags.

### Phase 4: Cloud deployment (week 8)
- [ ] GCP project + **billing budget alerts ($1 / $5)** + Terraform state bucket
- [ ] Terraform: Artifact Registry, two Cloud Run services (web, private api), service accounts and IAM, Secret Manager, GCS bucket, Firestore
- [ ] GitHub Actions deploy via Workload Identity Federation; Auth.js Google sign-in with an email allowlist
- [ ] Switch the interfaces to their cloud implementations; deploy; smoke-test against the eval subset

### Phase 5: Freshness & hardening (week 9)
- [ ] Cloud Scheduler → weekly ingest job → eval-gated index publish → "what changed" log (optional email digest)
- [ ] CI eval gate on every PR; uptime check; error alerting; measure cold starts (decide on `min-instances`)
- [ ] Load-test a little and confirm we stay inside the free tier

### Phase 6: Portfolio polish (week 10+)
- [ ] README: problem, architecture and deployment diagrams, ablation table and model comparison charts, cost breakdown ("runs for about $2/month"), security design, limitations
- [ ] 2-minute demo video and a short write-up ("Trustworthy policy RAG on a $2/month serverless stack")
- [ ] Stretch goals:
  - [ ] **AWS port** (Lambda container images + S3 + DynamoDB, with Terraform) to show multi-cloud
  - [ ] Ingest the cited G.S. 115C and 16 NCAC 06C sections as a secondary layer
  - [ ] Side-by-side policy version diffs
  - [ ] Saved anonymous "person profiles" for repeat scenarios

---

## 9. Risks & guardrails

| Risk | Mitigation |
|---|---|
| Wrong eligibility verdict | Separate facts from rules; ❓ for missing facts; arithmetic in code; per-row citations; verifier; eval category with ≥ 0.90 target |
| Pending or outdated text treated as current | Store pending and current separately; weekly re-crawl; "as of" date shown; temporal evals |
| Over-reliance | Sources shown alongside every row; persistent "not an official determination, confirm with DPI/HR" note |
| Privacy / unknown district AI policy | Public corpus only; PII warning; no question text stored by default; generic scenarios only |
| Simbli page layout changes break the parser | Parser tests on real quirks; `licn-ingest` reports warnings and failures per policy; the last good corpus and index stay in place until a re-ingest passes the evals |
| Policies change and nobody re-saves them | Re-save after the SBE's July updates; content hashes in `manifest.json` show what changed; "as of" date shown on answers |
| Surprise cloud bill | Budget alerts at $1 / $5; scale-to-zero everywhere; Cloud Run max-instances capped at 2; no always-on VMs or Cloud SQL |
| Cold-start latency | float32 Qwen3 embedder (or ONNX int8 for a smaller image); index loaded lazily and cached in memory; measure, then decide on `min-instances=1` |
| Public endpoint abuse | Google sign-in allowlist; private API (IAM-only); per-user rate limit; Anthropic workspace spend limit |
| Cost creep | Haiku by default; prompt caching; local embeddings; cost logged per question and per eval run |

---

## 10. Remaining open questions
1. ~~Golden set time~~ → **Haiku drafts the questions now. Grounded silver answers come after the scrape. She reviews to gold when she has time.**
2. ~~Interface~~ → **Next.js from day one.**
3. ~~Access~~ → **GCP Cloud Run, free tier.**
4. ~~Answering model / scraper~~ → **Haiku 5.5 / pages saved from a browser** (bot protection blocks automated scraping).
5. **Open:** is LICN-014 (Provisional License) still in force? It's cited by three policies but wasn't available to save.
6. **Partly answered:** the judge was rebuilt and validated against a model-graded reference set (§7). Still open: a one-time Sonnet cross-check (~$0.50) and her spot-check of the reference labels before headline numbers are published.
7. **Open:** does the agent replace RAG as the production pipeline? Decide after the test-split re-test.

---

## Sources
- Magesh et al., *Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools*: https://arxiv.org/pdf/2405.20362
- *Benchmarking Legal RAG: The Promise and Limits of AI Statutory Surveys* (CSLAW '26): https://dho.stanford.edu/wp-content/uploads/Stat_Surveys.pdf
- Garsia, *RAG Is Judgment* (SSRN): https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6945200
- *Structural, Temporal, and Causal Limitations of RAG in the Legal Domain*: https://arxiv.org/pdf/2606.09724
- LegalBench-RAG: https://www.emergentmind.com/topics/legalbench-rag
- Citation-enforced prompting study (MDPI): https://www.mdpi.com/2076-3417/16/6/3013
- Agentic retrieval preprints: https://arxiv.org/pdf/2605.05242 · https://arxiv.org/pdf/2605.05538
- Contextual + hybrid + rerank reference implementation: https://github.com/chatterjeesaurabh/Contextual-RAG-System-with-Hybrid-Search-and-Reranking
- Qwen3 Embedding / Reranker: https://arxiv.org/pdf/2506.05176 · model roundups: https://www.cognee.ai/best-embedding-models · https://builderai.tools/blog/best-open-source-embedding-models-2026
- Structure-aware parsing & chunking: https://arxiv.org/pdf/2604.12047
- Claude search result blocks / citations: https://platform.claude.com/docs/en/build-with-claude/search-results
- RAG evaluation tooling: https://blog.premai.io/rag-evaluation-metrics-frameworks-testing-2026
- Long context vs. RAG vs. GraphRAG: https://ayinedjimi-consultants.fr/articles/long-context-rag-graphrag-comparatif-2026
- GCP free tier overview (verify at cloud.google.com/free): https://freetier.co/gcp-free-tier · AWS free tier changes (July 2025): https://infratally.com/articles/aws-free-tier-2026/
- NC SBE Policy Manual (LICN series): https://simbli.eboardsolutions.com/Policy/PolicyListing.aspx?S=10399&ptid=muNUlKiR2jsXcslsh28JpBkiw==

*Many 2026 RAG write-ups are vendor blogs, and several of their headline percentages couldn't be traced to primary studies. This plan leans on academic sources where they exist, and on our own eval harness for everything else.*
