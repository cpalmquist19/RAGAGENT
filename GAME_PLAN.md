# Licensure RAG Assistant: Research Summary & Game Plan

*Prepared October 10, 2026 · Revision 3: scoped to the NC State Board of Education **LICN** policy series, a minimum-cost budget, scenario-based eligibility questions, **serverless hosting on Google Cloud's free tier**, and a **Next.js** frontend from day one*

**Decisions locked in:**
- **Hosting:** GCP Cloud Run on the always-free tier, managed with Terraform
- **Frontend:** Next.js
- **Answering model:** Claude Haiku 5.5
- **Scraper:** Playwright
- **Search index:** SQLite + sqlite-vec, stored as a file in Cloud Storage

**Goal:** Build an assistant that answers NC educator-licensure questions using only the **State Board of Education Licensure (LICN) policies**. Most questions will be hypothetical scenarios: *"Someone has X degree and Y license with Z years. What do they qualify for? Do they need more renewal credits?"* Each answer must cite the exact policy and section it relies on, show the user what facts it assumed, and say so plainly when the policies don't settle the question.

**Two audiences, one design:**
1. **Your wife (the real user):** she needs answers she can trust and check in one click.
2. **Hiring managers (the resume):** they want to see a modern architecture, measured quality (evals with numbers), and good engineering habits (tests, CI, observability, a clean deployment).

**Constraint: as cheap as possible.** The target is under $5/month in total. §4 shows how.

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
| **Reranking** | A cross-encoder reranker over the top ~50 fused results, keeping the best ~8 | Cheap (it runs locally), and consistently among the largest single quality gains. |
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
- **Long-context-only:** the whole LICN series may well fit in one prompt (see §2). We'll build it as a **baseline**, and if it ties RAG on our evals, that's an honest and interesting result for the write-up. RAG should still win on citation precision, cost per question, and latency.

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
- **Option B is still open:** asking SBE/DPI or eBoard Solutions for a bulk export or permission to automate. If that's granted, the Playwright crawler can come back with an honest user agent.
- **Actual size (October 10, 2026):** **24 policies, about 15,200 words, about 21k tokens.** That's far smaller than the original estimate.
  - LICN-006 (salary credit) and LICN-008 (fees, dated 2018) use an older page format, which the parser also handles.
  - LICN-014 (Provisional License) is cited by LICN-001, -010, and -011 but has not been saved.
  - Known errors in the source pages: LICN-006 numbers two sections "IV"; LICN-003 cites "GS 270.15" and is missing the PLT score.

> **Design implication:** the whole corpus fits in a single prompt for about $0.002–0.003 with Haiku, or less with prompt caching. The long-context baseline (§1.3) is now a serious contender, not a straw man. The evals will decide whether retrieval beats it on citation precision, latency, and cost. Whichever wins, the ablation table will show the comparison honestly.

### 2.3 Chunking
- Parse the policy body HTML into sections using its headings and numbered lists. **One chunk = one section or subsection, plus its heading path.** Tables (license types, requirement grids) stay whole and are converted to Markdown, so the model reads them as tables.
- Each chunk stores `policy_code`, `title`, `section_path`, `effective_date`, `last_revised`, `status`, `source_url`, and `statutory_citations[]`.
- **Deep links:** each citation opens the official Simbli page and uses a URL text fragment (`#:~:text=…`) to scroll to and highlight the cited passage. We can also show the cached passage inline, so it works even if the site changes.

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
- **Arithmetic is done in code, not by the LLM.** Credit hours, years, and deadlines go through a small `calculate` tool. LLMs make arithmetic slips, and this removes them.
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
| Fit for this app | Cloud Run runs ordinary **Docker containers** (FastAPI, Next.js, Playwright) and **scales to zero**. This is the simplest serverless path, and the same containers run locally. | Doable with Lambda container images + DynamoDB + S3 + CloudFront, but with more glue (API Gateway, Lambda size limits, cold starts with ML models). |
| Resume value | Cloud Run, Terraform, Workload Identity Federation, Secret Manager, IAM service-to-service auth | Equivalent, and AWS appears in more job postings |

**Recommendation: GCP**, because its free tier doesn't expire and Cloud Run matches our container-based design. Everything is in Terraform and Docker, so an AWS port later is a realistic stretch goal, and "deployed on both" is a good resume line.

> The free-tier figures above come from third-party 2026 guides. Confirm them at cloud.google.com/free when setting up. **Set a billing budget alert at $1 and $5 on day one.**

### 4.2 Cost by component
| Component | Choice | Cost |
|---|---|---|
| Web frontend | **Next.js** on Cloud Run (standalone output, scales to zero) | **$0** (free tier) |
| API | **FastAPI** on Cloud Run (scales to zero, 2 GiB memory) | **$0** (free tier) |
| Search index | **SQLite + `sqlite-vec` (vectors) + FTS5 (keyword/BM25)** in a single file. It is built by the ingestion job, uploaded to Cloud Storage, and loaded by the API at startup. The corpus is small and read-only between crawls, so a database server would be wasted money. *(Cloud SQL Postgres is not free.)* | **$0** |
| Query embeddings | **Qwen3-Embedding-0.6B**, ONNX int8, running inside the API container. Corpus embeddings are computed offline at ingestion. | **$0** |
| Reranking | **Default: Haiku as a listwise reranker** (top ~30 → best 8, about $0.001/question), which avoids a second large model in the container. *Ablation:* a local cross-encoder (bge-reranker-v2-m3), so the evals decide. | **~$0.001/question** |
| Bulk LLM work (chunk context headers, verification, eval judging) | **Claude Haiku 5.5**: $0.10 / $0.50 per M tokens (input/output). Use the **Batch API (–50%)** for ingestion and evals. | Ingesting the whole corpus once: **~$0.05** |
| Answering | **Claude Haiku 5.5** (decided). Sonnet 5.5 stays an eval comparison only. | ~15k input + 1.5k output ≈ **$0.002/question** |
| Weekly crawl | **Cloud Run Job** (Playwright container) triggered by **Cloud Scheduler** | **$0** (free tier) |
| Feedback & usage logs | **Firestore** (thumbs up/down, flagged citations, per-question cost). No question text stored by default. | **$0** |
| Secrets | **Secret Manager** (Anthropic API key) | **$0** (free tier) |
| Observability | **OpenTelemetry → Cloud Trace + Cloud Logging** (free allotments). Langfuse Cloud's free hobby tier is optional for LLM-specific trace views. | **$0** |
| Container images | **Artifact Registry**. Images with an ML model plus Playwright exceed the 0.5 GB free allowance. | **~$0.10–0.30/month** |
| Local-model mode | LM Studio adapter, for dev runs and eval comparisons only | **$0** |

**Realistic monthly cost:** about $0.30 for image storage + under $1 in Haiku calls + about $0.10–0.50 per full eval run. **Total: about $1–3/month.**

Prompt caching helps a lot here: the system prompt, tool definitions, and frequently retrieved policies are identical across questions, so cache reads cost a fraction of normal input.

**Cold starts:** with scale-to-zero, the first question after an idle period may take about 5–15 seconds while the container loads the embedding model and index. Later questions are fast. If this bothers her, setting `min-instances=1` on the API removes the delay, but that probably costs a few dollars a month. We'll measure before deciding.

---

## 5. Target architecture

```
  INGESTION (weekly, offline)
  Playwright crawler ──► LICN policy list ──► each ViewPolicy page (current + pending, kept separate)
        │  rendered HTML + header metadata (dates, status, G.S./NCAC citations) + hash
        ▼
  HTML section parser ──► chunks (one per section/subsection, tables as Markdown, heading path)
        │
        ├─► contextual header per chunk (Haiku, Batch API, prompt-cached per policy)
        ├─► local embeddings (Qwen3-Embedding) + full-text index
        ├─► cross-references: policy ↔ policy, policy ↔ G.S./NCAC (from headers + regex on body)
        └─► change log: diff vs previous version ("LICN-003 §2 changed on 07/01/2026")

  QUERY (online)
  Question ──► PII check ──► Claude agent loop (Haiku 5.5, max ~6 tool calls)
        ├─ lookup_policy("LICN-001")            exact fetch, no embedding guesswork
        ├─ search(query, filters)               hybrid (dense + BM25) → RRF → rerank → top 8
        ├─ get_section(id, with_children=True)  read the whole section, not a fragment
        ├─ list_license_types()                 the license catalog, so it reasons over the right options
        └─ calculate(expression)                credits, years, deadlines computed in code
        (search tools return `search_result` blocks → native citations with cited_text)
        │
        ▼
  Structured answer: facts used • assumptions/missing facts • requirement rows (each cited) • bottom line
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
  │ Google sign-in,      │   may invoke the API)        │ ONNX embedder + index.sqlite  │
  │ email allowlist      │                              │ (loaded from GCS at startup)  │
  └──────────────────────┘                              └──────┬──────────────┬─────────┘
                                                               │              │
                                     Secret Manager (API key) ─┘              ├─► Anthropic API (Haiku 5.5)
                                                                              └─► Firestore (feedback, cost logs)
  Cloud Scheduler (weekly)
          │
          ▼
  Cloud Run Job: ingest ──► Playwright crawl LICN ──► parse/chunk/embed ──► run retrieval eval smoke test
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
4. **Native citations through Claude's `search_result` blocks.** Claude cites whole text blocks and returns `cited_text` (which doesn't count as billed output tokens). We make each block one paragraph or list item, so citations are precise.
5. **A strict citation contract** in the system prompt:
   - Use only the provided policy text.
   - Every requirement and claim needs a citation.
   - If a fact is missing, ask for it rather than assume it.
   - If the policies don't answer the question, say so.
6. **Verification pass** to catch citations that don't support their claims.
7. **Time-awareness.** Answers use current adopted text by default, pending revisions are flagged rather than mixed in, and the "as of" date is always shown.
8. **Model-agnostic core.** Retrieval, chunking, evals, and the worksheet schema don't depend on Claude. We'll have a Claude adapter (native citations) and an LM Studio adapter (prompted `[S#]` citations validated by regex). The evals decide which model answers.

---

## 6. Tech stack

| Concern | Choice |
|---|---|
| Language | Python 3.12, `uv`, ruff, pyright, pytest |
| Scraping | Playwright (Python) |
| API | FastAPI with streaming answers; Pydantic v2 models at every boundary |
| Storage | **SQLite + sqlite-vec + FTS5**: a single index file, versioned in Cloud Storage. It sits behind a `Retriever` interface, so Postgres + pgvector can be swapped in if the corpus grows. |
| ML | `sentence-transformers` for offline corpus embedding; **ONNX Runtime** (int8) for query-time embedding in the API container |
| LLM | `anthropic` SDK: Messages API, tool use, prompt caching, `search_result` citations, Batch API |
| Frontend | **Next.js (App Router) + TypeScript + Tailwind**, mobile-first; **Auth.js** with Google sign-in and an email allowlist; streams answers from the API |
| Agent loop | Thin custom code (no LangChain). A few hundred readable lines, easier to debug and to explain in interviews. |
| Evals | pytest-based harness + RAGAS faithfulness; results stored as JSON, with charts in the README |
| Observability | OpenTelemetry → Cloud Trace / Cloud Logging (optional: Langfuse Cloud free tier) |
| Infrastructure | **Terraform** for all GCP resources (Cloud Run services and job, Scheduler, GCS, Firestore, Secret Manager, Artifact Registry, IAM, budget alerts) |
| Local dev | Docker Compose that mirrors production (web + api + a local GCS emulator or plain filesystem) |
| CI/CD | GitHub Actions: lint, type check, unit tests, and a **fast eval subset that fails the build if citation accuracy or Recall@8 drops**. On merge to `main`, build images and deploy to Cloud Run through Workload Identity Federation. |

---

## 7. Evaluation plan (the resume centerpiece)

**Golden dataset (`evals/golden.jsonl`).** Start with about 60 items and grow to 150+. It's built in three tiers, so we aren't blocked on her time:

1. **Draft questions (now, ungrounded):** Haiku generates questions with [evals/prompts/generate_questions.md](evals/prompts/generate_questions.md). These are **questions only, with no answers**. Without seeing the policies, an LLM would invent requirements.
2. **Silver answers (after the scrape, grounded):**
   - A script gives each draft question plus the relevant LICN policy text to a stronger model (Sonnet or Opus, run once through the Batch API for pennies). It drafts the expected verdict, requirement rows, and citations.
   - It also generates **new** questions *from* each policy section, so every section is covered.
   - Questions the policies can't answer are relabeled `unanswerable`.
   - Every item is marked `review_status: "silver"`.
3. **Gold (her review):** she spot-checks silver items when she has time and marks each `gold`, `fixed`, or `rejected`. Headline metrics are reported on gold items. Silver-only metrics are labeled as such.

Using a different, stronger model than the one under test to draft expected answers reduces "grading its own homework" bias. Her review is still what makes the numbers credible.

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

**Ablation table** (in the README). Run the same golden set through each pipeline and report the gains:

`long-context baseline` → `naive chunks + dense` → `+ section chunks` → `+ contextual headers` → `+ hybrid/BM25` → `+ reranker` → `+ agent tools` → `+ worksheet schema & calculator` → `+ verifier`

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

### Phase 0: Foundations & local skeleton (week 1)
- [ ] **Sit down with her for an hour:** collect 30–50 real (anonymized) scenario questions, plus her answers to them
- [ ] `git init` as a monorepo: `ingest/`, `retrieval/`, `agent/`, `api/` (Python, uv), `web/` (Next.js), `evals/`, `infra/` (Terraform). Add ruff, pyright, eslint, pre-commit, and Docker Compose for local dev.
- [ ] Docker Compose: `web` (Next.js) + `api` (FastAPI), with a **"hello world" page calling `/health`** running locally
- [ ] Storage / feedback / secrets interfaces with local implementations (table above)
- [ ] GitHub Actions CI (lint, type check, tests). No deploy yet.
- [ ] Generate draft eval questions with Haiku (tier 1 above)

### Phase 1: Corpus, silver eval set & retrieval (weeks 2–3)
- [ ] Playwright crawler: get the LICN list, then each policy (current and pending separately), with metadata and hashes saved
- [ ] Grounded silver-answer generation for the draft questions + per-section question generation (§7)
- [ ] HTML section parser → chunks; tables converted to Markdown; header citations → `citation_graph` table
- [ ] Contextual headers (Haiku batch); embeddings; SQLite index (sqlite-vec + FTS5); RRF; reranker (Haiku listwise, with a local cross-encoder as an ablation)
- [ ] Retrieval evals (Recall@k, MRR) running, with the first rows of the ablation table
- [ ] Measure total corpus tokens and build the long-context baseline
- [ ] Package ingestion as a **Cloud Run Job** that publishes `index-<date>.sqlite` to GCS

### Phase 2: Agent & eligibility reasoning (weeks 4–5)
- [ ] Tools: `lookup_policy`, `search`, `get_section`, `list_license_types`, `calculate`
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
| Simbli site changes break the scraper | Raw HTML archived in GCS; the job fails loudly with an alert; the last good index stays live (eval-gated publishing) |
| Surprise cloud bill | Budget alerts at $1 / $5; scale-to-zero everywhere; Cloud Run max-instances capped at 2; no always-on VMs or Cloud SQL |
| Cold-start latency | Small ONNX embedder; index loaded lazily and cached in memory; measure, then decide on `min-instances=1` |
| Public endpoint abuse | Google sign-in allowlist; private API (IAM-only); per-user rate limit; Anthropic workspace spend limit |
| Cost creep | Haiku by default; Batch API; prompt caching; cost logged per question |

---

## 10. Remaining open questions
1. ~~Golden set time~~ → **Haiku drafts the questions now. Grounded silver answers come after the scrape. She reviews to gold when she has time.**
2. ~~Interface~~ → **Next.js from day one.**
3. ~~Access~~ → **GCP Cloud Run, free tier.**
4. ~~Answering model / scraper~~ → **Haiku 5.5 / Playwright.**

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
