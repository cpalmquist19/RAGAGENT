# Plan 03 log: calculator / unit-conversion fix (Opus 5.5), 2026-10-11

Budget: $3.00 max on API calls. Running total is kept at the bottom of each entry.

## 1. Diagnosis (no spend)

Read README, `SESSION_REPORT_plan-03.md`, `agent/calculator.py`, `agent/tools.py`, `agent/prompt.py`, `pipelines/agent.py`.

**Root cause found: the `calculate` tool's own schema teaches the wrong conversion.**
`CalculateInput` in `src/licn_rag/agent/tools.py` gives the model this example:

```
description: "CEUs from 6 semester hours"
expression:  "6 * 2/3"        -> 4   (correct is 6 / (2/3) = 9)
```

Every research call sends this schema, so every time the model reaches for the calculator to convert
hours to CEUs, the example nudges it to multiply. That explains D2 ("models write conversions backwards,
including inside calculate") for both Haiku and Sonnet better than model capability does. The unit test
`test_calculate_records` asserts the same wrong pair (as plain arithmetic it's "correct", so nobody noticed).

Second problem: the direction of a conversion is the thing models get wrong, and a free-form calculator
can't catch it. A `convert` tool that takes named units (`6 semester_hour -> ceu`) and owns the factor
removes the direction choice from the model entirely.

Conversion rule in the corpus (only one): LICN-001 §2(g): 1 CEU = 2/3 semester credit hour = 10 clock hours.

**Test set.** n=7 was too small. I scanned all 220 answer keys for conversion math and picked the 16 whose
key computation converts units: renewal_math-001, 002, 003, 005, 007, 009, 010, 011, 012, 013, 021, 022,
027, 029, 030, missing_facts-005. I spot-checked the keys' directions; all correct (003 was hand-fixed earlier).

Spend so far: $0.00

## 2. Code changes (no spend)

- **New `convert` tool** (`src/licn_rag/agent/units.py` + `tools.py`). Input: `value`, `from_unit`, `to_unit`,
  each unit one of `ceu`, `semester_hour`, `clock_hour`. The code owns the factor from LICN-001 §2(g) and
  shows its working: `6 semester credit hours = 9 CEUs (1 CEU = 2/3 semester credit hour, so CEUs = 6 / (2/3))`.
  It also adds LICN-001 §2(g), (g)(i), (g)(ii) to the evidence so the answer step can cite the rule, and the
  result goes into the `<calculations>` block like calculator results do.
  Semester hours <-> clock hours is refused: the policy gives no such conversion (they are two separate
  ways to earn a CEU), so I didn't let the tool invent one.
- **Fixed the `calculate` schema.** Example is now `8 - 6.5` ("CEUs still needed"); description says
  conversions go through `convert`.
- **Guard in `calculate`**: an expression with a hand-written `2/3`, `3/2`, or `0.66/0.67` factor is refused
  with "use convert". Word boundaries so `12/3`, `16/3`, `13/2` still work. This is deterministic, so the
  calculator can no longer launder a backwards conversion into a "verified" number (that was D3).
- **Research prompt step 4**: conversions between CEUs / semester hours / clock hours go through `convert`
  (including obvious ones; ambiguous "credits" -> convert each reading); everything else through `calculate`.
- `agent_nocalc` now hides both math tools.
- Tests: fixed the test that asserted `6 * 2/3 = 4` for "CEUs from 6 semester hours"; added `tests/test_units.py`
  and tests for the guard and the new tool. 128 pass; ruff + pyright clean.

Not touched (plan rules): shared answer prompt/schema, answer keys, judge model.

Spend so far: $0.00

## 3. Deterministic conversion check + first test run ($0.10)

Wrote a checker (scratchpad script, not committed yet) that looks for the **right** converted value and the
**backwards** value in each answer, per question, with hand-written regexes from the answer keys. Fixed two
false positives in it (e.g. "4/3 semester hours" matched "3 semester"). Spot-checked every "backwards" hit by hand.

Run `20261011-195430-agent`: Haiku 5.5, new tool + prompt, the 16 conversion questions.

| Run (same question ids) | Backwards conversion in answer | Right value stated | Used a math tool | Judge correct |
|---|---|---|---|---|
| Full run `182005` (old prompt) | 3/16 | 10/16 | 5/16 | 4/16 |
| Checklist prompt `185706` (15 of the 16) | 3/15 | 8/15 | 13/15 | 6/15 |
| Haiku smoke `192739` (7) | 4/7 | 2/7 | 6/7 | 1/7 |
| Sonnet smoke `192907` (7) | 2/7 | 5/7 | 5/7 | 1/7 |
| **convert tool `195430`** | **0/16** | **12/16** | 10/16 (convert on 10) | 6/16 |

- All 13 `convert` calls were correct (by construction). Example: `6 semester credit hours = 9 CEUs (... CEUs = 6 / (2/3))`.
- 012 and 022, the two questions that flipped the bottom line earlier, are now right (9 and 18 CEUs).
- No calculator guard refusals fired (the model never even tried a hand-written 2/3).
- Judge "correct" didn't move meaningfully (6/16 vs 4/16 and 6/15; noise at this n). The judge grades the whole
  answer, and most of these answers were already "partially correct" for non-math reasons (missing exceptions etc.).
- Remaining: 6/16 used no math tool. 027, 029, mf-005 still stated the right value (the answer step did it);
  001, 013, 030 stated no conversion at all (not an error, the key just includes it as extra detail).
  002 converted the wrong *quantity* (read "short 2 credits" as 2 semester hours rather than a 2-CEU shortfall). That's an
  interpretation issue, not direction.
- Cosmetic bug spotted: tool says "1 CEU = 10 clock hour" (singular). Fixing.

Cost: pipeline $0.08 + judge $0.02 = $0.10. **Spend so far: $0.10**

## 4. Conversion check is now a harness command (no spend)

Made the checker generic and put it in the harness: `uv run licn-eval conversions [-v] <run dirs>`
(`src/licn_rag/evals/conversions.py`, 21 new tests). It finds conversion statements in any saved answer
("6 semester hours equal 9 CEUs", "eight CEUs = 80 clock hours") and arithmetic ("11.5 × 2/3 = 7.67 CEUs",
where the operator itself shows the direction), checks them against LICN-001 §2(g), and labels each
ok / backwards / wrong. Free and retroactive, so it works on every old run. It skips rates ("10 clock hours
per CEU"), negations, same-unit statements, and semester<->clock. It's deliberately narrow: what it flags has
been real every time I checked by hand (7/7 below), but it misses some phrasings (e.g. full-run 022 "they
convert to 8 CEUs" has no unit on the left), so treat error counts as a **lower bound**.

| Run | Answers | With a detected conversion | ok | backwards | Answers with an error |
|---|---|---|---|---|---|
| Long-context benchmark (220) | 220 | 19 | 22 | **2** | 2 |
| RAG k=20 (220) | 220 | 13 | 21 | **1** | 1 |
| Agent full run (220, old prompt) | 220 | 16 | 21 | **3** | 3 |
| Agent checklist prompt (30) | 30 | 13 | 22 | **1** | 1 |
| Agent + convert (16) | 16 | 11 | 18 | **0** | 0 |

Takeaway: **the long-context benchmark makes the same backwards error** (it has the whole corpus), so this is a
model reasoning problem, not retrieval. That's why it needs a tool that owns the direction, not more context or a
bigger model.

Also: added "questions using calculate/convert" and "convert calls" rows to `licn-eval report`; fixed
"1 CEU = 10 clock hour" wording. 149 tests pass; ruff + pyright clean.

Note: plan 03 says "only one tuning round (already used)". You asked me to optimize the prompt, so I'm treating
that as overridden; flagging it so the docs can say so.

Spend so far: $0.10. Next: full 220-question agent run with the new code (~$1.30 estimate).

## 5. Full 220-question run with convert ($1.35)

Run `20261011-200047-agent` (Haiku 5.5 answering and judging, same as all earlier runs).

| | Long-context | RAG k=20 | Agent, old (`182005`) | **Agent + convert (`200047`)** |
|---|---|---|---|---|
| Correct (judge) | 56% | 37% | 46% | **47%** |
| Key-point coverage | 81% | 69% | 76% | 77% |
| Answers with unsupported claim | 18% | 27% | 24% | 22% |
| Answers contradicting key | 3% | 7% | 3% | 2% |
| Verdict matches key | 80% | 82% | 80% | 81% |
| Key sections in context | – | – | 78% | 80% |
| **Backwards conversions (`licn-eval conversions`)** | 2 | 1 | 3 | **0** |
| Questions using calculate/convert | – | – | 6 | 21 |
| Cost / question | $0.0018 | $0.0024 | $0.0044 | $0.0047 |
| Latency p50 / p95 | 12.1 / 19.4s | 10.7 / 16.3s | 16.6 / 25.6s | 17.3 / 28.0s |

- **The defect is fixed where it occurs:** 0 backwards conversions in 220 answers (26 conversions detected, all
  correct), and 0/16 on the conversion set in both runs that used `convert`. Before: 3/16 to 4/7 depending on run.
- On the 16 conversion questions: judge correct 10/16 (old full run: 4/16). renewal_math-* ids: 16/30 vs 15/30.
- **Overall correctness is flat** (47% vs 46%; paired 36 wins vs 35 losses). Expected: conversions touch about 16 of
  220 questions, and most of those were already "partially correct" for other reasons. No regression on the other types.
- Costs: +$0.0003/question and +0.7s p50 (more tool calls: 5.3 vs 4.5).
- 002 looked like a miss in my per-question script but isn't: it converted every reading correctly (2 sem hrs = 3 CEUs;
  2 CEUs = 1.33 sem hrs), it just led with a different reading than the key. "Convert each plausible reading" works.
- Caveats: still Haiku judging Haiku; the conversion checker is a lower bound (narrow parser); the old runs were judged
  against the old renewal_math-003 key (1 question).

**Spend so far: $1.45** of $3.00 (test run $0.10 + full run $1.35).

## 6. Where this leaves things / recommendations

**Stopping here on spend ($1.45 used, $1.55 left on purpose).** I didn't run Sonnet again. Now that the code owns
the conversion direction, model size doesn't matter for this error. Sonnet was 10x the cost and still wrote 2/7 conversions backwards
inside the calculator. Haiku + `convert` beats both earlier setups on this.

What changed (all uncommitted, on `plan-03-agent-evaluation`; I didn't commit, that's your call):
- `src/licn_rag/agent/units.py` (new): the conversion logic.
- `src/licn_rag/agent/tools.py`: `convert` tool, fixed `calculate` example, refusal of hand-written 2/3 factors.
- `src/licn_rag/agent/prompt.py`: step 4 routes conversions to `convert`.
- `src/licn_rag/evals/conversions.py` (new) + `licn-eval conversions` subcommand; two new rows in `licn-eval report`.
- Tests: `tests/test_units.py`, `tests/test_conversions.py` (new), updated agent tool/pipeline/calculator tests. 149 pass.
- New run dirs: `evals/runs/20261011-195430-agent` (16 q), `evals/runs/20261011-200047-agent` (full 220).

How this answers the session report's open decisions:
1. **D2 (backwards conversions): fixed** with the `convert` tool, and D3 too (calculator can't launder a bad
   expression anymore). Measured 0 backwards in 220.
2. **Checklist prompt: keep it** (modified). The new full run uses it, so the headline numbers no longer predate it.
   **Use `200047` as the agent column** in README/GAME_PLAN instead of `182005`.
3. **6c ablation: still skip.** If the docs want it, `agent_nocalc` now removes both math tools.

Suggested next steps (not done):
- Step 7 docs with the `200047` numbers; mention `licn-eval conversions` and that long-context also made backwards
  conversions (a strong point for the tool approach in the README's "why this is harder" story).
- Update `SESSION_REPORT_plan-03.md` D1-D3 status and plan 02's stale "in progress" status (still pending from before).
- The RAG and long-context runs don't need re-running: `licn-eval conversions` already scores them for free.
- Possible further work: tool usage outside renewal math is still low (8 of 190 other questions use a math tool),
  but the checker found no conversion errors there, so I'd not push the prompt harder without evidence of harm.

---

# Session 2: docs + prompt tuning (budget $3.00, separate from session 1)

## 7. Why prompt tuning was locked, and what replaces the lock

The lock came from plans 02/03 and had two reasons:
1. **Fair comparisons.** All pipelines share one answer prompt, so RAG vs agent vs long-context measures retrieval,
   not wording. Still worth keeping, but it doesn't require freezing the prompt: change it for every pipeline at once and re-run them all.
2. **Overfitting.** Tuning against the same 220 questions you report scores on inflates the scores.
   The fix is a **dev/test split**, not a ban: tune on a dev slice with cheap smoke batches, and report on the untouched test
   slice when everything gets re-tested.

New approach (cheaper than full runs):
- **Dev/test split** (stratified by question type, fixed seed): tune only on dev.
- **Answer-step replay**: re-run only the answer step on the passages a saved agent run already collected. This takes
  retrieval variance out of answer-prompt experiments, and the research loop is about 60% of the agent's cost.
- Research-prompt changes are measured separately with `--no-judge` runs (context recall is deterministic and free to score).

## 8. Error analysis of the full run (free)

Read all 117 judge rationales for answers not marked "correct" in `200047`. Recurring failure modes, roughly by frequency:

| # | Failure | Example | Fix lives in |
|---|---|---|---|
| A | **Omits secondary points** the key includes: alternative license routes, fees, application documents, exam codes/scores | eligibility-029 omits emergency/residency routes; many omit the $60 fee | research (find them) + answer (include them) |
| B | **Overstates beyond the cited text**: "at least" for "more than", "must" for "may", maps an out-of-state credential to an NC area as settled, "X is a listed area" from a bare list entry | eligibility-004, -058, -019 | answer prompt |
| C | **Unsupported effective dates**: states "effective 2026-07-01" citing a passage that doesn't show it (rule 6 over-applied) | 6+ answers | answer prompt |
| D | **Verdict hedging**: needs_more_info / not_covered where the policies settle it (e.g. "no grace period exists" is an answer, not "not covered") | renewal_math-008, -025, eligibility-044, -045 | answer prompt |
| E | **False premise not corrected**: terms the policies don't use ("endorsement", "dual license", a digital-learning CEU requirement) | lookup-010, eligibility-026, -069 | answer prompt |
| F | Says the passages lack something that's in a table it was given | missing_facts-012, eligibility-030, -048 | answer prompt |

Other numbers: 87 of the 117 had only partial key-section coverage in context (retrieval matters), 25 had every key section
and still weren't "correct" (answering matters). Verdict mismatches are mostly hedges: needs_more_info -> conditional (8),
needs_more_info -> not_covered (6), informational -> conditional (6).

Caveat: the keys are Haiku-written and very inclusive (they list every fee and form), so some of A is the key being
exhaustive, not the answer being wrong. Still, A is what the analyst would want.

Spend so far (session 2): $0.00

## 9. Tuning tools built (no spend)

- `evals/data/splits.json`: **dev 54 / test 166**, stratified by question type, deterministic (hash of the id).
  Tune on dev only. Test stays untouched until the full re-test.
- `licn-eval run --split dev|test`, `--label "what changed"`, and `--replay <agent run>` (new `pipelines/replay.py`:
  re-runs only the answer step on the passages and calculator results a saved agent run collected).
  Replayed context is about 10% larger than the original (a partial exam-table hit comes back as the whole table), so replays are
  compared only with replays.
- Every run's `config.json` now records `prompt_version` (8-char hash of the answer + research prompts), split, replay
  source, and label, so results can be traced to the prompt that produced them.
- 3 new tests (152 total); ruff + pyright clean.

## 10. Experiment 0: noise floor ($0.37)

Two identical baseline replays (prompt v0) on the 54 dev questions:

| Run | Correct | Key-point coverage | Verdict match | Answers w/ unsupported claim |
|---|---|---|---|---|
| baseline replay 1 `202250` | 17/54 | 74% | 44/54 | 7 |
| baseline replay 2 `202526` | 14/54 | 73% | 45/54 | 10 |

**17 of 54 questions flipped correct/not-correct between identical runs.** The judge's "correct" call is mostly noise at
this n, so a prompt change that moves "correct" by fewer than about 6 questions on dev tells us nothing. **Key-point coverage is
stable (±1 point), so it's the main tuning metric**, with verdict match and unsupported claims as guardrails.
This also means the earlier n=7 smoke tests (session 1 and before) couldn't have measured anything on "correct".

Spend so far (session 2): $0.37

## 11. Judge review ($0.31 so far: v1 replay $0.15, two re-judges $0.16)

You asked whether the judge needed review. It did; I paused prompt tuning to fix it first, since everything else is
measured through it.

**Bugs found and fixed so far (free):**
1. **Cited text cut off at 6,000 characters.** The LICN-003 §5 exam table is 6,492, so claims about its later rows looked
   unsupported. Affected 111 citations in **87 of 220 answers** in the full run. Judge cap raised to 12,000.
2. **No policy header.** The answerer sees each policy's title and effective date; the judge didn't, so date claims looked
   invented. Judge now gets the same header.
3. **No parent context (found while hand-grading, fix in progress).** A cited list item like `LICN-013 §3(b)` reads only
   "Emergency License"; the rule ("DPI shall not add areas to the following") lives in `§3`. Answers cite the most specific
   passage (as the answer prompt tells them to) and the judge calls it "a bare list entry/truncated fragment". Hit **6 of 24**
   hand-graded answers (§3(b) add-on bar, "five-year cycle" from LICN-030 §1, the G.S. reference in LICN-019 §1(a), the
   Birth-K grouping, the classification list in LICN-012 §2, the CPL list in LICN-024 §1).

**New tool:** `licn-eval rejudge <run>` grades a saved run's answers again without re-answering (about $0.08 per 54 answers).

**Judge-only noise:** same 54 answers graded twice by the fixed judge: **9 of 54 overall labels flipped (17%)**. With answer
variance on top it was 17 of 54, so about half the "correct" noise comes from the judge. Key-point coverage is steady (about 2 points
per question, aggregate unchanged).

**Hand-graded reference set (24 dev answers, I graded them against the key and the policy text):**

| | Correct | Partial | Incorrect |
|---|---|---|---|
| Opus (me) | 21 | 3 | 0 |
| Haiku judge pass A | 11 | 13 | 0 |
| Haiku judge pass B | 10 | 14 | 0 |

The judge agrees with me on the bottom line nearly every time. The gap is all in correct vs partial, for three reasons: the judge demotes
answers for leaving out *peripheral* key points (fees, application documents, salary steps), for the parent-context bug above,
and for small inferential stretches. In practice its "correct" means "exhaustive and flawless", which is a vague line, and
that's why it flips. Caveat: I'm also a Claude model grading Claude answers; your wife's spot-check is the real reference.

**Plan:** fix parent context; replace the single holistic label with smaller, checkable judgments (does the bottom line agree,
material errors listed separately from minor issues, which key points are central), and compute correct/partial/incorrect
from those in code. Then re-validate against the 24 references and the noise test.

## 12. Judge v2 ($0.40)

Changes in `src/licn_rag/evals/score.py`:
- **Parent context:** a cited sub-item is shown after the lead-in lines of the sections above it (`context:` lines).
- **The single label is gone; the judge now makes smaller calls:** each key point marked `central` (does the bottom line depend on it) plus
  its coverage; `bottom_line` agrees / partly_agrees / disagrees; `material_errors` (wrong AND would change what the analyst
  does) kept separate from `minor_issues`. **correct / partially_correct / incorrect is computed in code**: correct = bottom
  line agrees + no material error + no central point missing; incorrect = bottom line disagrees or a central point contradicted.
- New metrics: `central_coverage`, `bottom_line`, `minor_issues`. `incorrect_statements` now holds only material errors.
- Reference labels saved in `evals/data/judge_reference.jsonl` (24 dev answers, marked as model-graded, pending human review).
- 7 new tests (159 total).

Validation on the same 54 answers, two passes each:

| Judge | Agrees with my 24 reference grades | Labels flipped between passes | "Correct" (pass A / B) |
|---|---|---|---|
| v1 (old rubric, cut-off fixed) | 14/24, 11/24 | 9/54 | 19 / 18 |
| **v2** | **20/24, 20/24** | 8/54 | 41 / 41 |

Agreement with the reference is much better and the aggregate is stable (41 vs 41). Per-question flips barely moved: most of
the remaining ones are agrees-vs-partly_agrees calls on hedged answers. That's a real gray area, so **compare aggregates, not single
questions**. Of my 4 disagreements, I'd concede 2 (the judge is right that missing_facts-005 hedges its verdict label). Not done:
a Sonnet-judge comparison (~$0.45-0.90). Worth doing once before the final test run, to check whether Haiku goes easy on Haiku.

**Note on scale:** v2 "correct" is about 75% where the old judge said about 35%. That's a definition change (correct now means "right bottom line,
central points, no material error" instead of "exhaustive and flawless"), not a quality jump. **Old and new "correct" numbers can't be
compared**; anything reported must be re-judged with v2 (re-judge cost is about $0.40 for a 220-question run).

## 13. Answer-prompt tuning on dev, judged by v2 ($1.02)

All runs: answer-step replays of the full agent run's passages, 54 dev questions.

| Prompt | Runs | Coverage | Central cov. | Verdict match | Answers w/ unsupported claim | Claim support | Correct |
|---|---|---|---|---|---|---|---|
| v0 (original) | 2 | 73-74% | 89-92% | 44-45 | 4-5 | 92% | 40-41 |
| v1 | 1 | 78% | 89% | **39** | 5 | 93% | 44 |
| v2 | 2 | 77% | 91-92% | 41-42 | **7** | 92-93% | 40-42 |
| **v3 (adopted)** | 1 | **78%** | 91% | **44** | **4** | **95%** | 42 |

What each version changed (`src/licn_rag/answer/prompt.py`, shared by every pipeline):
- **v1**, aimed at the failure modes from section 8: keep the cited text's qualifiers ("more than" is not "at least", "may" is not "must"); flag judgment calls (credential
  mapping, undefined terms) as needing confirmation; check table rows before saying something's missing; check every term
  the question assumes; effective dates only when timing matters; "no exception exists" is an answer; add related rules the
  analyst needs next (other routes, deadlines, fees, exams with codes and scores, application requirements).
  Coverage +4 points, but verdict labels drifted (needs_more_info -> conditional 7x; out-of-scope -> engaged).
- **v2**: explicit verdict definitions (needs_more_info = facts about the educator's present situation missing;
  conditional = depends on something that hasn't happened or someone else decides; not_covered = topic the policies don't
  govern, even if related rules are mentioned). Verdicts partly recovered; unsupported claims rose.
- **v3**: dropped "give the outcome if it's the same under every reading" (it turned needs_more_info into "no"); scoped
  "no exception = answer" to the educator's situation (topic not regulated -> not_covered); related rules must each be cited,
  no inferences. **Coverage gain kept, verdict match and unsupported claims back to v0 levels, best claim support.**

Caveats: v3 is one sample. The coverage gain shows up consistently across v1/v2/v3 (77-78 vs 73-74), so I trust that part.
Also, some of the remaining verdict mismatches are the keys' own labeling being inconsistent (e.g. "no such rule" is
not_covered in one key and needs_more_info in another), so I stopped chasing verdict match.
One mislabeled run: `205025` was meant to be v3 but my edit didn't apply; it's a second v2 sample, relabeled in its config.

## 14. Research-prompt experiment ($0.28), reverted

Added "what the analyst will need next (other routes, exams, fees, application requirements)" to the research prompt;
ran the full agent on dev without the judge. Context recall 79% vs 78% baseline (noise), with ~15% more tool calls and words.
No evidence it helps, so I reverted it. Candidate for the full re-test.

**Session 2 spend: $2.25** (of $3.00). Stopped spending here; docs next (free).

## 15. Docs (free)

**Decision: keep both files, but split their jobs and remove the duplication.**
- **README.md**: the public project page and the **single source of truth for status and results**. Rewritten:
  status line; agent described as evaluated with seven tools (incl. `convert` and why it exists); evaluation section now
  covers the dev/test split, context recall, the conversion check, the judge v2 design and the experiment tooling; results table
  gains the agent column plus rows for context recall and backwards conversions; "What this shows" rewritten; an explicit
  caveat that the "correct" rows are from judge v1 and aren't comparable with v2; a prompt-tuning section (dev, judge v2);
  new design decisions (code owns arithmetic and units, measure the measuring instrument, tune on dev / report on test);
  roadmap and next steps updated; new commands in the quickstart. No mention of human key review (plan rule kept).
- **GAME_PLAN.md**: the internal plan (your wife's use case, budget, research, cloud architecture, phases, risks). It
  can't merge into the README without putting personal notes on a public page and tripling its length. Changes: Revision 6
  line plus a note on how the two files split the work; §0 status trimmed to state + decisions, pointing to the README for
  numbers, with new "what the measurements changed" bullets (convert, judge, noise, tuning policy); §3.1 and §5 mention
  `convert`; §7 has the dev/test split, judge v2 design and validation, limitations, and why the `agent_nocalc` ablation wasn't run;
  §8 Phase 2 boxes ticked (the "close the gap" goal stays open: 47% vs 56%); §10 open questions updated.
- `implementation_plans/`: plan 02 -> done (smoke test was run in session 1), plan 03 -> done with a note on what it
  skipped and what went beyond it; index updated.
- Not touched: `SESSION_REPORT_plan-03.md` (Sonnet's handoff report). Its D1-D3 "open" items are now resolved by this log.

## 16. End of session 2: summary

- **Spend: $2.25 of $3.00** (session 1 was $1.45 separately).
- **Judge:** 3 bugs fixed (truncation, missing policy header, missing parent context), rubric rebuilt as smaller judgments,
  validated: 20/24 agreement with hand-graded references (was 11-14), stable aggregates.
- **Answer prompt v3** adopted: dev coverage 73-74% -> 78%, claim support 92% -> 95%, verdict match and unsupported claims
  unchanged. Research prompt unchanged (experiment showed nothing).
- **Tooling:** dev/test split, `--replay`, `rejudge`, `--label`, prompt fingerprint in every run.
- 159 tests pass; ruff, ruff format, pyright clean. **Nothing committed** (your call). New untracked files: `src/licn_rag/agent/units.py`,
  `src/licn_rag/evals/conversions.py`, `src/licn_rag/pipelines/replay.py`, `evals/data/splits.json`,
  `evals/data/judge_reference.jsonl`, 3 new test files, and the new run dirs.

**Recommended next (needs your OK and budget):**
1. Sonnet grades the 24 reference answers once (~$0.50): is Haiku going easy on Haiku?
2. Optional: your wife spot-checks `evals/data/judge_reference.jsonl` (24 rows, one-line notes); it's the yardstick for the judge.
3. Full re-test on the **test split** of long-context, RAG k=20, and agent with prompt v3 + judge v2 (~$2.50). These become the headline numbers.
4. Then decide agent vs RAG for production.
