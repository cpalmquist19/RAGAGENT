# Session report: plan 03 (agent evaluation), 2026-10-11

For an AI model picking this project up. Read this, then `implementation_plans/03-agent-p3-evaluation.md`
(the plan being executed) and `GAME_PLAN.md`. Branch: `plan-03-agent-evaluation`.

## 1. Current status

| Item | State |
|---|---|
| Plan 03 steps 1-5 (metrics, `licn-eval context`, tests) | **Done**, commit `efd8b84` |
| Step 6a sample runs (agent vs RAG, 30 q) | **Done**, passed decision point A |
| Step 6b full agent run (220 q) | **Done** (with the *old* research prompt) |
| Step 6c calculator ablation (`agent_nocalc`) | **Not run, deliberately** (see M4) |
| Step 7 docs (README.md, GAME_PLAN.md) | **Not started** |
| Second commit (runs + docs), plan/Index status -> `done` | **Not done** |
| Plan 02 status ("in progress: smoke test pending") | **Stale**: smoke test was run (M1) but plan 02's file and `implementation_plans/README.md` index were not updated |
| Tests / lint / types at `d716126` | 108 pass; ruff check, ruff format --check, pyright src all clean |
| Working tree | Clean except 6 untracked run dirs (below) |

Commits on the branch: `efd8b84` (harness), `d716126` (research prompt step 4 + key fix).

Untracked run directories in `evals/runs/` (all judged by Haiku 5.5):

| Dir | What | Prompt |
|---|---|---|
| `20261011-181318-agent` | agent, stratified sample 30 | old |
| `20261011-181543-rag` | rag, same 30 | n/a |
| `20261011-182005-agent` | **agent, full 220** (the headline numbers) | old |
| `20261011-185706-agent` | agent, 30 `renewal_math-*` ids (prompt check) | new (`d716126`) |
| `20261011-192739-agent` | smoke test, Haiku 5.5, 7 conversion questions | new |
| `20261011-192907-agent` | smoke test, **Sonnet 5.5**, same 7 questions | new |

**Open decisions for the user** (none answered yet):
1. Fix the unit-conversion problem (D2) with a `convert` tool taking named units (`6 semester_hours -> ceus`),
   or accept it as a documented open issue.
2. Keep, revert, or replace the checklist prompt in `d716126`. The full-run numbers were produced with the
   *old* prompt, so keeping it means either a full re-run (~$1.30) or a note in the docs that the numbers predate it.
3. Whether to run 6c at all (recommendation: no; the tool is rarely the deciding factor).

## 2. Results (full run, 220 questions, Haiku 5.5 answering and judging)

| | Long-context | RAG k=20 | **Agent** |
|---|---|---|---|
| Correct (judge) | 56% | 37% | **46%** |
| Key-point coverage | 81% | 69% | 76% |
| Claims supported | 90% | 87% | 89% |
| Answers with unsupported claim | 18% | 27% | 24% |
| Answers contradicting key | 3% | 7% | 3% |
| Verdict matches key | 80% | 82% | 80% |
| Key sections in model's context | n/a (whole corpus) | 69% | **78%** |
| Section recall (citations) | 72% | 56% | 63% |
| Cost / question | $0.0018 | $0.0024 | $0.0044 |
| Latency p50 / p95 | 12.1s / 19.4s | 10.7s / 16.3s | 16.6s / 25.6s |
| Policy words shown (median) | 17,453 (full corpus) | 3,189 (k=20), 1,901 (k=12) | 3,223 |

- Agent: 4.52 tool calls and 3.05 research turns per question; 0 hit the turn limit; 0 errors; 3 tool errors.
- Correct by type (long-context / RAG / agent): eligibility 50/27/37, lookup 58/39/48, missing_facts 53/17/36,
  renewal_math 54/54/54, temporal 33/50/33 (n=6), unanswerable 70/62/68.
- The agent closes about 47% of the RAG-to-benchmark gap on `correct` ((46-37)/(56-37)). Paired exact sign test:
  agent beats RAG (48 vs 28 wins, p=0.029) but is still below long-context (29 vs 50, p=0.024).
- Agent correct rate is 50% when all key sections were in context (n=44) vs 39% when some were missing (n=147).
- 30-question sample (agent 37% vs RAG 40% correct) was within noise; context recall 78% vs 69% was not.
- Spend this session: about $2.3 total (full run $1.28; sample runs $0.29; prompt check ~$0.18; smoke tests ~$0.49).
- The judge is Haiku grading Haiku (self-grading caveat in GAME_PLAN §7). Do not change it without asking.

## 3. Defects found in the product / eval (not process mistakes)

| # | Defect | Evidence | Status |
|---|---|---|---|
| D1 | Agent skipped `calculate` on simple arithmetic despite the prompt saying to use it for all arithmetic | Full run: used on 6 of 30 `renewal_math-*` questions, 0 of the other 190 | Prompt step 4 rewritten (`d716126`); adoption rose to 19 of 30, accuracy did not move (M8) |
| D2 | **Models write unit conversions backwards** (1 CEU = 2/3 semester hour, so CEUs = hours / (2/3)); both Haiku and Sonnet do it, including *inside* `calculate` | See section 4 table; the calculator is exact but only for the expression the model writes | **Open** |
| D3 | The answer step is told to trust `<calculations>`, so a wrong expression propagates | renewal_math-012, run `185706`: `6 * 2/3 = 4` presented as settled. (In the Haiku smoke run the answer step caught and corrected the same error, so not deterministic.) | Open, part of D2 |
| D4 | Answer key for renewal_math-003 had the same inversion | Key said 3 semester hours = 2 CEUs and 14 x 2/3 = 9.33; correct 4.5 and 21 | Fixed by hand (M6) |

## 4. Calculator smoke test: Haiku 5.5 vs Sonnet 5.5

Same 7 questions, same prompt, same Haiku judge. Correct direction: **CEUs = semester hours / (2/3)**;
semester hours = CEUs x 2/3; 1 CEU = 10 clock hours (LICN-001 §2(g)).

| Q | Correct | Haiku 5.5 | Sonnet 5.5 |
|---|---|---|---|
| 012 | 6 sem hrs = 9 CEUs | x calc `6 * 2/3 = 4`; final answer corrected to 9 | ok `6 / (2/3) = 9` |
| 022 | 12 = 18 CEUs | x `12 * 2/3 = 8` | x `12 * 2/3 = 8` |
| 029 | 11.5 = 17.25 CEUs | no calc, no conversion stated | no calc, stated 17.25 (wrote "x 2/3", number right) |
| 003 | 3 = 4.5 CEUs | ok `3 / (2/3) = 4.5` | x `3 * 2/3 = 2` |
| 005 | 3 = 4.5 CEUs | x `3 * 2/3 = 2` | ok `3 / (2/3) = 4.5` |
| 011 | 2.0 CEUs = 1.33 sem hrs | x `2.0 / (2/3) = 3` | ok `2.0 * 2/3 = 1.33` |
| 007 | 8 CEUs = 5.33 sem hrs = 80 clock hrs | ok (no calc) | ok (no calc) |

Totals: wrong final conversion Haiku 3 of 6 stated, Sonnet 2 of 7. Judged correct/partial/incorrect: 1/4/2 for both.
Cost $0.04 vs $0.42 (about 10x); median latency 24s vs 13s. n=7, one run each: Haiku also flipped from its own
earlier runs on 005, 007, 011, so this is noise-level. Conclusion: switching to Sonnet does not fix D2.

## 5. Mistakes made this session

Each has: what went wrong, corrective action, end result, current status.

**M1. Started plan 03 while its dependency (plan 02) was not `done`.**
Plan 02 was "code done; real-API smoke test pending", but plan 03 requires it `done`.
*Action:* did only the free steps 1-5, then asked the user and ran plan 02's smoke test (2 questions) before 6a.
*Result:* smoke test passed (2 research turns, ~1.9k words, $0.0033-0.0036, 16-19s).
*Status:* plan 02's status line and the Index in `implementation_plans/README.md` still say "in progress". **Not updated.**

**M2. Truncated the first smoke-test output with `| tail -80`.**
The verdict and summary of question 1 ("6.5 CEUs, enough to renew?") were never seen; only its trace (no `calculate` call).
*Action:* none at the time; not re-run to save cost.
*Result:* the trace was the thing plan 02 asked to check, and it was recorded.
*Status:* question 1's verdict is unverified. Similar questions were covered by the later runs.

**M3. Misleading calculator-usage count.**
I reported "2 of 13 renewal_math (6 of 220 overall)", counting by *corrected* question type, then framed 6/220 as a possible harness problem.
*Action:* the user asked whether the harness was broken. I read the prompt, tool registration and answer-step plumbing, and recounted by ID prefix.
*Result:* 6 of 30 `renewal_math-*` ids used it and 0 of the other 190; the tool is exposed, the prompt asks for it, and results reach the answer step. Harness is fine. (Of the 30 prefixed ids only 12 keep type `renewal_math`; others were retyped.)
*Status:* resolved. Beware the two groupings when quoting numbers.

**M4. Offered the calculator ablation (6c) as the response to calculator underuse.**
This conflated "does the calculator help?" with "does the agent follow the instruction to use it?". The user pointed out the ablation can't fix a non-compliance problem.
*Action:* withdrew 6c; moved to prompt/harness fixes.
*Result:* 6c not run; its plan item is unmet.
*Status:* open; the docs must say the calculator's contribution was not measured (and why).

**M5. Said I had found no wrong answer caused by skipping the calculator.**
That was true when said but wrong in substance: I had not looked.
*Action:* scanned the judge's `incorrect_statements` for conversion/arithmetic errors.
*Result:* found renewal_math-022 (12 sem hrs = 8 CEUs, should be 18) and -029 (11.5 sem hrs = 7.67 CEUs, should be 17.25), both with no calculator call and both flipping the bottom line.
*Status:* retracted in-session; it led to D1/D2.

**M6. Used "regenerate keys" confusingly, and misread the key error.**
I explained why I had not fixed the key as "the plan forbids regenerating keys", which conflated rewriting all 220 reference answers with hand-fixing one. The user was confused. I had also read the key's computation as `14 x 2/3 = 9` from truncated output; it was 9.33.
*Action:* clarified; the user authorized a hand fix. Edited the raw JSONL line for `renewal_math-003` with exact-string replacements (each asserted to occur once): 5 wrong spots in 4 fields (reference answer, key point, a missing-facts question, `computation` x2). Scanned all keys for the same pattern.
*Result:* one line changed in `evals/data/silver_answers.jsonl`; scan found no other inverted keys (regex-based, so not exhaustive).
*Status:* committed in `d716126`. The long-context, RAG and full agent runs were judged against the **old** key, so renewal_math-003 is slightly mis-scored in them (1 of 220).

**M7. Wrong ground truth for 007 and a false blanket claim.**
In the smoke-test table I listed 8 CEUs = 12 semester hours as correct; it is 5.33. I had also said "every conversion that went through `calculate` was right (6 questions, 7 expressions)"; the full-run 007 expression `8 / (2/3) = 12` was inverted.
*Action:* caught while reading Haiku's 007 output; corrected openly to the user.
*Result:* correct values are 5.33 sem hrs / 80 clock hrs; the accurate statement is "calculator conversions are right only when the model writes the right expression, roughly a coin flip".
*Status:* corrected here and in section 4. No file contains the wrong value.

**M8. Predicted the checklist prompt would fix the numbers; it raised usage but not accuracy.**
*Action:* rewrote research prompt step 4 as a "before DONE, list every derived figure and calculate each" checklist (generic wording, no policy facts), ran the 30 `renewal_math-*` ids, and compared with the same ids in the full run.
*Result:* questions using `calculate` 6 -> 19 (calls 7 -> 30); `correct` 50% -> 47% (52% -> 48% excluding 003; 5 questions up, 6 down: noise); key-point coverage 79% -> 80%; tool calls/question 4.4 -> 5.8; cost $0.0042 -> $0.0047; p50 latency 17.4s -> 19.8s. 022 fixed; 012 newly wrong via a bad expression (D3).
*Status:* committed (`d716126`), decision pending (open decision 2).

**M9. Did not report a small rounding inconsistency.**
For the 30-question agent sample `licn-eval report` printed context recall 78% while `licn-eval context` printed 77%: `aggregate()` rounds to 3 decimals before formatting. I noticed and did not mention it.
*Action:* none.
*Result:* cosmetic, up to 1 point.
*Status:* unfixed. When quoting context recall, use `licn-eval context` output and say so.

## 6. Notes for the next model

- User preferences seen: wants short, plain answers; wants honest numbers even when unfavorable; approves each API spend (state an estimate first); told me not to change code during investigations ("look and report").
- Plan rules still binding: do not regenerate answer keys, do not change the judge model, do not change the shared answer prompt/schema, do not mention human (gold) review of keys in README.md, only one tuning round (already used).
- Windows console prints `§` as `�` in ad-hoc scripts; the data files are fine (UTF-8).
- Step 7 doc work needs: README results table (agent column + three new rows), "What this shows" rewrite, Roadmap ticks, status blockquote, "Next steps"; GAME_PLAN §0 status, revision line (Revision 6), §7 ablation paragraph, §8 checklist. Do **not** tick the "close the gap" item: the agent reached 46% against RAG 37% and benchmark 56%.
- The plan also requires a recommendation on whether `agent` should replace `rag`: on these numbers it is +9 points correct for about 1.8x cost and 1.5x latency, still 10 points short of long-context, and it has the open D2 problem.
- Reproduce: `uv run licn-eval report evals/runs/20261010-153106-long_context evals/runs/20261010-164558-rag evals/runs/20261011-182005-agent`;
  `uv run licn-eval context <run dirs>`.
