"""Generate "silver" answer keys for the draft eval questions.

The model (ANTHROPIC_MODEL or --model) reads the *entire* LICN corpus (it is small
enough) and writes, for each question: the reference answer, cited key points, false
premises, missing facts, and out-of-scope parts. Every citation is checked against the
corpus. Items stay `review_status: "silver"` until a human reviews them. Ideally the key
writer is stronger than the model under test; the current keys are mostly Haiku 5.5.

    uv run licn-silver estimate                  # token count + cost estimate (free)
    uv run licn-silver pilot --n 3               # run a few now, at standard price
    uv run licn-silver run                       # everything now, in parallel (standard price)
    uv run licn-silver submit                    # or: via the Batch API (-50%, turnaround
    uv run licn-silver collect --wait            #     can be minutes to hours)
    uv run licn-silver revalidate                # re-check citations after a corpus change

Results merge into evals/data/silver_answers.jsonl keyed by question id, so re-running
only fills in what's missing (use --ids / --force to redo specific questions).
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, get_args

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from pydantic import BaseModel, ValidationError

from licn_rag.corpus import all_cite_ids, load_policies, normalize_citation, render_corpus
from licn_rag.llm import make_client, model_name, price, usage_cost, verify_workspace

QUESTIONS = Path("evals/data/draft_questions.jsonl")
OUTPUT = Path("evals/data/silver_answers.jsonl")
STATE_DIR = Path("data/silver")  # git-ignored: batch ids and raw API results
CORPUS_SNAPSHOT = "2026-10-10"
MAX_TOKENS = 16000


QuestionType = Literal[
    "lookup", "eligibility", "renewal_math", "missing_facts", "unanswerable", "temporal"
]


# ---------------------------------------------------------------- answer key schema


class KeyPoint(BaseModel):
    point: str
    citations: list[str]


class SilverAnswer(BaseModel):
    corrected_type: QuestionType
    answerable: Literal["yes", "partial", "no"]
    verdict: Literal["yes", "no", "conditional", "needs_more_info", "informational", "not_covered"]
    reference_answer: str
    key_points: list[KeyPoint]
    expected_citations: list[str]
    missing_facts_to_ask: list[str]
    false_premises: list[str]
    out_of_scope_parts: list[str]
    computation: str | None
    confidence: Literal["high", "medium", "low"]
    reviewer_notes: str


def json_schema() -> dict[str, Any]:
    """Hand-written to stay inside the structured-output JSON Schema subset."""
    strs = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(SilverAnswer.model_fields),
        "properties": {
            "corrected_type": {"type": "string", "enum": list(get_args(QuestionType))},
            "answerable": {"type": "string", "enum": ["yes", "partial", "no"]},
            "verdict": {
                "type": "string",
                "enum": [
                    "yes",
                    "no",
                    "conditional",
                    "needs_more_info",
                    "informational",
                    "not_covered",
                ],
            },
            "reference_answer": {"type": "string"},
            "key_points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["point", "citations"],
                    "properties": {"point": {"type": "string"}, "citations": strs},
                },
            },
            "expected_citations": strs,
            "missing_facts_to_ask": strs,
            "false_premises": strs,
            "out_of_scope_parts": strs,
            "computation": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "reviewer_notes": {"type": "string"},
        },
    }


# ---------------------------------------------------------------- prompt

INSTRUCTIONS = f"""\
You are writing the answer key for a test set. The test set evaluates a question-answering \
assistant used by a licensure analyst at Union County Public Schools (North Carolina). The \
analyst asks about the North Carolina State Board of Education's Licensure (LICN) policies, \
usually about an anonymized educator's situation.

The complete set of LICN policies the assistant can use is provided below (snapshot \
{CORPUS_SNAPSHOT}). Each citable passage is prefixed with its citation ID in square brackets, \
for example [LICN-030 §1(a)(ii)].

How to write the answer key:

1. Use only the policy text below. Do not use outside knowledge of North Carolina licensure, \
even if you believe it is correct. If the policies do not address something the question asks, \
list it in out_of_scope_parts instead of answering it.
2. Cite with the exact IDs shown in brackets, without the brackets. Cite the most specific \
passage that supports each point. Every key point needs at least one citation, except a point \
whose whole content is that the policies are silent on something. expected_citations is the \
deduplicated set of passages an ideal answer must rely on.
3. The questions were drafted by an AI that never saw these policies. They may use wrong \
license names, assume rules that do not exist, or misstate requirements. Record each premise \
the policies contradict in false_premises, and answer based on what the policies actually say.
4. If the answer depends on facts the question does not give (grade band, years of \
experience, degree, whether a test was passed, and so on), set verdict to "needs_more_info", \
list what to ask in missing_facts_to_ask, and still explain how the answer turns on each fact.
5. When the question involves counting (credits, CEUs, years, hours, deadlines), show the \
arithmetic in computation, using the policies' own definitions (for example, how a CEU is \
defined). Otherwise set computation to null.
6. The corpus holds only the currently adopted versions. For questions about older rules or \
past dates, give the current rule, note its effective date, and say the corpus cannot confirm \
what applied earlier. Some policies are older than others (check each policy's effective date) \
and some cite policies that are not in the corpus (for example LICN-014); say so when relevant.
7. corrected_type: re-label the question if its draft type is wrong. For example, a \
"renewal_math" question with no arithmetic may really be "eligibility", and a question the \
policies do not cover is "unanswerable".
8. answerable: "yes" if the policies fully answer it, "partial" if only some parts, "no" if \
none. verdict: "yes" / "no" for eligibility-style questions with a clear outcome, \
"conditional" when the outcome depends on stated conditions, "needs_more_info" as in rule 4, \
"informational" for lookups with no yes/no outcome, "not_covered" when the policies do not \
answer it.
9. reference_answer: the answer an excellent assistant should give the analyst, in 2-6 \
plain sentences, without citation IDs.
10. confidence and reviewer_notes: lower confidence and explain when policies conflict, \
wording is ambiguous, the source text looks defective (for example a missing number), or you \
had to interpret. The human reviewer reads reviewer_notes first.
"""


def build_system(policies_text: str) -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": INSTRUCTIONS},
        {
            "type": "text",
            "text": f"<corpus>\n{policies_text}\n</corpus>",
            "cache_control": {"type": "ephemeral"},
        },
    ]


def question_message(q: dict[str, Any]) -> str:
    facts = "; ".join(q.get("facts_given") or []) or "none listed"
    return (
        f"Question ID: {q['id']}\n"
        f"Draft type: {q['type']}\n"
        f"Draft notes (written without seeing the policies; may be wrong): "
        f"{q.get('notes') or 'none'}\n"
        f"Facts the drafter listed: {facts}\n\n"
        f"Question as the analyst typed it:\n{q['question']}"
    )


def request_params(model: str, system: list[dict[str, Any]], q: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": [{"role": "user", "content": question_message(q)}],
        # Effort is set explicitly: answer keys need careful reasoning, and Haiku 5.5
        # defaults to "medium".
        "output_config": {
            "effort": "high",
            "format": {"type": "json_schema", "schema": json_schema()},
        },
    }


# ---------------------------------------------------------------- results handling


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cost(model: str, usage: Any, batch: bool) -> float:
    total = usage_cost(model, usage)
    return total / 2 if batch else total


def to_record(
    q: dict[str, Any], message: Any, valid_cites: set[str], source: str, batch: bool
) -> dict[str, Any]:
    """Validate one API response into an answer-key record (or an error record)."""
    record: dict[str, Any] = {
        "id": q["id"],
        "question": q["question"],
        "draft_type": q["type"],
        "model": message.model,
        "source": source,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cost_usd": round(cost(message.model, message.usage, batch), 5),
        "review_status": "silver",
    }
    if message.stop_reason != "end_turn":
        return record | {"error": f"stop_reason={message.stop_reason}"}
    text = next((b.text for b in message.content if b.type == "text"), "")
    try:
        answer = SilverAnswer.model_validate_json(text)
    except ValidationError as e:
        return record | {"error": f"invalid answer JSON: {e.errors()[:3]}", "raw": text}

    data, bad = normalize_answer_citations(answer.model_dump(), valid_cites)
    return record | {"answer": data, "invalid_citations": bad}


def normalize_answer_citations(
    answer: dict[str, Any], valid: set[str]
) -> tuple[dict[str, Any], list[str]]:
    """Rewrite citations to canonical IDs; return the ones that match no real passage."""
    bad: set[str] = set()

    def fix(cites: list[str]) -> list[str]:
        out = []
        for c in cites:
            n = normalize_citation(c, valid)
            if n is None:
                bad.add(c)
            out.append(n or c)
        return list(dict.fromkeys(out))  # dedupe, keep order

    answer["expected_citations"] = fix(answer["expected_citations"])
    for kp in answer["key_points"]:
        kp["citations"] = fix(kp["citations"])
    return answer, sorted(bad)


def cmd_revalidate(_: argparse.Namespace) -> None:
    """Re-check saved answers against the current corpus (no API calls)."""
    valid = all_cite_ids(load_policies())
    rows = read_jsonl(OUTPUT)
    for r in rows:
        if "answer" in r:
            r["answer"], r["invalid_citations"] = normalize_answer_citations(r["answer"], valid)
    write_jsonl(OUTPUT, rows)
    summarize()


def merge(records: list[dict[str, Any]]) -> None:
    by_id = {r["id"]: r for r in read_jsonl(OUTPUT)}
    for r in records:
        if "error" in r and "answer" in by_id.get(r["id"], {}):
            continue  # don't overwrite a good answer with a failed retry
        by_id[r["id"]] = r
    order = [q["id"] for q in read_jsonl(QUESTIONS)]
    write_jsonl(OUTPUT, [by_id[i] for i in order if i in by_id])


def pending_questions(ids: list[str] | None, force: bool) -> list[dict[str, Any]]:
    questions = read_jsonl(QUESTIONS)
    if ids:
        wanted = set(ids)
        return [q for q in questions if q["id"] in wanted]
    if force:
        return questions
    # Answers with citations that don't exist in the corpus are redone too.
    done = {r["id"] for r in read_jsonl(OUTPUT) if "answer" in r and not r["invalid_citations"]}
    return [q for q in questions if q["id"] not in done]


def summarize() -> None:
    rows = read_jsonl(OUTPUT)
    ok = [r for r in rows if "answer" in r]
    errors = [r for r in rows if "error" in r]
    bad_cites = [r for r in ok if r["invalid_citations"]]
    print(
        f"\n{OUTPUT}: {len(ok)} answered, {len(errors)} errors, {len(bad_cites)} w/ bad citations"
    )
    if ok:

        def tally(key: str) -> str:
            counts: dict[str, int] = {}
            for r in ok:
                counts[r["answer"][key]] = counts.get(r["answer"][key], 0) + 1
            return ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))

        print(f"  answerable: {tally('answerable')}")
        print(f"  verdict:    {tally('verdict')}")
        print(f"  confidence: {tally('confidence')}")
        relabeled = sum(r["answer"]["corrected_type"] != r["draft_type"] for r in ok)
        print(f"  relabeled:  {relabeled}   total cost: ${sum(r['cost_usd'] for r in rows):.2f}")
    for r in errors:
        print(f"  ERROR {r['id']}: {r['error'][:120]}")
    for r in bad_cites:
        print(f"  BAD CITATIONS {r['id']}: {r['invalid_citations']}")


# ---------------------------------------------------------------- commands


def setup(args: argparse.Namespace) -> tuple[anthropic.Anthropic, str, list[dict[str, Any]], set]:
    policies = load_policies()
    client = make_client()
    model = args.model or model_name()
    workspace = verify_workspace(client, model)
    print(f"model {model} · workspace {workspace} · {len(policies)} policies")
    return client, model, build_system(render_corpus(policies)), all_cite_ids(policies)


def cmd_estimate(args: argparse.Namespace) -> None:
    client, model, system, _ = setup(args)
    questions = pending_questions(args.ids, args.force)
    if not questions:
        print("nothing pending")
        return
    sample = request_params(model, system, questions[0])
    counted = client.messages.count_tokens(
        model=model, system=sample["system"], messages=sample["messages"]
    )
    n, per_q_in, p = len(questions), counted.input_tokens, price(model)
    corpus_tokens = per_q_in - 300  # rough: everything but the question is shared
    out_low, out_high = 1500, 5000  # thinking + JSON answer, per question
    batch_in = (
        (
            corpus_tokens * p["cache_write"]
            + (n - 1) * corpus_tokens * p["cache_read"]
            + n * 300 * p["input"]
        )
        / 1e6
        / 2
    )
    no_cache_in = n * per_q_in * p["input"] / 1e6 / 2
    out_cost = [n * t * p["output"] / 1e6 / 2 for t in (out_low, out_high)]
    print(f"{n} questions · ~{per_q_in:,} input tokens each (corpus + instructions + question)")
    print(f"batch input: ${batch_in:.2f} if the cache hits, up to ${no_cache_in:.2f} if it doesn't")
    print(f"batch output: ${out_cost[0]:.2f}–${out_cost[1]:.2f}")
    print(
        f"=> roughly ${batch_in + out_cost[0]:.2f}–${no_cache_in + out_cost[1]:.2f} for the batch"
    )


def cmd_pilot(args: argparse.Namespace) -> None:
    client, model, system, valid = setup(args)
    questions = pending_questions(args.ids, args.force)
    if not args.ids:
        # One per question type, so the pilot shows the full range of behaviour.
        seen: dict[str, dict[str, Any]] = {}
        for q in questions:
            seen.setdefault(q["type"], q)
        questions = list(seen.values())[: args.n]
    records = []
    for q in questions:
        print(f"→ {q['id']}: {q['question'][:90]}")
        message = client.messages.create(**request_params(model, system, q))
        records.append(to_record(q, message, valid, source="sync", batch=False))
        u = message.usage
        print(
            f"  in {u.input_tokens} · cache w {u.cache_creation_input_tokens} "
            f"r {u.cache_read_input_tokens} · out {u.output_tokens} · ${records[-1]['cost_usd']}"
        )
    merge(records)
    summarize()


def cmd_run(args: argparse.Namespace) -> None:
    """Answer pending questions now, in parallel, at standard price."""
    client, model, system, valid = setup(args)
    client = client.with_options(max_retries=6)  # ride out rate limits per request
    questions = pending_questions(args.ids, args.force)
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        print("nothing pending")
        return

    lock = threading.Lock()
    done = 0

    def answer(q: dict[str, Any]) -> dict[str, Any]:
        nonlocal done
        try:
            message = client.messages.create(**request_params(model, system, q))
            record = to_record(q, message, valid, source="sync", batch=False)
        except anthropic.APIError as e:
            record = {
                "id": q["id"],
                "question": q["question"],
                "draft_type": q["type"],
                "error": f"{type(e).__name__}: {e}",
                "review_status": "silver",
                "cost_usd": 0.0,
            }
        with lock:  # save as we go, so an interrupted run keeps finished answers
            merge([record])
            done += 1
            status = "ERROR" if "error" in record else f"${record['cost_usd']}"
            print(f"[{done}/{len(questions)}] {q['id']} {status}", flush=True)
        return record

    # The first request writes the corpus to the prompt cache; the rest then read it.
    answer(questions[0])
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(answer, questions[1:]))
    summarize()


def cmd_submit(args: argparse.Namespace) -> None:
    client, model, system, _ = setup(args)
    questions = pending_questions(args.ids, args.force)
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        print("nothing pending")
        return
    batch = client.messages.batches.create(
        requests=[
            Request(
                custom_id=q["id"],
                params=MessageCreateParamsNonStreaming(**request_params(model, system, q)),
            )
            for q in questions
        ]
    )
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = STATE_DIR / "batches.jsonl"
    with state.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"batch_id": batch.id, "n": len(questions), "model": model}) + "\n")
    print(f"submitted batch {batch.id} with {len(questions)} questions; run `licn-silver collect`")


def cmd_collect(args: argparse.Namespace) -> None:
    client, _, _, valid = setup(args)
    state = read_jsonl(STATE_DIR / "batches.jsonl")
    if not state:
        print("no batches submitted")
        return
    batch_id = args.batch or state[-1]["batch_id"]
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        c = batch.request_counts
        print(
            f"{batch_id}: {batch.processing_status} · done {c.succeeded} · "
            f"errored {c.errored} · processing {c.processing}"
        )
        if batch.processing_status == "ended" or not args.wait:
            break
        time.sleep(60)
    if batch.processing_status != "ended":
        return

    questions = {q["id"]: q for q in read_jsonl(QUESTIONS)}
    records, raw = [], []
    for result in client.messages.batches.results(batch_id):
        q = questions[result.custom_id]
        raw.append(result.model_dump(mode="json"))
        if result.result.type == "succeeded":
            records.append(to_record(q, result.result.message, valid, source=batch_id, batch=True))
        else:
            records.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "draft_type": q["type"],
                    "error": f"batch result {result.result.type}",
                    "review_status": "silver",
                    "cost_usd": 0.0,
                }
            )
    write_jsonl(STATE_DIR / f"{batch_id}.raw.jsonl", raw)
    merge(records)
    summarize()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", help="override ANTHROPIC_MODEL")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("estimate", "pilot", "run", "submit"):
        p = sub.add_parser(name)
        p.add_argument("--ids", nargs="+", help="only these question ids")
        p.add_argument("--force", action="store_true", help="redo questions already answered")
    sub.choices["pilot"].add_argument("--n", type=int, default=3)
    sub.choices["submit"].add_argument("--limit", type=int)
    sub.choices["run"].add_argument("--limit", type=int)
    sub.choices["run"].add_argument("--workers", type=int, default=8)
    c = sub.add_parser("collect")
    c.add_argument("--batch", help="batch id (default: most recent)")
    c.add_argument("--wait", action="store_true", help="poll until the batch ends")
    sub.add_parser("summary")
    sub.add_parser("revalidate")
    args = ap.parse_args(argv)

    try:
        {
            "estimate": cmd_estimate,
            "pilot": cmd_pilot,
            "run": cmd_run,
            "submit": cmd_submit,
            "collect": cmd_collect,
            "summary": lambda _: summarize(),
            "revalidate": cmd_revalidate,
        }[args.cmd](args)
    except anthropic.AuthenticationError:
        print("authentication failed: check ANTHROPIC_API_KEY in .env", file=sys.stderr)
        return 1
    except anthropic.PermissionDeniedError as e:
        print(f"permission denied (workspace or model access?): {e.message}", file=sys.stderr)
        return 1
    except anthropic.RateLimitError:
        print("rate limited; wait a minute and re-run (finished answers are kept)", file=sys.stderr)
        return 1
    except anthropic.APIStatusError as e:
        print(f"API error {e.status_code}: {e.message}", file=sys.stderr)
        return 1
    except anthropic.APIConnectionError:
        print("could not reach the Anthropic API; check your connection", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
