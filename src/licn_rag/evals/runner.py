"""Run a pipeline over the eval questions, score every answer, and compare runs.

    uv run licn-eval run --pipeline long_context --sample 30   # quick stratified sample
    uv run licn-eval run --pipeline long_context               # all 220 questions
    uv run licn-eval report evals/runs/<run-a> evals/runs/<run-b>

Each run writes evals/runs/<timestamp>-<pipeline>/results.jsonl (one scored answer per
question) and summary.json (aggregate metrics, overall and per question type).
"""

import argparse
import json
import statistics
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic

from licn_rag.corpus import Corpus
from licn_rag.evals.score import deterministic, judge
from licn_rag.llm import make_client, verify_workspace
from licn_rag.pipelines.base import Pipeline
from licn_rag.pipelines.long_context import LongContextPipeline

QUESTIONS = Path("evals/data/draft_questions.jsonl")
ANSWER_KEYS = Path("evals/data/silver_answers.jsonl")
RUNS_DIR = Path("evals/runs")
DEFAULT_MODEL = "claude-haiku-5-5"

PIPELINES = {"long_context": LongContextPipeline}

# (metric, label, how to aggregate). "rate" averages booleans/fractions over the
# questions where the metric applies; "any" is the share of questions where it's > 0.
METRICS = [
    ("correct", "Correct (judge)", "rate"),
    ("correct_or_partial", "Correct or partially correct", "rate"),
    ("key_point_coverage", "Key-point coverage", "rate"),
    ("claim_support", "Claims supported by cited text", "rate"),
    ("unsupported_claims", "Answers with an unsupported claim", "any"),
    ("contradicted_key_points", "Answers contradicting the key", "any"),
    ("verdict_match", "Verdict matches key", "rate"),
    ("section_recall", "Section recall", "rate"),
    ("section_precision", "Section precision", "rate"),
    ("policy_recall", "Policy recall", "rate"),
    ("has_invalid_citation", "Answers citing a non-existent ID", "rate"),
    ("abstained_correctly", "Declines out-of-scope questions", "rate"),
    ("false_abstention", "Wrongly declines answerable questions", "rate"),
    ("asked_for_missing_facts", "Asks for missing facts when needed", "rate"),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def select(keys: dict[str, dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    questions = [q for q in read_jsonl(QUESTIONS) if q["id"] in keys]
    for q in questions:
        q["eval_type"] = keys[q["id"]]["corrected_type"]
    if args.ids:
        wanted = set(args.ids)
        questions = [q for q in questions if q["id"] in wanted]
    if args.types:
        questions = [q for q in questions if q["eval_type"] in args.types]
    if args.sample and args.sample < len(questions):
        # Stratified and deterministic: take questions round-robin across types.
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for q in questions:
            by_type[q["eval_type"]].append(q)
        picked: list[dict[str, Any]] = []
        while len(picked) < args.sample:
            for group in by_type.values():
                if group and len(picked) < args.sample:
                    picked.append(group.pop(0))
        questions = picked
    return questions


def score_one(
    pipeline: Pipeline,
    q: dict[str, Any],
    key: dict[str, Any],
    corpus: Corpus,
    client: anthropic.Anthropic,
    judge_model: str | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": q["id"],
        "type": q["eval_type"],
        "question": q["question"],
        "key_verdict": key["verdict"],
        "key_review": "silver",
    }
    try:
        result = pipeline.answer(q["question"])
    except anthropic.APIError as e:
        return record | {"error": f"{type(e).__name__}: {e}"}
    record |= {
        "model": result.model,
        "cost_usd": result.cost_usd,
        "latency_s": round(result.latency_s, 2),
        "context_ids": result.context_ids,
    }
    if result.answer is None:
        return record | {"error": result.error}
    record["answer"] = result.answer.model_dump()
    metrics = deterministic(result.answer, key, corpus)
    metrics["has_invalid_citation"] = bool(metrics["invalid_citations"])
    if judge_model:
        try:
            judged, judge_cost = judge(
                client, judge_model, q["question"], key, result.answer, corpus
            )
        except anthropic.APIError as e:
            judged, judge_cost = {"judge_error": f"{type(e).__name__}: {e}"}, 0.0
        metrics |= judged
        record["judge_cost_usd"] = judge_cost
        if "overall" in judged:
            metrics["correct"] = judged["overall"] == "correct"
            metrics["correct_or_partial"] = judged["overall"] != "incorrect"
    return record | {"metrics": metrics}


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in records if "metrics" in r]

    def agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(rows)}
        for key, _, how in METRICS:
            values = [r["metrics"].get(key) for r in rows]
            values = [v for v in values if v is not None]
            if not values:
                continue
            if how == "any":
                values = [v > 0 for v in values]
            out[key] = round(sum(float(v) for v in values) / len(values), 3)
            out[f"{key}_n"] = len(values)
        return out

    latencies = sorted(r["latency_s"] for r in scored)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in scored:
        by_type[r["type"]].append(r)
    n = len(scored)
    return {
        "overall": agg(scored),
        "by_type": {t: agg(rows) for t, rows in sorted(by_type.items())},
        "errors": [{"id": r["id"], "error": r["error"]} for r in records if "error" in r],
        "cost_per_question": round(sum(r["cost_usd"] for r in scored) / n, 5) if n else None,
        "judge_cost_total": round(sum(r.get("judge_cost_usd", 0) for r in records), 4),
        "pipeline_cost_total": round(sum(r.get("cost_usd", 0) for r in records), 4),
        "latency_p50_s": round(statistics.median(latencies), 2) if latencies else None,
        "latency_p95_s": round(latencies[int(0.95 * (n - 1))], 2) if latencies else None,
    }


def print_report(runs: list[tuple[str, dict[str, Any]]]) -> None:
    names = [name for name, _ in runs]
    width = max(14, *(len(n) for n in names))
    print(f"\n{'metric':40}" + "".join(f"{n:>{width + 2}}" for n in names))
    print("-" * (40 + (width + 2) * len(names)))

    def row(label: str, cells: list[str]) -> None:
        print(f"{label:40}" + "".join(f"{c:>{width + 2}}" for c in cells))

    row("questions scored", [str(s["overall"]["n"]) for _, s in runs])
    for key, label, _ in METRICS:
        cells = []
        for _, s in runs:
            v, n = s["overall"].get(key), s["overall"].get(f"{key}_n")
            cells.append(f"{v:.0%} ({n})" if v is not None else "–")
        row(label, cells)
    row("cost / question (pipeline)", [f"${s['cost_per_question']:.4f}" for _, s in runs])
    row("latency p50 / p95", [f"{s['latency_p50_s']}s / {s['latency_p95_s']}s" for _, s in runs])
    row("errors", [str(len(s["errors"])) for _, s in runs])

    if len(runs) == 1:
        _, s = runs[0]
        print("\nby question type (correct · key-point coverage · verdict match · section recall):")
        for t, m in s["by_type"].items():
            parts = [
                f"{m.get(k, 0):.0%}" if k in m else "–"
                for k in ("correct", "key_point_coverage", "verdict_match", "section_recall")
            ]
            print(f"  {t:15} n={m['n']:<4} " + " · ".join(parts))


def cmd_run(args: argparse.Namespace) -> None:
    keys = {r["id"]: r["answer"] for r in read_jsonl(ANSWER_KEYS) if "answer" in r}
    questions = select(keys, args)
    corpus = Corpus.load()
    client = make_client().with_options(max_retries=6)
    verify_workspace(client, args.model)  # fails fast if .env points at another workspace
    pipeline = PIPELINES[args.pipeline](client, args.model, corpus)
    judge_model = None if args.no_judge else args.judge_model

    run_dir = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{args.pipeline}"
    run_dir.mkdir(parents=True)
    config = {
        "pipeline": args.pipeline,
        "model": args.model,
        "judge_model": judge_model,
        "questions": len(questions),
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(
        f"{args.pipeline} · {args.model} · judge {judge_model} · {len(questions)} questions"
        f" → {run_dir}"
    )

    lock = threading.Lock()
    records: list[dict[str, Any]] = []
    out = (run_dir / "results.jsonl").open("w", encoding="utf-8")

    def work(q: dict[str, Any]) -> None:
        record = score_one(pipeline, q, keys[q["id"]], corpus, client, judge_model)
        with lock:
            records.append(record)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            m = record.get("metrics", {})
            status = record.get("error") or m.get("overall") or m.get("judge_error") or "scored"
            print(f"[{len(records)}/{len(questions)}] {q['id']:22} {status}", flush=True)

    # First question alone so the corpus is written to the prompt cache before the rest.
    work(questions[0])
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, questions[1:]))
    out.close()

    summary = aggregate(records)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_report([(run_dir.name, summary)])
    print(
        f"\npipeline ${summary['pipeline_cost_total']:.2f} + judge "
        f"${summary['judge_cost_total']:.2f}"
    )


def cmd_report(args: argparse.Namespace) -> None:
    runs = []
    for d in map(Path, args.runs):
        runs.append((d.name, json.loads((d / "summary.json").read_text(encoding="utf-8"))))
    print_report(runs)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--pipeline", choices=sorted(PIPELINES), required=True)
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--judge-model", default=DEFAULT_MODEL)
    r.add_argument("--no-judge", action="store_true", help="deterministic metrics only")
    r.add_argument("--sample", type=int, help="stratified sample of N questions")
    r.add_argument("--ids", nargs="+")
    r.add_argument("--types", nargs="+")
    r.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("report")
    p.add_argument("runs", nargs="+", help="run directories to compare")
    args = ap.parse_args(argv)
    try:
        {"run": cmd_run, "report": cmd_report}[args.cmd](args)
    except anthropic.AuthenticationError:
        print("authentication failed: check ANTHROPIC_API_KEY in .env", file=sys.stderr)
        return 1
    except anthropic.APIConnectionError:
        print("could not reach the Anthropic API", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
