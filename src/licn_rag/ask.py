"""Ask the assistant a question from the command line.

uv run licn-ask "How many CEUs does a K-5 teacher need to renew a CPL?"
uv run licn-ask --sources "..."     # also print the text of every cited passage
"""

import argparse
import json
import sys
import textwrap

from licn_rag.corpus import Corpus
from licn_rag.llm import make_client
from licn_rag.pipelines.registry import PIPELINES


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("question")
    ap.add_argument("--pipeline", choices=sorted(PIPELINES), default="rag")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--sources", action="store_true", help="print cited passage text")
    ap.add_argument("--trace", action="store_true", help="print the agent's tool calls")
    args = ap.parse_args(argv)

    corpus = Corpus.load()
    result = PIPELINES[args.pipeline](make_client(), args.model, corpus).answer(args.question)
    if result.answer is None:
        print(f"error: {result.error}", file=sys.stderr)
        return 1
    a = result.answer
    wrap = textwrap.TextWrapper(width=100, subsequent_indent="    ")
    print(f"\nVerdict: {a.verdict}\n\n{textwrap.fill(a.summary, 100)}\n")
    for claim in a.claims:
        print(wrap.fill(f"  • {claim.text}  [{'; '.join(claim.citations) or 'no citation'}]"))
    for label, items in [
        ("Questions to ask", a.questions_for_user),
        ("Premise corrections", a.premise_corrections),
        ("Not covered by the policies", a.not_covered),
    ]:
        if items:
            print(f"\n{label}:")
            for item in items:
                print(wrap.fill(f"  - {item}"))
    if args.sources:
        print("\nSources:")
        for c in a.citations():
            print(f"\n{corpus.passage(corpus.normalize(c) or c) or f'[{c}] (not found)'}")
    if args.trace and result.trace is not None:
        r = result.research or {}
        print(
            f"\nResearch ({r.get('turns', '?')} turns, stopped: {r.get('stop', '?')}, "
            f"{result.context_words or 0:,} words of policy text):"
        )
        for t in result.trace:
            status = "ERROR" if t["error"] else f"+{t['added_words']:,} words"
            args_text = json.dumps(t["input"], ensure_ascii=False)
            print(f"  {t['tool']:18} {textwrap.shorten(args_text, 60):60} {status}")
    print(f"\n({result.model} ·{result.latency_s:.1f}s · ${result.cost_usd:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
