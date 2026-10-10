"""Ingest saved Simbli policy pages into the corpus.

    uv run licn-ingest nc-licensure-papers --out corpus/licn

Writes, per policy, <CODE>.json (structured) and <CODE>.md (readable), plus
manifest.json summarizing every policy (dates, hashes, warnings). Only the main
.html of each saved page is read; the "_files" folders are ignored.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from licn_rag.ingest.simbli_html import parse_saved_page, to_markdown


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("src", type=Path, help="folder of browser-saved Simbli policy pages")
    ap.add_argument("--out", type=Path, default=Path("corpus/licn"))
    args = ap.parse_args(argv)

    pages = sorted(args.src.glob("*LICN-*.htm*"))
    if not pages:
        print(f"no saved LICN pages found in {args.src}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    manifest, failures = [], 0
    for page in pages:
        try:
            policy = parse_saved_page(page)
        except ValueError as e:
            print(f"FAIL {page.name}: {e}", file=sys.stderr)
            failures += 1
            continue
        (args.out / f"{policy.code}.json").write_text(
            policy.model_dump_json(indent=2), encoding="utf-8"
        )
        (args.out / f"{policy.code}.md").write_text(to_markdown(policy), encoding="utf-8")

        kinds = [b.kind for b in policy.blocks]
        manifest.append(
            {
                "code": policy.code,
                "title": policy.title,
                "status": policy.status,
                "effective_date": str(policy.effective_date),
                "last_revised": str(policy.last_revised),
                "source_url": policy.source_url,
                "sha256": policy.sha256,
                "blocks": len(kinds),
                "list_items": kinds.count("list_item"),
                "tables": kinds.count("table"),
                "words": sum(len(b.text.split()) for b in policy.blocks),
                "warnings": policy.warnings,
            }
        )

    manifest.sort(key=lambda m: m["code"])
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    today = date.today()
    print(f"{'code':9} {'eff.':10} {'blocks':>6} {'items':>5} {'tbls':>4} {'words':>6}  title")
    for m in manifest:
        flag = " (not yet effective)" if m["effective_date"] > str(today) else ""
        print(
            f"{m['code']:9} {m['effective_date']:10} {m['blocks']:6} {m['list_items']:5} "
            f"{m['tables']:4} {m['words']:6}  {m['title']}{flag}"
        )
        for w in m["warnings"]:
            print(f"          WARNING: {w}")
    total_words = sum(m["words"] for m in manifest)
    approx_tokens = int(total_words * 1.4)
    print(
        f"\n{len(manifest)} policies, {total_words:,} words (~{approx_tokens:,} tokens), "
        f"{failures} failed"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
