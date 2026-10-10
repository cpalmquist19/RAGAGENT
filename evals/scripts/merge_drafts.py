"""Merge per-batch question files into one draft eval set with unique, stable IDs.

Usage:
    python evals/scripts/merge_drafts.py analyst_questions evals/data/draft_questions.jsonl

IDs are reassigned as <type>-<nnn> in file order (files sorted by name), so batches
that reused the same numbering don't collide. The original id and file are kept in
`source` for traceability. Every row starts as review_status "draft".
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

# Playground artifacts like "terse: ..." at the start of a question.
PREFIX_ARTIFACT = re.compile(r"^(terse|short|informal)\s*:\s*", re.IGNORECASE)


def main(src_dir: str, out_path: str) -> None:
    counters: Counter[str] = Counter()
    rows = []
    for path in sorted(Path(src_dir).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            qtype = row["type"]
            counters[qtype] += 1
            rows.append(
                {
                    **row,
                    "id": f"{qtype}-{counters[qtype]:03d}",
                    "question": PREFIX_ARTIFACT.sub("", row["question"]).strip(),
                    "source": {"file": path.name, "original_id": row["id"]},
                    "review_status": "draft",
                }
            )

    seen: set[str] = set()
    unique = []
    for row in rows:
        key = row["question"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(row)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in unique:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(unique)} questions to {out} ({len(rows) - len(unique)} duplicates dropped)")
    for qtype, n in sorted(Counter(r["type"] for r in unique).items()):
        print(f"  {qtype}: {n}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
