"""Convert raw Claude API / playground responses into JSONL question files.

Usage:
    python evals/scripts/extract_playground.py analyst_questions/*.json

Each <name>.json becomes <name>.jsonl next to it. Text blocks are scanned line by
line; anything that isn't a JSON object (code fences, prose) is reported and skipped.
"""

import json
import sys
from pathlib import Path

REQUIRED = {"id", "type", "question"}


def extract(path: Path) -> list[dict]:
    response = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join(b["text"] for b in response["content"] if b.get("type") == "text")
    rows, skipped = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            skipped.append(line[:80])
            continue
        if isinstance(row, dict) and row.keys() >= REQUIRED:
            rows.append(row)
        else:
            skipped.append(line[:80])
    stop = response.get("stop_reason")
    print(f"{path.name}: {len(rows)} rows, {len(skipped)} skipped lines, stop_reason={stop}")
    for s in skipped:
        print(f"    skipped: {s}")
    if stop == "max_tokens":
        print("    WARNING: response was truncated (max_tokens); the last question may be missing")
    return rows


def main(paths: list[str]) -> None:
    for p in map(Path, paths):
        if p.suffix != ".json":
            continue
        rows = extract(p)
        out = p.with_suffix(".jsonl")
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
