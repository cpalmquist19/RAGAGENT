"""The passages the agent has read so far, kept in document order.

Everything a tool shows the model is added here as the same "[citation ID] text" lines
the RAG pipeline uses. The answer step sees exactly these passages, so the agent and RAG
pipelines share one answer prompt and differ only in how the passages were chosen.
"""

import re
from dataclasses import dataclass
from functools import cache

from licn_rag.corpus import Corpus, cite_id
from licn_rag.ingest.models import Block, Policy
from licn_rag.rag.chunking import Chunk, block_line

# A chunk's text is block lines joined by "\n"; each line opens with an optionally
# indented "[LICN-… ]". Table rows start with "|", so they stay with their table.
LINE_START = re.compile(r"\n(?=[ ]*\[LICN-\d{3}[^\]]*\] )")
LINE_ID = re.compile(r"\s*\[(LICN-\d{3}[^\]]*)\]")


@cache
def doc_order(corpus: Corpus) -> dict[str, int]:
    """Citation ID -> position of the first block with that ID in the corpus."""
    order: dict[str, int] = {}
    position = 0
    for policy in corpus.policies:
        for block in policy.blocks:
            if block.kind != "heading":
                order.setdefault(cite_id(policy, block), position)
            position += 1
    return order


def split_chunk(text: str) -> list[str]:
    return LINE_START.split(text)


@dataclass
class _Entry:
    line: str
    cite_id: str
    order: int
    arrival: int


class Evidence:
    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self._order = doc_order(corpus)
        self._entries: dict[str, _Entry] = {}

    def add_line(self, line: str) -> bool:
        """Add one rendered block line. Returns False if it was already there."""
        key = line.strip()
        if key in self._entries:
            return False
        m = LINE_ID.match(line)
        if m is None:
            raise ValueError(f"not a passage line: {line[:60]!r}")
        cid = m.group(1)
        order = self._order.get(cid, len(self._order))
        self._entries[key] = _Entry(line, cid, order, len(self._entries))
        return True

    def add_chunk(self, chunk: Chunk) -> int:
        return sum(self.add_line(line) for line in split_chunk(chunk.text))

    def add_blocks(self, policy: Policy, blocks: list[Block]) -> int:
        return sum(self.add_line(block_line(policy, b)) for b in blocks if b.kind != "heading")

    def covers(self, chunk: Chunk) -> bool:
        """True if every line of the chunk has already been read."""
        return all(line.strip() in self._entries for line in split_chunk(chunk.text))

    def words(self) -> int:
        return sum(len(e.line.split()) for e in self._entries.values())

    def _sorted(self) -> list[_Entry]:
        return sorted(self._entries.values(), key=lambda e: (e.order, e.arrival))

    def ids(self) -> list[str]:
        """Unique citation IDs of everything read, in document order."""
        return list(dict.fromkeys(e.cite_id for e in self._sorted()))

    def __len__(self) -> int:
        return len(self._entries)

    def render(self) -> str:
        """Passages grouped under their policy header, in document order (same wrapper
        format as pipelines.rag.render_passages)."""
        out: list[str] = []
        current = None
        for e in self._sorted():
            code = e.cite_id[:8]
            if code != current:
                if current:
                    out.append("</policy>")
                p = self.corpus.by_code[code]
                out.append(
                    f'<policy code="{p.code}" title="{p.title}" '
                    f'effective="{p.effective_date or "unknown"}">'
                )
                current = code
            out.append(e.line)
        if current:
            out.append("</policy>")
        return "\n".join(out)
