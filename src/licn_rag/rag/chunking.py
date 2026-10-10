"""Split the parsed policies into retrieval chunks along their own structure.

A chunk is a section subtree (an item plus its sub-items) when it fits the size budget.
Bigger subtrees are split: the item's own text becomes one chunk and its children are
chunked recursively, with small neighbouring siblings packed together so that, say, a
list of 30 one-line license areas doesn't become 30 near-empty chunks. Large tables are
split by their group rows, with the header row repeated.

Every chunk carries:
- `breadcrumb`: policy › section heading › parent lead-in text, so a fragment like
  "(ii) Three CEUs related to literacy" still says what it belongs to;
- `text`: its blocks, each prefixed with its citation ID, ready to show the model;
- `cite_ids`: every citation ID inside it, used to score retrieval.
"""

import re
from dataclasses import dataclass, field

from licn_rag.corpus import cite_id
from licn_rag.ingest.models import Block, Policy

MAX_WORDS = 300
MIN_PACK_WORDS = 120  # pack sibling subtrees together until a chunk reaches this size
TABLE_ROWS_PER_CHUNK = 25


@dataclass
class Node:
    block: Block
    children: list["Node"] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str
    policy_code: str
    policy_title: str
    breadcrumb: str
    text: str
    cite_ids: list[str]
    order: int
    """Position in the corpus, used to present retrieved chunks in document order."""
    context: str = ""
    """Optional LLM-written situating note (contextual retrieval)."""

    @property
    def words(self) -> int:
        return len(self.text.split())

    def index_text(self) -> str:
        """What gets embedded and keyword-indexed."""
        parts = [self.breadcrumb, self.context, strip_ids(self.text)]
        return "\n".join(p for p in parts if p)


def strip_ids(text: str) -> str:
    return re.sub(r"^\s*\[[^\]]+\]\s*", "", text, flags=re.MULTILINE)


def block_line(policy: Policy, b: Block) -> str:
    indent = "  " * b.depth
    if b.kind == "table":
        return f"{indent}[{cite_id(policy, b)}] (table)\n{b.text}"
    title = f"{b.heading}. " if b.heading else ""
    return f"{indent}[{cite_id(policy, b)}] {title}{b.text}".rstrip()


def parent_path(path: str) -> str | None:
    """'2(b)(iv)' -> '2(b)'; '2' -> None."""
    i = path.rfind("(")
    return path[:i] if i > 0 else None


def build_tree(policy: Policy) -> list[Node]:
    """Top-level nodes in document order; list items nest under their parent path."""
    roots: list[Node] = []
    by_path: dict[str, Node] = {}
    for b in policy.blocks:
        if b.kind == "heading":
            continue
        node = Node(b)
        parent = None
        if b.kind == "list_item" and b.path:
            parent = by_path.get(parent_path(b.path) or "")
            by_path[b.path] = node
        elif b.path:  # paragraph/table attached to an item (continuations, notes)
            parent = by_path.get(b.path)
        (parent.children if parent else roots).append(node)
    return roots


def subtree_blocks(node: Node) -> list[Block]:
    out = [node.block]
    for child in node.children:
        out += subtree_blocks(child)
    return out


def words(blocks: list[Block]) -> int:
    return sum(len(b.text.split()) + len((b.heading or "").split()) for b in blocks)


class Chunker:
    def __init__(self, policy: Policy, start_order: int) -> None:
        self.policy = policy
        self.order = start_order
        self.chunks: list[Chunk] = []

    def crumb(self, ancestors: list[Block], section: str | None) -> str:
        parts = [f"{self.policy.code} {self.policy.title}"]
        if section and (not ancestors or ancestors[0].heading != section):
            parts.append(section)
        for a in ancestors:
            label = f"{a.label} " if a.label else ""
            lead = a.heading or truncate(a.text, 25)
            parts.append(f"{label}{lead}".strip())
        return " › ".join(p for p in parts if p)

    def emit(self, blocks: list[Block], ancestors: list[Block]) -> None:
        if not blocks:
            return
        first = blocks[0]
        self.chunks.append(
            Chunk(
                chunk_id=f"{self.policy.code}#{len(self.chunks):03d}",
                policy_code=self.policy.code,
                policy_title=self.policy.title,
                breadcrumb=self.crumb(ancestors, first.section),
                text="\n".join(block_line(self.policy, b) for b in blocks),
                cite_ids=[cite_id(self.policy, b) for b in blocks],
                order=self.order,
            )
        )
        self.order += 1

    def chunk_nodes(
        self, nodes: list[Node], ancestors: list[Block], lead: list[Block] | None = None
    ) -> None:
        """Pack small sibling subtrees together; recurse into ones that are too big.

        `lead` is a split parent's own block; it opens the first chunk of its children
        instead of becoming a near-empty chunk by itself.
        """
        pack: list[Block] = list(lead or [])
        for node in nodes:
            blocks = subtree_blocks(node)
            size = words(blocks)
            if node.block.kind == "table" and len(node.block.rows or []) > TABLE_ROWS_PER_CHUNK:
                self.emit(pack, ancestors)
                pack = []
                self.chunk_table(node.block, ancestors)
                self.chunk_nodes(node.children, ancestors)
            elif size > MAX_WORDS:
                self.emit(pack, ancestors)
                pack = []
                if words([node.block]) >= MIN_PACK_WORDS:
                    self.emit([node.block], ancestors)
                    self.chunk_nodes(node.children, ancestors + [node.block])
                else:
                    self.chunk_nodes(node.children, ancestors + [node.block], lead=[node.block])
            else:
                if pack and words(pack) + size > MAX_WORDS:
                    self.emit(pack, ancestors)
                    pack = []
                pack += blocks
                if words(pack) >= MIN_PACK_WORDS:
                    self.emit(pack, ancestors)
                    pack = []
        self.emit(pack, ancestors)

    def chunk_table(self, table: Block, ancestors: list[Block]) -> None:
        rows = table.rows or []
        header, body = rows[0], rows[1:]
        # Start a new piece at each group row (a single cell spanning the table).
        groups: list[list[list[str]]] = [[]]
        for row in body:
            if len(row) == 1 and groups[-1] and len(groups[-1]) >= 3:
                groups.append([])
            groups[-1].append(row)
        pieces: list[list[list[str]]] = []
        for g in groups:
            for i in range(0, len(g), TABLE_ROWS_PER_CHUNK):
                pieces.append(g[i : i + TABLE_ROWS_PER_CHUNK])
        for piece in pieces:
            part = table.model_copy(
                update={"rows": [header, *piece], "text": md_table([header, *piece])}
            )
            self.emit([part], ancestors)


def md_table(rows: list[list[str]]) -> str:
    width = max(len(r) for r in rows)

    def fmt(cells: list[str]) -> str:
        if len(cells) == 1 and width > 1:
            cells = [f"**{cells[0]}**"]
        return "| " + " | ".join(cells + [""] * (width - len(cells))) + " |"

    return "\n".join([fmt(rows[0]), "| " + " | ".join(["---"] * width) + " |", *map(fmt, rows[1:])])


def truncate(text: str, n_words: int) -> str:
    ws = text.split()
    return " ".join(ws[:n_words]) + ("…" if len(ws) > n_words else "")


def chunk_corpus(policies: list[Policy]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for policy in policies:
        chunker = Chunker(policy, start_order=len(chunks))
        chunker.chunk_nodes(build_tree(policy), [])
        chunks += chunker.chunks
    return chunks
