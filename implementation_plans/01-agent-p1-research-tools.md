# Plan 01: Agent research tools

| | |
|---|---|
| **Plan** | 01 · agent phase 1 of 3 |
| **Status** | not started |
| **Depends on** | none |
| **Estimated size** | 5 new source files (~550 lines), 3 new test files (~250 lines) |
| **API cost** | $0. Nothing in this plan calls the Anthropic API. |

## Goal

Build the **tools** the research agent will use in plan 02, as plain Python with unit tests. No LLM calls yet. When this plan is done, a Python caller can do:

```python
box = ToolBox(corpus, index)
box.run("get_section", {"citation_id": "LICN-030 §1"})        # -> ToolOutcome(text=..., is_error=False)
box.run("follow_references", {"citation_id": "LICN-010 §5(c)"})
box.run("calculate", {"description": "CEUs from 6 semester hours", "expression": "6 * 2/3"})
box.evidence.render()   # every passage the tools showed, in document order, ready for the answer prompt
```

## Background (why this exists)

The single-shot RAG pipeline (`rag`) gets 37% "correct" from the judge; the long-context benchmark (whole corpus in the prompt) gets 56%. The gap is **multi-hop retrieval**: the median answer key cites 5 different policies (definitions in LICN-001, the license-type policy, exam tables in LICN-003, renewal rules, and so on), and one search on the question text finds only the main one. Retrieval recall@20 is 69%.

Phase 2 of [GAME_PLAN.md](../GAME_PLAN.md) fixes this with an agent that searches iteratively, reads whole sections, and follows cross-references. Plan 02 builds the agent loop. This plan builds the tools it calls:

| Tool | What it does | Adds passages to evidence? |
|---|---|---|
| `search` | Hybrid search (the existing index), optionally limited to one policy | yes |
| `get_policy_outline` | A policy's table of contents: section IDs, titles, word counts | no |
| `get_section` | Full text of a section and all its sub-items | yes |
| `follow_references` | Reads the sections a passage points to ("as provided in LICN-019", "Section 4 of this Policy") and lists passages in other policies that point back | yes |
| `list_license_types` | The license-type definitions (LICN-001 §2) and the teacher license list (LICN-010 §5) | yes |
| `calculate` | Safe arithmetic (CEU conversions, totals, shortfalls) | no (records the calculation) |

**Evidence** is the key idea. Every passage any tool shows the model is recorded in an `Evidence` object. In plan 02 the final answer is written by the *existing* `generate()` call using the *existing* answer prompt, with `Evidence.render()` as the passages. That keeps the agent and the RAG pipeline comparable: they share the answer prompt and differ only in **how the passages were chosen**, which is the project's evaluation principle ([answer/prompt.py](../src/licn_rag/answer/prompt.py) docstring).

## Read first

Read these files completely before writing code:

1. [src/licn_rag/corpus.py](../src/licn_rag/corpus.py): `Corpus`, `cite_id`, `normalize_citation`, `policy_of`, `Corpus.passage`. Citation IDs look like `LICN-030 §1(a)(ii)`, `LICN-030 (Policy Statement)`, or a bare policy code `LICN-008`.
2. [src/licn_rag/ingest/models.py](../src/licn_rag/ingest/models.py): `Policy`, `Block`. A block's `depth` equals the number of `(` in its `path` (`2` → 0, `2(a)` → 1, `2(a)(i)` → 2).
3. [src/licn_rag/rag/chunking.py](../src/licn_rag/rag/chunking.py): `Chunk`, `block_line`, `truncate`. **Chunk text is a sequence of `block_line()` outputs joined by `\n`.** Tables are multi-line (a `[ID] (table)` line followed by Markdown rows).
4. [src/licn_rag/rag/index.py](../src/licn_rag/rag/index.py): `Index.search(query, mode, k, candidates)` returns `Hit`s with `.chunk`. `Index(path, embedder=None)` works for BM25-only search.
5. [src/licn_rag/pipelines/rag.py](../src/licn_rag/pipelines/rag.py): `render_passages()`. `Evidence.render()` produces the same `<policy ...>` wrapper format.
6. [tests/test_score.py](../tests/test_score.py) and [tests/test_chunking.py](../tests/test_chunking.py): test style (module-scoped `corpus` fixture, plain asserts).

Facts about the corpus that this plan relies on (verified against `corpus/licn/` on 2026-10-10):

- 24 policies, 643 non-heading blocks, 623 unique citation IDs. **Citation IDs are not unique per block**: `LICN-003 §5` covers 6 blocks (paragraphs plus a large table), `LICN-006 §II` covers 4. Never assume one ID = one block.
- Two tables are split across chunks by the chunker: `LICN-003 §5` (89 rows) and the LICN-008 fee table (39 rows). A search hit can contain just *part* of a table.
- Policies reference each other in the text: `meets the requirements of LICN-019`, `LICN-007, Subsection 2(b)`, `LICN-014, Section 7`, and themselves: `Section 3.d. of this Policy`, `Subsection 9.a.ii`, `Subsection (2)(a)`, `Section 5 below`.
- `LICN-014` (Provisional License) is referenced by LICN-001, -010, and -011 but is **not in the corpus**. It is the only missing referenced policy.
- LICN-001 §2 is the definitions list. 12 definitions name a license type (their quoted term contains "license"): §2(b), (f), (h), (i), (k), (m), (n), (p), (q), (u), (w), (z). Definitions start with a curly quote: `“Residency license” means …`.

## Non-goals

- No LLM calls, no agent loop, no prompt text (plan 02).
- No changes to the eval runner or scoring (plan 03).
- Don't modify `corpus.py`, `chunking.py`, `index.py`, or any existing pipeline. Everything new goes in a new package `src/licn_rag/agent/`.
- No new dependencies.

## Steps

### Step 1: Create the package

Create an empty file `src/licn_rag/agent/__init__.py` (zero bytes, like the other packages' `__init__.py`).

**Check:** `uv run python -c "import licn_rag.agent"` exits 0.

---

### Step 2: The calculator (`src/licn_rag/agent/calculator.py`)

A safe arithmetic evaluator built on `ast`. **Never** use `eval()` or `exec()`. Write this file exactly:

```python
"""Safe arithmetic for the agent's `calculate` tool.

Models slip on unit conversions (semester hours to CEUs) and running totals, so the agent
does arithmetic here instead. Only numbers, + - * / // % **, parentheses, and the
functions round/min/max/abs are allowed; anything else raises ValueError.
"""

import ast
import operator
from collections.abc import Callable
from typing import Any

MAX_EXPRESSION_CHARS = 200
MAX_EXPONENT = 10

_BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_FUNCTIONS: dict[str, Callable[..., Any]] = {"round": round, "min": min, "max": max, "abs": abs}


def evaluate(expression: str) -> float:
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise ValueError(f"expression is longer than {MAX_EXPRESSION_CHARS} characters")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"not a valid arithmetic expression ({e.msg})") from None
    try:
        return _eval(tree.body)
    except (ZeroDivisionError, OverflowError, TypeError) as e:
        raise ValueError(f"cannot evaluate: {e}") from None


def _eval(node: ast.AST) -> Any:
    # type() rather than isinstance(): bool is a subclass of int and must be rejected.
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"exponents larger than {MAX_EXPONENT} are not allowed")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*(_eval(a) for a in node.args))
    raise ValueError("only numbers, + - * / // % **, and round/min/max/abs are allowed")


def format_number(value: float) -> str:
    """6.0 -> '6'; 3.3333333 -> '3.3333'."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return str(round(value, 4))
    return str(value)


def calculate(expression: str) -> str:
    """'6 * 2/3' -> '6 * 2/3 = 4'. Raises ValueError with a message the model can act on."""
    return f"{expression.strip()} = {format_number(evaluate(expression))}"
```

Create `tests/test_calculator.py`:

```python
"""The agent's safe calculator (no API calls)."""

import pytest

from licn_rag.agent.calculator import calculate, evaluate, format_number


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("6 * 2/3", "6 * 2/3 = 4"),
        ("8 - 5.5", "8 - 5.5 = 2.5"),
        ("10 / 3", "10 / 3 = 3.3333"),
        ("round(10 / 3, 2)", "round(10 / 3, 2) = 3.33"),
        ("max(0, 8 - 9.5)", "max(0, 8 - 9.5) = 0"),
        ("(3 + 2) * 10", "(3 + 2) * 10 = 50"),
        ("-4 + 1", "-4 + 1 = -3"),
    ],
)
def test_calculate(expression: str, expected: str) -> None:
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",
        "open('x')",
        "a + 1",
        "True + 1",
        "2 ** 100",
        "1 / 0",
        "'a' * 3",
        "[1, 2]",
        "1 +",
        "9" * 300,
    ],
)
def test_rejects_unsafe_or_invalid(expression: str) -> None:
    with pytest.raises(ValueError):
        evaluate(expression)


def test_format_number() -> None:
    assert format_number(6.0) == "6"
    assert format_number(7) == "7"
    assert format_number(2 / 3) == "0.6667"
```

**Check:** `uv run pytest tests/test_calculator.py -q` passes. If `round(10 / 3, 2)` fails, check that `_eval` evaluates `2` to an `int` (it does: `ast.Constant(2)` has an int value).

---

### Step 3: The evidence set (`src/licn_rag/agent/evidence.py`)

Stores every passage line a tool showed the model, deduplicated, and renders them grouped by policy in document order.

Design rules (read carefully, they handle the corpus quirks listed above):

- The unit of evidence is a **rendered block line**: exactly what `chunking.block_line(policy, block)` returns, e.g. `  [LICN-030 §1(a)] Teachers in Pre-Kindergarten …`. For tables, the "line" is multi-line (`[ID] (table)` plus the Markdown rows).
- Deduplicate on the line text with leading/trailing whitespace stripped.
- Chunks from search are split back into block lines with the regex `LINE_START` below. This was verified on all 113 chunks: the split always yields exactly one segment per entry in `chunk.cite_ids`. A partial table from a split table chunk is stored as its own line; if the full table is read later, both are kept (rare, acceptable).
- Sort key: the document position of the line's citation ID (the first block with that ID), then arrival order.

Write this file:

```python
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
```

`e.cite_id[:8]` is the policy code (`LICN-030`); every citation ID starts with it.

**Check:** `uv run pyright src` reports 0 errors. (Tests for this module come in step 6.)

---

### Step 4: Cross-references (`src/licn_rag/agent/references.py`)

Extracts references from the policy text into a graph. The regexes and the path conversion below were prototyped against the real corpus and produce 80 correct references (see "Expected references" in step 6).

Rules:

1. **Cross-policy references** match `CROSS_REF`: a policy code, optionally followed by `, Section X` / ` Subsection X`. A reference from a policy to itself by code is ignored.
2. **Same-policy references** match `SELF_REF` (`Section X` / `Subsection X`, capital S only), but are skipped when:
   - they fall inside a span already matched by `CROSS_REF` (so `LICN-007, Subsection 2(b)` isn't counted twice), or
   - the text right after the match is ` of <something other than "this">` (e.g. "Section 7 of Article 3" points outside the policy), or
   - they don't resolve to a real citation ID in the same policy.
3. **Path conversion** (`ref_path`): `9.a.ii` → `9(a)(ii)`; `3.d.` → `3(d)`; `(2)(a)` → `2(a)`; `2(b)` → `2(b)`; `5.` → `5`.
4. **Resolution** (`resolve`): try `CODE §path`, then `CODE §<first part of path>`, through `corpus.normalize`. If neither is valid but the policy exists, the target is the bare policy code. If the policy isn't in the corpus (LICN-014), keep the raw target and mark `in_corpus=False`.
5. Skip a reference whose target is the source passage itself or one of its ancestors (`LICN-007 §5(b)` saying "Subsection 5." points at its own parent).

Write this file:

```python
"""Cross-references between LICN passages, extracted from the policy text.

Policies point at each other ("meets the requirements of LICN-019", "LICN-007,
Subsection 2(b)") and at their own sections ("Section 4 of this Policy"). The agent's
follow_references tool uses this graph to read what a passage depends on and to find
passages in other policies that point back at it.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from functools import cache

from licn_rag.corpus import Corpus, cite_id, policy_of

CROSS_REF = re.compile(r"\b(LICN-\d{3})\b(?:,?\s+(?:Section|Subsection)\s+([0-9IVX][\w.()]*))?")
SELF_REF = re.compile(r"\b(?:Section|Subsection)\s+(\(?[0-9IVX][\w.()]*)")
OUTSIDE_TARGET = re.compile(r"\s+of\s+(?!this\b)")


@dataclass(frozen=True)
class Reference:
    source: str
    """Citation ID of the passage that contains the reference."""
    target: str
    """Citation ID ("LICN-007 §2(b)") or a whole policy ("LICN-019")."""
    in_corpus: bool


def ref_path(raw: str) -> str | None:
    """How a policy writes a section number -> our path: '9.a.ii' -> '9(a)(ii)'."""
    s = raw.strip().rstrip(".,")
    if s.startswith("("):
        parts = re.findall(r"\(([^)]+)\)", s)
    elif "(" in s:
        parts = [s[: s.index("(")].rstrip("."), *re.findall(r"\(([^)]+)\)", s)]
    else:
        parts = [p for p in s.split(".") if p]
    if not parts:
        return None
    return parts[0] + "".join(f"({p})" for p in parts[1:])


def resolve(corpus: Corpus, code: str, raw_section: str | None) -> tuple[str, bool]:
    path = ref_path(raw_section) if raw_section else None
    if path:
        for candidate in (f"{code} §{path}", f"{code} §{path.split('(')[0]}"):
            if found := corpus.normalize(candidate):
                return found, True
    if code in corpus.by_code:
        return code, True
    return (f"{code} §{path}" if path else code), False


def extract(corpus: Corpus) -> list[Reference]:
    refs: list[Reference] = []
    for policy in corpus.policies:
        for block in policy.blocks:
            if block.kind == "heading":
                continue
            source = cite_id(policy, block)
            text = block.text
            spans: list[tuple[int, int]] = []
            found: list[tuple[str, bool]] = []
            for m in CROSS_REF.finditer(text):
                spans.append(m.span())
                if m.group(1) != policy.code:
                    found.append(resolve(corpus, m.group(1), m.group(2)))
            for m in SELF_REF.finditer(text):
                if any(start <= m.start() < end for start, end in spans):
                    continue
                if OUTSIDE_TARGET.match(text, m.end()):
                    continue
                target, ok = resolve(corpus, policy.code, m.group(1))
                if ok and target != policy.code:
                    found.append((target, ok))
            for target, ok in found:
                if source == target or source.startswith(target + "("):
                    continue
                refs.append(Reference(source, target, ok))
    return list(dict.fromkeys(refs))  # dedupe, keep order


class ReferenceGraph:
    def __init__(self, corpus: Corpus) -> None:
        self.refs = extract(corpus)
        self._incoming: dict[str, list[Reference]] = defaultdict(list)
        for r in self.refs:
            target_code = policy_of(r.target)
            if target_code and target_code != policy_of(r.source):
                self._incoming[target_code].append(r)

    def from_passage(self, citation: str) -> list[Reference]:
        """References made by a passage or any of its sub-items, one per target."""
        hits = [
            r for r in self.refs if r.source == citation or r.source.startswith(citation + "(")
        ]
        if "§" not in citation and policy_of(citation) == citation:  # a whole policy
            hits = [r for r in self.refs if policy_of(r.source) == citation]
        return list({r.target: r for r in hits}.values())

    def citing(self, code: str) -> list[Reference]:
        """References from other policies that point at this policy or a section of it."""
        return list(self._incoming.get(code, []))

    def missing_policies(self) -> list[str]:
        return sorted({policy_of(r.target) or r.target for r in self.refs if not r.in_corpus})


@cache
def reference_graph(corpus: Corpus) -> ReferenceGraph:
    return ReferenceGraph(corpus)
```

Notes:

- `OUTSIDE_TARGET.match(text, m.end())` uses the compiled pattern's `pos` argument, so it checks only the text right after the match.
- `@cache` on a function that takes a `Corpus` works because `Corpus` objects are hashable by identity. The graph is built once per loaded corpus.

**Check:** `uv run pyright src` reports 0 errors.

---

### Step 5: The tools (`src/licn_rag/agent/tools.py`)

This is the biggest step. The module defines:

- one Pydantic input model per tool (the JSON schema sent to the API comes from it, so there's a single source of truth),
- `ToolOutcome`, the result of running a tool,
- `ToolBox`, which holds per-question state (evidence, calculations, trace) and runs tools by name.

**A `ToolBox` is created per question.** It is not thread-safe and must not be shared between questions. The `Corpus` and `Index` it receives are shared and read-only.

Constants (put them at the top of the module):

```python
SEARCH_K = 8                    # chunks per search call
SEED_K = 12                     # default for the seed search plan 02 runs before the loop
MAX_SECTION_WORDS = 1500        # get_section returns an outline instead above this
MAX_WHOLE_POLICY_WORDS = 800    # get_section on a bare policy code ("LICN-020") up to this size
OUTLINE_LEAD_WORDS = 15
MAX_CITING = 10                 # cap on "passages elsewhere that cite this policy"
LICENSE_TERM = re.compile(r'^\s*[“"]([^”"]+)[”"]')
```

#### 5a. Input models

Field descriptions matter: the model reads them. Use these exact descriptions.

```python
class SearchInput(BaseModel):
    query: str = Field(
        description="What to look for, in the policies' own vocabulary where possible "
        "(e.g. 'CEU renewal requirements for K-5 teachers', 'Praxis score elementary')."
    )
    policy: str | None = Field(
        default=None,
        description="Optional policy code such as 'LICN-030' to search only that policy.",
    )


class PolicyOutlineInput(BaseModel):
    code: str = Field(description="Policy code, e.g. 'LICN-019'.")


class CitationInput(BaseModel):
    citation_id: str = Field(
        description="A citation ID exactly as shown in brackets, e.g. 'LICN-030 §1(a)', "
        "'LICN-019 (Policy Statement)', or a bare policy code like 'LICN-020'."
    )


class NoInput(BaseModel):
    pass


class CalculateInput(BaseModel):
    description: str = Field(
        description="What is being computed, e.g. 'CEUs from 6 semester hours'."
    )
    expression: str = Field(
        description="Arithmetic only: numbers, + - * / // % **, parentheses, and "
        "round/min/max/abs. Example: '6 * 2/3'."
    )
```

`get_section` and `follow_references` both use `CitationInput`.

#### 5b. Tool specs

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    reads: bool
    """True if the tool adds passages to the evidence (subject to the reading budget)."""


SPECS: list[ToolSpec] = [
    ToolSpec(
        "search",
        "Hybrid keyword + semantic search over all LICN policy passages. Returns up to 8 "
        "passages with their citation IDs. Use specific terms; pass policy to search "
        "inside one policy. Passages you have already read are not repeated.",
        SearchInput,
        reads=True,
    ),
    ToolSpec(
        "get_policy_outline",
        "The table of contents of one policy: its sections and sub-sections with citation "
        "IDs, titles, and word counts. Use it to find the right section, then read it "
        "with get_section.",
        PolicyOutlineInput,
        reads=False,
    ),
    ToolSpec(
        "get_section",
        "The full text of a section and all of its sub-items, e.g. 'LICN-030 §1' returns "
        "§1, §1(a), §1(a)(i), and so on. A bare policy code returns the whole policy if it "
        "is short. Use this to read exceptions and conditions a search result cut off.",
        CitationInput,
        reads=True,
    ),
    ToolSpec(
        "follow_references",
        "Reads the sections a passage refers to (such as 'as provided in LICN-019' or "
        "'Section 4 of this Policy') and lists passages in other policies that refer to "
        "this passage's policy. Use it when a passage defers to another policy or section.",
        CitationInput,
        reads=True,
    ),
    ToolSpec(
        "list_license_types",
        "The definitions of every license type (LICN-001 §2) and the list of teacher "
        "license types with the policy that governs each (LICN-010 §5).",
        NoInput,
        reads=True,
    ),
    ToolSpec(
        "calculate",
        "Exact arithmetic for credits, CEUs, hours, years, and shortfalls. Use it instead "
        "of doing math yourself. Use only numbers from the question and the policies.",
        CalculateInput,
        reads=False,
    ),
]
```

#### 5c. Helpers

```python
@dataclass
class ToolOutcome:
    text: str
    is_error: bool = False
    added_words: int = 0


def tool_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema.pop("title", None)
    return schema


def blocks_under(policy: Policy, citation: str) -> list[Block]:
    """Non-heading blocks of a citation and its sub-items; a bare code means the policy."""
    if citation == policy.code:
        return [b for b in policy.blocks if b.kind != "heading"]
    return [
        b
        for b in policy.blocks
        if b.kind != "heading"
        and ((cid := cite_id(policy, b)) == citation or cid.startswith(citation + "("))
    ]


def word_count(policy: Policy, blocks: list[Block]) -> int:
    return sum(len(block_line(policy, b).split()) for b in blocks)


def citation_depth(citation: str) -> int:
    """'LICN-030 §1(a)' -> 1; '§1' -> 0; non-§ IDs and bare codes -> -1."""
    return citation.split("§", 1)[1].count("(") if "§" in citation else -1
```

#### 5d. `ToolBox`

Constructor and public methods:

```python
class ToolBox:
    def __init__(
        self,
        corpus: Corpus,
        index: Index | None,
        search_mode: str = "hybrid",
        max_evidence_words: int = 6000,
        calculator: bool = True,
    ) -> None:
        self.corpus = corpus
        self.index = index
        self.search_mode = search_mode
        self.max_evidence_words = max_evidence_words
        self.specs = [s for s in SPECS if calculator or s.name != "calculate"]
        self.graph = reference_graph(corpus)
        self.evidence = Evidence(corpus)
        self.calculations: list[str] = []
        self.trace: list[dict[str, Any]] = []
        self._handlers: dict[str, Callable[[Any], ToolOutcome]] = {
            "search": self._search,
            "get_policy_outline": self._get_policy_outline,
            "get_section": self._get_section,
            "follow_references": self._follow_references,
            "list_license_types": self._list_license_types,
            "calculate": self._calculate,
        }

    def definitions(self) -> list[ToolParam]:
        """Tool definitions for the API, sorted by name so the prompt cache stays valid."""
        return [
            {"name": s.name, "description": s.description, "input_schema": tool_schema(s.input_model)}
            for s in sorted(self.specs, key=lambda s: s.name)
        ]

    def run(self, name: str, raw_input: Any) -> ToolOutcome:
        """Run one tool call. Never raises: problems come back as is_error outcomes."""
        spec = next((s for s in self.specs if s.name == name), None)
        before = self.evidence.words()
        if spec is None:
            outcome = ToolOutcome(f"Unknown tool {name!r}.", is_error=True)
        elif spec.reads and before >= self.max_evidence_words:
            outcome = ToolOutcome(
                f"Reading budget reached ({before} words of policy text collected). "
                "Do not call more reading tools; reply DONE.",
                is_error=True,
            )
        else:
            try:
                args = spec.input_model.model_validate(raw_input)
            except ValidationError as e:
                outcome = ToolOutcome(f"Invalid input for {name}: {e.errors()[:2]}", is_error=True)
            else:
                outcome = self._handlers[name](args)
        outcome.added_words = self.evidence.words() - before
        self.trace.append(
            {"tool": name, "input": raw_input, "added_words": outcome.added_words, "error": outcome.is_error}
        )
        return outcome

    def seed_search(self, question: str, k: int = SEED_K) -> str:
        """The search plan 02 runs on the raw question before the agent's first turn."""
        before = self.evidence.words()
        outcome = self._show_chunks(self._search_chunks(question, None, k))
        self.trace.append(
            {"tool": "seed_search", "input": {"query": question, "k": k},
             "added_words": self.evidence.words() - before, "error": outcome.is_error}
        )
        return outcome.text
```

(Run `uv run ruff format` afterwards; it will re-wrap the long lines above.)

Now the handlers. Each takes the validated input model and returns a `ToolOutcome`. Each must handle bad input by returning an `is_error=True` outcome with a message that tells the model what to do instead. **Handlers must never raise.**

**`_search(args: SearchInput)`**

```python
def _search_chunks(self, query: str, policy: str | None, k: int) -> list[Chunk]:
    assert self.index is not None
    if policy is None:
        return [h.chunk for h in self.index.search(query, self.search_mode, k, 30)]
    n = len(self.index.chunks)
    hits = self.index.search(query, self.search_mode, n, n)
    return [h.chunk for h in hits if h.chunk.policy_code == policy][:k]

def _search(self, args: SearchInput) -> ToolOutcome:
    if self.index is None:
        return ToolOutcome("Search is unavailable; use get_policy_outline and get_section.", is_error=True)
    policy = args.policy.strip().upper() if args.policy else None
    if policy is not None and policy not in self.corpus.by_code:
        return ToolOutcome(f"Unknown policy {args.policy!r}. {self._known_codes()}", is_error=True)
    return self._show_chunks(self._search_chunks(args.query, policy, SEARCH_K))

def _show_chunks(self, chunks: list[Chunk]) -> ToolOutcome:
    shown: list[str] = []
    already: list[str] = []
    for chunk in chunks:
        if self.evidence.covers(chunk):
            already.append(chunk.cite_ids[0])
            continue
        self.evidence.add_chunk(chunk)
        shown.append(f"<!-- {chunk.breadcrumb} -->\n{chunk.text}")
    if not shown and not already:
        return ToolOutcome("No matching passages.")
    text = "\n\n".join(shown)
    if already:
        text += f"\n\n({len(already)} more results you have already read: {', '.join(already)})"
    return ToolOutcome(text.strip())

def _known_codes(self) -> str:
    return "Available policies: " + ", ".join(self.corpus.by_code)
```

When `self.index` is None, `seed_search` must also not crash. Guard it: if `self.index is None`, return `""` and don't append a trace entry.

**`_get_policy_outline(args: PolicyOutlineInput)`**

- Normalize: `code = args.code.strip().upper()`.
- If `code` is in `self.graph.missing_policies()`: error `"{code} is cited by other policies but is not in the available policy set."`.
- If `code` not in `self.corpus.by_code`: error with `self._known_codes()`.
- Otherwise build the outline with a helper you also reuse in `_get_section`:

```python
def _outline(self, policy: Policy, blocks: list[Block], max_depth: int) -> list[str]:
    """One line per citation ID at depth <= max_depth: '[ID] lead (N words)'."""
    lines: list[str] = []
    seen: set[str] = set()
    for b in blocks:
        cid = cite_id(policy, b)
        if b.depth > max_depth or cid in seen:
            continue
        seen.add(cid)
        size = word_count(policy, blocks_under(policy, cid))
        if b.kind == "table":
            lead = f"(table, {len(b.rows or [])} rows)"
        else:
            lead = b.heading or truncate(b.text, OUTLINE_LEAD_WORDS)
        lines.append(f"{'  ' * b.depth}[{cid}] {lead} ({size} words)")
    return lines
```

The outline text is:

```
LICN-019 Residency License · status ADOPTED · effective 2026-07-01 · 1101 words
Statutory citations: G.S. ...          (only if the policy has any)
NCAC citations: ...                    (only if the policy has any)
<outline lines for all non-heading blocks with max_depth=1>
Read any part with get_section.
```

Total policy words = `word_count(policy, blocks_under(policy, policy.code))`. The outline does **not** add evidence.

**`_get_section(args: CitationInput)`**

```python
cid = self.corpus.normalize(args.citation_id.strip())
```

- If `cid is None`: error `f"Unknown citation ID {args.citation_id!r}. Use get_policy_outline to see a policy's citation IDs."` (if the policy code is in `missing_policies()`, say that instead, as in the outline handler).
- `policy = self.corpus.by_code[cid[:8]]`, `blocks = blocks_under(policy, cid)`, `size = word_count(policy, blocks)`.
- `limit = MAX_WHOLE_POLICY_WORDS if cid == policy.code else MAX_SECTION_WORDS`.
- If `size > limit`: return a **non-error** outcome that lists the parts instead:
  `f"{cid} is {size} words, too long to read at once. Its parts:\n" + "\n".join(self._outline(policy, blocks, citation_depth(cid) + 1)) + "\nRead the parts you need with get_section."`
  For a bare policy code, `citation_depth` is -1, so this lists depth-0 items, which is what we want. Add no evidence in this case.
- Otherwise: `self.evidence.add_blocks(policy, blocks)` and return the lines `"\n".join(block_line(policy, b) for b in blocks)`.

**`_follow_references(args: CitationInput)`**

- Normalize like `_get_section`; error if unknown.
- `refs = self.graph.from_passage(cid)`.
- For each ref, build one entry:
  - `not r.in_corpus` → `f"- {r.target}: referenced, but not in the available policy set."`
  - `r.target` is a bare policy code (`"§" not in r.target and "(" not in r.target`) → `f"- {r.target} {title}: a whole policy. Use get_policy_outline('{r.target}') or search with policy='{r.target}'."` No evidence.
  - otherwise (a section): `blocks = blocks_under(target_policy, r.target)`. If `word_count(...) <= MAX_SECTION_WORDS` **and** `self.evidence.words() < self.max_evidence_words`: add the blocks to evidence and include their lines under a header `f"- {r.target} (referenced by {r.source}):"`. Otherwise give a pointer: `f"- {r.target}: use get_section to read it."`.
- Then, citing passages: `citing = [r.source for r in self.graph.citing(cid[:8])]`, deduped, first `MAX_CITING`. If any: append `f"\nPassages in other policies that refer to {cid[:8]}: " + ", ".join(citing)`.
- If there are no refs and no citing passages: return `ToolOutcome(f"No cross-references found in {cid}.")`.

**`_list_license_types(args: NoInput)`**

```python
defs = self.corpus.by_code.get("LICN-001")
types = self.corpus.by_code.get("LICN-010")
parts: list[str] = []
if defs:
    blocks: list[Block] = []
    for b in defs.blocks:
        m = LICENSE_TERM.match(b.text)
        if b.path and re.fullmatch(r"2\([a-z]+\)", b.path) and m and "license" in m.group(1).lower():
            blocks += blocks_under(defs, cite_id(defs, b))
    self.evidence.add_blocks(defs, blocks)
    parts.append("License definitions (LICN-001 §2):\n" + "\n".join(block_line(defs, b) for b in blocks))
if types:
    blocks = blocks_under(types, "LICN-010 §5")
    self.evidence.add_blocks(types, blocks)
    parts.append("Teacher license types and their policies (LICN-010 §5):\n" + "\n".join(block_line(types, b) for b in blocks))
return ToolOutcome("\n\n".join(parts) or "No license definitions found.", is_error=not parts)
```

`blocks_under` for `LICN-001 §2(f)` (Clear license) also includes its sub-items (i)–(v). That's intended.

**`_calculate(args: CalculateInput)`**

```python
try:
    result = calculate(args.expression)
except ValueError as e:
    return ToolOutcome(f"Calculation failed: {e}", is_error=True)
self.calculations.append(f"{args.description.strip()}: {result}")
return ToolOutcome(result)
```

#### 5e. Imports

The module needs roughly: `re`, `Callable` (from `collections.abc`), `dataclass`, `Any`, `ToolParam` (from `anthropic.types`), `BaseModel, Field, ValidationError` (from `pydantic`), `calculate`, `Evidence`, `reference_graph`, `Corpus, cite_id`, `Block, Policy`, `Chunk, block_line, truncate`, `Index`. Let ruff's import sorter (`ruff check --fix`) order them.

Module docstring:

```python
"""The research agent's tools: search, outline, read, follow references, calculate.

Each tool returns plain text for the model and records the passages it showed in an
Evidence set, which becomes the answer step's context. Tools never raise: bad input comes
back as an error message the model can act on.
"""
```

**Check:** `uv run ruff check . && uv run pyright src` pass. If pyright complains that a `dict` isn't assignable to `ToolParam` in `definitions()`, build each entry as a `ToolParam(...)` call instead (it's a `TypedDict`): `ToolParam(name=..., description=..., input_schema=...)`.

---

### Step 6: Tests

#### 6a. `tests/test_agent_references.py`

These expectations were verified against the current corpus:

```python
"""Cross-reference extraction over the real corpus (no API calls)."""

import pytest

from licn_rag.agent.references import ReferenceGraph, ref_path, reference_graph
from licn_rag.corpus import Corpus


@pytest.fixture(scope="module")
def graph() -> ReferenceGraph:
    return reference_graph(Corpus.load())


@pytest.mark.parametrize(
    ("raw", "path"),
    [("9.a.ii", "9(a)(ii)"), ("3.d.", "3(d)"), ("(2)(a)", "2(a)"), ("2(b)", "2(b)"), ("5.", "5")],
)
def test_ref_path(raw: str, path: str) -> None:
    assert ref_path(raw) == path


def targets(graph: ReferenceGraph, citation: str) -> set[str]:
    return {r.target for r in graph.from_passage(citation)}


def test_cross_policy_reference(graph: ReferenceGraph) -> None:
    assert targets(graph, "LICN-010 §5(c)") == {"LICN-019"}


def test_cross_policy_reference_to_a_subsection(graph: ReferenceGraph) -> None:
    assert "LICN-007 §2(b)" in targets(graph, "LICN-004 §1(h)(ii)(C)")


def test_same_policy_references(graph: ReferenceGraph) -> None:
    assert {"LICN-007 §3(d)", "LICN-007 §3", "LICN-007 §4"} <= targets(graph, "LICN-007 §5(c)")
    assert "LICN-019 §2(a)" in targets(graph, "LICN-019 §5(e)")


def test_parent_section_includes_children(graph: ReferenceGraph) -> None:
    # §5's sub-items point at nine license policies.
    assert {"LICN-019", "LICN-020", "LICN-024"} <= targets(graph, "LICN-010 §5")


def test_no_reference_to_own_ancestor(graph: ReferenceGraph) -> None:
    assert "LICN-007 §5" not in targets(graph, "LICN-007 §5(b)")


def test_citing(graph: ReferenceGraph) -> None:
    sources = {r.source for r in graph.citing("LICN-019")}
    assert {"LICN-001 §2(w)", "LICN-010 §5(c)", "LICN-028 §5"} <= sources


def test_missing_policy(graph: ReferenceGraph) -> None:
    assert graph.missing_policies() == ["LICN-014"]
    assert any(not r.in_corpus for r in graph.from_passage("LICN-001 §2(u)"))
```

#### 6b. `tests/test_agent_tools.py`

Corpus-only tests construct `ToolBox(corpus, None)`. Search tests need the built index (`data/index/licn.sqlite`, git-ignored) and are skipped when it's missing. They load the index **without an embedder** and use `search_mode="bm25"`, so they're fast and don't download the 1.2 GB embedding model.

Write tests for each of these behaviors (the test names are suggestions):

```python
"""The research agent's tools (no API calls)."""

from typing import Any

import pytest

from licn_rag.agent.evidence import Evidence, split_chunk
from licn_rag.agent.tools import ToolBox
from licn_rag.corpus import Corpus
from licn_rag.rag.chunking import chunk_corpus
from licn_rag.rag.index import INDEX_DIR, Index

INDEX_PATH = INDEX_DIR / "licn.sqlite"
needs_index = pytest.mark.skipif(not INDEX_PATH.exists(), reason="run `uv run licn-index build`")


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return Corpus.load()


@pytest.fixture
def box(corpus: Corpus) -> ToolBox:
    return ToolBox(corpus, None)
```

| Test | Assert |
|---|---|
| `test_definitions_sorted_and_valid` | `[d["name"] for d in box.definitions()]` equals the sorted list of 6 names; every `input_schema["type"] == "object"`. With `ToolBox(corpus, None, calculator=False)`, `"calculate"` is absent. |
| `test_get_section_reads_subtree` | `run("get_section", {"citation_id": "LICN-030 §1"})` is not an error; its text contains `[LICN-030 §1(a)]` and `[LICN-030 §1(a)(i)]`; `box.evidence.ids()` starts with `"LICN-030 §1"` and includes `"LICN-030 §1(a)(i)"`; `outcome.added_words > 0`. |
| `test_get_section_accepts_variants` | `{"citation_id": "LICN-030 Section 1(a)"}` works (normalize handles it). |
| `test_get_section_unknown_id` | `{"citation_id": "LICN-030 §99"}` → `is_error`, message mentions `get_policy_outline`. |
| `test_get_section_missing_policy` | `{"citation_id": "LICN-014"}` → `is_error`, message mentions "not in the available policy set". |
| `test_get_section_too_long_returns_outline` | `{"citation_id": "LICN-007"}` (3018 words, over 800) → **not** an error, text contains `"too long"` and `"[LICN-007 §1]"`, and `len(box.evidence) == 0`. |
| `test_get_section_short_whole_policy` | `{"citation_id": "LICN-020"}` (174 words) → not an error, evidence non-empty. |
| `test_outline` | `run("get_policy_outline", {"code": "licn-019"})` (lowercase on purpose) → text contains `"LICN-019 Residency License"` and `"[LICN-019 §1]"`; evidence stays empty. |
| `test_follow_references_reads_sections` | `{"citation_id": "LICN-004 §1(h)(ii)(C)"}` → text contains `"LICN-007 §2(b)"`; `"LICN-007 §2(b)"` in `box.evidence.ids()`. |
| `test_follow_references_whole_policy_pointer` | `{"citation_id": "LICN-010 §5(c)"}` → text contains `"get_policy_outline('LICN-019')"`. |
| `test_follow_references_lists_citing` | `{"citation_id": "LICN-019 §1"}` → text contains `"LICN-001 §2(w)"`. (Check `LICN-019 §1` exists with `corpus.normalize` first; if it doesn't, use the first citation ID in `get_policy_outline("LICN-019")`.) |
| `test_list_license_types` | not an error; text contains `"Residency license"` and `"[LICN-010 §5(c)]"`; `{"LICN-001 §2(w)", "LICN-001 §2(h)", "LICN-010 §5(c)"} <= set(box.evidence.ids())`. |
| `test_calculate_records` | `{"description": "CEUs from 6 semester hours", "expression": "6 * 2/3"}` → text `"6 * 2/3 = 4"`; `box.calculations == ["CEUs from 6 semester hours: 6 * 2/3 = 4"]`. |
| `test_calculate_error` | `{"description": "x", "expression": "import os"}` → `is_error`; `box.calculations == []`. |
| `test_invalid_input_and_unknown_tool` | `run("get_section", {})` → `is_error` mentioning "Invalid input"; `run("nope", {})` → `is_error`. Neither raises. |
| `test_reading_budget` | `ToolBox(corpus, None, max_evidence_words=50)`: first `get_section LICN-030 §1` succeeds; a second `get_section LICN-020` → `is_error`, message contains `"Reading budget"`; `calculate` still works after the budget is hit. |
| `test_trace` | after two calls, `len(box.trace) == 2` and each entry has keys `tool, input, added_words, error`. |
| `test_search_without_index` | `run("search", {"query": "CEU"})` with `index=None` → `is_error`; `box.seed_search("CEU") == ""`. |

Evidence tests (same file):

| Test | Assert |
|---|---|
| `test_split_chunk_matches_cite_ids` | for every chunk in `chunk_corpus(corpus.policies)`: `len(split_chunk(c.text)) == len(c.cite_ids)`. |
| `test_evidence_dedupes_and_orders` | add the LICN-030 chunks in reverse order, then add one of them again: second add returns 0; `ids()` is in document order (equal to `sorted(ids, key=doc_order(corpus).__getitem__)`). |
| `test_evidence_render` | after adding one LICN-030 and one LICN-001 chunk, `render()` contains exactly one `<policy code="LICN-001"`, one `<policy code="LICN-030"`, LICN-001 comes first, and the number of `</policy>` equals 2. |
| `test_add_line_rejects_non_passage` | `Evidence(corpus).add_line("hello")` raises `ValueError`. |

Index-backed tests (decorate with `@needs_index`; build the box with `ToolBox(corpus, Index(INDEX_PATH), search_mode="bm25")`):

| Test | Assert |
|---|---|
| `test_search_returns_passages` | `run("search", {"query": "renewal CEUs literacy"})` → not an error, text contains `"[LICN-"`, evidence non-empty. |
| `test_search_policy_filter` | `{"query": "license", "policy": "licn-030"}` → every `[LICN-…]` ID in the text starts with `LICN-030`. Use `re.findall(r"\[(LICN-\d{3})", text)`. |
| `test_search_skips_already_read` | run the same search twice: the second text contains `"already read"` and `added_words == 0`. |
| `test_search_unknown_policy` | `{"query": "x", "policy": "LICN-999"}` → `is_error`. |
| `test_seed_search` | `box.seed_search("Praxis elementary", k=5)` returns non-empty text and the trace's last entry has `tool == "seed_search"`. |

**Check:** `uv run pytest -q` passes (all old tests plus the new ones).

## Acceptance checklist

- [ ] `uv run pytest -q`: all tests pass, none skipped on a machine that has `data/index/licn.sqlite`.
- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass (run `uv run ruff format .` to fix formatting).
- [ ] `uv run pyright src` reports 0 errors.
- [ ] No file outside `src/licn_rag/agent/`, `tests/`, and `implementation_plans/` changed (`git status`).
- [ ] No `eval(` or `exec(` anywhere in `src/licn_rag/agent/`.
- [ ] Manual smoke test (free) prints sensible output:

```bash
uv run python -c "
from licn_rag.corpus import Corpus
from licn_rag.agent.tools import ToolBox
box = ToolBox(Corpus.load(), None)
for name, args in [('get_policy_outline', {'code': 'LICN-024'}),
                   ('follow_references', {'citation_id': 'LICN-024 §3'}),
                   ('calculate', {'description': 'CEUs short', 'expression': '8 - 6.5'})]:
    out = box.run(name, args)
    print('==', name, out.is_error, out.added_words); print(out.text[:800])
print(box.evidence.render()[:1500])
"
```

## Commit

Branch `plan-01-research-tools`. One commit:

```
Add agent research tools (implementation plan 01)

Search, policy outline, section reading, cross-reference following, license
type listing, and a safe calculator, plus an Evidence set that records every
passage the tools show. No LLM calls yet; the agent loop is plan 02.
```

Then set this plan's **Status** to `done` here and in [README.md](README.md)'s index, in the same commit.

## Gotchas

- **Windows console encoding.** `§` and curly quotes print as `�` in a cp1252 console. That's display only. If a quick script crashes on printing, set `PYTHONIOENCODING=utf-8`. Don't "fix" the data.
- **`corpus.normalize` returns IDs that may cover several blocks** (`LICN-003 §5`). Always go through `blocks_under`, never look up "the" block for an ID.
- **Don't change `Index`.** For policy-filtered search, ask the index for all 113 chunks and filter. It's fast enough at this corpus size.
- **Don't make the ToolBox a singleton.** Plan 02 runs 8 questions in parallel threads; each question gets its own ToolBox.
- **`model_json_schema()` output for `SearchInput.policy`** contains `anyOf: [{type: string}, {type: null}]` and `default: null`. That's fine for normal (non-strict) tools. Don't add `strict: true` to the tool definitions.
