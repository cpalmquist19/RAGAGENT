"""The research agent's tools: search, outline, read, follow references, calculate.

Each tool returns plain text for the model and records the passages it showed in an
Evidence set, which becomes the answer step's context. Tools never raise: bad input comes
back as an error message the model can act on.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from anthropic.types import ToolParam
from pydantic import BaseModel, Field, ValidationError

from licn_rag.agent.calculator import calculate
from licn_rag.agent.evidence import Evidence
from licn_rag.agent.references import reference_graph
from licn_rag.corpus import Corpus, cite_id
from licn_rag.ingest.models import Block, Policy
from licn_rag.rag.chunking import Chunk, block_line, truncate
from licn_rag.rag.index import Index

SEARCH_K = 8
SEED_K = 12
MAX_SECTION_WORDS = 1500
MAX_WHOLE_POLICY_WORDS = 800
OUTLINE_LEAD_WORDS = 15
MAX_CITING = 10
LICENSE_TERM = re.compile(r'^\s*[“"]([^”"]+)[”"]')


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
            ToolParam(
                name=s.name,
                description=s.description,
                input_schema=tool_schema(s.input_model),
            )
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
            {
                "tool": name,
                "input": raw_input,
                "added_words": outcome.added_words,
                "error": outcome.is_error,
            }
        )
        return outcome

    def seed_search(self, question: str, k: int = SEED_K) -> str:
        """The search plan 02 runs on the raw question before the agent's first turn."""
        if self.index is None:
            return ""
        before = self.evidence.words()
        outcome = self._show_chunks(self._search_chunks(question, None, k))
        self.trace.append(
            {
                "tool": "seed_search",
                "input": {"query": question, "k": k},
                "added_words": self.evidence.words() - before,
                "error": outcome.is_error,
            }
        )
        return outcome.text

    def _search_chunks(self, query: str, policy: str | None, k: int) -> list[Chunk]:
        assert self.index is not None
        if policy is None:
            return [h.chunk for h in self.index.search(query, self.search_mode, k, 30)]
        n = len(self.index.chunks)
        hits = self.index.search(query, self.search_mode, n, n)
        return [h.chunk for h in hits if h.chunk.policy_code == policy][:k]

    def _search(self, args: SearchInput) -> ToolOutcome:
        if self.index is None:
            return ToolOutcome(
                "Search is unavailable; use get_policy_outline and get_section.", is_error=True
            )
        policy = args.policy.strip().upper() if args.policy else None
        if policy is not None and policy not in self.corpus.by_code:
            return ToolOutcome(
                f"Unknown policy {args.policy!r}. {self._known_codes()}", is_error=True
            )
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

    def _policy_header_line(self, policy: Policy) -> str:
        eff = policy.effective_date or "unknown"
        status = policy.status or "unknown"
        total = word_count(policy, blocks_under(policy, policy.code))
        return f"{policy.code} {policy.title} · status {status} · effective {eff} · {total} words"

    def _get_policy_outline(self, args: PolicyOutlineInput) -> ToolOutcome:
        code = args.code.strip().upper()
        if code in self.graph.missing_policies():
            return ToolOutcome(
                f"{code} is cited by other policies but is not in the available policy set.",
                is_error=True,
            )
        if code not in self.corpus.by_code:
            return ToolOutcome(
                f"Unknown policy {args.code!r}. {self._known_codes()}", is_error=True
            )
        policy = self.corpus.by_code[code]
        blocks = blocks_under(policy, policy.code)
        lines = [self._policy_header_line(policy)]
        if policy.statutory_citations:
            lines.append(f"Statutory citations: {'; '.join(policy.statutory_citations)}")
        if policy.ncac_citations:
            lines.append(f"NCAC citations: {'; '.join(policy.ncac_citations)}")
        lines.extend(self._outline(policy, blocks, 1))
        lines.append("Read any part with get_section.")
        return ToolOutcome("\n".join(lines))

    def _unknown_citation_message(self, raw: str) -> ToolOutcome:
        code = raw.strip().upper()
        if re.match(r"LICN-\d{3}", code) and code[:8] in self.graph.missing_policies():
            return ToolOutcome(
                f"{code[:8]} is cited by other policies but is not in the available policy set.",
                is_error=True,
            )
        return ToolOutcome(
            f"Unknown citation ID {raw!r}. Use get_policy_outline to see a policy's citation IDs.",
            is_error=True,
        )

    def _get_section(self, args: CitationInput) -> ToolOutcome:
        cid = self.corpus.normalize(args.citation_id.strip())
        if cid is None:
            return self._unknown_citation_message(args.citation_id)
        policy = self.corpus.by_code[cid[:8]]
        blocks = blocks_under(policy, cid)
        size = word_count(policy, blocks)
        limit = MAX_WHOLE_POLICY_WORDS if cid == policy.code else MAX_SECTION_WORDS
        if size > limit:
            parts = self._outline(policy, blocks, citation_depth(cid) + 1)
            text = (
                f"{cid} is {size} words, too long to read at once. Its parts:\n"
                + "\n".join(parts)
                + "\nRead the parts you need with get_section."
            )
            return ToolOutcome(text)
        text = "\n".join(block_line(policy, b) for b in blocks)
        self.evidence.add_blocks(policy, blocks)
        return ToolOutcome(text)

    def _follow_references(self, args: CitationInput) -> ToolOutcome:
        cid = self.corpus.normalize(args.citation_id.strip())
        if cid is None:
            return self._unknown_citation_message(args.citation_id)
        refs = self.graph.from_passage(cid)
        parts: list[str] = []
        for r in refs:
            if not r.in_corpus:
                parts.append(f"- {r.target}: referenced, but not in the available policy set.")
                continue
            if "§" not in r.target and "(" not in r.target:
                title = self.corpus.by_code[r.target].title
                parts.append(
                    f"- {r.target} {title}: a whole policy. Use get_policy_outline('{r.target}') "
                    f"or search with policy='{r.target}'."
                )
                continue
            target_code = r.target[:8]
            target_policy = self.corpus.by_code[target_code]
            ref_blocks = blocks_under(target_policy, r.target)
            wc = word_count(target_policy, ref_blocks)
            if wc <= MAX_SECTION_WORDS and self.evidence.words() < self.max_evidence_words:
                self.evidence.add_blocks(target_policy, ref_blocks)
                body = "\n".join(block_line(target_policy, b) for b in ref_blocks)
                parts.append(f"- {r.target} (referenced by {r.source}):\n{body}")
            else:
                parts.append(f"- {r.target}: use get_section to read it.")
        citing = list(dict.fromkeys(r.source for r in self.graph.citing(cid[:8])))[:MAX_CITING]
        if citing:
            parts.append(
                f"\nPassages in other policies that refer to {cid[:8]}: " + ", ".join(citing)
            )
        if not parts:
            return ToolOutcome(f"No cross-references found in {cid}.")
        return ToolOutcome("\n".join(parts))

    def _list_license_types(self, args: NoInput) -> ToolOutcome:
        defs = self.corpus.by_code.get("LICN-001")
        types = self.corpus.by_code.get("LICN-010")
        parts: list[str] = []
        if defs:
            blocks: list[Block] = []
            for b in defs.blocks:
                m = LICENSE_TERM.match(b.text)
                if (
                    b.path
                    and re.fullmatch(r"2\([a-z]+\)", b.path)
                    and m
                    and "license" in m.group(1).lower()
                ):
                    blocks += blocks_under(defs, cite_id(defs, b))
            self.evidence.add_blocks(defs, blocks)
            parts.append(
                "License definitions (LICN-001 §2):\n"
                + "\n".join(block_line(defs, b) for b in blocks)
            )
        if types:
            blocks = blocks_under(types, "LICN-010 §5")
            self.evidence.add_blocks(types, blocks)
            parts.append(
                "Teacher license types and their policies (LICN-010 §5):\n"
                + "\n".join(block_line(types, b) for b in blocks)
            )
        return ToolOutcome(
            "\n\n".join(parts) or "No license definitions found.", is_error=not parts
        )

    def _calculate(self, args: CalculateInput) -> ToolOutcome:
        try:
            result = calculate(args.expression)
        except ValueError as e:
            return ToolOutcome(f"Calculation failed: {e}", is_error=True)
        self.calculations.append(f"{args.description.strip()}: {result}")
        return ToolOutcome(result)
