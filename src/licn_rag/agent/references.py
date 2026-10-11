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
        hits = [r for r in self.refs if r.source == citation or r.source.startswith(citation + "(")]
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
