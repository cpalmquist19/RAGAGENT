"""Load the parsed LICN corpus and give every block a stable, human-readable citation ID.

Citation IDs follow the policies' own numbering: "LICN-030 §2(b)(iv)". Blocks outside a
numbered list are cited by their section heading ("LICN-030 (Policy Statement)"). The
same IDs are used in prompts, answer keys, and the app's citations, so evals can compare
them directly.
"""

import re
from pathlib import Path

from licn_rag.ingest.models import Block, Policy

DEFAULT_CORPUS_DIR = Path("corpus/licn")


def load_policies(corpus_dir: Path = DEFAULT_CORPUS_DIR) -> list[Policy]:
    paths = sorted(corpus_dir.glob("LICN-*.json"))
    if not paths:
        raise FileNotFoundError(f"no LICN-*.json in {corpus_dir}; run `uv run licn-ingest` first")
    return [Policy.model_validate_json(p.read_text(encoding="utf-8")) for p in paths]


def cite_id(policy: Policy, block: Block) -> str:
    if block.path:
        return f"{policy.code} §{block.path}"
    if block.section:
        return f"{policy.code} ({block.section})"
    return f"{policy.code} (preamble)"


def all_cite_ids(policies: list[Policy]) -> set[str]:
    return {cite_id(p, b) for p in policies for b in p.blocks if b.kind != "heading"}


def citation_is_valid(citation: str, valid: set[str]) -> bool:
    """Exact match, or a parent of a real block (citing "LICN-030 §1" when §1(a) exists)."""
    if citation in valid:
        return True
    return any(v.startswith(citation + "(") for v in valid if "§" in citation)


def normalize_citation(citation: str, valid: set[str]) -> str | None:
    """Map a model-written citation onto a real citation ID, or None if it isn't one.

    Accepts the canonical form, a whole-policy citation ("LICN-008", for policies with no
    numbered sections or for "the policy is silent" points), and common variants:
    "LICN-019 Section 1(a)", "LICN-003 §5 (table)", "LICN-019 Policy Statement".
    """
    c = re.sub(r"\bSection\s+", "§", citation.strip())
    if citation_is_valid(c, valid):
        return c
    code = policy_of(c)
    if code is None or not any(v.startswith(code + " ") for v in valid):
        return None
    if c == code:
        return code
    if (m := re.match(r"(LICN-\d{3}) §([\w()]+)", c)) and citation_is_valid(m.group(), valid):
        return m.group()  # drop trailing descriptions like "(table)" or "footnote **"
    rest = c[len(code) :].strip().strip("()")
    if f"{code} ({rest})" in valid:
        return f"{code} ({rest})"
    return None


def policy_of(citation: str) -> str | None:
    m = re.match(r"LICN-\d{3}", citation)
    return m.group() if m else None


def render_policy(policy: Policy) -> str:
    """One policy as tagged text, every citable block prefixed with its [citation ID]."""
    attrs = {
        "code": policy.code,
        "title": policy.title,
        "status": policy.status or "unknown",
        "effective": str(policy.effective_date or "unknown"),
        "last_revised": str(
            policy.last_revised or next(iter(policy.history.revised), None) or "unknown"
        ),
    }
    head = " ".join(f'{k}="{v}"' for k, v in attrs.items())
    lines = [f"<policy {head}>"]
    if policy.statutory_citations:
        lines.append(f"Statutory citations: {'; '.join(policy.statutory_citations)}")
    if policy.ncac_citations:
        lines.append(f"NCAC citations: {'; '.join(policy.ncac_citations)}")
    for b in policy.blocks:
        indent = "  " * b.depth
        if b.kind == "heading":
            lines.append(f"\n## {b.text}")
        elif b.kind == "table":
            lines.append(f"{indent}[{cite_id(policy, b)}] (table)\n{b.text}")
        else:
            title = f"{b.heading}. " if b.heading else ""
            text = b.text.replace("\n", " ")
            lines.append(f"{indent}[{cite_id(policy, b)}] {title}{text}".rstrip())
    lines.append("</policy>")
    return "\n".join(lines)


def render_corpus(policies: list[Policy]) -> str:
    return "\n\n".join(render_policy(p) for p in policies)


class Corpus:
    """The loaded policies plus lookups by citation ID."""

    def __init__(self, policies: list[Policy]) -> None:
        self.policies = policies
        self.by_code = {p.code: p for p in policies}
        self.valid_ids = all_cite_ids(policies)

    @classmethod
    def load(cls, corpus_dir: Path = DEFAULT_CORPUS_DIR) -> "Corpus":
        return cls(load_policies(corpus_dir))

    def render(self) -> str:
        return render_corpus(self.policies)

    def normalize(self, citation: str) -> str | None:
        return normalize_citation(citation, self.valid_ids)

    def passage(self, citation: str, max_chars: int = 6000) -> str | None:
        """Text behind a citation: the block and its sub-items, or a whole policy."""
        code = policy_of(citation)
        policy = self.by_code.get(code or "")
        if policy is None:
            return None
        if citation == policy.code:
            return render_policy(policy)[:max_chars]
        lines = []
        for b in policy.blocks:
            cid = cite_id(policy, b)
            if b.kind != "heading" and (cid == citation or cid.startswith(citation + "(")):
                title = f"{b.heading}. " if b.heading else ""
                lines.append(f"[{cid}] {title}{b.text}")
        return "\n".join(lines)[:max_chars] or None
