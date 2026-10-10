"""Data model for parsed LICN policies.

A Policy is the cleaned, structured form of one saved Simbli policy page. Blocks keep
the policy's own numbering (e.g. path "2(a)(i)") so citations can match the way the
policies cite themselves ("Subsection 2(a)").
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

BlockKind = Literal["heading", "paragraph", "list_item", "table"]


class Block(BaseModel):
    kind: BlockKind
    text: str
    depth: int = 0
    path: str | None = None
    """Numbered-list position, e.g. "2(a)(i)". None outside numbered lists."""
    label: str | None = None
    """How the number is displayed: "2." at the top level, "(a)" / "(i)" below."""
    heading: str | None = None
    """Bold title that opens a list item, e.g. "Continuing Education"."""
    section: str | None = None
    """Nearest enclosing heading (top-level list item title or bold heading line)."""
    rows: list[list[str]] | None = None
    """Table cells, row by row. Only set for kind == "table"."""


class PolicyHistory(BaseModel):
    adopted: str | None = None
    revised: list[str] = Field(default_factory=list)
    raw: str = ""


class Policy(BaseModel):
    code: str
    title: str
    series: str | None = None
    status: str | None = None
    """"ADOPTED", or "PENDING" when the saved URL is a pending revision."""
    effective_date: date | None = None
    original_adopted: date | None = None
    last_revised: date | None = None
    last_reviewed: date | None = None
    statutory_citations: list[str] = Field(default_factory=list)
    ncac_citations: list[str] = Field(default_factory=list)
    source_url: str | None = None
    source_file: str
    sha256: str
    """Hash of the saved HTML file, used to detect changed re-saves."""
    blocks: list[Block]
    history: PolicyHistory
    warnings: list[str] = Field(default_factory=list)
