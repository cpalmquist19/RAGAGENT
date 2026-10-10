"""Parse a Simbli "View Policy" page saved from a browser into a structured Policy.

The saved pages carry a lot of site chrome. The policy itself lives in
`div.ICE-Tracking`: a header table (title, code, citations, effective date), the body,
and a Policy History table. Body markup varies a lot between policies (some are clean
nested <ol>s, some are Word exports with one <span> per word), so the parser does not
rely on layout. It flattens the DOM into a stream of text / line-break / list / table
events and rebuilds lines from that.
"""

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

from licn_rag.ingest.models import Block, Policy, PolicyHistory

WS = re.compile(r"\s+")
SAVED_FROM = re.compile(r"<!-- saved from url=\(\d+\)(\S+) -->")
DATE_NUMERIC = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")
NCAC_CITE = re.compile(r"(?:16\s*NCAC\s*)?(?P<cite>0?6C\s*\.\d{4}(?:\s*[–-]\s*\.\d{4})?)", re.I)
DATE_WORDS = re.compile(r"\b[A-Z][a-z]{2,8}\.? \d{1,2}, \d{4}\b")

BLOCK_TAGS = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "tr"}
LIST_TAGS = {"ol", "ul"}
BOLD_TAGS = {"strong", "b"}
DROP_TAGS = ["script", "style", "noscript", "iframe", "link"]

# Default numbering by nesting depth when an <ol> doesn't say; matches the 1 / (a) / (i)
# convention the policies use when they cite themselves.
DEFAULT_STYLES = ["decimal", "lower-alpha", "lower-roman", "upper-alpha"]
TYPE_ATTR_STYLES = {
    "1": "decimal",
    "a": "lower-alpha",
    "A": "upper-alpha",
    "i": "lower-roman",
    "I": "upper-roman",
}

HEADING_MAX_CHARS = 120


def clean(text: str) -> str:
    return WS.sub(" ", text.replace("\xa0", " ")).strip()


# ---------------------------------------------------------------- dates & citations


def parse_date(text: str | None) -> date | None:
    if not text:
        return None
    if m := DATE_NUMERIC.search(text):
        return datetime.strptime(m.group(), "%m/%d/%Y").date()
    if m := DATE_WORDS.search(text):
        raw = m.group().replace(".", "").replace("Sept ", "Sep ")
        for fmt in ("%B %d, %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(raw, fmt).date()
            except ValueError:
                continue
    return None


def split_statutes(raw: str) -> list[str]:
    """'G.S. 115C-12; 115C-270.1' -> ['G.S. 115C-12', 'G.S. 115C-270.1']."""
    out = []
    for part in re.split(r"[;,]", raw):
        part = clean(part)
        if not part:
            continue
        part = re.sub(r"^(G\.?S\.?|GS)\s*", "", part)
        # Citations never contain spaces; older pages have "115C - 12" or "11 5C - 270.20".
        out.append(f"G.S. {re.sub(r'\s+', '', part)}")
    return out


def split_ncac(raw: str) -> list[str]:
    """'16 NCAC 06C .0301; 06C .0302' -> ['16 NCAC 06C .0301', '16 NCAC 06C .0302'].

    Only the citation itself is kept; some header cells run into the following heading.
    """
    out = []
    for part in raw.split(";"):
        if m := NCAC_CITE.search(part):
            out.append(f"16 NCAC {clean(m.group('cite'))}")
    return out


# ---------------------------------------------------------------- DOM flattening


@dataclass
class TextEvent:
    text: str
    bold: bool


@dataclass
class BreakEvent:
    pass


@dataclass
class ListEvent:
    el: Tag


@dataclass
class TableEvent:
    el: Tag


Event = TextEvent | BreakEvent | ListEvent | TableEvent


def events(node: Tag, bold: bool = False) -> Iterator[Event]:
    """Flatten a subtree into text and line breaks; stop at nested lists and tables."""
    for child in node.children:
        if isinstance(child, NavigableString):
            # Keep whitespace-only strings: Word exports put each word in its own <span>
            # and the spaces between words live in these strings. Comments are skipped.
            if type(child) is NavigableString:
                yield TextEvent(str(child), bold)
            continue
        if not isinstance(child, Tag):
            continue
        if child.name in LIST_TAGS:
            yield ListEvent(child)
        elif child.name == "table":
            yield TableEvent(child)
        elif child.name == "br":
            yield BreakEvent()
        else:
            is_block = child.name in BLOCK_TAGS
            if is_block:
                yield BreakEvent()
            yield from events(child, bold or child.name in BOLD_TAGS)
            if is_block:
                yield BreakEvent()


@dataclass
class Line:
    text: str
    bold: bool


def lines_from(evts: list[TextEvent | BreakEvent]) -> list[Line]:
    """Join text events into lines, splitting on breaks. A line is bold if all its text is."""
    lines: list[Line] = []
    parts: list[TextEvent] = []

    def flush() -> None:
        text = clean("".join(p.text for p in parts))
        if text:
            lines.append(Line(text, all(p.bold for p in parts if p.text.strip())))
        parts.clear()

    for e in evts:
        if isinstance(e, BreakEvent):
            flush()
        else:
            parts.append(e)
    flush()
    return lines


def is_heading(line: Line) -> bool:
    return line.bold and len(line.text) <= HEADING_MAX_CHARS


# ---------------------------------------------------------------- numbering


def list_style(ol: Tag, depth: int) -> str:
    if m := re.search(r"list-style-type:\s*([\w-]+)", str(ol.get("style") or "")):
        return m.group(1)
    if (t := ol.get("type")) and str(t) in TYPE_ATTR_STYLES:
        return TYPE_ATTR_STYLES[str(t)]
    return DEFAULT_STYLES[min(depth, len(DEFAULT_STYLES) - 1)]


def to_roman(n: int) -> str:
    vals = [(10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out


def to_alpha(n: int) -> str:
    """1 -> a, 26 -> z, 27 -> aa, 28 -> ab: how browsers number lower-alpha past z."""
    out = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        out = chr(ord("a") + rem) + out
    return out


def number_token(n: int, style: str) -> str:
    match style:
        case "lower-alpha":
            return to_alpha(n)
        case "upper-alpha":
            return to_alpha(n).upper()
        case "lower-roman":
            return to_roman(n)
        case "upper-roman":
            return to_roman(n).upper()
        case _:
            return str(n)


# ---------------------------------------------------------------- body walk


class BodyBuilder:
    def __init__(self) -> None:
        self.blocks: list[Block] = []
        self.section: str | None = None
        # For lists that were split in the source (an <ol> ends, a note paragraph follows,
        # then a new <ol start="2" style="lower-alpha"> carries on the sub-list): remember
        # where each numbering style was last used so the new list can reattach there.
        self.style_context: dict[str, tuple[int, str | None]] = {}
        self.last_item_path: dict[int, str | None] = {}
        # Numbering style of the policy's top-level sections ("decimal", or "upper-roman"
        # in older policies), and the section we're currently inside. Loose paragraphs
        # and tables between lists are attributed to that section so they stay citable.
        self.top_style: str | None = None
        self.current_top: str | None = None

    def emit_lines(self, lines: list[Line], depth: int = 0) -> None:
        for line in lines:
            if is_heading(line):
                self.section = line.text
                self.current_top = None  # a standalone heading starts a non-numbered section
                self.blocks.append(Block(kind="heading", text=line.text, depth=depth))
            else:
                self.blocks.append(
                    Block(
                        kind="paragraph",
                        text=line.text,
                        depth=depth + 1 if self.current_top else depth,
                        path=self.current_top,
                        section=self.section,
                    )
                )

    def walk(self, evts: Iterator[Event]) -> None:
        pending: list[TextEvent | BreakEvent] = []
        for e in evts:
            if isinstance(e, TextEvent | BreakEvent):
                pending.append(e)
                continue
            self.emit_lines(lines_from(pending))
            pending = []
            if isinstance(e, ListEvent):
                self.walk_top_list(e.el)
            else:
                self.emit_table(e.el, depth=1 if self.current_top else 0, path=self.current_top)
        self.emit_lines(lines_from(pending))

    def walk_top_list(self, ol: Tag) -> None:
        """Place a list found outside any list item.

        Source pages often break lists apart, so a list at the top of the DOM is one of:
        a new top-level section list; the second half of a split sub-list (same style as
        a deeper level, start > 1); or a sub-list placed after its section's text.
        """
        style = list_style(ol, 0)
        start = int(str(ol.get("start") or "1"))
        if ol.name == "ul":
            self.walk_list(ol, 1 if self.current_top else 0, self.current_top)
        elif style != self.top_style and start > 1 and style in self.style_context:
            depth, parent = self.style_context[style]
            # Paragraphs between the two halves of the list belong to the last item.
            anchor = self.last_item_path.get(depth)
            for b in reversed(self.blocks):
                if b.kind != "paragraph":
                    break
                b.path, b.depth = anchor, depth + 1
            self.walk_list(ol, depth, parent)
        elif self.top_style is not None and style != self.top_style and self.current_top:
            # Keep the numbering the page shows (style computed at depth 0), not the
            # depth-1 default, so "1. 2. 3." under section II doesn't become "II(a)".
            self.walk_list(ol, 1, self.current_top, style=style)
        else:
            self.top_style = self.top_style or style
            self.walk_list(ol, 0, None)

    def walk_list(
        self, ol: Tag, depth: int, parent_path: str | None, style: str | None = None
    ) -> None:
        ordered = ol.name == "ol"
        style = style or list_style(ol, depth)
        start = int(str(ol.get("start") or "1")) if ordered else 1
        items = [li for li in ol.find_all("li", recursive=False)]
        if ordered:
            self.style_context[style] = (depth, parent_path)
        for i, li in enumerate(items):
            path, label = None, None
            if ordered:
                token = number_token(start + i, style)
                if depth == 0:
                    path, label = token, f"{token}."
                else:
                    path = f"{parent_path or ''}({token})"
                    label = f"{token}." if style == "decimal" else f"({token})"
            else:
                path = parent_path
            self.last_item_path[depth] = path
            if depth == 0:
                self.current_top = path
            self.walk_item(li, depth, path, label)

    def walk_item(self, li: Tag, depth: int, path: str | None, label: str | None) -> None:
        pending: list[TextEvent | BreakEvent] = []
        emitted_item = False

        def flush() -> None:
            nonlocal emitted_item, pending
            lines = lines_from(pending)
            pending = []
            if not lines:
                return
            if not emitted_item:
                heading = None
                if is_heading(lines[0]) and len(lines) > 1:
                    heading = lines.pop(0).text
                elif is_heading(lines[0]) and len(lines) == 1:
                    # A bold-only item is a title whose content is entirely in sub-items.
                    heading = lines.pop(0).text
                if depth == 0 and heading:
                    self.section = heading
                self.blocks.append(
                    Block(
                        kind="list_item",
                        text="\n".join(line.text for line in lines),
                        depth=depth,
                        path=path,
                        label=label,
                        heading=heading,
                        section=self.section,
                    )
                )
                emitted_item = True
            else:
                # Text that follows a nested list still belongs to this item.
                self.blocks.append(
                    Block(
                        kind="paragraph",
                        text="\n".join(line.text for line in lines),
                        depth=depth + 1,
                        path=path,
                        section=self.section,
                    )
                )

        for e in events(li):
            if isinstance(e, TextEvent | BreakEvent):
                pending.append(e)
                continue
            flush()
            if not emitted_item:
                # Item opens directly with a nested list; still record its number.
                self.blocks.append(
                    Block(
                        kind="list_item",
                        text="",
                        depth=depth,
                        path=path,
                        label=label,
                        section=self.section,
                    )
                )
                emitted_item = True
            if isinstance(e, ListEvent):
                self.walk_list(e.el, depth + 1, path)
            else:
                self.emit_table(e.el, depth + 1, path)
        flush()

    def emit_table(self, table: Tag, depth: int, path: str | None) -> None:
        rows = table_grid(table)
        if rows:
            self.blocks.append(
                Block(
                    kind="table",
                    text=table_markdown(rows),
                    depth=depth,
                    path=path,
                    section=self.section,
                    rows=rows,
                )
            )


def table_grid(table: Tag) -> list[list[str]]:
    """Table cells as a grid, with rowspan/colspan values copied into every covered cell.

    Repeated header rows (Word repeats the header before each group) are dropped, and
    group rows (a single label spanning the whole width) are kept as one-cell rows.
    """
    grid: list[list[str]] = []
    carry: dict[int, tuple[str, int]] = {}  # column -> (value, rows remaining)
    for tr in table.find_all("tr"):
        row: list[str] = []
        cells = tr.find_all(["td", "th"], recursive=False)
        col = 0
        for td in cells:
            while col in carry:
                row.append(carry[col][0])
                col += 1
            value = clean(td.get_text(" "))
            colspan = int(str(td.get("colspan") or "1"))
            rowspan = int(str(td.get("rowspan") or "1"))
            for _ in range(colspan):
                if rowspan > 1:
                    carry[col] = (value, rowspan)
                row.append(value)
                col += 1
        while col in carry:
            row.append(carry[col][0])
            col += 1
        carry = {c: (v, n - 1) for c, (v, n) in carry.items() if n - 1 > 0}
        if len(cells) == 1 and len(set(row)) == 1:
            row = row[:1]  # group label spanning the table, e.g. "Preschool/Elementary"
        if any(row):
            grid.append(row)
    if grid:

        def key(r: list[str]) -> list[str]:
            return [re.sub(r"\s+", "", c).lower() for c in r]

        header = key(grid[0])
        grid = grid[:1] + [r for r in grid[1:] if key(r) != header]
    return grid


def table_markdown(rows: list[list[str]]) -> str:
    width = max(len(r) for r in rows)

    def fmt(cells: list[str]) -> str:
        cells = [c.replace("|", "\\|") for c in cells]
        if len(cells) == 1 and width > 1:
            cells = [f"**{cells[0]}**"]  # group row, e.g. "Preschool/Elementary"
        return "| " + " | ".join(cells + [""] * (width - len(cells))) + " |"

    header, *body = rows
    return "\n".join([fmt(header), "| " + " | ".join(["---"] * width) + " |", *map(fmt, body)])


# ---------------------------------------------------------------- page-level parsing


def header_fields(table: Tag) -> dict[str, str]:
    fields = {}
    for tr in table.find_all("tr"):
        cells = [clean(td.get_text(" ")) for td in tr.find_all("td", recursive=False)]
        if len(cells) >= 2:
            fields[cells[0].lower()] = cells[1]
    return fields


def pick(fields: dict[str, str], *needles: str) -> str | None:
    """First field whose label equals a needle, else the first whose label contains one."""
    for n in needles:
        if n in fields:
            return fields[n]
    for key, value in fields.items():
        if any(n in key for n in needles):
            return value
    return None


def history_from_legacy_header(fields: dict[str, str]) -> PolicyHistory:
    """Older pages put dates in the header ("Policy Date", "Previous Policy Dates")."""
    current = pick(fields, "policy date")
    previous = [clean(d) for d in (pick(fields, "previous policy dates") or "").split(",")]
    revised = [d for d in [current, *reversed(previous)] if d]
    raw = f"Policy Date: {current or ''}; Previous Policy Dates: {', '.join(previous)}"
    return PolicyHistory(adopted=previous[0] if previous[0] else current, revised=revised, raw=raw)


def parse_history(table: Tag) -> PolicyHistory:
    raw = clean(table.get_text(" "))
    adopted = re.search(r"Adopted\s*:\s*(.+?)(?:Revised|$)", raw)
    revised = re.search(r"Revised\s*:\s*(.*)$", raw)
    revised_list = (
        [clean(r) for r in re.split(r"[;,](?=\s*[A-Z0-9])", revised.group(1))] if revised else []
    )
    # Dates written "Jul. 1, 2026" contain a comma; re-join "Month D" + "YYYY" fragments.
    merged: list[str] = []
    for r in revised_list:
        if merged and re.fullmatch(r"\d{4}", r) and not DATE_NUMERIC.search(merged[-1]):
            merged[-1] = f"{merged[-1]}, {r}"
        elif r:
            merged.append(r)
    return PolicyHistory(
        adopted=clean(adopted.group(1)) if adopted else None, revised=merged, raw=raw
    )


def page_meta(soup: BeautifulSoup) -> dict[str, str]:
    """Status and dates shown above the policy (outside the policy body)."""
    text = clean(soup.get_text(" "))
    out = {}
    for key, pattern in {
        "status": r"Status:\s*([A-Z]+)",
        "original_adopted": r"Original Adopted Date:\s*([\d/]+)",
        "last_revised": r"Last Revised Date:\s*([\d/]+)",
        "last_reviewed": r"Last Reviewed Date:\s*([\d/]+)",
    }.items():
        if m := re.search(pattern, text):
            out[key] = m.group(1)
    return out


def parse_saved_page(path: Path) -> Policy:
    raw_bytes = path.read_bytes()
    html = raw_bytes.decode("utf-8")
    warnings: list[str] = []

    source_url = m.group(1) if (m := SAVED_FROM.search(html)) else None
    if not source_url:
        warnings.append("no 'saved from url' comment; citation links will be missing")

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(DROP_TAGS):
        tag.decompose()
    meta = page_meta(soup)

    body = soup.select_one("div.ICE-Tracking")
    if body is None:
        raise ValueError(f"{path.name}: policy body (div.ICE-Tracking) not found")

    tables = body.find_all("table")
    # Current pages label the code "Policy Code"; older-format pages (e.g. LICN-006, -008)
    # use "Policy ID", "Policy Date", "Statutory Reference" and have no history table.
    header = next(
        (t for t in tables if re.search(r"Policy (Code|ID)", clean(t.get_text(" ")))), None
    )
    history = next((t for t in tables if clean(t.get_text(" ")).startswith("Policy History")), None)
    if header is None:
        raise ValueError(f"{path.name}: header table with 'Policy Code' / 'Policy ID' not found")

    fields = header_fields(header)
    legacy = "policy id" in fields
    if history is None and not legacy:
        warnings.append("no Policy History table")
    code = pick(fields, "policy code", "policy id") or ""
    if not re.fullmatch(r"LICN-\d{3}", code):
        warnings.append(f"unexpected policy code {code!r}")
    effective_raw = pick(fields, "effective date", "effective", "policy date")
    effective = parse_date(effective_raw)
    if effective is None:
        warnings.append(f"could not parse effective date from {effective_raw!r}")

    # Body = everything after the header table, minus the history table. The banner
    # lines above the header ("NORTH CAROLINA STATE BOARD OF EDUCATION / Policy Manual")
    # are dropped by starting only once the header has been passed.
    started = False
    evts: list[Event] = []
    for e in events(body):
        if isinstance(e, TableEvent) and e.el is header:
            started = True
        elif started and not (isinstance(e, TableEvent) and e.el is history):
            evts.append(e)
    builder = BodyBuilder()
    builder.walk(iter(evts))
    blocks = builder.blocks

    if not blocks:
        warnings.append("policy body produced no blocks")
    paths = [b.path for b in blocks if b.kind == "list_item" and b.path]
    if dupes := sorted({p for p in paths if paths.count(p) > 1}):
        warnings.append(f"duplicate list paths (a list restarted its numbering?): {dupes}")

    status = meta.get("status")
    if source_url and parse_qs(urlparse(source_url).query).get("isPndg", ["false"])[0] == "true":
        status = "PENDING"

    return Policy(
        code=code,
        title=pick(fields, "policy title") or "",
        series=pick(fields, "policy series", "policy category"),
        status=status,
        effective_date=effective,
        original_adopted=parse_date(meta.get("original_adopted")),
        last_revised=parse_date(meta.get("last_revised")),
        last_reviewed=parse_date(meta.get("last_reviewed")),
        statutory_citations=split_statutes(pick(fields, "statutory") or ""),
        ncac_citations=split_ncac(pick(fields, "ncac", "administrative code") or ""),
        source_url=source_url,
        source_file=path.name,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        blocks=blocks,
        history=(
            parse_history(history)
            if history
            else history_from_legacy_header(fields)
            if legacy
            else PolicyHistory()
        ),
        warnings=warnings,
    )


# ---------------------------------------------------------------- rendering


def to_markdown(policy: Policy) -> str:
    """Human-readable rendering, used for review and for the long-context baseline."""
    head = [
        f"# {policy.code}: {policy.title}",
        "",
        f"- **Status:** {policy.status or 'unknown'}",
        f"- **Effective:** {policy.effective_date or 'unknown'}",
        f"- **Last revised:** {policy.last_revised or 'unknown'}",
        f"- **Statutes:** {'; '.join(policy.statutory_citations) or 'none listed'}",
        f"- **NCAC:** {'; '.join(policy.ncac_citations) or 'none listed'}",
        f"- **Source:** {policy.source_url or 'unknown'}",
        "",
    ]
    body = []
    for b in policy.blocks:
        indent = "  " * b.depth
        match b.kind:
            case "heading":
                body += ["", f"## {b.text}", ""]
            case "list_item":
                title = f"**{b.heading}**" if b.heading else ""
                text = b.text.replace("\n", f"\n{indent}  ")
                sep = " " if title and text else ""
                body.append(f"{indent}{b.label or '-'} {title}{sep}{text}".rstrip())
            case "table":
                body += ["", b.text, ""]
            case _:
                body.append(f"{indent}{b.text}" if b.depth else f"\n{b.text}")
    return "\n".join(head + body).strip() + "\n"
