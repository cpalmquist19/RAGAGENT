"""Parser tests on a synthetic page that reproduces the quirks found in real saved pages."""

from datetime import date
from pathlib import Path

import pytest

from licn_rag.ingest.models import Policy
from licn_rag.ingest.simbli_html import (
    number_token,
    parse_saved_page,
    split_ncac,
    split_statutes,
    to_markdown,
)

URL = (
    "https://simbli.eboardsolutions.com/Policy/ViewPolicy.aspx?S=10399&revid=abc"
    "&ptid=xyz&secid=&PG=6&IRP=0&isPndg={pending}"
)

PAGE = """<!DOCTYPE html>
<!-- saved from url=(0100){url} -->
<html><head><meta charset="utf-8"><script>var junk = 1;</script></head><body>
<div>Policy LICN-099: Test Policy Status: ADOPTED Original Adopted Date: 05/13/2021
  | Last Revised Date: 07/01/2026 | Last Reviewed Date: 07/02/2026</div>
<div class="ICE-Tracking">
  <div><strong>NORTH CAROLINA STATE BOARD OF EDUCATION</strong></div>
  <table>
    <tr><td><strong>Policy Title</strong></td><td>Test Policy</td></tr>
    <tr><td>Policy Series</td><td>Licensure (LICN)</td></tr>
    <tr><td>Policy Code</td><td>LICN-099</td></tr>
    <tr><td>Statutory Citation(s)</td><td>G.S. 115C-12(9); 115C-270.20</td></tr>
    <tr><td>North Carolina Administrative Code (NCAC) Citation(s)</td>
        <td>16 NCAC 06C .0301; 06C .0302</td></tr>
    <tr><td>Effective Date</td><td>Effective date: July&nbsp;1, 2026</td></tr>
  </table>
  <span><strong>Policy Statement</strong></span><br/>
  <span>This</span>
  <span>Policy</span>
  <span>tests things.</span>
  <ol>
    <li><strong>Continuing Education</strong><br/><br/>Renew every five years.
      <ol style="list-style-type:lower-alpha">
        <li>Grades K-5 need:
          <ol style="list-style-type:lower-roman">
            <li>Three CEUs in literacy.</li>
            <li>Two general CEUs.</li>
          </ol>
        </li>
      </ol>
      <div>A note that belongs to 1(a).</div>
      <ol start="2" style="list-style-type:lower-alpha">
        <li>Grades 6-12 need eight CEUs.</li>
      </ol>
    </li>
    <li><strong>Evaluations</strong>
      <ol style="list-style-type:lower-alpha"><li>Be proficient.</li></ol>
    </li>
  </ol>
  <table>
    <tr><td>AREA</td><td>CODE</td><td>TEST</td></tr>
    <tr><td colspan="3">Elementary</td></tr>
    <tr><td rowspan="2">Elementary (K-6)</td><td rowspan="2">025</td><td>Reading 890</td></tr>
    <tr><td>Math 7813</td></tr>
    <tr><td>AREA</td><td>CODE </td><td>TEST</td></tr>
  </table>
  <table><tr><td>Policy History</td>
    <td>Adopted : Jul. 1, 2026 Revised : Jul. 1, 2026; May 2, 2024</td></tr></table>
</div></body></html>
"""


def write_page(tmp_path: Path, pending: str = "false", top_level_split: bool = False) -> Path:
    html = PAGE.format(url=URL.format(pending=pending))
    if top_level_split:
        # Real-world variant (LICN-010): the sub-list continues as a *top-level* <ol>.
        html = html.replace(
            """      <div>A note that belongs to 1(a).</div>
      <ol start="2" style="list-style-type:lower-alpha">
        <li>Grades 6-12 need eight CEUs.</li>
      </ol>
    </li>""",
            "    </li>",
        ).replace(
            """    <li><strong>Evaluations</strong>
      <ol style="list-style-type:lower-alpha"><li>Be proficient.</li></ol>
    </li>
  </ol>""",
            """  </ol>
  <div>A note that belongs to 1(a).</div>
  <ol start="2" style="list-style-type:lower-alpha"><li>Grades 6-12 need eight CEUs.</li></ol>""",
        )
    path = tmp_path / "View Policy LICN-099_ Test Policy _ NC State Board of Education.html"
    path.write_text(html, encoding="utf-8")
    return path


@pytest.fixture
def policy(tmp_path: Path) -> Policy:
    return parse_saved_page(write_page(tmp_path))


def by_path(policy: Policy, path: str):
    return next(b for b in policy.blocks if b.kind == "list_item" and b.path == path)


def test_header_and_page_metadata(policy: Policy) -> None:
    assert policy.code == "LICN-099"
    assert policy.title == "Test Policy"
    assert policy.status == "ADOPTED"
    assert policy.effective_date == date(2026, 7, 1)
    assert policy.original_adopted == date(2021, 5, 13)
    assert policy.last_reviewed == date(2026, 7, 2)
    assert policy.statutory_citations == ["G.S. 115C-12(9)", "G.S. 115C-270.20"]
    assert policy.ncac_citations == ["16 NCAC 06C .0301", "16 NCAC 06C .0302"]
    assert policy.source_url and "revid=abc" in policy.source_url
    assert policy.warnings == []


def test_banner_and_scripts_are_excluded(policy: Policy) -> None:
    text = " ".join(b.text for b in policy.blocks)
    assert "NORTH CAROLINA STATE BOARD" not in text
    assert "junk" not in text


def test_word_per_span_text_keeps_spaces(policy: Policy) -> None:
    assert any(b.text == "This Policy tests things." for b in policy.blocks)


def test_nested_numbering_matches_policy_citation_style(policy: Policy) -> None:
    item = by_path(policy, "1")
    assert item.heading == "Continuing Education"
    assert item.text == "Renew every five years."
    assert by_path(policy, "1(a)(ii)").text == "Two general CEUs."
    assert by_path(policy, "1(a)(ii)").label == "(ii)"
    assert by_path(policy, "1(b)").text == "Grades 6-12 need eight CEUs."
    assert by_path(policy, "1(b)").section == "Continuing Education"


def test_bold_only_item_is_title_not_text(policy: Policy) -> None:
    item = by_path(policy, "2")
    assert item.heading == "Evaluations"
    assert item.text == ""


def test_split_top_level_list_reattaches(tmp_path: Path) -> None:
    policy = parse_saved_page(write_page(tmp_path, top_level_split=True))
    assert by_path(policy, "1(b)").text == "Grades 6-12 need eight CEUs."
    note = next(b for b in policy.blocks if b.text == "A note that belongs to 1(a).")
    assert note.path == "1(a)"


def test_table_spans_are_filled_and_repeated_headers_dropped(policy: Policy) -> None:
    table = next(b for b in policy.blocks if b.kind == "table")
    assert table.rows == [
        ["AREA", "CODE", "TEST"],
        ["Elementary"],
        ["Elementary (K-6)", "025", "Reading 890"],
        ["Elementary (K-6)", "025", "Math 7813"],
    ]


def test_history(policy: Policy) -> None:
    assert policy.history.adopted == "Jul. 1, 2026"
    assert policy.history.revised == ["Jul. 1, 2026", "May 2, 2024"]


def test_pending_revision_is_flagged(tmp_path: Path) -> None:
    assert parse_saved_page(write_page(tmp_path, pending="true")).status == "PENDING"


def test_markdown_render(policy: Policy) -> None:
    md = to_markdown(policy)
    assert md.startswith("# LICN-099: Test Policy")
    assert "    (ii) Two general CEUs." in md


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("G.S. 115C-12; 115C-270.1", ["G.S. 115C-12", "G.S. 115C-270.1"]),
        ("GS 270.15", ["G.S. 270.15"]),
    ],
)
def test_split_statutes(raw: str, expected: list[str]) -> None:
    assert split_statutes(raw) == expected


def test_split_ncac_range_kept_whole() -> None:
    assert split_ncac("16 NCAC 06C .0601–.0608") == ["16 NCAC 06C .0601–.0608"]


def test_split_ncac_ignores_trailing_text() -> None:
    assert split_ncac("16 NCAC 06C .0371 LICENSURE FEES") == ["16 NCAC 06C .0371"]


LEGACY_PAGE = """<!-- saved from url=(10)https://example/x?isPndg=false -->
<html><body><div class="ICE-Tracking">
  <table>
    <tr><td>Policy Title</td><td>Old Style</td></tr>
    <tr><td>Policy Category</td><td>Licensure (LICN)</td></tr>
    <tr><td>Policy ID</td><td>LICN-098</td></tr>
    <tr><td>Policy Date</td><td>07/01/2025</td></tr>
    <tr><td>Previous Policy Dates</td><td>10/01/1998, 04/07/2022</td></tr>
    <tr><td>Statutory Reference</td><td>G.S. 115C - 12; 11 5C - 270.20</td></tr>
    <tr><td>NC Administrative Code</td><td>16 NCAC 06C .0364</td></tr>
  </table>
  <ol style="list-style-type:upper-roman">
    <li><strong>Experience Credit</strong><br/>Rules:
      <ol style="list-style-type:lower-alpha"><li>Full-time work counts.</li></ol></li>
  </ol>
  <div>The Licensure Section shall consider:</div>
  <div><ol><li>Alignment with standards.</li><li>Transferable skills.</li></ol></div>
  <ol start="2" style="list-style-type:upper-roman">
    <li><strong>JROTC</strong><br/>Up to 10 years.</li></ol>
</div></body></html>
"""


def test_legacy_header_and_roman_sections(tmp_path: Path) -> None:
    path = tmp_path / "View Policy LICN-098_ Old Style.html"
    path.write_text(LEGACY_PAGE, encoding="utf-8")
    policy = parse_saved_page(path)
    assert policy.code == "LICN-098"
    assert policy.series == "Licensure (LICN)"
    assert policy.effective_date == date(2025, 7, 1)
    assert policy.statutory_citations == ["G.S. 115C-12", "G.S. 115C-270.20"]
    assert policy.history.revised == ["07/01/2025", "04/07/2022", "10/01/1998"]
    assert policy.warnings == []
    assert by_path(policy, "I(a)").text == "Full-time work counts."
    # Loose paragraph and the separately-placed "1. 2." list belong to section I.
    note = next(b for b in policy.blocks if b.text == "The Licensure Section shall consider:")
    assert note.path == "I"
    assert by_path(policy, "I(2)").label == "2."
    assert by_path(policy, "II").heading == "JROTC"


@pytest.mark.parametrize(
    ("n", "style", "token"),
    [
        (1, "lower-alpha", "a"),
        (26, "lower-alpha", "z"),
        (27, "lower-alpha", "aa"),
        (28, "upper-alpha", "AB"),
        (4, "lower-roman", "iv"),
        (9, "upper-roman", "IX"),
    ],
)
def test_number_token(n: int, style: str, token: str) -> None:
    assert number_token(n, style) == token
