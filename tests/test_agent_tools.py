"""The research agent's tools (no API calls)."""

import re

import pytest

from licn_rag.agent.evidence import Evidence, doc_order, split_chunk
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


def test_definitions_sorted_and_valid(corpus: Corpus) -> None:
    box = ToolBox(corpus, None)
    names = [d["name"] for d in box.definitions()]
    expected = sorted(
        [
            "calculate",
            "follow_references",
            "get_policy_outline",
            "get_section",
            "list_license_types",
            "search",
        ]
    )
    assert names == expected
    for d in box.definitions():
        assert d["input_schema"]["type"] == "object"
    box_no_calc = ToolBox(corpus, None, calculator=False)
    assert "calculate" not in [d["name"] for d in box_no_calc.definitions()]


def test_get_section_reads_subtree(box: ToolBox) -> None:
    out = box.run("get_section", {"citation_id": "LICN-030 §1"})
    assert not out.is_error
    assert "[LICN-030 §1(a)]" in out.text
    assert "[LICN-030 §1(a)(i)]" in out.text
    assert box.evidence.ids()[0] == "LICN-030 §1"
    assert "LICN-030 §1(a)(i)" in box.evidence.ids()
    assert out.added_words > 0


def test_get_section_accepts_variants(box: ToolBox) -> None:
    out = box.run("get_section", {"citation_id": "LICN-030 Section 1(a)"})
    assert not out.is_error


def test_get_section_unknown_id(box: ToolBox) -> None:
    out = box.run("get_section", {"citation_id": "LICN-030 §99"})
    assert out.is_error
    assert "get_policy_outline" in out.text


def test_get_section_missing_policy(box: ToolBox) -> None:
    out = box.run("get_section", {"citation_id": "LICN-014"})
    assert out.is_error
    assert "not in the available policy set" in out.text


def test_get_section_too_long_returns_outline(box: ToolBox) -> None:
    out = box.run("get_section", {"citation_id": "LICN-007"})
    assert not out.is_error
    assert "too long" in out.text
    assert "[LICN-007 §1]" in out.text
    assert len(box.evidence) == 0


def test_get_section_short_whole_policy(box: ToolBox) -> None:
    out = box.run("get_section", {"citation_id": "LICN-020"})
    assert not out.is_error
    assert len(box.evidence) > 0


def test_outline(box: ToolBox) -> None:
    out = box.run("get_policy_outline", {"code": "licn-019"})
    assert not out.is_error
    assert "LICN-019 Residency License" in out.text
    assert "[LICN-019 §1]" in out.text
    assert len(box.evidence) == 0


def test_follow_references_reads_sections(box: ToolBox) -> None:
    out = box.run("follow_references", {"citation_id": "LICN-004 §1(h)(ii)(C)"})
    assert "LICN-007 §2(b)" in out.text
    assert "LICN-007 §2(b)" in box.evidence.ids()


def test_follow_references_whole_policy_pointer(box: ToolBox) -> None:
    out = box.run("follow_references", {"citation_id": "LICN-010 §5(c)"})
    assert "get_policy_outline('LICN-019')" in out.text


def test_follow_references_lists_citing(corpus: Corpus, box: ToolBox) -> None:
    cid = corpus.normalize("LICN-019 §1")
    assert cid is not None
    out = box.run("follow_references", {"citation_id": cid})
    assert "LICN-001 §2(w)" in out.text


def test_list_license_types(box: ToolBox) -> None:
    out = box.run("list_license_types", {})
    assert not out.is_error
    assert "Residency license" in out.text
    assert "[LICN-010 §5(c)]" in out.text
    assert {"LICN-001 §2(w)", "LICN-001 §2(h)", "LICN-010 §5(c)"} <= set(box.evidence.ids())


def test_calculate_records(box: ToolBox) -> None:
    out = box.run(
        "calculate",
        {"description": "CEUs from 6 semester hours", "expression": "6 * 2/3"},
    )
    assert out.text == "6 * 2/3 = 4"
    assert box.calculations == ["CEUs from 6 semester hours: 6 * 2/3 = 4"]


def test_calculate_error(box: ToolBox) -> None:
    out = box.run("calculate", {"description": "x", "expression": "import os"})
    assert out.is_error
    assert box.calculations == []


def test_invalid_input_and_unknown_tool(box: ToolBox) -> None:
    out = box.run("get_section", {})
    assert out.is_error
    assert "Invalid input" in out.text
    out2 = box.run("nope", {})
    assert out2.is_error


def test_reading_budget(corpus: Corpus) -> None:
    box = ToolBox(corpus, None, max_evidence_words=50)
    out1 = box.run("get_section", {"citation_id": "LICN-030 §1"})
    assert not out1.is_error
    out2 = box.run("get_section", {"citation_id": "LICN-020"})
    assert out2.is_error
    assert "Reading budget" in out2.text
    out3 = box.run("calculate", {"description": "x", "expression": "1 + 1"})
    assert not out3.is_error


def test_trace(box: ToolBox) -> None:
    box.run("get_section", {"citation_id": "LICN-020"})
    box.run("calculate", {"description": "x", "expression": "2 + 2"})
    assert len(box.trace) == 2
    for entry in box.trace:
        assert set(entry) == {"tool", "input", "added_words", "error"}


def test_search_without_index(box: ToolBox) -> None:
    out = box.run("search", {"query": "CEU"})
    assert out.is_error
    assert box.seed_search("CEU") == ""


def test_split_chunk_matches_cite_ids(corpus: Corpus) -> None:
    for c in chunk_corpus(corpus.policies):
        assert len(split_chunk(c.text)) == len(c.cite_ids)


def test_evidence_dedupes_and_orders(corpus: Corpus) -> None:
    chunks = [c for c in chunk_corpus(corpus.policies) if c.policy_code == "LICN-030"]
    ev = Evidence(corpus)
    order = doc_order(corpus)
    for c in reversed(chunks):
        ev.add_chunk(c)
    added = ev.add_chunk(chunks[0])
    assert added == 0
    ids = ev.ids()
    assert ids == sorted(ids, key=order.__getitem__)


def test_evidence_render(corpus: Corpus) -> None:
    chunks = chunk_corpus(corpus.policies)
    c030 = next(c for c in chunks if c.policy_code == "LICN-030")
    c001 = next(c for c in chunks if c.policy_code == "LICN-001")
    ev = Evidence(corpus)
    ev.add_chunk(c030)
    ev.add_chunk(c001)
    rendered = ev.render()
    assert rendered.count('<policy code="LICN-001"') == 1
    assert rendered.count('<policy code="LICN-030"') == 1
    assert rendered.index("LICN-001") < rendered.index("LICN-030")
    assert rendered.count("</policy>") == 2


def test_add_line_rejects_non_passage(corpus: Corpus) -> None:
    with pytest.raises(ValueError):
        Evidence(corpus).add_line("hello")


@needs_index
def test_search_returns_passages(corpus: Corpus) -> None:
    box = ToolBox(corpus, Index(INDEX_PATH), search_mode="bm25")
    out = box.run("search", {"query": "renewal CEUs literacy"})
    assert not out.is_error
    assert "[LICN-" in out.text
    assert len(box.evidence) > 0


@needs_index
def test_search_policy_filter(corpus: Corpus) -> None:
    box = ToolBox(corpus, Index(INDEX_PATH), search_mode="bm25")
    out = box.run("search", {"query": "license", "policy": "licn-030"})
    ids = re.findall(r"\[(LICN-\d{3})", out.text)
    assert ids
    assert all(i == "LICN-030" for i in ids)


@needs_index
def test_search_skips_already_read(corpus: Corpus) -> None:
    box = ToolBox(corpus, Index(INDEX_PATH), search_mode="bm25")
    box.run("search", {"query": "renewal CEUs literacy"})
    out2 = box.run("search", {"query": "renewal CEUs literacy"})
    assert "already read" in out2.text
    assert out2.added_words == 0


@needs_index
def test_search_unknown_policy(corpus: Corpus) -> None:
    box = ToolBox(corpus, Index(INDEX_PATH), search_mode="bm25")
    out = box.run("search", {"query": "x", "policy": "LICN-999"})
    assert out.is_error


@needs_index
def test_seed_search(corpus: Corpus) -> None:
    box = ToolBox(corpus, Index(INDEX_PATH), search_mode="bm25")
    text = box.seed_search("Praxis elementary", k=5)
    assert text
    assert box.trace[-1]["tool"] == "seed_search"
