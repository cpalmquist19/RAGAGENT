"""Chunking invariants on the real corpus, plus index helpers that need no models."""

import pytest

from licn_rag.corpus import Corpus
from licn_rag.rag.chunking import MAX_WORDS, chunk_corpus, parent_path, strip_ids
from licn_rag.rag.index import fts_query


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return Corpus.load()


@pytest.fixture(scope="module")
def chunks(corpus: Corpus):
    return chunk_corpus(corpus.policies)


def test_every_passage_is_retrievable(corpus: Corpus, chunks) -> None:
    covered = {cid for c in chunks for cid in c.cite_ids}
    assert covered == corpus.valid_ids


def test_chunks_stay_within_budget(chunks) -> None:
    # A single block larger than the budget is allowed through whole; nothing else is.
    oversized = [c for c in chunks if c.words > MAX_WORDS + 100]
    assert oversized == []


def test_chunks_are_ordered_and_unique(chunks) -> None:
    assert [c.order for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_sub_items_keep_their_parent_context(chunks) -> None:
    literacy = next(c for c in chunks if "LICN-030 §1(a)(ii)" in c.cite_ids)
    assert "LICN-030 License Renewal for Teachers" in literacy.breadcrumb
    assert "LICN-030 §1(a)" in literacy.cite_ids  # lead-in kept with its sub-items


def test_big_table_is_split_with_header_repeated(chunks) -> None:
    table_chunks = [c for c in chunks if "LICN-003 §5" in c.cite_ids and "| ---" in c.text]
    assert len(table_chunks) >= 5
    assert all("AREA OF LICENSURE" in c.text for c in table_chunks)


def test_parent_path() -> None:
    assert parent_path("2(b)(iv)") == "2(b)"
    assert parent_path("II(1)") == "II"
    assert parent_path("2") is None


def test_strip_ids() -> None:
    assert strip_ids("[LICN-030 §1] Text\n  [LICN-030 §1(a)] More") == "Text\nMore"


def test_fts_query_is_safe_and_drops_stopwords() -> None:
    q = fts_query('What does "LICN-030" say about CEUs? (renewal) AND OR')
    assert '"licn-030"' in q and '"ceus"' in q and '"renewal"' in q
    assert '"what"' not in q
    assert fts_query("the and of") == '""'
