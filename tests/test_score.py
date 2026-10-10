"""Deterministic scoring and schema helpers (no API calls)."""

from typing import Any

import pytest

from licn_rag.answer.schema import AppAnswer, Claim
from licn_rag.corpus import Corpus
from licn_rag.evals.score import deterministic, section_of
from licn_rag.llm import output_schema


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return Corpus.load()


def key(**overrides: Any) -> dict[str, Any]:
    base = {
        "answerable": "yes",
        "verdict": "no",
        "expected_citations": ["LICN-030 §1(a)", "LICN-001 §2(g)"],
        "key_points": [{"point": "8 CEUs", "citations": ["LICN-030 §1(a)(i)"]}],
    }
    return base | overrides


def answer(verdict: str = "no", cites: list[str] | None = None, **kw: Any) -> AppAnswer:
    return AppAnswer(
        verdict=verdict,  # type: ignore[arg-type]
        summary="s",
        claims=[Claim(text="c", citations=cites if cites is not None else ["LICN-030 §1(a)(ii)"])],
        questions_for_user=kw.get("questions", []),
        premise_corrections=[],
        not_covered=[],
    )


@pytest.mark.parametrize(
    ("cite", "section"),
    [
        ("LICN-030 §1(a)(ii)", "LICN-030 §1"),
        ("LICN-006 §II(1)", "LICN-006 §II"),
        ("LICN-030 (Policy Statement)", "LICN-030 (Policy Statement)"),
        ("LICN-008", "LICN-008"),
    ],
)
def test_section_of(cite: str, section: str) -> None:
    assert section_of(cite) == section


def test_section_level_citation_overlap(corpus: Corpus) -> None:
    m = deterministic(answer(), key(), corpus)
    assert m["verdict_match"] is True
    assert m["section_recall"] == 0.5  # cited LICN-030 §1, missed LICN-001 §2
    assert m["section_precision"] == 1.0
    assert m["policy_recall"] == 0.5
    assert m["invalid_citations"] == []


def test_invalid_citation_detected(corpus: Corpus) -> None:
    m = deterministic(answer(cites=["LICN-030 §9(z)"]), key(), corpus)
    assert m["invalid_citations"] == ["LICN-030 §9(z)"]


def test_abstention_metrics(corpus: Corpus) -> None:
    out_of_scope = key(answerable="no", verdict="not_covered")
    assert deterministic(answer("not_covered", []), out_of_scope, corpus)["abstained_correctly"]
    m = deterministic(answer("not_covered", []), key(), corpus)
    assert m["false_abstention"] is True
    assert m["abstained_correctly"] is None


def test_asking_for_missing_facts(corpus: Corpus) -> None:
    needs = key(verdict="needs_more_info")
    assert deterministic(answer("no", questions=["grade band?"]), needs, corpus)[
        "asked_for_missing_facts"
    ]
    assert not deterministic(answer("no"), needs, corpus)["asked_for_missing_facts"]


def test_output_schema_is_strict() -> None:
    schema = output_schema(AppAnswer)
    assert "$defs" not in str(schema)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(AppAnswer.model_fields)
    claim = schema["properties"]["claims"]["items"]
    assert claim["additionalProperties"] is False
    assert claim["required"] == ["text", "citations"]
