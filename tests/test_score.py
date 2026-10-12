"""Deterministic scoring and schema helpers (no API calls)."""

from typing import Any

import pytest

from licn_rag.answer.schema import AppAnswer, Claim
from licn_rag.corpus import Corpus
from licn_rag.evals.score import context_recall, deterministic, section_of
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


def test_context_recall() -> None:
    assert context_recall(["LICN-030 §1(b)", "LICN-019 §2"], key()) == 0.5
    assert context_recall(["LICN-030 §1", "LICN-001 §2(w)"], key()) == 1.0
    assert context_recall([], key()) == 0.0
    assert context_recall(None, key()) is None  # full corpus in context
    assert context_recall(["LICN-030 §1"], key(answerable="no")) is None


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


def test_parent_context_gives_list_items_their_lead_in(corpus: Corpus) -> None:
    from licn_rag.evals.score import parent_context

    lines = parent_context(corpus, "LICN-013 §3(b)")
    assert len(lines) == 1 and lines[0].startswith("[LICN-013 §3] ")
    assert "shall not add" in lines[0]
    assert [line[:16] for line in parent_context(corpus, "LICN-019 §1(a)(ii)")] == [
        "[LICN-019 §1] El",
        "[LICN-019 §1(a)]",
    ]
    assert parent_context(corpus, "LICN-030 §1") == []


def _judgment(bottom_line: str, errors: list[str], central: list[str]) -> Any:
    from licn_rag.evals.score import Judgment, KeyPointGrade

    points = [KeyPointGrade(index=i, central=True, status=s) for i, s in enumerate(central)]  # type: ignore[arg-type]
    points.append(KeyPointGrade(index=len(points), central=False, status="missing"))
    return Judgment(
        key_points=points,
        claims=[],
        bottom_line=bottom_line,  # type: ignore[arg-type]
        material_errors=errors,
        minor_issues=["peripheral wording"],
        rationale="",
    )


@pytest.mark.parametrize(
    ("bottom_line", "errors", "central", "label"),
    [
        ("agrees", [], ["covered", "partial"], "correct"),  # a missing peripheral point is fine
        ("agrees", ["wrong fee"], ["covered"], "partially_correct"),
        ("agrees", [], ["covered", "missing"], "partially_correct"),
        ("partly_agrees", [], ["covered"], "partially_correct"),
        ("agrees", [], ["contradicted"], "incorrect"),
        ("disagrees", [], ["covered"], "incorrect"),
    ],
)
def test_overall_label_is_computed(
    bottom_line: str, errors: list[str], central: list[str], label: str
) -> None:
    from licn_rag.evals.score import overall_label

    assert overall_label(_judgment(bottom_line, errors, central)) == label
