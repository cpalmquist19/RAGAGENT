"""The eval's deterministic unit-conversion check (no API calls)."""

import pytest

from licn_rag.evals.conversions import answer_text, find_conversions


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ("6 semester hours equal 9 CEUs.", "ok"),
        ("6 graduate semester hours convert to 4 CEUs.", "backwards"),
        ("Eight CEUs therefore equal 12 semester credit hours.", "backwards"),
        ("8 CEUs equal about 5.33 semester credit hours.", "ok"),
        ("8 CEUs = 80 clock hours.", "ok"),
        ("45 clock hours equals 4.5 CEUs.", "ok"),
        ("4 clock hours is 0.4 CEU.", "ok"),
        ("2 CEUs equal 1.3 semester hours.", "wrong"),
        ("11.5 × 2/3 = about 7.67 CEUs", "backwards"),
        ("12 / (2/3) = 18 CEUs", "ok"),
        ("3 x 2/3 = 2 CEUs", "backwards"),
        ("8 x 2/3 = 5.33 semester credit hours", "ok"),
        ("0.5 ÷ (2/3) = 0.75 semester credit hour", "backwards"),
        ("45 / 10 = 4.5 CEUs", "ok"),
        ("8 CEUs x 10 = 80 clock hours", "ok"),
    ],
)
def test_find_conversions(text: str, status: str) -> None:
    found = find_conversions(text)
    assert [c.status for c in found] == [status]


@pytest.mark.parametrize(
    "text",
    [
        "10 clock hours per CEU equals 0.4 CEU",  # a rate, not an amount
        "Five semester credit hours do not equal five CEUs.",
        "The teacher has 8 CEUs and needs 4 CEUs in the license area.",  # same unit
        "3 semester hours equal 45 clock hours.",  # no such conversion in the policies
        "Teachers need 8 CEUs. 6 semester hours are required for the add-on.",
    ],
)
def test_ignores_non_conversions(text: str) -> None:
    assert find_conversions(text) == []


def test_answer_text_covers_all_fields() -> None:
    answer = {
        "summary": "s",
        "claims": [{"text": "c1", "citations": []}],
        "questions_for_user": ["q"],
        "premise_corrections": ["p"],
        "not_covered": ["n"],
    }
    assert answer_text(answer).split("\n") == ["s", "c1", "q", "p", "n"]
