"""The agent's safe calculator (no API calls)."""

import pytest

from licn_rag.agent.calculator import calculate, evaluate, format_number


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("6 / (2/3)", "6 / (2/3) = 9"),
        ("8 - 5.5", "8 - 5.5 = 2.5"),
        ("10 / 3", "10 / 3 = 3.3333"),
        ("round(10 / 3, 2)", "round(10 / 3, 2) = 3.33"),
        ("max(0, 8 - 9.5)", "max(0, 8 - 9.5) = 0"),
        ("(3 + 2) * 10", "(3 + 2) * 10 = 50"),
        ("-4 + 1", "-4 + 1 = -3"),
    ],
)
def test_calculate(expression: str, expected: str) -> None:
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",
        "open('x')",
        "a + 1",
        "True + 1",
        "2 ** 100",
        "1 / 0",
        "'a' * 3",
        "[1, 2]",
        "1 +",
        "9" * 300,
    ],
)
def test_rejects_unsafe_or_invalid(expression: str) -> None:
    with pytest.raises(ValueError):
        evaluate(expression)


def test_format_number() -> None:
    assert format_number(6.0) == "6"
    assert format_number(7) == "7"
    assert format_number(2 / 3) == "0.6667"
