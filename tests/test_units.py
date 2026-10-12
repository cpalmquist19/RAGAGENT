"""The agent's unit conversions (no API calls)."""

import pytest

from licn_rag.agent.units import convert


@pytest.mark.parametrize(
    ("value", "from_unit", "to_unit", "expected", "rule"),
    [
        (6, "semester_hour", "ceu", "6 semester credit hours = 9 CEUs", "LICN-001 §2(g)(i)"),
        (3, "semester_hour", "ceu", "3 semester credit hours = 4.5 CEUs", "LICN-001 §2(g)(i)"),
        (
            11.5,
            "semester_hour",
            "ceu",
            "11.5 semester credit hours = 17.25 CEUs",
            "LICN-001 §2(g)(i)",
        ),
        (2, "ceu", "semester_hour", "2 CEUs = 1.3333 semester credit hours", "LICN-001 §2(g)(i)"),
        (8, "ceu", "semester_hour", "8 CEUs = 5.3333 semester credit hours", "LICN-001 §2(g)(i)"),
        (8, "ceu", "clock_hour", "8 CEUs = 80 clock hours", "LICN-001 §2(g)(ii)"),
        (45, "clock_hour", "ceu", "45 clock hours = 4.5 CEUs", "LICN-001 §2(g)(ii)"),
        (4, "clock_hour", "ceu", "4 clock hours = 0.4 CEUs", "LICN-001 §2(g)(ii)"),
    ],
)
def test_convert(value: float, from_unit: str, to_unit: str, expected: str, rule: str) -> None:
    line, cited = convert(value, from_unit, to_unit)  # type: ignore[arg-type]
    assert line.startswith(expected + " (")
    assert cited == rule


def test_convert_explains_direction() -> None:
    line, _ = convert(6, "semester_hour", "ceu")
    assert "1 CEU = 2/3 semester credit hour" in line and "6 / (2/3)" in line


def test_same_unit_and_bad_pairs() -> None:
    assert convert(5, "ceu", "ceu") == ("5 CEUs = 5 CEUs (same unit)", None)
    with pytest.raises(ValueError, match="no direct conversion"):
        convert(3, "semester_hour", "clock_hour")
    with pytest.raises(ValueError, match="negative"):
        convert(-1, "ceu", "clock_hour")
