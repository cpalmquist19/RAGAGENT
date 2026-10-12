"""Unit conversions for the agent's `convert` tool.

Models reliably pick the wrong direction for CEU conversions (they write 6 semester hours
as 6 * 2/3 = 4 CEUs; it is 6 / (2/3) = 9), and a calculator cannot catch that because
the expression is the model's. Here the model names the units and the code owns the factor.

The only conversion rule in the LICN policies is the CEU definition, LICN-001 §2(g):
one CEU is two-thirds of a semester credit hour (i) or ten clock hours of professional
development (ii). The policies define no direct semester-hour/clock-hour conversion, so
that pair is refused instead of being inferred through CEUs.
"""

from fractions import Fraction
from typing import Literal

from licn_rag.agent.calculator import format_number

Unit = Literal["ceu", "semester_hour", "clock_hour"]
RULE_CITATIONS = {"semester_hour": "LICN-001 §2(g)(i)", "clock_hour": "LICN-001 §2(g)(ii)"}
PER_CEU: dict[str, Fraction] = {"semester_hour": Fraction(2, 3), "clock_hour": Fraction(10)}
LABELS = {"ceu": "CEUs", "semester_hour": "semester credit hours", "clock_hour": "clock hours"}


def convert(value: float, from_unit: Unit, to_unit: Unit) -> tuple[str, str | None]:
    """(result line, citation ID of the rule used). Raises ValueError for unsupported pairs.

    convert(6, "semester_hour", "ceu") ->
    ("6 semester credit hours = 9 CEUs (1 CEU = 2/3 semester credit hour, so CEUs = 6 / (2/3))",
     "LICN-001 §2(g)(i)")
    """
    if value < 0:
        raise ValueError("value must not be negative")
    if from_unit == to_unit:
        line = f"{format_number(value)} {LABELS[from_unit]} = {format_number(value)} "
        return line + LABELS[to_unit] + " (same unit)", None
    if "ceu" not in (from_unit, to_unit):
        raise ValueError(
            "the policies define no direct conversion between semester credit hours and "
            "clock hours; they are two separate ways to earn CEUs (LICN-001 §2(g)). Convert "
            "each to ceu instead."
        )
    other = to_unit if from_unit == "ceu" else from_unit
    per = PER_CEU[other]
    per_text = f"{per.numerator}/{per.denominator}" if per.denominator != 1 else str(per)
    v = format_number(value)
    unit = LABELS[other] if per > 1 else LABELS[other].removesuffix("s")
    rule = f"1 CEU = {per_text} {unit}"
    if from_unit == "ceu":
        result = float(Fraction(value) * per)
        how = f"{LABELS[other]} = {v} x {per_text}"
    else:
        result = float(Fraction(value) / per)
        how = f"CEUs = {v} / {per_text}" if per.denominator == 1 else f"CEUs = {v} / ({per_text})"
    line = f"{v} {LABELS[from_unit]} = {format_number(result)} {LABELS[to_unit]} ({rule}, so {how})"
    return line, RULE_CITATIONS[other]
