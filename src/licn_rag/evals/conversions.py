"""Deterministic check of unit conversions stated in an answer (no LLM calls).

Finds conversions between CEUs, semester credit hours, and clock hours and checks them
against the CEU definition in LICN-001 §2(g): one CEU is two-thirds of a semester credit
hour or ten clock hours. Two forms are recognized:

- statements: "6 semester hours equal 9 CEUs", "eight CEUs = 80 clock hours"
- arithmetic: "11.5 x 2/3 = 7.67 CEUs", "12 / (2/3) = 18 CEUs" (the operator and factor
  show which direction the writer converted, even without a unit on the left)

A wrong conversion is labelled "backwards" when it used the inverted factor (6 semester
hours = 4 CEUs), the error models make most. The parser is deliberately narrow, so what it
flags is almost always a real conversion; it misses conversions phrased other ways, so
read its error counts as a lower bound.
"""

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

PER_CEU = {"ceu": Fraction(1), "semester_hour": Fraction(2, 3), "clock_hour": Fraction(10)}
WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "eighteen": 18,
    "twenty": 20, "twenty-four": 24, "thirty": 30,
}  # fmt: skip
NUM = r"(\d+(?:\.\d+)?(?:/\d+)?|" + "|".join(sorted(WORDS, key=len, reverse=True)) + ")"
DIGITS = r"(\d+(?:\.\d+)?)"
UNIT = (
    r"(CEUs?|continuing education units?|(?:graduate |college )?semester (?:credit )?hours?"
    r"|clock hours?|contact hours?)"
)
LINK = (
    r"(?:=|equals?|equal to|(?:is |are )?equivalent to|converts? (?:in)?to|converted (?:in)?to"
    r"|works? out to|amounts? to|translates? (?:in)?to|is|are|yields?|gives?)"
)
HEDGE = r"(?:about |approximately |roughly |only |just |exactly )?"
# "per" right after a unit is a rate ("10 clock hours per CEU"), not an amount.
NOT_RATE = r"(?![^\S\n]+per\b)"
STATEMENT = re.compile(
    rf"(?<![\w./]){NUM}\s+{UNIT}\b{NOT_RATE}[^.;,:()\n]{{0,30}}?\s{LINK}\s+{HEDGE}{NUM}\s+{UNIT}",
    re.IGNORECASE,
)
ARITHMETIC = re.compile(
    rf"(?<![\w./]){DIGITS}\s*(?:{UNIT}\s*)?(×|x|\*|times|÷|/|divided by)\s*"
    rf"\(?\s*(2/3|two-thirds|3/2|1\.5|10|ten)\s*\)?\s*(?:=|equals?|is|gives)\s*{HEDGE}"
    rf"{DIGITS}\s+{UNIT}",
    re.IGNORECASE,
)
FACTORS = {"2/3": Fraction(2, 3), "two-thirds": Fraction(2, 3), "3/2": Fraction(3, 2),
           "1.5": Fraction(3, 2), "10": Fraction(10), "ten": Fraction(10)}  # fmt: skip
NEGATION = re.compile(r"\b(?:not|never|no)\b|n't\b", re.IGNORECASE)
TOLERANCE = 0.02  # relative: "1.33" for 4/3 is fine, "1.3" is not


@dataclass(frozen=True)
class Conversion:
    text: str
    status: str
    """'ok', 'backwards', or 'wrong'."""
    expected: float


def unit_of(raw: str) -> str:
    raw = raw.lower()
    if raw.startswith(("ceu", "continuing")):
        return "ceu"
    if "semester" in raw:
        return "semester_hour"
    return "clock_hour"  # clock or contact hours


def number(raw: str) -> float:
    raw = raw.lower()
    if raw in WORDS:
        return float(WORDS[raw])
    return float(Fraction(raw)) if "/" in raw else float(raw)


def close(a: float, b: float) -> bool:
    return abs(a - b) <= TOLERANCE * abs(b) + 0.005


def find_conversions(text: str) -> list[Conversion]:
    out: list[Conversion] = []
    spans: list[tuple[int, int]] = []
    for m in STATEMENT.finditer(text):
        a, ua, b, ub = number(m[1]), unit_of(m[2]), number(m[3]), unit_of(m[4])
        if ua == ub or "ceu" not in (ua, ub) or a == 0:
            continue  # same unit, or semester/clock (the policies define no such conversion)
        if NEGATION.search(m[0]):
            continue  # "5 semester hours do not equal 5 CEUs"
        factor = float(PER_CEU[ub] / PER_CEU[ua])  # amount of ub in one ua
        if close(b, a * factor):
            status = "ok"
        elif close(b, a / factor):
            status = "backwards"
        else:
            status = "wrong"
        out.append(Conversion(m[0], status, round(a * factor, 4)))
        spans.append(m.span())
    for m in ARITHMETIC.finditer(text):
        if any(s < m.end() and m.start() < e for s, e in spans):
            continue  # already checked as a statement
        if m[2] and re.match(r"[^\S\n]+per\b", text[m.end(2) :]):
            continue  # "4 clock hours per ..." is a rate
        a, op, f, b = float(m[1]), m[3].lower(), FACTORS[m[4].lower()], float(m[5])
        target = unit_of(m[6])
        if target == "ceu":
            source = "clock_hour" if f == 10 else "semester_hour"
        elif m[2] and unit_of(m[2]) != "ceu":
            continue  # semester <-> clock: the policies define no such conversion
        else:
            source = "ceu"
        if f == 10 and "semester_hour" in (source, target):
            continue  # a factor of 10 next to semester hours is not a CEU conversion
        used = f if op in ("×", "x", "*", "times") else 1 / f
        correct = PER_CEU[target] / PER_CEU[source]
        if a == 0 or not close(b, a * float(used)):
            status = "wrong"  # the arithmetic itself is off
        elif used == correct:
            status = "ok"
        elif used == 1 / correct:
            status = "backwards"
        else:
            status = "wrong"
        out.append(Conversion(m[0], status, round(a * float(correct), 4)))
    return out


def answer_text(answer: dict[str, Any]) -> str:
    """Every sentence of a saved AppAnswer, one field per line."""
    parts = [answer.get("summary", "")]
    parts += [c["text"] for c in answer.get("claims", [])]
    for key in ("questions_for_user", "premise_corrections", "not_covered"):
        parts += answer.get(key) or []
    return "\n".join(parts)
