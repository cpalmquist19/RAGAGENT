"""The structured answer every pipeline returns.

The verdict vocabulary matches the answer keys (evals/data/silver_answers.jsonl), so
pipelines can be scored against them directly.
"""

from typing import Literal

from pydantic import BaseModel

Verdict = Literal["yes", "no", "conditional", "needs_more_info", "informational", "not_covered"]


class Claim(BaseModel):
    text: str
    citations: list[str]
    """Citation IDs of the passages that support this claim, e.g. "LICN-030 §1(a)"."""


class AppAnswer(BaseModel):
    verdict: Verdict
    summary: str
    """The bottom line the analyst can act on, in 1-3 sentences."""
    claims: list[Claim]
    """Every factual statement the answer relies on, each with its supporting citations."""
    questions_for_user: list[str]
    """Facts the analyst should supply when the outcome depends on them."""
    premise_corrections: list[str]
    """Assumptions in the question that the policies contradict."""
    not_covered: list[str]
    """Parts of the question the policies don't address."""

    def citations(self) -> list[str]:
        return list(dict.fromkeys(c for claim in self.claims for c in claim.citations))
