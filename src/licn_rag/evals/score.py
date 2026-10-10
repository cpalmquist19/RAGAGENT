"""Score one pipeline answer against its answer key.

Two layers:
- Deterministic checks: verdict, abstention, asking for missing facts, and citation
  overlap with the answer key at the policy and section level.
- An LLM judge: does the answer cover the key's points without contradicting them, and
  does each cited passage actually support the claim it is attached to?

Answer keys over-cite (they list every relevant passage), so citation *recall* is
measured at the section level ("LICN-030 §1" rather than "LICN-030 §1(a)(ii)").
"""

import re
from typing import Any, Literal

import anthropic
from pydantic import BaseModel, ValidationError

from licn_rag.answer.schema import AppAnswer
from licn_rag.corpus import Corpus
from licn_rag.llm import first_text, output_schema, usage_cost

# ---------------------------------------------------------------- deterministic metrics


def section_of(citation: str) -> str:
    """'LICN-030 §1(a)(ii)' -> 'LICN-030 §1'; other forms are returned unchanged."""
    m = re.match(r"(LICN-\d{3}) §([^()\s]+)", citation)
    return f"{m.group(1)} §{m.group(2)}" if m else citation


def policy_of(citation: str) -> str:
    return citation[:8]


def ratio(hit: int, total: int) -> float | None:
    return hit / total if total else None


def deterministic(answer: AppAnswer, key: dict[str, Any], corpus: Corpus) -> dict[str, Any]:
    raw = answer.citations()
    normalized = [corpus.normalize(c) for c in raw]
    cited = [c for c in normalized if c]
    expected = list(
        dict.fromkeys(
            key["expected_citations"] + [c for kp in key["key_points"] for c in kp["citations"]]
        )
    )
    exp_sec, got_sec = {section_of(c) for c in expected}, {section_of(c) for c in cited}
    exp_pol, got_pol = {policy_of(c) for c in expected}, {policy_of(c) for c in cited}

    answerable = key["answerable"]
    out_of_scope = answerable == "no"
    needs_info = key["verdict"] == "needs_more_info"
    asked = answer.verdict == "needs_more_info" or bool(answer.questions_for_user)

    return {
        "verdict_match": answer.verdict == key["verdict"],
        "invalid_citations": [c for c, n in zip(raw, normalized, strict=True) if n is None],
        "uncited_claims": sum(1 for c in answer.claims if not c.citations),
        # Citation overlap is only meaningful when the key expects citations.
        "policy_recall": None if out_of_scope else ratio(len(exp_pol & got_pol), len(exp_pol)),
        "section_recall": None if out_of_scope else ratio(len(exp_sec & got_sec), len(exp_sec)),
        "section_precision": None if out_of_scope else ratio(len(exp_sec & got_sec), len(got_sec)),
        # Out-of-scope questions should be declined; in-scope ones should not be.
        "abstained_correctly": (answer.verdict == "not_covered") if out_of_scope else None,
        "false_abstention": (answer.verdict == "not_covered") if answerable == "yes" else None,
        "asked_for_missing_facts": asked if needs_info else None,
    }


# ---------------------------------------------------------------- LLM judge


class KeyPointGrade(BaseModel):
    index: int
    status: Literal["covered", "partial", "missing", "contradicted"]


class ClaimGrade(BaseModel):
    index: int
    support: Literal["supported", "partially_supported", "unsupported"]


class Judgment(BaseModel):
    key_points: list[KeyPointGrade]
    claims: list[ClaimGrade]
    incorrect_statements: list[str]
    overall: Literal["correct", "partially_correct", "incorrect"]
    rationale: str


JUDGE_INSTRUCTIONS = """\
You grade answers from a reference assistant about North Carolina educator licensure \
(LICN) policies. You get the question, an answer key written from the policies, and the \
assistant's answer: its claims, each followed by the policy text it cited.

Grade three things:

1. key_points: for each numbered key point in the answer key, is it "covered" (the \
answer states it or something equivalent), "partial", "missing", or "contradicted" (the \
answer says something incompatible with it)? The key is deliberately thorough; missing a \
peripheral point is normal.
2. claims: for each numbered assistant claim, judge ONLY against the cited policy text \
shown under it. "supported" if that text backs the claim, "partially_supported" if it backs \
part of it, "unsupported" if it does not (including claims with no citations). Arithmetic \
done correctly from supported numbers counts as supported.
3. overall: "correct" if the answer's bottom line and essential points agree with the \
answer key and it states nothing materially wrong; "partially_correct" if the bottom line \
is right but something important is missing or slightly wrong, or it hedges where the key \
is definite; "incorrect" if the bottom line is wrong, it misses the central point, or it \
states something materially false. Asking for missing facts when the key also does is \
correct behavior. Declining to answer is correct only if the key says the policies do not \
cover the question.

List any materially wrong statements in incorrect_statements. Keep rationale to 1-3 \
sentences.
"""


def judge_prompt(question: str, key: dict[str, Any], answer: AppAnswer, corpus: Corpus) -> str:
    lines = [f"QUESTION:\n{question}\n", "ANSWER KEY:"]
    lines.append(f"Verdict: {key['verdict']} (answerable from policies: {key['answerable']})")
    lines.append(f"Reference answer: {key['reference_answer']}")
    for i, kp in enumerate(key["key_points"]):
        lines.append(f"  Key point {i}: {kp['point']}")
    for label, field in [
        ("Facts the assistant should ask for", "missing_facts_to_ask"),
        ("False premises to correct", "false_premises"),
        ("Not covered by the policies", "out_of_scope_parts"),
    ]:
        if key[field]:
            lines.append(f"{label}: " + " | ".join(key[field]))

    lines += ["\nASSISTANT ANSWER:", f"Verdict: {answer.verdict}", f"Summary: {answer.summary}"]
    for i, claim in enumerate(answer.claims):
        lines.append(f"\n  Claim {i}: {claim.text}")
        if not claim.citations:
            lines.append("    (no citations)")
        for c in claim.citations:
            text = corpus.passage(corpus.normalize(c) or c)
            lines.append(f"    Cited {c}: {text or '(citation does not exist in the policies)'}")
    for label, items in [
        ("Asked the analyst", answer.questions_for_user),
        ("Premise corrections", answer.premise_corrections),
        ("Said not covered", answer.not_covered),
    ]:
        if items:
            lines.append(f"\n{label}: " + " | ".join(items))
    return "\n".join(lines)


def judge(
    client: anthropic.Anthropic,
    model: str,
    question: str,
    key: dict[str, Any],
    answer: AppAnswer,
    corpus: Corpus,
) -> tuple[dict[str, Any], float]:
    message = client.messages.create(
        model=model,
        max_tokens=8000,
        system=JUDGE_INSTRUCTIONS,
        messages=[{"role": "user", "content": judge_prompt(question, key, answer, corpus)}],
        output_config={
            "effort": "medium",
            "format": {"type": "json_schema", "schema": output_schema(Judgment)},
        },
    )
    cost = usage_cost(message.model, message.usage)
    if message.stop_reason != "end_turn":
        return {"judge_error": f"stop_reason={message.stop_reason}"}, cost
    try:
        j = Judgment.model_validate_json(first_text(message))
    except ValidationError as e:
        return {"judge_error": f"invalid judgment JSON: {e.errors()[:2]}"}, cost

    kp_value = {"covered": 1.0, "partial": 0.5, "missing": 0.0, "contradicted": 0.0}
    claim_value = {"supported": 1.0, "partially_supported": 0.5, "unsupported": 0.0}
    n_kp, n_claims = len(key["key_points"]), len(answer.claims)
    return {
        "overall": j.overall,
        "key_point_coverage": ratio_f(sum(kp_value[g.status] for g in j.key_points), n_kp),
        "contradicted_key_points": sum(g.status == "contradicted" for g in j.key_points),
        "claim_support": ratio_f(sum(claim_value[g.support] for g in j.claims), n_claims),
        "unsupported_claims": sum(g.support == "unsupported" for g in j.claims),
        "incorrect_statements": j.incorrect_statements,
        "judge_rationale": j.rationale,
    }, cost


def ratio_f(total: float, n: int) -> float | None:
    return total / n if n else None
