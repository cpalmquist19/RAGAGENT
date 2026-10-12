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


def key_sections(key: dict[str, Any]) -> set[str]:
    """Sections the answer key cites, in its expected citations and its key points."""
    cites = set(key["expected_citations"]) | {
        c for kp in key["key_points"] for c in kp["citations"]
    }
    return {section_of(c) for c in cites}


def context_recall(context_ids: list[str] | None, key: dict[str, Any]) -> float | None:
    """Share of the key's sections that were in the answering model's context.

    None when the whole corpus was in context (context_ids is None) or the question is
    out of scope. Same definition as `licn-eval retrieval`'s Recall@k, so the numbers are
    comparable.
    """
    if context_ids is None or key["answerable"] == "no":
        return None
    return ratio(
        len(key_sections(key) & {section_of(c) for c in context_ids}), len(key_sections(key))
    )


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
    central: bool
    """True if the key's bottom line depends on this point; False for supporting detail."""
    status: Literal["covered", "partial", "missing", "contradicted"]


class ClaimGrade(BaseModel):
    index: int
    support: Literal["supported", "partially_supported", "unsupported"]


class Judgment(BaseModel):
    key_points: list[KeyPointGrade]
    claims: list[ClaimGrade]
    bottom_line: Literal["agrees", "partly_agrees", "disagrees"]
    material_errors: list[str]
    minor_issues: list[str]
    rationale: str


# Big enough for the longest cited passage (the LICN-003 §5 exam table is ~6,500
# characters; whole short policies up to ~9,300). A cut-off passage made the judge mark
# claims about the table's later rows as unsupported.
JUDGE_PASSAGE_CHARS = 12000
PARENT_CHARS = 600

JUDGE_INSTRUCTIONS = """\
You grade answers from a reference assistant about North Carolina educator licensure \
(LICN) policies. You get the question, an answer key written from the policies, and the \
assistant's answer: its claims, each followed by the policy text it cited. A cited list \
item is shown after the lead-in text of the sections above it (marked "context:"), \
because a list item means what its lead-in says; judge the item together with that \
context.

Make these judgments:

1. key_points: for each numbered key point in the answer key:
   - central: true if the key's bottom line depends on this point (the rule that decides \
the question, or a deciding fact to ask for); false for supporting detail such as fees, \
application documents, salary placement, or adjacent routes. The key is deliberately \
thorough, so most keys have only one to three central points.
   - status: "covered" (the answer states it or something equivalent), "partial", \
"missing", or "contradicted" (the answer says something incompatible with it).
2. claims: for each numbered assistant claim, judge ONLY against the cited policy text \
shown under it (with its context). "supported" if that text backs the claim, \
"partially_supported" if it backs part of it, "unsupported" if it does not (including \
claims with no citations). Arithmetic done correctly from supported numbers counts as \
supported.
3. bottom_line: compare the conclusion an analyst would take away from the answer (its \
summary and verdict together) with the key's. "agrees" if it is the same conclusion, \
including when the answer asks for the same deciding facts the key asks for; \
"partly_agrees" if it is right on part of a multi-part question or hedges where the key \
gives a definite outcome; "disagrees" if it reaches a different outcome, or declines to \
answer when the key says the policies do answer it.
4. material_errors: statements in the answer that are wrong according to the policy text \
or the key AND would change what the analyst concludes or does about this question. A \
statement of policy that the passages do not support, presented as fact, counts if the \
analyst could act on it. Do not list omissions here.
5. minor_issues: other problems: wording that overstates the cited text, peripheral \
inaccuracies, small unsupported inferences. Do not list omissions of supporting detail.

Keep rationale to 1-3 sentences.
"""


def overall_label(j: Judgment) -> str:
    """correct / partially_correct / incorrect, computed from the parts of the judgment."""
    central = [g for g in j.key_points if g.central]
    if j.bottom_line == "disagrees" or any(g.status == "contradicted" for g in central):
        return "incorrect"
    if (
        j.bottom_line == "agrees"
        and not j.material_errors
        and all(g.status in ("covered", "partial") for g in central)
    ):
        return "correct"
    return "partially_correct"


def parent_context(corpus: Corpus, cid: str) -> list[str]:
    """Lead-in lines of the sections above a cited item, outermost first.

    'LICN-013 §3(b)' reads only "Emergency License"; its meaning ("DPI shall not add
    license areas to the following") is in 'LICN-013 §3'.
    """
    ancestors = []
    current = cid
    while (parent := re.sub(r"\([^()]*\)$", "", current)) != current:
        ancestors.append(parent)
        current = parent
    lines = []
    for anc in reversed(ancestors):
        text = corpus.passage(anc, max_chars=10**6) or ""
        own = [line for line in text.split("\n") if line.startswith(f"[{anc}]")]
        if own:
            lines.append(own[0][:PARENT_CHARS])
    return lines


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
            cid = corpus.normalize(c) or c
            text = corpus.passage(cid, max_chars=JUDGE_PASSAGE_CHARS)
            if text is None:
                lines.append(f"    Cited {c}: (citation does not exist in the policies)")
                continue
            # The answering model sees each policy's title and effective date; so does the judge.
            p = corpus.by_code[cid[:8]]
            header = f"{p.code} {p.title}, effective {p.effective_date or 'unknown'}"
            lines.append(f"    Cited {c} [{header}]:")
            lines += [f"      context: {line}" for line in parent_context(corpus, cid)]
            lines.append(f"      {text}")
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
    central = [g for g in j.key_points if g.central]
    return {
        "overall": overall_label(j),
        "bottom_line": j.bottom_line,
        "key_point_coverage": ratio_f(sum(kp_value[g.status] for g in j.key_points), n_kp),
        "central_coverage": ratio_f(sum(kp_value[g.status] for g in central), len(central)),
        "contradicted_key_points": sum(g.status == "contradicted" for g in j.key_points),
        "claim_support": ratio_f(sum(claim_value[g.support] for g in j.claims), n_claims),
        "unsupported_claims": sum(g.support == "unsupported" for g in j.claims),
        "incorrect_statements": j.material_errors,
        "minor_issues": j.minor_issues,
        "judge_rationale": j.rationale,
    }, cost


def ratio_f(total: float, n: int) -> float | None:
    return total / n if n else None
