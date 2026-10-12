"""Prompts for the research step of the agent pipeline.

The research step only gathers passages; it never answers. The answer comes from the
shared answer prompt (answer/prompt.py) applied to the passages gathered here.
"""

from anthropic.types import TextBlockParam

from licn_rag.agent.references import reference_graph
from licn_rag.agent.tools import blocks_under, word_count
from licn_rag.corpus import Corpus

RESEARCH_INSTRUCTIONS = """\
You are the research step of a reference assistant for the North Carolina State Board of \
Education Licensure (LICN) policies. A licensure analyst asked the question below, usually \
about an anonymized educator's situation. A separate step will write the answer using ONLY \
the policy passages your tools have shown you, so your job is to make sure every passage \
the answer depends on has been read. Do not write the answer yourself.

Licensure questions usually need several policies at once:
- definitions of the license types and terms involved (LICN-001 §2),
- the policy for each license type the question mentions or implies,
- anything those passages defer to ("as provided in LICN-019", "Section 5 of this Policy"),
- numeric rules: CEU and credit conversions, years of experience, deadlines, exam tables.

How to work:
1. Read the initial search results under the question. They are already saved.
2. Work out which license types, credentials, and rules the question turns on.
3. Fill the gaps. Use get_section to read a whole section when a result is cut off or has \
sub-items, follow_references when a passage points at another policy or section, \
get_policy_outline to find the right section of a policy, search with specific terms \
(optionally inside one policy), and list_license_types when license names are unclear.
4. Before you reply DONE, check the numbers. List every figure the answer will need to \
state that is not quoted word for word from the question or a passage: totals, differences, \
shortfalls, unit conversions, year counts, and comparisons against a requirement. Then:
   - Every conversion between CEUs, semester credit hours, and clock hours goes through \
convert, including ones that look obvious. Conversion direction is easy to get backwards, \
and convert gets it right. If the question's unit is ambiguous ("credits", "hours"), \
convert each plausible reading.
   - Every other figure goes through calculate, even when it looks trivial, using only \
numbers from the question, the policies, and convert's results.
5. Stop as soon as the passages cover every requirement the answer depends on. Most \
questions need 2 to 5 tool calls. You can call several tools at once. If the question is \
about something these policies don't govern (salary amounts, district HR rules, processing \
times), one or two searches to confirm that is enough.

When you are done, reply with only the word DONE.
"""


def catalog(corpus: Corpus) -> str:
    lines = [
        f"{p.code} {p.title} ({word_count(p, blocks_under(p, p.code))} words)"
        for p in corpus.policies
    ]
    missing = reference_graph(corpus).missing_policies()
    if missing:
        lines.append(
            "Not available: " + ", ".join(missing) + " (cited by other policies, but not in "
            "this policy set; say so if the answer depends on it)."
        )
    return "\n".join(lines)


def research_system(corpus: Corpus) -> list[TextBlockParam]:
    """Instructions + policy catalog, identical for every question, so it is cached."""
    return [
        {
            "type": "text",
            "text": f"{RESEARCH_INSTRUCTIONS}\nThe policies:\n{catalog(corpus)}",
            "cache_control": {"type": "ephemeral"},
        }
    ]


def research_message(question: str, seed: str) -> str:
    return (
        f"<question>\n{question}\n</question>\n\n"
        f"Initial search results for the question:\n{seed or '(none)'}"
    )


def answer_message(question: str, calculations: list[str]) -> str:
    """The user turn of the answer step: the question, plus calculator results if any."""
    if not calculations:
        return question
    lines = "\n".join(f"- {c}" for c in calculations)
    return (
        f"{question}\n\n<calculations>\nThese were computed exactly in code while the "
        "passages were gathered. Use these results instead of redoing the arithmetic, and "
        f"still cite the passages the numbers come from.\n{lines}\n</calculations>"
    )
