# Prompt: generate draft eval questions (ungrounded, questions only)

Use this prompt in batches: run it once per category block below and ask for about 40 questions each time. Asking for 200 in one go causes repetition and drift.
Questions generated this way are **drafts**. They carry no answers. Expected answers and citations are attached later from the actual LICN policy text and reviewed by a human (see GAME_PLAN.md §7).

---

## Prompt (paste everything below this line, then fill in the CATEGORY block)

You are helping build a test set for a reference assistant. The assistant answers questions about North Carolina State Board of Education **Licensure (LICN) policies**. Its user is a **licensure analyst at Union County Public Schools (NC)**. While reviewing an educator's file, the analyst types questions *into the assistant* to look up what the policies require and how those requirements apply to a specific (anonymized) person.

Write questions **the analyst would type into the assistant**.

Rules:
- The question is addressed to the reference assistant, **not** to the educator. Never write "Have you…", "Did you…", or "Are you…" aimed at a candidate.
- Refer to the educator in the third person and anonymously: "a teacher", "this candidate", "an applicant". Never use names, IDs, or schools.
- Write the way a busy professional types: short, practical, sometimes informal. Vary the length and phrasing.
- Scenario questions must state concrete facts (degree, current license type, years of experience, credits earned, tests passed, program enrollment, out-of-state license, and so on). Pick realistic facts, and vary them across questions.
- Do **not** include answers, and do **not** state policy requirements as facts in the question. Describe only the person's situation and ask what applies.
- Cover a wide spread of license types and situations. For example: Residency License, Initial Professional License, Continuing Professional License, Emergency License, Permit to Teach, Limited License, add-on licensure areas, out-of-state and international applicants, Career and Technical Education, Exceptional Children, Birth–Kindergarten, school services personnel (counselor, school psychologist, school social worker, media), administrators, renewal, expired or lapsed licenses, reinstatement, and National Board certification. If you are unsure whether a license type exists in NC, phrase the question as "Is there a license that…" rather than asserting it.

Good examples:
- "Teacher has a BS in biology, is on year 2 of a Residency License, and hasn't passed the content test yet. What happens if they don't pass before the RL expires?"
- "Candidate holds a current professional license from South Carolina in middle grades math with 4 years of experience. What NC license can they get and what do they still need?"
- "How many renewal credits does a CPL holder need per renewal cycle, and do any have to be in specific areas?"
- "Teacher has 6.5 CEUs this cycle, 1.0 of them in digital learning. Are they good to renew?"
- "Is a teacher allowed to teach a class outside their licensure area, and for how long?"

Bad examples (do NOT write questions like these):
- "Have you kept copies of all documents submitted to NCDPI?" ← addressed to the candidate
- "Are you seeking a Birth through Kindergarten endorsement?" ← addressed to the candidate
- "Teachers need 8 CEUs to renew, right?" ← states a requirement as fact

Output **JSON Lines only** (one JSON object per line, no other text), with these fields:
```
{"id": "<category>-<nn>", "type": "<category>", "question": "...", "license_types": ["..."], "facts_given": ["..."], "missing_fact": "<for missing_facts type: the deciding fact the question leaves out; otherwise null>", "notes": "<optional: what makes this question tricky>"}
```

CATEGORY: <paste ONE block from below>

---

## Category blocks (run one per batch)

**lookup** (about 40 questions): direct questions about what a policy says, with no specific person. Example: "What's the difference between an IPL and a CPL?"

**eligibility** (about 80 questions; run 2 batches of 40): a specific anonymized person's facts → which license they qualify for, or whether they meet requirements for a named license. Include upgrades, out-of-state transfers, add-on areas, career changers, and lapsed licenses.

**renewal_math** (about 30 questions): questions that require counting credits, hours, years, or deadlines from stated numbers. Include some where the person is just short, some exactly at the threshold, and some where credits fall in the wrong category.

**missing_facts** (about 20 questions): scenarios that leave out exactly one fact that decides the answer (for example, the degree is never mentioned, or the years of experience are unclear). Put the missing fact in `missing_fact`. The correct system behavior is to ask for it.

**unanswerable** (about 20 questions): plausible questions that NC SBE licensure policy would not answer. Examples: salary schedules, district hiring decisions, union or HR leave rules, another state's rules, how long DPI takes to process an application. The correct system behavior is to say so.

**temporal** (about 10 questions): questions that depend on *when* a rule applied. Examples: "a teacher licensed before 2019 under the old rules…", "did the renewal requirements change this year?"
