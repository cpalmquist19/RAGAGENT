# Plan 02: Agent loop

| | |
|---|---|
| **Plan** | 02 · agent phase 2 of 3 |
| **Status** | in progress (code done; real-API smoke test pending) |
| **Depends on** | 01 (research tools) must be `done` |
| **Estimated size** | 2 new source files (~250 lines), 4 small edits to existing files, 1 new test file (~200 lines) |
| **API cost** | $0 for tests (they use a fake client). 💲 About $0.01–0.03 for the manual smoke test in step 7. |

## Goal

Add an answering pipeline named `agent` (plus an ablation variant `agent_nocalc`) that:

1. runs a **seed search** on the question (the same hybrid search RAG uses, 12 chunks),
2. runs a **research loop**: Claude Haiku 5.5 calls the plan 01 tools until it says it's done or hits the turn limit,
3. answers with the **existing** `generate()` call and the **existing** answer prompt, showing the model every passage the tools collected (`Evidence.render()`), plus any calculator results.

After this plan, these work:

```bash
uv run licn-ask --pipeline agent --trace "Teacher has 6.5 CEUs this cycle on a CPL and teaches 4th grade. Enough to renew?"
uv run licn-eval run --pipeline agent --sample 30     # plan 03 runs this
```

## Background (why this design)

- **Two stages, not one.** The research loop and the answer are separate API calls. The answer call is byte-for-byte the same kind of request the RAG pipeline makes (same `build_system("retrieved", …)`, same `AppAnswer` schema), so the evaluation compares *retrieval strategies*, not prompts. It also keeps tool use and structured output in separate requests, which is simpler to debug.
- **Seed search first.** The loop starts with the 12 chunks hybrid search finds for the raw question, so the agent never starts from nothing and its recall can't fall far below RAG's.
- **Bounded.** At most `max_turns` research calls (default 8) and at most `max_evidence_words` of policy text (default 6,000 words, about a third of the corpus; RAG's 20 chunks are about 3,000). Both are enforced in code, not just requested in the prompt.
- **Cheap.** Haiku 5.5 throughout. Prompt caching: a breakpoint on the research system prompt (tools + system are cached together), plus top-level automatic caching so each loop turn reuses the previous turn's prefix. Expected cost is roughly $0.004–0.008 per question; plan 03 measures it.
- **Thin code.** A manual `while` loop over `client.messages.create`, no Tool Runner, no framework, matching the project's "thin code over frameworks" principle.

### Claude API facts this plan relies on (checked against the SDK in `uv.lock`, anthropic 1.13.0)

- `client.messages.create(...)` accepts `tools=`, `output_config={"effort": ...}`, and a **top-level** `cache_control={"type": "ephemeral"}` (automatic caching of the last cacheable block).
- Claude Haiku 5.5 runs adaptive thinking by default; don't pass `thinking`. Its default effort is `medium`; pass it explicitly anyway.
- **Forced tool choice is not used.** Leave `tool_choice` unset (`auto`).
- **Append `response.content` unchanged** as the assistant turn. It may contain `thinking` blocks that must be passed back exactly as received. Never edit or drop earlier messages in the loop.
- Return **all** `tool_result` blocks for one assistant turn in **one** user message, each with the matching `tool_use_id`. Failed tools use `"is_error": True`, never a dropped result.
- Check `stop_reason` before reading content: `"tool_use"` → run tools and continue; `"end_turn"` or `"max_tokens"` → stop researching; `"refusal"` → stop and return an error result.
- Tool `input` arrives already parsed as a `dict`. Don't do string matching on it.

## Read first

1. [implementation_plans/01-agent-p1-research-tools.md](01-agent-p1-research-tools.md) and the code it produced in `src/licn_rag/agent/`, especially `ToolBox` (`definitions`, `run`, `seed_search`, `evidence`, `calculations`, `trace`).
2. [src/licn_rag/pipelines/base.py](../src/licn_rag/pipelines/base.py): `PipelineResult`, `generate`, `Effort`.
3. [src/licn_rag/pipelines/rag.py](../src/licn_rag/pipelines/rag.py): `shared_index`, `Mode`, `make_rag` (copy its factory pattern).
4. [src/licn_rag/pipelines/registry.py](../src/licn_rag/pipelines/registry.py), [src/licn_rag/ask.py](../src/licn_rag/ask.py).
5. [src/licn_rag/answer/prompt.py](../src/licn_rag/answer/prompt.py): `build_system`. Don't change it.
6. [src/licn_rag/llm.py](../src/licn_rag/llm.py): `usage_cost`.

## Non-goals

- Don't change the answer prompt (`answer/prompt.py`) or the answer schema (`answer/schema.py`).
- Don't change the eval runner or scoring; plan 03 does that.
- No streaming, no async, no Tool Runner, no Batch API.
- No citation verifier, no worksheet format (later plans).
- Don't change the tools from plan 01 unless a test here exposes a real bug. If you do, say so in your report.

## Steps

### Step 1: Extend `PipelineResult` (`src/licn_rag/pipelines/base.py`)

Add three optional fields at the end of the dataclass, after `usage`. Existing pipelines don't set them, so nothing else changes.

```python
    trace: list[dict[str, Any]] | None = None
    """Agent pipelines: every tool call in order (tool, input, added_words, error)."""
    research: dict[str, Any] | None = None
    """Agent pipelines: research turns, why research stopped, and calculator results."""
    context_words: int | None = None
    """Words of policy text shown to the answering model; None if not measured."""
```

**Check:** `uv run pytest -q` still passes (nothing uses the fields yet).

---

### Step 2: The research prompt (`src/licn_rag/agent/prompt.py`)

Write this file. The instructions are deliberately general (no answers, no hints tied to specific eval questions); the only corpus-specific content is a generated catalog of policy codes and titles, which is what a person would see on the policy index page.

```python
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
4. Use calculate for every piece of arithmetic the answer will need (unit conversions, \
totals, shortfalls, year counts), using only numbers from the question and the policies.
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
```

Importing `blocks_under` and `word_count` from `agent.tools` is fine (no import cycle: `tools` doesn't import `prompt`).

**Check:** `uv run python -c "from licn_rag.corpus import Corpus; from licn_rag.agent.prompt import catalog; print(catalog(Corpus.load()))"` prints 24 policy lines and a `Not available: LICN-014 …` line. (Set `PYTHONIOENCODING=utf-8` if the console complains.)

---

### Step 3: The pipeline (`src/licn_rag/pipelines/agent.py`)

Write this file. Read the comments; they explain the parts that are easy to get wrong.

```python
"""Agentic retrieval: the model researches with tools, then answers from what it read.

    question ─► seed search (hybrid, 12 chunks)
             ─► research loop: Haiku calls tools (search, outline, read section, follow
                references, list license types, calculate) for up to max_turns calls
             ─► every passage the tools showed, in document order
             ─► the same structured-answer call the RAG pipeline makes

The answer step reuses generate() and the shared answer prompt, so this pipeline differs
from RAG only in how the passages were chosen. Calculator results are passed to the answer
step with the question.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import anthropic
from anthropic.types import MessageParam, ToolResultBlockParam

from licn_rag.agent.prompt import answer_message, research_message, research_system
from licn_rag.agent.tools import SEED_K, ToolBox
from licn_rag.answer.prompt import build_system
from licn_rag.corpus import Corpus
from licn_rag.llm import usage_cost
from licn_rag.pipelines.base import Effort, PipelineResult, generate
from licn_rag.pipelines.rag import Mode, shared_index
from licn_rag.rag.index import Index

RESEARCH_MAX_TOKENS = 8000
USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


@dataclass(frozen=True)
class AgentConfig:
    seed_k: int = SEED_K
    """Chunks from the seed search on the raw question; 0 skips it."""
    max_turns: int = 8
    """Research API calls before the loop stops regardless."""
    max_evidence_words: int = 6000
    """Reading tools refuse once this much policy text has been collected."""
    calculator: bool = True
    research_effort: Effort = "medium"
    answer_effort: Effort = "medium"
    search_mode: Mode = "hybrid"


VARIANTS: dict[str, AgentConfig] = {
    "agent": AgentConfig(),
    "agent_nocalc": AgentConfig(calculator=False),  # ablation: does the calculator help?
}


def add_usage(total: dict[str, int], usage: Any) -> None:
    for key in USAGE_FIELDS:
        total[key] = total.get(key, 0) + (getattr(usage, key, None) or 0)


class AgentPipeline:
    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        corpus: Corpus,
        config: AgentConfig,
        index: Index | None,
        name: str = "agent",
    ) -> None:
        self.client = client
        self.model = model
        self.corpus = corpus
        self.config = config
        self.index = index
        self.name = name
        self.system = research_system(corpus)

    def answer(self, question: str) -> PipelineResult:
        # Everything per-question lives in locals: eval runs call answer() from 8 threads.
        start = time.perf_counter()
        cfg = self.config
        box = ToolBox(
            self.corpus, self.index, cfg.search_mode, cfg.max_evidence_words, cfg.calculator
        )
        seed = box.seed_search(question, cfg.seed_k) if cfg.seed_k else ""
        messages: list[MessageParam] = [
            {"role": "user", "content": research_message(question, seed)}
        ]
        tools = box.definitions()
        cost = 0.0
        usage: dict[str, int] = {}
        turns = 0
        stop = "max_turns"

        while turns < cfg.max_turns:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=RESEARCH_MAX_TOKENS,
                system=self.system,
                tools=tools,
                messages=messages,
                output_config={"effort": cfg.research_effort},
                cache_control={"type": "ephemeral"},  # cache the growing conversation too
            )
            turns += 1
            cost += usage_cost(response.model, response.usage)
            add_usage(usage, response.usage)
            if response.stop_reason == "refusal":
                return PipelineResult(
                    answer=None,
                    model=response.model,
                    cost_usd=cost,
                    latency_s=time.perf_counter() - start,
                    error="research step refused (stop_reason=refusal)",
                    usage={"research": usage},
                    trace=box.trace,
                    research={"turns": turns, "stop": "refusal", "calculations": []},
                )
            if response.stop_reason != "tool_use":
                stop = str(response.stop_reason)  # "end_turn", or "max_tokens"
                break
            # Pass the assistant turn back exactly as received (it can hold thinking blocks).
            messages.append({"role": "assistant", "content": response.content})
            results: list[ToolResultBlockParam] = []
            for block in response.content:
                if block.type == "tool_use":
                    outcome = box.run(block.name, block.input)
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": outcome.text,
                            "is_error": outcome.is_error,
                        }
                    )
            # All results for one turn go back in a single user message.
            messages.append({"role": "user", "content": results})

        result = generate(
            self.client,
            self.model,
            build_system("retrieved", box.evidence.render()),
            answer_message(question, box.calculations),
            cfg.answer_effort,
            box.evidence.ids(),
        )
        result.cost_usd += cost
        result.latency_s = time.perf_counter() - start
        result.usage = {"research": usage, "answer": result.usage}
        result.trace = box.trace
        result.research = {"turns": turns, "stop": stop, "calculations": box.calculations}
        result.context_words = box.evidence.words()
        return result


def make_agent(name: str) -> Callable[[anthropic.Anthropic, str, Corpus], AgentPipeline]:
    def factory(client: anthropic.Anthropic, model: str, corpus: Corpus) -> AgentPipeline:
        config = VARIANTS[name]
        index = shared_index(True, config.search_mode != "bm25")
        return AgentPipeline(client, model, corpus, config, index, name=name)

    return factory
```

Notes:

- `generate()` returns a `PipelineResult` whose `cost_usd` and `latency_s` cover only the answer call. The code above adds the research cost and replaces the latency with the end-to-end time.
- If the research loop ends because of `max_turns`, the last message in `messages` is a tool-results message with no reply. That's fine: `messages` is thrown away; the answer step is a new request.
- `box.seed_search` returns `""` when the index is `None` (plan 01 guarantees this). Tests rely on it.
- **Possible pyright complaint:** `messages.append({"role": "assistant", "content": response.content})`. The SDK's `MessageParam` accepts response content blocks, so this should type-check. If pyright still reports an error on that line, add `# type: ignore[typeddict-item]` (or the code pyright names) to that one line. **Do not** convert the blocks to dicts or strip fields: thinking blocks must be passed back unchanged.

**Check:** `uv run pyright src` reports 0 errors.

---

### Step 4: Register the pipelines (`src/licn_rag/pipelines/registry.py`)

Add the agent variants. The module should end up as:

```python
"""All answering pipelines by name: the long-context benchmark, RAG variants, and agents."""

from collections.abc import Callable

import anthropic

from licn_rag.corpus import Corpus
from licn_rag.pipelines.agent import VARIANTS as AGENT_VARIANTS
from licn_rag.pipelines.agent import make_agent
from licn_rag.pipelines.base import Pipeline
from licn_rag.pipelines.long_context import LongContextPipeline
from licn_rag.pipelines.rag import VARIANTS, make_rag

Factory = Callable[[anthropic.Anthropic, str, Corpus], Pipeline]

PIPELINES: dict[str, Factory] = {
    "long_context": LongContextPipeline,
    **{name: make_rag(name) for name in VARIANTS},
    **{name: make_agent(name) for name in AGENT_VARIANTS},
}
```

Let `ruff check --fix` merge or order the imports however it wants.

Note: `evals/runner.py` imports `VARIANTS` from `pipelines.rag` for `licn-eval retrieval`. Leave that alone; the agent variants are not retrieval-only pipelines.

**Check:** `uv run licn-eval run --help` lists `agent` and `agent_nocalc` among the `--pipeline` choices.

---

### Step 5: `--trace` in the CLI (`src/licn_rag/ask.py`)

Add a flag:

```python
ap.add_argument("--trace", action="store_true", help="print the agent's tool calls")
```

After printing the answer and before the final `(model · latency · cost)` line, if `args.trace` and `result.trace` is not None, print:

```
Research (3 turns, stopped: end_turn, 2,410 words of policy text):
  seed_search      {"query": "...", "k": 12}                  +1,203 words
  get_section      {"citation_id": "LICN-030 §1"}            +391 words
  calculate        {"description": "...", "expression": "..."}  +0 words
  follow_references {...}                                     ERROR
```

Implementation sketch:

```python
if args.trace and result.trace is not None:
    r = result.research or {}
    print(
        f"\nResearch ({r.get('turns', '?')} turns, stopped: {r.get('stop', '?')}, "
        f"{result.context_words or 0:,} words of policy text):"
    )
    for t in result.trace:
        status = "ERROR" if t["error"] else f"+{t['added_words']:,} words"
        args_text = json.dumps(t["input"], ensure_ascii=False)
        print(f"  {t['tool']:18} {textwrap.shorten(args_text, 60):60} {status}")
```

Add `import json` at the top. Keep the rest of `ask.py` unchanged.

**Check:** `uv run licn-ask --help` shows `--trace`.

---

### Step 6: Tests (`tests/test_agent_pipeline.py`)

These tests use a **fake client**, so they make no API calls and cost nothing. They pass `index=None` and `AgentConfig(seed_k=0)`, so they don't need the search index either.

Fake client and response helpers (these constructors were checked against anthropic 1.13.0):

```python
"""The agent pipeline's loop, with a scripted fake client (no API calls)."""

from typing import Any

import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

from licn_rag.answer.schema import AppAnswer, Claim
from licn_rag.corpus import Corpus
from licn_rag.llm import usage_cost
from licn_rag.pipelines.agent import VARIANTS, AgentConfig, AgentPipeline
from licn_rag.pipelines.registry import PIPELINES

MODEL = "claude-haiku-5-5"
USAGE = Usage(input_tokens=100, output_tokens=20)


def message(content: list[Any], stop_reason: str) -> Message:
    return Message(
        id="msg_test",
        type="message",
        role="assistant",
        model=MODEL,
        content=content,
        stop_reason=stop_reason,  # type: ignore[arg-type]
        stop_sequence=None,
        usage=USAGE,
    )


def tool_use(id: str, name: str, **input: Any) -> ToolUseBlock:
    return ToolUseBlock(id=id, type="tool_use", name=name, input=input)


def text(t: str) -> TextBlock:
    return TextBlock(type="text", text=t)


ANSWER_JSON = AppAnswer(
    verdict="no",
    summary="Short by 1.5 CEUs.",
    claims=[Claim(text="K-5 teachers need eight CEUs.", citations=["LICN-030 §1(a)"])],
    questions_for_user=[],
    premise_corrections=[],
    not_covered=[],
).model_dump_json()


class FakeMessages:
    def __init__(self, responses: list[Message]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Message:
        # Copy the messages list: the pipeline keeps appending to the same list object.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[Message]) -> None:
        self.messages = FakeMessages(responses)


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return Corpus.load()


def run(corpus: Corpus, responses: list[Message], **config: Any) -> tuple[Any, FakeClient]:
    client = FakeClient(responses)
    pipeline = AgentPipeline(
        client,  # type: ignore[arg-type]
        MODEL,
        corpus,
        AgentConfig(seed_k=0, **config),
        index=None,
    )
    return pipeline.answer("Teacher has 6.5 CEUs on a CPL and teaches 4th grade. Enough?"), client
```

Write these tests:

**`test_research_then_answer`**: script = [tool_use turn with two calls (`get_section` `citation_id="LICN-030 §1"` as `toolu_1`, and `calculate` `description="CEUs short"`, `expression="8 - 6.5"` as `toolu_2`) with stop `"tool_use"`; then `[text("DONE")]` with `"end_turn"`; then `[text(ANSWER_JSON)]` with `"end_turn"`]. Assert:

- `result.answer is not None and result.answer.verdict == "no"` and `result.error is None`
- `len(client.messages.calls) == 3`
- the second call's messages have length 3: `[user, assistant, user]`; the last one's `content` is a list of 2 `tool_result` dicts with `tool_use_id`s `["toolu_1", "toolu_2"]`, both `is_error` False
- the assistant message in the second call is the exact `content` list from the first response (`is` or `==` comparison)
- the answer call (`calls[2]`) has **no** `tools` key; its `system[1]["text"]` contains `"[LICN-030 §1(a)]"`; its `messages[0]["content"]` contains `"<calculations>"` and `"8 - 6.5 = 1.5"`
- `"LICN-030 §1(a)" in result.context_ids`
- `[t["tool"] for t in result.trace] == ["get_section", "calculate"]`
- `result.research == {"turns": 2, "stop": "end_turn", "calculations": ["CEUs short: 8 - 6.5 = 1.5"]}`
- `result.context_words > 0`
- `result.cost_usd == pytest.approx(3 * usage_cost(MODEL, USAGE))`
- `set(result.usage) == {"research", "answer"}` and `result.usage["research"]["input_tokens"] == 200`

**`test_research_request_shape`**: same script as above. On `calls[0]`:

- `[t["name"] for t in calls[0]["tools"]]` equals the sorted list of the 6 tool names
- `calls[0]["cache_control"] == {"type": "ephemeral"}`
- `calls[0]["system"][-1]["cache_control"] == {"type": "ephemeral"}`
- `calls[0]["output_config"] == {"effort": "medium"}`
- `"LICN-019 Residency License"` and `"LICN-014"` both appear in `calls[0]["system"][0]["text"]`
- `"tool_choice" not in calls[0]`
- `calls[0]["messages"][0]["content"]` contains `"<question>"` and `"(none)"` (no seed with `seed_k=0`)
- `calls[0]["tools"] == calls[1]["tools"]` and `calls[0]["system"] == calls[1]["system"]` (stable prefix, so caching works)

**`test_turn_limit`**: `max_turns=2`; script = two identical tool_use turns (one `calculate` call each), then the answer. Assert 3 calls total, `result.research["stop"] == "max_turns"`, `result.research["turns"] == 2`, and `result.answer is not None`.

**`test_refusal_stops_without_answer`**: script = `[message([], "refusal")]`. Assert `result.answer is None`, `"refusal" in result.error`, exactly 1 call.

**`test_done_immediately`**: script = `[text("DONE")]` end_turn, then the answer. Assert 2 calls, `result.context_ids == []`, `result.context_words == 0`, `result.trace == []`, and the answer call's user message equals the question exactly (no calculations block).

**`test_max_tokens_in_research_still_answers`**: script = `[text("partial")]` with `"max_tokens"`, then the answer. Assert `result.research["stop"] == "max_tokens"` and `result.answer is not None`.

**`test_tool_error_is_reported_not_raised`**: script = tool_use `get_section` with `citation_id="LICN-030 §99"`, then DONE, then the answer. Assert the tool_result in `calls[1]` has `is_error` True and the trace entry has `error` True.

**`test_nocalc_variant`**: `VARIANTS["agent_nocalc"].calculator is False`; run a pipeline with `calculator=False` and the DONE-then-answer script; assert `"calculate"` is not among `calls[0]["tools"]` names.

**`test_registry`**: `{"agent", "agent_nocalc"} <= set(PIPELINES)`.

**Check:** `uv run pytest -q` passes.

---

### Step 7: 💲 Smoke test against the real API (about $0.01–0.03)

**Tell the user before running this**, since it calls the API. It needs `ANTHROPIC_API_KEY` in `.env` and the built index (`data/index/licn.sqlite`). The first run loads the Qwen3 embedding model from the local cache, which takes a little while.

```bash
uv run licn-ask --pipeline agent --trace --sources "Teacher has 6.5 CEUs this cycle on a CPL and teaches 4th grade. Enough to renew?"
uv run licn-ask --pipeline agent --trace "Someone holds a clear teaching license from Virginia with 4 years of experience. What NC license can they get?"
```

Look for:

- a sensible verdict and summary, with citations;
- a trace with a `seed_search` entry plus a few tool calls, `stopped: end_turn`, and well under 6,000 words of policy text;
- for the first question, a `calculate` call for the CEU shortfall;
- for the second, reads from LICN-025 (out-of-state applicants) and probably LICN-024/LICN-020;
- cost per question printed at the end, expected under $0.01.

If the agent never calls tools, or loops until `max_turns` every time, don't tune the prompt in this plan. Record what happened (paste the trace) in your report; plan 03 handles tuning with measurements.

## Acceptance checklist

- [x] `uv run pytest -q`: all tests pass (104).
- [x] `uv run ruff check .` and `uv run ruff format --check .` pass.
- [x] `uv run pyright src`: 0 errors.
- [x] `uv run licn-eval run --help` lists `agent` and `agent_nocalc`.
- [x] `answer/prompt.py`, `answer/schema.py`, `evals/`, `rag/` are unchanged (`git diff --stat`).
- [ ] **Still open:** smoke test done (or explicitly skipped by the user), with the two traces and costs in your report.

## Commit

Branch `plan-02-agent-loop`. One commit:

```
Add agentic retrieval pipeline (implementation plan 02)

A seed search, then a bounded Haiku tool-use loop over the plan 01 research
tools, then the shared structured-answer call over every passage the tools
showed. Registered as `agent` and `agent_nocalc`; `licn-ask --trace` prints
the tool calls.
```

Set this plan's **Status** to `done` here and in [README.md](README.md).

## Gotchas

- **The fake client must copy `messages`.** The pipeline appends to the same list after each call, so a fake that stores the list by reference will show later state in earlier calls and the assertions will look wrong.
- **Don't put the question or anything per-request into `self.system`.** It must be identical across questions so it caches. The question goes in the first user message.
- **Don't sort, filter, or rebuild `tools` between turns.** `box.definitions()` is computed once per question and is identical across questions (same specs, sorted by name).
- **Don't reuse a `ToolBox` across questions** and don't store per-question state on `self`.
- **`stop_reason` can be `"pause_turn"` only with server tools**, which this pipeline doesn't use. Treat any value other than `"tool_use"` and `"refusal"` as "stop researching".
- **API errors** (`anthropic.APIError`) from either stage should propagate; the eval runner already catches them per question. Don't swallow them.
