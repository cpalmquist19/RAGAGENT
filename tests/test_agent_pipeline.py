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
QUESTION = "Teacher has 6.5 CEUs on a CPL and teaches 4th grade. Enough?"


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
    return pipeline.answer(QUESTION), client


def research_script() -> list[Message]:
    """Two tool calls in one turn, DONE, then the answer."""
    return [
        message(
            [
                tool_use("toolu_1", "get_section", citation_id="LICN-030 §1"),
                tool_use("toolu_2", "calculate", description="CEUs short", expression="8 - 6.5"),
            ],
            "tool_use",
        ),
        message([text("DONE")], "end_turn"),
        message([text(ANSWER_JSON)], "end_turn"),
    ]


def test_research_then_answer(corpus: Corpus) -> None:
    script = research_script()
    first_content = script[0].content
    result, client = run(corpus, script)
    calls = client.messages.calls

    assert result.answer is not None and result.answer.verdict == "no"
    assert result.error is None
    assert len(calls) == 3

    second = calls[1]["messages"]
    assert len(second) == 3
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert second[1]["content"] == first_content
    tool_results = second[2]["content"]
    assert [r["tool_use_id"] for r in tool_results] == ["toolu_1", "toolu_2"]
    assert all(r["type"] == "tool_result" and not r["is_error"] for r in tool_results)

    answer_call = calls[2]
    assert "tools" not in answer_call
    assert "[LICN-030 §1(a)]" in answer_call["system"][1]["text"]
    content = answer_call["messages"][0]["content"]
    assert "<calculations>" in content and "8 - 6.5 = 1.5" in content

    assert "LICN-030 §1(a)" in result.context_ids
    assert [t["tool"] for t in result.trace] == ["get_section", "calculate"]
    assert result.research == {
        "turns": 2,
        "stop": "end_turn",
        "calculations": ["CEUs short: 8 - 6.5 = 1.5"],
    }
    assert result.context_words > 0
    assert result.cost_usd == pytest.approx(3 * usage_cost(MODEL, USAGE))
    assert set(result.usage) == {"research", "answer"}
    assert result.usage["research"]["input_tokens"] == 200


def test_research_request_shape(corpus: Corpus) -> None:
    _, client = run(corpus, research_script())
    calls = client.messages.calls

    names = [t["name"] for t in calls[0]["tools"]]
    assert names == sorted(
        ["search", "get_policy_outline", "get_section", "follow_references"]
        + ["list_license_types", "calculate"]
    )
    assert calls[0]["cache_control"] == {"type": "ephemeral"}
    assert calls[0]["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert calls[0]["output_config"] == {"effort": "medium"}
    system_text = calls[0]["system"][0]["text"]
    assert "LICN-019 Residency License" in system_text and "LICN-014" in system_text
    assert "tool_choice" not in calls[0]
    first_message = calls[0]["messages"][0]["content"]
    assert "<question>" in first_message and "(none)" in first_message
    # A stable prefix across turns is what makes prompt caching work.
    assert calls[0]["tools"] == calls[1]["tools"]
    assert calls[0]["system"] == calls[1]["system"]


def test_turn_limit(corpus: Corpus) -> None:
    turn = [tool_use("toolu_1", "calculate", description="x", expression="1 + 1")]
    script = [
        message(turn, "tool_use"),
        message(turn, "tool_use"),
        message([text(ANSWER_JSON)], "end_turn"),
    ]
    result, client = run(corpus, script, max_turns=2)
    assert len(client.messages.calls) == 3
    assert result.research["stop"] == "max_turns"
    assert result.research["turns"] == 2
    assert result.answer is not None


def test_refusal_stops_without_answer(corpus: Corpus) -> None:
    result, client = run(corpus, [message([], "refusal")])
    assert result.answer is None
    assert "refusal" in result.error
    assert len(client.messages.calls) == 1


def test_done_immediately(corpus: Corpus) -> None:
    script = [message([text("DONE")], "end_turn"), message([text(ANSWER_JSON)], "end_turn")]
    result, client = run(corpus, script)
    calls = client.messages.calls
    assert len(calls) == 2
    assert result.context_ids == []
    assert result.context_words == 0
    assert result.trace == []
    assert calls[1]["messages"][0]["content"] == QUESTION


def test_max_tokens_in_research_still_answers(corpus: Corpus) -> None:
    script = [message([text("partial")], "max_tokens"), message([text(ANSWER_JSON)], "end_turn")]
    result, _ = run(corpus, script)
    assert result.research["stop"] == "max_tokens"
    assert result.answer is not None


def test_tool_error_is_reported_not_raised(corpus: Corpus) -> None:
    script = [
        message([tool_use("toolu_1", "get_section", citation_id="LICN-030 §99")], "tool_use"),
        message([text("DONE")], "end_turn"),
        message([text(ANSWER_JSON)], "end_turn"),
    ]
    result, client = run(corpus, script)
    tool_result = client.messages.calls[1]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True
    assert result.trace[0]["error"] is True
    assert result.answer is not None


def test_nocalc_variant(corpus: Corpus) -> None:
    assert VARIANTS["agent_nocalc"].calculator is False
    script = [message([text("DONE")], "end_turn"), message([text(ANSWER_JSON)], "end_turn")]
    _, client = run(corpus, script, calculator=False)
    assert "calculate" not in [t["name"] for t in client.messages.calls[0]["tools"]]


def test_registry() -> None:
    assert {"agent", "agent_nocalc"} <= set(PIPELINES)
