"""What every answering pipeline returns, plus the shared call-and-parse step."""

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import anthropic
from anthropic.types import OutputConfigParam, TextBlockParam
from pydantic import ValidationError

from licn_rag.answer.schema import AppAnswer
from licn_rag.llm import first_text, output_schema, usage_cost

MAX_TOKENS = 16000

Effort = Literal["low", "medium", "high", "xhigh", "max"]


@dataclass
class PipelineResult:
    answer: AppAnswer | None
    model: str
    cost_usd: float = 0.0
    latency_s: float = 0.0
    error: str | None = None
    context_ids: list[str] | None = None
    """Citation IDs of the passages the model saw; None means the full corpus."""
    usage: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] | None = None
    """Agent pipelines: every tool call in order (tool, input, added_words, error)."""
    research: dict[str, Any] | None = None
    """Agent pipelines: research turns, why research stopped, and calculator results."""
    context_words: int | None = None
    """Words of policy text shown to the answering model; None if not measured."""


class Pipeline(Protocol):
    name: str

    def answer(self, question: str) -> PipelineResult: ...


def generate(
    client: anthropic.Anthropic,
    model: str,
    system: list[TextBlockParam],
    question: str,
    effort: Effort,
    context_ids: list[str] | None,
) -> PipelineResult:
    """One structured-output call: question in, validated AppAnswer out."""
    output_config: OutputConfigParam = {
        "effort": effort,
        "format": {"type": "json_schema", "schema": output_schema(AppAnswer)},
    }
    start = time.perf_counter()
    message = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": question}],
        output_config=output_config,
    )
    result = PipelineResult(
        answer=None,
        model=message.model,
        cost_usd=usage_cost(message.model, message.usage),
        latency_s=time.perf_counter() - start,
        context_ids=context_ids,
        usage=message.usage.model_dump(mode="json"),
    )
    if message.stop_reason != "end_turn":
        result.error = f"stop_reason={message.stop_reason}"
        return result
    try:
        result.answer = AppAnswer.model_validate_json(first_text(message))
    except ValidationError as e:
        result.error = f"invalid answer JSON: {e.errors()[:2]}"
    return result
