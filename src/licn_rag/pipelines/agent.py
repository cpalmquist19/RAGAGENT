"""Agentic retrieval: the model researches with tools, then answers from what it read.

    question ─► seed search (hybrid, 12 chunks)
             ─► research loop: Haiku calls tools (search, outline, read section, follow
                references, list license types, calculate, convert) for up to max_turns calls
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
    "agent_nocalc": AgentConfig(calculator=False),  # ablation: do calculate and convert help?
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
