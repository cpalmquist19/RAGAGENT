"""Answer-step replay: re-answer from the passages a saved agent run already collected.

For answer-prompt experiments. The research loop is skipped, so a prompt change is measured
without retrieval variance, at roughly a third of an agent run's cost:

    saved run (context_ids, calculator results) ─► same passages ─► answer step only

The passages are rebuilt from the saved citation IDs as whole block lines. That is close to,
but not byte-identical with, what the agent showed (a search hit on part of the exam table
comes back as the whole table), so compare a replay with a replay of the same run, never
with the original run.
"""

import json
import time
from pathlib import Path
from typing import Any

import anthropic

from licn_rag.agent.evidence import Evidence
from licn_rag.agent.prompt import answer_message
from licn_rag.answer.prompt import build_system
from licn_rag.corpus import Corpus, cite_id
from licn_rag.ingest.models import Block, Policy
from licn_rag.pipelines.base import Effort, PipelineResult, generate


def load_records(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Saved agent records keyed by question text."""
    lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line]
    return {r["question"]: r for r in records if r.get("context_ids") is not None}


def blocks_by_id(corpus: Corpus) -> dict[str, tuple[Policy, list[Block]]]:
    out: dict[str, tuple[Policy, list[Block]]] = {}
    for policy in corpus.policies:
        for block in policy.blocks:
            if block.kind != "heading":
                out.setdefault(cite_id(policy, block), (policy, []))[1].append(block)
    return out


class ReplayPipeline:
    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        corpus: Corpus,
        records: dict[str, dict[str, Any]],
        effort: Effort = "medium",
        name: str = "replay",
    ) -> None:
        self.client = client
        self.model = model
        self.corpus = corpus
        self.records = records
        self.effort: Effort = effort
        self.name = name
        self._blocks = blocks_by_id(corpus)

    def answer(self, question: str) -> PipelineResult:
        start = time.perf_counter()
        saved = self.records.get(question)
        if saved is None:
            return PipelineResult(answer=None, model=self.model, error="not in the replayed run")
        evidence = Evidence(self.corpus)
        for cid in saved["context_ids"]:
            if cid in self._blocks:
                policy, blocks = self._blocks[cid]
                evidence.add_blocks(policy, blocks)
        calculations = (saved.get("research") or {}).get("calculations", [])
        result = generate(
            self.client,
            self.model,
            build_system("retrieved", evidence.render()),
            answer_message(question, calculations),
            self.effort,
            evidence.ids(),
        )
        result.latency_s = time.perf_counter() - start
        result.context_words = evidence.words()
        result.research = saved.get("research")
        return result
