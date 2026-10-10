"""The production pipeline: retrieve passages, then answer only from them.

    question ─► hybrid search (dense + BM25, fused with RRF) ─► top-k chunks
             ─► passages grouped by policy, in document order ─► structured answer

Variants (dense-only, BM25-only, reranked, no context notes) exist for the ablation
table; they share everything except the retrieval step.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Literal

import anthropic

from licn_rag.answer.prompt import build_system
from licn_rag.corpus import Corpus
from licn_rag.pipelines.base import Effort, PipelineResult, generate
from licn_rag.rag.chunking import Chunk
from licn_rag.rag.index import INDEX_DIR, Embedder, Index, Reranker

Mode = Literal["dense", "bm25", "hybrid"]


@dataclass(frozen=True)
class RetrievalConfig:
    mode: Mode = "hybrid"
    rerank: bool = False
    contextual: bool = True
    k: int = 20
    candidates: int = 30


# Heavy models load once per process and are shared by every pipeline instance.
@cache
def shared_embedder() -> Embedder:
    return Embedder()


@cache
def shared_reranker() -> Reranker:
    return Reranker()


@cache
def shared_index(contextual: bool, with_embedder: bool) -> Index:
    path = INDEX_DIR / ("licn.sqlite" if contextual else "licn-plain.sqlite")
    return Index(path, shared_embedder() if with_embedder else None)


def render_passages(chunks: list[Chunk], corpus: Corpus) -> str:
    """Retrieved chunks grouped under their policy header, in document order."""
    out: list[str] = []
    current = None
    for chunk in sorted(chunks, key=lambda c: c.order):
        if chunk.policy_code != current:
            if current:
                out.append("</policy>")
            p = corpus.by_code[chunk.policy_code]
            out.append(
                f'<policy code="{p.code}" title="{p.title}" '
                f'effective="{p.effective_date or "unknown"}">'
            )
            current = chunk.policy_code
        out.append(f"<!-- {chunk.breadcrumb} -->\n{chunk.text}")
    if current:
        out.append("</policy>")
    return "\n".join(out)


class RagPipeline:
    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        corpus: Corpus,
        config: RetrievalConfig,
        name: str = "rag",
        effort: Effort = "medium",
    ) -> None:
        self.client = client
        self.model = model
        self.corpus = corpus
        self.config = config
        self.name = name
        self.effort: Effort = effort
        self.index = shared_index(config.contextual, config.mode != "bm25")
        self.reranker = shared_reranker() if config.rerank else None

    def retrieve(self, question: str) -> list[Chunk]:
        hits = self.index.search(
            question, self.config.mode, self.config.k, self.config.candidates, self.reranker
        )
        return [h.chunk for h in hits]

    def answer(self, question: str) -> PipelineResult:
        chunks = self.retrieve(question)
        context_ids = list(dict.fromkeys(i for c in chunks for i in c.cite_ids))
        system = build_system("retrieved", render_passages(chunks, self.corpus))
        return generate(self.client, self.model, system, question, self.effort, context_ids)


VARIANTS: dict[str, RetrievalConfig] = {
    # Production: contextual chunks, hybrid dense + BM25 fused with RRF. No reranker: the
    # small cross-encoder lowered recall and added ~7s per query on CPU (see ablation).
    "rag": RetrievalConfig(rerank=False),
    "rag_k12": RetrievalConfig(k=12),
    "rag_rerank": RetrievalConfig(rerank=True),
    "rag_dense": RetrievalConfig(mode="dense", rerank=False),
    "rag_bm25": RetrievalConfig(mode="bm25", rerank=False),
    "rag_plain": RetrievalConfig(contextual=False, rerank=False),  # no context notes
}


def make_rag(name: str) -> Callable[[anthropic.Anthropic, str, Corpus], RagPipeline]:
    def factory(client: anthropic.Anthropic, model: str, corpus: Corpus) -> RagPipeline:
        return RagPipeline(client, model, corpus, VARIANTS[name], name=name)

    return factory
