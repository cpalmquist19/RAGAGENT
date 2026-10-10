"""All answering pipelines by name: the long-context benchmark and the RAG variants."""

from collections.abc import Callable

import anthropic

from licn_rag.corpus import Corpus
from licn_rag.pipelines.base import Pipeline
from licn_rag.pipelines.long_context import LongContextPipeline
from licn_rag.pipelines.rag import VARIANTS, make_rag

Factory = Callable[[anthropic.Anthropic, str, Corpus], Pipeline]

PIPELINES: dict[str, Factory] = {
    "long_context": LongContextPipeline,
    **{name: make_rag(name) for name in VARIANTS},
}
