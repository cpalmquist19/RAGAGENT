"""Benchmark pipeline: the whole corpus in the prompt, no retrieval.

Not the production design. It answers "what would the model do with perfect recall?",
which is the bar the RAG pipeline is measured against.
"""

import anthropic

from licn_rag.answer.prompt import build_system
from licn_rag.corpus import Corpus
from licn_rag.pipelines.base import Effort, PipelineResult, generate


class LongContextPipeline:
    name = "long_context"

    def __init__(
        self, client: anthropic.Anthropic, model: str, corpus: Corpus, effort: Effort = "medium"
    ) -> None:
        self.client = client
        self.model = model
        self.effort: Effort = effort
        rendered = corpus.render()
        self.context_words = len(rendered.split())
        self.system = build_system("full_corpus", rendered)

    def answer(self, question: str) -> PipelineResult:
        result = generate(self.client, self.model, self.system, question, self.effort, None)
        result.context_words = self.context_words
        return result
