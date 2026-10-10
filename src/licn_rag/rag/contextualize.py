"""Contextual retrieval: a short LLM-written note situating each chunk in its policy.

Before indexing, each chunk gets 1-2 sentences saying where it sits and what it is about
(Anthropic's "contextual retrieval"). The note is embedded and keyword-indexed with the
chunk, so a fragment like "(ii) Three CEUs related to literacy" is findable by searches
about teacher license renewal.

Notes are cached in corpus/chunk_contexts.json keyed by a hash of the chunk text, so
rebuilding the index only calls the API for chunks that changed.
"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

from licn_rag.corpus import Corpus, render_policy
from licn_rag.llm import first_text, usage_cost
from licn_rag.rag.chunking import Chunk

CACHE = Path("corpus/chunk_contexts.json")
MODEL = "claude-haiku-5-5"

PROMPT = """\
Here is a chunk from the policy above:
<chunk>
{chunk}
</chunk>

Write one or two sentences that situate this chunk within the policy, to improve search \
retrieval of the chunk. Say which license, process, or requirement it concerns and who it \
applies to. Answer with only the sentences."""


def chunk_hash(chunk: Chunk) -> str:
    return hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:16]


def load_cache() -> dict[str, str]:
    return json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}


def contextualize(chunks: list[Chunk], corpus: Corpus, client: anthropic.Anthropic) -> float:
    """Fill chunk.context from the cache, calling the API for missing ones. Returns cost."""
    cache = load_cache()
    missing = [c for c in chunks if chunk_hash(c) not in cache]
    cost = 0.0

    def note(chunk: Chunk) -> tuple[str, str, float]:
        policy = corpus.by_code[chunk.policy_code]
        message = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=[
                {
                    "type": "text",
                    "text": f"<policy>\n{render_policy(policy)}\n</policy>",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": PROMPT.format(chunk=chunk.text)}],
            output_config={"effort": "low"},
        )
        return chunk_hash(chunk), first_text(message).strip(), usage_cost(MODEL, message.usage)

    if missing:
        print(f"writing context notes for {len(missing)} chunks with {MODEL}…")
        with ThreadPoolExecutor(max_workers=8) as pool:
            for h, text, c in pool.map(note, missing):
                cache[h] = text
                cost += c
        CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")

    for chunk in chunks:
        chunk.context = cache[chunk_hash(chunk)]
    return cost
