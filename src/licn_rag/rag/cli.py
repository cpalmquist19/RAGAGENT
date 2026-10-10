"""Build or query the retrieval index.

uv run licn-index build                 # contextual index (writes context notes if needed)
uv run licn-index build --plain         # same chunks without context notes (ablation)
uv run licn-index search "CEUs to renew a K-5 teacher CPL" --mode hybrid --rerank
"""

import argparse
import sys
import time
from pathlib import Path

from licn_rag.corpus import Corpus
from licn_rag.llm import make_client
from licn_rag.rag.chunking import chunk_corpus
from licn_rag.rag.contextualize import contextualize
from licn_rag.rag.index import INDEX_DIR, Embedder, Index, Reranker, build


def index_path(plain: bool) -> Path:
    return INDEX_DIR / ("licn-plain.sqlite" if plain else "licn.sqlite")


def cmd_build(args: argparse.Namespace) -> None:
    corpus = Corpus.load()
    chunks = chunk_corpus(corpus.policies)
    if not args.plain:
        cost = contextualize(chunks, corpus, make_client())
        print(f"context notes ready (${cost:.3f} spent on new notes)")
    start = time.perf_counter()
    path = index_path(args.plain)
    build(chunks, Embedder(), path)
    print(f"indexed {len(chunks)} chunks → {path} in {time.perf_counter() - start:.0f}s")


def cmd_search(args: argparse.Namespace) -> None:
    index = Index(index_path(args.plain), Embedder() if args.mode != "bm25" else None)
    reranker = Reranker() if args.rerank else None
    start = time.perf_counter()
    hits = index.search(args.query, args.mode, args.k, reranker=reranker)
    print(f"{len(hits)} hits in {time.perf_counter() - start:.2f}s\n")
    for i, h in enumerate(hits, 1):
        ranks = f"dense #{h.dense_rank or '-'} · bm25 #{h.bm25_rank or '-'}"
        print(f"{i}. {h.chunk.chunk_id}  score {h.score:.3f}  ({ranks})\n   {h.chunk.breadcrumb}")
        print(f"   {h.chunk.cite_ids[0]} … {len(h.chunk.cite_ids)} passages")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--plain", action="store_true", help="skip contextual notes")
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    s.add_argument("--rerank", action="store_true")
    s.add_argument("--plain", action="store_true")
    s.add_argument("-k", type=int, default=8)
    args = ap.parse_args(argv)
    {"build": cmd_build, "search": cmd_search}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
