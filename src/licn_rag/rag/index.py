"""Build and query the retrieval index: one SQLite file holding chunks, vectors, and BM25.

- Dense: Qwen3-Embedding vectors in a sqlite-vec `vec0` table (cosine distance).
- Sparse: an FTS5 table ranked with BM25.
- Hybrid: reciprocal rank fusion of the two lists, optionally reranked by a local
  cross-encoder.

The index is a derived artifact (data/index/, git-ignored); rebuild it with
`uv run licn-index build`.
"""

import json
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sqlite_vec

from licn_rag.rag.chunking import Chunk

INDEX_DIR = Path("data/index")
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANK_MODEL = "mixedbread-ai/mxbai-rerank-xsmall-v1"
RRF_K = 60

# Words that carry no retrieval signal for BM25 (FTS5 has no stopword list of its own).
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "so",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "would",
        "you",
        "your",
        "there",
        "any",
        "all",
        "just",
        "now",
        "still",
        "get",
        "got",
        "need",
        "needs",
    }
)


def connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path, check_same_thread=False)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


class Embedder:
    """Qwen3-Embedding with its query instruction; loaded once, lazily."""

    _lock = threading.Lock()

    def __init__(self, model: str = EMBED_MODEL) -> None:
        import torch
        from sentence_transformers import SentenceTransformer  # slow import; only when needed

        # float32, not the model's default bfloat16: CPUs without native bf16 support
        # (most laptops, Cloud Run) run bf16 matmuls ~6x slower.
        self.model = SentenceTransformer(model, model_kwargs={"dtype": torch.float32})
        self.dim = self.model.get_embedding_dimension()

    def documents(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            return self.model.encode(texts, normalize_embeddings=True, batch_size=8)

    def query(self, text: str) -> np.ndarray:
        with self._lock:
            return self.model.encode([text], prompt_name="query", normalize_embeddings=True)[0]


class Reranker:
    _lock = threading.Lock()

    def __init__(self, model: str = RERANK_MODEL) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model, model_kwargs={"dtype": torch.float32})

    def scores(self, query: str, texts: list[str]) -> list[float]:
        with self._lock:
            return [float(s) for s in self.model.predict([(query, t) for t in texts])]


def build(chunks: list[Chunk], embedder: Embedder, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    db = connect(path)
    db.executescript(
        f"""
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE, policy_code TEXT, policy_title TEXT,
            ord INTEGER, breadcrumb TEXT, context TEXT, text TEXT, cite_ids TEXT);
        CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[{embedder.dim}]
            distance_metric=cosine);
        CREATE VIRTUAL TABLE chunks_fts USING fts5(body, tokenize='porter unicode61');
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        """
    )
    vectors = embedder.documents([c.index_text() for c in chunks])
    for i, (c, v) in enumerate(zip(chunks, vectors, strict=True), start=1):
        db.execute(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?)",
            (
                i,
                c.chunk_id,
                c.policy_code,
                c.policy_title,
                c.order,
                c.breadcrumb,
                c.context,
                c.text,
                json.dumps(c.cite_ids),
            ),
        )
        db.execute(
            "INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)",
            (i, v.astype(np.float32).tobytes()),
        )
        db.execute("INSERT INTO chunks_fts(rowid, body) VALUES (?, ?)", (i, c.index_text()))
    db.execute("INSERT INTO meta VALUES ('embed_model', ?)", (EMBED_MODEL,))
    db.execute("INSERT INTO meta VALUES ('contextual', ?)", (str(any(c.context for c in chunks)),))
    db.commit()
    db.close()


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None


def fts_query(text: str) -> str:
    """Free text -> an FTS5 OR-query of quoted terms (no syntax errors from user input)."""
    terms = [
        t
        for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-]*", text.lower())
        if t not in STOPWORDS and len(t) > 1
    ]
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(terms)) or '""'


class Index:
    def __init__(self, path: Path, embedder: Embedder | None = None) -> None:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `uv run licn-index build`")
        self.db = connect(path)
        self.embedder = embedder
        self.chunks: dict[int, Chunk] = {}
        for row in self.db.execute(
            "SELECT id, chunk_id, policy_code, policy_title, ord, breadcrumb, context, text, "
            "cite_ids FROM chunks"
        ):
            self.chunks[row[0]] = Chunk(
                chunk_id=row[1],
                policy_code=row[2],
                policy_title=row[3],
                order=row[4],
                breadcrumb=row[5],
                context=row[6],
                text=row[7],
                cite_ids=json.loads(row[8]),
            )

    def dense(self, query: str, k: int) -> list[int]:
        assert self.embedder is not None, "dense search needs an embedder"
        vec = self.embedder.query(query).astype(np.float32).tobytes()
        rows = self.db.execute(
            "SELECT rowid FROM chunks_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (vec, k),
        )
        return [r[0] for r in rows]

    def bm25(self, query: str, k: int) -> list[int]:
        rows = self.db.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) "
            "LIMIT ?",
            (fts_query(query), k),
        )
        return [r[0] for r in rows]

    def search(
        self, query: str, mode: str, k: int, candidates: int = 30, reranker: Reranker | None = None
    ) -> list[Hit]:
        dense = self.dense(query, candidates) if mode in ("dense", "hybrid") else []
        sparse = self.bm25(query, candidates) if mode in ("bm25", "hybrid") else []
        fused: dict[int, Hit] = {}
        for rank, rid in enumerate(dense, start=1):
            hit = fused.setdefault(rid, Hit(self.chunks[rid], 0.0))
            hit.score += 1 / (RRF_K + rank)
            hit.dense_rank = rank
        for rank, rid in enumerate(sparse, start=1):
            hit = fused.setdefault(rid, Hit(self.chunks[rid], 0.0))
            hit.score += 1 / (RRF_K + rank)
            hit.bm25_rank = rank
        hits = sorted(fused.values(), key=lambda h: h.score, reverse=True)[:candidates]
        if reranker and hits:
            scores = reranker.scores(query, [h.chunk.index_text() for h in hits])
            for h, s in zip(hits, scores, strict=True):
                h.score = s
            hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]
