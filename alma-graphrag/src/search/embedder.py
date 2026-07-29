"""
Local dense embedder for semantic search.

Wraps a sentence-transformers model (default all-MiniLM-L6-v2, 384-dim). Runs
fully offline on CPU and is loaded lazily + cached, so importing this module is
cheap and does not pull torch until the first embed() call.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from src.config import ST_EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _model():
    # Imported here so module import stays light (torch loads on first use).
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(ST_EMBEDDING_MODEL)


def embed(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts into unit-normalised dense vectors."""
    model = _model()
    vecs = model.encode(
        list(texts),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vecs]


@lru_cache(maxsize=1024)
def _embed_one_cached(text: str) -> tuple:
    return tuple(embed([text])[0])


def embed_one(text: str) -> List[float]:
    """Embed a single text. Cached — the evaluation harness embeds the same
    question once per system (Semantic, Hybrid), and re-runs repeat queries."""
    return list(_embed_one_cached(text))
