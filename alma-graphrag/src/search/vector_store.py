"""
pgvector-backed search store — keyword (Postgres full-text) + semantic (dense
embeddings), plus a hybrid fusion of the two.

One table, `hotel_search`, holds per-hotel text with:
  - tsv        : a tsvector for keyword / full-text ranking (GIN index)
  - embedding  : a dense vector for semantic search (HNSW cosine index)

All heavy imports (psycopg2, pgvector) are lazy so importing this module never
fails when the optional deps or the database are absent; callers should guard
with is_available().
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.config import (
    PG_DB,
    PG_HOST,
    PG_PASSWORD,
    PG_PORT,
    PG_USER,
    ST_EMBEDDING_DIM,
)

logger = logging.getLogger("alma.vector")

TABLE = "hotel_search"


def _connect(register_vector_type: bool = True):
    import psycopg2
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASSWORD
    )
    if register_vector_type:
        from pgvector.psycopg2 import register_vector
        register_vector(conn)
    return conn


# Cached read-only connection for the search paths — the evaluation harness
# issues hundreds of small queries and a connect per call dominates latency.
_search_conn = None


def _query(sql: str, params: tuple) -> List[tuple]:
    """Run a read-only query on the cached connection, reconnecting once if the
    connection has gone away."""
    global _search_conn
    import psycopg2
    for attempt in (1, 2):
        try:
            if _search_conn is None or _search_conn.closed:
                _search_conn = _connect(register_vector_type=False)
            with _search_conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
            _search_conn.rollback()  # end the implicit read transaction
            return rows
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            _search_conn = None
            if attempt == 2:
                raise
    return []  # unreachable


# ---------------------------------------------------------------------------
# Schema + indexing
# ---------------------------------------------------------------------------

def init_schema() -> None:
    conn = _connect(register_vector_type=False)
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        from pgvector.psycopg2 import register_vector
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TABLE} (
                    id        text PRIMARY KEY,
                    city      text,
                    name      text,
                    doc       text,
                    tsv       tsvector,
                    embedding vector({ST_EMBEDDING_DIM})
                )
                """
            )
            cur.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_tsv_idx ON {TABLE} USING GIN (tsv)")
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {TABLE}_emb_idx ON {TABLE} "
                f"USING hnsw (embedding vector_cosine_ops)"
            )
        conn.commit()
    finally:
        conn.close()


def index_hotels(rows: List[Dict[str, Any]]) -> int:
    """Upsert hotel rows. Each row: {id, city, name, doc, embedding}."""
    conn = _connect(register_vector_type=False)
    try:
        with conn.cursor() as cur:
            for r in rows:
                cur.execute(
                    f"""
                    INSERT INTO {TABLE} (id, city, name, doc, tsv, embedding)
                    VALUES (%s, %s, %s, %s, to_tsvector('english', %s), %s::vector)
                    ON CONFLICT (id) DO UPDATE SET
                        city = EXCLUDED.city, name = EXCLUDED.name,
                        doc = EXCLUDED.doc, tsv = EXCLUDED.tsv,
                        embedding = EXCLUDED.embedding
                    """,
                    (r["id"], r["city"], r["name"], r["doc"], r["doc"], _vec_literal(r["embedding"])),
                )
        conn.commit()
    finally:
        conn.close()
    return len(rows)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _vec_literal(embedding: List[float]) -> str:
    """Render an embedding as a pgvector string literal, e.g. '[0.1,0.2]'.

    Passing this with an explicit `::vector` cast avoids relying on
    driver-side type adaptation (psycopg2 would otherwise send a plain list
    as `numeric[]`, which has no `<=>` operator against `vector`).
    """
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


def semantic_search(city: str, query_embedding: List[float], k: int) -> List[str]:
    """Dense cosine nearest-neighbour search (pgvector `<=>`)."""
    rows = _query(
        f"""
        SELECT id FROM {TABLE}
        WHERE lower(city) = lower(%s)
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (city, _vec_literal(query_embedding), k),
    )
    return [row[0] for row in rows]


def keyword_search(city: str, query_text: str, k: int) -> List[str]:
    """Lexical full-text search ranked by ts_rank.

    OR semantics: `plainto_tsquery` normalises the query and drops stopwords,
    producing an AND query (`a & b & c`); we swap the `&` for `|` so a hotel
    matching *any* query term is retrieved, then rank by ts_rank (documents
    matching more/better terms rank higher). Hard-AND would only match hotels
    containing every word — a strawman for NL questions. Note ts_rank is a
    term-frequency/proximity ranking, NOT BM25 (no IDF or BM25 length
    normalisation) — describe it as "Postgres full-text", not BM25.
    """
    rows = _query(
        f"""
        WITH q AS (
            SELECT replace(plainto_tsquery('english', %s)::text, '&', '|')::tsquery AS tq
        )
        SELECT id FROM {TABLE}, q
        WHERE lower(city) = lower(%s)
          AND q.tq <> ''::tsquery
          AND tsv @@ q.tq
        ORDER BY ts_rank(tsv, q.tq) DESC
        LIMIT %s
        """,
        (query_text, city, k),
    )
    return [row[0] for row in rows]


def _rrf(*rankings: List[str], K: int = 60) -> List[str]:
    """Reciprocal Rank Fusion — the standard parameter-light list combiner."""
    score: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            score[doc] = score.get(doc, 0.0) + 1.0 / (K + rank)
    return [doc for doc, _ in sorted(score.items(), key=lambda kv: -kv[1])]


def hybrid_search(city: str, query_text: str, query_embedding: List[float], k: int) -> List[str]:
    """Fuse keyword + semantic rankings with RRF (pull 2k from each first)."""
    sem = semantic_search(city, query_embedding, k * 2)
    kw = keyword_search(city, query_text, k * 2)
    return _rrf(sem, kw)[:k]


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def count(city: Optional[str] = None) -> int:
    if city:
        rows = _query(f"SELECT count(*) FROM {TABLE} WHERE lower(city) = lower(%s)", (city,))
    else:
        rows = _query(f"SELECT count(*) FROM {TABLE}", ())
    return int(rows[0][0])


def is_available(city: Optional[str] = None) -> bool:
    """True when Postgres is reachable and the index has rows for `city`."""
    try:
        return count(city) > 0
    except Exception as exc:  # missing deps, DB down, table absent
        logger.info("pgvector search unavailable: %s", exc)
        return False
