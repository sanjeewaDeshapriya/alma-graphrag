"""
Retrieval baselines for comparative evaluation (proposal Phase 5).

Each implements ``retrieve(question, city, k) -> List[hotel_id]`` over the SAME
city candidate pool, so differences reflect the retrieval method only:

  1. FilterBaseline   — traditional filter-and-sort: apply structured filters
                        (price/rating/star) parsed from the query, rank by rating.
                        Sees no spatial/accessibility/graph structure.
  2. KeywordBaseline  — Postgres full-text keyword search (tsvector / ts_rank)
                        over the hotel text. Lexical; exact-word matching.
  3. SemanticBaseline — dense semantic search: sentence-transformers embeddings
                        in pgvector, cosine nearest-neighbour. Captures meaning,
                        not just keywords.
  4. HybridBaseline   — Reciprocal Rank Fusion of keyword + semantic.
  5. WeightedGraph    — the proposed system: feasibility-first weighted multi-hop
                        GraphRAG retriever (uses spatial + live-traffic
                        accessibility + facility + economic + disruption edges).

The keyword/semantic/hybrid systems require the pgvector index (build it with
scripts/build_vector_index.py); when it is absent they are skipped so the rest
of the harness still runs.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.crag.query_parser import parse_query
from src.graph.query import _get_driver
from src.graph.retriever import WeightedRetriever
from src.search import vector_store as vs
from src.search.embedder import embed_one

logger = logging.getLogger("alma.eval.baselines")

# Shared candidate fetch — full attribute set incl. description for the vector baseline.
_FETCH_QUERY = """
MATCH (h:Hotel)-[loc:LOCATED_IN]->(c:City)
WHERE toLower(c.name) = toLower($city)
OPTIONAL MATCH (h)-[:HAS_AMENITY]->(a:Amenity)
OPTIONAL MATCH (h)-[:NEAR_ATTRACTION]->(at:AttractionType)
WITH h, loc,
     collect(DISTINCT a.name)  AS amenities,
     collect(DISTINCT at.name) AS attractions
RETURN h.id AS id, h.name AS name, h.description AS description,
       h.rating AS rating, h.star_rating AS star,
       h.price_per_night_lkr AS price,
       loc.travel_time_min AS travel_time_min,
       loc.travel_time_traffic_min AS travel_time_traffic_min,
       amenities, attractions
"""


def fetch_city_hotels(city: str) -> List[Dict[str, Any]]:
    driver = _get_driver()
    with driver.session() as session:
        return session.run(_FETCH_QUERY, {"city": city}).data()


# ---------------------------------------------------------------------------
# Baseline 1 — filter-and-sort
# ---------------------------------------------------------------------------

class FilterBaseline:
    name = "Filter"

    def retrieve(self, question: str, city: str, k: int) -> List[str]:
        intent = parse_query(question, default_city=city)
        hotels = fetch_city_hotels(city)
        out = []
        for h in hotels:
            price, rating, star = h.get("price"), h.get("rating"), h.get("star")
            if intent.max_price_lkr is not None and (price is None or float(price) > intent.max_price_lkr):
                continue
            if intent.min_price_lkr is not None and (price is None or float(price) < intent.min_price_lkr):
                continue
            if intent.min_rating is not None and (rating is None or float(rating) < intent.min_rating):
                continue
            if intent.min_star is not None and (star is None or float(star) < intent.min_star):
                continue
            out.append(h)
        # Classic behaviour: rank survivors by rating (then cheaper first).
        out.sort(key=lambda h: (-(h.get("rating") or 0), h.get("price") or float("inf")))
        return [str(h["id"]) for h in out[:k]]


# ---------------------------------------------------------------------------
# Baseline 2 — keyword search (Postgres full-text)
# ---------------------------------------------------------------------------

class KeywordBaseline:
    name = "Keyword"

    def retrieve(self, question: str, city: str, k: int) -> List[str]:
        return vs.keyword_search(city, question, k)


# ---------------------------------------------------------------------------
# Baseline 3 — semantic search (dense embeddings in pgvector)
# ---------------------------------------------------------------------------

class SemanticBaseline:
    name = "SemanticVec"

    def retrieve(self, question: str, city: str, k: int) -> List[str]:
        return vs.semantic_search(city, embed_one(question), k)


# ---------------------------------------------------------------------------
# Baseline 4 — hybrid (keyword + semantic via RRF)
# ---------------------------------------------------------------------------

class HybridBaseline:
    name = "Hybrid"

    def retrieve(self, question: str, city: str, k: int) -> List[str]:
        return vs.hybrid_search(city, question, embed_one(question), k)


# ---------------------------------------------------------------------------
# Baseline 3 — proposed weighted GraphRAG
# ---------------------------------------------------------------------------

class WeightedGraphBaseline:
    name = "WeightedGraphRAG"

    def __init__(self) -> None:
        self._retriever = WeightedRetriever()

    def retrieve(self, question: str, city: str, k: int) -> List[str]:
        intent = parse_query(question, default_city=city)
        if not intent.city:
            intent.city = city
        result = self._retriever.retrieve(intent, limit=k)
        return [h.id for h in result.hotels]


def all_baselines() -> List[Any]:
    """Filter + GraphRAG are always available; the pgvector-backed keyword /
    semantic / hybrid systems are included only when the index is populated."""
    baselines: List[Any] = [FilterBaseline()]
    if vs.is_available():
        baselines += [KeywordBaseline(), SemanticBaseline(), HybridBaseline()]
    else:
        logger.warning(
            "pgvector index unavailable — skipping Keyword/SemanticVec/Hybrid "
            "baselines. Start the pgvector container and run "
            "scripts/build_vector_index.py to enable them."
        )
    baselines.append(WeightedGraphBaseline())
    return baselines
