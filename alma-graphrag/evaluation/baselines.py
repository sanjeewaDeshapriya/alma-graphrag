"""
Retrieval baselines for comparative evaluation (proposal Phase 5).

All three implement ``retrieve(question, city, k) -> List[hotel_id]`` over the
SAME city candidate pool, so differences reflect the retrieval method only:

  1. FilterBaseline   — traditional filter-and-sort: apply structured filters
                        (price/rating/star) parsed from the query, rank by rating.
                        Sees no spatial/accessibility/graph structure.
  2. VectorBaseline   — vector RAG: TF-IDF cosine similarity between the query
                        and hotel text (name/description/amenities). Lexical-dense
                        retrieval; cannot see numeric edge attributes like traffic
                        travel time. (Swap in dense embeddings when populated.)
  3. WeightedGraph    — the proposed system: feasibility-first weighted multi-hop
                        GraphRAG retriever (uses spatial + live-traffic
                        accessibility + facility + economic + disruption edges).
"""
from __future__ import annotations

from typing import Any, Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.crag.query_parser import parse_query
from src.graph.query import _get_driver
from src.graph.retriever import WeightedRetriever

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
# Baseline 2 — vector RAG (TF-IDF)
# ---------------------------------------------------------------------------

class VectorBaseline:
    name = "VectorRAG"

    def _doc(self, h: Dict[str, Any]) -> str:
        parts = [
            h.get("name") or "",
            h.get("description") or "",
            " ".join(h.get("amenities") or []),
            " ".join(h.get("attractions") or []),
        ]
        return " ".join(parts)

    def retrieve(self, question: str, city: str, k: int) -> List[str]:
        hotels = fetch_city_hotels(city)
        if not hotels:
            return []
        docs = [self._doc(h) for h in hotels]
        vec = TfidfVectorizer(stop_words="english")
        try:
            matrix = vec.fit_transform(docs + [question])
        except ValueError:
            return [str(h["id"]) for h in hotels[:k]]
        sims = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        order = sims.argsort()[::-1]
        return [str(hotels[i]["id"]) for i in order[:k]]


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
    return [FilterBaseline(), VectorBaseline(), WeightedGraphBaseline()]
