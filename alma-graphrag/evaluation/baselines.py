"""
Retrieval baselines for comparative evaluation.

Each implements ``retrieve(question, city, k, intent=None) -> List[hotel_id]``
over the SAME city candidate pool, so differences reflect the retrieval method
only.

Shared parsed intent
--------------------
`intent` is threaded through every baseline and, when the harness supplies it,
each system receives the IDENTICAL parsed representation of the query.

This matters more than it looks. `parse_query` runs a regex pass and,
when a key is configured, an LLM slot-fill pass whose output is not
deterministic. Previously each baseline called `parse_query` itself, so
Filter and WeightedGraphRAG could be answering *different readings of the
same question* — one seeing `max_price=25000`, the other seeing nothing.
Any measured gap then confounds retrieval quality with parser luck, and the
LLM pass was billed once per system per query. Parsing once and sharing the
result removes both problems; `intent=None` keeps the old self-parsing
behaviour so the baselines still work standalone.

The line-up
-----------
Classical (pre-neural):
  1. Random       — shuffle. The floor: any system not beating it is broken.
  2. Popularity   — rank by rating, ignoring the query entirely. A
                    surprisingly strong floor in recommendation, and the one
                    that exposes benchmarks where the query does not matter.
  3. Filter       — filter-and-sort on parsed constraints, rank by rating.
                    Stands in for the real-world OTA facet search.
  4. Keyword      — Postgres full-text (tsvector / ts_rank). Lexical.

Neural:
  5. SemanticVec  — dense sentence-transformer embeddings in pgvector, cosine.
  6. Hybrid       — Reciprocal Rank Fusion of keyword + semantic.
  7. CrossEncoder — a cross-encoder re-ranker over the hybrid candidates. Runs
                    locally; this is the standard modern two-stage IR pipeline
                    and the strongest non-LLM baseline available offline.
  8. LLMReranker  — hand the pool to an instruction-tuned model and ask it to
                    rank. The baseline a 2026 reviewer expects; without it the
                    comparison is against pre-neural methods only.

Learned:
  9. LTR          — supervised learning-to-rank (gradient-boosted trees) on the
                    SAME features the graph retriever scores on, trained under
                    k-fold CV over queries. This separates "the weighting is
                    learned" from "the weighting is learned by our method": if
                    plain LTR matches the proposed system, the contribution is
                    the feature set, not the scorer.

Proposed:
 10. WeightedGraphRAG — feasibility-first weighted multi-hop GraphRAG.

Availability
------------
Everything degrades gracefully. pgvector-backed systems are skipped when the
index is absent, the cross-encoder when sentence-transformers cannot load a
model, the LLM re-ranker when no API key is configured. `all_baselines()`
reports what it skipped rather than failing, so a partial environment still
produces a valid (smaller) table.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.crag.query_parser import QueryIntent, parse_query
from src.graph.query import _get_driver
from src.graph.retriever import WeightedRetriever
from src.search import vector_store as vs
from src.search.embedder import embed_one

logger = logging.getLogger("alma.eval.baselines")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "evaluation" / ".cache"

# Shared candidate fetch — full attribute set incl. description for the vector
# baseline, plus the disruption evidence the graded gold needs for its
# `max_added_delay_min` predicate.
_FETCH_QUERY = """
MATCH (h:Hotel)-[loc:LOCATED_IN]->(c:City)
WHERE toLower(c.name) = toLower($city)
OPTIONAL MATCH (h)-[:HAS_AMENITY]->(a:Amenity)
OPTIONAL MATCH (h)-[:NEAR_ATTRACTION]->(at:AttractionType)
OPTIONAL MATCH (h)-[:HAS_SIGNAL]->(ts:TrafficSignal)
OPTIONAL MATCH (h)-[ae:AFFECTED_BY]->(:Event)
WITH h, loc,
     collect(DISTINCT a.name)  AS amenities,
     collect(DISTINCT at.name) AS attractions,
     max(coalesce(ts.eta_change_min, 0.0)) AS max_eta_change_min,
     max(coalesce(ae.impact_score, 0.0))   AS event_impact
RETURN h.id AS id, h.name AS name, h.description AS description,
       h.rating AS rating, h.star_rating AS star,
       h.price_per_night_lkr AS price,
       h.lat AS lat, h.lng AS lng,
       loc.distance_km AS distance_km,
       loc.travel_time_min AS travel_time_min,
       loc.travel_time_traffic_min AS travel_time_traffic_min,
       amenities, attractions, max_eta_change_min, event_impact
"""


def fetch_city_hotels(city: str) -> List[Dict[str, Any]]:
    driver = _get_driver()
    with driver.session() as session:
        return session.run(_FETCH_QUERY, {"city": city}).data()


def _resolve_intent(question: str, city: str,
                    intent: Optional[QueryIntent]) -> QueryIntent:
    """Use the shared intent when the harness passed one; else parse locally."""
    if intent is not None:
        return intent
    parsed = parse_query(question, default_city=city)
    if not parsed.city:
        parsed.city = city
    return parsed


# ---------------------------------------------------------------------------
# Floors
# ---------------------------------------------------------------------------

class RandomBaseline:
    """Shuffled pool. Seeded per query so a run is reproducible."""

    name = "Random"

    def retrieve(self, question: str, city: str, k: int,
                 intent: Optional[QueryIntent] = None) -> List[str]:
        hotels = [str(h["id"]) for h in fetch_city_hotels(city)]
        seed = int(hashlib.md5(question.encode("utf-8")).hexdigest()[:8], 16)
        random.Random(seed).shuffle(hotels)
        return hotels[:k]


class PopularityBaseline:
    """Rank by rating, ignoring the query.

    Reported because it is the sharpest diagnostic in the table: if a system
    barely beats it, the benchmark's queries are not doing any work.
    """

    name = "Popularity"

    def retrieve(self, question: str, city: str, k: int,
                 intent: Optional[QueryIntent] = None) -> List[str]:
        hotels = fetch_city_hotels(city)
        hotels.sort(key=lambda h: (-(h.get("rating") or 0),
                                   h.get("price") or float("inf")))
        return [str(h["id"]) for h in hotels[:k]]


# ---------------------------------------------------------------------------
# Classical
# ---------------------------------------------------------------------------

class FilterBaseline:
    """Traditional filter-and-sort — the real-world OTA facet search."""

    name = "Filter"

    def retrieve(self, question: str, city: str, k: int,
                 intent: Optional[QueryIntent] = None) -> List[str]:
        intent = _resolve_intent(question, city, intent)
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


class KeywordBaseline:
    name = "Keyword"

    def retrieve(self, question: str, city: str, k: int,
                 intent: Optional[QueryIntent] = None) -> List[str]:
        return vs.keyword_search(city, question, k)


# ---------------------------------------------------------------------------
# Neural
# ---------------------------------------------------------------------------

class SemanticBaseline:
    name = "SemanticVec"

    def retrieve(self, question: str, city: str, k: int,
                 intent: Optional[QueryIntent] = None) -> List[str]:
        return vs.semantic_search(city, embed_one(question), k)


class HybridBaseline:
    name = "Hybrid"

    def retrieve(self, question: str, city: str, k: int,
                 intent: Optional[QueryIntent] = None) -> List[str]:
        return vs.hybrid_search(city, question, embed_one(question), k)


class CrossEncoderBaseline:
    """Two-stage retrieve-then-rerank: hybrid first stage, cross-encoder second.

    The standard modern IR pipeline. A bi-encoder embeds query and document
    separately (fast, approximate); a cross-encoder reads the pair jointly and
    scores it (slow, accurate), so it is run only over a shallow candidate list.

    `ms-marco-MiniLM-L-6-v2` is trained on web-search relevance, not on hotel
    attributes, so it judges textual relevance of the verbalised hotel document.
    That is exactly the comparison worth having: how far does strong general
    text matching get you on a task whose constraints are numeric?
    """

    name = "CrossEncoder"
    MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    FIRST_STAGE = 50

    def __init__(self) -> None:
        self._model = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            from sentence_transformers import CrossEncoder  # noqa: F401
        except ImportError:
            return False
        return vs.is_available()

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.MODEL, max_length=384)
        return self._model

    def retrieve(self, question: str, city: str, k: int,
                 intent: Optional[QueryIntent] = None) -> List[str]:
        first = vs.hybrid_search(city, question, embed_one(question), self.FIRST_STAGE)
        if not first:
            return []
        by_id = {str(h["id"]): h for h in fetch_city_hotels(city)}
        pairs, ids = [], []
        for hid in first:
            h = by_id.get(hid)
            if h:
                pairs.append((question, _verbalise(h)))
                ids.append(hid)
        if not pairs:
            return []
        scores = self._load().predict(pairs)
        ranked = sorted(zip(ids, scores), key=lambda t: -t[1])
        return [hid for hid, _ in ranked[:k]]


class LLMRerankerBaseline:
    """Give an instruction-tuned model the pool and ask it to rank.

    Results are cached on disk keyed by (model, city, question, pool digest), so
    a re-run costs nothing and the numbers are reproducible. The pool digest is
    part of the key: if the graph changes, the cache correctly misses rather
    than silently serving a ranking of hotels that no longer exist.

    The prompt is deliberately plain. Prompt engineering it into a win would
    make the baseline a strawman in the opposite direction — the point is to
    measure what a competent, obvious use of an LLM achieves.
    """

    name = "LLMReranker"
    POOL_LIMIT = 60

    def __init__(self, model: Optional[str] = None) -> None:
        from src.config import LLM_MODEL
        self.model = model or LLM_MODEL
        self._client = None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_path = CACHE_DIR / "llm_reranker.json"
        self._cache: Dict[str, List[str]] = {}
        if self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("LLM reranker cache is corrupt; starting fresh")

    @staticmethod
    def is_available() -> bool:
        from src.config import LLM_API_KEY
        return bool(LLM_API_KEY)

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            from src.config import LLM_API_KEY, LLM_BASE_URL
            self._client = OpenAI(api_key=LLM_API_KEY,
                                  base_url=LLM_BASE_URL or None)
        return self._client

    def _key(self, question: str, city: str, hotels: List[Dict[str, Any]]) -> str:
        digest = hashlib.md5(
            "|".join(sorted(str(h["id"]) for h in hotels)).encode("utf-8")
        ).hexdigest()[:10]
        return f"{self.model}::{city}::{question}::{digest}"

    def _flush(self) -> None:
        self.cache_path.write_text(json.dumps(self._cache, indent=1), encoding="utf-8")

    def retrieve(self, question: str, city: str, k: int,
                 intent: Optional[QueryIntent] = None) -> List[str]:
        hotels = fetch_city_hotels(city)[: self.POOL_LIMIT]
        if not hotels:
            return []
        key = self._key(question, city, hotels)
        if key in self._cache:
            return self._cache[key][:k]

        listing = "\n".join(
            f"[{i}] {_verbalise(h)}" for i, h in enumerate(hotels)
        )
        prompt = (
            f"A traveller asks: {question}\n\n"
            f"Here are the hotels available in {city}:\n{listing}\n\n"
            f"Rank the {k} hotels that best answer the traveller's request, best "
            f"first. Consider every stated requirement. Reply with ONLY a JSON "
            f'array of the numeric indices, e.g. [3,0,7]. No other text.'
        )
        try:
            resp = self._get_client().chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            text = (resp.choices[0].message.content or "").strip()
            start, end = text.find("["), text.rfind("]")
            idxs = json.loads(text[start:end + 1]) if start >= 0 < end else []
            ranked = [str(hotels[i]["id"]) for i in idxs
                      if isinstance(i, int) and 0 <= i < len(hotels)]
        except Exception as exc:
            logger.warning("LLM reranker failed for %r: %s", question, exc)
            return []

        # De-duplicate while preserving order — models repeat indices.
        seen, out = set(), []
        for hid in ranked:
            if hid not in seen:
                seen.add(hid)
                out.append(hid)
        self._cache[key] = out
        self._flush()
        return out[:k]


# ---------------------------------------------------------------------------
# Learned
# ---------------------------------------------------------------------------

class LTRBaseline:
    """Supervised learning-to-rank over the same features the graph scorer uses.

    Trained with k-fold cross-validation over QUERIES and only ever asked to
    rank queries from its held-out fold, so no query's gold contributes to the
    model that ranks it. Without that discipline an LTR baseline trained and
    tested on the same gold scores near 1.0 and means nothing.

    This is the baseline that isolates the contribution. The proposed system's
    claim is that a *particular* weighting of these features helps; LTR learns
    an arbitrary (non-linear, non-simplex) function of the same features. If LTR
    matches it, the features are doing the work, not the scorer.

    CIRCULARITY WARNING — read before quoting the `LTR` row
    -------------------------------------------------------
    Against the RULE-BASED gold, this baseline is partly measuring the gold's
    own definition rather than ranking ability.

    `evaluation/gold.py` computes relevance as ``min(verdicts)`` over a fixed
    set of predicates: price within cap, rating above floor, star above floor,
    travel time within cap, amenities present. Several PAIR_FEATURES —
    `price_within_cap`, `rating_meets_floor`, `star_meets_floor`,
    `amenity_match_frac` — are those verdicts, handed to the model as inputs.
    A gradient-boosted tree given the terms of a min() can reconstruct the min()
    to high accuracy. Its held-out nDCG then measures how learnable the
    labelling function is, not how good the recommendations are.

    Measured on the 60-query Colombo set: `LTR` scores 0.954 nDCG@10 against
    WeightedGraphRAG's 0.796, winning five of six categories with several exact
    1.000s — the same saturation signature the graded gold was introduced to
    remove. This is the circular-gold problem resurfacing on the baseline side.

    `LTR[doc-only]` (use_pair_features=False) is therefore reported alongside
    it. It sees hotel attributes only, with no verdict indicators, so it cannot
    reconstruct the labelling rule. The GAP BETWEEN THE TWO ROWS is the useful
    quantity: it measures how much of the benchmark is recoverable from its own
    predicates, and belongs in the write-up as a benchmark diagnostic.

    Against the CHOICE-BASED gold (`evaluation/human_eval.py`) the concern does
    not apply — human bookings are not generated by these predicates — so the
    full-feature row is a fair baseline there.
    """

    name = "LTR"
    # Signals the harness to pass query_id, so the model that ranks a query is
    # always the one that never saw it in training.
    wants_query_id = True

    # Document features — properties of the hotel alone.
    DOC_FEATURES = ("price", "rating", "star", "distance_km",
                    "travel_time_traffic_min", "max_eta_change_min",
                    "event_impact", "n_amenities")

    # Query-document features are computed by `_pair_features`. They are what
    # make this a ranking model rather than a global popularity prior: without
    # them the model sees the same input for every query and can only learn
    # "which hotels are generally good", which is exactly the Popularity
    # baseline with extra steps. It scores 0.000 on the economic category
    # precisely because a per-query price threshold is invisible to it.
    PAIR_FEATURES = ("has_price_cap", "price_over_cap_ratio", "price_within_cap",
                     "has_rating_floor", "rating_margin", "rating_meets_floor",
                     "has_star_floor", "star_margin", "star_meets_floor",
                     "has_travel_cap", "travel_over_cap_ratio", "travel_within_cap",
                     "amenity_match_frac", "wants_cheapest", "wants_top_rated",
                     "wants_accessible", "avoid_traffic", "prefers_far")

    def __init__(self, folds: int = 5, seed: int = 17,
                 use_pair_features: bool = True) -> None:
        self.folds = folds
        self.seed = seed
        # See the CIRCULARITY WARNING below. `use_pair_features=False` drops the
        # constraint-satisfaction indicators and leaves a genuinely independent
        # learner; the gap between the two rows measures how much of the
        # benchmark is reconstructible from its own predicates.
        self.use_pair_features = use_pair_features
        self.name = "LTR" if use_pair_features else "LTR[doc-only]"
        self._models: Dict[str, Any] = {}   # query id -> model for its fold
        self._intents: Dict[str, QueryIntent] = {}
        self._medians: Dict[str, float] = {}
        self._fitted = False

    @staticmethod
    def is_available() -> bool:
        try:
            import sklearn  # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def _doc_features(cls, h: Dict[str, Any],
                      pool_medians: Dict[str, float]) -> List[float]:
        out = []
        for f in cls.DOC_FEATURES:
            if f == "n_amenities":
                out.append(float(len(h.get("amenities") or [])))
                continue
            v = h.get(f)
            out.append(float(v) if v is not None else pool_medians.get(f, 0.0))
        return out

    @staticmethod
    def _pair_features(h: Dict[str, Any], intent: QueryIntent) -> List[float]:
        """Query-document interaction features, in PAIR_FEATURES order.

        Each constraint contributes three signals: whether the query states it,
        how far the hotel is from satisfying it, and whether it satisfies it.
        A missing attribute is encoded as a large violation rather than as
        zero, so "no price" is not read as "free".
        """
        price, rating = h.get("price"), h.get("rating")
        star = h.get("star")
        tt = h.get("travel_time_traffic_min") or h.get("travel_time_min")

        cap = intent.max_price_lkr
        if cap:
            ratio = (float(price) / cap) if price else 5.0   # unknown = far over
            price_feats = [1.0, min(ratio, 5.0), 1.0 if price and float(price) <= cap else 0.0]
        else:
            price_feats = [0.0, 0.0, 0.0]

        floor = intent.min_rating
        if floor:
            margin = (float(rating) - floor) if rating is not None else -5.0
            rating_feats = [1.0, max(-5.0, min(margin, 5.0)),
                            1.0 if rating is not None and float(rating) >= floor else 0.0]
        else:
            rating_feats = [0.0, 0.0, 0.0]

        sfloor = intent.min_star
        if sfloor:
            margin = (float(star) - sfloor) if star else -5.0
            star_feats = [1.0, max(-5.0, min(margin, 5.0)),
                          1.0 if star and float(star) >= sfloor else 0.0]
        else:
            star_feats = [0.0, 0.0, 0.0]

        # QueryIntent carries no travel-time cap slot, so this stays inactive
        # unless one is added; kept for symmetry with the gold predicates.
        travel_cap = getattr(intent, "max_travel_time_min", None)
        if travel_cap:
            ratio = (float(tt) / travel_cap) if tt is not None else 5.0
            travel_feats = [1.0, min(ratio, 5.0),
                            1.0 if tt is not None and float(tt) <= travel_cap else 0.0]
        else:
            travel_feats = [0.0, 0.0, 0.0]

        req = [a.lower() for a in intent.required_amenities]
        have = [a.lower() for a in (h.get("amenities") or [])]
        amen = (sum(1 for a in req if any(a in x for x in have)) / len(req)) if req else 0.0

        return price_feats + rating_feats + star_feats + travel_feats + [
            amen,
            1.0 if intent.sort_intent == "cheapest" else 0.0,
            1.0 if intent.sort_intent == "highest_rated" else 0.0,
            1.0 if intent.sort_intent == "most_accessible" else 0.0,
            1.0 if intent.avoid_traffic else 0.0,
            1.0 if intent.proximity_preference == "far" else 0.0,
        ]

    def _vector(self, h: Dict[str, Any], intent: QueryIntent) -> List[float]:
        doc = self._doc_features(h, self._medians)
        return doc + self._pair_features(h, intent) if self.use_pair_features else doc

    def fit(self, pool: List[Dict[str, Any]], queries: List[Dict[str, Any]],
            gains_for: Any, intents: Optional[Dict[str, QueryIntent]] = None) -> None:
        """Fit one model per fold. `gains_for(query) -> {hotel_id: grade}`.

        `intents` is the harness's shared parse, so the LTR baseline conditions
        on exactly the same query reading every other system received.
        """
        import numpy as np
        from sklearn.ensemble import GradientBoostingRegressor

        self._medians = {}
        for f in self.DOC_FEATURES:
            vals = [float(h[f]) for h in pool
                    if f != "n_amenities" and h.get(f) is not None]
            self._medians[f] = float(np.median(vals)) if vals else 0.0

        self._intents = dict(intents or {})
        for q in queries:
            if q["id"] not in self._intents:
                self._intents[q["id"]] = parse_query(q["question"])

        idx = list(range(len(queries)))
        random.Random(self.seed).shuffle(idx)
        folds = [idx[i::self.folds] for i in range(self.folds)]

        for fi, held in enumerate(folds):
            held_set = set(held)
            X: List[List[float]] = []
            y: List[float] = []
            for qi, q in enumerate(queries):
                if qi in held_set:
                    continue  # never train on a query this model will rank
                gains = gains_for(q)
                intent = self._intents[q["id"]]
                for h in pool:
                    X.append(self._vector(h, intent))
                    y.append(float(gains.get(str(h["id"]), 0)))
            if not X:
                continue
            model = GradientBoostingRegressor(
                n_estimators=120, max_depth=3, learning_rate=0.08,
                random_state=self.seed + fi,
            )
            model.fit(np.array(X), np.array(y))
            for qi in held:
                self._models[queries[qi]["id"]] = model

        self._fitted = True

    def retrieve(self, question: str, city: str, k: int,
                 intent: Optional[QueryIntent] = None,
                 query_id: Optional[str] = None) -> List[str]:
        if not self._fitted or query_id not in self._models:
            # Unfitted, or a query outside the CV design: return nothing rather
            # than a ranking from a model that saw this query's gold.
            return []
        import numpy as np
        model = self._models[query_id]
        intent = intent or self._intents.get(query_id) or parse_query(question, default_city=city)
        pool = fetch_city_hotels(city)
        X = np.array([self._vector(h, intent) for h in pool])
        scores = model.predict(X)
        ranked = sorted(zip((str(h["id"]) for h in pool), scores), key=lambda t: -t[1])
        return [hid for hid, _ in ranked[:k]]


# ---------------------------------------------------------------------------
# Proposed system
# ---------------------------------------------------------------------------

class WeightedGraphBaseline:
    """The proposed system.

    `weight_profile` picks a static starting vector; `weight_policy` names a
    policy from src/graph/weight_policy.py ("handtuned" / "learned" / a profile
    name) and supersedes it. `price_policy` selects the missing-price rule.
    Each distinct configuration reports under its own name so they appear as
    separate rows scored on identical gold.
    """

    def __init__(self, weight_profile: Optional[str] = None,
                 price_policy: str = "neutral",
                 weight_policy: Optional[str] = None,
                 self_weight: float = 0.7,
                 label: Optional[str] = None) -> None:
        self.weight_profile = weight_profile
        tags = []
        if weight_policy:
            tags.append(weight_policy)
        elif weight_profile:
            tags.append(weight_profile)
        if price_policy != "neutral":
            tags.append(f"price={price_policy}")
        if self_weight >= 1.0:
            tags.append("no-diffusion")
        self.name = label or (
            "WeightedGraphRAG" if not tags else f"WeightedGraphRAG[{','.join(tags)}]"
        )

        model = None
        if weight_policy:
            from src.graph.weight_policy import get_policy
            model = get_policy(weight_policy)

        self._retriever = WeightedRetriever(
            weight_profile=weight_profile, price_policy=price_policy,
            self_weight=self_weight, weight_model=model,
            # Safe here and only here: the graph is static for the duration of
            # an evaluation run, and without it a ten-system table issues the
            # expensive multi-hop query several hundred times identically.
            cache_candidates=True,
        )

    def retrieve(self, question: str, city: str, k: int,
                 intent: Optional[QueryIntent] = None) -> List[str]:
        intent = _resolve_intent(question, city, intent)
        result = self._retriever.retrieve(intent, limit=k)
        return [h.id for h in result.hotels]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verbalise(h: Dict[str, Any]) -> str:
    """Render a hotel as the plain sentence the text baselines see.

    Numeric attributes are spelled out because a cross-encoder or an LLM cannot
    read a column it is never shown; withholding them would make those
    baselines strawmen.
    """
    bits = [str(h.get("name") or "Hotel")]
    if h.get("rating") is not None:
        bits.append(f"rated {float(h['rating']):.1f}/5")
    if h.get("star"):
        bits.append(f"{int(h['star'])}-star")
    if h.get("price"):
        bits.append(f"{float(h['price']):.0f} LKR per night")
    else:
        bits.append("price not listed")
    if h.get("distance_km") is not None:
        bits.append(f"{float(h['distance_km']):.1f} km from the centre")
    tt = h.get("travel_time_traffic_min") or h.get("travel_time_min")
    if tt is not None:
        bits.append(f"about {float(tt):.0f} min travel time")
    amen = ", ".join((h.get("amenities") or [])[:8])
    if amen:
        bits.append(f"amenities: {amen}")
    attr = ", ".join((h.get("attractions") or [])[:5])
    if attr:
        bits.append(f"near: {attr}")
    return "; ".join(bits)


def all_baselines(weight_profiles: Optional[List[str]] = None,
                  price_policies: Optional[List[str]] = None,
                  weight_policies: Optional[List[str]] = None,
                  include_floors: bool = True,
                  include_llm: bool = True,
                  include_cross_encoder: bool = True,
                  include_ltr: bool = True,
                  include_ablations: bool = True) -> List[Any]:
    """Assemble the comparison line-up, skipping whatever is unavailable.

    Every optional system logs the reason it was skipped, so a thin table can be
    explained rather than being mistaken for a missing comparison.
    """
    baselines: List[Any] = []

    if include_floors:
        baselines += [RandomBaseline(), PopularityBaseline()]

    baselines.append(FilterBaseline())

    if vs.is_available():
        baselines += [KeywordBaseline(), SemanticBaseline(), HybridBaseline()]
        if include_cross_encoder:
            if CrossEncoderBaseline.is_available():
                baselines.append(CrossEncoderBaseline())
            else:
                logger.warning(
                    "CrossEncoder skipped — sentence-transformers not importable."
                )
    else:
        logger.warning(
            "pgvector index unavailable — skipping Keyword/SemanticVec/Hybrid/"
            "CrossEncoder. Start the pgvector container and run "
            "scripts/build_vector_index.py to enable them."
        )

    if include_llm:
        if LLMRerankerBaseline.is_available():
            baselines.append(LLMRerankerBaseline())
        else:
            logger.warning(
                "LLMReranker skipped — no LLM_API_KEY configured. This is the "
                "baseline reviewers expect; set a key before publishing."
            )

    if include_ltr:
        if LTRBaseline.is_available():
            # Both variants always, never the full-feature one alone: quoted by
            # itself the `LTR` row is misleading against rule-based gold. See
            # the CIRCULARITY WARNING in LTRBaseline.
            baselines.append(LTRBaseline())
            baselines.append(LTRBaseline(use_pair_features=False))
        else:
            logger.warning("LTR skipped — scikit-learn not importable.")

    baselines.append(WeightedGraphBaseline())

    for name in (weight_profiles or []):
        baselines.append(WeightedGraphBaseline(weight_profile=name))
    for policy in (weight_policies or []):
        try:
            baselines.append(WeightedGraphBaseline(weight_policy=policy))
        except (FileNotFoundError, KeyError) as exc:
            logger.warning("weight policy %r skipped — %s", policy, exc)
    for pp in (price_policies or []):
        baselines.append(WeightedGraphBaseline(price_policy=pp))

    if include_ablations:
        # self_weight = 1.0 disables neighbourhood diffusion, isolating what the
        # multi-hop traversal contributes.
        baselines.append(WeightedGraphBaseline(
            self_weight=1.0, label="WeightedGraphRAG[no-diffusion]"
        ))

    return baselines
