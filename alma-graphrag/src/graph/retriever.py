"""
Feasibility-first weighted multi-hop GraphRAG retriever.

Implements the proposal's Algorithm 1 (Weighted Multi-Hop Traversal):

    score = w_spatial      * spatial_score
          + w_accessibility * accessibility_score
          + w_facility      * facility_score
          + w_economic      * economic_score
          + w_disruption    * disruption_score

The traversal is genuine multi-hop: City -> Hotel (hop 1) -> {Amenity,
AttractionType, TrafficSignal, Event} (hop 2). Raw per-hotel metrics are pulled
in a single parameterised Cypher query (safe — no string interpolation of user
input), then normalised and combined in Python so each component is inspectable
(supports the thesis's explainability requirement and the P2 evaluation harness).

Weights are *dynamic*: they shift based on QueryIntent (e.g. a "quiet seeker"
up-weights disruption avoidance and inverts the spatial preference). This is the
extension point for P3 personalisation — a UserProfile simply supplies weights.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.crag.query_parser import QueryIntent
from src.graph.query import _get_driver

logger = logging.getLogger("alma.graph.retriever")


# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

@dataclass
class ScoringWeights:
    spatial: float = 0.25
    accessibility: float = 0.20
    facility: float = 0.25
    economic: float = 0.15
    disruption: float = 0.15

    def normalised(self) -> "ScoringWeights":
        total = self.spatial + self.accessibility + self.facility + self.economic + self.disruption
        if total <= 0:
            return ScoringWeights()
        return ScoringWeights(
            spatial=self.spatial / total,
            accessibility=self.accessibility / total,
            facility=self.facility / total,
            economic=self.economic / total,
            disruption=self.disruption / total,
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "spatial": round(self.spatial, 3),
            "accessibility": round(self.accessibility, 3),
            "facility": round(self.facility, 3),
            "economic": round(self.economic, 3),
            "disruption": round(self.disruption, 3),
        }


def weights_for_intent(intent: QueryIntent) -> ScoringWeights:
    """Derive dynamic scoring weights from query intent (P3 personalisation hook)."""
    w = ScoringWeights()

    if intent.sort_intent == "cheapest":
        w.economic += 0.20
        w.facility -= 0.05
    elif intent.sort_intent == "highest_rated":
        w.facility += 0.20
        w.economic -= 0.05
    elif intent.sort_intent == "most_accessible":
        w.accessibility += 0.20
        w.disruption += 0.10

    if intent.accessibility_priority == "high":
        w.accessibility += 0.15

    if intent.avoid_traffic:
        w.disruption += 0.15
        w.accessibility += 0.05

    # Quiet seeker: disruption avoidance dominates, proximity matters less.
    if intent.proximity_preference == "far":
        w.disruption += 0.10
        w.spatial = max(0.05, w.spatial - 0.10)

    if intent.required_amenities:
        w.facility += 0.10

    # Clamp negatives, then normalise to sum 1.
    w.spatial = max(0.0, w.spatial)
    w.accessibility = max(0.0, w.accessibility)
    w.facility = max(0.0, w.facility)
    w.economic = max(0.0, w.economic)
    w.disruption = max(0.0, w.disruption)
    return w.normalised()


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ScoredHotel:
    id: str
    name: str
    score: float
    components: Dict[str, float]          # normalised sub-scores [0,1]
    weighted_components: Dict[str, float]  # sub-score * weight
    raw: Dict[str, Any]                   # raw metrics for context formatting
    reasons: List[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    city: Optional[str]
    intent: QueryIntent
    weights: ScoringWeights
    hotels: List[ScoredHotel]
    filters_relaxed: bool = False
    candidate_count: int = 0


# ---------------------------------------------------------------------------
# Cypher — multi-hop candidate fetch (parameterised, injection-safe)
# ---------------------------------------------------------------------------

_CANDIDATE_QUERY = """
MATCH (h:Hotel)-[loc:LOCATED_IN]->(c:City)
WHERE toLower(c.name) = toLower($city)
OPTIONAL MATCH (h)-[:HAS_AMENITY]->(a:Amenity)
OPTIONAL MATCH (h)-[:NEAR_ATTRACTION]->(at:AttractionType)
OPTIONAL MATCH (h)-[:NEAR]->(l:Location)
OPTIONAL MATCH (h)-[:HAS_SIGNAL]->(ts:TrafficSignal)
OPTIONAL MATCH (h)-[:AFFECTED_BY]->(e:Event)
WITH h, loc,
     collect(DISTINCT toLower(a.name))  AS amenities,
     collect(DISTINCT toLower(at.name)) AS attractions,
     collect(DISTINCT toLower(l.name))  AS locations,
     collect(DISTINCT ts.severity)      AS signal_severities,
     collect(DISTINCT ts.eta_change_min) AS signal_etas,
     count(DISTINCT e)                  AS event_count
RETURN h.id                                AS id,
       h.name                              AS name,
       h.rating                            AS rating,
       h.star_rating                       AS star,
       h.price_per_night_lkr               AS price,
       h.price_range                       AS price_range,
       h.address                           AS address,
       h.source                            AS source,
       coalesce(loc.distance_km, loc.distance_from_center_km) AS distance_km,
       loc.travel_time_min                 AS travel_time_min,
       loc.travel_time_traffic_min         AS travel_time_traffic_min,
       amenities, attractions, locations,
       signal_severities, signal_etas, event_count
"""


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _minmax(values: List[Optional[float]]) -> Tuple[float, float]:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return 0.0, 0.0
    return min(nums), max(nums)


def _norm_lower_better(v: Optional[float], lo: float, hi: float, default: float = 0.5) -> float:
    """Lower raw value -> higher score (e.g. distance, travel time, price)."""
    if v is None:
        return default
    if hi <= lo:
        return 1.0
    return 1.0 - (float(v) - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class WeightedRetriever:
    """Multi-hop weighted GraphRAG retriever (proposal Algorithm 1)."""

    def retrieve(self, intent: QueryIntent, limit: int = 10) -> RetrievalResult:
        city = intent.city
        if not city:
            return RetrievalResult(city=None, intent=intent, weights=ScoringWeights(), hotels=[])

        candidates = self._fetch_candidates(city)
        weights = weights_for_intent(intent)

        result = RetrievalResult(
            city=city, intent=intent, weights=weights, hotels=[],
            candidate_count=len(candidates),
        )
        if not candidates:
            return result

        # Hard filters (soft-relax if they wipe out all candidates).
        filtered = self._apply_filters(candidates, intent)
        if not filtered:
            filtered = candidates
            result.filters_relaxed = True

        scored = self._score(filtered, intent, weights)
        scored.sort(key=lambda h: h.score, reverse=True)
        result.hotels = scored[:limit]
        logger.info(
            "Weighted retrieve: city=%s candidates=%d filtered=%d returned=%d weights=%s",
            city, len(candidates), len(filtered), len(result.hotels), weights.to_dict(),
        )
        return result

    # -- data access --------------------------------------------------------

    def _fetch_candidates(self, city: str) -> List[Dict[str, Any]]:
        driver = _get_driver()
        with driver.session() as session:
            return session.run(_CANDIDATE_QUERY, {"city": city}).data()

    # -- filtering ----------------------------------------------------------

    def _apply_filters(self, cands: List[Dict[str, Any]], intent: QueryIntent) -> List[Dict[str, Any]]:
        out = []
        for c in cands:
            rating = c.get("rating")
            star = c.get("star")
            price = c.get("price")

            if intent.min_rating is not None and rating is not None and float(rating) < intent.min_rating:
                continue
            if intent.min_star is not None and star is not None and float(star) < intent.min_star:
                continue
            if intent.max_price_lkr is not None and price:
                if float(price) > intent.max_price_lkr:
                    continue
            if intent.min_price_lkr is not None and price:
                if float(price) < intent.min_price_lkr:
                    continue
            out.append(c)
        return out

    # -- scoring ------------------------------------------------------------

    def _score(
        self,
        cands: List[Dict[str, Any]],
        intent: QueryIntent,
        weights: ScoringWeights,
    ) -> List[ScoredHotel]:
        # Precompute min/max for normalisation across the candidate set.
        dist_lo, dist_hi = _minmax([c.get("distance_km") for c in cands])
        tt_lo, tt_hi = _minmax([
            c.get("travel_time_traffic_min") or c.get("travel_time_min") for c in cands
        ])
        price_lo, price_hi = _minmax([c.get("price") for c in cands])
        amen_counts = [len(c.get("amenities") or []) for c in cands]
        max_amen = max(amen_counts) if amen_counts else 0

        req_amen = {a.lower() for a in intent.required_amenities}
        req_attr = {a.lower() for a in intent.near_attractions}

        scored: List[ScoredHotel] = []
        for c in cands:
            reasons: List[str] = []

            # --- spatial ---------------------------------------------------
            spatial = _norm_lower_better(c.get("distance_km"), dist_lo, dist_hi)
            if intent.proximity_preference == "far":
                spatial = 1.0 - spatial  # quiet seeker wants distance from centre
            elif intent.proximity_preference == "close" and c.get("distance_km") is not None:
                if spatial > 0.7:
                    reasons.append("central / walkable location")

            # --- accessibility (uses live traffic travel time) -------------
            tt = c.get("travel_time_traffic_min") or c.get("travel_time_min")
            accessibility = _norm_lower_better(tt, tt_lo, tt_hi)
            if c.get("travel_time_traffic_min") and accessibility > 0.7:
                reasons.append(f"fast access (~{float(c['travel_time_traffic_min']):.0f} min in traffic)")

            # --- facility --------------------------------------------------
            amenities = set(c.get("amenities") or [])
            attractions = set(c.get("attractions") or [])
            if req_amen:
                matched = sum(1 for a in req_amen if any(a in x for x in amenities))
                amen_match = matched / len(req_amen)
                if matched:
                    reasons.append(f"matches {matched}/{len(req_amen)} requested amenities")
            else:
                amen_match = (len(amenities) / max_amen) if max_amen else 0.5
            attr_match = 0.0
            if req_attr:
                am = sum(1 for a in req_attr if any(a in x for x in attractions | set(c.get("locations") or [])))
                attr_match = am / len(req_attr)
                if am:
                    reasons.append(f"near {am}/{len(req_attr)} requested attractions")
            star_score = (float(c["star"]) / 5.0) if c.get("star") else 0.0
            rating_score = (float(c["rating"]) / 5.0) if c.get("rating") else 0.0
            facility = (
                0.40 * amen_match
                + 0.20 * attr_match
                + 0.20 * rating_score
                + 0.20 * star_score
            )

            # --- economic --------------------------------------------------
            if c.get("price"):
                economic = _norm_lower_better(c.get("price"), price_lo, price_hi)
                if intent.max_price_lkr and float(c["price"]) <= intent.max_price_lkr:
                    reasons.append("within budget")
            else:
                economic = 0.5  # unknown price = neutral

            # --- disruption (traffic signals + active events) -------------
            disruption = 1.0
            sev = [s for s in (c.get("signal_severities") or []) if s]
            heavy = sum(1 for s in sev if s == "heavy")
            moderate = sum(1 for s in sev if s == "moderate")
            events = int(c.get("event_count") or 0)
            disruption -= 0.40 * heavy + 0.20 * moderate + 0.30 * min(events, 2)
            disruption = max(0.0, min(1.0, disruption))
            if heavy:
                reasons.append(f"⚠ heavy traffic nearby (x{heavy})")
            elif disruption > 0.85:
                reasons.append("low disruption / stable conditions")

            components = {
                "spatial": round(spatial, 3),
                "accessibility": round(accessibility, 3),
                "facility": round(facility, 3),
                "economic": round(economic, 3),
                "disruption": round(disruption, 3),
            }
            weighted = {
                "spatial": spatial * weights.spatial,
                "accessibility": accessibility * weights.accessibility,
                "facility": facility * weights.facility,
                "economic": economic * weights.economic,
                "disruption": disruption * weights.disruption,
            }
            total = sum(weighted.values())

            scored.append(ScoredHotel(
                id=str(c.get("id")),
                name=c.get("name") or "Unknown",
                score=round(total, 4),
                components=components,
                weighted_components={k: round(v, 4) for k, v in weighted.items()},
                raw=c,
                reasons=reasons,
            ))
        return scored


# ---------------------------------------------------------------------------
# Context formatting for the LLM generator
# ---------------------------------------------------------------------------

def format_retrieval_context(result: RetrievalResult) -> str:
    """Render ranked, scored hotels into LLM-ready context (replaces dump)."""
    lines: List[str] = []
    city = result.city or "the area"
    lines.append(f"=== Ranked hotels in {city} (feasibility-first weighted GraphRAG) ===")
    lines.append(
        "Ranking weights — "
        + ", ".join(f"{k}:{v}" for k, v in result.weights.to_dict().items())
    )
    if result.filters_relaxed:
        lines.append("(Note: strict filters returned no hotels; constraints were relaxed.)")
    lines.append("")

    for rank, h in enumerate(result.hotels, 1):
        r = h.raw
        price = r.get("price")
        price_str = f"{float(price):.0f} LKR/night" if price else (r.get("price_range") or "price N/A")
        tt = r.get("travel_time_traffic_min") or r.get("travel_time_min")
        tt_str = f"{float(tt):.0f} min" if tt else "n/a"
        dist = r.get("distance_km")
        dist_str = f"{float(dist):.1f} km" if dist is not None else "n/a"
        star_str = f" | {int(r['star'])}-star" if r.get("star") else ""
        amen = ", ".join((r.get("amenities") or [])[:8]) or "none listed"
        attractions = ", ".join((r.get("attractions") or [])[:6]) or "none"
        comp = h.components
        reasons = "; ".join(h.reasons) if h.reasons else "-"

        lines.append(
            f"#{rank} {h.name}  [score={h.score:.3f}]\n"
            f"    Rating: {r.get('rating', 'N/A')}/5{star_str}"
            f" | Price: {price_str} | Distance: {dist_str} | Travel time: {tt_str}\n"
            f"    Amenities: [{amen}] | Near: [{attractions}]\n"
            f"    Score breakdown - spatial:{comp['spatial']} accessibility:{comp['accessibility']} "
            f"facility:{comp['facility']} economic:{comp['economic']} disruption:{comp['disruption']}\n"
            f"    Why: {reasons}"
        )
        lines.append("")

    if not result.hotels:
        lines.append("No hotels found for this city in the knowledge graph.")

    return "\n".join(lines)
