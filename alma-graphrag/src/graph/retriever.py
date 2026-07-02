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
    event: float = 0.0  # only active when an ActiveEvent is in play

    def normalised(self) -> "ScoringWeights":
        total = (
            self.spatial + self.accessibility + self.facility
            + self.economic + self.disruption + self.event
        )
        if total <= 0:
            return ScoringWeights()
        return ScoringWeights(
            spatial=self.spatial / total,
            accessibility=self.accessibility / total,
            facility=self.facility / total,
            economic=self.economic / total,
            disruption=self.disruption / total,
            event=self.event / total,
        )

    def to_dict(self) -> Dict[str, float]:
        d = {
            "spatial": round(self.spatial, 3),
            "accessibility": round(self.accessibility, 3),
            "facility": round(self.facility, 3),
            "economic": round(self.economic, 3),
            "disruption": round(self.disruption, 3),
        }
        if self.event:
            d["event"] = round(self.event, 3)
        return d


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


def weights_for_profile(profile: Any, intent: QueryIntent, event_active: bool) -> ScoringWeights:
    """Resolve scoring weights for a personalised request.

    A UserProfile (duck-typed: .weights, .event_preference) overrides the
    intent-derived weights. When an event is in play and the profile expresses a
    seek/avoid preference, an ``event`` weight is added and everything is
    renormalised.
    """
    if profile is not None and getattr(profile, "weights", None) is not None:
        base = profile.weights
        w = ScoringWeights(
            spatial=base.spatial, accessibility=base.accessibility,
            facility=base.facility, economic=base.economic,
            disruption=base.disruption, event=base.event,
        )
    else:
        w = weights_for_intent(intent)

    if event_active and profile is not None and getattr(profile, "event_preference", "neutral") in ("seek", "avoid"):
        w.event = 0.30  # strong event influence on a personalised ranking
    return w.normalised()


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    from math import radians, sin, cos, sqrt, atan2
    R = 6371.0
    la1, lo1, la2, lo2 = map(radians, [lat1, lng1, lat2, lng2])
    dlat, dlon = la2 - la1, lo2 - lo1
    a = sin(dlat / 2) ** 2 + cos(la1) * cos(la2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


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
       h.lat                               AS lat,
       h.lng                               AS lng,
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

    def retrieve(
        self,
        intent: QueryIntent,
        limit: int = 10,
        profile: Any = None,
        event: Any = None,
    ) -> RetrievalResult:
        city = intent.city
        if not city:
            return RetrievalResult(city=None, intent=intent, weights=ScoringWeights(), hotels=[])

        # A user profile overrides the proximity preference (e.g. quiet seeker
        # wants distance from the centre / event).
        if profile is not None and getattr(profile, "proximity_preference", "any") != "any":
            intent.proximity_preference = profile.proximity_preference

        candidates = self._fetch_candidates(city)
        if profile is not None:
            weights = weights_for_profile(profile, intent, event_active=event is not None)
        else:
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

        scored = self._score(filtered, intent, weights, profile=profile, event=event)
        scored.sort(key=lambda h: h.score, reverse=True)
        result.hotels = scored[:limit]
        logger.info(
            "Weighted retrieve: city=%s candidates=%d filtered=%d returned=%d weights=%s profile=%s event=%s",
            city, len(candidates), len(filtered), len(result.hotels), weights.to_dict(),
            getattr(profile, "id", None), getattr(event, "name", None),
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
        profile: Any = None,
        event: Any = None,
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

        # --- Event impact zone: distance from each hotel to the event ----------
        event_pref = getattr(profile, "event_preference", "neutral") if profile else "neutral"
        ev_dist: Dict[str, float] = {}
        if event is not None:
            for c in cands:
                lat, lng = c.get("lat"), c.get("lng")
                if lat and lng:
                    ev_dist[str(c.get("id"))] = _haversine_km(event.lat, event.lng, float(lat), float(lng))
            ed_lo, ed_hi = _minmax(list(ev_dist.values()))

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
            # Combine coarse severity buckets with a graded penalty from the
            # actual route-delay minutes, so disruption discriminates even when
            # signals are nominally "light".
            disruption = 1.0
            sev = [s for s in (c.get("signal_severities") or []) if s]
            heavy = sum(1 for s in sev if s == "heavy")
            moderate = sum(1 for s in sev if s == "moderate")
            events = int(c.get("event_count") or 0)
            etas = [float(e) for e in (c.get("signal_etas") or []) if e]
            max_eta = max(etas) if etas else 0.0
            disruption -= 0.40 * heavy + 0.20 * moderate + 0.30 * min(events, 2)
            disruption -= min(max_eta / 20.0, 0.5)  # +10 min route delay -> -0.5
            disruption = max(0.0, min(1.0, disruption))
            if heavy:
                reasons.append(f"heavy traffic on route (x{heavy})")
            elif max_eta >= 3:
                reasons.append(f"+{max_eta:.0f} min traffic delay on route")
            elif disruption > 0.85:
                reasons.append("low disruption / stable conditions")

            # --- event affinity (personalised, only when an event is active) ---
            event_affinity = 0.0
            if event is not None and weights.event > 0:
                ed = ev_dist.get(str(c.get("id")))
                if ed is None:
                    event_affinity = 0.5  # unknown location = neutral
                else:
                    norm_far = 0.5 if ed_hi <= ed_lo else (ed - ed_lo) / (ed_hi - ed_lo)
                    in_zone = ed <= event.impact_radius_km
                    if event_pref == "seek":
                        event_affinity = 1.0 - norm_far  # closer = better
                        if in_zone:
                            event_affinity = min(1.0, event_affinity + 0.15)
                            reasons.append(f"inside {event.name} zone ({ed:.1f} km) - great for attending")
                    elif event_pref == "avoid":
                        event_affinity = norm_far  # farther = better
                        if in_zone:
                            # The event inflates traffic/noise in its zone.
                            sev_pen = {"high": 0.45, "medium": 0.30, "low": 0.15}.get(event.severity, 0.30)
                            disruption = max(0.0, disruption - sev_pen)
                            reasons.append(f"inside {event.name} impact zone ({ed:.1f} km) - crowds/traffic")
                        else:
                            reasons.append(f"{ed:.1f} km from {event.name} - calm")
                    else:
                        event_affinity = 0.5

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
            if event is not None and weights.event > 0:
                components["event"] = round(event_affinity, 3)
                weighted["event"] = event_affinity * weights.event
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
