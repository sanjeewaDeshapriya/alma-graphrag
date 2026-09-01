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

from src.config import SCORING_WEIGHTS_PROFILE
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


# ---------------------------------------------------------------------------
# Named weight profiles
# ---------------------------------------------------------------------------
#
# `handset` is the original hand-tuned prior. `elicited` and `blended` come from
# the discrete-choice experiment in studies/weight-elicitation (material
# v4-rooms-20260818-minmax): 95 clean-cohort participants, 949 choices over a
# 32-hotel pool, conditional logit, weights bootstrapped by participant.
# Full provenance: studies/weight-elicitation/analysis/DATA_AUDIT.md
#
# Held-out validation (73 participants never seen during fitting) put elicited
# ahead of handset on every metric — nDCG@10 0.708 vs 0.596, p < 0.001.
#
# The catch: `facility` and `economic` fitted NEGATIVE and clip to zero on the
# simplex. That is a real, sign-stable effect in the booking task, but it is not
# transferable to constraint-satisfaction queries — a retriever with
# economic = 0 cannot answer "cheapest hotel", and facility = 0 ignores
# requested amenities. `blended` therefore keeps the hand-set prior mass for
# those two (0.25 + 0.15 = 0.40) and distributes the remaining 0.60 in the
# elicited proportions. Which profile to run is an empirical question — set
# SCORING_WEIGHTS_PROFILE and compare in evaluation/run_eval.py.

HANDSET_WEIGHTS = ScoringWeights(
    spatial=0.25, accessibility=0.20, facility=0.25, economic=0.15, disruption=0.15,
)

ELICITED_WEIGHTS = ScoringWeights(
    spatial=0.563, accessibility=0.401, facility=0.000, economic=0.000, disruption=0.036,
)

BLENDED_WEIGHTS = ScoringWeights(
    spatial=0.338, accessibility=0.240, facility=0.250, economic=0.150, disruption=0.022,
)

WEIGHT_PROFILES: Dict[str, ScoringWeights] = {
    "handset": HANDSET_WEIGHTS,
    "elicited": ELICITED_WEIGHTS,
    "blended": BLENDED_WEIGHTS,
}


def base_weights(profile: Optional[str] = None) -> ScoringWeights:
    """Starting weights before intent/profile adjustment.

    Falls back to the hand-set prior for an unknown name rather than raising —
    a typo in an env var should not take the retriever down.
    """
    name = (profile or SCORING_WEIGHTS_PROFILE or "handset").lower()
    base = WEIGHT_PROFILES.get(name)
    if base is None:
        logger.warning("Unknown SCORING_WEIGHTS_PROFILE %r; using 'handset'", name)
        base = HANDSET_WEIGHTS
    # Copy — callers mutate the returned object.
    return ScoringWeights(
        spatial=base.spatial, accessibility=base.accessibility,
        facility=base.facility, economic=base.economic,
        disruption=base.disruption, event=base.event,
    )


def weights_for_intent(intent: QueryIntent, profile: Optional[str] = None) -> ScoringWeights:
    """Derive dynamic scoring weights from query intent (P3 personalisation hook)."""
    w = base_weights(profile)

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


def weights_for_profile(profile: Any, intent: QueryIntent, event_active: bool,
                        weight_profile: Optional[str] = None) -> ScoringWeights:
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
        w = weights_for_intent(intent, weight_profile)

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

# Hops of NEAR_HOTEL traversal used for neighbourhood diffusion. Cypher cannot
# parameterise a variable-length bound, so this is templated into the query as a
# validated integer (never user input).
DEFAULT_MAX_HOPS = 2

# Per-hop attenuation for the diffusion. 0.5 means a 2-hop neighbour carries
# half the weight of a 1-hop neighbour before the distance term is applied.
DEFAULT_HOP_DECAY = 0.5

# How much of a hotel's disruption exposure comes from its OWN route signal
# versus its neighbourhood. 1.0 reproduces the pre-diffusion behaviour exactly,
# which is the ablation to report.
DEFAULT_SELF_WEIGHT = 0.7


def _candidate_query(max_hops: int = DEFAULT_MAX_HOPS) -> str:
    """Multi-hop candidate fetch with neighbourhood diffusion.

    Two-stage traversal:

      Stage 1 (hop 1) — the hotel's own attribute neighbourhood: amenities,
      attraction types, named locations, its route traffic signal, and any
      events it is linked to. This is a star-join and always was.

      Stage 2 (hops 1..N) — diffusion over the ``NEAR_HOTEL`` spatial proximity
      graph built by ``scripts/build_graph_topology.py``. Each hotel's
      congestion and event exposure is re-estimated as an inverse-distance,
      hop-attenuated weighted mean over the hotels it is spatially coupled to,
      including hotels reached only transitively (A near B, B near C, so C's
      state informs A at reduced weight).

    Stage 2 is the part that needs a graph. It is transitive inference: a hotel
    with no signal of its own still receives an exposure estimate from its
    neighbourhood, and two hotels with identical own-signals separate because
    their neighbourhoods differ. A relational star-join cannot express it
    without a recursive CTE, and the value differs per hotel, so — unlike the
    constant it replaces — it can actually change a ranking.

    The weight for a neighbour at path length ``hops`` and straight-line
    distance ``km`` is::

        w = hop_decay^(hops-1) / (1 + km)

    Weights are normalised by their own sum, so a hotel in a dense
    neighbourhood is not penalised for having many neighbours; only the
    distance-weighted *state* of the neighbourhood matters.
    """
    hops = int(max_hops)
    if not 1 <= hops <= 4:
        raise ValueError(f"max_hops must be in 1..4, got {max_hops}")
    return f"""
MATCH (h:Hotel)-[loc:LOCATED_IN]->(c:City)
WHERE toLower(c.name) = toLower($city)

// --- hop 1: the hotel's own attribute neighbourhood ---------------------
OPTIONAL MATCH (h)-[:HAS_AMENITY]->(a:Amenity)
OPTIONAL MATCH (h)-[:NEAR_ATTRACTION]->(at:AttractionType)
OPTIONAL MATCH (h)-[:NEAR]->(l:Location)
OPTIONAL MATCH (h)-[:HAS_SIGNAL]->(ts:TrafficSignal)
OPTIONAL MATCH (h)-[ae:AFFECTED_BY]->(e:Event)
WITH h, loc,
     collect(DISTINCT toLower(a.name))   AS amenities,
     collect(DISTINCT toLower(at.name))  AS attractions,
     collect(DISTINCT toLower(l.name))   AS locations,
     collect(DISTINCT ts.severity)       AS signal_severities,
     collect(DISTINCT ts.eta_change_min) AS signal_etas,
     count(DISTINCT e)                   AS event_count,
     max(coalesce(ae.impact_score, 0.0)) AS event_impact,
     min(ae.distance_km)                 AS event_distance_km

// --- hops 1..{hops}: diffusion over the NEAR_HOTEL proximity graph ---------
CALL {{
    WITH h
    MATCH path = (h)-[:NEAR_HOTEL*1..{hops}]-(nb:Hotel)
    WHERE nb <> h AND nb.lat IS NOT NULL AND nb.lng IS NOT NULL
    WITH h, nb, min(length(path)) AS hops
    OPTIONAL MATCH (nb)-[:HAS_SIGNAL]->(nts:TrafficSignal)
    OPTIONAL MATCH (nb)-[nae:AFFECTED_BY]->(:Event)
    WITH h, nb, hops,
         max(coalesce(nts.eta_change_min, 0.0)) AS nb_eta,
         max(CASE nts.severity
                 WHEN 'heavy'    THEN 1.0
                 WHEN 'moderate' THEN 0.5
                 ELSE 0.0 END)              AS nb_sev,
         max(coalesce(nae.impact_score, 0.0)) AS nb_impact
    WITH hops, nb_eta, nb_sev, nb_impact,
         point.distance(
             point({{latitude: h.lat,  longitude: h.lng}}),
             point({{latitude: nb.lat, longitude: nb.lng}})
         ) / 1000.0 AS km
    WITH (($hop_decay) ^ (hops - 1)) / (1.0 + km) AS w, nb_eta, nb_sev, nb_impact
    RETURN
        CASE WHEN sum(w) = 0 THEN 0.0 ELSE sum(w * nb_eta)    / sum(w) END AS nbr_eta,
        CASE WHEN sum(w) = 0 THEN 0.0 ELSE sum(w * nb_sev)    / sum(w) END AS nbr_severity,
        CASE WHEN sum(w) = 0 THEN 0.0 ELSE sum(w * nb_impact) / sum(w) END AS nbr_event_impact,
        count(*) AS nbr_count
}}

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
       signal_severities, signal_etas, event_count,
       event_impact, event_distance_km,
       nbr_eta, nbr_severity, nbr_event_impact, nbr_count
"""


# Kept for backwards compatibility with anything importing the old constant.
_CANDIDATE_QUERY = _candidate_query()


# ---------------------------------------------------------------------------
# Candidate cache (opt-in)
# ---------------------------------------------------------------------------
#
# The multi-hop query is expensive, and an evaluation run issues it once per
# system per query — with ten systems and sixty queries that is 600 identical
# round trips against a graph that does not change during the run.
#
# Caching is OPT-IN rather than default because the live API must not serve
# stale traffic: signals refresh every 15-30 minutes and a cached pool would
# quietly defeat the entire premise of a live disruption-aware system. The
# evaluation harness turns it on for the duration of a run; nothing else should.
_CANDIDATE_CACHE: Dict[Tuple[str, int, float], List[Dict[str, Any]]] = {}


def clear_candidate_cache() -> None:
    """Drop cached candidate pools. Call after any write to the graph."""
    _CANDIDATE_CACHE.clear()


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
# Missing-price policy
# ---------------------------------------------------------------------------
#
# 24 of 58 Colombo hotels (41%) carry no price: LiteAPI returns rates only for
# hotels with live availability, and google_places deliberately stores NULL
# rather than a fabricated default (see src/ingest/google_places.py).
#
# Imputation is therefore not a detail — it decides the economic sub-score for
# four hotels in ten, and the original choice (a neutral 0.5) is the single most
# consequential undocumented constant in the scorer. A neutral 0.5 ranks an
# unpriced hotel ABOVE every hotel more expensive than the pool midpoint, so a
# missing value actively helps a candidate. That is an indefensible default to
# leave unexamined in a paper.
#
# The policy is now explicit and swept by the evaluation harness, so the
# write-up can report whether conclusions survive all four choices instead of
# asserting one.
#
#   neutral  — 0.5. The original behaviour; preserves published numbers.
#   worst    — 0.0. Conservative: no price evidence means no economic credit.
#              Never rewards missingness.
#   median   — impute the pool median price, then normalise as usual. Standard
#              statistical imputation; assumes missing-at-random, which is
#              questionable here (missingness correlates with supplier).
#   exclude  — drop unpriced hotels from the candidate set entirely. Cleanest
#              inference, smallest pool; report the pool size when using it.
#
PRICE_POLICIES = ("neutral", "worst", "median", "exclude")
DEFAULT_PRICE_POLICY = "neutral"


def _median(values: List[Optional[float]]) -> Optional[float]:
    nums = sorted(float(v) for v in values if v is not None)
    if not nums:
        return None
    mid = len(nums) // 2
    return nums[mid] if len(nums) % 2 else (nums[mid - 1] + nums[mid]) / 2.0


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class WeightedRetriever:
    """Multi-hop weighted GraphRAG retriever (proposal Algorithm 1).

    `weight_profile` selects the starting weight vector — one of
    WEIGHT_PROFILES. None follows SCORING_WEIGHTS_PROFILE from config. The
    evaluation harness instantiates one retriever per profile so the profiles
    can be compared head-to-head on the same query set.
    """

    def __init__(
        self,
        weight_profile: Optional[str] = None,
        price_policy: str = DEFAULT_PRICE_POLICY,
        max_hops: int = DEFAULT_MAX_HOPS,
        hop_decay: float = DEFAULT_HOP_DECAY,
        self_weight: float = DEFAULT_SELF_WEIGHT,
        weight_model: Any = None,
        cache_candidates: bool = False,
    ) -> None:
        self.weight_profile = weight_profile
        if price_policy not in PRICE_POLICIES:
            raise ValueError(
                f"price_policy must be one of {PRICE_POLICIES}, got {price_policy!r}"
            )
        self.price_policy = price_policy
        self.max_hops = int(max_hops)
        self.hop_decay = float(hop_decay)
        # self_weight = 1.0 disables diffusion — the ablation baseline.
        self.self_weight = min(1.0, max(0.0, float(self_weight)))
        # Optional learned weight model (src/graph/weight_policy.py). When set it
        # supersedes the static profile + hand-written intent rules; duck-typed
        # on `.predict(context) -> ScoringWeights` so tests can pass a stub.
        self.weight_model = weight_model
        # Only ever True inside an evaluation run — see _CANDIDATE_CACHE.
        self.cache_candidates = bool(cache_candidates)
        self._query = _candidate_query(self.max_hops)

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

        # Weight resolution, in precedence order:
        #   1. an explicit UserProfile (personalised request) — always wins
        #   2. a learned weight model, if one was supplied
        #   3. the hand-written intent rules over the configured profile
        if profile is not None:
            weights = weights_for_profile(profile, intent, event_active=event is not None,
                                          weight_profile=self.weight_profile)
        elif self.weight_model is not None:
            # Imported lazily: src.graph.weight_policy imports this module, so a
            # module-level import here would be circular.
            from src.graph.weight_policy import PoolConditions
            weights = self.weight_model.predict(
                intent, PoolConditions.from_candidates(candidates)
            ).normalised()
        else:
            weights = weights_for_intent(intent, self.weight_profile)

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
        key = (city.lower(), self.max_hops, self.hop_decay)
        if self.cache_candidates and key in _CANDIDATE_CACHE:
            # Deep-copied on the way out: _score writes provenance keys
            # (price_imputed, diffusion, ...) onto the candidate dicts, and a
            # shared mutable cache entry would leak one system's annotations
            # into the next system's inputs.
            return [dict(c) for c in _CANDIDATE_CACHE[key]]

        driver = _get_driver()
        with driver.session() as session:
            rows = session.run(
                self._query, {"city": city, "hop_decay": self.hop_decay}
            ).data()

        if self.cache_candidates:
            _CANDIDATE_CACHE[key] = [dict(r) for r in rows]
        return rows

    # -- filtering ----------------------------------------------------------

    def _apply_filters(self, cands: List[Dict[str, Any]], intent: QueryIntent) -> List[Dict[str, Any]]:
        """Feasibility-first pruning of hard constraints.

        A hotel with an UNKNOWN value for a constrained attribute is excluded:
        feasibility requires positive evidence (recommending a hotel with no
        price for an "under 25,000" query is a guess, not a recommendation).
        Hotels the constraints don't mention are untouched.

        Under the `exclude` price policy, unpriced hotels are dropped whether or
        not the query mentions price — that policy's premise is that a candidate
        without price evidence should never be ranked at all."""
        out = []
        for c in cands:
            rating = c.get("rating")
            star = c.get("star")
            price = c.get("price")

            if self.price_policy == "exclude" and not price:
                continue
            if intent.min_rating is not None and (rating is None or float(rating) < intent.min_rating):
                continue
            if intent.min_star is not None and (star is None or float(star) < intent.min_star):
                continue
            if intent.max_price_lkr is not None and (not price or float(price) > intent.max_price_lkr):
                continue
            if intent.min_price_lkr is not None and (not price or float(price) < intent.min_price_lkr):
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

        # Missing-price handling — see PRICE_POLICIES.
        price_median = _median([c.get("price") for c in cands])
        n_missing_price = sum(1 for c in cands if not c.get("price"))
        if n_missing_price and self.price_policy == "neutral":
            logger.debug(
                "%d/%d candidates have no price and score a neutral 0.5 economic "
                "(policy=neutral). Sweep other policies before publishing.",
                n_missing_price, len(cands),
            )

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
            price_imputed = False
            if c.get("price"):
                economic = _norm_lower_better(c.get("price"), price_lo, price_hi)
                if intent.max_price_lkr and float(c["price"]) <= intent.max_price_lkr:
                    reasons.append("within budget")
            else:
                price_imputed = True
                if self.price_policy == "worst":
                    economic = 0.0
                elif self.price_policy == "median" and price_median is not None:
                    economic = _norm_lower_better(price_median, price_lo, price_hi)
                else:  # "neutral", or "median" with an all-null pool
                    economic = 0.5
                reasons.append(f"price unknown (imputed: {self.price_policy})")

            # --- disruption (own signals + diffused neighbourhood exposure) ---
            #
            # Exposure has two sources, blended by self_weight:
            #
            #   own           — this hotel's route signal and linked events
            #   neighbourhood — the inverse-distance, hop-attenuated weighted
            #                   mean over hotels reachable across NEAR_HOTEL,
            #                   computed in Cypher (see _candidate_query)
            #
            # The neighbourhood term is what makes this graph retrieval rather
            # than a star-join: a hotel with no signal of its own still gets a
            # non-trivial, hotel-specific exposure estimate, and two hotels with
            # identical own-signals separate on the state of their surroundings.
            # Setting self_weight = 1.0 removes the term entirely and is the
            # ablation to report against.
            sev = [s for s in (c.get("signal_severities") or []) if s]
            heavy = sum(1 for s in sev if s == "heavy")
            moderate = sum(1 for s in sev if s == "moderate")
            events = int(c.get("event_count") or 0)
            etas = [float(e) for e in (c.get("signal_etas") or []) if e]
            max_eta = max(etas) if etas else 0.0
            event_impact = float(c.get("event_impact") or 0.0)

            own_exposure = (
                0.40 * min(heavy, 1) + 0.20 * min(moderate, 1)
                + min(max_eta / 20.0, 0.5)          # +10 min route delay -> 0.5
                + 0.30 * min(event_impact, 1.0)
            )

            nbr_eta = float(c.get("nbr_eta") or 0.0)
            nbr_sev = float(c.get("nbr_severity") or 0.0)
            nbr_event = float(c.get("nbr_event_impact") or 0.0)
            nbr_count = int(c.get("nbr_count") or 0)
            nbr_exposure = (
                0.40 * nbr_sev
                + min(nbr_eta / 20.0, 0.5)
                + 0.30 * min(nbr_event, 1.0)
            )

            # No neighbours (isolated node) -> fall back to own exposure rather
            # than crediting the hotel with a spuriously calm neighbourhood.
            sw = self.self_weight if nbr_count else 1.0
            exposure = sw * own_exposure + (1.0 - sw) * nbr_exposure
            disruption = max(0.0, min(1.0, 1.0 - exposure))

            if heavy:
                reasons.append(f"heavy traffic on route (x{heavy})")
            elif max_eta >= 3:
                reasons.append(f"+{max_eta:.0f} min traffic delay on route")
            if event_impact > 0.05:
                reasons.append(f"near a live event (impact {event_impact:.2f})")
            if nbr_count and nbr_exposure > own_exposure + 0.05:
                reasons.append(
                    f"congested neighbourhood ({nbr_count} nearby hotels affected)"
                )
            elif nbr_count and nbr_exposure < 0.05 and own_exposure < 0.05:
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
            # Provenance for the explanation panel and the evaluation harness:
            # how the disruption score was arrived at, and whether the economic
            # score rests on a real price or an imputed one.
            c["price_imputed"] = price_imputed
            c["max_eta_change_min"] = max_eta
            c["own_exposure"] = round(own_exposure, 4)
            c["nbr_exposure"] = round(nbr_exposure, 4)
            c["diffusion"] = {
                "self_weight": round(sw, 3),
                "neighbours": nbr_count,
                "own_exposure": round(own_exposure, 4),
                "neighbourhood_exposure": round(nbr_exposure, 4),
                "blended_exposure": round(exposure, 4),
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
