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
# `handset` is the original hand-tuned prior. `elicited` and `blended` are fitted
# from the discrete-choice experiment in studies/weight-elicitation (material
# v4-rooms-20260818) by `weight_elicitation/fit_weights.py`, which reads the hosted
# study's raw JSON dump directly. Primary specification: all 2,232 choice sets
# from 241 participants (32-hotel pool), conditional logit, bootstrapped by
# participant. The attention check fails for 43.7% of participants, so excluding
# them is a judgement call made after seeing the data — the all-cohort fit is
# therefore primary and the clean cohort is a sensitivity analysis
# (`--drop-failed-attention`), not the reverse.
# Full provenance: docs/Weight_Elicitation_Data_Audit.md
#
# Two corrections separate these numbers from the earlier published ones
# (0.563 / 0.401 / 0 / 0 / 0.036):
#
#   1. A log-rank term is fitted alongside the components and then discarded.
#      46% of participants picked the hotel at rank 1 on a list sorted by one of
#      the components being estimated; rank correlates +0.686 with `spatial` and
#      +0.670 with `accessibility`. Without that term the logit books position
#      bias as a preference for proximity. Adding it improves the training
#      log-likelihood by 1,702 on 1 df — position explains more than all five
#      components together.
#   2. `facility` is recomputed without the `min(n_facilities / 40, 1)` ceiling
#      that had saturated 30 of the 32 hotels. (Component vectors were never
#      shown to participants, so rebuilding the feature matrix invalidates no
#      choice — only our description of the alternatives changes.)
#
# Held-out (72 participants never seen in fitting), macro-averaged over sort
# mode so the 75% proximity-sorted majority cannot win on position bias alone:
#
#     handset   nDCG@10 0.325      elicited  0.394      blended  0.380
#
# Bootstrap 95% CIs on `elicited` (400 replicates, resampled by participant):
#
#     spatial       0.381  [0.293, 0.433]
#     accessibility 0.472  [0.293, 0.614]
#     facility      0.097  [0.000, 0.238]   NOT distinguishable from zero
#     economic      0.000  [0.000, 0.107]   NOT distinguishable from zero
#     disruption    0.050  [0.000, 0.136]   NOT distinguishable from zero
#
#     spatial + accessibility  0.816  [0.582, 1.000]  <- the identified estimand
#     spatial share of that    0.447  [0.379, 0.504]
#
# The TOTAL location weight is what this design identifies. Because spatial and
# accessibility correlate 0.907 in Colombo, how that mass divides between them is
# far weaker evidence than the two separate numbers suggest — do not report
# "travel time beats proximity" off this fit.
#
# ADDENDUM 2026-09-02 — the split is a PRIOR DIAL, not an estimate.
#
# The bracket above is a bootstrap of the SHRUNKEN (MAP) estimator, so it
# inherits the hand-set prior's shape (which sits at spatial share 0.556). The
# unregularised profile likelihood — fix the split, re-optimise location mass,
# facility, economic, disruption and the rank term at each point — says
# something different and much sharper:
#
#     spatial share s   0.000  0.100  0.200  0.447  0.500  1.000
#     -logL            3105.8 3106.5 3107.4 3110.0 3110.7 3117.7
#     vs best             .00    .72   1.58   4.25   4.91  11.89
#
# A likelihood-ratio test rejects a split costing more than 1.92. Only
# s in [0.000, 0.200] survives: the data wants travel time and essentially
# nothing else, and it rejects BOTH the shipped 0.447 and an even 0.500.
#
# So lambda is what picks the split, not the choice data: lambda = 0 gives
# 0.00 / 0.45, lambda -> infinity gives the hand-set 0.25 / 0.20, and the
# lambda = 250 chosen on validation gives 0.201 / 0.249. The direction is
# robust (every estimator puts accessibility >= spatial); the magnitude is not.
#
# Variance inflation on the study's feature matrix, for the record:
#
#     spatial 5.43   accessibility 5.46   facility 2.30   economic 2.38
#     disruption 1.18
#
# None of this reaches the served ranking. Swapping the two weights
# (0.249/0.201) or setting them equal (0.225/0.225) changes the top-10 on
# 0 of 60 benchmark queries and reorders none of them, because at serving time
# the two components correlate +0.966 — distance and travel time come out of
# the same Google Distance Matrix element, and the current traffic snapshot was
# taken near midnight (mean delay -1.58 min, 54 of 58 hotels FASTER "in
# traffic"), so accessibility is road distance divided by a near-constant
# 24.5 km/h. Moving the location SUM does matter: 0.45 -> 0.30 changes 13 of 60.
#
# Write-up guidance: report ONE location weight and declare the split as an
# assumption. A claim about spatial versus accessibility is not supportable from
# this design in either direction. Facility, economic and disruption
# all have CIs including zero, and that holds under all four candidate `facility`
# definitions tested.
#
# Read `elicited` as "this booking task, on a sorted list, measured location and
# little else" — not as evidence that price and disruption do not matter.
# Participants opened a median of 1 hotel out of 32, so no comparison ever
# happened for a trade-off model to read.
#
# `blended` is therefore the deployable profile and a stated product decision,
# not a statistical one: an equal mixture of the elicited posterior and the
# hand-set prior. A retriever with economic = 0 cannot answer "cheapest hotel
# near Galle Face" and one with disruption = 0 discards the thesis's whole
# contribution. The study measured booking behaviour on a sorted list; it never
# tested constraint satisfaction, so it does not get to zero out a capability it
# did not measure.
#
# Which profile to run is an empirical question — set SCORING_WEIGHTS_PROFILE
# and compare in evaluation/run_eval.py.

HANDSET_WEIGHTS = ScoringWeights(
    spatial=0.25, accessibility=0.20, facility=0.25, economic=0.15, disruption=0.15,
)

ELICITED_WEIGHTS = ScoringWeights(
    spatial=0.381, accessibility=0.472, facility=0.097, economic=0.000, disruption=0.050,
)

BLENDED_WEIGHTS = ScoringWeights(
    spatial=0.316, accessibility=0.336, facility=0.173, economic=0.075, disruption=0.100,
)

# `balanced` — the profile for real-world use, and the answer to "why is
# economic only 0.075?".
#
# The study cannot price. Four reasons, each measured on the collected data
# (weight_elicitation diagnostics, 2026-09-02), none of them "travellers do not
# care what a room costs":
#
#   1. COLLINEARITY. facility and economic correlate -0.745, so only their
#      DIFFERENCE is identified. Unconstrained that difference is
#      beta_fac - beta_econ = -1.120 - (-1.015) = -0.105, i.e. ~0. Which of the
#      two got 0.097 and which got 0.000 is where the non-negativity bound
#      happened to land — it is not evidence about price.
#
#   2. THE NUISANCE TERM EATS THE SIGNAL. -log(rank) exists to absorb position,
#      but on a price-sorted list position IS price: corr(economic, -log rank)
#      = +0.919 on price_asc sets, against +0.094 on distance-sorted ones. Fit
#      those 188 sets on their own and economic comes back POSITIVE (+0.580)
#      with rank still controlled.
#
#   3. THE POOLED FIT IS THE WRONG AVERAGE. 75% of sets were proximity-sorted,
#      where price is simply not what the participant is doing. Those sets
#      decide the pooled coefficient.
#
#   4. A COMPENSATORY MODEL ON NONCOMPENSATORY BEHAVIOUR. Conditional logit
#      assumes trade-offs, but 99.9% of participants opened <= 1 hotel out of
#      32 — no trade-off ever occurred. The literature on hotel search
#      ("Determinants of consumers' choices in hotel online searches: a
#      comparison of consideration and booking stages", Int. J. Hospitality
#      Management, 2019) finds shoppers screen noncompensatorily while building
#      a consideration set and only trade attributes off at the booking stage.
#      Here price was expressed by CHOOSING TO SORT BY IT — 38.6% of
#      participants did so at least once — and the fit treats sort mode as a
#      stratification variable, not as a preference.
#
# `economic ~ 0` is therefore a fact about the instrument, not about travellers.
#
# Booking-stage conjoint studies, which show attributes side by side and so do
# elicit real trade-offs, put price at 16.5% relative importance, hotel rating
# at 16.3% and location at 15.3% — within two points of each other (Assaker &
# O'Connor, "The Importance of Green Certification Labels/Badges in Online Hotel
# Booking Choice", J. Hospitality & Tourism Research, 2023). Hotel conjoints
# elsewhere put price as high as 26%.
#
# `balanced` is FITTED, not chosen, and fitted against the RANKING THE
# RETRIEVER ACTUALLY PRODUCES. `scripts/fit_weight_profile.py` learns it:
#
#     w* = argmax_{w in simplex}  mean_q nDCG@10( rank(C_q . f(w, intent_q)), gold_q )
#
# where f is `apply_intent_adjustments` — the same intent ladder `retrieve()`
# applies. That detail is the difference between a fitted number and a fiction:
# an earlier version optimised the raw composite score instead, and its
# objective disagreed with the evaluation harness by up to 0.118 nDCG. Ranking
# through the serving path, the fitting environment now reproduces
# evaluation/run_eval.py exactly (+0.0000 on all three profiles on the main set,
# within 0.003 on the price slice).
#
# Search: seeded Dirichlet sampling over the simplex plus deterministic
# coordinate refinement — nDCG is piecewise constant in w, so derivative-free is
# a necessity. Nested 5-fold CV, inner split for the prior-mixing coefficient,
# every seed fixed and the parser's intents frozen to disk:
#
#     nested CV nDCG@10 = 0.8402 +/- 0.0292     (the honest performance claim)
#     lambda (prior mixing) = 0.00 in all 5 folds
#
# Rerun and it returns this vector bit for bit:
#
#     python scripts/fit_weight_profile.py --check-reproducible --emit-profile
#
# Two dimensions are not left to the fit, and the script prints the measurement
# behind each rather than asserting it:
#
#   disruption 0.150 is RESERVED at the hand-set value. Its evaluation category
#     is saturated at nDCG 1.000 for every weight vector tested, so the query set
#     prices the dimension's COST and never its BENEFIT; freed, the fit drove it
#     to 0.052. That is the instrument reporting its blind spot, not a finding
#     about congestion. Re-fit it once there are queries where disruption
#     avoidance actually discriminates.
#
#   spatial / accessibility: the fit divides the location mass almost arbitrarily
#     because the two components correlate +0.966 at serving time (distance and
#     travel time come from the same Distance Matrix element, and the traffic
#     snapshot is a near-constant 24.5 km/h). Across folds the split wandered
#     from 0.000 to 0.108 accessibility while the TOTAL stayed near 0.17. So the
#     total is taken from the data and the split from the study's 0.447/0.553
#     ratio — the script verifies that substitution is free before applying it
#     (measured delta +0.0001 here) and keeps the fitted split when it is not.
#
# What the fit landed on:
#
#   location total  0.174   spatial 0.078 + accessibility 0.096
#   facility        0.365   above the literature's 0.288. The weakest number
#                           here: most of its support is the amenity category,
#                           whose vocabulary is four values, two of them present
#                           on all 58 hotels.
#   economic        0.311   close to the literature's renormalised 0.291, and the
#                           dimension the price slice exists to make visible
#   disruption      0.150   reserved, see above
#
# SCALE — fixed 2026-09-02. Every profile here was fitted on PERCENTILE
# features, and until that date the retriever scored MIN-MAX ones, so no weight
# in this module delivered its nominal share at serving time (`economic` worst:
# sd 0.289 -> 0.206, mean 0.484 -> 0.744 on the study pool). Scoring is now
# percentile throughout — see _pct_lower_better. Retrieval numbers published
# before that date were produced on the old scale and are not comparable.

BALANCED_WEIGHTS = ScoringWeights(
    spatial=0.078, accessibility=0.096, facility=0.365, economic=0.311, disruption=0.150,
)

# `human` — fitted from the discrete-choice study and NOTHING ELSE. No retrieval
# benchmark, no conjoint literature, no hand-set prior.
#
#     python -m weight_elicitation.fit_human_weights --emit-profile
#
# The estimator is a non-negative conditional logit with a log-rank nuisance
# term, fitted SEPARATELY PER DISPLAY CONDITION and then macro-averaged, so no
# sort mode can win by being popular. That is the whole difference from
# `elicited`, which pools every set into one fit and is therefore decided by the
# 75% of sets left sorted by distance or travel time — the lists ordered by the
# very components being estimated. Per stratum:
#
#     distance n=986   spa .531  acc .462  fac .000  eco .007  dis .000
#     travel   n=696   spa .260  acc .527  fac .000  eco .011  dis .201
#     rating   n=278   spa .000  acc .000  fac 1.000 eco .000  dis .000
#     price    n=272   spa .401  acc .137  fac .000  eco .462  dis .000
#
# Sort mode is a participant variable, not a design variable: the frozen
# material assigns no sort, `final_sort` varies inside every task, and it tracks
# the scenario framing (the economic-framed tasks draw 64 and 50 price-sorts
# against 11 for a proximity-framed one).
#
# The result that matters: `economic` = 0.120 with a clustered bootstrap CI of
# [0.049, 0.256] — it EXCLUDES ZERO. The pooled fit reported 0.000 with a CI
# spanning zero, and that was an artefact of averaging over lists where price
# was not what the participant was doing. Every dimension's CI now excludes
# zero, and held-out macro nDCG@10 is 0.4001 against 0.394 for `elicited`,
# 0.380 for `blended` and 0.325 for `handset` on the study's own metric.
#
# Two honest caveats:
#   * `facility` = 0.250 is mechanical, not graded: the rating-sorted stratum
#     returns a degenerate 1.000 and the other three return 0.000, so the value
#     is exactly one quarter of one vote. Its bootstrap CI [0.250, 0.253] is
#     tight for that reason and should not be read as precision.
#   * Participants opened a median of ONE hotel of 32, so every number here is a
#     consideration-stage quantity. `spatial` and `accessibility` correlate
#     +0.898 in the material; their TOTAL (0.580) is the estimand, not the split.

# ECONOMIC IS PINNED AT 0.200 AND WAS NOT ESTIMATED.
#
#   python -m weight_elicitation.fit_human_weights --reserve economic=0.20 #          --emit-profile
#
# The study's own point estimate is 0.120. 0.200 is a declared constraint. It is
# defensible — it sits inside the clustered bootstrap CI [0.049, 0.256], so the
# data does not reject it, and it matches booking-stage conjoint work once the
# attributes this retriever does not model are removed — but it is not something
# this study found, and it costs 0.019 on the study's own held-out metric
# (macro nDCG@10 0.4001 unconstrained -> 0.3810 constrained). The remaining four
# dimensions are rescaled by a single factor, so every ratio the data DID
# establish among them is preserved exactly.
#
# Unconstrained macro-average, for the record:
#     spatial .298  accessibility .282  facility .250  economic .120  disruption .050

HUMAN_WEIGHTS = ScoringWeights(
    spatial=0.271, accessibility=0.256, facility=0.227, economic=0.200, disruption=0.046,
)

WEIGHT_PROFILES: Dict[str, ScoringWeights] = {
    "handset": HANDSET_WEIGHTS,
    "elicited": ELICITED_WEIGHTS,
    "blended": BLENDED_WEIGHTS,
    "balanced": BALANCED_WEIGHTS,
    "human": HUMAN_WEIGHTS,
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


def apply_intent_adjustments(base: ScoringWeights, intent: QueryIntent) -> ScoringWeights:
    """Apply the intent ladder to an ALREADY-CHOSEN base vector.

    Split out of `weights_for_intent` so that the weight fitter
    (scripts/fit_weight_profile.py) can rank exactly the way serving does. The
    fitter used to optimise the base vector against the raw composite score,
    which is not the function the retriever actually applies — the ladder below
    is worth roughly 0.03 nDCG on the benchmark, and a base vector tuned without
    it is tuned for a scorer that never runs.

    This is deliberately the ONLY place these constants live. Training and
    serving must not be able to drift apart.
    """
    w = ScoringWeights(
        spatial=base.spatial, accessibility=base.accessibility,
        facility=base.facility, economic=base.economic,
        disruption=base.disruption, event=base.event,
    )

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


def weights_for_intent(intent: QueryIntent, profile: Optional[str] = None) -> ScoringWeights:
    """Derive dynamic scoring weights from query intent (P3 personalisation hook)."""
    return apply_intent_adjustments(base_weights(profile), intent)


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


def _pct_rank(v: float, pool: List[float]) -> float:
    """Midrank percentile of `v` within `pool`, in [0, 1].

    This is the scale the discrete-choice study used for every component it
    could rank (`1 - pct_rank(price)`, `1 - pct_rank(distance)`, ...), so
    scoring the same way here is what lets a fitted weight mean at serving time
    what it meant in the fit.

    The study's own helper counted `v_i <= x`; this uses the midrank
    `(#below + 0.5 * #equal) / n`. The two differ by a constant 0.5/n when all
    values are distinct, and an additive constant applied to every candidate
    cannot reorder them — so this reproduces the study's ranking while
    degrading sanely on ties. An all-equal pool returns 0.5 for everyone
    (nothing to separate) instead of collapsing to 0.0.
    """
    n = len(pool)
    if n == 0:
        return 0.5
    below = sum(1 for x in pool if x < v)
    equal = sum(1 for x in pool if x == v)
    return (below + 0.5 * equal) / n


def _pct_lower_better(v: Optional[float], pool: List[float],
                      default: float = 0.5) -> float:
    """Lower raw value -> higher score, on the percentile scale.

    Percentile replaces the min-max normalisation this retriever used until
    2026-09-02. Min-max was measurably the wrong scale: on the study's own
    32-hotel pool it compressed `economic`'s spread from sd 0.289 to 0.206 and
    pushed its mean from 0.484 to 0.744, so 47% of hotels scored above 0.8
    against 19% in the study. A couple of luxury properties stretch `hi` and
    everything below the median piles up near 1.0, where it can no longer
    separate the hotels users are actually choosing between. Price is the most
    skewed input in the pool and took the worst of it, which is most of why
    `economic` looked inert at serving time whatever weight it was given.
    """
    if v is None:
        return default
    return 1.0 - _pct_rank(float(v), pool)


def _pct_higher_better(v: Optional[float], pool: List[float],
                       default: float = 0.5) -> float:
    """Higher raw value -> higher score, on the percentile scale."""
    if v is None:
        return default
    return _pct_rank(float(v), pool)


def _norm_lower_better(v: Optional[float], lo: float, hi: float, default: float = 0.5) -> float:
    """Lower raw value -> higher score, min-max scaled.

    RETAINED for the `event` proximity term and for callers that genuinely want
    an absolute 0-1 range. The five composite-score components no longer use it
    — see `_pct_lower_better` for why they moved to percentiles.
    """
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
        # Percentile pools for normalisation across the candidate set.
        #
        # These are the empirical distributions each component is ranked
        # against, and they are the serving-time counterpart of the study's
        # 32-hotel pool. Nulls are excluded rather than imputed into the pool:
        # a hotel with no price should not shift where the priced ones rank.
        dist_pool = [float(c["distance_km"]) for c in cands
                     if c.get("distance_km") is not None]
        tt_pool = [float(c.get("travel_time_traffic_min") or c.get("travel_time_min"))
                   for c in cands
                   if (c.get("travel_time_traffic_min") or c.get("travel_time_min")) is not None]
        price_pool = [float(c["price"]) for c in cands if c.get("price")]
        star_pool = [float(c["star"]) for c in cands if c.get("star")]
        rating_pool = [float(c["rating"]) for c in cands if c.get("rating")]
        amen_counts = [len(c.get("amenities") or []) for c in cands]
        amen_pool = [float(n) for n in amen_counts]
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
            spatial = _pct_lower_better(c.get("distance_km"), dist_pool)
            if intent.proximity_preference == "far":
                spatial = 1.0 - spatial  # quiet seeker wants distance from centre
            elif intent.proximity_preference == "close" and c.get("distance_km") is not None:
                if spatial > 0.7:
                    reasons.append("central / walkable location")

            # --- accessibility (uses live traffic travel time) -------------
            tt = c.get("travel_time_traffic_min") or c.get("travel_time_min")
            accessibility = _pct_lower_better(tt, tt_pool)
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
                # No requested amenities: fall back to "how well equipped is
                # this hotel relative to the pool". The study ranked its
                # facility count the same way (pct_rank(n_facilities)); the old
                # count/max_count divided by a single best-equipped outlier.
                amen_match = _pct_higher_better(float(len(amenities)), amen_pool)
            attr_match = 0.0
            if req_attr:
                am = sum(1 for a in req_attr if any(a in x for x in attractions | set(c.get("locations") or [])))
                attr_match = am / len(req_attr)
                if am:
                    reasons.append(f"near {am}/{len(req_attr)} requested attractions")
            # Percentile, not value/5 — matching the study's
            # 0.45*pct_rank(star) + 0.35*pct_rank(n_facilities) + 0.20*pct_rank(rating).
            # Colombo hotels cluster at 3-4 stars and 4.0-4.5 rating, so /5
            # squeezed both into a narrow band and left `facility` unable to
            # separate anything.
            star_score = _pct_higher_better(c.get("star"), star_pool, default=0.0)
            rating_score = _pct_higher_better(c.get("rating"), rating_pool, default=0.0)
            facility = (
                0.40 * amen_match
                + 0.20 * attr_match
                + 0.20 * rating_score
                + 0.20 * star_score
            )

            # --- economic --------------------------------------------------
            price_imputed = False
            if c.get("price"):
                economic = _pct_lower_better(c.get("price"), price_pool)
                if intent.max_price_lkr and float(c["price"]) <= intent.max_price_lkr:
                    reasons.append("within budget")
            else:
                price_imputed = True
                if self.price_policy == "worst":
                    economic = 0.0
                elif self.price_policy == "median" and price_median is not None:
                    economic = _pct_lower_better(price_median, price_pool)
                else:  # "neutral", or "median" with an all-null pool
                    economic = 0.5
                reasons.append(f"price unknown (imputed: {self.price_policy})")

            # `economic` is defined cheaper-is-better, so a premium query would
            # otherwise be served its cheapest hotels first — and the larger the
            # economic weight, the harder it pulls the wrong way. Flip the
            # component rather than zero the weight, exactly as
            # proximity_preference="far" flips `spatial` above: the dimension
            # still carries its share of the score, it just points the other way.
            if intent.price_preference == "high":
                economic = 1.0 - economic
                if economic > 0.7:
                    reasons.append("upmarket property")

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
