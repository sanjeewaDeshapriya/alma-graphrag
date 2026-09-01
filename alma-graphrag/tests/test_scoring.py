"""Tests for WeightedRetriever._score — the composite scorer itself.

This function had 0% coverage while being the object of the whole thesis: every
reported nDCG is a consequence of it, and a refactor could invert a sub-score
without a single test failing. These tests pin the behaviour that the results
depend on.

They never touch Neo4j. `_score` operates on plain candidate dicts, so a
hand-built pool is both sufficient and preferable — the arithmetic is checked
against values chosen to make the expected answer obvious.
"""
from __future__ import annotations

import pytest

from src.crag.query_parser import QueryIntent
from src.graph.retriever import (
    PRICE_POLICIES,
    ScoringWeights,
    WeightedRetriever,
    _median,
    _minmax,
    _norm_lower_better,
)


def hotel(hid: str, **kw):
    """Candidate dict shaped like a row from the multi-hop Cypher query."""
    base = {
        "id": hid, "name": f"Hotel {hid}",
        "rating": 4.0, "star": 4, "price": 20000.0,
        "distance_km": 3.0, "travel_time_min": 20.0,
        "travel_time_traffic_min": 22.0,
        "lat": 6.92, "lng": 79.86,
        "amenities": ["pool", "wifi"], "attractions": [], "locations": [],
        "signal_severities": [], "signal_etas": [], "event_count": 0,
        "event_impact": 0.0, "event_distance_km": None,
        "nbr_eta": 0.0, "nbr_severity": 0.0, "nbr_event_impact": 0.0,
        "nbr_count": 0,
    }
    base.update(kw)
    return base


EQUAL_WEIGHTS = ScoringWeights(
    spatial=0.2, accessibility=0.2, facility=0.2, economic=0.2, disruption=0.2
).normalised()


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def test_minmax_ignores_none():
    assert _minmax([None, 5.0, None, 1.0]) == (1.0, 5.0)


def test_minmax_all_none_is_zero_range():
    assert _minmax([None, None]) == (0.0, 0.0)


def test_norm_lower_better_inverts():
    # Lowest raw value scores 1.0, highest scores 0.0.
    assert _norm_lower_better(1.0, 1.0, 5.0) == pytest.approx(1.0)
    assert _norm_lower_better(5.0, 1.0, 5.0) == pytest.approx(0.0)
    assert _norm_lower_better(3.0, 1.0, 5.0) == pytest.approx(0.5)


def test_norm_lower_better_degenerate_range():
    # Every candidate identical -> nothing to discriminate -> full credit.
    assert _norm_lower_better(4.0, 4.0, 4.0) == pytest.approx(1.0)


def test_median_ignores_none():
    assert _median([None, 1.0, 3.0, None]) == pytest.approx(2.0)
    assert _median([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert _median([None, None]) is None


# ---------------------------------------------------------------------------
# Direction of each sub-score
# ---------------------------------------------------------------------------

def test_cheaper_hotel_scores_higher_economic():
    r = WeightedRetriever()
    cands = [hotel("cheap", price=5000.0), hotel("dear", price=50000.0)]
    scored = {h.id: h for h in r._score(cands, QueryIntent(), EQUAL_WEIGHTS)}
    assert scored["cheap"].components["economic"] > scored["dear"].components["economic"]


def test_closer_hotel_scores_higher_spatial():
    r = WeightedRetriever()
    cands = [hotel("near", distance_km=0.5), hotel("far", distance_km=12.0)]
    scored = {h.id: h for h in r._score(cands, QueryIntent(), EQUAL_WEIGHTS)}
    assert scored["near"].components["spatial"] > scored["far"].components["spatial"]


def test_far_proximity_preference_inverts_spatial():
    """A quiet seeker wants distance from the centre, not proximity to it."""
    r = WeightedRetriever()
    cands = [hotel("near", distance_km=0.5), hotel("far", distance_km=12.0)]
    intent = QueryIntent(proximity_preference="far")
    scored = {h.id: h for h in r._score(cands, intent, EQUAL_WEIGHTS)}
    assert scored["far"].components["spatial"] > scored["near"].components["spatial"]


def test_faster_travel_scores_higher_accessibility():
    r = WeightedRetriever()
    cands = [hotel("quick", travel_time_traffic_min=5.0),
             hotel("slow", travel_time_traffic_min=60.0)]
    scored = {h.id: h for h in r._score(cands, QueryIntent(), EQUAL_WEIGHTS)}
    assert (scored["quick"].components["accessibility"]
            > scored["slow"].components["accessibility"])


def test_requested_amenity_match_raises_facility():
    r = WeightedRetriever()
    cands = [hotel("has", amenities=["swimming pool", "gym"]),
             hotel("lacks", amenities=["parking"])]
    intent = QueryIntent(required_amenities=["pool"])
    scored = {h.id: h for h in r._score(cands, intent, EQUAL_WEIGHTS)}
    assert scored["has"].components["facility"] > scored["lacks"].components["facility"]


# ---------------------------------------------------------------------------
# Composite arithmetic
# ---------------------------------------------------------------------------

def test_score_is_weighted_sum_of_components():
    r = WeightedRetriever()
    cands = [hotel("a"), hotel("b", price=40000.0, distance_km=8.0)]
    w = ScoringWeights(spatial=0.3, accessibility=0.1, facility=0.2,
                       economic=0.3, disruption=0.1)
    for h in r._score(cands, QueryIntent(), w):
        expected = sum(
            h.components[c] * getattr(w, c)
            for c in ("spatial", "accessibility", "facility", "economic", "disruption")
        )
        assert h.score == pytest.approx(expected, abs=1e-3)


def test_weighted_components_sum_to_score():
    r = WeightedRetriever()
    for h in r._score([hotel("a"), hotel("b", price=9000.0)],
                      QueryIntent(), EQUAL_WEIGHTS):
        assert sum(h.weighted_components.values()) == pytest.approx(h.score, abs=1e-3)


def test_zero_weight_component_cannot_affect_score():
    """A component with weight 0 must contribute exactly nothing.

    This is what makes the `elicited` profile (facility = economic = 0) safe to
    deploy: those sub-scores are still computed and displayed, but must not
    leak into the ranking.
    """
    r = WeightedRetriever()
    w = ScoringWeights(spatial=1.0, accessibility=0.0, facility=0.0,
                       economic=0.0, disruption=0.0)
    cheap = r._score([hotel("a", price=1.0), hotel("b", price=99999.0)],
                     QueryIntent(), w)
    dear = r._score([hotel("a", price=99999.0), hotel("b", price=1.0)],
                    QueryIntent(), w)
    assert {h.id: h.score for h in cheap} == {h.id: h.score for h in dear}


def test_components_stay_in_unit_interval():
    r = WeightedRetriever()
    cands = [
        hotel("a", price=None, rating=None, star=None, distance_km=None),
        hotel("b", price=10 ** 9, signal_severities=["heavy"], signal_etas=[120.0]),
        hotel("c", event_impact=1.0, nbr_eta=99.0, nbr_severity=1.0, nbr_count=5),
    ]
    for h in r._score(cands, QueryIntent(), EQUAL_WEIGHTS):
        for name, v in h.components.items():
            assert 0.0 <= v <= 1.0, f"{h.id}.{name} = {v} out of range"


# ---------------------------------------------------------------------------
# Missing-price policy
# ---------------------------------------------------------------------------

def test_all_price_policies_are_constructible():
    for p in PRICE_POLICIES:
        WeightedRetriever(price_policy=p)


def test_unknown_price_policy_rejected():
    with pytest.raises(ValueError):
        WeightedRetriever(price_policy="wishful")


def test_neutral_policy_scores_unknown_price_at_half():
    r = WeightedRetriever(price_policy="neutral")
    cands = [hotel("known", price=20000.0), hotel("unknown", price=None)]
    scored = {h.id: h for h in r._score(cands, QueryIntent(), EQUAL_WEIGHTS)}
    assert scored["unknown"].components["economic"] == pytest.approx(0.5)


def test_worst_policy_gives_unknown_price_no_credit():
    r = WeightedRetriever(price_policy="worst")
    cands = [hotel("known", price=20000.0), hotel("unknown", price=None)]
    scored = {h.id: h for h in r._score(cands, QueryIntent(), EQUAL_WEIGHTS)}
    assert scored["unknown"].components["economic"] == pytest.approx(0.0)


def test_neutral_policy_can_rank_unknown_above_expensive():
    """The precise reason the default needs documenting and sweeping.

    Under `neutral`, a hotel with NO price outranks a genuinely expensive one on
    the economic component — missing data is rewarded. `worst` removes that.
    """
    cheap, dear, unknown = (hotel("cheap", price=1000.0),
                            hotel("dear", price=100000.0),
                            hotel("unknown", price=None))
    neutral = {h.id: h for h in WeightedRetriever(price_policy="neutral")
               ._score([cheap, dear, unknown], QueryIntent(), EQUAL_WEIGHTS)}
    assert (neutral["unknown"].components["economic"]
            > neutral["dear"].components["economic"])

    worst = {h.id: h for h in WeightedRetriever(price_policy="worst")
             ._score([cheap, dear, unknown], QueryIntent(), EQUAL_WEIGHTS)}
    assert (worst["unknown"].components["economic"]
            <= worst["dear"].components["economic"])


def test_median_policy_lands_between_extremes():
    r = WeightedRetriever(price_policy="median")
    cands = [hotel("a", price=1000.0), hotel("b", price=100000.0),
             hotel("u", price=None)]
    scored = {h.id: h for h in r._score(cands, QueryIntent(), EQUAL_WEIGHTS)}
    econ = scored["u"].components["economic"]
    assert scored["b"].components["economic"] <= econ <= scored["a"].components["economic"]


def test_exclude_policy_drops_unpriced_in_filters():
    r = WeightedRetriever(price_policy="exclude")
    cands = [hotel("priced", price=20000.0), hotel("unpriced", price=None)]
    kept = {c["id"] for c in r._apply_filters(cands, QueryIntent())}
    assert kept == {"priced"}


def test_non_exclude_policy_keeps_unpriced_when_no_price_constraint():
    r = WeightedRetriever(price_policy="neutral")
    cands = [hotel("priced", price=20000.0), hotel("unpriced", price=None)]
    kept = {c["id"] for c in r._apply_filters(cands, QueryIntent())}
    assert kept == {"priced", "unpriced"}


def test_price_imputation_is_recorded_on_the_candidate():
    r = WeightedRetriever(price_policy="worst")
    cands = [hotel("known", price=20000.0), hotel("unknown", price=None)]
    r._score(cands, QueryIntent(), EQUAL_WEIGHTS)
    flags = {c["id"]: c["price_imputed"] for c in cands}
    assert flags == {"known": False, "unknown": True}


# ---------------------------------------------------------------------------
# Neighbourhood diffusion (the multi-hop contribution)
# ---------------------------------------------------------------------------

def test_congested_neighbourhood_lowers_disruption_score():
    """A hotel with a clean own-signal is still penalised for its surroundings.

    This is the behaviour the multi-hop traversal exists to produce, and the one
    a star-join cannot express.
    """
    r = WeightedRetriever(self_weight=0.5)
    calm = hotel("calm", nbr_eta=0.0, nbr_severity=0.0, nbr_count=6)
    surrounded = hotel("surrounded", nbr_eta=15.0, nbr_severity=1.0, nbr_count=6)
    scored = {h.id: h for h in r._score([calm, surrounded], QueryIntent(), EQUAL_WEIGHTS)}
    assert (scored["surrounded"].components["disruption"]
            < scored["calm"].components["disruption"])


def test_self_weight_one_disables_diffusion():
    """The ablation must be exact: self_weight = 1.0 ignores neighbours entirely."""
    r = WeightedRetriever(self_weight=1.0)
    calm = hotel("a", nbr_eta=0.0, nbr_severity=0.0, nbr_count=6)
    surrounded = hotel("b", nbr_eta=18.0, nbr_severity=1.0, nbr_count=6)
    scored = {h.id: h for h in r._score([calm, surrounded], QueryIntent(), EQUAL_WEIGHTS)}
    assert (scored["a"].components["disruption"]
            == pytest.approx(scored["b"].components["disruption"]))


def test_isolated_hotel_falls_back_to_own_exposure():
    """No neighbours must not mean a free pass.

    With nbr_count = 0 the neighbourhood term is undefined; blending it in as
    zero would credit an isolated hotel with a perfectly calm neighbourhood it
    was never observed to have.
    """
    r = WeightedRetriever(self_weight=0.5)
    isolated = hotel("iso", signal_severities=["heavy"], signal_etas=[20.0],
                     nbr_count=0, nbr_eta=0.0, nbr_severity=0.0)
    scored = r._score([isolated], QueryIntent(), EQUAL_WEIGHTS)[0]
    assert scored.components["disruption"] < 0.5
    assert scored.raw["diffusion"]["self_weight"] == pytest.approx(1.0)


def test_own_heavy_traffic_lowers_disruption_score():
    r = WeightedRetriever()
    clean = hotel("clean")
    jammed = hotel("jammed", signal_severities=["heavy"], signal_etas=[15.0])
    scored = {h.id: h for h in r._score([clean, jammed], QueryIntent(), EQUAL_WEIGHTS)}
    assert scored["jammed"].components["disruption"] < scored["clean"].components["disruption"]


def test_event_impact_lowers_disruption_score():
    r = WeightedRetriever()
    away = hotel("away", event_impact=0.0)
    at_event = hotel("at_event", event_impact=1.0, event_count=1)
    scored = {h.id: h for h in r._score([away, at_event], QueryIntent(), EQUAL_WEIGHTS)}
    assert scored["at_event"].components["disruption"] < scored["away"].components["disruption"]


def test_diffusion_provenance_is_recorded():
    r = WeightedRetriever(self_weight=0.7)
    cands = [hotel("a", nbr_count=4, nbr_eta=6.0)]
    scored = r._score(cands, QueryIntent(), EQUAL_WEIGHTS)[0]
    d = scored.raw["diffusion"]
    assert set(d) == {"self_weight", "neighbours", "own_exposure",
                      "neighbourhood_exposure", "blended_exposure"}
    assert d["neighbours"] == 4
    assert d["self_weight"] == pytest.approx(0.7)


def test_max_hops_bounds_are_enforced():
    for bad in (0, 5, -1):
        with pytest.raises(ValueError):
            WeightedRetriever(max_hops=bad)
    for good in (1, 2, 3, 4):
        WeightedRetriever(max_hops=good)
