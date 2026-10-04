"""SCORING_MODE: weighted GraphRAG vs graph-only (unweighted) GraphRAG.

Unweighted mode must combine the five components with equal weights no matter
what the query intent, a traveller profile or the configured weight profile
says, and it must leave the weighted path exactly as it was. None of these tests
touch Neo4j: `_fetch_candidates` is replaced by a hand-built pool.
"""
from __future__ import annotations

import pytest

from evaluation import baselines as bl
from src.crag.query_parser import QueryIntent
from src.graph import retriever as rt
from src.graph.retriever import (HANDSET_WEIGHTS, UNIFORM_WEIGHTS, WeightedRetriever,
                                 apply_intent_adjustments, resolve_scoring_mode)


def hotel(hid, **kw):
    base = {
        "id": hid, "name": f"Hotel {hid}", "rating": 4.0, "star": 4, "price": 20000.0,
        "distance_km": 3.0, "travel_time_min": 20.0, "travel_time_traffic_min": 22.0,
        "lat": 6.92, "lng": 79.86, "amenities": ["pool", "wifi"], "attractions": [],
        "locations": [], "signal_severities": [], "signal_etas": [], "event_count": 0,
        "event_impact": 0.0, "event_distance_km": None, "nbr_eta": 0.0,
        "nbr_severity": 0.0, "nbr_event_impact": 0.0, "nbr_count": 0,
    }
    base.update(kw)
    return base


POOL = [hotel("a", price=10000.0, rating=3.5, distance_km=6.0, travel_time_traffic_min=30.0),
        hotel("b", price=30000.0, rating=4.8, distance_km=1.0, travel_time_traffic_min=6.0),
        hotel("c", price=20000.0, rating=4.2, distance_km=3.0, travel_time_traffic_min=15.0)]


def _retriever(monkeypatch, mode):
    r = WeightedRetriever(scoring_mode=mode)
    monkeypatch.setattr(r, "_fetch_candidates", lambda city: [dict(h) for h in POOL])
    return r


def _as_tuple(w):
    return tuple(round(getattr(w, d), 6) for d in
                 ("spatial", "accessibility", "facility", "economic", "disruption"))


# -- mode resolution --------------------------------------------------------

def test_default_mode_is_weighted(monkeypatch):
    monkeypatch.setattr(rt, "SCORING_MODE", "weighted")
    assert resolve_scoring_mode() == "weighted"


def test_env_value_selects_unweighted(monkeypatch):
    monkeypatch.setattr(rt, "SCORING_MODE", "unweighted")
    assert resolve_scoring_mode() == "unweighted"
    assert WeightedRetriever().scoring_mode == "unweighted"


def test_explicit_mode_overrides_env(monkeypatch):
    monkeypatch.setattr(rt, "SCORING_MODE", "unweighted")
    assert WeightedRetriever(scoring_mode="weighted").scoring_mode == "weighted"


def test_unknown_mode_falls_back_to_weighted():
    assert resolve_scoring_mode("equalish") == "weighted"


# -- unweighted ranking ----------------------------------------------------------

def test_unweighted_uses_equal_weights_whatever_the_intent(monkeypatch):
    r = _retriever(monkeypatch, "unweighted")
    for sort_intent in (None, "cheapest", "highest_rated", "most_accessible"):
        res = r.retrieve(QueryIntent(city="Colombo", sort_intent=sort_intent), limit=3)
        assert _as_tuple(res.weights) == _as_tuple(UNIFORM_WEIGHTS)
        assert res.scoring_mode == "unweighted"


def test_unweighted_ignores_profile_weights(monkeypatch):
    class Profile:
        id = "p"
        weights = HANDSET_WEIGHTS
        event_preference = "seek"
        proximity_preference = "any"
    r = _retriever(monkeypatch, "unweighted")
    res = r.retrieve(QueryIntent(city="Colombo"), limit=3, profile=Profile())
    assert _as_tuple(res.weights) == _as_tuple(UNIFORM_WEIGHTS)


def test_unweighted_still_applies_explicit_weight_model(monkeypatch):
    """An ablation sets a weight model on purpose; it must still take effect."""
    class OnlyEconomic:
        def predict(self, intent, pool):
            return rt.ScoringWeights(0.0, 0.0, 0.0, 1.0, 0.0)
    r = _retriever(monkeypatch, "unweighted")
    r.weight_model = OnlyEconomic()
    res = r.retrieve(QueryIntent(city="Colombo"), limit=3)
    assert res.weights.economic == pytest.approx(1.0)


def test_unweighted_score_is_mean_of_components(monkeypatch):
    res = _retriever(monkeypatch, "unweighted").retrieve(QueryIntent(city="Colombo"), limit=3)
    for h in res.hotels:
        mean = sum(h.components[d] for d in
                   ("spatial", "accessibility", "facility", "economic", "disruption")) / 5
        assert h.score == pytest.approx(mean, abs=1e-6)


def test_feasibility_filter_applies_in_unweighted_mode(monkeypatch):
    res = _retriever(monkeypatch, "unweighted").retrieve(
        QueryIntent(city="Colombo", max_price_lkr=25000.0), limit=3)
    assert {h.id for h in res.hotels} == {"a", "c"}


# -- weighted path is unchanged ----------------------------------------------------

def test_weighted_mode_still_applies_intent_ladder(monkeypatch):
    monkeypatch.setattr(rt, "SCORING_WEIGHTS_PROFILE", "handset")
    r = _retriever(monkeypatch, "weighted")
    intent = QueryIntent(city="Colombo", sort_intent="cheapest")
    res = r.retrieve(intent, limit=3)
    expected = apply_intent_adjustments(rt.base_weights("handset"), intent)
    assert _as_tuple(res.weights) == _as_tuple(expected)
    assert res.scoring_mode == "weighted"


# -- evaluation wiring -------------------------------------------------------------

def test_graph_system_names():
    assert bl.graph_system_name("weighted") == "WeightedGraphRAG"
    assert bl.graph_system_name("unweighted") == "GraphRAG[unweighted]"


def test_baseline_names_follow_mode():
    assert bl.WeightedGraphBaseline(scoring_mode="unweighted").name == "GraphRAG[unweighted]"
    assert bl.WeightedGraphBaseline(scoring_mode="weighted").name == "WeightedGraphRAG"
    assert (bl.WeightedGraphBaseline(scoring_mode="unweighted", self_weight=1.0).name
            == "GraphRAG[unweighted,no-diffusion]")


def _lineup(monkeypatch, **kw):
    monkeypatch.setattr(bl.vs, "is_available", lambda: False)
    return [b.name for b in bl.all_baselines(include_llm=False, include_ltr=False, **kw)]


def test_unweighted_lineup_skips_weight_profiles_and_policies(monkeypatch):
    names = _lineup(monkeypatch, scoring_mode="unweighted",
                    weight_profiles=["handset"], weight_policies=["human-price-aware"])
    assert "GraphRAG[unweighted]" in names
    assert not any(n.startswith("WeightedGraphRAG") for n in names)


def test_compare_modes_adds_the_other_row(monkeypatch):
    names = _lineup(monkeypatch, scoring_mode="unweighted", compare_modes=True)
    assert "GraphRAG[unweighted]" in names and "WeightedGraphRAG" in names
