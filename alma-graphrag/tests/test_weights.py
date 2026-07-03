import pytest

from src.crag.query_parser import QueryIntent
from src.crag.user_profile import PRESETS, get_profile
from src.graph.retriever import (
    ScoringWeights,
    _haversine_km,
    weights_for_intent,
    weights_for_profile,
)


def _total(w: ScoringWeights) -> float:
    return w.spatial + w.accessibility + w.facility + w.economic + w.disruption + w.event


def test_normalised_sums_to_one():
    w = ScoringWeights(spatial=2, accessibility=1, facility=1, economic=1, disruption=1).normalised()
    assert _total(w) == pytest.approx(1.0)


def test_normalised_handles_all_zero():
    w = ScoringWeights(0, 0, 0, 0, 0, 0).normalised()
    assert _total(w) == pytest.approx(1.0)  # falls back to defaults


def test_cheapest_intent_boosts_economic():
    base = weights_for_intent(QueryIntent())
    cheap = weights_for_intent(QueryIntent(sort_intent="cheapest"))
    assert cheap.economic > base.economic
    assert max(
        cheap.spatial, cheap.accessibility, cheap.facility, cheap.economic, cheap.disruption
    ) == cheap.economic


def test_avoid_traffic_boosts_disruption():
    base = weights_for_intent(QueryIntent())
    quiet = weights_for_intent(QueryIntent(avoid_traffic=True, proximity_preference="far"))
    assert quiet.disruption > base.disruption


def test_intent_weights_always_normalised():
    w = weights_for_intent(
        QueryIntent(sort_intent="most_accessible", avoid_traffic=True, required_amenities=["pool"])
    )
    assert _total(w) == pytest.approx(1.0)


def test_profile_weights_override_intent():
    profile = PRESETS["budget_traveler"]
    w = weights_for_profile(profile, QueryIntent(sort_intent="highest_rated"), event_active=False)
    # Budget profile dominates despite a quality-leaning intent.
    assert w.economic == max(w.spatial, w.accessibility, w.facility, w.economic, w.disruption)
    assert w.event == 0.0


def test_event_weight_added_only_for_seek_or_avoid():
    intent = QueryIntent()
    seeker = weights_for_profile(PRESETS["event_seeker"], intent, event_active=True)
    neutral = weights_for_profile(PRESETS["budget_traveler"], intent, event_active=True)
    no_event = weights_for_profile(PRESETS["event_seeker"], intent, event_active=False)
    assert seeker.event > 0.0
    assert neutral.event == 0.0
    assert no_event.event == 0.0
    assert _total(seeker) == pytest.approx(1.0)


def test_get_profile_case_insensitive_and_unknown():
    assert get_profile("EVENT_SEEKER") is PRESETS["event_seeker"]
    assert get_profile("nope") is None
    assert get_profile(None) is None


def test_haversine_colombo_to_kandy():
    d = _haversine_km(6.9271, 79.8612, 7.2906, 80.6337)
    assert 85 < d < 105  # ~94 km great-circle


def test_haversine_zero_distance():
    assert _haversine_km(6.9, 79.8, 6.9, 79.8) == pytest.approx(0.0)
