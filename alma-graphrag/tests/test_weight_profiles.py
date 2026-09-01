"""Tests for the named composite-weight profiles.

`handset` is the original hand-tuned prior; `elicited` and `blended` come from
the discrete-choice study in studies/weight-elicitation. The profiles are
selectable at runtime (SCORING_WEIGHTS_PROFILE) and per-retriever, so the
evaluation harness can score them head-to-head on identical gold.

These tests run offline — no Neo4j, no pgvector.
"""
import pytest

from evaluation.baselines import WeightedGraphBaseline, all_baselines
from src.crag.query_parser import QueryIntent
from src.crag.user_profile import PRESETS
from src.graph.retriever import (
    BLENDED_WEIGHTS,
    ELICITED_WEIGHTS,
    HANDSET_WEIGHTS,
    WEIGHT_PROFILES,
    ScoringWeights,
    WeightedRetriever,
    base_weights,
    weights_for_intent,
    weights_for_profile,
)

ALL_PROFILES = ["handset", "elicited", "blended"]


def _total(w: ScoringWeights) -> float:
    return w.spatial + w.accessibility + w.facility + w.economic + w.disruption + w.event


# ---------------------------------------------------------------------------
# The profile table
# ---------------------------------------------------------------------------

def test_registry_holds_exactly_the_documented_profiles():
    assert set(WEIGHT_PROFILES) == set(ALL_PROFILES)


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_every_profile_sums_to_one(name):
    assert _total(WEIGHT_PROFILES[name]) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_no_profile_has_a_negative_weight(name):
    w = WEIGHT_PROFILES[name]
    assert min(w.spatial, w.accessibility, w.facility, w.economic, w.disruption) >= 0.0


def test_elicited_puts_location_first():
    """The study's headline finding: spatial + accessibility dominate."""
    w = ELICITED_WEIGHTS
    assert w.spatial > HANDSET_WEIGHTS.spatial
    assert w.spatial + w.accessibility > 0.75


def test_elicited_zeroes_facility_and_economic():
    """Documents a known limitation rather than hiding it.

    Both components fitted negative and clip to zero on the simplex. That is a
    real effect in the booking task but does NOT transfer to constraint queries,
    which is exactly why `blended` exists. If a future change makes these
    non-zero it should be a deliberate decision, not a silent drift.
    """
    assert ELICITED_WEIGHTS.facility == 0.0
    assert ELICITED_WEIGHTS.economic == 0.0


def test_blended_keeps_the_handset_prior_for_the_untransferable_components():
    assert BLENDED_WEIGHTS.facility == HANDSET_WEIGHTS.facility
    assert BLENDED_WEIGHTS.economic == HANDSET_WEIGHTS.economic
    # ...and follows the elicited ordering for the rest.
    assert BLENDED_WEIGHTS.spatial > BLENDED_WEIGHTS.accessibility > BLENDED_WEIGHTS.disruption


# ---------------------------------------------------------------------------
# base_weights()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_PROFILES)
def test_base_weights_returns_the_named_profile(name):
    assert base_weights(name).to_dict() == WEIGHT_PROFILES[name].to_dict()


def test_base_weights_is_case_insensitive():
    assert base_weights("ELICITED").to_dict() == ELICITED_WEIGHTS.to_dict()


def test_unknown_profile_falls_back_to_handset_without_raising():
    """A typo in an env var must not take the retriever down."""
    assert base_weights("no-such-profile").to_dict() == HANDSET_WEIGHTS.to_dict()


def test_base_weights_returns_a_copy_not_the_shared_constant():
    """weights_for_intent mutates what base_weights hands back.

    Returning the module-level constant would let one query permanently corrupt
    the profile for every subsequent request in the process.
    """
    w = base_weights("handset")
    w.economic += 5.0
    assert HANDSET_WEIGHTS.economic == 0.15
    assert base_weights("handset").economic == 0.15


# ---------------------------------------------------------------------------
# Intent adjustment on top of a profile
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_PROFILES)
def test_intent_weights_normalised_for_every_profile(name):
    w = weights_for_intent(
        QueryIntent(sort_intent="most_accessible", avoid_traffic=True,
                    required_amenities=["pool"]),
        name,
    )
    assert _total(w) == pytest.approx(1.0)


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_cheapest_intent_raises_economic_on_every_profile(name):
    base = weights_for_intent(QueryIntent(), name)
    cheap = weights_for_intent(QueryIntent(sort_intent="cheapest"), name)
    assert cheap.economic > base.economic


def test_elicited_recovers_a_price_signal_on_cheapest_queries():
    """Base economic is 0.0, but the +0.20 intent bump must still bite.

    Without this the elicited profile would be permanently price-blind, which
    would make "cheapest hotel" unanswerable.
    """
    w = weights_for_intent(QueryIntent(sort_intent="cheapest"), "elicited")
    assert w.economic > 0.10


def test_profiles_produce_different_rankings_signal():
    intent = QueryIntent()
    vecs = {name: weights_for_intent(intent, name).to_dict() for name in ALL_PROFILES}
    assert vecs["handset"] != vecs["elicited"]
    assert vecs["blended"] != vecs["elicited"]
    assert vecs["blended"] != vecs["handset"]


def test_user_profile_still_overrides_the_weight_profile():
    """A UserProfile preset supplies its own weights and must win."""
    w = weights_for_profile(PRESETS["budget_traveler"], QueryIntent(),
                            event_active=False, weight_profile="elicited")
    assert w.economic == max(w.spatial, w.accessibility, w.facility,
                             w.economic, w.disruption)


def test_weight_profile_used_when_no_user_profile_supplied():
    w = weights_for_profile(None, QueryIntent(), event_active=False,
                            weight_profile="elicited")
    assert w.facility == pytest.approx(0.0)
    assert _total(w) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Wiring into the retriever and the evaluation harness
# ---------------------------------------------------------------------------

def test_retriever_records_its_profile():
    assert WeightedRetriever().weight_profile is None
    assert WeightedRetriever(weight_profile="blended").weight_profile == "blended"


def test_baseline_names_distinguish_profiles():
    """Distinct names keep the systems as separate rows in results.json."""
    assert WeightedGraphBaseline().name == "WeightedGraphRAG"
    assert WeightedGraphBaseline("elicited").name == "WeightedGraphRAG[elicited]"
    assert WeightedGraphBaseline("blended").name == "WeightedGraphRAG[blended]"


def test_all_baselines_adds_one_row_per_requested_profile(monkeypatch):
    import evaluation.baselines as bl
    monkeypatch.setattr(bl.vs, "is_available", lambda *a, **k: False)

    without = [b.name for b in all_baselines()]
    with_profiles = [b.name for b in all_baselines(ALL_PROFILES)]

    assert len(with_profiles) == len(without) + len(ALL_PROFILES)
    assert "WeightedGraphRAG" in with_profiles          # significance reference
    for name in ALL_PROFILES:
        assert f"WeightedGraphRAG[{name}]" in with_profiles


def test_all_baselines_names_are_unique(monkeypatch):
    """Duplicate names would silently overwrite each other in the results dict."""
    import evaluation.baselines as bl
    monkeypatch.setattr(bl.vs, "is_available", lambda *a, **k: False)
    names = [b.name for b in all_baselines(ALL_PROFILES)]
    assert len(names) == len(set(names))
