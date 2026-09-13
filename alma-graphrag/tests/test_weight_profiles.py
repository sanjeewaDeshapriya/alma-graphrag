"""Tests for the named composite-weight profiles.

`handset` is the original hand-tuned prior; `elicited` and `blended` come from
the discrete-choice study in studies/weight-elicitation; `balanced` re-weights
price and quality back to booking-stage literature values, because the study
measured consideration-stage behaviour and cannot identify them. The profiles are
selectable at runtime (SCORING_WEIGHTS_PROFILE) and per-retriever, so the
evaluation harness can score them head-to-head on identical gold.

These tests run offline — no Neo4j, no pgvector.
"""
import pytest

from evaluation.baselines import WeightedGraphBaseline, all_baselines
from src.crag.query_parser import QueryIntent
from src.crag.user_profile import PRESETS
from src.graph.retriever import (
    BALANCED_WEIGHTS,
    HUMAN_WEIGHTS,
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

ALL_PROFILES = ["handset", "elicited", "blended", "balanced", "human"]


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


def test_elicited_leaves_the_unidentified_components_at_about_zero():
    """Documents a known limitation rather than hiding it.

    After controlling for list position, the bootstrap 95% CIs for these three
    all include zero — facility [0.000, 0.238], economic [0.000, 0.107],
    disruption [0.000, 0.136] — so none is distinguishable from no effect at
    all, whatever its point estimate happens to be. What the design does
    identify is the TOTAL location weight, 0.816 [0.582, 1.000]; even the split
    of that between spatial and accessibility is weak, since the two correlate
    0.907 in Colombo.

    That is a statement about what this booking task could measure — participants
    opened a median of 1 hotel out of 32, so no comparison happened — NOT
    evidence that price or disruption are irrelevant. It is exactly why
    `blended` exists. If a future refit makes these substantial it should be a
    deliberate decision backed by a design that can identify them, not silent
    drift. See docs/Weight_Elicitation_Data_Audit.md.
    """
    location = ELICITED_WEIGHTS.spatial + ELICITED_WEIGHTS.accessibility
    for dim in ("facility", "economic", "disruption"):
        # Each unidentified component carries less mass than either identified
        # one, and the three together carry less than the location pair.
        assert getattr(ELICITED_WEIGHTS, dim) < ELICITED_WEIGHTS.spatial
    assert location > 0.75


def test_blended_is_the_equal_mixture_of_elicited_and_the_prior():
    for dim in ("spatial", "accessibility", "facility", "economic", "disruption"):
        expected = 0.5 * getattr(ELICITED_WEIGHTS, dim) + 0.5 * getattr(HANDSET_WEIGHTS, dim)
        assert getattr(BLENDED_WEIGHTS, dim) == pytest.approx(expected, abs=1e-3)


def test_blended_keeps_every_component_usable():
    """The reason `blended` is the deployable profile.

    A retriever scoring with economic = 0 cannot answer "cheapest hotel near
    Galle Face", and one with disruption = 0 discards the thesis's contribution.
    The study measured booking behaviour on a sorted list; it never tested
    constraint satisfaction, so it does not get to zero out a capability it did
    not measure.
    """
    for dim in ("spatial", "accessibility", "facility", "economic", "disruption"):
        assert getattr(BLENDED_WEIGHTS, dim) > 0.0

    # Location still leads, as the study found — just not to the exclusion of all else.
    assert BLENDED_WEIGHTS.spatial + BLENDED_WEIGHTS.accessibility > 0.5


# ---------------------------------------------------------------------------
# base_weights()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# `balanced` — the real-world profile
# ---------------------------------------------------------------------------

def test_balanced_gives_price_a_literature_sized_share():
    """The whole point of the profile.

    `elicited` pins economic at 0.000 and `blended` only lifts it to 0.075,
    both because the study could not identify price, not because travellers
    ignore it.

    The band here is the RENORMALISED literature share, not the raw one. The
    published 16.5% is price's share of seven attributes, three of which this
    retriever does not model at all (cancellation policy, photos, brand).
    Across the three it does model, price is 34.3% -- and after reserving
    `disruption` at 0.150 that is 0.29 of the profile.
    """
    assert 0.25 <= BALANCED_WEIGHTS.economic <= 0.35
    assert BALANCED_WEIGHTS.economic > 3 * BLENDED_WEIGHTS.economic


def test_human_profile_matches_its_artifact():
    """`human` is fitted from the study alone; keep code and artifact in step.

    Regenerate with:
        python -m weight_elicitation.fit_human_weights --emit-profile
    """
    import json
    from pathlib import Path
    art = (Path(__file__).resolve().parents[1] / "weight_elicitation" / "out"
           / "human_weights.json")
    if not art.exists():
        pytest.skip("human_weights.json not present")
    for dim, v in json.loads(art.read_text(encoding="utf-8"))["weights"].items():
        assert getattr(HUMAN_WEIGHTS, dim) == pytest.approx(v, abs=5e-4)


def test_human_profile_gives_price_a_nonzero_weight():
    """The point of the stratified estimator.

    Pooling every display condition into one logit reports economic = 0.000 with
    a CI spanning zero. Estimating per display condition and macro-averaging
    recovers 0.120, CI [0.049, 0.256], excluding zero.
    """
    assert HUMAN_WEIGHTS.economic > 0.05
    assert HUMAN_WEIGHTS.economic > ELICITED_WEIGHTS.economic


def test_human_profile_location_total_below_the_pooled_fit():
    """Removing the proximity-sorted majority's vote lowers location mass."""
    human_loc = HUMAN_WEIGHTS.spatial + HUMAN_WEIGHTS.accessibility
    elicited_loc = ELICITED_WEIGHTS.spatial + ELICITED_WEIGHTS.accessibility
    assert human_loc < elicited_loc


def test_balanced_matches_the_fitted_artifact():
    """The shipped vector must equal what the fitter last produced.

    `balanced` is no longer hand-picked: scripts/fit_weight_profile.py learns it
    and writes evaluation/fitted_weights.json. If someone edits the constant by
    hand, or re-runs the fit and forgets to paste the result, these two drift
    apart and the profile stops being reproducible. This is the test that keeps
    the claim honest.

    Regenerate both together with:
        python scripts/fit_weight_profile.py --emit-profile
    """
    import json
    from pathlib import Path
    art = Path(__file__).resolve().parents[1] / "evaluation" / "fitted_weights.json"
    if not art.exists():
        pytest.skip("fitted_weights.json not present; run scripts/fit_weight_profile.py")
    fitted = json.loads(art.read_text(encoding="utf-8"))["weights_final"]
    for dim, v in fitted.items():
        assert getattr(BALANCED_WEIGHTS, dim) == pytest.approx(v, abs=5e-4), (
            f"{dim}: profile has {getattr(BALANCED_WEIGHTS, dim)}, "
            f"fitted_weights.json has {v}")


def test_balanced_puts_price_and_quality_above_any_single_location_dimension():
    """What the fit found, and what the booking-stage literature independently
    says: price and quality each outweigh either half of the location term."""
    for dim in ("spatial", "accessibility"):
        assert BALANCED_WEIGHTS.economic > getattr(BALANCED_WEIGHTS, dim)
        assert BALANCED_WEIGHTS.facility > getattr(BALANCED_WEIGHTS, dim)


def test_balanced_location_split_favours_travel_time():
    """The fit puts location almost entirely on `accessibility`.

    NOT a finding that travel time beats proximity: the two correlate +0.966 at
    serving time. It is what a fit does with two near-redundant features. The
    fitter measures whether redividing the mass by the study's 0.447/0.553 ratio
    is free and keeps the fitted split only when it is not -- at the fitted mass
    that substitution costs 0.0138 nDCG, so the fitted split stands.
    """
    assert BALANCED_WEIGHTS.accessibility > BALANCED_WEIGHTS.spatial


def test_no_balanced_component_is_exactly_zero():
    """Every dimension must still be able to move a ranking.

    The threshold is deliberately low: `spatial` is fitted at 0.051 because
    `accessibility` already carries the location signal, and a fit over two
    redundant features concentrates the mass. Zero would be different -- it
    would make the component dead code and the explanation bars misleading.
    """
    for dim, v in BALANCED_WEIGHTS.to_dict().items():
        if dim == "event":
            continue
        assert v > 0.02, f"{dim} is effectively dead: {v}"


def test_balanced_leaves_disruption_at_the_hand_set_value():
    """Neither the study nor the literature can speak to it, so it is not moved."""
    assert BALANCED_WEIGHTS.disruption == pytest.approx(HANDSET_WEIGHTS.disruption, abs=1e-9)


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
    # Compare against the profile itself rather than a literal, so a refit of
    # the study data does not have to be mirrored by hand in this assertion.
    assert w.facility == pytest.approx(ELICITED_WEIGHTS.facility)
    assert w.accessibility == pytest.approx(ELICITED_WEIGHTS.accessibility)
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
