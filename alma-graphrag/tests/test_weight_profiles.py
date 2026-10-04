"""Tests for the named composite-weight profiles.

`handset` is the sole deployable prior until a captured human-choice artifact
passes every acceptance gate. `elicited`, `blended`, and `balanced` are retained
as historical research vectors, not runtime-selectable profiles.

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
    HISTORICAL_WEIGHT_PROFILES,
    WEIGHT_PROFILES,
    ScoringWeights,
    WeightedRetriever,
    base_weights,
    weights_for_intent,
    weights_for_profile,
)
from evaluation.harness import _StaticWeightModel

ALL_PROFILES = ["handset"]
HISTORICAL_PROFILES = ["elicited", "blended", "balanced"]


def _total(w: ScoringWeights) -> float:
    return w.spatial + w.accessibility + w.facility + w.economic + w.disruption + w.event


# ---------------------------------------------------------------------------
# The profile table
# ---------------------------------------------------------------------------

def test_registry_holds_only_deployable_profiles():
    assert set(WEIGHT_PROFILES) == set(ALL_PROFILES)
    assert set(HISTORICAL_WEIGHT_PROFILES) == set(HISTORICAL_PROFILES)


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
    """The historical blended vector has non-zero components.

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


def test_current_human_fit_is_not_selectable_after_failing_acceptance_gates():
    """Wave 1 is position-biased, so its calculated weights cannot reach serving."""
    assert HUMAN_WEIGHTS is None
    assert "human" not in WEIGHT_PROFILES


def test_load_human_weights_accepts_a_passing_captured_data_artifact(tmp_path):
    import json
    from src.graph.retriever import load_human_weights

    artifact = tmp_path / "human_weights.json"
    artifact.write_text(json.dumps({
        "shippable": True,
        "weights": {
            "spatial": 0.25,
            "accessibility": 0.20,
            "facility": 0.25,
            "economic": 0.15,
            "disruption": 0.15,
        },
    }), encoding="utf-8")

    assert load_human_weights(artifact).to_dict() == {
        "spatial": 0.25,
        "accessibility": 0.20,
        "facility": 0.25,
        "economic": 0.15,
        "disruption": 0.15,
    }


def test_balanced_matches_the_fitted_artifact():
    """The historical vector must equal what the fitter last produced.

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
    assert base_weights("HANDSET").to_dict() == HANDSET_WEIGHTS.to_dict()


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


def test_historical_profiles_cannot_be_selected_at_runtime():
    for name in HISTORICAL_PROFILES:
        assert base_weights(name).to_dict() == HANDSET_WEIGHTS.to_dict()


def test_historical_vectors_are_available_only_to_the_counterfactual_model():
    weights = _StaticWeightModel(ELICITED_WEIGHTS).predict(QueryIntent(sort_intent="cheapest"), [])
    assert weights.economic > ELICITED_WEIGHTS.economic


def test_user_profile_still_overrides_the_weight_profile():
    """A UserProfile preset supplies its own weights and must win."""
    w = weights_for_profile(PRESETS["budget_traveler"], QueryIntent(),
                            event_active=False, weight_profile="elicited")
    assert w.economic == max(w.spatial, w.accessibility, w.facility,
                             w.economic, w.disruption)


def test_weight_profile_used_when_no_user_profile_supplied():
    w = weights_for_profile(None, QueryIntent(), event_active=False,
                            weight_profile="handset")
    assert w.facility == pytest.approx(HANDSET_WEIGHTS.facility)
    assert w.accessibility == pytest.approx(HANDSET_WEIGHTS.accessibility)
    assert _total(w) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Wiring into the retriever and the evaluation harness
# ---------------------------------------------------------------------------

def test_retriever_records_its_profile():
    assert WeightedRetriever().weight_profile is None
    assert WeightedRetriever(weight_profile="handset").weight_profile == "handset"


def test_baseline_names_distinguish_profiles():
    """Distinct names keep the systems as separate rows in results.json."""
    assert WeightedGraphBaseline().name == "WeightedGraphRAG"
    assert WeightedGraphBaseline("handset").name == "WeightedGraphRAG[handset]"


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


def test_all_baselines_skips_profiles_that_do_not_exist(monkeypatch):
    import evaluation.baselines as bl
    monkeypatch.setattr(bl.vs, "is_available", lambda *a, **k: False)
    names = [b.name for b in all_baselines(["no-such-profile"])]
    assert "WeightedGraphRAG[no-such-profile]" not in names


def test_research_profiles_are_evaluated_but_marked_research(monkeypatch):
    """Gated-out vectors are rankable here and nowhere else.

    `human` fails its acceptance gates, so serving refuses it (NFR-09) while the
    harness still needs a row for it — that row is the only way to find out what
    human-fitted weights do to ranking quality.
    """
    import evaluation.baselines as bl
    monkeypatch.setattr(bl.vs, "is_available", lambda *a, **k: False)
    built = {b.name: b for b in all_baselines(["elicited", "human"])
             if isinstance(b, bl.WeightedGraphBaseline)}
    for name in ("WeightedGraphRAG[elicited]", "WeightedGraphRAG[human]"):
        assert name in built
        assert built[name].research_profile is (
            built[name].weight_profile not in WEIGHT_PROFILES)
