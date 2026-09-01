"""Tests for the learned weight policy (src/graph/weight_policy.py).

The properties worth pinning are the ones whose violation would silently
corrupt every reported number:

  * the weight vector must stay on the simplex, or the composite score stops
    being a convex combination and the explanation bars stop being comparable;
  * the feature vector's LAYOUT must be stable, because a checkpoint stores
    weights against column positions — a reordering silently mismatches
    features to neurons and degrades accuracy with no error;
  * inference must be deterministic, or an evaluation run is not reproducible.
"""
from __future__ import annotations

import pytest

from src.crag.query_parser import QueryIntent
from src.graph.retriever import ScoringWeights
from src.graph.weight_policy import (
    COMPONENTS,
    FEATURE_NAMES,
    N_FEATURES,
    HandTunedPolicy,
    PoolConditions,
    StaticProfilePolicy,
    featurise,
    get_policy,
)

torch = pytest.importorskip("torch", reason="learned policy needs PyTorch")


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------

def test_featurise_length_matches_declared_names():
    assert len(featurise(QueryIntent())) == N_FEATURES == len(FEATURE_NAMES)


def test_feature_names_are_unique():
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_featurise_is_deterministic():
    intent = QueryIntent(sort_intent="cheapest", required_amenities=["pool"])
    assert featurise(intent) == featurise(intent)


def test_sort_intent_is_one_hot():
    x = featurise(QueryIntent(sort_intent="cheapest"))
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    flags = [x[idx[n]] for n in ("sort_best_overall", "sort_cheapest",
                                 "sort_highest_rated", "sort_most_accessible")]
    assert sum(flags) == 1.0
    assert x[idx["sort_cheapest"]] == 1.0


def test_proximity_is_one_hot():
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    for pref, flag in (("close", "proximity_close"), ("far", "proximity_far"),
                       ("any", "proximity_any")):
        x = featurise(QueryIntent(proximity_preference=pref))
        assert x[idx[flag]] == 1.0
        assert sum(x[idx[n]] for n in ("proximity_close", "proximity_far",
                                       "proximity_any")) == 1.0


def test_amenity_count_is_squashed_to_unit_interval():
    idx = FEATURE_NAMES.index("n_required_amenities")
    many = featurise(QueryIntent(required_amenities=["a"] * 20))
    assert many[idx] == pytest.approx(1.0)


def test_features_stay_in_unit_interval():
    intent = QueryIntent(sort_intent="cheapest", max_price_lkr=5000,
                         min_rating=4.5, min_star=5, avoid_traffic=True,
                         required_amenities=["pool", "spa", "gym", "bar"])
    cond = PoolConditions(disruption_mean=1.0, disruption_spread=1.0,
                          price_missing_rate=1.0, event_active=1.0)
    assert all(0.0 <= v <= 1.0 for v in featurise(intent, cond))


# ---------------------------------------------------------------------------
# Pool conditions
# ---------------------------------------------------------------------------

def test_pool_conditions_from_empty_pool_is_neutral():
    c = PoolConditions.from_candidates([])
    assert (c.disruption_mean, c.disruption_spread,
            c.price_missing_rate, c.event_active) == (0.0, 0.0, 0.0, 0.0)


def test_pool_conditions_measures_missing_price_rate():
    cands = [{"price": 100.0}, {"price": None}, {"price": None}, {"price": 5.0}]
    assert PoolConditions.from_candidates(cands).price_missing_rate == pytest.approx(0.5)


def test_pool_conditions_detects_active_events():
    assert PoolConditions.from_candidates(
        [{"price": 1.0, "event_impact": 0.4}]
    ).event_active == 1.0
    assert PoolConditions.from_candidates(
        [{"price": 1.0, "event_impact": 0.0}]
    ).event_active == 0.0


def test_pool_conditions_spread_is_zero_when_uniform():
    cands = [{"price": 1.0, "signal_etas": [5.0]}] * 4
    assert PoolConditions.from_candidates(cands).disruption_spread == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Static policies
# ---------------------------------------------------------------------------

def _simplex(w: ScoringWeights) -> bool:
    vals = [getattr(w, c) for c in COMPONENTS]
    return all(v >= 0 for v in vals) and abs(sum(vals) + w.event - 1.0) < 1e-6


def test_handtuned_policy_returns_simplex_weights():
    assert _simplex(HandTunedPolicy().predict(QueryIntent()))


def test_static_profile_policy_ignores_intent():
    p = StaticProfilePolicy("handset")
    a = p.predict(QueryIntent(sort_intent="cheapest"))
    b = p.predict(QueryIntent(sort_intent="highest_rated"))
    assert a.to_dict() == b.to_dict()


def test_handtuned_policy_does_react_to_intent():
    p = HandTunedPolicy()
    cheap = p.predict(QueryIntent(sort_intent="cheapest"))
    rated = p.predict(QueryIntent(sort_intent="highest_rated"))
    assert cheap.to_dict() != rated.to_dict()


def test_static_profile_rejects_unknown_name():
    with pytest.raises(KeyError):
        StaticProfilePolicy("nonexistent")


def test_get_policy_resolves_known_names():
    assert isinstance(get_policy("handtuned"), HandTunedPolicy)
    assert isinstance(get_policy("elicited"), StaticProfilePolicy)


def test_get_policy_rejects_unknown_name():
    with pytest.raises(KeyError):
        get_policy("magic")


def test_get_policy_reports_missing_checkpoint_clearly():
    with pytest.raises(FileNotFoundError, match="train_weight_policy"):
        get_policy("learned", checkpoint="does/not/exist.pt")


# ---------------------------------------------------------------------------
# Dirichlet network
# ---------------------------------------------------------------------------

def test_network_alpha_is_above_one():
    """softplus + 1 keeps the Dirichlet unimodal.

    Without the +1 the density can spike at the simplex corners and the policy
    collapses to "one component takes everything" before learning anything.
    """
    from src.graph.weight_policy import build_network
    net = build_network()
    x = torch.randn(16, N_FEATURES)
    assert (net.alpha(x) > 1.0).all()


def test_network_mean_weights_lie_on_simplex():
    from src.graph.weight_policy import build_network
    net = build_network()
    w = net.mean_weights(torch.randn(32, N_FEATURES))
    assert torch.allclose(w.sum(dim=-1), torch.ones(32), atol=1e-5)
    assert (w >= 0).all()


def test_network_samples_lie_on_simplex():
    from src.graph.weight_policy import build_network
    net = build_network()
    w = net.distribution(torch.randn(8, N_FEATURES)).sample()
    assert torch.allclose(w.sum(dim=-1), torch.ones(8), atol=1e-5)
    assert (w >= 0).all()


def test_network_output_width_matches_components():
    from src.graph.weight_policy import build_network
    net = build_network()
    assert net.alpha(torch.zeros(1, N_FEATURES)).shape[-1] == len(COMPONENTS)


def test_log_prob_is_differentiable_wrt_parameters():
    """REINFORCE needs a gradient through log p(w | x), not through nDCG."""
    from src.graph.weight_policy import build_network
    net = build_network()
    dist = net.distribution(torch.randn(4, N_FEATURES))
    loss = -dist.log_prob(dist.rsample()).mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in net.parameters())


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip_is_deterministic(tmp_path):
    from src.graph.weight_policy import LearnedPolicy, build_network, save_checkpoint

    net = build_network()
    path = tmp_path / "policy.pt"
    save_checkpoint(net, path, meta={"hidden": 64, "note": "test"})

    policy = LearnedPolicy(path)
    intent = QueryIntent(sort_intent="cheapest")
    first = policy.predict(intent)
    second = policy.predict(intent)
    assert first.to_dict() == second.to_dict()
    assert _simplex(first)


def test_checkpoint_rejects_changed_feature_layout(tmp_path):
    """A silent feature reordering must fail loudly, not degrade quietly."""
    import src.graph.weight_policy as wp
    from src.graph.weight_policy import LearnedPolicy, build_network, save_checkpoint

    net = build_network()
    path = tmp_path / "policy.pt"
    save_checkpoint(net, path, meta={"hidden": 64})

    original = wp.FEATURE_NAMES
    try:
        wp.FEATURE_NAMES = tuple(reversed(original))
        with pytest.raises(ValueError, match="feature layout"):
            LearnedPolicy(path)
    finally:
        wp.FEATURE_NAMES = original


def test_saved_checkpoint_has_json_sidecar(tmp_path):
    from src.graph.weight_policy import build_network, save_checkpoint
    path = tmp_path / "policy.pt"
    save_checkpoint(build_network(), path, meta={"hidden": 64})
    assert (tmp_path / "policy.pt.json").exists()
