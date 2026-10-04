"""The shared conditional-logit core: position specifications, likelihood,
inference, cross-validation.

The central test reproduces the wave-1 failure on synthetic data with a KNOWN
answer: choices that fall off a cliff after rank 3 (as wave 1's did) on lists
sorted by one component. A -log(rank) control misattributes the cliff to that
component; rank dummies recover the truth. Everything runs offline.
"""
from __future__ import annotations

import numpy as np
import pytest

from weight_elicitation.choice_model import (
    DIMS,
    PositionSpec,
    build_design,
    choice_data_from_responses,
    cluster_bootstrap,
    cross_validate,
    fit_model,
    loglik,
    lr_test_components,
    model_se,
    percentile_ci,
    to_simplex,
    wilson_interval,
)


def make_cliff_study(beta, n_people=300, n_hotels=12, n_tasks=4, cliff=(0.0, -0.35, -0.8, -6.0),
                     sort_dim=0, seed=5, random_order=False):
    """Rank utility is a cliff after rank 3 (wave 1's shape).

    By default every list is sorted by `sort_dim`, so rank is confounded with that
    component and, with few questions, only a handful of hotel profiles are ever
    really weighed: wave 1 in miniature. `random_order=True` shuffles the order per
    participant x question, the wave-2 design, under which effects are identified.
    """
    rng = np.random.default_rng(seed)
    comps = {t: rng.uniform(0, 1, (n_hotels, 5)) for t in range(n_tasks)}
    responses = []
    for p in range(n_people):
        for t in range(n_tasks):
            order = (rng.permutation(n_hotels) if random_order
                     else np.argsort(-comps[t][:, sort_dim]))
            rank = np.empty(n_hotels, int)
            rank[order] = np.arange(1, n_hotels + 1)
            pos_u = np.array([cliff[min(r - 1, len(cliff) - 1)] for r in rank])
            u = comps[t] @ beta + pos_u
            prob = np.exp(u - u.max())
            prob /= prob.sum()
            pick = int(rng.choice(n_hotels, p=prob))
            responses.append({
                "participantId": f"p{p}", "taskId": f"t{t}", "isAttentionCheck": False,
                "timing": {"final_sort": "distance"},
                "options": [{"hotel_id": f"h{h}", "displayed_position": int(rank[h]),
                             "chosen": h == pick, "components": dict(zip(DIMS, comps[t][h]))}
                            for h in range(n_hotels)]})
    return choice_data_from_responses(responses)


NULL_BETA = np.zeros(5)
TRUE_BETA = np.array([0.0, 1.2, 0.0, 0.8, 0.0])


# --------------------------------------------------------------------------- #
# Position specifications
# --------------------------------------------------------------------------- #
def test_parse_and_label_round_trip():
    for text in ("neglog", "none", "dummies:3", "topk:4"):
        assert PositionSpec.parse(text).label() == text
    assert PositionSpec.parse("dummies").k == 3
    with pytest.raises(ValueError):
        PositionSpec("curve")


def test_dummies_encode_ranks_and_pool_the_tail():
    pos = np.array([[1, 2, 3, 4, 9]])
    mask = np.ones_like(pos, bool)
    P, m, names = PositionSpec("dummies", 3).columns(pos, mask)
    assert names == ["position_2", "position_3", "position_gt_3"]
    assert P[0].tolist() == [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
    assert m.all()


def test_topk_restricts_the_choice_set_and_drops_rows_chosen_below_k():
    data = make_cliff_study(TRUE_BETA, n_people=40)
    d = build_design(data, PositionSpec("topk", 3))
    assert (d.mask.sum(axis=1) == 3).all()
    chosen_rank = data.pos[np.arange(len(data)), data.y]
    assert d.n_dropped == int((chosen_rank > 3).sum())


def test_dead_position_columns_are_dropped():
    """Sets of 3 have no 'beyond 3' rank; that column must not survive."""
    data = make_cliff_study(TRUE_BETA, n_people=30, n_hotels=3)
    d = build_design(data, PositionSpec("dummies", 3))
    assert "position_gt_3" not in d.names


# --------------------------------------------------------------------------- #
# The wave-1 failure, reproduced with a known answer
# --------------------------------------------------------------------------- #
def test_neglog_books_a_position_cliff_as_preference_and_dummies_do_not():
    """Lists sorted by `spatial`, spatial has NO true effect, choices follow a cliff."""
    data = make_cliff_study(NULL_BETA, n_people=400, sort_dim=0)
    neglog = fit_model(data, PositionSpec("neglog"))
    dummies = fit_model(data, PositionSpec("dummies", 3))
    i = DIMS.index("spatial")
    # -log misfit shows up as a spatial effect several SEs from zero ...
    assert abs(neglog.beta[i] / neglog.se[i]) > 2.5
    # ... which the correctly shaped control does not produce.
    assert abs(dummies.beta[i] / dummies.se[i]) < 2.5
    assert lr_test_components(data, PositionSpec("dummies", 3))["p"] > 0.01


def test_dummies_recover_true_coefficients():
    data = make_cliff_study(TRUE_BETA, n_people=500, sort_dim=0)
    f = fit_model(data, PositionSpec("dummies", 3))
    assert np.all(np.abs(f.components - TRUE_BETA) < 3.5 * f.se[:5])
    assert lr_test_components(data, PositionSpec("dummies", 3))["p"] < 1e-6


# --------------------------------------------------------------------------- #
# Likelihood and inference
# --------------------------------------------------------------------------- #
def test_gradient_matches_finite_differences():
    data = make_cliff_study(TRUE_BETA, n_people=20)
    d = build_design(data, PositionSpec("dummies", 3))
    beta = np.linspace(-0.5, 0.5, d.X.shape[2])
    _, grad, _ = loglik(beta, d.X, d.mask, d.y)
    eps = 1e-6
    for j in range(len(beta)):
        e = np.zeros_like(beta)
        e[j] = eps
        num = (loglik(beta + e, d.X, d.mask, d.y)[0] - loglik(beta - e, d.X, d.mask, d.y)[0]) / (2 * eps)
        assert grad[j] == pytest.approx(num, rel=1e-4, abs=1e-4)


def test_hessian_se_agrees_with_clustered_bootstrap_roughly():
    data = make_cliff_study(TRUE_BETA, n_people=250, n_tasks=8, random_order=True)
    spec = PositionSpec("dummies", 3)
    f = fit_model(data, spec)
    draws = cluster_bootstrap(data, lambda s: fit_model(s, spec, with_se=False).components, 60, 1)
    boot_sd = draws.std(axis=0)
    assert np.all(boot_sd / f.se[:5] < 2.0) and np.all(boot_sd / f.se[:5] > 0.5)
    lo, hi = percentile_ci(draws)
    assert np.all(lo <= hi)


def test_non_negative_fit_never_goes_below_zero():
    data = make_cliff_study(np.array([-1.0, 1.0, 0, 0, 0]), n_people=100)
    f = fit_model(data, PositionSpec("dummies", 3), non_negative=True)
    assert (f.components >= -1e-9).all()


def test_to_simplex_never_invents_a_flat_vector():
    assert to_simplex(np.array([-1.0, -2, 0, -0.1, 0])).tolist() == [0.0] * 5
    w = to_simplex(np.array([2.0, 0, 1, -3, 1]))
    assert w.sum() == pytest.approx(1.0) and w[3] == 0.0


def test_prior_direction_penalty_returns_the_prior_shape_when_strong():
    data = make_cliff_study(TRUE_BETA, n_people=100)
    prior = np.array([0.25, 0.20, 0.25, 0.15, 0.15])
    f = fit_model(data, PositionSpec("dummies", 3), non_negative=True, l2=1e6,
                  prior_direction=prior, with_se=False)
    assert np.allclose(to_simplex(f.components), prior, atol=0.02)


def test_cross_validation_is_paired_and_split_by_participant():
    data = make_cliff_study(TRUE_BETA, n_people=150)
    cv = cross_validate(data, {"pos": {"components": False}, "full": {}},
                        PositionSpec("dummies", 3), n_folds=5, seed=2)
    a, b = cv["pos"]["per_choice"], cv["full"]["per_choice"]
    assert np.array_equal(np.isfinite(a), np.isfinite(b))
    assert cv["full"]["ll_per_choice"] > cv["pos"]["ll_per_choice"]


def test_model_se_is_finite_on_a_well_posed_fit():
    data = make_cliff_study(TRUE_BETA, n_people=60)
    d = build_design(data, PositionSpec("dummies", 3))
    f = fit_model(data, PositionSpec("dummies", 3))
    assert np.isfinite(model_se(f.beta, d.X, d.mask, d.y)).all()


def test_wilson_interval_contains_the_estimate_and_handles_edges():
    lo, hi = wilson_interval(77, 240)
    assert lo < 77 / 240 < hi
    assert wilson_interval(0, 50)[0] == 0.0
    assert wilson_interval(5, 0) == (0.0, 0.0)
