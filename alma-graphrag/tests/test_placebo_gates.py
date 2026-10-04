"""Placebo test, estimator registry and acceptance gates.

What must hold for the gates to mean anything:
  * the placebo null produces choices that ignore the components;
  * a clip-and-average estimator is NOT rescued by the placebo (its statistic on
    noise is as large as on the data) — the wave-1 share failure;
  * real effects pass every gate and a true-zero dimension does not;
  * `deployable_vector` never presents prior mass as an estimate.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.test_choice_model import make_cliff_study
from weight_elicitation.choice_model import DIMS, PositionSpec, fit_model
from weight_elicitation.estimators import ESTIMATORS, bind
from weight_elicitation.gates import GateConfig, deployable_vector, evaluate_gates
from weight_elicitation.placebo import placebo_test, position_only_null, restricted_null, simulate

SPEC = PositionSpec("dummies", 3)
FAST = GateConfig(placebo_reps=39, bootstrap_reps=40, cv_folds=4, cv_bootstrap_reps=200,
                  min_loqo_groups=3, seed=11)


def test_position_only_null_zeroes_every_component():
    data = make_cliff_study(np.array([0.0, 1.0, 0, 0.5, 0]), n_people=60)
    beta = position_only_null(data, SPEC)
    assert np.allclose(beta[:5], 0.0)
    assert beta[5] < 0 and beta[6] < 0              # ranks 2 and 3 less likely


def test_restricted_null_keeps_the_other_effects():
    data = make_cliff_study(np.array([0.0, 1.5, 0, 0.8, 0]), n_people=300)
    beta = restricted_null(data, SPEC, zero=("economic",))
    assert beta[DIMS.index("economic")] == 0.0
    assert beta[DIMS.index("accessibility")] > 0.5


def test_simulated_placebo_choices_carry_no_component_signal():
    data = make_cliff_study(np.array([0.0, 2.0, 0, 0, 0]), n_people=300)
    sim = simulate(data, position_only_null(data, SPEC), SPEC, np.random.default_rng(3))
    f = fit_model(sim, SPEC)
    assert abs(f.components[1] / f.se[1]) < 3.0


def test_placebo_p_value_is_add_one_smoothed():
    data = make_cliff_study(np.zeros(5), n_people=40, n_tasks=2)
    out = placebo_test(data, bind("pooled", SPEC, raw=True), spec=SPEC, n_reps=9, seed=1)
    assert out["reps"] == 9
    assert all(0.1 - 1e-9 <= p <= 1.0 for p in out["p_value"].values())


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_every_estimator_returns_a_simplex_or_zeros_and_a_raw_statistic(name):
    data = make_cliff_study(np.array([0.2, 1.0, 0.3, 0.8, 0.0]), n_people=120)
    kw = {"min_sets": 10} if "macro" in name else {}
    w = ESTIMATORS[name](data, SPEC, **kw)
    raw = ESTIMATORS[name](data, SPEC, raw=True, **kw)
    assert w.shape == (5,) and (w >= 0).all()
    assert w.sum() == pytest.approx(1.0) or w.sum() == 0.0
    assert (raw >= 0).all()


def test_real_effects_are_identified_and_a_zero_effect_is_not():
    # Randomised order and several questions: the design the gates exist for.
    data = make_cliff_study(np.array([0.0, 1.4, 0.0, 1.0, 0.0]), n_people=300, n_tasks=8,
                            random_order=True)
    rep = evaluate_gates(data, "pooled", spec=SPEC, config=FAST)
    assert "accessibility" in rep["identified_dimensions"]
    assert "economic" in rep["identified_dimensions"]
    for d in ("spatial", "facility", "disruption"):
        assert d not in rep["identified_dimensions"]
    assert rep["shippable"] is False


def test_nothing_is_identified_when_choices_follow_position_only():
    data = make_cliff_study(np.zeros(5), n_people=300, n_tasks=4, sort_dim=0)
    rep = evaluate_gates(data, "per_task_macro", estimator_kwargs={"min_sets": 10},
                         spec=SPEC, config=FAST)
    assert rep["identified_dimensions"] == []
    # The macro estimator still hands out simplex mass on noise, which is exactly
    # why a weight on its own is not evidence.
    assert sum(rep["weights"].values()) == pytest.approx(1.0)


def test_callable_estimator_requires_a_scale_bearing_statistic():
    data = make_cliff_study(np.zeros(5), n_people=30, n_tasks=2)
    with pytest.raises(ValueError, match="statistic"):
        evaluate_gates(data, bind("pooled", SPEC), spec=SPEC, config=FAST)


def test_deployable_vector_declares_the_prior_it_used():
    report = {"dimensions": {d: {"identified": d in ("spatial", "economic"),
                                 "weight": {"spatial": 0.6, "economic": 0.4}.get(d, 0.0)}
                             for d in DIMS}}
    prior = [0.25, 0.20, 0.25, 0.15, 0.15]
    out = deployable_vector(report, prior)
    w = out["weights"]
    assert sum(w.values()) == pytest.approx(1.0)
    assert out["estimated"] == ["spatial", "economic"]
    assert out["declared_from_prior"] == {"accessibility": 0.20, "facility": 0.25, "disruption": 0.15}
    assert w["spatial"] / w["economic"] == pytest.approx(1.5)


def test_gate_report_is_json_serialisable():
    import json
    data = make_cliff_study(np.array([0, 1.0, 0, 0, 0]), n_people=80, n_tasks=3)
    rep = evaluate_gates(data, "pooled", spec=SPEC,
                         config=GateConfig(placebo_reps=20, bootstrap_reps=5, cv_folds=3,
                                           cv_bootstrap_reps=20, min_loqo_groups=3))
    json.dumps(rep)


def test_too_few_placebo_replicates_for_alpha_is_refused():
    data = make_cliff_study(np.zeros(5), n_people=30, n_tasks=2)
    with pytest.raises(ValueError, match="cannot reach"):
        evaluate_gates(data, "pooled", spec=SPEC, config=GateConfig(placebo_reps=19))
