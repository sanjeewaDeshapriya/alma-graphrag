"""The three wave-1 fitters after the position-specification retrofit.

`fit_weights.apply_position_spec` must build exactly the design the shared core
builds, and the per-display-condition estimator must drop a stratum whose fit
collapses instead of averaging it in as 0.2 on every dimension.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.test_choice_model import make_cliff_study
from weight_elicitation.choice_model import DIMS, PositionSpec, build_design
from weight_elicitation.fit_human_weights import fit_strata
from weight_elicitation.fit_weights import ChoiceSets, apply_position_spec, fit_mnl, to_simplex


def to_choice_sets(data):
    X = np.concatenate([data.F, np.where(data.mask, -np.log(np.maximum(data.pos, 1)), 0.0)[:, :, None]],
                       axis=2)
    return ChoiceSets(X, data.mask, data.y, data.participants, data.sorts, data.tasks)


@pytest.mark.parametrize("spec", [PositionSpec("neglog"), PositionSpec("dummies", 3),
                                  PositionSpec("topk", 3)], ids=lambda s: s.label())
def test_apply_position_spec_matches_the_shared_design(spec):
    data = make_cliff_study(np.array([0.3, 1.0, 0.2, 0.6, 0.0]), n_people=60)
    cs = apply_position_spec(to_choice_sets(data), spec)
    d = build_design(data, spec)
    assert len(cs) == len(d.y)
    assert cs.X.shape[2] == d.X.shape[2]
    assert np.allclose(cs.X[cs.mask], d.X[d.mask])


def test_fit_mnl_accepts_any_number_of_position_columns():
    data = make_cliff_study(np.array([0.0, 1.2, 0.0, 0.8, 0.0]), n_people=200, n_tasks=8,
                            random_order=True)
    cs = apply_position_spec(to_choice_sets(data), PositionSpec("dummies", 3))
    beta = fit_mnl(cs, use_position=True, non_negative=False)
    assert len(beta) == cs.X.shape[2]
    assert np.argmax(to_simplex(beta)) == DIMS.index("accessibility")


def test_a_collapsed_stratum_is_dropped_not_averaged_as_flat():
    data = make_cliff_study(np.array([0.0, 1.5, 0.0, 0.0, 0.0]), n_people=300, n_tasks=6,
                            random_order=True)
    cs = to_choice_sets(data)
    # Two strata: one with signal, one whose components are identical (no information).
    sorts = np.where(np.arange(len(cs)) % 2 == 0, "distance", "rating")
    X = cs.X.copy()
    X[sorts == "rating", :, :5] = 0.5
    cs = ChoiceSets(X, cs.mask, cs.y, cs.participants, sorts, cs.tasks)
    cs = apply_position_spec(cs, PositionSpec("dummies", 3))
    per, macro = fit_strata(cs, l2=0.0)
    assert "rating" not in per
    assert not np.allclose(macro, 0.2)
    assert macro[DIMS.index("accessibility")] > 0.5
