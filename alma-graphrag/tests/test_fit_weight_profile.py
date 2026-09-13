"""The fitted weight profile must be re-derivable, not just plausible.

`scripts/fit_weight_profile.py` learns the composite weight vector by maximising
nDCG@10 over the simplex. A hand-picked vector cannot be checked by a reviewer;
a fitted one can only be checked if the fit is deterministic. These tests pin
that down on a stub environment, so they run offline — no Neo4j, no pgvector,
no LLM.

The stub replaces `RankingEnv.reward` with a fixed synthetic objective. That is
enough: the search, the blending and the fold construction are what must be
reproducible, and none of them know where the reward came from.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.fit_weight_profile import (
    LAMBDA_GRID,
    as_vec,
    blend,
    fit_vector,
    k_folds,
    literature_prior,
    mean_ndcg,
    normalise,
)
from src.crag.query_parser import QueryIntent
from src.graph.retriever import WEIGHT_PROFILES
from src.graph.weight_policy import COMPONENTS


class StubEnv:
    """A deterministic, non-trivial objective with a known optimum.

    The reward peaks at `target`, so a working search must approach it, and a
    reproducible search must approach it identically every time.
    """

    def __init__(self, target):
        self.target = normalise(np.asarray(target, dtype=float))
        self.calls = 0

    def reward(self, row, weights):
        self.calls += 1
        w = np.asarray(weights, dtype=float)
        # Negative squared distance, shifted positive; row acts as a mild tilt
        # so that different "queries" disagree, as real ones do.
        tilt = np.zeros(len(COMPONENTS))
        tilt[row["i"] % len(COMPONENTS)] = 0.02
        return float(1.0 - np.sum((w - self.target - tilt) ** 2))


# A neutral QueryIntent triggers no rung of the intent ladder, so the stub's
# objective is untouched — but the rows still carry one, which means these tests
# exercise the same `effective_weights` path the real fit uses. Without it the
# suite would pass while the fitter was ranking through a different function
# than the retriever, which is precisely the bug this indirection exists to stop.
ROWS = [{"i": i, "intent": QueryIntent()} for i in range(12)]


# ---------------------------------------------------------------------------
# Reproducibility — the point of the whole script
# ---------------------------------------------------------------------------

def test_fit_is_bit_identical_at_the_same_seed():
    env = StubEnv([0.10, 0.15, 0.30, 0.30, 0.15])
    a = fit_vector(env, ROWS, n_samples=200, rng=np.random.default_rng(7))
    b = fit_vector(env, ROWS, n_samples=200, rng=np.random.default_rng(7))
    assert np.array_equal(a, b), "same seed must give a bit-identical vector"


def test_fit_differs_at_a_different_seed_but_stays_close():
    """Different seeds explore differently; they must still find the same peak.

    If two seeds disagreed materially the reported vector would be an artefact
    of the seed rather than of the data.
    """
    env = StubEnv([0.10, 0.15, 0.30, 0.30, 0.15])
    a = fit_vector(env, ROWS, n_samples=400, rng=np.random.default_rng(1))
    b = fit_vector(env, ROWS, n_samples=400, rng=np.random.default_rng(2))
    assert np.abs(a - b).max() < 0.05


def test_folds_are_deterministic_and_partition_exactly():
    f1 = k_folds(74, 5, np.random.default_rng(3))
    f2 = k_folds(74, 5, np.random.default_rng(3))
    assert [x.tolist() for x in f1] == [x.tolist() for x in f2]
    allidx = sorted(int(i) for f in f1 for i in f)
    assert allidx == list(range(74)), "folds must partition every query exactly once"


# ---------------------------------------------------------------------------
# The search actually searches
# ---------------------------------------------------------------------------

def test_fit_recovers_a_known_optimum():
    target = [0.05, 0.10, 0.35, 0.35, 0.15]
    env = StubEnv(target)
    w = fit_vector(env, ROWS, n_samples=600, rng=np.random.default_rng(11))
    assert np.abs(w - normalise(np.array(target))).max() < 0.06


def test_fit_never_returns_worse_than_the_incumbent_profiles():
    """The named profiles are seeded into the candidate set on purpose."""
    env = StubEnv([0.20, 0.20, 0.20, 0.20, 0.20])
    w = fit_vector(env, ROWS, n_samples=100, rng=np.random.default_rng(5))
    got = mean_ndcg(env, ROWS, w)
    for name, prof in WEIGHT_PROFILES.items():
        assert got >= mean_ndcg(env, ROWS, as_vec(prof)) - 1e-9, \
            f"search returned something worse than the {name} profile"


# ---------------------------------------------------------------------------
# Simplex and blending invariants
# ---------------------------------------------------------------------------

def test_fit_returns_a_point_on_the_simplex():
    env = StubEnv([0.10, 0.15, 0.30, 0.30, 0.15])
    w = fit_vector(env, ROWS, n_samples=200, rng=np.random.default_rng(9))
    assert w.min() >= 0.0
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("lam", LAMBDA_GRID)
def test_blend_stays_on_the_simplex(lam):
    fit = normalise(np.array([0.5, 0.2, 0.1, 0.1, 0.1]))
    out = blend(fit, literature_prior(), lam)
    assert out.min() >= 0.0
    assert out.sum() == pytest.approx(1.0, abs=1e-9)


def test_blend_endpoints_are_the_pure_components():
    fit = normalise(np.array([0.5, 0.2, 0.1, 0.1, 0.1]))
    prior = literature_prior()
    assert blend(fit, prior, 0.0) == pytest.approx(fit, abs=1e-12)
    assert blend(fit, prior, 1.0) == pytest.approx(prior, abs=1e-12)


def test_blend_is_monotone_between_the_endpoints():
    """lambda has to mean something: each step must move toward the prior."""
    fit = normalise(np.array([0.6, 0.2, 0.1, 0.05, 0.05]))
    prior = literature_prior()
    d = [float(np.abs(blend(fit, prior, L) - prior).sum()) for L in LAMBDA_GRID]
    assert d == sorted(d, reverse=True)


# ---------------------------------------------------------------------------
# The literature prior is what it claims to be
# ---------------------------------------------------------------------------

def test_literature_prior_is_on_the_simplex():
    v = literature_prior()
    assert v.min() >= 0.0
    assert v.sum() == pytest.approx(1.0, abs=1e-9)


def test_literature_prior_reserves_disruption_at_the_hand_set_value():
    """Neither the conjoint literature nor the choice study addresses it."""
    assert literature_prior()[COMPONENTS.index("disruption")] == pytest.approx(0.150)


def test_literature_prior_puts_price_and_quality_above_either_location_dim():
    v = literature_prior()
    eco = v[COMPONENTS.index("economic")]
    fac = v[COMPONENTS.index("facility")]
    for dim in ("spatial", "accessibility"):
        assert eco > v[COMPONENTS.index(dim)]
        assert fac > v[COMPONENTS.index(dim)]
