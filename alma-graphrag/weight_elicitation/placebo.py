"""
Placebo test: does an estimator find weights in data that contain none?

    from weight_elicitation.placebo import position_only_null, placebo_test

Every participant keeps exactly the list they saw — same hotels, same
positions — but their choices are re-drawn from a model with NO component
effect (only the fitted position effect). The estimator under test is then run
on each simulated dataset. Its output on those datasets is the distribution a
weight takes when there is nothing to find, and the observed weight is compared
against it.

This is the check that caught the wave-1 share estimator: on position-only
placebo data it returned economic .233 and facility .289 on average, MORE than
the real data gave. Clipping each question's coefficients at zero and
renormalising turns noise into simplex corners, and the average of corners
cannot be zero, so its bootstrap interval excluded zero by construction.

A second null, `restricted_null`, keeps the fitted effects of some dimensions
and zeroes others. It asks the sharper question "is this weight what we would
see if THIS dimension had no effect but everything else stayed as estimated?".
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np

from weight_elicitation.choice_model import (
    DEFAULT_POSITION,
    DIMS,
    ChoiceData,
    PositionSpec,
    build_design,
    fit_logit,
)

Estimator = Callable[[ChoiceData], np.ndarray]


def _full_names(data: ChoiceData, spec: PositionSpec):
    return build_design(data, spec)


def restricted_null(data: ChoiceData, spec: PositionSpec = DEFAULT_POSITION,
                    zero: Sequence[str] = DIMS) -> np.ndarray:
    """Maximum-likelihood coefficients with the named dimensions pinned at 0.

    Returned aligned to the FULL design (components then position), ready for
    `simulate`. `zero=DIMS` is the position-only null.
    """
    d = _full_names(data, spec)
    bounds = [(0.0, 0.0) if nm in zero else (None, None) for nm in d.names]
    beta, _, _ = fit_logit(d.X, d.mask, d.y, bounds=bounds)
    for i, nm in enumerate(d.names):
        if nm in zero:
            beta[i] = 0.0
    return beta


def position_only_null(data: ChoiceData, spec: PositionSpec = DEFAULT_POSITION) -> np.ndarray:
    return restricted_null(data, spec, DIMS)


def simulate(data: ChoiceData, beta: np.ndarray, spec: PositionSpec,
             rng: np.random.Generator) -> ChoiceData:
    """Re-draw every choice from the logit with coefficients `beta`.

    Rows the position spec drops (chosen alternative outside top-k) are removed
    first, so the simulated dataset has exactly the choice sets the model sees.
    """
    d = build_design(data, spec)
    if len(beta) != d.X.shape[2]:
        raise ValueError("beta does not match the design; build it with restricted_null")
    u = np.where(d.mask, d.X @ beta, -np.inf)
    u = u - u.max(axis=1, keepdims=True)
    p = np.where(d.mask, np.exp(u), 0.0)
    p /= p.sum(axis=1, keepdims=True)
    cum = p.cumsum(axis=1)
    draw = rng.random((len(d.y), 1))
    y = np.minimum((cum < draw).sum(axis=1), p.shape[1] - 1)
    return data.subset(d.rows).with_choices(y)


def placebo_test(data: ChoiceData, estimator: Estimator, *,
                 observed: Optional[np.ndarray] = None,
                 null_beta: Optional[np.ndarray] = None,
                 spec: PositionSpec = DEFAULT_POSITION,
                 n_reps: int = 200, seed: int = 20260913) -> Dict[str, object]:
    """Run `estimator` on `n_reps` simulated null datasets.

    p-value per dimension is one-sided and add-one smoothed,
    (1 + #{null >= observed}) / (1 + reps), so it is never exactly 0 and a
    two-hundred-replicate run cannot claim more than p = 0.005.
    """
    if observed is None:
        observed = np.asarray(estimator(data), float)
    if null_beta is None:
        null_beta = position_only_null(data, spec)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_reps):
        try:
            draws.append(np.asarray(estimator(simulate(data, null_beta, spec, rng)), float))
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
    D = np.array(draws) if draws else np.zeros((0, len(observed)))
    ge = (D >= observed[None, :] - 1e-12).sum(axis=0) if len(D) else np.zeros(len(observed))
    p = (1 + ge) / (1 + len(D))
    names = list(DIMS) if len(observed) == 5 else [f"x{i}" for i in range(len(observed))]
    return {
        "reps": int(len(D)),
        "position_spec": spec.label(),
        "observed": dict(zip(names, map(float, observed))),
        "null_mean": dict(zip(names, map(float, D.mean(axis=0)))) if len(D) else {},
        "null_p95": dict(zip(names, map(float, np.percentile(D, 95, axis=0)))) if len(D) else {},
        "p_value": dict(zip(names, map(float, p))),
        "draws": D,
    }
