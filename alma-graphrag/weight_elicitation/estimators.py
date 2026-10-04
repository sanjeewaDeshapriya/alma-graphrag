"""
Every weight estimator the project has used, as a plain `ChoiceData -> weights`
function, so the placebo test and the acceptance gates can run any of them.

    ESTIMATORS["pooled"](data, spec)            unconstrained pooled logit, clipped to the simplex
    ESTIMATORS["pooled_nonneg"](data, spec)     non-negative pooled logit
    ESTIMATORS["prior_map"](data, spec, l2=..)  fit_weights.py `deployed` (shrinks toward hand-set shape)
    ESTIMATORS["per_sort_macro"](data, spec)    fit_human_weights.py (per display condition, averaged)
    ESTIMATORS["per_task_macro"](data, spec)    fit_share_weights.py (per question, averaged)

Weights are returned on the simplex. An estimator whose every component is
pinned at zero returns all zeros — never 0.2 each, which would read as a finding.

`raw=True` returns the estimator's SCALE-BEARING statistic instead: the clipped,
un-normalised coefficients (averaged over groups for the macro estimators, with
degenerate groups counted as the zeros they are). The placebo gate must use this.
Normalising to the simplex discards magnitude, so on pure noise a clipped vector
normalises to weights of about 0.2 each and real weights of that size cannot
beat it: a simulated wave with true effects of 0.5-0.8 utility failed a
simplex-based placebo on every dimension. The un-normalised statistic is near
zero on noise and large on signal, which is the contrast a placebo needs.
"""
from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import numpy as np

from weight_elicitation.choice_model import (
    DEFAULT_POSITION,
    ChoiceData,
    PositionSpec,
    fit_model,
    to_simplex,
)

HANDSET = np.array([0.25, 0.20, 0.25, 0.15, 0.15])

#: fit_human_weights.py strata: price_asc and price_desc pooled (84 desc sets
#: will not fit alone).
SORT_STRATA: Dict[str, Tuple[str, ...]] = {
    "distance": ("distance",),
    "travel": ("travel",),
    "rating": ("rating",),
    "price": ("price_asc", "price_desc"),
}


def _out(beta: np.ndarray, raw: bool) -> np.ndarray:
    clipped = np.clip(np.asarray(beta[:5], float), 0.0, None)
    return clipped if raw else to_simplex(clipped)


def pooled(data: ChoiceData, spec: PositionSpec = DEFAULT_POSITION, raw: bool = False,
           **_) -> np.ndarray:
    return _out(fit_model(data, spec, with_se=False).components, raw)


def pooled_nonneg(data: ChoiceData, spec: PositionSpec = DEFAULT_POSITION,
                  l2: float = 0.0, raw: bool = False, **_) -> np.ndarray:
    return _out(fit_model(data, spec, non_negative=True, l2=l2, with_se=False).components, raw)


def prior_map(data: ChoiceData, spec: PositionSpec = DEFAULT_POSITION,
              l2: float = 1.0, prior: Sequence[float] = HANDSET, raw: bool = False,
              **_) -> np.ndarray:
    return _out(fit_model(data, spec, non_negative=True, l2=l2,
                          prior_direction=np.asarray(prior, float),
                          with_se=False).components, raw)


def _macro(data: ChoiceData, spec: PositionSpec, groups: Dict[str, np.ndarray],
           min_sets: int, l2: float, raw: bool) -> np.ndarray:
    betas, ws = [], []
    for idx in groups.values():
        if len(idx) < min_sets:
            continue
        b = np.clip(fit_model(data.subset(idx), spec, non_negative=True, l2=l2,
                              with_se=False).components, 0.0, None)
        betas.append(b)
        if b.sum() > 0:                   # degenerate group: dropped, not averaged as flat
            ws.append(to_simplex(b))
    if raw:
        return np.mean(betas, axis=0) if betas else np.zeros(5)
    return np.mean(ws, axis=0) if ws else np.zeros(5)


def per_sort_macro(data: ChoiceData, spec: PositionSpec = DEFAULT_POSITION,
                   l2: float = 1.0, min_sets: int = 100, raw: bool = False, **_) -> np.ndarray:
    groups = {name: np.where(np.isin(data.sorts, list(modes)))[0]
              for name, modes in SORT_STRATA.items()}
    return _macro(data, spec, groups, min_sets, l2, raw)


def per_task_macro(data: ChoiceData, spec: PositionSpec = DEFAULT_POSITION,
                   l2: float = 0.0, min_sets: int = 30, raw: bool = False, **_) -> np.ndarray:
    groups = {t: np.where(data.tasks == t)[0] for t in np.unique(data.tasks)}
    return _macro(data, spec, groups, min_sets, l2, raw)


ESTIMATORS: Dict[str, Callable[..., np.ndarray]] = {
    "pooled": pooled,
    "pooled_nonneg": pooled_nonneg,
    "prior_map": prior_map,
    "per_sort_macro": per_sort_macro,
    "per_task_macro": per_task_macro,
}


def bind(name: str, spec: PositionSpec, **kw) -> Callable[[ChoiceData], np.ndarray]:
    """Fix the spec and hyper-parameters so the estimator is `data -> weights`."""
    fn = ESTIMATORS[name]
    return lambda data: fn(data, spec, **kw)
