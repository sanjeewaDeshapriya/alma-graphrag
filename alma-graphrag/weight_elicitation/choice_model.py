"""
The conditional-logit machinery every fitter in this package shares.

    from weight_elicitation.choice_model import (
        PositionSpec, choice_data_from_responses, fit_model, lr_test_components,
        cluster_bootstrap, cross_validate)

Why this module exists
----------------------
The 2026-09-13 audit (docs/Weight_Elicitation_Share_Audit.md) found that all
three wave-1 fitters controlled list position with a single -log(position)
column. The data do not have that shape: 45.8% / 32.6% / 21.0% of choices went
to positions 1 / 2 / 3 and 13 of 2,232 went anywhere below, in every sort mode.
A smooth -log curve over-predicts position 1 by ~17 points and under-predicts
2-3, and whichever component tracks "rows 2-3 rather than 4+" (the sort key
itself) absorbs the misfit and is booked as a preference. Under -log every
per-question likelihood-ratio test was significant; under position dummies
almost none were.

So position is now a first-class, explicit modelling choice, `PositionSpec`,
and there is ONE implementation of the likelihood, its gradient, its Hessian,
the participant-clustered bootstrap, the likelihood-ratio test and the
participant-level cross-validation. The fitters, the placebo test and the
acceptance gates all call into it, so they cannot drift apart.

Position specifications
-----------------------
    none      no position term (only for describing what an uncontrolled fit does)
    neglog    one column, -log(position). Kept for reproducing wave-1 numbers.
    dummies   one indicator per position 2..k plus one for "beyond k"
              (position 1 is the reference). The default: it is saturated where
              the choices are and pools the empty tail, so it cannot misplace
              probability the way a parametric curve does.
    topk      restrict every set to positions 1..k (the consideration set the
              participant actually weighed) with indicators for 2..k. Rows whose
              chosen alternative sat below k are dropped and counted.

Columns that are identically zero on the rows in play are dropped rather than
left to produce a singular Hessian, and their names go with them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import minimize

DIMS: Tuple[str, ...] = ("spatial", "accessibility", "facility", "economic", "disruption")
POSITION_KINDS = ("none", "neglog", "dummies", "topk")


# --------------------------------------------------------------------------- #
# Position
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PositionSpec:
    kind: str = "dummies"
    k: int = 3

    def __post_init__(self) -> None:
        if self.kind not in POSITION_KINDS:
            raise ValueError(f"unknown position spec {self.kind!r}; use one of {POSITION_KINDS}")
        if self.kind in ("dummies", "topk") and self.k < 1:
            raise ValueError("position k must be >= 1")

    @classmethod
    def parse(cls, text: str) -> "PositionSpec":
        """`dummies`, `dummies:4`, `topk:3`, `neglog`, `none`."""
        kind, _, k = text.partition(":")
        return cls(kind.strip(), int(k) if k else 3)

    def label(self) -> str:
        return self.kind if self.kind in ("none", "neglog") else f"{self.kind}:{self.k}"

    def columns(self, pos: np.ndarray, mask: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Position features (n, m, p), the possibly-restricted mask, and names."""
        pos = np.where(mask, pos, 0).astype(int)
        if self.kind == "none":
            return np.zeros(pos.shape + (0,)), mask, []
        if self.kind == "neglog":
            col = np.where(mask, -np.log(np.maximum(pos, 1)), 0.0)
            return col[:, :, None], mask, ["neg_log_position"]
        cols, names = [], []
        for r in range(2, self.k + 1):
            cols.append((pos == r).astype(float))
            names.append(f"position_{r}")
        if self.kind == "dummies":
            cols.append((pos > self.k).astype(float))
            names.append(f"position_gt_{self.k}")
            new_mask = mask
        else:
            new_mask = mask & (pos >= 1) & (pos <= self.k)
        P = np.stack(cols, axis=2) if cols else np.zeros(pos.shape + (0,))
        return np.where(new_mask[:, :, None], P, 0.0), new_mask, names


DEFAULT_POSITION = PositionSpec("dummies", 3)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class ChoiceData:
    """Padded choice sets with everything a model or a resample needs.

    F      (n, m, 5) component vectors, in DIMS order
    pos    (n, m)    displayed position, 1-based (0 on padding)
    mask   (n, m)    alternative was on screen
    y      (n,)      index of the chosen alternative
    participants, tasks, sorts (n,) grouping labels
    extra  optional (n, m, q) observed attributes (e.g. price in LKR) with names
    """
    F: np.ndarray
    pos: np.ndarray
    mask: np.ndarray
    y: np.ndarray
    participants: np.ndarray
    tasks: np.ndarray
    sorts: np.ndarray
    extra: Optional[np.ndarray] = None
    extra_names: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx: np.ndarray) -> "ChoiceData":
        idx = np.asarray(idx)
        return ChoiceData(self.F[idx], self.pos[idx], self.mask[idx], self.y[idx],
                          self.participants[idx], self.tasks[idx], self.sorts[idx],
                          None if self.extra is None else self.extra[idx],
                          list(self.extra_names))

    def with_choices(self, y: np.ndarray) -> "ChoiceData":
        out = self.subset(np.arange(len(self)))
        out.y = np.asarray(y, dtype=int)
        return out

    def rows_by_participant(self) -> Dict[str, np.ndarray]:
        people = np.unique(self.participants)
        return {p: np.where(self.participants == p)[0] for p in people}


def choice_data_from_responses(responses: Iterable[dict],
                               facility: Optional[Dict[str, float]] = None,
                               pool_size: Optional[int] = None,
                               exclude_participants: Optional[set] = None,
                               extra: Optional[Dict[str, Callable[[dict], float]]] = None,
                               ) -> ChoiceData:
    """One choice set per response, from the raw `?format=raw` dump rows.

    Only alternatives that were on screen (non-null `displayed_position`) and
    carry a component vector enter. A hotel filtered out of view was never
    rejected, so treating it as rejected would bias every coefficient.
    `facility` optionally overrides the stored facility component (the wave-1
    ceiling repair). `extra` maps a name to a function of the option dict, for
    observed attributes such as price.
    """
    facility = facility or {}
    exclude_participants = exclude_participants or set()
    extra = extra or {}
    rows, positions, labels, pids, tasks, sorts, extras = [], [], [], [], [], [], []
    for r in responses:
        if r.get("isAttentionCheck"):
            continue
        if pool_size is not None and len(r.get("options") or []) != pool_size:
            continue
        if r.get("participantId") in exclude_participants:
            continue
        opts = [o for o in (r.get("options") or [])
                if o.get("displayed_position") is not None and o.get("components")]
        if len(opts) < 2 or sum(1 for o in opts if o.get("chosen")) != 1:
            continue
        rows.append(np.array([[o["components"]["spatial"], o["components"]["accessibility"],
                               facility.get(o["hotel_id"], o["components"]["facility"]),
                               o["components"]["economic"], o["components"]["disruption"]]
                              for o in opts], dtype=float))
        positions.append(np.array([int(o["displayed_position"]) for o in opts]))
        labels.append(next(i for i, o in enumerate(opts) if o.get("chosen")))
        pids.append(r.get("participantId"))
        tasks.append(r.get("taskId"))
        sorts.append((r.get("timing") or {}).get("final_sort") or "unknown")
        if extra:
            extras.append(np.array([[fn(o) for fn in extra.values()] for o in opts], dtype=float))
    if not rows:
        raise ValueError("no usable choice sets in these responses")
    n, m = len(rows), max(len(a) for a in rows)
    F = np.zeros((n, m, 5))
    P = np.zeros((n, m), dtype=int)
    M = np.zeros((n, m), dtype=bool)
    E = np.zeros((n, m, len(extra))) if extra else None
    for i, a in enumerate(rows):
        F[i, :len(a)] = a
        P[i, :len(a)] = positions[i]
        M[i, :len(a)] = True
        if E is not None:
            E[i, :len(a)] = extras[i]
    return ChoiceData(F, P, M, np.array(labels), np.array(pids), np.array(tasks),
                      np.array(sorts), E, list(extra))


def choice_data_from_choice_sets(cs) -> ChoiceData:
    """Adapter for `fit_weights.ChoiceSets`, whose column 5 is -log(position)."""
    pos = np.where(cs.mask, np.rint(np.exp(-cs.X[:, :, 5])), 0).astype(int)
    tasks = getattr(cs, "tasks", None)
    if tasks is None:
        tasks = np.array(["?"] * len(cs.y))
    return ChoiceData(cs.X[:, :, :5].copy(), pos, cs.mask.copy(), cs.y.copy(),
                      cs.participants.copy(), np.asarray(tasks), cs.sorts.copy())


# --------------------------------------------------------------------------- #
# Design matrices
# --------------------------------------------------------------------------- #
@dataclass
class Design:
    X: np.ndarray            # (n', m, k)
    mask: np.ndarray         # (n', m)
    y: np.ndarray            # (n',)
    rows: np.ndarray         # indices into the ChoiceData rows kept
    names: List[str]
    n_components: int        # leading columns that are components (0 or 5)
    n_dropped: int           # rows lost to a topk restriction

    @property
    def component_slice(self) -> slice:
        return slice(0, self.n_components)


def build_design(data: ChoiceData, spec: PositionSpec = DEFAULT_POSITION, *,
                 components: bool = True, task_specific: bool = False,
                 score_weights: Optional[np.ndarray] = None,
                 extra: bool = False) -> Design:
    """Stack components (or a fixed score), extra attributes and position.

    `score_weights` replaces the five component columns with ONE column,
    F @ w, so a fixed weight profile can be scored with only a scale and the
    position terms estimated — the fair way to ask whether a profile predicts
    held-out choices better than position alone.
    """
    P, mask, pnames = spec.columns(data.pos, data.mask)
    keep = mask[np.arange(len(data)), data.y]
    rows = np.where(keep)[0]
    mask = mask[rows]
    parts, names, n_comp = [], [], 0
    if score_weights is not None:
        parts.append((data.F[rows] @ np.asarray(score_weights, float))[:, :, None])
        names.append("score")
        n_comp = 1
    elif components and task_specific:
        labels = sorted(set(data.tasks.tolist()))
        T = {t: i for i, t in enumerate(labels)}
        Xt = np.zeros((len(rows), data.F.shape[1], 5 * len(labels)))
        for i, r in enumerate(rows):
            j = T[data.tasks[r]]
            Xt[i, :, 5 * j:5 * j + 5] = data.F[r]
        parts.append(Xt)
        names += [f"{t}:{d}" for t in labels for d in DIMS]
        n_comp = 5 * len(labels)
    elif components:
        parts.append(data.F[rows])
        names += list(DIMS)
        n_comp = 5
    if extra and data.extra is not None:
        parts.append(data.extra[rows])
        names += list(data.extra_names)
    Pr = P[rows]
    if Pr.shape[2]:
        live = np.array([np.any(Pr[:, :, j][mask]) for j in range(Pr.shape[2])], dtype=bool)
        Pr = Pr[:, :, live]
        pnames = [nm for nm, ok in zip(pnames, live) if ok]
    parts.append(Pr)
    names += pnames
    X = np.concatenate(parts, axis=2) if parts else np.zeros(mask.shape + (0,))
    X = np.where(mask[:, :, None], X, 0.0)
    return Design(X, mask, data.y[rows], rows, names, n_comp, int(len(data) - len(rows)))


# --------------------------------------------------------------------------- #
# Likelihood
# --------------------------------------------------------------------------- #
def loglik(beta: np.ndarray, X: np.ndarray, mask: np.ndarray, y: np.ndarray
           ) -> Tuple[float, np.ndarray, np.ndarray]:
    """Log-likelihood, its gradient, and the choice probabilities."""
    n = len(y)
    u = np.where(mask, X @ beta if X.shape[2] else 0.0, -np.inf)
    mx = u.max(axis=1, keepdims=True)
    e = np.where(mask, np.exp(u - mx), 0.0)
    s = e.sum(axis=1, keepdims=True)
    p = e / s
    ll = float((u[np.arange(n), y] - (np.log(s[:, 0]) + mx[:, 0])).sum())
    if X.shape[2] == 0:
        return ll, np.zeros(0), p
    grad = X[np.arange(n), y].sum(axis=0) - np.einsum("nh,nhk->k", p, X)
    return ll, grad, p


def per_choice_loglik(beta: np.ndarray, X: np.ndarray, mask: np.ndarray,
                      y: np.ndarray) -> np.ndarray:
    _, _, p = loglik(beta, X, mask, y)
    return np.log(np.maximum(p[np.arange(len(y)), y], 1e-300))


def fit_logit(X: np.ndarray, mask: np.ndarray, y: np.ndarray, *,
              bounds: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,
              l2: float = 0.0, l2_idx: Optional[np.ndarray] = None,
              prior_direction: Optional[np.ndarray] = None,
              start: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, bool]:
    """Maximum likelihood with an optional penalty on the `l2_idx` block.

    Plain ridge penalises ||b||^2. With `prior_direction` h (unit vector) it
    penalises ||b||^2 - (b.h)^2 instead: only the part of b orthogonal to the
    prior is shrunk, so lambda -> inf returns the prior's SHAPE rather than
    zero (the estimator `fit_weights.py` ships as `deployed`).
    """
    k = X.shape[2]
    if k == 0:
        return np.zeros(0), loglik(np.zeros(0), X, mask, y)[0], True
    idx = np.arange(k) if l2_idx is None else np.asarray(l2_idx)
    h = None
    if prior_direction is not None:
        h = np.asarray(prior_direction, float)
        h = h / np.linalg.norm(h)

    def obj(b):
        ll, g, _ = loglik(b, X, mask, y)
        f, gr = -ll, -g
        if l2:
            bi = b[idx]
            gr = gr.copy()
            if h is None:
                f += l2 * float(bi @ bi)
                gr[idx] += 2 * l2 * bi
            else:
                proj = float(bi @ h)
                f += l2 * (float(bi @ bi) - proj ** 2)
                gr[idx] += 2 * l2 * (bi - proj * h)
        return f, gr

    res = minimize(obj, np.zeros(k) if start is None else start, jac=True,
                   method="L-BFGS-B", bounds=bounds, options={"maxiter": 2000})
    beta = res.x
    return beta, loglik(beta, X, mask, y)[0], bool(res.success)


def information_matrix(beta: np.ndarray, X: np.ndarray, mask: np.ndarray,
                       y: np.ndarray) -> np.ndarray:
    _, _, p = loglik(beta, X, mask, y)
    xbar = np.einsum("nh,nhk->nk", p, X)
    return np.einsum("nh,nhk,nhl->kl", p, X, X) - xbar.T @ xbar


def model_se(beta: np.ndarray, X: np.ndarray, mask: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Observed-information standard errors (assume independent choices)."""
    cov = np.linalg.pinv(information_matrix(beta, X, mask, y))
    return np.sqrt(np.clip(np.diag(cov), 0.0, None))


@dataclass
class ModelFit:
    spec: PositionSpec
    names: List[str]
    beta: np.ndarray
    se: np.ndarray
    ll: float
    n: int
    n_dropped: int
    converged: bool
    non_negative: bool

    def coef(self, name: str) -> float:
        return float(self.beta[self.names.index(name)])

    @property
    def components(self) -> np.ndarray:
        return np.array([self.coef(d) for d in DIMS]) if DIMS[0] in self.names else np.zeros(5)

    def as_dict(self) -> dict:
        return {"position_spec": self.spec.label(), "n": self.n, "n_dropped": self.n_dropped,
                "log_likelihood": self.ll, "converged": self.converged,
                "non_negative": self.non_negative,
                "coefficients": {nm: {"beta": float(b), "se": float(s)}
                                 for nm, b, s in zip(self.names, self.beta, self.se)}}


def fit_model(data: ChoiceData, spec: PositionSpec = DEFAULT_POSITION, *,
              components: bool = True, non_negative: bool = False, l2: float = 0.0,
              prior_direction: Optional[np.ndarray] = None,
              task_specific: bool = False, score_weights: Optional[np.ndarray] = None,
              extra: bool = False, with_se: bool = True) -> ModelFit:
    d = build_design(data, spec, components=components, task_specific=task_specific,
                     score_weights=score_weights, extra=extra)
    k = d.X.shape[2]
    bounds = None
    if non_negative and d.n_components:
        bounds = [(0.0, None)] * d.n_components + [(None, None)] * (k - d.n_components)
    l2_idx = np.arange(d.n_components) if (l2 and d.n_components) else None
    beta, ll, ok = fit_logit(d.X, d.mask, d.y, bounds=bounds, l2=l2, l2_idx=l2_idx,
                             prior_direction=prior_direction if d.n_components == 5 else None)
    se = model_se(beta, d.X, d.mask, d.y) if (with_se and k) else np.full(k, np.nan)
    return ModelFit(spec, d.names, beta, se, ll, len(d.y), d.n_dropped, ok, non_negative)


def to_simplex(beta: np.ndarray) -> np.ndarray:
    """Clip to non-negative and normalise. All-zero stays all-zero (NOT 0.2 each):
    a flat vector would read as a finding when it is the absence of one."""
    w = np.clip(np.asarray(beta[:5], float), 0.0, None)
    s = w.sum()
    return w / s if s > 0 else np.zeros(5)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def lr_test_components(data: ChoiceData, spec: PositionSpec = DEFAULT_POSITION) -> dict:
    """Do the five components add anything once position is modelled?"""
    full = fit_model(data, spec, with_se=False)
    base = fit_model(data, spec, components=False, with_se=False)
    null = float(-np.log(build_design(data, spec, components=False).mask.sum(axis=1)).sum())
    lr = max(0.0, 2.0 * (full.ll - base.ll))
    return {"position_spec": spec.label(), "n": full.n, "df": 5,
            "ll_null": null, "ll_position": base.ll, "ll_full": full.ll,
            "lr": lr, "p": float(stats.chi2.sf(lr, 5)),
            "mcfadden_position": 1 - base.ll / null if null else float("nan"),
            "mcfadden_full": 1 - full.ll / null if null else float("nan")}


def cluster_bootstrap(data: ChoiceData, estimator: Callable[[ChoiceData], np.ndarray],
                      n_reps: int, seed: int) -> np.ndarray:
    """Resample PARTICIPANTS with replacement; each contributed ~10 correlated
    choices, so resampling rows would shrink intervals to fiction."""
    rng = np.random.default_rng(seed)
    by = data.rows_by_participant()
    people = np.array(list(by))
    draws = []
    for _ in range(n_reps):
        pick = rng.choice(people, size=len(people), replace=True)
        idx = np.concatenate([by[p] for p in pick])
        try:
            draws.append(np.asarray(estimator(data.subset(idx)), float))
        except (ValueError, np.linalg.LinAlgError, RuntimeError):
            continue
    return np.array(draws)


def percentile_ci(draws: np.ndarray, level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    a = (1 - level) / 2
    return np.percentile(draws, 100 * a, axis=0), np.percentile(draws, 100 * (1 - a), axis=0)


def participant_folds(data: ChoiceData, n_folds: int, seed: int) -> List[np.ndarray]:
    people = np.unique(data.participants)
    perm = np.random.default_rng(seed).permutation(people)
    return [np.where(np.isin(data.participants, f))[0] for f in np.array_split(perm, n_folds)]


def cross_validate(data: ChoiceData, models: Dict[str, dict],
                   spec: PositionSpec = DEFAULT_POSITION, n_folds: int = 5,
                   seed: int = 3) -> Dict[str, dict]:
    """Held-out log-likelihood per choice, split by PARTICIPANT.

    `models` maps a name to keyword arguments for `fit_model` (e.g.
    `{"components": False}` for position only, `{"score_weights": w}` for a fixed
    profile). Every model is scored on the same held-out choices (the rows the
    position spec keeps), so per-choice differences are paired.
    """
    base = build_design(data, spec, components=False)
    keep = np.zeros(len(data), dtype=bool)
    keep[base.rows] = True
    out = {name: np.full(len(data), np.nan) for name in models}
    top1 = {name: np.full(len(data), np.nan) for name in models}
    for te in participant_folds(data, n_folds, seed):
        tr = np.setdiff1d(np.arange(len(data)), te)
        for name, kw in models.items():
            fitted = fit_model(data.subset(tr), spec, with_se=False, **kw)
            d = build_design(data.subset(te), spec, **{k: v for k, v in kw.items()
                                                      if k in ("components", "task_specific",
                                                               "score_weights", "extra")})
            if d.X.shape[2] != len(fitted.beta):
                # a position column absent from this fold's training rows
                beta = np.zeros(d.X.shape[2])
                for j, nm in enumerate(d.names):
                    if nm in fitted.names:
                        beta[j] = fitted.beta[fitted.names.index(nm)]
            else:
                beta = fitted.beta
            _, _, p = loglik(beta, d.X, d.mask, d.y)
            rows = te[d.rows]
            out[name][rows] = np.log(np.maximum(p[np.arange(len(d.y)), d.y], 1e-300))
            top1[name][rows] = (np.argmax(np.where(d.mask, p, -1.0), axis=1) == d.y).astype(float)
    result = {}
    for name in models:
        v = out[name][keep]
        result[name] = {"ll_per_choice": float(np.nanmean(v)),
                        "perplexity": float(math.exp(-np.nanmean(v))),
                        "top1": float(np.nanmean(top1[name][keep])),
                        "n": int(np.isfinite(v).sum()),
                        "per_choice": out[name]}
    return result


def paired_participant_bootstrap(diff: np.ndarray, participants: np.ndarray,
                                 n_reps: int, seed: int) -> Tuple[float, float, float]:
    """Mean per-choice difference with a participant-clustered 95% interval."""
    ok = np.isfinite(diff)
    diff, participants = diff[ok], participants[ok]
    people = np.unique(participants)
    sums = np.array([diff[participants == p].sum() for p in people])
    counts = np.array([(participants == p).sum() for p in people])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_reps):
        i = rng.integers(0, len(people), len(people))
        draws.append(sums[i].sum() / counts[i].sum())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def wilson_interval(k: float, n: float, z: float = 1.959964) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)
