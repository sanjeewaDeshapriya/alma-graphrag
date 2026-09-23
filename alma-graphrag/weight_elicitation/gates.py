"""
Acceptance gates: when may a human-elicited weight be quoted or shipped?

    from weight_elicitation.gates import GateConfig, evaluate_gates, deployable_vector

Pre-registered in docs/Weight_Elicitation_Share_Audit.md before wave 2 is
collected. A dimension's weight is IDENTIFIED only if all five hold:

  G1 placebo          the shipped estimator's SCALE-BEARING statistic (clipped,
                      un-normalised coefficients; estimators.py `raw=True`)
                      beats position-only placebo data: one-sided p < alpha.
                      Never the simplex weights: normalising removes the
                      magnitude a placebo needs to see.
  G2 likelihood ratio the five components jointly improve on position alone:
                      chi2(5) p < alpha. (Global — fails every dimension.)
  G3 interval         the UNCONSTRAINED pooled coefficient's participant-
                      clustered 95% CI lies entirely above zero. A clipped or
                      averaged estimator cannot pass this by construction.
  G4 held-out         adding the components raises held-out log-likelihood per
                      choice over position alone, participant-clustered 95% CI
                      above zero. (Global.)
  G5 stability        the unconstrained coefficient stays positive in at least
                      `loqo_min_share` of leave-one-task-out refits. Fewer than
                      `min_loqo_groups` tasks fails closed.

A dimension that fails is reported UNIDENTIFIED. `deployable_vector` then
gives it a declared prior value, recorded as a constraint, never as a finding.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from weight_elicitation.choice_model import (
    DEFAULT_POSITION,
    DIMS,
    ChoiceData,
    PositionSpec,
    cluster_bootstrap,
    cross_validate,
    fit_model,
    lr_test_components,
    paired_participant_bootstrap,
    percentile_ci,
)
from weight_elicitation.placebo import placebo_test, position_only_null


@dataclass
class GateConfig:
    alpha: float = 0.05
    placebo_reps: int = 200
    bootstrap_reps: int = 300
    cv_folds: int = 5
    cv_bootstrap_reps: int = 1000
    loqo_min_share: float = 0.8
    min_loqo_groups: int = 5
    seed: int = 20260913


def evaluate_gates(data: ChoiceData,
                   estimator: Union[str, Callable[[ChoiceData], np.ndarray]], *,
                   spec: PositionSpec = DEFAULT_POSITION,
                   config: Optional[GateConfig] = None,
                   name: Optional[str] = None,
                   statistic: Optional[Callable[[ChoiceData], np.ndarray]] = None,
                   estimator_kwargs: Optional[dict] = None,
                   progress: Optional[Callable[[str], None]] = None) -> Dict[str, object]:
    """`estimator` is a registry name from estimators.py (preferred: its weights
    and its scale-bearing placebo statistic are then bound consistently) or a
    callable returning weights, in which case `statistic` is required."""
    cfg = config or GateConfig()
    say = progress or (lambda _msg: None)
    # The smallest achievable placebo p is 1 / (reps + 1). If that is not below
    # alpha, G1 can never pass and every dimension would be reported
    # unidentified for a reason that has nothing to do with the data.
    if 1.0 / (cfg.placebo_reps + 1) >= cfg.alpha:
        raise ValueError(f"placebo_reps={cfg.placebo_reps} cannot reach p < alpha={cfg.alpha}; "
                         f"use at least {int(round(1 / cfg.alpha))} replicates")
    if isinstance(estimator, str):
        from weight_elicitation.estimators import bind
        kw = estimator_kwargs or {}
        name = name or estimator
        statistic = bind(estimator, spec, raw=True, **kw)
        estimator = bind(estimator, spec, **kw)
    elif statistic is None:
        raise ValueError("pass `statistic` (a scale-bearing vector) with a callable estimator")
    name = name or "estimator"

    weights = np.asarray(estimator(data), float)
    observed_stat = np.asarray(statistic(data), float)

    say("G2 likelihood-ratio test")
    lr = lr_test_components(data, spec)
    g2 = lr["p"] < cfg.alpha

    say(f"G1 placebo ({cfg.placebo_reps} reps)")
    null_beta = position_only_null(data, spec)
    plc = placebo_test(data, statistic, observed=observed_stat, null_beta=null_beta, spec=spec,
                       n_reps=cfg.placebo_reps, seed=cfg.seed)
    g1 = {d: plc["p_value"][d] < cfg.alpha for d in DIMS}

    say(f"G3 clustered bootstrap ({cfg.bootstrap_reps} reps)")
    full = fit_model(data, spec)
    draws = cluster_bootstrap(data, lambda s: fit_model(s, spec, with_se=False).components,
                              cfg.bootstrap_reps, cfg.seed)
    lo, hi = percentile_ci(draws) if len(draws) else (np.full(5, np.nan), np.full(5, np.nan))
    g3 = {d: bool(lo[i] > 0) for i, d in enumerate(DIMS)}

    say(f"G4 {cfg.cv_folds}-fold participant cross-validation")
    cv = cross_validate(data, {"position_only": {"components": False},
                               "components": {}}, spec, cfg.cv_folds, cfg.seed)
    diff = cv["components"]["per_choice"] - cv["position_only"]["per_choice"]
    d_mean, d_lo, d_hi = paired_participant_bootstrap(diff, data.participants,
                                                      cfg.cv_bootstrap_reps, cfg.seed)
    g4 = d_lo > 0

    say("G5 leave-one-task-out")
    tasks = [t for t in np.unique(data.tasks)]
    signs: List[np.ndarray] = []
    if len(tasks) >= cfg.min_loqo_groups:
        for t in tasks:
            sub = data.subset(np.where(data.tasks != t)[0])
            signs.append(fit_model(sub, spec, with_se=False).components > 0)
    share_pos = (np.mean(signs, axis=0) if signs else np.zeros(5))
    g5 = {d: bool(signs) and bool(share_pos[i] >= cfg.loqo_min_share)
          for i, d in enumerate(DIMS)}

    per_dim = {}
    for i, d in enumerate(DIMS):
        checks = {"G1_placebo": bool(g1[d]), "G2_likelihood_ratio": bool(g2),
                  "G3_interval": g3[d], "G4_held_out": bool(g4), "G5_stability": g5[d]}
        per_dim[d] = {
            "weight": float(weights[i]),
            "placebo_statistic": float(observed_stat[i]),
            "beta_unconstrained": float(full.components[i]),
            "ci95": [float(lo[i]), float(hi[i])],
            "placebo_p": plc["p_value"][d],
            "placebo_null_mean": plc["null_mean"].get(d),
            "loqo_positive_share": float(share_pos[i]),
            "checks": checks,
            "identified": all(checks.values()),
        }
    identified = [d for d in DIMS if per_dim[d]["identified"]]
    return _plain({
        "estimator": name,
        "position_spec": spec.label(),
        "config": asdict(cfg),
        "n_choices": int(len(data)),
        "n_participants": int(len(np.unique(data.participants))),
        "weights": dict(zip(DIMS, map(float, weights))),
        "likelihood_ratio": lr,
        "held_out": {"position_only": {k: v for k, v in cv["position_only"].items()
                                       if k != "per_choice"},
                     "components": {k: v for k, v in cv["components"].items()
                                    if k != "per_choice"},
                     "delta_ll_per_choice": d_mean, "delta_ci95": [d_lo, d_hi]},
        "loqo_tasks": len(signs),
        "dimensions": per_dim,
        "identified_dimensions": identified,
        "shippable": len(identified) == len(DIMS),
    })


def _plain(obj):
    """Numpy scalars -> Python, recursively, so every report is JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def deployable_vector(report: Dict[str, object], prior: Sequence[float]
                      ) -> Dict[str, object]:
    """Identified dimensions keep their estimated proportions; the rest take the
    prior. The prior mass is DECLARED in the output, so nobody can later read it
    back as something the study found."""
    prior = np.asarray(prior, float)
    prior = prior / prior.sum()
    dims = report["dimensions"]
    ident = [i for i, d in enumerate(DIMS) if dims[d]["identified"]]
    w = prior.copy()
    if ident:
        est = np.array([dims[DIMS[i]]["weight"] for i in ident])
        free_mass = 1.0 - sum(prior[i] for i in range(5) if i not in ident)
        if est.sum() > 0:
            w[ident] = est / est.sum() * free_mass
    return {"weights": dict(zip(DIMS, map(float, w))),
            "estimated": [DIMS[i] for i in ident],
            "declared_from_prior": {DIMS[i]: float(prior[i]) for i in range(5) if i not in ident}}


def format_report(report: Dict[str, object]) -> str:
    lines = [f"ACCEPTANCE GATES - {report['estimator']}  (position {report['position_spec']}, "
             f"{report['n_choices']} choices / {report['n_participants']} participants)"]
    lr = report["likelihood_ratio"]
    ho = report["held_out"]
    lines.append(f"  G2 LR components vs position: chi2(5) = {lr['lr']:.2f}, p = {lr['p']:.4g}")
    lines.append(f"  G4 held-out delta log-lik/choice: {ho['delta_ll_per_choice']:+.4f} "
                 f"[{ho['delta_ci95'][0]:+.4f}, {ho['delta_ci95'][1]:+.4f}]  "
                 f"(perplexity {ho['position_only']['perplexity']:.3f} -> "
                 f"{ho['components']['perplexity']:.3f})")
    lines.append(f"  {'dimension':14s} {'weight':>7s} {'beta':>7s} {'95% CI':>18s} "
                 f"{'placebo p':>9s} {'LOTO+':>6s}  G1 G2 G3 G4 G5  verdict")
    for d, r in report["dimensions"].items():
        c = r["checks"]
        marks = "  ".join("Y" if v else "." for v in c.values())
        lines.append(f"  {d:14s} {r['weight']:7.3f} {r['beta_unconstrained']:+7.2f} "
                     f"[{r['ci95'][0]:+6.2f},{r['ci95'][1]:+6.2f}] {r['placebo_p']:9.3f} "
                     f"{r['loqo_positive_share']:6.2f}   {marks}   "
                     f"{'IDENTIFIED' if r['identified'] else 'unidentified'}")
    lines.append(f"  shippable: {report['shippable']}   identified: "
                 f"{', '.join(report['identified_dimensions']) or 'none'}")
    return "\n".join(lines)
