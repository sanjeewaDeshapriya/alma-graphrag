"""
Audit every human-elicited weight estimator and every shipped retriever profile.

    python -m weight_elicitation.audit_profiles
    python -m weight_elicitation.audit_profiles --placebo-reps 200 --bootstrap 300
    python -m weight_elicitation.audit_profiles --quick          # 40 reps, for a smoke run

Three questions, each answered under every position specification:

1. DO THE COMPONENTS ADD ANYTHING BEYOND POSITION?
   Likelihood-ratio test and participant-level cross-validated log-likelihood,
   position-only vs position + components. Under `neglog` wave 1 says yes
   (chi2 = 121); under `dummies:3` and `topk:3` it says no.

2. DOES EACH ESTIMATOR PASS THE ACCEPTANCE GATES?
   The estimators behind `elicited` (prior_map), `human` (per_sort_macro) and
   the share weights (per_task_macro), plus the plain pooled fits, are run
   through gates.evaluate_gates — placebo, LR, clustered interval, held-out,
   leave-one-task-out — and each dimension is reported identified or not.

3. DOES ANY SHIPPED PROFILE PREDICT HELD-OUT CHOICES BETTER THAN POSITION?
   Each profile in src/graph/retriever.py WEIGHT_PROFILES is scored as a fixed
   composite (only its scale and the position terms are estimated on the
   training folds), so it gets no advantage from being re-fitted. The
   difference from position-only is reported with a participant-clustered CI.

Outputs `out/audit.json` and `out/audit.md`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from weight_elicitation import MATERIAL, OUT, latest_dump
from weight_elicitation.choice_model import (
    DIMS,
    PositionSpec,
    choice_data_from_responses,
    cross_validate,
    lr_test_components,
    paired_participant_bootstrap,
)
from weight_elicitation.fit_weights import (
    facility_scores,
    failed_attention,
    load_dump,
    load_material,
)
from weight_elicitation.gates import GateConfig, evaluate_gates, format_report

DEFAULT_SPECS = ("neglog", "dummies:3", "topk:3")

#: estimator name -> (registry key, hyper-parameters, what it produced)
ESTIMATOR_PLAN = {
    "pooled": ("pooled", {}, "unconstrained pooled logit, clipped (inference reference)"),
    "pooled_nonneg": ("pooled_nonneg", {}, "non-negative pooled logit"),
    "elicited (prior_map)": ("prior_map", {"l2": 1.0}, "fit_weights.py deployed -> ELICITED_WEIGHTS"),
    "human (per_sort_macro)": ("per_sort_macro", {"l2": 1.0}, "fit_human_weights.py -> HUMAN_WEIGHTS"),
    "share (per_task_macro)": ("per_task_macro", {"l2": 0.0}, "fit_share_weights.py, never shipped"),
}


def retriever_profiles() -> Dict[str, np.ndarray]:
    try:
        from src.graph.retriever import WEIGHT_PROFILES
    except Exception as exc:                            # retriever deps missing
        print(f"  (could not import retriever profiles: {exc})")
        return {}
    return {name: np.array([getattr(w, d) for d in DIMS], float)
            for name, w in WEIGHT_PROFILES.items()}


def profile_cv(data, spec: PositionSpec, profiles: Dict[str, np.ndarray], folds: int,
               seed: int, boot: int) -> Dict[str, dict]:
    models = {"position_only": {"components": False}, "components": {}}
    for name, w in profiles.items():
        models[f"profile:{name}"] = {"score_weights": w / w.sum()}
    cv = cross_validate(data, models, spec, folds, seed)
    base = cv["position_only"]["per_choice"]
    out = {}
    for name, r in cv.items():
        row = {k: v for k, v in r.items() if k != "per_choice"}
        if name != "position_only":
            m, lo, hi = paired_participant_bootstrap(r["per_choice"] - base,
                                                     data.participants, boot, seed)
            row.update(delta_vs_position=m, delta_ci95=[lo, hi])
        out[name] = row
    return out


def to_markdown(result: dict) -> str:
    L: List[str] = ["# Weight-estimator audit", "",
                    f"Source `{result['source_dump']}`, {result['n_choices']} choices from "
                    f"{result['n_participants']} participants. Placebo reps "
                    f"{result['config']['placebo_reps']}, bootstrap reps "
                    f"{result['config']['bootstrap_reps']}.", ""]
    L += ["## Do the components add anything beyond position?", "",
          "| Position spec | Choices | LR chi2(5) | p | CV perplexity, position only | "
          "CV perplexity, + components | delta log-lik/choice [95% CI] |",
          "|---|---:|---:|---:|---:|---:|---|"]
    for spec, s in result["specs"].items():
        lr, cv = s["likelihood_ratio"], s["profiles_cv"]
        c = cv["components"]
        L.append(f"| {spec} | {lr['n']} | {lr['lr']:.2f} | {lr['p']:.3g} | "
                 f"{cv['position_only']['perplexity']:.3f} | {c['perplexity']:.3f} | "
                 f"{c['delta_vs_position']:+.4f} [{c['delta_ci95'][0]:+.4f}, "
                 f"{c['delta_ci95'][1]:+.4f}] |")
    L += ["", "## Acceptance gates per estimator", "",
          "A dimension is identified only if it passes all five gates "
          "(placebo, LR, clustered CI > 0, held-out gain, leave-one-task-out sign).", "",
          "| Position spec | Estimator | Weights (spa/acc/fac/eco/dis) | Placebo p | Identified |",
          "|---|---|---|---|---|"]
    for spec, s in result["specs"].items():
        for name, g in s["gates"].items():
            w = "/".join(f"{g['weights'][d]:.3f}" for d in DIMS)
            pp = "/".join(f"{g['dimensions'][d]['placebo_p']:.2f}" for d in DIMS)
            L.append(f"| {spec} | {name} | {w} | {pp} | "
                     f"{', '.join(g['identified_dimensions']) or 'none'} |")
    L += ["", "## Shipped retriever profiles as fixed composites", "",
          "Held-out log-likelihood per choice relative to position only "
          "(only a scale and the position terms are fitted).", "",
          "| Position spec | Profile | Perplexity | delta vs position [95% CI] |",
          "|---|---|---:|---|"]
    for spec, s in result["specs"].items():
        for name, r in s["profiles_cv"].items():
            if not name.startswith("profile:"):
                continue
            L.append(f"| {spec} | {name[8:]} | {r['perplexity']:.3f} | "
                     f"{r['delta_vs_position']:+.4f} [{r['delta_ci95'][0]:+.4f}, "
                     f"{r['delta_ci95'][1]:+.4f}] |")
    L += ["", "Generated by `python -m weight_elicitation.audit_profiles`."]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, default=None)
    ap.add_argument("--material", type=Path, default=MATERIAL)
    ap.add_argument("--facility-def", default="all_ranks")
    ap.add_argument("--pool-size", type=int, default=32)
    ap.add_argument("--specs", nargs="+", default=list(DEFAULT_SPECS))
    ap.add_argument("--estimators", nargs="+", default=list(ESTIMATOR_PLAN))
    ap.add_argument("--placebo-reps", type=int, default=200)
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--quick", action="store_true", help="40 placebo / 60 bootstrap reps")
    ap.add_argument("--out", type=Path, default=OUT / "audit.json")
    args = ap.parse_args()
    if args.quick:
        args.placebo_reps, args.bootstrap = 40, 60

    dump = args.dump or latest_dump()
    _, responses, version = load_dump(dump)
    material = load_material(args.material)
    data = choice_data_from_responses(responses, facility_scores(material, args.facility_def),
                                      pool_size=args.pool_size)
    profiles = retriever_profiles()
    cfg = GateConfig(placebo_reps=args.placebo_reps, bootstrap_reps=args.bootstrap,
                     cv_folds=args.cv_folds, seed=args.seed)
    print(f"audit of {dump.name}: {len(data)} choices, "
          f"{len(np.unique(data.participants))} participants; profiles: {', '.join(profiles)}")

    result = {"source_dump": dump.name, "material_version": version,
              "n_choices": int(len(data)),
              "n_participants": int(len(np.unique(data.participants))),
              "config": {"placebo_reps": cfg.placebo_reps, "bootstrap_reps": cfg.bootstrap_reps,
                         "cv_folds": cfg.cv_folds, "seed": cfg.seed},
              "specs": {}}
    for label in args.specs:
        spec = PositionSpec.parse(label)
        t0 = time.time()
        print(f"\n=== position {spec.label()} ===")
        block = {"likelihood_ratio": lr_test_components(data, spec),
                 "profiles_cv": profile_cv(data, spec, profiles, cfg.cv_folds, cfg.seed, 1000),
                 "gates": {}}
        lr = block["likelihood_ratio"]
        print(f"  LR components vs position: chi2(5) = {lr['lr']:.2f}, p = {lr['p']:.3g}")
        for name in args.estimators:
            key, kw, _what = ESTIMATOR_PLAN[name]
            print(f"  gates: {name}")
            rep = evaluate_gates(data, key, estimator_kwargs=kw, spec=spec, config=cfg, name=name)
            print("    " + format_report(rep).replace("\n", "\n    "))
            block["gates"][name] = rep
        result["specs"][spec.label()] = block
        print(f"  ({time.time() - t0:.0f}s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = args.out.with_suffix(".md")
    md.write_text(to_markdown(result), encoding="utf-8")
    print(f"\nwrote {args.out}\nwrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
