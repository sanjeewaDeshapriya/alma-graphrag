"""
Fit the wave-2 discrete choice experiment, gate it, and say what may ship.

    python -m weight_elicitation.fit_dce --dump data/study_data_<wave2>.json --material m_dce.json
    python -m weight_elicitation.fit_dce ... --bootstrap 500 --placebo-reps 400
    python -m weight_elicitation.fit_dce ... --emit-profile

Pre-registered analysis (docs/Weight_Elicitation_Share_Audit.md §Revised plan)
-----------------------------------------------------------------------------
Primary model   pooled conditional logit, UNCONSTRAINED, five components plus
                one indicator per display position 2..J (J = set size),
                participant-clustered bootstrap intervals.
Weights         the unconstrained coefficients clipped at zero and normalised —
                the estimator the gates evaluate (`pooled`). Per-question fits are
                a sensitivity analysis only and never ship.
Money scale     a second model replaces `economic` by the hotel's price in LKR
                (10k LKR units). WTP for one unit of a component (its full 0-1
                range) is -beta_component / beta_price, with a clustered interval.
                Reported only when price is significantly negative.
Gates           gates.evaluate_gates on the primary estimator. A dimension that
                fails is UNIDENTIFIED.
Shipping        all five identified -> `shippable`. Otherwise
                `gates.deployable_vector` keeps the identified dimensions'
                proportions and gives the rest the declared hand-set prior;
                `--emit-profile` prints that vector with the prior mass named,
                and refuses when nothing is identified (it would just be the prior).

Refuses wave-1-shaped data (32-option browse lists) unless --allow-browse,
because this analysis assumes small randomised sets.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from weight_elicitation import MATERIAL, OUT, latest_dump
from weight_elicitation.choice_model import (
    DIMS,
    PositionSpec,
    choice_data_from_responses,
    cluster_bootstrap,
    fit_model,
    lr_test_components,
    percentile_ci,
    to_simplex,
)
from weight_elicitation.estimators import HANDSET, per_task_macro
from weight_elicitation.fit_weights import facility_scores, load_dump, load_material
from weight_elicitation.gates import GateConfig, deployable_vector, evaluate_gates, format_report


def wtp_analysis(responses, material, facility, spec: PositionSpec, reps: int, seed: int) -> dict:
    """Swap `economic` for price (10k LKR) and express components in rupees."""
    hotels = material["hotels"]
    data = choice_data_from_responses(
        responses, facility,
        extra={"price_10k_lkr": lambda o: hotels[o["hotel_id"]]["attributes"]["price_lkr"] / 1e4})
    econ = DIMS.index("economic")
    keep = [i for i in range(5) if i != econ]

    def fit(d):
        d2 = d.subset(np.arange(len(d)))
        d2.F = d2.F.copy()
        d2.F[:, :, econ] = 0.0                      # economic replaced by price itself
        f = fit_model(d2, spec, extra=True, with_se=False)
        b = {nm: f.beta[f.names.index(nm)] for nm in f.names}
        price = b["price_10k_lkr"]
        return np.array([price] + [-b[DIMS[i]] / price * 1e4 if price < 0 else np.nan for i in keep])

    point = fit(data)
    draws = cluster_bootstrap(data, fit, reps, seed)
    lo, hi = percentile_ci(draws) if len(draws) else (np.full(5, np.nan),) * 2
    price_ok = bool(hi[0] < 0)
    out = {"beta_price_per_10k_lkr": float(point[0]),
           "beta_price_ci95": [float(lo[0]), float(hi[0])],
           "price_significantly_negative": price_ok,
           "wtp_lkr_per_full_component_range": {}}
    for j, i in enumerate(keep, start=1):
        out["wtp_lkr_per_full_component_range"][DIMS[i]] = (
            {"wtp": float(point[j]), "ci95": [float(np.nanpercentile(draws[:, j], 2.5)),
                                              float(np.nanpercentile(draws[:, j], 97.5))]}
            if price_ok else None)
    if not price_ok:
        out["note"] = "price coefficient not significantly negative: WTP not reported"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, default=None)
    ap.add_argument("--material", type=Path, default=MATERIAL,
                    help="the DCE material the wave was fielded with")
    ap.add_argument("--facility-def", default="all_ranks")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--placebo-reps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--drop-failed-attention", action="store_true",
                    help="sensitivity analysis only; the primary cohort is everyone")
    ap.add_argument("--allow-browse", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT / "dce_weights.json")
    ap.add_argument("--emit-profile", action="store_true")
    args = ap.parse_args()

    dump = args.dump or latest_dump()
    _, responses, version = load_dump(dump)
    material = load_material(args.material)
    design = material.get("design", {})
    if design.get("mode") != "dce" and not args.allow_browse:
        raise SystemExit(f"material {material.get('version')} is not a DCE design "
                         "(design.mode != 'dce'); pass --allow-browse to analyse it anyway")
    J = int(design.get("set_size") or max(len(r.get("options") or []) for r in responses))
    sizes = [len(r.get("options") or []) for r in responses if not r.get("isAttentionCheck")]
    if sizes and max(sizes) > J and not args.allow_browse:
        raise SystemExit(f"responses carry sets of up to {max(sizes)} options but the design "
                         f"set size is {J}: this dump mixes browse-mode data")

    failed = set()
    if args.drop_failed_attention:
        failed = {r["participantId"] for r in responses
                  if r.get("isAttentionCheck") and r.get("attentionPass") is False}
    facility = facility_scores(material, args.facility_def)
    data = choice_data_from_responses(responses, facility, exclude_participants=failed)
    spec = PositionSpec("dummies", J)
    print(f"{dump.name}: {len(data)} choices from {len(np.unique(data.participants))} participants; "
          f"sets of {J}; position control {spec.label()}")

    primary = fit_model(data, spec)
    draws = cluster_bootstrap(data, lambda d: fit_model(d, spec, with_se=False).beta,
                              args.bootstrap, args.seed)
    lo, hi = percentile_ci(draws)
    lr = lr_test_components(data, spec)
    weights = to_simplex(primary.components)
    loc = draws[:, 0] + draws[:, 1]
    print(f"\nPRIMARY MODEL (unconstrained, clustered {args.bootstrap}-rep bootstrap)")
    for i, nm in enumerate(primary.names):
        print(f"  {nm:18s} {primary.beta[i]:+7.3f}  se {primary.se[i]:.3f}  "
              f"[{lo[i]:+.3f}, {hi[i]:+.3f}]")
    print(f"  location sum       {primary.components[0] + primary.components[1]:+7.3f}  "
          f"[{np.percentile(loc, 2.5):+.3f}, {np.percentile(loc, 97.5):+.3f}]")
    print(f"  LR components vs position: chi2(5) = {lr['lr']:.2f}, p = {lr['p']:.3g}")

    sensitivity = per_task_macro(data, spec)
    wtp = wtp_analysis(responses, material, facility, spec, min(args.bootstrap, 300), args.seed)

    report = evaluate_gates(data, "pooled", spec=spec,
                            config=GateConfig(placebo_reps=args.placebo_reps,
                                              bootstrap_reps=args.bootstrap, seed=args.seed),
                            name="fit_dce pooled", progress=lambda m: print(f"  running {m}"))
    print("\n" + format_report(report))
    deploy = deployable_vector(report, HANDSET)

    payload = {
        "source_dump": dump.name, "material_version": material.get("version"),
        "response_material_version": version, "position_spec": spec.label(),
        "cohort": {"choices": int(len(data)), "participants": int(len(np.unique(data.participants))),
                   "dropped_failed_attention": sorted(failed)},
        "primary": {**primary.as_dict(),
                    "bootstrap_ci95": {nm: [float(lo[i]), float(hi[i])]
                                       for i, nm in enumerate(primary.names)},
                    "location_sum_ci95": [float(np.percentile(loc, 2.5)),
                                          float(np.percentile(loc, 97.5))]},
        "likelihood_ratio": lr,
        "weights": dict(zip(DIMS, map(float, weights))),
        "sensitivity_per_question_macro": dict(zip(DIMS, map(float, sensitivity))),
        "willingness_to_pay": wtp,
        "gates": report,
        "shippable": report["shippable"],
        "deployable": deploy,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")

    if args.emit_profile:
        if not deploy["estimated"]:
            raise SystemExit("\nREFUSING --emit-profile: no dimension passed the gates; the "
                             "vector would be the hand-set prior relabelled as a finding.")
        print("\n# paste into src/graph/retriever.py")
        if deploy["declared_from_prior"]:
            print("# estimated from wave 2: " + ", ".join(deploy["estimated"]))
            print("# DECLARED from the hand-set prior (not identified): " +
                  ", ".join(f"{d}={v:.3f}" for d, v in deploy["declared_from_prior"].items()))
        print("DCE_WEIGHTS = ScoringWeights(")
        print("    " + ", ".join(f"{d}={v:.3f}" for d, v in deploy["weights"].items()) + ",")
        print(")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
