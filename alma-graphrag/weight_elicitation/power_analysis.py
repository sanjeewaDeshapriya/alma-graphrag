"""
Simulation-based power analysis for the wave-2 DCE — run BEFORE recruiting.

    python -m weight_elicitation.power_analysis --material material_dce.json
    python -m weight_elicitation.power_analysis --material m.json --participants 100 150 200 300 --reps 200
    python -m weight_elicitation.power_analysis --material m.json --beta 0.6,0.4,0.5,0.3,0 --gates 20

For each sample size it simulates complete studies on the DESIGN AS FIELDED —
the material's blocks and choice sets, random option order per participant x
question, a position effect of the assumed size — then fits the primary model
(pooled unconstrained conditional logit with position dummies) and records:

  power        P(coefficient > 0 and z > 1.96) for each dimension whose true
               coefficient is positive
  false pos.   the same rate for a dimension whose true coefficient is 0
  bias, RMSE   of the estimated coefficient
  coverage     share of nominal 95% Wald intervals containing the truth
  (--gates N)  share of N simulated studies in which each dimension passes
               every acceptance gate — slow, but it is the criterion that decides
               whether a weight ships, so it is the honest power figure

Defaults: true coefficients 2 x hand-set weights (0.50 / 0.40 / 0.50 / 0.30 /
0.30 utility per unit of a 0-1 component) and position effects extrapolated from
wave 1's top-3 fit (rank 2: -0.34, rank 3: -0.76, then -1.0, -1.2). Both are
assumptions; state them next to any sample size chosen from this output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from weight_elicitation import MATERIAL, OUT
from weight_elicitation.choice_model import (
    DIMS,
    PositionSpec,
    choice_data_from_responses,
    fit_model,
)
from weight_elicitation.design_choice_sets import task_attributes
from weight_elicitation.fit_weights import facility_scores

HANDSET = np.array([0.25, 0.20, 0.25, 0.15, 0.15])
DEFAULT_POSITION_EFFECTS = (-0.34, -0.76, -1.0, -1.2)


def simulate_study(material: dict, beta: np.ndarray, position_effects: Sequence[float],
                   n_participants: int, rng: np.random.Generator,
                   facility: Dict[str, float]) -> List[dict]:
    """Responses in the raw-dump shape, so the real loaders and fitters run on them."""
    design = material["design"]
    n_blocks = int(design["n_blocks"])
    tasks = [t for t in material["tasks"] if not t.get("is_attention_check")]
    comp_cache = {}
    for t in tasks:
        for sets in t["choice_sets"]:
            for hid in sets:
                if (t["id"], hid) not in comp_cache:
                    comp_cache[(t["id"], hid)] = task_attributes(material, t, [hid], facility)[0]
    pos_u = np.concatenate([[0.0], np.asarray(position_effects, float)])
    responses = []
    for p in range(n_participants):
        pid = f"sim{p}"
        block = p % n_blocks
        for t in rng.permutation(len(tasks)):
            task = tasks[t]
            ids = list(rng.permutation(task["choice_sets"][block]))
            X = np.array([comp_cache[(task["id"], h)] for h in ids])
            u = X @ beta + pos_u[np.minimum(np.arange(len(ids)), len(pos_u) - 1)]
            prob = np.exp(u - u.max())
            prob /= prob.sum()
            pick = int(rng.choice(len(ids), p=prob))
            responses.append({
                "participantId": pid, "taskId": task["id"], "isAttentionCheck": False,
                "timing": {"final_sort": "random"},
                "options": [{"hotel_id": h, "displayed_position": i + 1, "chosen": i == pick,
                             "components": dict(zip(DIMS, map(float, X[i])))}
                            for i, h in enumerate(ids)],
            })
    return responses


def run(material: dict, beta: np.ndarray, position_effects: Sequence[float],
        participants: Sequence[int], reps: int, seed: int, gate_reps: int = 0,
        facility_def: str = "all_ranks") -> Dict[str, dict]:
    facility = facility_scores(material, facility_def)
    J = int(material["design"]["set_size"])
    spec = PositionSpec("dummies", J)
    out = {}
    for n in participants:
        rng = np.random.default_rng(seed + n)
        est, se = [], []
        for _ in range(reps):
            data = choice_data_from_responses(
                simulate_study(material, beta, position_effects, n, rng, facility))
            f = fit_model(data, spec)
            est.append(f.components)
            se.append(f.se[:5])
        E, S = np.array(est), np.array(se)
        z = E / np.where(S > 0, S, np.nan)
        detect = (z > 1.959964) & (E > 0)
        cover = (E - 1.959964 * S <= beta) & (beta <= E + 1.959964 * S)
        row = {}
        for i, d in enumerate(DIMS):
            row[d] = {"true": float(beta[i]),
                      ("power" if beta[i] > 0 else "false_positive_rate"): float(np.nanmean(detect[:, i])),
                      "mean_estimate": float(E[:, i].mean()),
                      "bias": float(E[:, i].mean() - beta[i]),
                      "rmse": float(np.sqrt(((E[:, i] - beta[i]) ** 2).mean())),
                      "coverage95": float(np.nanmean(cover[:, i])),
                      "mean_se": float(np.nanmean(S[:, i]))}
        if gate_reps:
            from weight_elicitation.gates import GateConfig, evaluate_gates
            cfg = GateConfig(placebo_reps=60, bootstrap_reps=80, cv_bootstrap_reps=300, seed=seed)
            passed = np.zeros(5)
            for _ in range(gate_reps):
                data = choice_data_from_responses(
                    simulate_study(material, beta, position_effects, n, rng, facility))
                rep = evaluate_gates(data, "pooled", spec=spec, config=cfg)
                passed += [rep["dimensions"][d]["identified"] for d in DIMS]
            for i, d in enumerate(DIMS):
                row[d]["gate_pass_rate"] = float(passed[i] / gate_reps)
        out[str(n)] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--material", type=Path, required=True,
                    help="a DCE material written by design_choice_sets.py")
    ap.add_argument("--participants", type=int, nargs="+", default=[100, 150, 200, 250])
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--beta", default=None,
                    help="comma-separated true coefficients (default 2 x hand-set)")
    ap.add_argument("--position-effects", default=",".join(map(str, DEFAULT_POSITION_EFFECTS)))
    ap.add_argument("--gates", type=int, default=0,
                    help="also run the full acceptance gates on this many simulated studies")
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--out", type=Path, default=OUT / "power_analysis.json")
    args = ap.parse_args()

    material = json.loads(args.material.read_text(encoding="utf-8"))
    if material.get("design", {}).get("mode") != "dce":
        raise SystemExit("material has no DCE design; run design_choice_sets.py first")
    beta = (np.array([float(x) for x in args.beta.split(",")]) if args.beta else 2.0 * HANDSET)
    pos = [float(x) for x in args.position_effects.split(",")]

    result = run(material, beta, pos, args.participants, args.reps, args.seed, args.gates)
    tasks = sum(1 for t in material["tasks"] if not t.get("is_attention_check"))
    print(f"design {material['version']}: {tasks} questions x {material['design']['n_blocks']} blocks "
          f"x {material['design']['set_size']} hotels; {args.reps} simulated studies per N")
    print(f"true beta {dict(zip(DIMS, beta.round(3)))}; position effects {pos}\n")
    head = f"{'N':>5}  " + "  ".join(f"{d[:10]:>14s}" for d in DIMS)
    print(head + "\n" + "-" * len(head))
    for n, row in result.items():
        cells = []
        for d in DIMS:
            r = row[d]
            rate = r.get("power", r.get("false_positive_rate"))
            tag = "pw" if "power" in r else "fp"
            g = f" g{r['gate_pass_rate']:.2f}" if "gate_pass_rate" in r else ""
            cells.append(f"{tag}{rate:.2f} se{r['mean_se']:.2f}{g}")
        print(f"{n:>5}  " + "  ".join(f"{c:>14s}" for c in cells))
    print("\npw = power (z > 1.96, positive); fp = false-positive rate for a zero coefficient; "
          "se = mean standard error; g = acceptance-gate pass rate")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"material_version": material["version"],
                                    "true_beta": dict(zip(DIMS, map(float, beta))),
                                    "position_effects": pos, "reps": args.reps,
                                    "results": result}, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
