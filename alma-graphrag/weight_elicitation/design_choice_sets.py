"""
Wave-2 discrete choice experiment: D-efficient choice sets from the frozen pool.

    python -m weight_elicitation.design_choice_sets
    python -m weight_elicitation.design_choice_sets --set-size 5 --blocks 6 --restarts 8
    python -m weight_elicitation.design_choice_sets --material in.json --out out.json

Why the design changes (docs/Weight_Elicitation_Share_Audit.md)
---------------------------------------------------------------
Wave 1 showed all 32 hotels in one sortable list. Participants picked from rows
1-3 of whatever sort they chose (99.4% of choices), so the components were never
traded off against each other and no weight was identified. The fix is the
textbook discrete choice experiment the study originally planned: each question
shows a SMALL set of hotels (default 5, one screen, no scrolling, no sort
control, random order), and WHICH hotels appear together is chosen by design so
that the attributes vary independently inside each set.

What "D-efficient" means here
-----------------------------
For a conditional logit with coefficients b, the information a choice set s
contributes is

    I_s(b) = sum_j p_j (x_j - xbar_s)(x_j - xbar_s)^T,   p = softmax(X_s b),

and the design maximises log det(sum_s I_s). With the default prior b = 0 this is
the within-set covariance of the attributes: sets whose hotels differ on every
component, in directions that are not collinear. That is exactly what wave 1
lacked. A non-zero prior (`--prior-scale`) makes it a locally D-optimal design
around plausible weights.

The search is coordinate exchange with random restarts: every slot of every set
is tried against every hotel not already in the set, and an exchange is kept if
it raises

    log det(I) - dominance_penalty * #dominated pairs - balance_penalty * var(appearances)
               - correlation_penalty * sum_{i<j} max(0, |r_ij| - correlation_target)^2

where r_ij is the correlation of components i and j over the within-set
deviations of the whole design (what the logit actually sees). The penalty only
bites ABOVE the target, so it acts as a soft constraint: below it the search is
pure D-efficiency. A flat penalty on all correlation was tried first and
over-corrected (r fell to 0.09 but the design became less efficient than a
random one), because it keeps buying decorrelation nobody needs.

A dominated pair (one hotel at least as good on all five components) is an
uninformative question; appearance balance stops a few "extreme" hotels from
being shown in every set.

Output and gate
---------------
Writes a copy of the material with `design.mode = "dce"`, per-task
`choice_sets` (one list of hotel ids per block), `option_ids` narrowed to the
union of the task's sets, and a human-readable `noise_level` attribute on every
hotel so the disruption component is something a participant can actually see.
It REFUSES to write a design whose within-set component correlation exceeds
`--max-abs-r` or whose within-set spread on any component is below `--min-sd` —
the wave-1 lesson, applied to the design rather than the pool.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from weight_elicitation import MATERIAL
from weight_elicitation.choice_model import DIMS
from weight_elicitation.fit_weights import facility_scores

HANDSET = np.array([0.25, 0.20, 0.25, 0.15, 0.15])

NOISE_LABELS = ((0.66, "Quiet street"), (0.33, "Some traffic noise"), (-1.0, "Busy, noisy area"))


# --------------------------------------------------------------------------- #
# Attributes
# --------------------------------------------------------------------------- #
def task_attributes(material: dict, task: dict, hotel_ids: List[str],
                    facility: Dict[str, float]) -> np.ndarray:
    """(H, 5) component matrix a task's alternatives carry."""
    ac = material["anchor_components"][task["anchor_id"]]
    rows = []
    for hid in hotel_ids:
        g = material["hotels"][hid]["components_global"]
        rows.append([ac[hid]["spatial"], ac[hid]["accessibility"],
                     facility.get(hid, g["facility"]), g["economic"], g["disruption"]])
    return np.asarray(rows, float)


def noise_level(disruption: float) -> str:
    for cut, label in NOISE_LABELS:
        if disruption >= cut:
            return label
    return NOISE_LABELS[-1][1]


# --------------------------------------------------------------------------- #
# Information
# --------------------------------------------------------------------------- #
def set_information(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    u = X @ beta
    p = np.exp(u - u.max())
    p /= p.sum()
    xc = X - p @ X
    return (xc * p[:, None]).T @ xc


def dominated_pairs(X: np.ndarray) -> int:
    n = 0
    for i in range(len(X)):
        for j in range(len(X)):
            if i != j and np.all(X[i] >= X[j]) and np.any(X[i] > X[j]):
                n += 1
    return n


def logdet(M: np.ndarray) -> float:
    sign, val = np.linalg.slogdet(M)
    return val if sign > 0 else -1e9


def d_error(total_info: np.ndarray, n_sets: int) -> float:
    """det(I / S)^(-1/K): lower is better, comparable across designs of one size."""
    k = total_info.shape[0]
    ld = logdet(total_info / max(n_sets, 1))
    return float(math.exp(-ld / k)) if ld > -1e8 else float("inf")


class Design:
    """sets[(task_index, block)] -> list of hotel indices into the pool."""

    def __init__(self, attrs: List[np.ndarray], n_blocks: int, set_size: int,
                 beta: np.ndarray, dominance_penalty: float, balance_penalty: float,
                 correlation_penalty: float = 0.0):
        self.attrs, self.B, self.J = attrs, n_blocks, set_size
        self.H = attrs[0].shape[0]
        self.beta = beta
        self.dom_pen, self.bal_pen = dominance_penalty, balance_penalty
        self.corr_pen = correlation_penalty
        self.corr_target = 0.5
        self.sets: Dict[Tuple[int, int], List[int]] = {}

    def keys(self):
        return [(t, b) for t in range(len(self.attrs)) for b in range(self.B)]

    def random_init(self, rng: np.random.Generator) -> None:
        # Deal hotels from a reshuffled deck per task so appearances start balanced.
        for t in range(len(self.attrs)):
            deck: List[int] = []
            for b in range(self.B):
                chosen: List[int] = []
                while len(chosen) < self.J:
                    if not deck:
                        deck = list(rng.permutation(self.H))
                    h = deck.pop()
                    if h not in chosen:
                        chosen.append(h)
                self.sets[(t, b)] = chosen

    def info(self, key) -> np.ndarray:
        return set_information(self.attrs[key[0]][self.sets[key]], self.beta)

    def counts(self) -> np.ndarray:
        c = np.zeros(self.H)
        for s in self.sets.values():
            c[s] += 1
        return c

    def scatter(self, key, members=None) -> np.ndarray:
        X = self.attrs[key[0]][self.sets[key] if members is None else members]
        xc = X - X.mean(axis=0)
        return xc.T @ xc

    def objective(self, total: np.ndarray, dom: int, counts: np.ndarray,
                  scatter: np.ndarray) -> float:
        val = logdet(total) - self.dom_pen * dom - self.bal_pen * float(counts.var())
        if self.corr_pen:
            d = np.sqrt(np.maximum(np.diag(scatter), 1e-12))
            r = scatter / np.outer(d, d)
            excess = np.clip(np.abs(np.triu(r, 1)) - self.corr_target, 0.0, None)
            val -= self.corr_pen * float((excess ** 2).sum())
        return val

    def optimise(self, max_passes: int = 30) -> float:
        infos = {k: self.info(k) for k in self.keys()}
        scats = {k: self.scatter(k) for k in self.keys()}
        doms = {k: dominated_pairs(self.attrs[k[0]][self.sets[k]]) for k in self.keys()}
        total = sum(infos.values())
        scat = sum(scats.values())
        counts = self.counts()
        best = self.objective(total, sum(doms.values()), counts, scat)
        for _ in range(max_passes):
            improved = False
            for key in self.keys():
                X = self.attrs[key[0]]
                for slot in range(self.J):
                    current = self.sets[key][slot]
                    for h in range(self.H):
                        if h in self.sets[key]:
                            continue
                        trial = list(self.sets[key])
                        trial[slot] = h
                        I_new = set_information(X[trial], self.beta)
                        S_new = self.scatter(key, trial)
                        d_new = dominated_pairs(X[trial])
                        counts[current] -= 1
                        counts[h] += 1
                        tot_new = total - infos[key] + I_new
                        scat_new = scat - scats[key] + S_new
                        val = self.objective(tot_new, sum(doms.values()) - doms[key] + d_new,
                                             counts, scat_new)
                        if val > best + 1e-9:
                            self.sets[key] = trial
                            total, infos[key], doms[key], best = tot_new, I_new, d_new, val
                            scat, scats[key] = scat_new, S_new
                            current = h
                            improved = True
                        else:
                            counts[current] += 1
                            counts[h] -= 1
            if not improved:
                break
        return best

    def total_info(self) -> np.ndarray:
        return sum(self.info(k) for k in self.keys())

    def within_set_deviations(self) -> np.ndarray:
        rows = []
        for key, s in self.sets.items():
            X = self.attrs[key[0]][s]
            rows.append(X - X.mean(axis=0))
        return np.vstack(rows)


def diagnostics(design: Design) -> Dict[str, object]:
    dev = design.within_set_deviations()
    sd = dev.std(axis=0)
    corr = np.corrcoef(dev, rowvar=False)
    off = np.abs(corr - np.eye(5))
    i, j = np.unravel_index(np.argmax(off), off.shape)
    counts = design.counts()
    return {
        "d_error": d_error(design.total_info(), len(design.sets)),
        "within_set_sd": dict(zip(DIMS, map(float, sd))),
        "within_set_correlation": {a: dict(zip(DIMS, map(float, corr[k])))
                                   for k, a in enumerate(DIMS)},
        "max_abs_within_set_r": float(off.max()),
        "max_abs_pair": [DIMS[i], DIMS[j]],
        "dominated_pairs": int(sum(dominated_pairs(design.attrs[k[0]][s])
                                   for k, s in design.sets.items())),
        "appearances": {"min": int(counts.min()), "max": int(counts.max()),
                        "mean": float(counts.mean())},
    }


def build_design(material: dict, *, set_size: int = 5, n_blocks: int = 6,
                 restarts: int = 6, seed: int = 20260913, facility_def: str = "all_ranks",
                 prior_scale: float = 0.0, dominance_penalty: float = 0.5,
                 balance_penalty: float = 0.05, correlation_penalty: float = 200.0,
                 correlation_target: float = 0.5, max_passes: int = 30,
                 random_baseline: int = 50) -> Tuple[Design, Dict[str, object], List[dict]]:
    tasks = [t for t in material["tasks"] if not t.get("is_attention_check")]
    pool = list(material["hotels"])
    if set_size >= len(pool):
        raise ValueError("set size must be smaller than the hotel pool")
    facility = facility_scores(material, facility_def)
    attrs = [task_attributes(material, t, pool, facility) for t in tasks]
    beta = prior_scale * HANDSET
    rng = np.random.default_rng(seed)

    best: Optional[Design] = None
    best_val = -np.inf
    for _ in range(restarts):
        d = Design(attrs, n_blocks, set_size, beta, dominance_penalty, balance_penalty,
                   correlation_penalty)
        d.corr_target = correlation_target
        d.random_init(rng)
        val = d.optimise(max_passes)
        if val > best_val:
            best, best_val = d, val
    assert best is not None

    random_errors = []
    for _ in range(random_baseline):
        r = Design(attrs, n_blocks, set_size, beta, 0.0, 0.0)
        r.random_init(rng)
        random_errors.append(d_error(r.total_info(), len(r.sets)))
    diag = diagnostics(best)
    diag["random_design_d_error_mean"] = float(np.mean(random_errors))
    diag["efficiency_vs_random"] = float(np.mean(random_errors) / diag["d_error"])
    diag["prior_beta"] = dict(zip(DIMS, map(float, beta)))
    diag["search"] = {"restarts": restarts, "dominance_penalty": dominance_penalty,
                      "balance_penalty": balance_penalty,
                      "correlation_penalty": correlation_penalty,
                      "correlation_target": correlation_target, "seed": seed}
    return best, diag, tasks


def apply_to_material(material: dict, design: Design, tasks: List[dict], diag: dict, *,
                      seed: int, facility_def: str) -> dict:
    out = copy.deepcopy(material)
    pool = list(material["hotels"])
    by_id = {t["id"]: t for t in out["tasks"]}
    for ti, task in enumerate(tasks):
        sets = [[pool[h] for h in design.sets[(ti, b)]] for b in range(design.B)]
        t = by_id[task["id"]]
        t["choice_sets"] = sets
        t["option_ids"] = sorted({h for s in sets for h in s}, key=pool.index)
    rng = np.random.default_rng(seed + 1)
    for t in out["tasks"]:
        if not t.get("is_attention_check"):
            continue
        answer = t["attention_answer_hotel_id"]
        others = [h for h in pool if h != answer]
        sets = []
        for _ in range(design.B):
            pick = list(rng.choice(others, size=design.J - 1, replace=False)) + [answer]
            sets.append([str(h) for h in rng.permutation(pick)])
        t["choice_sets"] = sets
        t["option_ids"] = sorted({h for s in sets for h in s}, key=pool.index)
    for hid, h in out["hotels"].items():
        h["attributes"]["noise_level"] = noise_level(h["components_global"]["disruption"])
        h["attributes"]["n_facilities"] = h.get("detail", {}).get("n_facilities")
    out["design"] = {**material.get("design", {}),
                     "mode": "dce", "set_size": design.J, "n_blocks": design.B,
                     "all_options_shown": False, "sort_controls": False,
                     "randomised_option_order": True, "randomised_task_order": True,
                     "block_assignment": "hashSeed(participantId, 'block') % n_blocks",
                     "facility_definition": facility_def,
                     "dce": diag}
    base = material.get("version", "material")
    out["version"] = f"{base.split('+')[0]}+dce{design.J}x{design.B}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--material", type=Path, default=MATERIAL)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <material stem>_dce.json next to the input")
    ap.add_argument("--set-size", type=int, default=5)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--restarts", type=int, default=6)
    ap.add_argument("--max-passes", type=int, default=30)
    ap.add_argument("--prior-scale", type=float, default=0.0,
                    help="locally optimal around prior_scale x hand-set weights (0 = utility-neutral)")
    ap.add_argument("--correlation-penalty", type=float, default=200.0)
    ap.add_argument("--correlation-target", type=float, default=0.5,
                    help="within-set |r| the search aims below (the gate is --max-abs-r)")
    ap.add_argument("--facility-def", default="all_ranks")
    ap.add_argument("--max-abs-r", type=float, default=0.7)
    ap.add_argument("--min-sd", type=float, default=0.08)
    ap.add_argument("--force", action="store_true",
                    help="write even if the gate fails (recorded in the material)")
    ap.add_argument("--seed", type=int, default=20260913)
    args = ap.parse_args()

    material = json.loads(args.material.read_text(encoding="utf-8"))
    design, diag, tasks = build_design(
        material, set_size=args.set_size, n_blocks=args.blocks, restarts=args.restarts,
        seed=args.seed, facility_def=args.facility_def, prior_scale=args.prior_scale,
        correlation_penalty=args.correlation_penalty,
        correlation_target=args.correlation_target, max_passes=args.max_passes)

    print(f"{len(tasks)} questions x {args.blocks} blocks x {args.set_size} hotels "
          f"from a pool of {design.H}")
    print(f"D-error {diag['d_error']:.4f}  (random designs {diag['random_design_d_error_mean']:.4f}; "
          f"{diag['efficiency_vs_random']:.2f}x as efficient)")
    print("within-set SD: " + "  ".join(f"{d[:5]}={v:.3f}" for d, v in diag["within_set_sd"].items()))
    print(f"largest within-set |r| = {diag['max_abs_within_set_r']:.3f} "
          f"({' / '.join(diag['max_abs_pair'])}); dominated pairs {diag['dominated_pairs']}; "
          f"appearances {diag['appearances']}")

    failures = []
    if diag["max_abs_within_set_r"] > args.max_abs_r:
        failures.append(f"within-set |r| {diag['max_abs_within_set_r']:.3f} > {args.max_abs_r} "
                        f"({' / '.join(diag['max_abs_pair'])})")
    thin = [d for d, v in diag["within_set_sd"].items() if v < args.min_sd]
    if thin:
        failures.append(f"within-set SD below {args.min_sd} for {', '.join(thin)}")
    diag["gate"] = {"max_abs_r": args.max_abs_r, "min_sd": args.min_sd,
                    "failures": failures, "forced": bool(failures and args.force)}
    if failures and not args.force:
        raise SystemExit("\nREFUSING TO WRITE DESIGN: " + "; ".join(failures) +
                         "\nA wave fielded on it could not separate those weights. Change the "
                         "pool (decorrelate anchors / oversample off-diagonal hotels) or pass "
                         "--force to record a deliberate override.")

    out_material = apply_to_material(material, design, tasks, diag, seed=args.seed,
                                     facility_def=args.facility_def)
    out = args.out or args.material.with_name(args.material.stem + "_dce.json")
    out.write_text(json.dumps(out_material, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}  (version {out_material['version']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
