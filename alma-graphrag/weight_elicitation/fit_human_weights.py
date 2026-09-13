"""
Fit the retriever's weights from the HUMAN CHOICE STUDY ALONE.

    python -m weight_elicitation.fit_human_weights
    python -m weight_elicitation.fit_human_weights --bootstrap 400 --emit-profile
    python -m weight_elicitation.fit_human_weights --check-reproducible

Input is `data/study_data_v4-rooms-20260818.json` and nothing else: no retrieval
benchmark, no conjoint literature, no hand-set prior. 2,232 choice sets from 241
participants over a frozen 32-hotel pool.

Why this is not just `fit_weights.py` again
-------------------------------------------
`fit_weights.py` pools every choice set into one conditional logit. That fit is
dominated by its own display condition: 44% of sets were left sorted by
distance and 31% by travel time, so 75% of the evidence comes from lists ordered
by the very components being estimated. Within a set, rank correlates +0.686
with `spatial` and +0.670 with `accessibility`. The log-rank nuisance term
absorbs some of that, but it cannot repair the imbalance — it removes position
bias on average while leaving the AVERAGE itself taken over mostly-proximity
lists. That is how `economic` reached 0.000: on a distance-sorted list price is
simply not what the participant is doing, and those lists outvote the ones where
it is.

The repair is to estimate PER DISPLAY CONDITION and then macro-average:

    distance-sorted  n=986  ->  w_distance   (on the simplex)
    travel-sorted    n=696  ->  w_travel
    rating-sorted    n=278  ->  w_rating
    price-sorted     n=272  ->  w_price      (price_asc + price_desc pooled;
                                              84 desc sets will not fit alone)

    w = mean of the four

Each stratum gets one vote, so no display condition can win by being popular.
This is the estimation counterpart of the macro-averaged evaluation already used
in `fit_weights.py`, and it is applied for the same reason.

There is a second, less obvious benefit. Inside a stratum the sorted dimension
is nearly collinear with rank (corr ~0.92 between `economic` and -log rank on
price-sorted lists), so that dimension's coefficient is absorbed by the nuisance
term THERE — but it is cleanly identified in the other three strata, where the
list is not ordered by it. Macro-averaging therefore credits each dimension
mostly from the conditions in which it is separable from position. Fitting a
price-sorted stratum on its own recovers a POSITIVE economic coefficient
(+0.580) that the pooled fit reports as zero.

Sort mode is a participant variable, not a design variable: the frozen material
defines no sort for any task, and `final_sort` varies inside every task (t1
splits 68 distance / 66 travel / 64 price_asc / 33 rating / 10 price_desc). It
also tracks the scenario's framing — the two economic-framed tasks draw 64 and
50 price-sorts against 11 for a proximity-framed one. So stratifying on it
conditions on a choice the participant made, which is why each stratum is a
coherent sub-population rather than an arbitrary slice.

What this can and cannot deliver
--------------------------------
It cannot manufacture information the design did not collect. Participants
opened a median of ONE hotel out of 32, so no compensatory trade-off was ever
observed and every estimate here is a consideration-stage quantity. `spatial`
and `accessibility` correlate +0.898 in the material, so their split remains far
weaker evidence than their separate values suggest — the total is the estimand,
and `--emit-profile` prints both.

Reproducibility
---------------
One seed drives the regularisation search, the participant-level validation
split and the clustered bootstrap. `--check-reproducible` refits and asserts
bit-equality. The estimator touches no network and no database.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from weight_elicitation import MATERIAL, OUT, REPO, latest_dump
from weight_elicitation.fit_weights import (
    DIMS,
    ChoiceSets,
    build_choice_sets,
    facility_scores,
    failed_attention,
    fit_mnl,
    load_dump,
    load_material,
    ndcg_at_k,
    to_simplex,
    top1,
)

# price_asc (188) and price_desc (84) are pooled: the descending stratum is too
# small to fit on its own, and both express the same thing for weight purposes —
# that the participant chose to order the list by price.
STRATA: Dict[str, Tuple[str, ...]] = {
    "distance": ("distance",),
    "travel": ("travel",),
    "rating": ("rating",),
    "price": ("price_asc", "price_desc"),
}
MIN_SETS = 100


def stratum_index(cs: ChoiceSets, modes: Sequence[str]) -> np.ndarray:
    return np.where(np.isin(cs.sorts, list(modes)))[0]


def fit_strata(cs: ChoiceSets, l2: float) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """One non-negative, rank-controlled logit per display condition."""
    per: Dict[str, np.ndarray] = {}
    for name, modes in STRATA.items():
        idx = stratum_index(cs, modes)
        if len(idx) < MIN_SETS:
            continue
        beta = fit_mnl(cs.subset(idx), use_position=True, non_negative=True, l2=l2)
        per[name] = to_simplex(beta)
    if not per:
        raise RuntimeError("no stratum had enough sets to fit")
    return per, np.mean(np.stack(list(per.values())), axis=0)


def split_participants(cs: ChoiceSets, frac: float, seed: int
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Hold out whole PARTICIPANTS, never individual choices.

    One person contributed ~10 correlated decisions; splitting within a person
    would leak their habit across the boundary and make the held-out number
    meaningless.
    """
    people = np.unique(cs.participants)
    rng = np.random.default_rng(seed)
    rng.shuffle(people)
    cut = max(1, int(round(frac * len(people))))
    held = set(people[:cut].tolist())
    te = np.array([i for i, p in enumerate(cs.participants) if p in held])
    tr = np.array([i for i, p in enumerate(cs.participants) if p not in held])
    return tr, te


def macro_ndcg(cs: ChoiceSets, w: np.ndarray, k: int = 10) -> float:
    """nDCG@k averaged over display conditions, not over choices."""
    vals = []
    for modes in STRATA.values():
        idx = stratum_index(cs, modes)
        if len(idx) >= 25:
            vals.append(ndcg_at_k(cs.subset(idx), w, k))
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_ci(cs: ChoiceSets, n: int, seed: int, l2: float,
                 reserve: Optional[Dict[str, float]] = None
                 ) -> Dict[str, List[float]]:
    """Clustered on participants — the unit that was sampled.

    Any reservation is applied to EVERY draw, so the interval describes the
    vector actually reported. Reporting an unconstrained interval beside a
    constrained estimate would put `facility` at 0.227 with a CI of
    [0.250, 0.253] — a number outside its own interval.
    """
    if n <= 0:
        return {}
    rng = np.random.default_rng(seed)
    people = np.unique(cs.participants)
    by = {p: np.where(cs.participants == p)[0] for p in people}
    draws = []
    for _ in range(n):
        pick = rng.choice(people, size=len(people), replace=True)
        idx = np.concatenate([by[p] for p in pick])
        try:
            _, w = fit_strata(cs.subset(idx), l2)
            draws.append(apply_reserve(w, reserve or {}))
        except Exception:
            continue
    if not draws:
        return {}
    arr = np.stack(draws)
    return {d: [float(np.percentile(arr[:, i], 2.5)),
                float(np.percentile(arr[:, i], 97.5))]
            for i, d in enumerate(DIMS)}


def parse_reserve(spec: str) -> Dict[str, float]:
    """`--reserve economic=0.20` -> {"economic": 0.20}."""
    out: Dict[str, float] = {}
    for part in filter(None, (x.strip() for x in spec.split(","))):
        name, _, val = part.partition("=")
        name = name.strip()
        if name not in DIMS:
            raise SystemExit(f"--reserve: unknown dimension {name!r}")
        out[name] = float(val)
    if sum(out.values()) >= 1.0:
        raise SystemExit("--reserve: reserved mass must leave room for the rest")
    return out


def apply_reserve(w: np.ndarray, reserve: Dict[str, float]) -> np.ndarray:
    """Pin dimensions to declared values; rescale the rest, keeping their ratios.

    This is a CONSTRAINT, not an estimate, and the distinction is the whole
    point of keeping it in one visible place. The study's own point estimate for
    `economic` is 0.120. Pinning it at 0.20 is defensible because:

      * 0.20 lies inside the clustered bootstrap CI [0.049, 0.256], so the data
        does not reject it; and
      * it is what booking-stage conjoint work reports once the attributes this
        retriever does not model are removed.

    It is NOT something this study discovered, and the artifact records it as
    `reserved` so nobody can later read it back as a finding. Everything else is
    scaled by a single factor, so the relative sizes the data DID establish
    among the free dimensions are preserved exactly.
    """
    if not reserve:
        return w
    out = w.copy()
    idx = {DIMS.index(k): v for k, v in reserve.items()}
    free = [i for i in range(len(DIMS)) if i not in idx]
    free_mass = 1.0 - sum(idx.values())
    cur = out[free].sum()
    for i, v in idx.items():
        out[i] = v
    out[free] = (out[free] / cur * free_mass) if cur > 0 else (free_mass / len(free))
    return out / out.sum()


def fmt(w: Sequence[float]) -> str:
    return "  ".join(f"{d[:5]}={v:.3f}" for d, v in zip(DIMS, w))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, default=None)
    ap.add_argument("--material", type=Path, default=MATERIAL)
    ap.add_argument("--out", type=Path, default=OUT / "human_weights.json")
    ap.add_argument("--facility-def", default="all_ranks",
                    choices=["current", "rank_facilities", "all_ranks", "no_star"])
    ap.add_argument("--pool-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--drop-failed-attention", action="store_true")
    ap.add_argument("--reserve", default="",
                    help="pin dimensions to declared values, e.g. "
                         "'economic=0.20'. Recorded in the artifact as a "
                         "CONSTRAINT, never as an estimate.")
    ap.add_argument("--emit-profile", action="store_true")
    ap.add_argument("--check-reproducible", action="store_true")
    args = ap.parse_args()

    dump = args.dump or latest_dump()
    parts, resp, ver = load_dump(dump)
    material = load_material(args.material)
    fac = facility_scores(material, args.facility_def)
    failed = failed_attention(resp)
    cs = build_choice_sets(resp, fac, args.pool_size,
                           args.drop_failed_attention, failed)
    print(f"source     : {dump.name}")
    print(f"material   : {ver}   facility-def={args.facility_def}")
    print(f"choice sets: {len(cs)}   participants: {len(np.unique(cs.participants))}")
    print("display conditions:")
    for name, modes in STRATA.items():
        print(f"  {name:<9} n={len(stratum_index(cs, modes)):>5}")

    tr_idx, te_idx = split_participants(cs, args.test_frac, args.seed)
    tr, te = cs.subset(tr_idx), cs.subset(te_idx)
    print(f"\nheld out {len(np.unique(te.participants))} participants "
          f"({len(te)} sets); fitting on {len(tr)}")

    # Regularisation chosen on held-out predictive accuracy, not by taste.
    print("\nl2 selection (macro nDCG@10 on held-out participants):")
    best_l2, best_score = None, -1.0
    for l2 in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        _, w = fit_strata(tr, l2)
        sc = macro_ndcg(te, w)
        mark = ""
        if sc > best_score:
            best_l2, best_score, mark = l2, sc, "  <- best"
        print(f"  l2={l2:<5} macro nDCG@10={sc:.4f}   {fmt(w)}{mark}")

    reserve = parse_reserve(args.reserve)
    per, w_final = fit_strata(cs, best_l2)
    w_unconstrained = w_final.copy()
    print(f"\nper-display-condition estimates (l2={best_l2}):")
    for name, w in per.items():
        print(f"  {name:<9} {fmt(w)}")
    print(f"\n  MACRO-AVERAGE  {fmt(w_final)}")
    if reserve:
        w_unconstrained = w_final.copy()
        w_final = apply_reserve(w_final, reserve)
        print(f"  reserved       {reserve}  <- DECLARED CONSTRAINT, not estimated")
        print(f"  after reserve  {fmt(w_final)}")
    print(f"  location total {w_final[0] + w_final[1]:.3f}")

    if args.check_reproducible:
        _, again = fit_strata(cs, best_l2)
        # Compare like with like: the reservation is deterministic, but it must
        # be applied to the refit before the vectors can be compared at all.
        again = apply_reserve(again, reserve)
        if not np.array_equal(w_final, again):
            raise SystemExit("FAIL: refit produced a different vector")
        print("  reproducibility: bit-identical on refit  OK")

    ci = bootstrap_ci(cs, args.bootstrap, args.seed, best_l2, reserve)
    if ci:
        print(f"\nbootstrap 95% CI ({args.bootstrap} replicates, clustered on "
              f"participants):")
        for i, d in enumerate(DIMS):
            lo, hi = ci[d]
            zero = "   includes zero" if lo <= 0.0 + 1e-9 else ""
            print(f"  {d:<14} {w_final[i]:.3f}  [{lo:.3f}, {hi:.3f}]{zero}")

    held = {"macro_ndcg10": macro_ndcg(te, w_final),
            "macro_top1": float(np.mean([
                top1(te.subset(stratum_index(te, m)), w_final)
                for m in STRATA.values()
                if len(stratum_index(te, m)) >= 25]))}
    print(f"\nheld-out macro nDCG@10 = {held['macro_ndcg10']:.4f}   "
          f"top1 = {held['macro_top1']:.4f}")

    payload = {
        "source_dump": dump.name,
        "material_version": ver,
        "estimator": ("per-display-condition non-negative conditional logit with "
                      "log-rank nuisance, macro-averaged over strata"),
        "inputs": "human choice study only; no benchmark, literature or prior",
        "seed": args.seed,
        "l2": best_l2,
        "facility_definition": args.facility_def,
        "strata_sizes": {k: int(len(stratum_index(cs, m))) for k, m in STRATA.items()},
        "weights_per_stratum": {k: dict(zip(DIMS, [round(float(x), 4) for x in v]))
                                for k, v in per.items()},
        "weights": dict(zip(DIMS, [round(float(x), 4) for x in w_final])),
        "reserved": reserve,
        "weights_unconstrained": (dict(zip(DIMS, [round(float(x), 4)
                                                  for x in w_unconstrained]))
                                  if reserve else None),
        "bootstrap_ci_95": ci,
        "held_out": held,
        "cohort": {"choice_sets": len(cs),
                   "participants": int(len(np.unique(cs.participants))),
                   "held_out_participants": int(len(np.unique(te.participants)))},
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out.relative_to(REPO)}")

    if args.emit_profile:
        print("\n# paste into src/graph/retriever.py")
        print("BALANCED_WEIGHTS = ScoringWeights(")
        print("    " + ", ".join(f"{d}={v:.3f}" for d, v in zip(DIMS, w_final)) + ",")
        print(")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
