"""
Fit the retriever's composite-score weights from the hosted study's raw dump.

Reads `study_data_v4-rooms-20260818.json` (the `?format=raw` admin export)
DIRECTLY — no CSV in the path. The CSVs are for humans; this is the model.

    python -m weight_elicitation.fit_weights
    python -m weight_elicitation.fit_weights --dump study_data_v4-rooms-20260818.json
    python -m weight_elicitation.fit_weights --facility-def all_ranks --bootstrap 400
    python -m weight_elicitation.fit_weights --emit-profile   # print retriever code

What it estimates
-----------------
A conditional (multinomial) logit over choice sets: each response is one set of
candidate hotels, exactly one labelled chosen, and the utility of a candidate is
a linear function of its five component scores. The fitted coefficients, put on
the simplex, are the retriever's `ScoringWeights`.

Three things make this more than a plain logit, and all three come out of
`docs/Weight_Elicitation_Data_Audit.md`:

1. POSITION IS A NUISANCE PARAMETER.
   46% of participants picked the hotel at rank 1 and 78% picked rank 1-2, on a
   list sorted by one of the very components being estimated. Within a choice
   set, rank correlates +0.686 with `spatial` and +0.670 with `accessibility`.
   A logit without a rank term therefore books position bias as a preference for
   proximity — which is exactly how the published weights reached
   spatial + accessibility = 0.964.

   So a log-rank term is estimated alongside the components and then DISCARDED.
   At retrieval time there is no pre-existing position: the retriever produces
   the ordering. Position belongs in the fit and nowhere else.

2. `facility` IS RECOMPUTED, NOT READ.
   The stored vectors carry the `min(n_facilities / 40, 1.0)` ceiling that
   saturated 30 of 32 hotels and left `facility` with a third of the spread of
   every other dimension. Because component vectors were NEVER SHOWN to
   participants — `componentsFor()` is server-only — the feature matrix can be
   rebuilt without invalidating a single choice. The behaviour is untouched;
   only our description of the alternatives changes.

3. EVALUATION IS MACRO-AVERAGED OVER SORT MODE.
   75% of choices were made on a proximity-sorted list. Micro-averaged held-out
   accuracy would hand the win to whichever weights correlate with proximity —
   scoring the artifact instead of the preference. Averaging per sort mode
   (distance / travel / rating / price_asc / price_desc) asks the harder and
   more honest question: do these weights predict choice when the list is NOT
   sorted the way they point?

Outputs `weight_elicitation/out/weights.json` plus a report on stdout.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

# The report uses en-dashes, Δ and × in its labels; a Windows console defaults to
# cp1252 and dies on all three mid-run, after the expensive fitting is done.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from weight_elicitation import DUMPS, MATERIAL, OUT, REPO, latest_dump

DIMS = ["spatial", "accessibility", "facility", "economic", "disruption"]
SORT_MODES = ["distance", "travel", "rating", "price_asc", "price_desc"]

# The hand-tuned prior the retriever ships with, as the comparison baseline.
HANDSET = np.array([0.25, 0.20, 0.25, 0.15, 0.15])


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_dump(path: Path) -> Tuple[List[dict], List[dict], str]:
    with path.open(encoding="utf-8") as fh:
        d = json.load(fh)
    return d["participants"], d["responses"], d.get("version", "unknown")


def load_material(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def pct_rank(x: float, arr: Sequence[float]) -> float:
    """Share of the pool at or below `x` — the scale every other component uses."""
    if len(arr) == 0:
        return 0.5
    return sum(1 for v in arr if v <= x) / len(arr)


def facility_scores(material: dict, definition: str) -> Dict[str, float]:
    """Rebuild the `facility` component under a chosen definition.

    `current` reproduces the shipped material (the saturated one) so the
    published result stays reproducible. The others repair it; see DATA_AUDIT.md
    §3b for the measured spread and identifying power of each.
    """
    hotels = material["hotels"]
    ids = list(hotels)
    nf = [float(hotels[i]["detail"]["n_facilities"]) for i in ids]
    st = [float(hotels[i]["attributes"].get("star") or 0) for i in ids]
    rt = [float(hotels[i]["attributes"].get("rating") or 0) for i in ids]

    out: Dict[str, float] = {}
    for k, hid in enumerate(ids):
        if definition == "current":
            v = 0.45 * st[k] / 5 + 0.35 * min(nf[k] / 40.0, 1.0) + 0.20 * rt[k] / 5
        elif definition == "rank_facilities":
            v = 0.45 * st[k] / 5 + 0.35 * pct_rank(nf[k], nf) + 0.20 * rt[k] / 5
        elif definition == "all_ranks":
            v = (0.45 * pct_rank(st[k], st)
                 + 0.35 * pct_rank(nf[k], nf)
                 + 0.20 * pct_rank(rt[k], rt))
        elif definition == "no_star":
            v = 0.60 * pct_rank(nf[k], nf) + 0.40 * pct_rank(rt[k], rt)
        else:
            raise ValueError(f"unknown facility definition {definition!r}")
        out[hid] = max(0.0, min(1.0, v))
    return out


# --------------------------------------------------------------------------- #
# Choice sets
# --------------------------------------------------------------------------- #
class ChoiceSets:
    """Padded tensor of choice sets: X (n, m, k), mask (n, m), y (n,).

    Padding to a rectangle lets every likelihood evaluation be one vectorised
    pass instead of a Python loop over 2,232 sets — which is what makes a
    clustered bootstrap of a few hundred replicates finish in seconds.
    """

    def __init__(self, X: np.ndarray, mask: np.ndarray, y: np.ndarray,
                 participants: np.ndarray, sorts: np.ndarray):
        self.X, self.mask, self.y = X, mask, y
        self.participants, self.sorts = participants, sorts

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx: np.ndarray) -> "ChoiceSets":
        return ChoiceSets(self.X[idx], self.mask[idx], self.y[idx],
                          self.participants[idx], self.sorts[idx])


def build_choice_sets(responses: List[dict],
                      facility: Dict[str, float],
                      pool_size: int,
                      drop_failed_attention: bool,
                      participants_failed: set) -> ChoiceSets:
    """One choice set per response, with the recomputed feature matrix.

    Only candidates that were actually ON SCREEN are included. A hotel the
    participant filtered out was never rejected — it was never seen — and
    treating the two as the same event would bias every coefficient.
    """
    rows: List[np.ndarray] = []
    labels: List[int] = []
    pids: List[str] = []
    sorts: List[str] = []

    for r in responses:
        if r.get("isAttentionCheck"):
            continue
        if len(r.get("options") or []) != pool_size:
            continue                       # legacy 5- and 35-option strata
        if drop_failed_attention and r["participantId"] in participants_failed:
            continue

        opts = [o for o in r["options"]
                if o.get("displayed_position") is not None and o.get("components")]
        if len(opts) < 2 or not any(o.get("chosen") for o in opts):
            continue

        feats = []
        for o in opts:
            c = o["components"]
            feats.append([
                c["spatial"],
                c["accessibility"],
                facility.get(o["hotel_id"], c["facility"]),   # recomputed
                c["economic"],
                c["disruption"],
                # Rank enters as -log(rank): click-through falls off roughly
                # geometrically down a list, so log rank is the natural scale.
                -math.log(o["displayed_position"]),
            ])
        rows.append(np.asarray(feats, dtype=float))
        labels.append(next(i for i, o in enumerate(opts) if o.get("chosen")))
        pids.append(r["participantId"])
        sorts.append((r.get("timing") or {}).get("final_sort") or "unknown")

    m = max(len(a) for a in rows)
    k = rows[0].shape[1]
    X = np.zeros((len(rows), m, k))
    mask = np.zeros((len(rows), m), dtype=bool)
    for i, a in enumerate(rows):
        X[i, : len(a)] = a
        mask[i, : len(a)] = True
    return ChoiceSets(X, mask, np.asarray(labels), np.asarray(pids),
                      np.asarray(sorts))


def failed_attention(responses: List[dict]) -> set:
    """Participants who failed the check — dropped wholesale, not task by task.

    Someone who was not paying attention on the trap task was not paying
    attention on the others either; excluding only the trap keeps their noise.
    """
    return {r["participantId"] for r in responses
            if r.get("isAttentionCheck") and r.get("attentionPass") is False}


# --------------------------------------------------------------------------- #
# Conditional logit
# --------------------------------------------------------------------------- #
PRIOR_DIR = HANDSET / np.linalg.norm(HANDSET)


def neg_loglik(beta: np.ndarray, cs: ChoiceSets, l2: float,
               toward_prior: bool = False) -> float:
    u = cs.X @ beta
    u = np.where(cs.mask, u, -np.inf)
    u = u - u.max(axis=1, keepdims=True)
    lse = np.log(np.exp(np.where(cs.mask, u, -np.inf)).sum(axis=1))
    chosen = u[np.arange(len(cs)), cs.y]

    b5 = beta[:5]
    if toward_prior:
        # MAP against the hand-set prior, shrinking DIRECTION but not magnitude.
        #
        # A plain ridge pulls toward zero, which here just deletes the weakly
        # identified dimensions and hands the retriever a degenerate vector
        # (`accessibility = 1.0`, everything else 0). What we actually believe a
        # priori is the hand-set profile's SHAPE, not that the coefficients are
        # small.
        #
        # Penalising ||beta - s*h||^2 with a free prior scale `s` and profiling
        # `s` out analytically (s* = beta·h / ||h||^2) leaves exactly the part of
        # beta orthogonal to the prior direction. So lambda = 0 is the
        # unconstrained fit, lambda -> inf returns the hand-set profile, and
        # anything between is a tuned interpolation — which is what the ad-hoc
        # `blended` profile was reaching for by hand.
        pen = l2 * float(b5 @ b5 - (b5 @ PRIOR_DIR) ** 2)
    else:
        # Plain ridge. Excludes the position term: that is a nuisance we want
        # fitted freely, and shrinking it would push its share onto the
        # components — the very confound the term exists to absorb.
        pen = l2 * float(b5 @ b5)
    return float(-(chosen - lse).sum()) + pen


def fit_mnl(cs: ChoiceSets, *, use_position: bool, non_negative: bool,
            l2: float = 0.0, toward_prior: bool = False) -> np.ndarray:
    k = 6 if use_position else 5
    view = cs if use_position else ChoiceSets(
        cs.X[:, :, :5], cs.mask, cs.y, cs.participants, cs.sorts)
    if non_negative:
        # Components are all "higher is better", and the retriever cannot use a
        # negative weight. Constraining during estimation is not the same as
        # clipping afterwards: the optimiser redistributes the mass properly
        # instead of leaving the survivors distorted by a partner it later cut.
        bounds = [(0.0, None)] * 5 + ([(None, None)] if use_position else [])
    else:
        bounds = [(None, None)] * k
    res = minimize(neg_loglik, np.zeros(k), args=(view, l2, toward_prior),
                   method="L-BFGS-B", bounds=bounds)
    return res.x


def to_simplex(beta: np.ndarray) -> np.ndarray:
    w = np.clip(beta[:5], 0.0, None)
    s = w.sum()
    return w / s if s > 0 else np.full(5, 0.2)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def ndcg_at_k(cs: ChoiceSets, w: np.ndarray, k: int = 10) -> float:
    """Mean nDCG@k with exactly one relevant item per set: 1/log2(rank+1)."""
    scores = np.where(cs.mask, cs.X[:, :, :5] @ w, -np.inf)
    chosen_score = scores[np.arange(len(cs)), cs.y]
    rank = (scores > chosen_score[:, None]).sum(axis=1) + 1
    return float(np.where(rank <= k, 1.0 / np.log2(rank + 1), 0.0).mean())


def top1(cs: ChoiceSets, w: np.ndarray) -> float:
    scores = np.where(cs.mask, cs.X[:, :, :5] @ w, -np.inf)
    return float((scores.argmax(axis=1) == cs.y).mean())


def macro_by_sort(cs: ChoiceSets, w: np.ndarray, fn) -> Tuple[float, Dict[str, float]]:
    """Average the metric over sort modes, not over choices.

    Three quarters of the data was collected on a proximity-sorted list. A
    micro-average would let those sets decide the winner, which rewards exactly
    the position bias the fit is trying to remove. Weighting each display
    condition equally asks whether the weights survive a list that is NOT
    ordered the way they point.
    """
    per: Dict[str, float] = {}
    for mode in SORT_MODES:
        idx = np.where(cs.sorts == mode)[0]
        if len(idx) >= 25:
            per[mode] = fn(cs.subset(idx), w)
    return (float(np.mean(list(per.values()))) if per else float("nan")), per


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def bootstrap(cs: ChoiceSets, n: int, seed: int, **fit_kw) -> np.ndarray:
    """Resample PARTICIPANTS, not choices.

    Each person contributed ~10 correlated decisions, so resampling rows would
    treat one participant's habit as ten independent facts and shrink the
    intervals to fiction.
    """
    rng = np.random.default_rng(seed)
    people = np.unique(cs.participants)
    by_person = {p: np.where(cs.participants == p)[0] for p in people}
    out = []
    for _ in range(n):
        pick = rng.choice(people, size=len(people), replace=True)
        idx = np.concatenate([by_person[p] for p in pick])
        try:
            out.append(to_simplex(fit_mnl(cs.subset(idx), **fit_kw)))
        except Exception:                      # a degenerate resample: skip it
            continue
    return np.asarray(out)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def fmt(w: Sequence[float]) -> str:
    return "  ".join(f"{d[:6]}={v:.3f}" for d, v in zip(DIMS, w))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, default=None,
                    help="raw ?format=raw JSON export from the hosted study "
                         "(default: newest study_data_*.json in "
                         "weight_elicitation/data/)")
    ap.add_argument("--material", type=Path, default=MATERIAL)
    ap.add_argument("--out", type=Path, default=OUT / "weights.json")
    ap.add_argument("--facility-def", default="all_ranks",
                    choices=["current", "rank_facilities", "all_ranks", "no_star"])
    ap.add_argument("--pool-size", type=int, default=32,
                    help="analyse only this stratum (32 = the real study)")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--val-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=20260818)
    # The attention check fails for 43.7% of participants, so excluding them is a
    # judgement call made AFTER seeing the data, not a pre-specified rule. The
    # all-cohort fit is therefore the primary one and the clean cohort is a
    # sensitivity analysis — not the other way round.
    ap.add_argument("--drop-failed-attention", action="store_true",
                    help="sensitivity analysis: exclude the participants who "
                         "failed the attention check (NOT the primary spec)")
    ap.add_argument("--emit-profile", action="store_true",
                    help="print ScoringWeights source for src/graph/retriever.py")
    args = ap.parse_args()

    if args.dump is None:
        args.dump = latest_dump()
    elif not args.dump.exists():
        # Accept a bare filename for a dump sitting in the usual place.
        cand = DUMPS / args.dump.name
        if not cand.exists():
            raise SystemExit(f"dump not found: {args.dump}")
        args.dump = cand

    participants, responses, version = load_dump(args.dump)
    material = load_material(args.material)
    facility = facility_scores(material, args.facility_def)
    failed = failed_attention(responses)

    cs = build_choice_sets(responses, facility, args.pool_size,
                           args.drop_failed_attention, failed)

    print("=" * 78)
    print(f"ALMA-GraphRAG — weight elicitation fit   (material {version})")
    print("=" * 78)
    print(f"source dump        : {args.dump.name}")
    print(f"participants (all) : {len(participants)}   responses: {len(responses)}")
    print(f"failed attention   : {len(failed)} participants"
          f"{' (dropped — sensitivity)' if args.drop_failed_attention else ' (kept — primary spec)'}")
    print(f"analysis cohort    : {len(cs)} choice sets from "
          f"{len(np.unique(cs.participants))} participants")
    print(f"facility definition: {args.facility_def}")
    fac_sd = np.array([cs.X[i][cs.mask[i]][:, 2].std() for i in range(len(cs))]).mean()
    spa_sd = np.array([cs.X[i][cs.mask[i]][:, 0].std() for i in range(len(cs))]).mean()
    print(f"  facility within-set sd {fac_sd:.4f}  vs spatial {spa_sd:.4f}"
          f"  ({fac_sd / spa_sd:.2f}x)")

    # ---- three-way split by participant ------------------------------------ #
    #
    # Split on PEOPLE, never on choices: one participant's ten decisions are
    # correlated, so a row-wise split would leak their habits across the
    # boundary and report a held-out score that is partly memorisation.
    #
    # Three ways, not two, because lambda is selected on held-out ranking
    # quality — doing that on the test split would make the final number a
    # best-of-grid, not an estimate of generalisation.
    rng = np.random.default_rng(args.seed)
    people = np.unique(cs.participants)
    rng.shuffle(people)
    n_test = max(1, int(len(people) * args.test_frac))
    n_val = max(1, int(len(people) * args.val_frac))
    test_people = set(people[:n_test])
    val_people = set(people[n_test:n_test + n_val])
    where = np.array([2 if p in test_people else 1 if p in val_people else 0
                      for p in cs.participants])
    train, valid, test = cs.subset(where == 0), cs.subset(where == 1), cs.subset(where == 2)
    for nm, s in (("train", train), ("valid", valid), ("test", test)):
        print(f"  {nm:6s}: {len(s):5d} sets / "
              f"{len(np.unique(s.participants)):3d} people")

    print(f"\nsort modes in cohort: "
          f"{dict(Counter(cs.sorts).most_common())}")

    # ---- models ------------------------------------------------------------ #
    models: Dict[str, Dict[str, Any]] = {}

    b_naive = fit_mnl(train, use_position=False, non_negative=False)
    models["naive"] = dict(
        beta=b_naive, weights=to_simplex(b_naive),
        label="components only, no position control (reproduces the published fit)")

    b_pos = fit_mnl(train, use_position=True, non_negative=False)
    models["position_controlled"] = dict(
        beta=b_pos, weights=to_simplex(b_pos),
        label="components + log-rank, unconstrained (inference)")

    # `deployed` is the one the retriever actually gets. Lambda is chosen on the
    # validation people by macro-averaged held-out nDCG, so the strength of the
    # pull toward the hand-set prior is set by evidence rather than taste.
    grid = [0.0, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 1000.0]
    sel = []
    for lam in grid:
        b = fit_mnl(train, use_position=True, non_negative=True,
                    l2=lam, toward_prior=True)
        score, _ = macro_by_sort(valid, to_simplex(b), ndcg_at_k)
        sel.append((score, lam, b))
    best_lam_score, best_lam, b_dep = max(sel, key=lambda t: t[0])
    models["deployed"] = dict(
        beta=b_dep, weights=to_simplex(b_dep),
        label=f"components + log-rank nuisance, non-negative, "
              f"MAP toward hand-set prior (lambda={best_lam:g}, chosen on validation)")

    print("\n" + "-" * 78)
    print("LAMBDA SELECTION — pull toward the hand-set prior's direction")
    print("(validation macro-nDCG@10; lambda=0 is the free fit, "
          "large lambda returns the prior)")
    print("-" * 78)
    for score, lam, b in sel:
        star = "  <-- selected" if lam == best_lam else ""
        print(f"  lambda={lam:8.1f}   valid nDCG@10 {score:.4f}   "
              f"{fmt(to_simplex(b))}{star}")

    print("\n" + "-" * 78)
    print("FITTED COEFFICIENTS (on the training split)")
    print("-" * 78)
    for name, m in models.items():
        print(f"\n{name}  —  {m['label']}")
        print("  raw     : " + "  ".join(f"{d[:6]}={v:+.3f}"
                                         for d, v in zip(DIMS, m["beta"][:5])))
        if len(m["beta"]) == 6:
            print(f"  position: {m['beta'][5]:+.3f}   "
                  f"(positive = pull toward the top of the list)")
        print("  simplex : " + fmt(m["weights"]))

    ll_naive = -neg_loglik(np.append(b_naive, 0.0), train, 0.0)
    ll_pos = -neg_loglik(b_pos, train, 0.0)
    print(f"\nlog-likelihood  no position {ll_naive:>12,.0f}")
    print(f"                + position  {ll_pos:>12,.0f}"
          f"   (Δ2LL = {2 * (ll_pos - ll_naive):,.0f} on 1 df)")

    # ---- held-out evaluation ----------------------------------------------- #
    print("\n" + "-" * 78)
    print("HELD-OUT PERFORMANCE — macro-averaged over sort mode")
    print("(each display condition counts once, so the 75% proximity-sorted")
    print(" majority cannot win the comparison on position bias alone)")
    print("-" * 78)
    # A 50/50 mixture of the elicited posterior and the hand-set prior.
    #
    # This is a deployability decision, stated openly rather than smuggled into
    # the estimator. The fit pins `economic` and `disruption` at zero because
    # their coefficients want to be negative (participants took the dearest room
    # when the list was sorted dearest-first), but a retriever with economic = 0
    # cannot answer "cheapest hotel near Galle Face" and one with disruption = 0
    # throws away the thesis's whole contribution. The study measured booking
    # behaviour on a sorted list; it did not measure constraint satisfaction, so
    # it does not get to zero out a capability it never tested.
    #
    # An equal mixture says exactly that: trust the evidence and the prior
    # equally on the dimensions where the evidence is weak.
    blended = 0.5 * models["deployed"]["weights"] + 0.5 * HANDSET
    models["blended"] = dict(beta=np.append(blended, 0.0), weights=blended,
                             label="0.5 x deployed + 0.5 x hand-set prior")

    cands = {"handset (shipped prior)": HANDSET,
             **{k: v["weights"] for k, v in models.items()}}
    print(f"\n{'weights':28s} {'nDCG@10':>9s} {'top-1':>8s}   per-sort-mode nDCG@10")
    results = {}
    for name, w in cands.items():
        nd, per = macro_by_sort(test, w, ndcg_at_k)
        t1, _ = macro_by_sort(test, w, top1)
        results[name] = dict(ndcg10=nd, top1=t1, per_mode=per)
        detail = " ".join(f"{m[:5]}={v:.3f}" for m, v in per.items())
        print(f"{name:28s} {nd:9.3f} {t1:8.3f}   {detail}")

    best = max(results, key=lambda k: results[k]["ndcg10"])
    print(f"\nbest by macro nDCG@10: {best}")

    # ---- refit on everything for the shipped vector ------------------------- #
    #
    # The splits exist to choose lambda and to earn an unbiased generalisation
    # number. Both jobs are done, so the vector that actually ships is refitted
    # on all the data at the selected lambda — standard practice, and it is what
    # makes the point estimate agree with the bootstrap below, which also
    # resamples the full cohort.
    b_final = fit_mnl(cs, use_position=True, non_negative=True,
                      l2=best_lam, toward_prior=True)
    w_final = to_simplex(b_final)
    w_blend_final = 0.5 * w_final + 0.5 * HANDSET
    models["deployed"]["weights"] = w_final
    models["deployed"]["beta"] = b_final
    models["blended"]["weights"] = w_blend_final
    print("\n" + "-" * 78)
    print(f"SHIPPED VECTORS — refitted on all {len(cs)} sets at lambda={best_lam:g}")
    print("-" * 78)
    print("  elicited: " + fmt(w_final))
    print("  blended : " + fmt(w_blend_final))

    # ---- bootstrap ---------------------------------------------------------- #
    print("\n" + "-" * 78)
    print(f"BOOTSTRAP — {args.bootstrap} replicates, resampled by participant")
    print("-" * 78)
    # Must mirror the `deployed` specification exactly — same position nuisance,
    # same non-negativity, same selected lambda. A bootstrap of a different
    # estimator describes the sampling variation of a model nobody is shipping.
    boot = bootstrap(cs, args.bootstrap, args.seed, use_position=True,
                     non_negative=True, l2=best_lam, toward_prior=True)
    ci: Dict[str, List[float]] = {}
    if len(boot):
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        med = np.median(boot, axis=0)
        print(f"\n{'dimension':16s} {'median':>8s} {'95% CI':>20s}   stable?")
        for i, d in enumerate(DIMS):
            stable = "yes" if lo[i] > 0.02 else "NO — includes ~0"
            print(f"{d:16s} {med[i]:8.3f}   [{lo[i]:6.3f}, {hi[i]:6.3f}]   {stable}")
            ci[d] = [float(lo[i]), float(hi[i])]

        # `spatial` and `accessibility` correlate 0.907, so the logit cannot say
        # how the location mass divides between them — only how much of it there
        # is. Their SUM is the estimand this design actually identifies; the
        # split is an artefact of which of two near-duplicates the optimiser
        # happened to favour, and reading it as "travel time beats proximity"
        # (or the reverse) is over-reading the fit.
        loc = boot[:, 0] + boot[:, 1]
        l_lo, l_hi = np.percentile(loc, [2.5, 97.5])
        print(f"\n{'spatial+access':16s} {np.median(loc):8.3f}   "
              f"[{l_lo:6.3f}, {l_hi:6.3f}]   <-- the identified quantity")
        print(f"{'  split ratio':16s} {np.median(boot[:, 0] / np.maximum(loc, 1e-9)):8.3f}   "
              f"[{np.percentile(boot[:, 0] / np.maximum(loc, 1e-9), 2.5):6.3f}, "
              f"{np.percentile(boot[:, 0] / np.maximum(loc, 1e-9), 97.5):6.3f}]"
              f"   <-- NOT identified (r = 0.907)")
        ci["spatial_plus_accessibility"] = [float(l_lo), float(l_hi)]

    # ---- stage 2: willingness to pay ---------------------------------------- #
    print("\n" + "-" * 78)
    print("STAGE 2 — room choice, for the monetary scale")
    print("-" * 78)
    wtp = fit_room_logit(responses, args.pool_size)
    if wtp:
        print(f"  room choice sets      : {wtp['n_sets']}")
        print(f"  price coefficient     : {wtp['beta_price']:+.6f} per LKR")
        print(f"  refundable            : {wtp['beta_refundable']:+.3f}"
              f"  =  LKR {wtp['wtp_refundable']:,.0f}")
        print(f"  breakfast included    : {wtp['beta_breakfast']:+.3f}"
              f"  =  LKR {wtp['wtp_breakfast']:,.0f}")
    else:
        print("  not enough room-choice variation to fit")

    # ---- persist ------------------------------------------------------------ #
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "material_version": version,
        "source_dump": args.dump.name,
        "facility_definition": args.facility_def,
        "cohort": {
            "choice_sets": len(cs),
            "participants": int(len(np.unique(cs.participants))),
            "pool_size": args.pool_size,
            "failed_attention_dropped": sorted(failed) if args.drop_failed_attention else [],
        },
        "models": {k: {"beta": [float(x) for x in v["beta"]],
                       "weights": {d: float(w) for d, w in zip(DIMS, v["weights"])},
                       "label": v["label"]}
                   for k, v in models.items()},
        "bootstrap_ci_95": ci,
        "held_out": {k: {"ndcg10": v["ndcg10"], "top1": v["top1"],
                         "per_sort_mode": v["per_mode"]}
                     for k, v in results.items()},
        "stage2_wtp": wtp,
        "recommended": best,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out.relative_to(REPO)}")

    if args.emit_profile:
        w = models["deployed"]["weights"]
        print("\n# paste into src/graph/retriever.py")
        print("ELICITED_WEIGHTS = ScoringWeights(")
        print("    " + ", ".join(f"{d}={v:.3f}" for d, v in zip(DIMS, w)) + ",")
        print(")")


# --------------------------------------------------------------------------- #
# Stage 2 — room choice
# --------------------------------------------------------------------------- #
def fit_room_logit(responses: List[dict], pool_size: int) -> Optional[dict]:
    """Logit over the offers inside the chosen hotel.

    These attributes are OBSERVED in rupees rather than latent, so the ratio of
    any coefficient to the price coefficient is a willingness-to-pay — the only
    thing in the study that puts a real currency scale on any of this.
    """
    rows, labels = [], []
    for r in responses:
        if r.get("isAttentionCheck") or len(r.get("options") or []) != pool_size:
            continue
        offers = r.get("roomOptions") or []
        if len(offers) < 2 or not any(o.get("chosen") for o in offers):
            continue
        feats = []
        for o in offers:
            board = (o.get("board_name") or "").lower()
            feats.append([
                float(o["price_lkr"]),
                1.0 if o.get("refundable") else 0.0,
                1.0 if any(b in board for b in
                           ("breakfast", "half board", "full board", "all inclusive")) else 0.0,
                float(o.get("size_sqm") or 0.0),
            ])
        rows.append(np.asarray(feats))
        labels.append(next(i for i, o in enumerate(offers) if o.get("chosen")))
    if len(rows) < 50:
        return None

    m = max(len(a) for a in rows)
    X = np.zeros((len(rows), m, 4))
    mask = np.zeros((len(rows), m), dtype=bool)
    for i, a in enumerate(rows):
        X[i, : len(a)] = a
        mask[i, : len(a)] = True
    y = np.asarray(labels)

    # Price is scaled to units of 10k LKR for conditioning; unscaled after.
    Xs = X.copy()
    Xs[:, :, 0] /= 10_000.0

    def nll(b):
        u = np.where(mask, Xs @ b, -np.inf)
        u = u - u.max(axis=1, keepdims=True)
        lse = np.log(np.exp(np.where(mask, u, -np.inf)).sum(axis=1))
        return float(-(u[np.arange(len(y)), y] - lse).sum())

    b = minimize(nll, np.zeros(4), method="L-BFGS-B").x
    beta_price = b[0] / 10_000.0
    if abs(beta_price) < 1e-12:
        return None
    return {
        "n_sets": len(y),
        "beta_price": float(beta_price),
        "beta_refundable": float(b[1]),
        "beta_breakfast": float(b[2]),
        "wtp_refundable": float(-b[1] / beta_price),
        "wtp_breakfast": float(-b[2] / beta_price),
    }


if __name__ == "__main__":
    main()
