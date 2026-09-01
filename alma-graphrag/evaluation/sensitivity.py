"""
Sensitivity analysis — do the conclusions survive the arbitrary constants?

Two constants in this benchmark were chosen by hand and decide a lot:

1. **Gold tolerance bands** (`evaluation/gold.py`). Price +15%, rating 0.2,
   star 1, travel +2 min. These decide every relevance label. A reviewer is
   entitled to ask whether the reported system ordering is an artefact of them.

2. **Missing-price policy** (`src/graph/retriever.py`). 41% of the Colombo pool
   has no price, and the original default scored those hotels a neutral 0.5 —
   which ranks an unpriced hotel above every hotel more expensive than the pool
   midpoint. That is a strong, undocumented assumption applied to four hotels in
   ten.

Neither can be argued away, but both can be *swept*. If the ordering of systems
is stable across the grid, the constants are not load-bearing and the sweep
table settles the question in one figure. If the ordering flips, that is a
finding and must be reported rather than buried under one preferred setting.

Usage
-----
    python evaluation/sensitivity.py --bands
    python evaluation/sensitivity.py --price
    python evaluation/sensitivity.py --bands --price --out evaluation/results_sensitivity.json

Runtime note: every cell re-runs the whole evaluation. Pass --no-llm (the
default here) to keep it offline, and use --band-scales to shorten the grid.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
import logging
from typing import Any, Dict, List, Sequence

from evaluation.gold import DEFAULT_BANDS
from evaluation.harness import DEFAULT_QUERYSET, run_evaluation
from src.graph.retriever import PRICE_POLICIES

logging.basicConfig(level=logging.ERROR)

# Scale factors applied uniformly to every band. 1.0 is the published default.
# 0.0 collapses the bands entirely (strict pass/fail — the pre-graded gold),
# which is a useful extreme to include.
DEFAULT_SCALES: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)


def _ordering(overall: Dict[str, Dict[str, float]], k: int) -> List[str]:
    return [name for name, _ in sorted(
        overall.items(), key=lambda kv: -kv[1].get(f"nDCG@{k}", 0.0)
    )]


def _kendall_tau(a: Sequence[str], b: Sequence[str]) -> float:
    """Rank correlation between two system orderings. 1.0 = identical."""
    common = [x for x in a if x in b]
    if len(common) < 2:
        return 1.0
    pos_b = {name: i for i, name in enumerate(b)}
    concordant = discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            x, y = common[i], common[j]
            if pos_b[x] < pos_b[y]:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 1.0


def sweep_bands(queryset: str, scales: Sequence[float],
                **eval_opts: Any) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    reference_order: List[str] = []

    for scale in scales:
        bands = DEFAULT_BANDS.scaled(scale)
        out = run_evaluation(queryset, no_human=True, bands=bands,
                             include_sensitivity=False, **eval_opts)
        k = out["k"]
        order = _ordering(out["overall"], k)
        if not reference_order:
            reference_order = order
        gold_sizes = [r["n_relevant"] for r in out["per_query"]]
        rows.append({
            "scale": scale,
            "bands": bands.to_dict(),
            "winner": order[0],
            "order": order,
            "tau_vs_default": round(_kendall_tau(reference_order, order), 4),
            "mean_gold_size": round(sum(gold_sizes) / len(gold_sizes), 2),
            "ndcg": {name: round(m.get(f"nDCG@{k}", 0.0), 4)
                     for name, m in out["overall"].items()},
        })
        print(f"  scale {scale:>4.1f}  winner={order[0]:<32s} "
              f"mean gold {rows[-1]['mean_gold_size']:>6.2f}  "
              f"tau {rows[-1]['tau_vs_default']:+.3f}")

    winners = {r["winner"] for r in rows}
    return {
        "parameter": "gold tolerance bands",
        "scales": list(scales),
        "rows": rows,
        "winner_stable": len(winners) == 1,
        "winners": sorted(winners),
        "min_tau": round(min(r["tau_vs_default"] for r in rows), 4),
    }


def sweep_price(queryset: str, policies: Sequence[str],
                **eval_opts: Any) -> Dict[str, Any]:
    """Compare the proposed system under each missing-price policy.

    All policies appear as rows in ONE run, so they are scored on identical gold
    and identical queries and the comparison is exactly paired.
    """
    out = run_evaluation(queryset, no_human=True,
                         price_policies=list(policies),
                         include_sensitivity=False, **eval_opts)
    k = out["k"]
    graph_rows = {name: m for name, m in out["overall"].items()
                  if name.startswith("WeightedGraphRAG")}
    order = _ordering(graph_rows, k)
    for name in order:
        print(f"  {name:<44s} nDCG@{k} {graph_rows[name][f'nDCG@{k}']:.4f}")
    return {
        "parameter": "missing-price policy",
        "policies": list(policies),
        "pool_missing_rate": out["price_policy"]["pool_missing_rate"],
        "pool_missing_price": out["price_policy"]["pool_missing_price"],
        "best": order[0] if order else None,
        "ndcg": {name: round(m.get(f"nDCG@{k}", 0.0), 4)
                 for name, m in graph_rows.items()},
        "spread": round(
            max(m.get(f"nDCG@{k}", 0.0) for m in graph_rows.values())
            - min(m.get(f"nDCG@{k}", 0.0) for m in graph_rows.values()), 4
        ) if graph_rows else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queryset", default=str(DEFAULT_QUERYSET))
    ap.add_argument("--out", default="evaluation/results_sensitivity.json")
    ap.add_argument("--bands", action="store_true", help="sweep gold tolerance bands")
    ap.add_argument("--price", action="store_true", help="sweep missing-price policy")
    ap.add_argument("--band-scales", default=",".join(str(s) for s in DEFAULT_SCALES))
    ap.add_argument("--with-llm", action="store_true",
                    help="include the LLM re-ranker (costs one API call per "
                         "query per cell; cached, but the first run is slow)")
    args = ap.parse_args()

    if not (args.bands or args.price):
        args.bands = args.price = True

    eval_opts: Dict[str, Any] = {
        "include_llm": args.with_llm,
        "include_floors": False,
        "include_ablations": False,
        "include_ltr": False,
    }

    report: Dict[str, Any] = {"queryset": args.queryset}

    if args.bands:
        scales = [float(s) for s in args.band_scales.split(",") if s.strip()]
        print(f"Sweeping gold tolerance bands over {len(scales)} scales...")
        report["bands"] = sweep_bands(args.queryset, scales, **eval_opts)
        b = report["bands"]
        print(f"\n  winner stable across all scales: {b['winner_stable']}"
              f"  (winners: {', '.join(b['winners'])}, min tau {b['min_tau']:+.3f})")
        if not b["winner_stable"]:
            print("  The reported winner DEPENDS on the tolerance bands. "
                  "Report this table, not a single setting.")

    if args.price:
        print(f"\nSweeping missing-price policies {PRICE_POLICIES}...")
        report["price"] = sweep_price(args.queryset, PRICE_POLICIES, **eval_opts)
        p = report["price"]
        print(f"\n  {p['pool_missing_price']} of the pool "
              f"({p['pool_missing_rate']:.0%}) have no price.")
        print(f"  nDCG spread across policies: {p['spread']:.4f}")
        if p["spread"] > 0.02:
            print("  The policy materially changes the result — state which one "
                  "the headline number uses and show this table.")

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
