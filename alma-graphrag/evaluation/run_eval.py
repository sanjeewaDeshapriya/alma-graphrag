"""
Script: Comparative evaluation of retrieval baselines (proposal RQ3 / Phase 5).

Runs Filter, VectorRAG, and WeightedGraphRAG over the evaluation query set and
reports Precision@K, Recall@K, nDCG@K, MRR — overall and per query category.
Computation lives in evaluation/harness.py so this CLI and the /eval API return
identical numbers.

Usage:
    python evaluation/run_eval.py
    python evaluation/run_eval.py --queryset evaluation/queryset.json --out evaluation/results.json
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
import logging
from typing import Dict, List

from evaluation.harness import run_evaluation

logging.basicConfig(level=logging.WARNING)  # keep the table clean


def _fmt_row(label: str, m: Dict[str, float], keys: List[str]) -> str:
    cells = " ".join(f"{m.get(key, 0.0):>8.3f}" for key in keys)
    return f"{label:<28s} {cells}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval baselines")
    parser.add_argument("--queryset", default="evaluation/queryset.json")
    parser.add_argument("--out", default="evaluation/results.json")
    parser.add_argument("--gold-human", default="evaluation/gold_human.json",
                        help="human annotation file; used per-query when present "
                             "(--no-human forces rule-based gold)")
    parser.add_argument("--no-human", action="store_true",
                        help="ignore human gold even if the file exists")
    parser.add_argument("--weight-profiles", default="",
                        help="comma-separated composite-weight profiles to add as "
                             "extra WeightedGraphRAG rows, e.g. "
                             "'handset,elicited,blended' (see WEIGHT_PROFILES)")
    parser.add_argument("--price-policies", default="",
                        help="comma-separated missing-price policies to add as "
                             "extra rows: neutral,worst,median,exclude. 41%% of "
                             "the pool has no price, so this choice matters — "
                             "report the sweep, not one setting.")
    parser.add_argument("--weight-policies", default="",
                        help="comma-separated weight policies to add as extra "
                             "rows: handtuned,learned (see src/graph/weight_policy.py)")
    parser.add_argument("--no-llm", action="store_true",
                        help="skip the LLM re-ranker baseline (no API calls)")
    parser.add_argument("--no-floors", action="store_true",
                        help="skip the Random and Popularity floor baselines")
    parser.add_argument("--no-ablations", action="store_true",
                        help="skip the no-diffusion ablation row")
    parser.add_argument("--no-sensitivity", action="store_true",
                        help="skip the weight-sensitivity diagnostic (faster)")
    parser.add_argument("--band-scale", type=float, default=1.0,
                        help="uniformly scale the gold tolerance bands; 1.0 is "
                             "the published default. Use evaluation/sensitivity.py "
                             "for a full sweep.")
    args = parser.parse_args()

    profiles = [p.strip() for p in args.weight_profiles.split(",") if p.strip()]
    price_policies = [p.strip() for p in args.price_policies.split(",") if p.strip()]
    weight_policies = [p.strip() for p in args.weight_policies.split(",") if p.strip()]

    from evaluation.gold import DEFAULT_BANDS
    bands = DEFAULT_BANDS.scaled(args.band_scale) if args.band_scale != 1.0 else DEFAULT_BANDS

    out = run_evaluation(
        args.queryset, args.gold_human, args.no_human, profiles,
        price_policies=price_policies, weight_policies=weight_policies,
        bands=bands,
        include_sensitivity=not args.no_sensitivity,
        include_llm=not args.no_llm,
        include_floors=not args.no_floors,
        include_ablations=not args.no_ablations,
    )
    city, k = out["city"], out["k"]
    system_order = out["system_order"]
    metric_keys = [f"P@{k}", f"R@{k}", f"nDCG@{k}", "MRR"]

    meta = out["gold_meta"]
    if meta["used_human"]:
        print(f"Using human gold for {meta['human_queries']} queries "
              f"(alpha={meta['alpha']}, {args.gold_human}); rule-based gold for the rest.")

    print(f"\nEvaluation: city={city}, k={k}, queries={out['n_queries']}, "
          f"pool={out['pool_size']} hotels\n")

    wp = out.get("weight_profiles") or {}
    if wp.get("vectors"):
        print("Composite-weight profiles in play (default=%s):" % wp.get("default"))
        for name, vec in wp["vectors"].items():
            print(f"  {name:<10s} " + ", ".join(f"{d}={v}" for d, v in vec.items()))
        print()

    # ---- Overall table -----------------------------------------------------
    print("=" * 92)
    print("OVERALL (mean over all queries)")
    print("-" * 92)
    print(f"{'System':<28s} {' '.join(f'{key:>8s}' for key in metric_keys)}")
    for name in system_order:
        print(_fmt_row(name, out["overall"][name], metric_keys))

    # ---- Per-category table ------------------------------------------------
    print("\n" + "=" * 92)
    print("BY CATEGORY (mean nDCG@%d)" % k)
    print("-" * 92)
    cats = sorted(out["by_category"].keys())
    print(f"{'System':<28s} " + " ".join(f"{c[:10]:>12s}" for c in cats))
    for name in system_order:
        cells = [f"{out['by_category'][c][name].get(f'nDCG@{k}', 0.0):>12.3f}" for c in cats]
        print(f"{name:<28s} " + " ".join(cells))

    # ---- Winner summary ----------------------------------------------------
    print("\n" + "=" * 92)
    print(f"Best overall nDCG@{k}: {out['best_system']} ({out['best_ndcg']:.3f})")
    print("=" * 92)

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nDetailed results written to {args.out}")


if __name__ == "__main__":
    main()
