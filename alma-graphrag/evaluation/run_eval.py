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
    return f"{label:<20s} {cells}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval baselines")
    parser.add_argument("--queryset", default="evaluation/queryset.json")
    parser.add_argument("--out", default="evaluation/results.json")
    parser.add_argument("--gold-human", default="evaluation/gold_human.json",
                        help="human annotation file; used per-query when present "
                             "(--no-human forces rule-based gold)")
    parser.add_argument("--no-human", action="store_true",
                        help="ignore human gold even if the file exists")
    args = parser.parse_args()

    out = run_evaluation(args.queryset, args.gold_human, args.no_human)
    city, k = out["city"], out["k"]
    system_order = out["system_order"]
    metric_keys = [f"P@{k}", f"R@{k}", f"nDCG@{k}", "MRR"]

    meta = out["gold_meta"]
    if meta["used_human"]:
        print(f"Using human gold for {meta['human_queries']} queries "
              f"(alpha={meta['alpha']}, {args.gold_human}); rule-based gold for the rest.")

    print(f"\nEvaluation: city={city}, k={k}, queries={out['n_queries']}, "
          f"pool={out['pool_size']} hotels\n")

    # ---- Overall table -----------------------------------------------------
    print("=" * 70)
    print("OVERALL (mean over all queries)")
    print("-" * 70)
    print(f"{'System':<20s} {' '.join(f'{key:>8s}' for key in metric_keys)}")
    for name in system_order:
        print(_fmt_row(name, out["overall"][name], metric_keys))

    # ---- Per-category table ------------------------------------------------
    print("\n" + "=" * 70)
    print("BY CATEGORY (mean nDCG@%d)" % k)
    print("-" * 70)
    cats = sorted(out["by_category"].keys())
    print(f"{'System':<20s} " + " ".join(f"{c[:10]:>12s}" for c in cats))
    for name in system_order:
        cells = [f"{out['by_category'][c][name].get(f'nDCG@{k}', 0.0):>12.3f}" for c in cats]
        print(f"{name:<20s} " + " ".join(cells))

    # ---- Winner summary ----------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Best overall nDCG@{k}: {out['best_system']} ({out['best_ndcg']:.3f})")
    print("=" * 70)

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nDetailed results written to {args.out}")


if __name__ == "__main__":
    main()
