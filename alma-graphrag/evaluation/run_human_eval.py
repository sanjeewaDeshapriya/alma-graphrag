"""
Script: choice-based evaluation against real human bookings.

The companion to run_eval.py. That script grades against rules the author wrote
(evaluation/gold.py); this one grades against what 247 participants in the
discrete-choice study actually booked. Report both — the rule gold cannot see
ranking quality, and the choice data cannot see price sensitivity.

Usage:
    python evaluation/run_human_eval.py
    python evaluation/run_human_eval.py --cohort all --holdout 0.3
    python evaluation/run_human_eval.py --deployed-index   # asymmetric, for comparison
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
import logging

from evaluation.human_eval import (DEFAULT_HUMAN_RESULTS, DEFAULT_MATERIAL,
                                   DEFAULT_RESPONSES, run_human_evaluation)

logging.basicConfig(level=logging.WARNING)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate against human booking choices")
    ap.add_argument("--responses", default=str(DEFAULT_RESPONSES))
    ap.add_argument("--material", default=str(DEFAULT_MATERIAL))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--cohort", choices=["all", "attention", "clean"], default="clean")
    ap.add_argument("--holdout", type=float, default=0.3,
                    help="fraction of PARTICIPANTS held out (never rows)")
    ap.add_argument("--min-votes", type=int, default=2,
                    help="votes needed for a hotel to enter the per-task gold set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deployed-index", action="store_true",
                    help="score the text baselines through the deployed pgvector "
                         "index (city-centre travel time) instead of rebuilding "
                         "anchor-relative documents. Reproduces production, but "
                         "the baselines cannot see the task anchor.")
    ap.add_argument("--restrict-to-indexed", action="store_true",
                    help="keep only hotels present in the pgvector index")
    ap.add_argument("--out", default=str(DEFAULT_HUMAN_RESULTS))
    args = ap.parse_args()

    out = run_human_evaluation(
        responses_path=args.responses, material_path=args.material, k=args.k,
        cohort=args.cohort, holdout=args.holdout, min_votes=args.min_votes,
        seed=args.seed, anchor_fair=not args.deployed_index,
        restrict_to_indexed=args.restrict_to_indexed,
    )
    k = out["k"]

    print(f"\nChoice-based evaluation — material {out['material_version']} "
          f"({out['normalisation']})")
    print(f"  cohort '{out['cohort']}': {out['cohort_stats']}")
    print(f"  train {out['n_train_participants']} people / {out['n_train_choices']} choices"
          f"   |   test {out['n_test_participants']} people / {out['n_test_choices']} choices")
    print(f"  candidate set {out['candidate_set_size']} hotels per task"
          f"   |   baselines: {'anchor-fair' if out['anchor_fair'] else 'deployed index'}")
    if out["skipped"]:
        print(f"  skipped rows: {out['skipped']}")

    print("\n" + "=" * 96)
    print(f"A. PER-CHOICE — ground truth = the hotel that participant booked "
          f"({out['n_test_choices']} choices)")
    print("-" * 96)
    # The cap is 1/K, not 1/pool: with exactly one relevant item, a perfect
    # ranking puts it at position 1 and P@K is 1/K. Dividing by the candidate
    # set size printed 0.031 while the systems were scoring 0.089, which reads
    # as an impossible result rather than a mislabelled ceiling.
    print(f"  one relevant item per query, so P@{k} is capped at {1/k:.3f} "
          f"and R@{k} equals the hit rate")
    keys = [f"P@{k}", f"R@{k}", f"nDCG@{k}", "MRR", "mean_rank"]
    print(f"\n{'System':<36}" + "".join(f"{key:>11}" for key in keys))
    for name in out["system_order"]:
        m = out["per_choice"][name]
        print(f"{name:<36}" + "".join(f"{m.get(key, 0.0):>11.3f}" for key in keys))
    rb = out["random_baseline"]
    print(f"{'Random':<36}{rb[f'P@{k}']:>11.3f}{rb[f'R@{k}']:>11.3f}"
          f"{'-':>11}{'-':>11}{rb['mean_rank']:>11.2f}")

    print("\n" + "=" * 96)
    print(f"B. PER-TASK — ground truth = hotels chosen by >= {out['min_votes']} held-out participants")
    print("-" * 96)
    tkeys = [f"P@{k}", f"R@{k}", f"nDCG@{k}", f"gradedNDCG@{k}", "MRR"]
    print(f"\n{'System':<36}" + "".join(f"{key:>15}" for key in tkeys))
    for name in out["system_order"]:
        m = out["per_task"][name]
        print(f"{name:<36}" + "".join(f"{m.get(key, 0.0):>15.3f}" for key in tkeys))

    print("\n" + "=" * 96)
    sig = out["significance"]
    print(f"C. SIGNIFICANCE — {sig['tests']}")
    print(f"   reference: {sig['reference']}")
    print("-" * 96)
    print(f"\n{'vs':<36}{'mean diff':>12}{'95% CI':>24}{'p':>10}")
    for name, block in sig["vs"].items():
        ci = f"[{block['ci_low']:+.3f}, {block['ci_high']:+.3f}]"
        p = "<0.001" if block["p"] < 0.001 else f"{block['p']:.3f}"
        print(f"{name:<36}{block['mean_diff']:>+12.3f}{ci:>24}{p:>10}")

    print("\n" + "=" * 96)
    print(f"Best per-choice nDCG@{k}: {out['best_system']} ({out['best_ndcg']:.3f})")
    print("=" * 96)

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nDetailed results written to {args.out}")


if __name__ == "__main__":
    main()
