"""
Script: Comparative evaluation of retrieval baselines (proposal RQ3 / Phase 5).

Runs Filter, VectorRAG, and WeightedGraphRAG over the evaluation query set and
reports Precision@K, Recall@K, nDCG@K, MRR — overall and per query category.

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
from collections import defaultdict
from typing import Any, Dict, List

from evaluation.baselines import all_baselines, fetch_city_hotels
from evaluation.gold import relevant_set
from evaluation.metrics import evaluate_ranking, mean_metrics

logging.basicConfig(level=logging.WARNING)  # keep the table clean


def _fmt_row(label: str, m: Dict[str, float], keys: List[str]) -> str:
    cells = " ".join(f"{m.get(key, 0.0):>8.3f}" for key in keys)
    return f"{label:<20s} {cells}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval baselines")
    parser.add_argument("--queryset", default="evaluation/queryset.json")
    parser.add_argument("--out", default="evaluation/results.json")
    args = parser.parse_args()

    spec = json.loads(Path(args.queryset).read_text(encoding="utf-8"))
    city = spec["city"]
    k = int(spec.get("k", 10))
    queries = spec["queries"]

    # Full city pool — used to compute gold relevant sets once per query.
    pool = fetch_city_hotels(city)
    print(f"\nEvaluation: city={city}, k={k}, queries={len(queries)}, pool={len(pool)} hotels\n")

    baselines = all_baselines()
    metric_keys = [f"P@{k}", f"R@{k}", f"nDCG@{k}", "MRR"]

    # results[baseline_name] = list of per-query metric dicts
    results: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    # per category aggregation
    cat_results: Dict[str, Dict[str, List[Dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    per_query_out: List[Dict[str, Any]] = []

    for q in queries:
        gold_set = relevant_set(pool, q["gold"])
        row_out: Dict[str, Any] = {
            "id": q["id"], "question": q["question"],
            "category": q.get("category", "general"),
            "gold": q["gold"], "n_relevant": len(gold_set), "scores": {},
        }
        for b in baselines:
            ranked = b.retrieve(q["question"], city, k)
            m = evaluate_ranking(ranked, gold_set, k)
            results[b.name].append(m)
            cat_results[q.get("category", "general")][b.name].append(m)
            row_out["scores"][b.name] = {kk: round(v, 4) for kk, v in m.items()}
        per_query_out.append(row_out)

    # ---- Overall table -----------------------------------------------------
    print("=" * 70)
    print("OVERALL (mean over all queries)")
    print("-" * 70)
    print(f"{'System':<20s} {' '.join(f'{key:>8s}' for key in metric_keys)}")
    overall: Dict[str, Dict[str, float]] = {}
    for b in baselines:
        agg = mean_metrics(results[b.name])
        overall[b.name] = agg
        print(_fmt_row(b.name, agg, metric_keys))

    # ---- Per-category table ------------------------------------------------
    print("\n" + "=" * 70)
    print("BY CATEGORY (mean nDCG@%d)" % k)
    print("-" * 70)
    cats = sorted(cat_results.keys())
    print(f"{'System':<20s} " + " ".join(f"{c[:10]:>12s}" for c in cats))
    for b in baselines:
        cells = []
        for c in cats:
            agg = mean_metrics(cat_results[c][b.name])
            cells.append(f"{agg.get(f'nDCG@{k}', 0.0):>12.3f}")
        print(f"{b.name:<20s} " + " ".join(cells))

    # ---- Winner summary ----------------------------------------------------
    print("\n" + "=" * 70)
    best = max(overall.items(), key=lambda kv: kv[1].get(f"nDCG@{k}", 0.0))
    print(f"Best overall nDCG@{k}: {best[0]} ({best[1].get(f'nDCG@{k}', 0.0):.3f})")
    print("=" * 70)

    out = {
        "city": city, "k": k, "n_queries": len(queries),
        "pool_size": len(pool),
        "overall": {b: overall[b] for b in overall},
        "by_category": {
            c: {b.name: mean_metrics(cat_results[c][b.name]) for b in baselines}
            for c in cats
        },
        "per_query": per_query_out,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nDetailed results written to {args.out}")


if __name__ == "__main__":
    main()
