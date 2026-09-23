"""
Generate annotation sheets (one CSV per annotator) for the human gold standard.

Candidate pooling follows standard IR practice: for each query, the pool is the
union of top-k results from every baseline system, plus a seeded random sample
of extra hotels so pool bias can be estimated. Each annotator receives the SAME
query/hotel pairs in a DIFFERENT (seeded) order, with system provenance hidden.

Usage:
    python evaluation/annotation/make_sheets.py                 # 3 annotators
    python evaluation/annotation/make_sheets.py --annotators 5 --extras 3

Requires a running Neo4j with the ingested graph (same as run_eval.py).
Outputs to evaluation/annotation/sheets/:
    annotator_1.csv ... annotator_N.csv   (relevance column left blank)
    pool_meta.json                        (pooling provenance, for the paper)
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import json
import logging
import random

from evaluation.baselines import all_baselines, fetch_city_hotels

logging.basicConfig(level=logging.WARNING)

SHEETS_DIR = Path(__file__).resolve().parent / "sheets"


def build_pool(question: str, city: str, k: int, baselines, hotels_by_id, rng, extras: int):
    """Union of top-k across systems + seeded random extras from the rest."""
    pooled: set[str] = set()
    provenance: dict[str, list[str]] = {}
    for b in baselines:
        ranked = [str(h) for h in b.retrieve(question, city, k)]
        provenance[b.name] = ranked
        pooled.update(ranked)

    remainder = sorted(set(hotels_by_id) - pooled)
    extra_ids = rng.sample(remainder, min(extras, len(remainder)))
    pooled.update(extra_ids)
    provenance["random_extras"] = extra_ids
    return sorted(pooled), provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate annotation sheets")
    parser.add_argument("--queryset", default="evaluation/queryset.json")
    parser.add_argument("--annotators", type=int, default=3)
    parser.add_argument("--extras", type=int, default=2,
                        help="random non-retrieved hotels added per query")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-floors", action="store_true",
                        help="also pool from the random and popularity floors")
    parser.add_argument("--pool-circular", action="store_true",
                        help="also pool from the rule-aligned LTR diagnostic")
    args = parser.parse_args()

    spec = json.loads(Path(args.queryset).read_text(encoding="utf-8"))
    city, k, queries = spec["city"], int(spec.get("k", 10)), spec["queries"]

    hotels = fetch_city_hotels(city)
    hotels_by_id = {str(h["id"]): h for h in hotels}
    # Deterministic systems only. The pool a judgement is made against is part
    # of the gold standard's provenance, so it has to be reproducible: the LLM
    # re-ranker returns a different ranking on consecutive calls at the same
    # temperature, which would mean two runs of this script produce different
    # sheets from identical inputs. It also costs one model call per query for
    # a set the other systems already cover.
    #
    # The floors are excluded from POOLING (they still run in the evaluation).
    # Pooling covers the hotels a credible system would show a traveller; a
    # random ranker contributes arbitrary ones, which inflates every annotator's
    # workload without making the gold more complete. The rule-aligned LTR is
    # excluded for the same reason in reverse: it reconstructs the rule gold, so
    # pooling from it would seed the human labels with the labels they replace.
    baselines = all_baselines(include_llm=False, include_floors=args.pool_floors)
    pool_systems = [b for b in baselines
                    if args.pool_circular or b.name != "LTR"]
    rng = random.Random(args.seed)

    rows = []
    meta = {"city": city, "k": k, "seed": args.seed,
            "pooled_from": [b.name for b in pool_systems],
            "excluded_from_pooling": sorted(
                {b.name for b in baselines} - {b.name for b in pool_systems}),
            "queries": {}}
    for q in queries:
        pool_ids, provenance = build_pool(
            q["question"], city, k, pool_systems, hotels_by_id, rng, args.extras
        )
        meta["queries"][q["id"]] = {"pool_size": len(pool_ids), "provenance": provenance}
        for hid in pool_ids:
            h = hotels_by_id[hid]
            tt = h.get("travel_time_traffic_min") or h.get("travel_time_min")
            rows.append({
                "query_id": q["id"],
                "question": q["question"],
                "hotel_id": hid,
                "hotel_name": h.get("name") or "",
                "price_lkr": h.get("price") if h.get("price") is not None else "",
                "rating": h.get("rating") if h.get("rating") is not None else "",
                "star": h.get("star") if h.get("star") is not None else "",
                "travel_time_min": tt if tt is not None else "",
                "amenities": "; ".join(h.get("amenities") or []),
                "relevance": "",  # annotator fills 0 / 1 / 2
            })

    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for i in range(1, args.annotators + 1):
        shuffled = rows[:]
        random.Random(args.seed + i).shuffle(shuffled)
        path = SHEETS_DIR / f"annotator_{i}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(shuffled)
        print(f"Wrote {path} ({len(shuffled)} judgments)")

    (SHEETS_DIR / "pool_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    total = len(rows)
    print(f"\n{len(queries)} queries, mean pool size {total / len(queries):.1f}, "
          f"{total} judgments per annotator.")
    print("Next: distribute sheets, then run evaluation/annotation/aggregate.py "
          "on the filled copies.")


if __name__ == "__main__":
    main()
