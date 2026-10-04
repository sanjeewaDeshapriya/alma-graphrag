"""
Build the delay-graded query set: evaluation/queryset_disruption.json.

The main set's ten "disruption" queries are graded on `max_travel_time`, so no
query in the benchmark ever graded a hotel on how much traffic delay its route
carries (results_robustness.json: queries_graded_on_delay = 0). These queries
use `max_added_delay_min` instead, which evaluation/gold.py already grades with
a 3-minute tolerance band.

Thresholds are 3, 5 and 10 minutes of ADDED delay on top of free-flow time:
  3  - a traveller with a fixed appointment; almost any congestion matters
  5  - a normal commute margin
  10 - only a serious jam or closure counts
They are research assumptions, frozen in the file, not fitted to any system.

On the live July snapshot every route delay is at most 0.2 min, so every hotel
passes and the gold is degenerate. The set is meant for peak-hour snapshots
(scripts/collect_peak_traffic.py) and for the controlled scenarios in
evaluation/disruption_scenarios.py, which skip degenerate queries explicitly.

Every template contains a phrase in query_parser._TRAFFIC_TERMS, so the regex
pass alone sets avoid_traffic; tests/test_delay_queries.py enforces that.

Usage:
    python evaluation/build_delay_queries.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
from typing import Any, Dict, List

THRESHOLDS = (3, 5, 10)

PURE_TEMPLATES = (
    "hotels in colombo where traffic delay stays under {t} minutes",
    "avoid congestion, no more than {t} minutes of added drive time",
    "hotels not stuck in traffic, under {t} min added delay",
    "rush hour safe hotels with under {t} extra minutes on the road",
    "hotels unaffected by a road closure, less than {t} min added delay",
)

# (template, extra gold constraints). Mixed thresholds use 5 and 10 only: at 3
# minutes the joint gold is often empty once price is also required, and 41% of
# the pool has no price.
MIXED_TEMPLATES = (
    ("hotels under 30000 lkr with less than {t} minutes traffic delay", {"max_price": 30000}),
    ("hotels under 45000 lkr that avoid congestion, under {t} min added delay", {"max_price": 45000}),
    ("hotels with rating 4.3 or higher and under {t} minutes traffic delay", {"min_rating": 4.3}),
)
MIXED_THRESHOLDS = (5, 10)


def build_queries() -> List[Dict[str, Any]]:
    queries: List[Dict[str, Any]] = []
    for t in THRESHOLDS:
        for tpl in PURE_TEMPLATES:
            queries.append({
                "question": tpl.format(t=t),
                "gold": {"max_added_delay_min": float(t)},
                "category": "delay",
            })
    for t in MIXED_THRESHOLDS:
        for tpl, extra in MIXED_TEMPLATES:
            queries.append({
                "question": tpl.format(t=t),
                "gold": {**extra, "max_added_delay_min": float(t)},
                "category": "delay_multi",
            })
    for i, q in enumerate(queries, start=1):
        q["id"] = f"d{i:03d}"
    return [{"id": q["id"], **{k: v for k, v in q.items() if k != "id"}} for q in queries]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "queryset_disruption.json")
    args = ap.parse_args()
    queries = build_queries()
    spec = {
        "description": ("Delay-graded queries (max_added_delay_min, thresholds 3/5/10 min). "
                        "Degenerate on off-peak snapshots; use with peak-hour data or "
                        "evaluation/disruption_scenarios.py. Built by evaluation/build_delay_queries.py."),
        "city": "Colombo",
        "k": 10,
        "queries": queries,
    }
    args.out.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"wrote {len(queries)} queries to {args.out}")


if __name__ == "__main__":
    main()
