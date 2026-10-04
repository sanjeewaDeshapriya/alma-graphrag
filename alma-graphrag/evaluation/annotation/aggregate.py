"""
Aggregate filled annotation sheets into a human gold standard.

Reads every ``annotator_*.csv`` in evaluation/annotation/sheets/ (the filled
copies), reports Krippendorff's alpha (interval metric over the 0/1/2 scale),
and writes evaluation/gold_human.json with majority-vote relevant sets.
run_eval.py automatically prefers gold_human.json when it exists.

Usage:
    python evaluation/annotation/aggregate.py
    python evaluation/annotation/aggregate.py --sheets-dir path/to/filled --min-alpha 0.667
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import json
from collections import defaultdict
from datetime import date

from evaluation.annotation.agreement import (aggregate_gold, aggregate_graded,
                                             krippendorff_alpha)

SHEETS_DIR = Path(__file__).resolve().parent / "sheets"
GOLD_OUT = Path(__file__).resolve().parents[1] / "gold_human.json"
MIN_ANNOTATORS = 3


def read_sheets(sheets_dir: Path):
    """-> {query_id: {hotel_id: [score per annotator (None if blank/missing)]}}"""
    files = sorted(sheets_dir.glob("annotator_*.csv"))
    if not files:
        sys.exit(f"No annotator_*.csv files found in {sheets_dir}")

    labels: dict = defaultdict(lambda: defaultdict(lambda: [None] * len(files)))
    for idx, path in enumerate(files):
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                judgments = labels[row["query_id"]][row["hotel_id"]]
                raw = (row.get("relevance") or "").strip()
                if raw == "":
                    continue
                score = float(raw)
                if score not in (0.0, 1.0, 2.0):
                    sys.exit(f"{path.name}: invalid relevance {raw!r} for "
                             f"{row['query_id']}/{row['hotel_id']} (must be 0, 1 or 2)")
                judgments[idx] = score
    return labels, [p.name for p in files]


def annotation_status(labels: dict, annotator_count: int) -> dict:
    """Summarise coverage before human labels are allowed to become gold."""
    units = [judgments for per_hotel in labels.values() for judgments in per_hotel.values()]
    complete = [unit for unit in units if len(unit) == annotator_count and all(v is not None for v in unit)]
    return {
        "items": len(units),
        "judgments": sum(1 for unit in units for value in unit if value is not None),
        "complete_items": len(complete),
        "missing_judgments": sum(1 for unit in units for value in unit if value is None),
        "complete_units": complete,
    }


def validate_annotations(labels: dict, annotator_count: int, min_alpha: float) -> tuple[dict, float]:
    """Fail closed: incomplete or unreliable annotation is not evaluation gold."""
    status = annotation_status(labels, annotator_count)
    if status["items"] == 0:
        raise ValueError("no annotation items found")
    if status["missing_judgments"]:
        raise ValueError(
            f"{status['missing_judgments']} judgments are missing across "
            f"{status['items'] - status['complete_items']} items"
        )
    alpha = krippendorff_alpha(status["complete_units"])
    if alpha < min_alpha:
        raise ValueError(
            f"Krippendorff's alpha {alpha:.3f} is below the required {min_alpha:.3f}"
        )
    return status, alpha


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate annotation sheets")
    parser.add_argument("--sheets-dir", default=str(SHEETS_DIR))
    parser.add_argument("--out", default=str(GOLD_OUT))
    parser.add_argument("--min-alpha", type=float, default=0.667,
                        help="minimum required Krippendorff alpha")
    parser.add_argument("--min-annotators", type=int, default=MIN_ANNOTATORS,
                        help="minimum independent judgments required per item")
    args = parser.parse_args()

    labels, sheet_names = read_sheets(Path(args.sheets_dir))
    if len(sheet_names) < args.min_annotators:
        raise SystemExit(
            f"Need at least {args.min_annotators} annotation sheets; found {len(sheet_names)}"
        )
    try:
        status, alpha = validate_annotations(labels, len(sheet_names), args.min_alpha)
    except ValueError as exc:
        raise SystemExit(f"REFUSING human gold: {exc}") from exc

    gold = aggregate_gold(labels)
    graded = aggregate_graded(labels)
    out = {
        "method": "majority vote over graded 0/1/2 judgments at >=1 decides "
                  "relevance (ties non-relevant); the lower median judgment is "
                  "kept as the nDCG gain",
        "annotators": sheet_names,
        "date": date.today().isoformat(),
        "krippendorff_alpha_interval": round(alpha, 4),
        "n_items": status["items"],
        "n_judgments": status["judgments"],
        "relevant": {qid: sorted(ids) for qid, ids in sorted(gold.items())},
        "graded": {qid: dict(sorted(per_hotel.items()))
                   for qid, per_hotel in sorted(graded.items())},
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Annotators: {len(sheet_names)}  items: {status['items']}  "
          f"judgments: {status['judgments']}")
    print(f"Krippendorff's alpha (interval): {alpha:.3f}")
    fully = sum(1 for per in graded.values() for g in per.values() if g == 2)
    partly = sum(1 for per in graded.values() for g in per.values() if g == 1)
    print(f"Human gold written to {args.out} "
          f"({sum(len(v) for v in gold.values())} relevant pairs across {len(gold)} "
          f"queries; {fully} graded 2, {partly} graded 1)")


if __name__ == "__main__":
    main()
