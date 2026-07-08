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

from evaluation.annotation.agreement import aggregate_gold, krippendorff_alpha

SHEETS_DIR = Path(__file__).resolve().parent / "sheets"
GOLD_OUT = Path(__file__).resolve().parents[1] / "gold_human.json"


def read_sheets(sheets_dir: Path):
    """-> {query_id: {hotel_id: [score per annotator (None if blank/missing)]}}"""
    files = sorted(sheets_dir.glob("annotator_*.csv"))
    if not files:
        sys.exit(f"No annotator_*.csv files found in {sheets_dir}")

    labels: dict = defaultdict(lambda: defaultdict(lambda: [None] * len(files)))
    for idx, path in enumerate(files):
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                raw = (row.get("relevance") or "").strip()
                if raw == "":
                    continue
                score = float(raw)
                if score not in (0.0, 1.0, 2.0):
                    sys.exit(f"{path.name}: invalid relevance {raw!r} for "
                             f"{row['query_id']}/{row['hotel_id']} (must be 0, 1 or 2)")
                labels[row["query_id"]][row["hotel_id"]][idx] = score
    return labels, [p.name for p in files]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate annotation sheets")
    parser.add_argument("--sheets-dir", default=str(SHEETS_DIR))
    parser.add_argument("--out", default=str(GOLD_OUT))
    parser.add_argument("--min-alpha", type=float, default=0.667,
                        help="warn (not fail) below this agreement level")
    args = parser.parse_args()

    labels, sheet_names = read_sheets(Path(args.sheets_dir))

    units = [
        judgments
        for per_hotel in labels.values()
        for judgments in per_hotel.values()
    ]
    n_judged = sum(1 for u in units for v in u if v is not None)
    alpha = krippendorff_alpha(units)

    gold = aggregate_gold(labels)
    out = {
        "method": "majority vote over graded 0/1/2 judgments, binarized at >=1; "
                  "ties resolved as non-relevant",
        "annotators": sheet_names,
        "date": date.today().isoformat(),
        "krippendorff_alpha_interval": round(alpha, 4),
        "n_items": len(units),
        "n_judgments": n_judged,
        "relevant": {qid: sorted(ids) for qid, ids in sorted(gold.items())},
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Annotators: {len(sheet_names)}  items: {len(units)}  judgments: {n_judged}")
    print(f"Krippendorff's alpha (interval): {alpha:.3f}")
    if alpha < args.min_alpha:
        print(f"WARNING: alpha below {args.min_alpha} — labels are not reliable enough "
              "to publish. Review disagreements with annotators (adjudication) and re-run.")
    print(f"Human gold written to {args.out} "
          f"({sum(len(v) for v in gold.values())} relevant pairs across {len(gold)} queries)")


if __name__ == "__main__":
    main()
