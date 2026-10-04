"""
Did the annotators judge, or did they rubber-stamp the model?

Model-assisted pre-labelling only produces human labels if the humans actually
exercised judgement. This module measures that from the filled sheets, using
the blind control pairs that `prelabel.py` withheld a suggestion for:

    shown agreement    how often a final label equals the model's grade on
                       pairs where the model's grade was visible
    blind agreement    the same quantity on control pairs, where the annotator
                       could not see it
    anchoring          shown minus blind, with a bootstrap interval. A large
                       positive gap means the suggestion moved the label
                       rather than matched it, and the gold standard has to be
                       described accordingly.
    override rate      share of shown pairs the annotator changed, and in
                       which direction

`verdict()` turns those numbers into one of three states, so a marginal result
cannot be quietly read as a clean one:

    human          anchoring below `tolerance`; labels stand as human labels
    assisted       anchoring measurable; labels are human-verified, and must
                   be reported that way alongside the gap
    model-driven   blind agreement is itself near-total, or the override rate
                   is near zero: the exercise measured the model, not people

Usage:
    python evaluation/annotation/prelabel_audit.py
    python evaluation/annotation/prelabel_audit.py --tolerance 0.05
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.append(str(Path(__file__).resolve().parents[2]))

SHEETS_DIR = Path(__file__).resolve().parent / "sheets"
REPORT = Path(__file__).resolve().parents[1] / "prelabel_audit.json"


def read_filled(sheets_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(sheets_dir.glob("annotator_*.csv")):
        for row in csv.DictReader(path.open(encoding="utf-8")):
            row["_annotator"] = path.stem
            rows.append(row)
    if not rows:
        raise SystemExit(f"no annotator_*.csv in {sheets_dir}")
    return rows


def _grade(value: Optional[str]) -> Optional[int]:
    text = (value or "").strip()
    return int(text) if text in {"0", "1", "2"} else None


def split_rows(rows: Sequence[Dict[str, str]]) -> Tuple[List[Tuple[int, int]],
                                                        List[Tuple[int, int]]]:
    """-> (shown pairs, blind pairs) as (human grade, model grade) tuples."""
    shown: List[Tuple[int, int]] = []
    blind: List[Tuple[int, int]] = []
    for row in rows:
        human, model = _grade(row.get("relevance")), _grade(row.get("model_prelabel"))
        if human is None or model is None:
            continue
        target = shown if (row.get("prelabel_shown") or "").strip() == "yes" else blind
        target.append((human, model))
    return shown, blind


def agreement(pairs: Sequence[Tuple[int, int]]) -> float:
    return sum(1 for h, m in pairs if h == m) / len(pairs) if pairs else 0.0


def bootstrap_gap(shown: Sequence[Tuple[int, int]], blind: Sequence[Tuple[int, int]],
                  reps: int = 5000, seed: int = 7) -> Tuple[float, float]:
    """Percentile interval for (shown agreement - blind agreement)."""
    if not shown or not blind:
        return (0.0, 0.0)
    rng = random.Random(seed)
    diffs = []
    for _ in range(reps):
        a = [shown[rng.randrange(len(shown))] for _ in shown]
        b = [blind[rng.randrange(len(blind))] for _ in blind]
        diffs.append(agreement(a) - agreement(b))
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))]
    return (round(lo, 4), round(hi, 4))


def verdict(shown_agreement: float, blind_agreement: float, override_rate: float,
            tolerance: float = 0.10) -> str:
    if override_rate < 0.02 or blind_agreement > 0.95:
        return "model-driven"
    if shown_agreement - blind_agreement > tolerance:
        return "assisted"
    return "human"


def audit(rows: Sequence[Dict[str, str]], tolerance: float = 0.10) -> Dict[str, Any]:
    shown, blind = split_rows(rows)
    shown_agreement, blind_agreement = agreement(shown), agreement(blind)
    overrides = [(h, m) for h, m in shown if h != m]
    override_rate = len(overrides) / len(shown) if shown else 0.0
    lo, hi = bootstrap_gap(shown, blind)
    labelled = sum(1 for row in rows if _grade(row.get("relevance")) is not None)
    return {
        "rows_total": len(rows),
        "rows_labelled": labelled,
        "shown_pairs": len(shown),
        "blind_control_pairs": len(blind),
        "shown_agreement": round(shown_agreement, 4),
        "blind_agreement": round(blind_agreement, 4),
        "anchoring_gap": round(shown_agreement - blind_agreement, 4),
        "anchoring_gap_ci95": [lo, hi],
        "override_rate": round(override_rate, 4),
        "overrides_up": sum(1 for h, m in overrides if h > m),
        "overrides_down": sum(1 for h, m in overrides if h < m),
        "tolerance": tolerance,
        "verdict": verdict(shown_agreement, blind_agreement, override_rate, tolerance),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sheets-dir", default=str(SHEETS_DIR))
    ap.add_argument("--tolerance", type=float, default=0.10,
                    help="anchoring gap above which labels are only 'assisted'")
    ap.add_argument("--out", default=str(REPORT))
    args = ap.parse_args()

    report = audit(read_filled(Path(args.sheets_dir)), args.tolerance)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"labelled rows        : {report['rows_labelled']} of {report['rows_total']}")
    print(f"shown / blind pairs  : {report['shown_pairs']} / {report['blind_control_pairs']}")
    print(f"agreement shown      : {report['shown_agreement']:.3f}")
    print(f"agreement blind      : {report['blind_agreement']:.3f}")
    print(f"anchoring gap        : {report['anchoring_gap']:+.3f} "
          f"CI {report['anchoring_gap_ci95']}")
    print(f"override rate        : {report['override_rate']:.3f} "
          f"({report['overrides_up']} up, {report['overrides_down']} down)")
    print(f"\nverdict: {report['verdict']}")
    if report["verdict"] == "model-driven":
        print("Report these as model labels verified by people, not as human labels.")
    elif report["verdict"] == "assisted":
        print("Report as human-verified labels and quote the anchoring gap beside them.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
