"""
Model-assisted pre-labelling for the human relevance gold standard.

WHY
---
The sheets hold 2,054 query-hotel pairs per annotator. Judging each from
scratch is 3-4 hours of work per person, which is the single reason the human
gold standard has not been collected. A model can propose a grade for every
pair; a person then confirms or overrides it. Verification is far faster than
origination, and the resulting labels are still human labels -- provided the
design below is respected.

THE HAZARD, AND THE CONTROLS
----------------------------
Pre-labels anchor annotators: a person shown a suggested grade agrees with it
more often than they would have chosen it unprompted, so naive pre-labelling
quietly replaces human judgement with model judgement wearing a human name.
Three controls keep that measurable rather than hidden:

1. HOLD-OUT CONTROL. A seeded random share of pairs (default 20%) is served
   with NO pre-label. Agreement with the model on pre-labelled pairs can then
   be compared against agreement on control pairs; a large gap is anchoring,
   not accuracy, and the gap is reported with the gold standard.
2. INDEPENDENCE. The pre-labeller sees the query and the hotel's attributes
   only. It is never shown which system retrieved the hotel, at what rank, or
   what the rule gold says. It is not the system under evaluation, so its
   judgements cannot make that system look better by construction.
3. OVERRIDE TRACKING. The sheet keeps the model's grade in its own column.
   aggregate.py can therefore report how often annotators overrode it, in
   which direction, and whether inter-annotator agreement was computed on
   labels that mostly came from the model.

Anything the model proposes is a SUGGESTION. The relevance column an annotator
fills is the label; the pre-label column is provenance.

USAGE
    python evaluation/annotation/prelabel.py --limit-queries 2      # pilot
    python evaluation/annotation/prelabel.py                        # full run
    python evaluation/annotation/prelabel.py --resume               # continue

Writes evaluation/annotation/sheets/prelabels.json (a cache keyed by
query/hotel, so an interrupted run resumes without re-spending calls) and
rewrites the annotator sheets with `model_prelabel` and `prelabel_shown`
columns.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.append(str(Path(__file__).resolve().parents[2]))

from openai import OpenAI  # noqa: E402

from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL  # noqa: E402
from src.llm_utils import chat_completion_with_retry  # noqa: E402

logger = logging.getLogger("alma.annotation.prelabel")

SHEETS_DIR = Path(__file__).resolve().parent / "sheets"
CACHE = SHEETS_DIR / "prelabels.json"

SYSTEM_PROMPT = (
    "You grade hotel search results for an information retrieval study in "
    "Colombo, Sri Lanka. For each numbered item you are given a traveller's "
    "request and one hotel's attributes. Grade how well that hotel answers "
    "that request:\n"
    "2 = fully relevant: satisfies the main need with no disqualifying "
    "attribute.\n"
    "1 = partially relevant: reasonable but flawed - slightly over budget, "
    "slightly slower to reach, or satisfies a secondary aspect while missing "
    "part of the main one.\n"
    "0 = not relevant: a traveller asking this would consider it a wrong "
    "answer.\n\n"
    "Rules: judge as a traveller would read the request, not as a database "
    "filter; 'cheap' without a number means cheap relative to the other "
    "hotels in this city. A missing attribute is not automatically "
    "irrelevant - judge from the rest and use 1 when genuinely uncertain. "
    "For a multi-constraint request all main constraints must hold for a 2; "
    "one clearly failed constraint caps the grade at 1; two failed gives 0. "
    "Travel time is drive time under traffic: under about 5 minutes is quick "
    "for Colombo, over about 10 minutes is not.\n\n"
    "Reply with one line per item in the form 'id=grade', nothing else."
)


def hotel_line(row: Dict[str, str], index: int) -> str:
    bits = [f"[{index}] request: {row['question']}", f"hotel: {row['hotel_name']}"]
    if row.get("price_lkr"):
        bits.append(f"price {row['price_lkr']} LKR")
    else:
        bits.append("price not listed")
    if row.get("rating"):
        bits.append(f"rating {row['rating']}/5")
    if row.get("star"):
        bits.append(f"{row['star']}-star")
    if row.get("travel_time_min"):
        bits.append(f"{row['travel_time_min']} min travel time")
    amenities = (row.get("amenities") or "").strip()
    bits.append(f"amenities: {amenities}" if amenities else "amenities: none listed")
    return "; ".join(bits)


def parse_reply(text: str, expected: Sequence[int]) -> Dict[int, int]:
    """Read 'id=grade' lines, ignoring anything the model adds around them."""
    out: Dict[int, int] = {}
    for raw in (text or "").splitlines():
        line = raw.strip().strip("-•*[] ")
        if "=" not in line:
            continue
        left, _, right = line.partition("=")
        try:
            idx, grade = int(left.strip().strip("[]")), int(right.strip()[:1])
        except ValueError:
            continue
        if idx in expected and grade in (0, 1, 2):
            out[idx] = grade
    return out


def _client() -> OpenAI:
    """The configured chat client, or a clear error rather than a None deref."""
    if not LLM_API_KEY:
        raise SystemExit(
            "No LLM key configured. Set GEMINI_API_KEY or OPENAI_API_KEY in .env "
            "before pre-labelling, or annotate without suggestions."
        )
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def grade_batch(rows: List[Dict[str, str]], batch_start: int) -> Dict[int, int]:
    """One model call for a batch of pairs. Returns {row index: grade}."""
    listing = "\n".join(hotel_line(row, batch_start + i) for i, row in enumerate(rows))
    expected = range(batch_start, batch_start + len(rows))
    client = _client()
    response = chat_completion_with_retry(
        client=client, model=LLM_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": listing}],
        temperature=0.0,
    )
    return parse_reply(response.choices[0].message.content, list(expected))


def load_cache() -> Dict[str, int]:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8")).get("grades", {})
    return {}


def save_cache(grades: Dict[str, int], meta: Dict[str, Any]) -> None:
    CACHE.write_text(json.dumps({"meta": meta, "grades": grades}, indent=2),
                     encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sheets-dir", default=str(SHEETS_DIR))
    ap.add_argument("--batch", type=int, default=25,
                    help="pairs per model call")
    ap.add_argument("--control-share", type=float, default=0.20,
                    help="share of pairs served WITHOUT a pre-label")
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--limit-queries", type=int, default=0,
                    help="pilot on the first N queries only")
    ap.add_argument("--resume", action="store_true",
                    help="keep grades already cached and only fill the gaps")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be called, spend nothing")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    sheets = Path(args.sheets_dir)
    first = sheets / "annotator_1.csv"
    if not first.exists():
        raise SystemExit(f"no sheets in {sheets}; run make_sheets.py first")
    rows = list(csv.DictReader(first.open(encoding="utf-8")))

    # One canonical row per (query, hotel); the sheets are shuffled copies.
    pairs: Dict[str, Dict[str, str]] = {}
    for row in rows:
        pairs.setdefault(f"{row['query_id']}::{row['hotel_id']}", row)
    keys = sorted(pairs)
    if args.limit_queries:
        keep = sorted({k.split("::")[0] for k in keys})[:args.limit_queries]
        keys = [k for k in keys if k.split("::")[0] in keep]

    grades = load_cache() if args.resume else {}
    todo = [k for k in keys if k not in grades]
    calls = (len(todo) + args.batch - 1) // args.batch
    print(f"{len(keys)} pairs in scope, {len(grades)} cached, {len(todo)} to grade "
          f"in {calls} model calls of up to {args.batch}")
    if args.dry_run:
        return

    for start in range(0, len(todo), args.batch):
        chunk = todo[start:start + args.batch]
        try:
            graded = grade_batch([pairs[k] for k in chunk], start)
        except Exception as exc:                       # quota, network, parse
            print(f"stopped after {len(grades)} grades: {exc}")
            break
        for offset, key in enumerate(chunk):
            if start + offset in graded:
                grades[key] = graded[start + offset]
        save_cache(grades, {"batch": args.batch, "pairs_in_scope": len(keys)})
        print(f"  graded {len(grades)}/{len(keys)}")

    # Which pairs are served blind, so anchoring can be measured later.
    rng = random.Random(args.seed)
    control = {k for k in keys if rng.random() < args.control_share}

    written = 0
    for path in sorted(sheets.glob("annotator_*.csv")):
        sheet = list(csv.DictReader(path.open(encoding="utf-8")))
        for row in sheet:
            key = f"{row['query_id']}::{row['hotel_id']}"
            grade = grades.get(key)
            shown = key not in control and grade is not None
            row["model_prelabel"] = "" if grade is None else str(grade)
            row["prelabel_shown"] = "yes" if shown else "no"
            # The suggestion is pre-filled only where it is shown; the control
            # pairs stay blank so the annotator judges them unprompted.
            if shown and not (row.get("relevance") or "").strip():
                row["relevance"] = str(grade)
        fields = [f for f in sheet[0].keys()]
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sheet)
        written += 1

    shown_n = sum(1 for k in keys if k not in control and k in grades)
    print(f"\nrewrote {written} sheets")
    print(f"  pre-labelled and shown : {shown_n}")
    print(f"  blind control pairs    : {len(control & set(keys))}")
    print(f"  ungraded (no suggestion): {len(keys) - len(grades)}")
    print("\nAnnotators confirm or override every row. The pre-label is a "
          "suggestion, not a label.")


if __name__ == "__main__":
    main()
