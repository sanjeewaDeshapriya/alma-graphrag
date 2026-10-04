"""
Export the collected choices as a LABELLED RANKING DATASET.

This script does not fit weights. It turns what participants did into the
standard learning-to-rank shape so that any downstream model - LambdaMART,
a linear scorer, a conditional logit, whatever you decide later - can be
trained and compared on identical data.

One "query" (qid) = one participant working one task. Its candidates are the
whole hotel pool for that task's anchor: exactly one is labelled 1 (chosen),
the rest 0. Candidates the participant filtered out are still exported, marked
`shown=0`, because "not shown" and "shown and rejected" are different facts and
collapsing them would bias anything trained on the result.

Outputs (into weight_elicitation/out/ by default)
    dataset.csv        one row per candidate - the human-readable master file
    dataset.jsonl      same rows, one JSON object per line
    dataset.ltr.txt    RankLib / LibSVM / XGBoost-rank format:
                           <label> qid:<n> 1:<f1> ... 6:<f6> # <hotel_id>
    groups.txt         group sizes, in qid order (XGBoost `set_group`)
    dataset_card.md    row/participant counts, feature list, known caveats

Usage
    python -m weight_elicitation.export_dataset
    python -m weight_elicitation.export_dataset --database-url "$DATABASE_URL"
    python -m weight_elicitation.export_dataset --exclude-failed-attention
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from weight_elicitation import LOCAL_CAPTURE, OUT
DIMENSIONS = ["spatial", "accessibility", "facility", "economic", "disruption"]

# `position` is a feature, not a nuisance to be dropped. The list is
# distance-sorted, so rank and proximity are entangled in the raw click; keeping
# it as an explicit column lets a downstream model control for it (or lets you
# estimate a propensity and reweight). Discarding it would bake the bias in
# silently, which is the one outcome that cannot be repaired later.
FEATURES = DIMENSIONS + ["position"]

# Stage-2 features. Unlike the hotel components these are all OBSERVED, and
# `price_lkr` is in rupees — which is the point of exporting them. A coefficient
# ratio against price converts any other coefficient into a willingness-to-pay,
# so the stage-1 weights can be reported on a monetary scale instead of only up
# to an arbitrary scale factor.
ROOM_FEATURES = [
    "price_delta_lkr",   # vs the cheapest offer in the same hotel
    "breakfast",
    "refundable",
    "size_sqm",
    "max_occupancy",
    "room_position",
]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_local(data_dir: Path) -> tuple[list[dict], list[dict]]:
    participants = read_jsonl(data_dir / "participants.jsonl")
    demo = {d["id"]: d.get("demographics") or {}
            for d in read_jsonl(data_dir / "demographics.jsonl")}
    for p in participants:
        p["demographics"] = demo.get(p["id"], p.get("demographics") or {})
    return participants, read_jsonl(data_dir / "responses.jsonl")


def load_database(url: str) -> tuple[list[dict], list[dict]]:
    import psycopg2
    import psycopg2.extras
    conn = psycopg2.connect(url)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id, consent, demographics, material_version,"
                        " created_at FROM participants ORDER BY created_at")
            participants = [dict(r) for r in cur.fetchall()]
            cur.execute(
                "SELECT participant_id, task_id, scenario_id, anchor_id,"
                " primary_dimension, secondary_dimension, repeat_of,"
                " chosen_hotel_id, options, chosen_room_id, room, room_options,"
                " is_attention_check, attention_pass,"
                " timing, interactions, submitted_at"
                " FROM responses ORDER BY submitted_at")
            responses = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
    return participants, responses


def get(rec: Dict[str, Any], *names: str, default: Any = None) -> Any:
    """Rows arrive either snake_case (Postgres) or camelCase (JSONL fallback)."""
    for n in names:
        if rec.get(n) is not None:
            return rec[n]
    return default


# --------------------------------------------------------------------------- #
# Building rows
# --------------------------------------------------------------------------- #
def build_rows(participants: List[Dict[str, Any]],
               responses: List[Dict[str, Any]],
               exclude_failed_attention: bool) -> tuple[list[dict], dict]:
    demo_by_pid = {p["id"]: (p.get("demographics") or {}) for p in participants}

    # Participants who failed the attention check, so they can be dropped as a
    # set rather than task by task - a participant who was not paying attention
    # was not paying attention for the whole session.
    failed = {
        get(r, "participant_id", "participantId")
        for r in responses
        if get(r, "is_attention_check", "isAttentionCheck")
        and get(r, "attention_pass", "attentionPass") is False
    }

    rows: List[Dict[str, Any]] = []
    qid = 0
    skipped_attention_tasks = 0
    skipped_failed_participants = 0

    for r in responses:
        pid = get(r, "participant_id", "participantId")
        if get(r, "is_attention_check", "isAttentionCheck"):
            skipped_attention_tasks += 1
            continue                       # never train on the attention check
        if exclude_failed_attention and pid in failed:
            skipped_failed_participants += 1
            continue

        options = r.get("options") or []
        if not options:
            continue
        qid += 1

        timing = r.get("timing") or {}
        dwell = timing.get("dwell_ms") or {}
        demo = demo_by_pid.get(pid, {}) or {}
        n_shown = sum(1 for o in options if o.get("displayed_position") is not None)

        for o in options:
            comps = o.get("components") or {}
            pos = o.get("displayed_position")
            rows.append({
                "qid": qid,
                "participant_id": pid,
                "task_id": get(r, "task_id", "taskId"),
                "anchor_id": get(r, "anchor_id", "anchorId"),
                "primary_dimension": get(r, "primary_dimension", "primaryDimension"),
                "secondary_dimension": get(r, "secondary_dimension", "secondaryDimension"),
                "repeat_of": get(r, "repeat_of", "repeatOf"),
                "hotel_id": o.get("hotel_id"),
                "label": 1 if o.get("chosen") else 0,
                # 1-based rank as displayed; 0 when the hotel was filtered out.
                "position": pos if pos is not None else 0,
                "shown": 1 if pos is not None else 0,
                **{d: comps.get(d) for d in DIMENSIONS},
                "dwell_ms": dwell.get(o.get("hotel_id"), 0),
                "n_shown": n_shown,
                "n_total": len(options),
                "decision_ms": timing.get("decision_ms"),
                "time_to_first_interaction_ms": timing.get("time_to_first_interaction_ms"),
                "revisions": timing.get("revisions"),
                "final_sort": timing.get("final_sort"),
                "final_query": timing.get("final_query"),
                "final_max_price": timing.get("final_max_price"),
                "age_band": demo.get("age_band"),
                "occupation": demo.get("occupation"),
                "salary_band": demo.get("salary_band"),
                "home_area": demo.get("home_area"),
            })

    stats = {
        "n_participants": len({r["participant_id"] for r in rows}),
        "n_queries": qid,
        "n_rows": len(rows),
        "n_positive": sum(r["label"] for r in rows),
        "attention_tasks_skipped": skipped_attention_tasks,
        "failed_attention_participants": len(failed),
        "rows_dropped_failed_attention": skipped_failed_participants,
    }
    return rows, stats


def build_room_rows(participants: List[Dict[str, Any]],
                    responses: List[Dict[str, Any]],
                    exclude_failed_attention: bool) -> tuple[list[dict], dict]:
    """Stage 2: one row per room offer inside each chosen hotel.

    A separate ranking dataset with its own qid space. The groups are small
    (2-5 offers) and the choice set is whatever the chosen hotel actually
    offered, so groups are not comparable in size across rows — which is normal
    for a nested design and is why this is exported alongside, not merged into,
    the hotel dataset.
    """
    demo_by_pid = {p["id"]: (p.get("demographics") or {}) for p in participants}
    failed = {
        get(r, "participant_id", "participantId")
        for r in responses
        if get(r, "is_attention_check", "isAttentionCheck")
        and get(r, "attention_pass", "attentionPass") is False
    }

    rows: List[Dict[str, Any]] = []
    qid = 0
    no_room = 0

    for r in responses:
        pid = get(r, "participant_id", "participantId")
        if get(r, "is_attention_check", "isAttentionCheck"):
            continue
        if exclude_failed_attention and pid in failed:
            continue

        offers = get(r, "room_options", "roomOptions") or []
        if not offers or not any(o.get("chosen") for o in offers):
            # Pre-two-stage responses, or a client that posted without a room.
            no_room += 1
            continue
        qid += 1

        timing = r.get("timing") or {}
        demo = demo_by_pid.get(pid, {}) or {}
        cheapest = min(o.get("price_lkr") or 0 for o in offers)
        hotel_id = get(r, "chosen_hotel_id", "chosenHotelId")

        for o in offers:
            price = o.get("price_lkr") or 0
            board = (o.get("board_type") or "").upper()
            rows.append({
                "qid": qid,
                "participant_id": pid,
                "task_id": get(r, "task_id", "taskId"),
                "anchor_id": get(r, "anchor_id", "anchorId"),
                "hotel_id": hotel_id,
                "room_id": o.get("room_id"),
                "room_name": o.get("name"),
                "label": 1 if o.get("chosen") else 0,
                "price_lkr": price,
                "price_delta_lkr": price - cheapest,
                "board_name": o.get("board_name"),
                # RO is "Room Only"; anything else includes at least breakfast.
                "breakfast": 0 if board == "RO" else 1,
                "refundable": 1 if o.get("refundable") else 0,
                "size_sqm": o.get("size_sqm"),
                "max_occupancy": o.get("max_occupancy"),
                "beds": o.get("beds"),
                "room_position": o.get("displayed_position") or 0,
                "n_offers": len(offers),
                "panel_ms": (timing.get("panel_ms") or {}).get(hotel_id),
                "room_revisions": timing.get("room_revisions"),
                "decision_ms": timing.get("decision_ms"),
                "age_band": demo.get("age_band"),
                "occupation": demo.get("occupation"),
                "salary_band": demo.get("salary_band"),
                "home_area": demo.get("home_area"),
            })

    stats = {
        "n_participants": len({r["participant_id"] for r in rows}),
        "n_queries": qid,
        "n_rows": len(rows),
        "n_positive": sum(r["label"] for r in rows),
        "responses_without_room": no_room,
    }
    return rows, stats


def write_room_outputs(rows: List[Dict[str, Any]], stats: Dict[str, Any],
                       out_dir: Path) -> None:
    if not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "rooms.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with (out_dir / "rooms.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    ordered = sorted(rows, key=lambda r: (r["qid"], r["room_position"]))
    with (out_dir / "rooms.ltr.txt").open("w", encoding="utf-8") as fh:
        for r in ordered:
            feats = " ".join(
                f"{i}:{(r.get(f) if r.get(f) is not None else 0):g}"
                for i, f in enumerate(ROOM_FEATURES, start=1)
            )
            fh.write(f"{r['label']} qid:{r['qid']} {feats} # {r['room_id']}\n")

    sizes: Dict[int, int] = {}
    for r in ordered:
        sizes[r["qid"]] = sizes.get(r["qid"], 0) + 1
    with (out_dir / "rooms_groups.txt").open("w", encoding="utf-8") as fh:
        for q in sorted(sizes):
            fh.write(f"{sizes[q]}\n")


def pearson(xs: List[float], ys: List[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < 2:
        return float("nan")
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    dy = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    return num / (dx * dy) if dx and dy else float("nan")


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def write_outputs(rows: List[Dict[str, Any]], stats: Dict[str, Any],
                  out_dir: Path, shown_only: bool,
                  room_stats: Optional[Dict[str, Any]] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    export = [r for r in rows if r["shown"]] if shown_only else rows

    # --- CSV ---------------------------------------------------------------
    csv_path = out_dir / "dataset.csv"
    if export:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(export[0].keys()))
            w.writeheader()
            w.writerows(export)

    # --- JSONL -------------------------------------------------------------
    with (out_dir / "dataset.jsonl").open("w", encoding="utf-8") as fh:
        for r in export:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- RankLib / LibSVM --------------------------------------------------
    # Rows must be grouped by qid and each group contiguous.
    export_sorted = sorted(export, key=lambda r: (r["qid"], r["position"]))
    with (out_dir / "dataset.ltr.txt").open("w", encoding="utf-8") as fh:
        for r in export_sorted:
            feats = " ".join(
                f"{i}:{(r.get(f) if r.get(f) is not None else 0):g}"
                for i, f in enumerate(FEATURES, start=1)
            )
            fh.write(f"{r['label']} qid:{r['qid']} {feats} # {r['hotel_id']}\n")

    # --- group sizes (XGBoost ranking) -------------------------------------
    sizes: Dict[int, int] = {}
    for r in export_sorted:
        sizes[r["qid"]] = sizes.get(r["qid"], 0) + 1
    with (out_dir / "groups.txt").open("w", encoding="utf-8") as fh:
        for q in sorted(sizes):
            fh.write(f"{sizes[q]}\n")

    # --- dataset card ------------------------------------------------------
    if room_stats and room_stats["n_rows"]:
        room_section = (
            f"| | |\n|---|---|\n"
            f"| Queries (participant x task) | {room_stats['n_queries']} |\n"
            f"| Rows (offers) | {room_stats['n_rows']} |\n"
            f"| Positive labels | {room_stats['n_positive']} |\n"
            f"| Responses with no room recorded | {room_stats['responses_without_room']} |"
        )
    else:
        room_section = (
            "No room-level responses in this export (all responses predate the "
            "two-stage design, or were posted without a room)."
        )

    corr_lines = []
    for i, a in enumerate(DIMENSIONS):
        for b in DIMENSIONS[i + 1:]:
            r = pearson([x[a] for x in export], [x[b] for x in export])
            if not math.isnan(r) and abs(r) >= 0.5:
                corr_lines.append(f"  - `{a}` vs `{b}`: r = {r:.2f}")

    card = f"""# ALMA hotel-choice dataset

Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}

## Shape

| | |
|---|---|
| Participants | {stats['n_participants']} |
| Queries (participant x task) | {stats['n_queries']} |
| Rows (candidates) | {len(export)} |
| Positive labels | {stats['n_positive']} |
| Rows per query | up to {stats['n_rows'] // max(stats['n_queries'], 1)} |

Attention-check tasks excluded from the dataset: {stats['attention_tasks_skipped']}.
Participants who failed the attention check: {stats['failed_attention_participants']}
(rows dropped: {stats['rows_dropped_failed_attention']}).

## Files

| File | Format |
|---|---|
| `dataset.csv` | one row per candidate, all columns |
| `dataset.jsonl` | same rows as JSON |
| `dataset.ltr.txt` | `<label> qid:<n> 1:.. 6:.. # <hotel_id>` (RankLib / XGBoost) |
| `groups.txt` | group sizes in qid order |
| `rooms.csv` / `rooms.jsonl` | stage 2: one row per room offer |
| `rooms.ltr.txt` / `rooms_groups.txt` | stage 2 in ranking format |

## Features (in `dataset.ltr.txt` order)

{chr(10).join(f'{i}. `{f}`' for i, f in enumerate(FEATURES, start=1))}

`position` is the 1-based rank the hotel was displayed at, or 0 if the
participant had filtered it out. `shown` in the CSV distinguishes the two.

## Stage 2 — room choice within the chosen hotel

{room_section}

Features (in `rooms.ltr.txt` order):

{chr(10).join(f'{i}. `{f}`' for i, f in enumerate(ROOM_FEATURES, start=1))}

Its own qid space, unrelated to the hotel `qid`. Groups are small (2-5) and
vary in size, because the choice set is whatever the chosen hotel actually
offered. `price_delta_lkr` is measured against the cheapest offer in the SAME
hotel, so it is a within-hotel contrast and carries no hotel-level price
information — that lives in the stage-1 `economic` component.

The reason to fit this: `price_delta_lkr` is in rupees, so the ratio of any
other coefficient to the price coefficient is a willingness-to-pay. That is
what lets the five stage-1 weights be reported on a monetary scale rather than
only up to an arbitrary scale factor.

## Caveats a user of this data must know

- **Position is confounded with proximity.** Each task's list opens sorted by
  distance to the task anchor, so the nearest hotels are also the ones at the
  top. `position` is exported so this can be modelled or corrected; a model
  trained without it will attribute the effect of rank to the spatial feature.
- **`spatial` and `accessibility` are strongly correlated** (~0.9 across the
  pool: in Colombo, being close and being quick to reach mostly coincide).
  Treat them as one location signal unless you have a design that separates
  them. Observed correlations at or above 0.5 in this export:
{chr(10).join(corr_lines) if corr_lines else '  - none'}
- **Choice sets are participant-dependent.** Search and filters were available,
  so the candidates actually on screen differ between participants for the same
  task. `shown`, `n_shown`, `final_sort`, `final_query` and `final_max_price`
  describe what each participant had narrowed the list to.
- **Repeated task.** `t10` deliberately repeats `t2`; `repeat_of` marks it. Use
  the pair to measure test-retest consistency, and drop one of them before
  training so the same decision is not counted twice.
"""
    (out_dir / "dataset_card.md").write_text(card, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=LOCAL_CAPTURE)
    ap.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--exclude-failed-attention", action="store_true",
                    help="drop every response from participants who failed the check")
    ap.add_argument("--shown-only", action="store_true",
                    help="export only candidates that were actually on screen")
    args = ap.parse_args()

    if args.database_url:
        participants, responses = load_database(args.database_url)
        source = "postgres"
    else:
        participants, responses = load_local(args.data_dir)
        source = str(args.data_dir)

    if not responses:
        raise SystemExit(f"No responses found in {source}.")

    rows, stats = build_rows(participants, responses, args.exclude_failed_attention)
    if not rows:
        raise SystemExit("No usable rows after filtering.")
    room_rows, room_stats = build_room_rows(
        participants, responses, args.exclude_failed_attention
    )
    write_outputs(rows, stats, args.out, args.shown_only, room_stats)
    write_room_outputs(room_rows, room_stats, args.out)

    print(f"source            : {source}")
    print(f"participants      : {stats['n_participants']}")
    print(f"queries (groups)  : {stats['n_queries']}")
    print(f"rows (candidates) : {stats['n_rows']}")
    print(f"positive labels   : {stats['n_positive']}")
    print(f"\nstage 2 (rooms)")
    print(f"  queries         : {room_stats['n_queries']}")
    print(f"  rows (offers)   : {room_stats['n_rows']}")
    print(f"  no room recorded: {room_stats['responses_without_room']}")

    written = ["dataset.csv", "dataset.jsonl", "dataset.ltr.txt", "groups.txt",
               "dataset_card.md"]
    if room_rows:
        written += ["rooms.csv", "rooms.jsonl", "rooms.ltr.txt", "rooms_groups.txt"]
    print(f"\nwrote -> {args.out}")
    for f in written:
        print(f"  {f}")


if __name__ == "__main__":
    main()
