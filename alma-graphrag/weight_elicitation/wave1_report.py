"""
Thesis-ready wave-1 tables, generated from the dump rather than typed by hand.

    python -m weight_elicitation.wave1_report
    python -m weight_elicitation.wave1_report --top 5 --out weight_elicitation/out/wave1_report.md

Tables
  1. Selection share per question: top hotels with Wilson 95% interval, the share
     expected from list position alone, and the z-score of the difference, plus
     how concentrated the choices were.
  2. The position cliff: where the chosen hotel sat, by the sort in use.
  3. Sort switches: where participants moved the sort to, per question
     (from sort_choice.py).
  4. Estimator audit summary, if `out/audit.json` exists (audit_profiles.py).

Every number is recomputed on each run, so the tables cannot drift from the data.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from weight_elicitation import MATERIAL, OUT, latest_dump
from weight_elicitation.choice_model import DIMS, choice_data_from_responses, wilson_interval
from weight_elicitation.fit_share_weights import build_panel, iter_cells
from weight_elicitation.fit_weights import (facility_scores, failed_attention, load_dump,
                                            load_material)
from weight_elicitation.sort_choice import SORTS, extract_events

SORT_LABEL = {"distance": "Distance", "travel": "Travel time", "rating": "Rating",
              "price_asc": "Price, low first", "price_desc": "Price, high first"}


def share_table(responses, material, facility, failed, top: int) -> List[str]:
    panel = build_panel(responses, material, facility, 32, "display", False, failed)
    sc = panel.counts()
    L = ["## Table 1. Selection share per question", "",
         "Share = chosen / shown. *Expected by position* applies the empirical "
         "P(choose | rank, sort mode) to the ranks each hotel was actually shown at; "
         "*z* = (chosen − expected) / SD under that baseline.", "",
         "| Q | Persona (primed) | Hotel | Share | 95% CI | Expected by position | z |",
         "|---|---|---|---:|---:|---:|---:|"]
    conc = ["", "| Q | Respondents | Hotels ever chosen | Effective number of hotels |",
            "|---|---:|---:|---:|"]
    for key, n_resp, exposed, chosen, share, _m, _r, _gi, expected, var in iter_cells(sc, False):
        t = key[0]
        meta = panel.task_meta.get(t, {})
        for k, h in enumerate(np.argsort(-share)[:top]):
            lo, hi = wilson_interval(float(chosen[h]), float(exposed[h]))
            sd = float(np.sqrt(var[h]))
            z = (chosen[h] - expected[h]) / sd if sd > 0 else float("nan")
            label = (f"{t} | {meta.get('persona', '')} ({meta.get('primary_dimension', '')})"
                     if k == 0 else " | ")
            L.append(f"| {label} | {panel.names[panel.hotels[h]]} | {100 * share[h]:.1f}% | "
                     f"{100 * lo:.1f}–{100 * hi:.1f} | {100 * expected[h] / n_resp:.1f}% | "
                     f"{z:+.1f} |")
        sh = chosen / n_resp
        hhi = float((sh ** 2).sum())
        conc.append(f"| {t} | {int(n_resp)} | {int((chosen > 0).sum())} | {1 / hhi:.1f} |")
    return L + conc


def position_table(data) -> List[str]:
    pos = data.pos[np.arange(len(data)), data.y]
    L = ["## Table 2. Where the chosen hotel sat", "",
         "| Sort in use | Choices | Position 1 | Position 2 | Position 3 | Below 3 |",
         "|---|---:|---:|---:|---:|---:|"]
    for s in SORTS + ("all",):
        sel = np.ones(len(data), bool) if s == "all" else data.sorts == s
        n = int(sel.sum())
        if not n:
            continue
        c = [np.mean(pos[sel] == r) for r in (1, 2, 3)] + [np.mean(pos[sel] > 3)]
        name = "**All**" if s == "all" else SORT_LABEL[s]
        L.append(f"| {name} | {n} | " + " | ".join(f"{100 * v:.1f}%" for v in c) + " |")
    return L


def sort_table(responses) -> List[str]:
    events = extract_events(responses)
    tasks = sorted({e["task"] for e in events}, key=lambda t: int(t[1:]) if t[1:].isdigit() else 99)
    sw = [e for e in events if e["switched"]]
    used = np.mean([e["n_sort_events"] > 0 for e in events])
    carry = 1 - len(sw) / max(sum(1 for e in events if e["previous_sort"]), 1)
    L = ["## Table 3. Sort switches by question", "",
         f"{100 * used:.1f}% of responses used the sort control; the sort carried over "
         f"from the previous question {100 * carry:.1f}% of the time, leaving "
         f"{len(sw)} switches. Destination shares of those switches:", "",
         "| Q | Primed | Switchers | " + " | ".join(SORT_LABEL[s] for s in SORTS) + " |",
         "|---|---|---:|" + "---:|" * len(SORTS)]
    for t in tasks:
        here = [e for e in sw if e["task"] == t]
        if not here:
            continue
        c = Counter(e["final_sort"] for e in here)
        primary = next((e["primary"] for e in events if e["task"] == t and e["primary"]), "")
        L.append(f"| {t} | {primary} | {len(here)} | " +
                 " | ".join(f"{100 * c[s] / len(here):.0f}%" for s in SORTS) + " |")
    return L


def audit_table(path: Path) -> List[str]:
    if not path.exists():
        return ["## Table 4. Estimator audit", "",
                "_Not generated: run `python -m weight_elicitation.audit_profiles` first._"]
    a = json.loads(path.read_text(encoding="utf-8"))
    L = ["## Table 4. Estimator audit", "",
         f"{a['config']['placebo_reps']} placebo and {a['config']['bootstrap_reps']} bootstrap "
         "replicates per estimator.", "",
         "| Position control | LR χ²(5) | p | Estimator | Identified dimensions |",
         "|---|---:|---:|---|---|"]
    for spec, block in a["specs"].items():
        lr = block["likelihood_ratio"]
        for i, (name, g) in enumerate(block["gates"].items()):
            head = f"{spec} | {lr['lr']:.1f} | {lr['p']:.3g}" if i == 0 else " | | "
            L.append(f"| {head} | {name} | {', '.join(g['identified_dimensions']) or 'none'} |")
    return L


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, default=None)
    ap.add_argument("--material", type=Path, default=MATERIAL)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--audit", type=Path, default=OUT / "audit.json")
    ap.add_argument("--out", type=Path, default=OUT / "wave1_report.md")
    args = ap.parse_args()

    dump = args.dump or latest_dump()
    _, responses, version = load_dump(dump)
    material = load_material(args.material)
    facility = facility_scores(material, "all_ranks")
    failed = failed_attention(responses)
    data = choice_data_from_responses(responses, facility, pool_size=32)

    parts = [f"# Wave 1 tables — {version}", "",
             f"Source `{dump.name}`: {len(data)} hotel choices from "
             f"{len(np.unique(data.participants))} participants (all cohorts).", ""]
    for block in (share_table(responses, material, facility, failed, args.top),
                  position_table(data), sort_table(responses), audit_table(args.audit)):
        parts += block + [""]
    parts.append("Generated by `python -m weight_elicitation.wave1_report`.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
