"""
Sort choice as revealed search strategy — the preference signal wave 1 does carry.

    python -m weight_elicitation.sort_choice
    python -m weight_elicitation.sort_choice --permutations 5000

Wave-1 hotel choices are explained by list position (see audit_profiles.py), but
WHICH list a participant looked at was their own choice: the default sort was
distance, 87% of responses used the sort control, and the median first sort
came 7.5 s into the task. Sorts carry over between questions, so the informative
event is a SWITCH: a participant changing the sort they arrived with. Where a
switch lands is a choice of one attribute to prioritise.

What this estimates
-------------------
* Destination shares of switches per question, with Wilson intervals. A
  multinomial logit of destination on question fixed effects is saturated, so
  its MLE is exactly these shares; they are reported directly.
* Whether destination depends on the question: a chi-square statistic whose null
  distribution comes from permuting question labels WITHIN each participant, so
  a person who always sorts by travel time cannot manufacture an association.
* A primed-sort lift for each question whose primed dimension has a matching
  sort (spatial -> distance, accessibility -> travel, economic -> price_asc,
  facility -> rating): P(switch to that sort | this question) divided by the
  same probability on the other questions, participant-clustered 95% CI.

What it cannot estimate: compensatory weights. A sort ranks by ONE attribute;
it carries no trade-off rate. `disruption` had no sort at all, so it could not
be expressed. Question order was fixed, so framing and fatigue are confounded.

Outputs `out/sort_choice.json` and `out/sort_switches.csv`.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from weight_elicitation import OUT, latest_dump
from weight_elicitation.choice_model import wilson_interval
from weight_elicitation.fit_weights import load_dump

SORTS = ("distance", "travel", "rating", "price_asc", "price_desc")
PRIMED_SORT = {"spatial": "distance", "accessibility": "travel",
               "economic": "price_asc", "facility": "rating"}


def extract_events(responses: List[dict]) -> List[dict]:
    """One row per non-attention response, in each participant's task order."""
    rows = []
    for r in responses:
        if r.get("isAttentionCheck"):
            continue
        t = r.get("timing") or {}
        final = t.get("final_sort")
        if final is None:
            continue
        sort_events = [x for x in (r.get("interactions") or [])
                       if isinstance(x, dict) and x.get("kind") == "sort"]
        rows.append({
            "participant": r.get("participantId"),
            "task": r.get("taskId"),
            "task_index": t.get("task_index"),
            "primary": r.get("primaryDimension"),
            "final_sort": final,
            "n_sort_events": len(sort_events),
            "first_sort_ms": sort_events[0].get("at_ms") if sort_events else None,
        })
    by = defaultdict(list)
    for row in rows:
        by[row["participant"]].append(row)
    out = []
    for items in by.values():
        items.sort(key=lambda x: (x["task_index"] is None, x["task_index"]))
        prev = None
        for row in items:
            row["previous_sort"] = prev
            row["switched"] = prev is not None and row["final_sort"] != prev
            prev = row["final_sort"]
            out.append(row)
    return out


def _chi2(table: np.ndarray) -> float:
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    if table.size == 0:
        return 0.0
    exp = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / table.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nansum((table - exp) ** 2 / np.where(exp > 0, exp, np.nan)))


def association_test(switches: List[dict], tasks: List[str], n_perm: int, seed: int) -> dict:
    """Question x destination chi-square, null by within-participant permutation."""
    t_idx = {t: i for i, t in enumerate(tasks)}
    s_idx = {s: i for i, s in enumerate(SORTS)}

    def table(labels):
        m = np.zeros((len(tasks), len(SORTS)))
        for sw, t in zip(switches, labels):
            m[t_idx[t], s_idx[sw["final_sort"]]] += 1
        return m

    observed_labels = [sw["task"] for sw in switches]
    obs = _chi2(table(observed_labels))
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for i, sw in enumerate(switches):
        groups[sw["participant"]].append(i)
    exceed = 0
    for _ in range(n_perm):
        labels = list(observed_labels)
        for idx in groups.values():
            if len(idx) > 1:
                perm = rng.permutation(idx)
                for a, b in zip(idx, perm):
                    labels[a] = observed_labels[b]
        exceed += _chi2(table(labels)) >= obs - 1e-9
    return {"chi2": obs, "permutations": n_perm,
            "p_within_participant_permutation": (1 + exceed) / (1 + n_perm),
            "note": "participants with a single switch cannot be permuted, so the "
                    "test is conservative"}


def primed_lift(switches: List[dict], events: List[dict], n_boot: int, seed: int) -> Dict[str, dict]:
    """P(switch to the primed sort | question) / P(same | other questions)."""
    tasks = sorted({e["task"] for e in events if e["primary"] in PRIMED_SORT},
                   key=lambda t: int(t[1:]) if t[1:].isdigit() else 99)
    people = sorted({e["participant"] for e in events})
    by_person = defaultdict(list)
    for e in events:
        by_person[e["participant"]].append(e)
    rng = np.random.default_rng(seed)
    out = {}
    for t in tasks:
        primary = next((e["primary"] for e in events if e["task"] == t and e["primary"]), None)
        target = PRIMED_SORT[primary]

        def rates(sample):
            here = [e for p in sample for e in by_person[p] if e["task"] == t and e["previous_sort"]]
            other = [e for p in sample for e in by_person[p] if e["task"] != t and e["previous_sort"]]
            a = sum(e["switched"] and e["final_sort"] == target for e in here) / max(len(here), 1)
            b = sum(e["switched"] and e["final_sort"] == target for e in other) / max(len(other), 1)
            return a, b

        if not any(e["task"] == t and e["previous_sort"] for e in events):
            continue                      # the first question has no sort to switch from
        a, b = rates(people)
        draws = []
        for _ in range(n_boot):
            sample = rng.choice(people, size=len(people), replace=True)
            x, y = rates(sample)
            if y > 0:
                draws.append(x / y)
        lo, hi = (np.percentile(draws, [2.5, 97.5]) if draws else (np.nan, np.nan))
        out[t] = {"primary": primary, "primed_sort": target,
                  "rate_this_question": a, "rate_other_questions": b,
                  "lift": a / b if b > 0 else None, "lift_ci95": [float(lo), float(hi)]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, default=None)
    ap.add_argument("--permutations", type=int, default=2000)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--out", type=Path, default=OUT / "sort_choice.json")
    ap.add_argument("--csv", type=Path, default=OUT / "sort_switches.csv")
    args = ap.parse_args()

    dump = args.dump or latest_dump()
    _, responses, version = load_dump(dump)
    events = extract_events(responses)
    if not events:
        raise SystemExit("no responses carry a final_sort (a design without sort controls?)")
    tasks = sorted({e["task"] for e in events}, key=lambda t: int(t[1:]) if t[1:].isdigit() else 99)

    untouched = Counter(e["final_sort"] for e in events if e["n_sort_events"] == 0)
    default_sort = untouched.most_common(1)[0][0] if untouched else None
    deliberate = [e for e in events if e["n_sort_events"] > 0]
    first_ms = [e["first_sort_ms"] for e in deliberate if e["first_sort_ms"] is not None]
    with_prev = [e for e in events if e["previous_sort"] is not None]
    switches = [e for e in with_prev if e["switched"]]

    per_task = {}
    for t in tasks:
        sw = [e for e in switches if e["task"] == t]
        n = len(sw)
        c = Counter(e["final_sort"] for e in sw)
        per_task[t] = {
            "primary": next((e["primary"] for e in events if e["task"] == t and e["primary"]), None),
            "responses": sum(1 for e in events if e["task"] == t),
            "switchers": n,
            "destinations": {s: {"n": c[s], "share": c[s] / n if n else 0.0,
                                 "ci95": list(wilson_interval(c[s], n))} for s in SORTS},
        }
    first_task = tasks[0]
    first_c = Counter(e["final_sort"] for e in events if e["task"] == first_task)
    pooled_c = Counter(e["final_sort"] for e in switches)

    result = {
        "source_dump": dump.name, "material_version": version,
        "responses": len(events),
        "participants": len({e["participant"] for e in events}),
        "inferred_default_sort": default_sort,
        "used_sort_control_share": len(deliberate) / len(events),
        "first_sort_ms_quartiles": (np.percentile(first_ms, [25, 50, 75]).tolist()
                                    if first_ms else None),
        "first_sort_under_300ms_share": (float(np.mean(np.array(first_ms) < 300))
                                         if first_ms else None),
        "carry_over_share": 1 - len(switches) / max(len(with_prev), 1),
        "switches": len(switches),
        "first_task": {"task": first_task,
                       "sorts": {s: first_c[s] / sum(first_c.values()) for s in SORTS}},
        "pooled_switch_destinations": {s: pooled_c[s] / max(len(switches), 1) for s in SORTS},
        "per_task": per_task,
        "association": association_test(switches, tasks, args.permutations, args.seed),
        "primed_lift": primed_lift(switches, events, args.bootstrap, args.seed),
        "limits": ["a sort is one attribute: no trade-off rate is identified",
                   "no sort existed for disruption",
                   "question order was fixed: framing and fatigue are confounded"],
    }

    print(f"{result['responses']} responses / {result['participants']} participants")
    print(f"default sort (responses that never touched the control): {default_sort}")
    print(f"used the sort control: {result['used_sort_control_share']:.1%}; median first sort "
          f"at {result['first_sort_ms_quartiles'][1] / 1000:.1f}s" if first_ms else "")
    print(f"sort carried over from the previous question: {result['carry_over_share']:.1%}; "
          f"{len(switches)} switches")
    print(f"\n{'task':5s} {'primed':14s} {'n':>4s}  " + "  ".join(f"{s:>10s}" for s in SORTS))
    for t, r in per_task.items():
        if t == first_task:
            print(f"{t:5s} {str(r['primary']):14s}  first question - sorts chosen: " +
                  ", ".join(f"{s} {100 * v:.0f}%" for s, v in result["first_task"]["sorts"].items()))
            continue
        print(f"{t:5s} {str(r['primary']):14s} {r['switchers']:4d}  " +
              "  ".join(f"{100 * r['destinations'][s]['share']:9.0f}%" for s in SORTS))
    a = result["association"]
    print(f"\nquestion x destination: chi2 = {a['chi2']:.1f}, within-participant permutation "
          f"p = {a['p_within_participant_permutation']:.4f}")
    print("\nprimed-sort lift (switch to the primed sort here vs on other questions):")
    for t, r in result["primed_lift"].items():
        lift = "n/a" if r["lift"] is None else f"{r['lift']:.2f}"
        print(f"  {t:4s} {r['primary']:13s} -> {r['primed_sort']:9s} {100 * r['rate_this_question']:5.1f}% "
              f"vs {100 * r['rate_other_questions']:4.1f}%  lift {lift} "
              f"[{r['lift_ci95'][0]:.2f}, {r['lift_ci95'][1]:.2f}]")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with args.csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(events[0].keys()))
        w.writeheader()
        w.writerows(events)
    print(f"\nwrote {args.out}\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
