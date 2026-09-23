"""
Split-robustness check for the held-out human-choice evaluation.

run_human_eval.py reports ONE participant split (seed 42). A difference measured
on 28 held-out people could owe as much to which 28 were drawn as to the systems,
so this script repeats the split-fit-score loop over many seeds for the two
weighted rankers that do not need a language model or embeddings: the hand-set
profile and the vector fitted on the training participants.

Usage:
    python -m evaluation.human_eval_seeds --seeds 50
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from evaluation.human_eval import (DEFAULT_MATERIAL, DEFAULT_RESPONSES, DIMS, Study,
                                   _per_choice, _per_task, fit_weights, load_material,
                                   load_responses, select_cohort)
from src.graph.retriever import RESEARCH_WEIGHT_PROFILES, WEIGHT_PROFILES

OUT = Path(__file__).resolve().parent / "results_human_seeds.json"


def one_split(study: Study, seed: int, holdout: float, k: int, min_votes: int):
    rng = np.random.default_rng(seed)
    people = list(study.participants())
    rng.shuffle(people)
    n_test = max(1, int(len(people) * holdout))
    test_p = set(people[:n_test])
    train = [o for o in study.observations if o[0] not in test_p]
    test = [o for o in study.observations if o[0] in test_p]

    known = {**RESEARCH_WEIGHT_PROFILES, **WEIGHT_PROFILES}
    vectors = {"handset": np.array([getattr(known["handset"], d) for d in DIMS]),
               "fitted-on-train": fit_weights(study, train)}
    if "human" in known:
        vectors["human"] = np.array([getattr(known["human"], d) for d in DIMS])
    votes = collections.defaultdict(collections.Counter)
    for _p, task, hid in test:
        votes[task][hid] += 1

    out = {}
    for name, vec in vectors.items():
        ranks = {}
        for t, spec in study.tasks.items():
            ids = study.pool(spec["anchor_id"])
            ranks[t] = [ids[i] for i in np.argsort(-(study.features(spec["anchor_id"], ids) @ vec))]
        pc = _per_choice(ranks, test, k)
        pt = _per_task(ranks, votes, k, min_votes)
        out[name] = {"choice_ndcg": pc[f"nDCG@{k}"], "mrr": pc["MRR"],
                     "mean_rank": pc["mean_rank"], "task_graded_ndcg": pt.get(f"gradedNDCG@{k}")}
    out["fitted_weights"] = {d: round(float(v), 4) for d, v in zip(DIMS, vectors["fitted-on-train"])}
    return out


def summarise(values):
    arr = np.array(values, dtype=float)
    return {"mean": round(float(arr.mean()), 4), "sd": round(float(arr.std(ddof=1)), 4),
            "p2_5": round(float(np.percentile(arr, 2.5)), 4),
            "p97_5": round(float(np.percentile(arr, 97.5)), 4),
            "min": round(float(arr.min()), 4), "max": round(float(arr.max()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--min-votes", type=int, default=2)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    material = load_material(DEFAULT_MATERIAL)
    rows = load_responses(DEFAULT_RESPONSES)
    keep, stats = select_cohort(rows, "clean")
    study = Study(material, rows, keep)

    runs = [one_split(study, s, args.holdout, args.k, args.min_votes) for s in range(args.seeds)]
    systems = [name for name in ("handset", "human", "fitted-on-train") if name in runs[0]]
    gain = [r["fitted-on-train"]["choice_ndcg"] - r["handset"]["choice_ndcg"] for r in runs]
    task_gain = [r["fitted-on-train"]["task_graded_ndcg"] - r["handset"]["task_graded_ndcg"] for r in runs]
    weights = {d: summarise([r["fitted_weights"][d] for r in runs]) for d in DIMS}
    zero_share = {d: round(sum(1 for r in runs if r["fitted_weights"][d] == 0.0) / len(runs), 3)
                  for d in DIMS}
    result = {
        "seeds": args.seeds, "holdout": args.holdout, "k": args.k, "min_votes": args.min_votes,
        "cohort_stats": stats, "participants": len(study.participants()),
        "systems": systems,
        "choice_ndcg": {n: summarise([r[n]["choice_ndcg"] for r in runs]) for n in systems},
        "task_graded_ndcg": {n: summarise([r[n]["task_graded_ndcg"] for r in runs])
                             for n in systems},
        "mean_rank": {n: summarise([r[n]["mean_rank"] for r in runs]) for n in systems},
        "fitted_minus_handset_choice_ndcg": summarise(gain),
        "fitted_beats_handset_share": round(sum(1 for g in gain if g > 0) / len(gain), 3),
        "fitted_minus_handset_task_graded_ndcg": summarise(task_gain),
        "human_minus_handset_choice_ndcg": summarise(
            [r["human"]["choice_ndcg"] - r["handset"]["choice_ndcg"] for r in runs]
        ) if "human" in runs[0] else None,
        "human_beats_handset_share": round(
            sum(1 for r in runs if r["human"]["choice_ndcg"] > r["handset"]["choice_ndcg"])
            / len(runs), 3) if "human" in runs[0] else None,
        "fitted_weights": weights,
        "fitted_weight_exactly_zero_share": zero_share,
        "runs": runs,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
