"""Human-choice evaluation: graph-only GraphRAG vs text retrieval, judged ONLY by
what study participants actually chose. No rule-based relevance is used.

Ground truth (per query = per study task)
    Each of the 10 study tasks is one query: a written travel scenario anchored at
    a real Colombo location, answered by choosing one of 32 real hotels. For a
    task, every hotel that at least one participant chose is relevant, and its
    relevance grade is its SELECTION SHARE - the percentage of that task's
    participants who chose it. nDCG@10 uses the share as the gain, so a hotel
    chosen by 40% of people counts four times as much as one chosen by 10%.

Metrics per query (then averaged over the 10 queries)
    nDCG@10            share-graded ranking quality (primary)
    P@10               share of the top 10 that someone chose
    R@10               share of the chosen hotels that appear in the top 10
    Choice coverage@10 percentage of all participants' choices whose hotel is in
                       the top 10 (the "how many people would find their hotel")
    MRR                1 / rank of the first chosen hotel
    Top-1 accuracy     the first result is a hotel someone chose
    Top-1 share        percentage of participants who chose the #1 result
    Rank of favourite  position of the task's most-chosen hotel

Uncertainty
    Participant-clustered bootstrap: participants are resampled with replacement,
    the ground truth is rebuilt from the resample, and every system is re-scored.
    Rankings do not depend on participants, so this measures how much the result
    depends on which people happened to take part.

Systems
    GraphRAG[unweighted]  equal-weight mean of the five graph criteria (spatial and
                          accessibility measured from the task's anchor location)
    Keyword / SemanticVec / Hybrid / CrossEncoder   over per-anchor documents that
                          verbalise each hotel's price, rating, stars, facilities and
                          travel time from the task anchor (anchor-fair)
    Filter                current-practice reference; Random is an analytic floor

    python -m evaluation.human_choice_eval                 # all 247 participants
    python -m evaluation.human_choice_eval --cohort clean  # 95 attentive ones
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from evaluation.human_eval import (DEFAULT_MATERIAL, DEFAULT_RESPONSES, DIMS, Study,
                                   build_rankers, load_material, load_responses,
                                   select_cohort)
from evaluation.metrics import ndcg_at_k_graded

logger = logging.getLogger("alma.eval.human_choice")
EV = Path(__file__).resolve().parent
G = "GraphRAG[unweighted]"
ORDER = [G, "Keyword", "SemanticVec", "Hybrid", "CrossEncoder", "Filter"]
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# --------------------------------------------------------------------------- rankers

def add_cross_encoder(study: Study, rankers: Dict[str, Any]) -> None:
    """Cross-encoder over the same anchor-relative documents the text systems use.

    The candidate set is only 32 hotels, so the cross-encoder re-scores all of
    them rather than a hybrid shortlist - the strongest form of the baseline.
    """
    from sentence_transformers import CrossEncoder
    from evaluation.human_eval import AnchorFairIndex

    index = AnchorFairIndex(study)
    model = CrossEncoder(CROSS_ENCODER_MODEL, max_length=384)
    cache: Dict[Tuple[str, str], List[str]] = {}

    def rank(anchor: str, question: str) -> List[str]:
        key = (anchor, question)
        if key not in cache:
            ids, docs = index.ids[anchor], index.docs[anchor]
            scores = model.predict([(question, d) for d in docs])
            cache[key] = [ids[i] for i in np.argsort(-np.asarray(scores))]
        return cache[key]

    rankers["CrossEncoder"] = rank


# --------------------------------------------------------------------------- metrics

def task_metrics(ranked: Sequence[str], counter: collections.Counter, k: int) -> Dict[str, float]:
    total = sum(counter.values())
    share = {h: v / total for h, v in counter.items()}
    chosen = set(share)
    top = list(ranked[:k])
    hits = [h for h in top if h in chosen]
    first = next((i for i, h in enumerate(ranked, 1) if h in chosen), None)
    favourite = max(counter, key=lambda h: (counter[h], h))
    return {
        f"nDCG@{k}": ndcg_at_k_graded(list(ranked), share, k),
        f"P@{k}": len(hits) / k,
        f"R@{k}": len(hits) / len(chosen),
        f"coverage@{k}": sum(share[h] for h in hits),
        "MRR": 1.0 / first if first else 0.0,
        "top1_accuracy": 1.0 if ranked[0] in chosen else 0.0,
        "top1_share": share.get(ranked[0], 0.0),
        "favourite_rank": float(list(ranked).index(favourite) + 1),
    }


def score(rank_lists: Dict[str, Dict[str, List[str]]], votes: Dict[str, collections.Counter],
          k: int) -> Dict[str, Dict[str, Any]]:
    out = {}
    for name, per_task in rank_lists.items():
        rows = {t: task_metrics(per_task[t], votes[t], k) for t in votes if votes[t]}
        keys = next(iter(rows.values())).keys()
        out[name] = {"mean": {m: float(np.mean([r[m] for r in rows.values()])) for m in keys},
                     "per_task": rows}
    return out


def random_expectation(pool_size: int, votes: Dict[str, collections.Counter], k: int,
                       reps: int, seed: int, pools: Dict[str, List[str]]) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    acc = collections.defaultdict(list)
    for _ in range(reps):
        lists = {t: list(rng.permutation(pools[t])) for t in votes}
        for t, c in votes.items():
            for m, v in task_metrics(lists[t], c, k).items():
                acc[m].append(v)
    return {m: float(np.mean(v)) for m, v in acc.items()}


def bootstrap(study: Study, rank_lists, k: int, reps: int, seed: int,
              metric: str) -> Dict[str, Dict[str, float]]:
    """Participant-clustered bootstrap of (GraphRAG - system) on the task-mean metric."""
    by_person: Dict[str, List[Tuple[str, str]]] = collections.defaultdict(list)
    for p, t, h in study.observations:
        by_person[p].append((t, h))
    people = sorted(by_person)
    rng = np.random.default_rng(seed)
    diffs = collections.defaultdict(list)
    for _ in range(reps):
        sample = rng.choice(people, size=len(people), replace=True)
        votes: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for p in sample:
            for t, h in by_person[p]:
                votes[t][h] += 1
        means = {name: np.mean([task_metrics(rank_lists[name][t], votes[t], k)[metric]
                                for t in votes]) for name in rank_lists}
        for name in rank_lists:
            if name != G:
                diffs[name].append(means[G] - means[name])
    out = {}
    for name, d in diffs.items():
        arr = np.array(d)
        p_two = 2 * min((arr <= 0).mean(), (arr >= 0).mean())
        out[name] = {"mean_diff": float(arr.mean()),
                     "ci_low": float(np.percentile(arr, 2.5)),
                     "ci_high": float(np.percentile(arr, 97.5)),
                     "p": float(max(p_two, 1.0 / reps))}
    # Holm correction across the comparisons
    order = sorted(out, key=lambda n: out[n]["p"])
    m, running = len(order), 0.0
    for i, name in enumerate(order):
        running = max(running, min(1.0, (m - i) * out[name]["p"]))
        out[name]["p_holm"] = running
        out[name]["significant"] = running < 0.05
    return out


def per_choice(study: Study, rank_lists, k: int) -> Dict[str, Dict[str, float]]:
    out = {}
    for name, lists in rank_lists.items():
        ranks = [lists[t].index(h) + 1 for _p, t, h in study.observations]
        out[name] = {"hit@10": float(np.mean([r <= k for r in ranks])),
                     "mean_rank": float(np.mean(ranks)),
                     "median_rank": float(np.median(ranks))}
    return out


# --------------------------------------------------------------------------- main

def run(cohort: str = "all", k: int = 10, reps: int = 2000, seed: int = 20260927) -> Dict[str, Any]:
    material = load_material(DEFAULT_MATERIAL)
    rows = load_responses(DEFAULT_RESPONSES)
    keep, stats = select_cohort(rows, cohort)
    study = Study(material, rows, keep)

    rankers = build_rankers(study, anchor_fair=True,
                            weight_vectors={G: np.full(len(DIMS), 1.0 / len(DIMS))})
    add_cross_encoder(study, rankers)
    rank_lists = {name: {t: rankers[name](spec["anchor_id"], spec["context"])
                         for t, spec in study.tasks.items()}
                  for name in ORDER}

    votes: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for _p, t, h in study.observations:
        votes[t][h] += 1

    scores = score(rank_lists, votes, k)
    pools = {t: study.pool(spec["anchor_id"]) for t, spec in study.tasks.items()}
    rnd = random_expectation(32, votes, k, 2000, seed, pools)
    boot = {m: bootstrap(study, rank_lists, k, reps, seed, m)
            for m in (f"nDCG@{k}", f"coverage@{k}")}

    wtl = {}
    for name in ORDER[1:]:
        d = [scores[G]["per_task"][t][f"nDCG@{k}"] - scores[name]["per_task"][t][f"nDCG@{k}"]
             for t in scores[G]["per_task"]]
        wtl[name] = [sum(x > 1e-9 for x in d), sum(abs(x) <= 1e-9 for x in d),
                     sum(x < -1e-9 for x in d)]

    tasks = []
    for t, spec in sorted(study.tasks.items(), key=lambda kv: int(kv[0][1:])):
        c = votes[t]
        n = sum(c.values())
        tasks.append({
            "task": t, "persona": spec["persona"], "anchor": spec["anchor_id"],
            "criterion": spec.get("primary_dimension"), "question": spec["context"],
            "choices": n, "distinct_hotels": len(c),
            "chosen": [{"hotel": material["hotels"][h]["name"], "id": h, "votes": v,
                        "share": round(v / n, 4),
                        **{f"rank_{name}": rank_lists[name][t].index(h) + 1 for name in ORDER}}
                       for h, v in c.most_common()],
        })

    return {
        "evaluation": "human-choice only (selection-share graded); no rule-based labels",
        "cohort": cohort, "cohort_stats": stats, "participants": len(study.participants()),
        "choices": len(study.observations), "queries": len(votes), "k": k,
        "candidates_per_query": 32, "systems": ORDER,
        "scores": {name: scores[name]["mean"] for name in ORDER},
        "per_task_scores": {name: scores[name]["per_task"] for name in ORDER},
        "random_expected": rnd, "per_choice": per_choice(study, rank_lists, k),
        "significance": {"method": f"participant-clustered bootstrap ({reps} resamples), "
                                   "Holm-corrected; diff = GraphRAG minus system",
                         **boot},
        "wins_ties_losses_nDCG": wtl, "tasks": tasks,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", choices=["all", "attention", "clean"], default="all")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)
    out = run(args.cohort, reps=args.reps)
    path = Path(args.out) if args.out else EV / f"results_human_choice_{args.cohort}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"cohort {args.cohort}: {out['participants']} participants, {out['choices']} choices, "
          f"{out['queries']} queries")
    keys = ["nDCG@10", "P@10", "R@10", "coverage@10", "MRR", "top1_accuracy",
            "top1_share", "favourite_rank"]
    print(f"{'system':22s}" + "".join(f"{k:>14s}" for k in keys))
    for name in ORDER:
        print(f"{name:22s}" + "".join(f"{out['scores'][name][k]:14.3f}" for k in keys))
    print(f"{'Random (expected)':22s}" + "".join(f"{out['random_expected'][k]:14.3f}" for k in keys))
    for m in ("nDCG@10", "coverage@10"):
        print(f"\nGraphRAG minus system, {m}:")
        for name, v in out["significance"][m].items():
            print(f"  {name:14s} {v['mean_diff']:+.3f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}] "
                  f"Holm p={v['p_holm']:.4f} {'SIG' if v['significant'] else 'ns'}")
    print("\nper-query nDCG@10 wins/ties/losses:", out["wins_ties_losses_nDCG"])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
