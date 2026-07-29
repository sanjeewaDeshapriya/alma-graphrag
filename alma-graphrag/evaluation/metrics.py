"""
Information-retrieval metrics for the GraphRAG evaluation harness.

Metrics over ranked hotel-id lists:
    Precision@K, Recall@K, MRR   — binary relevance (in/out of the gold set)
    nDCG@K                       — graded gains when a {id: grade} map is given
                                   (grade 2 = fully relevant, 1 = partially),
                                   binary gains otherwise

Graded nDCG is the headline metric: with the tolerance-band gold in
evaluation/gold.py, a system that ranks fully-relevant hotels above
borderline ones must score higher than one that treats them alike.
"""
from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Set


def precision_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = ranked[:k]
    if not topk:
        return 0.0
    hits = sum(1 for x in topk if x in relevant)
    return hits / len(topk)


def recall_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for x in ranked[:k] if x in relevant)
    return hits / len(relevant)


def dcg_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    dcg = 0.0
    for i, x in enumerate(ranked[:k], start=1):
        if x in relevant:
            dcg += 1.0 / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg_at_k(ranked, relevant, k) / idcg if idcg > 0 else 0.0


def ndcg_at_k_graded(ranked: Sequence[str], gains: Mapping[str, float], k: int) -> float:
    """nDCG@K with graded gains ({hotel_id: grade}, grade > 0).

    Ideal DCG places the highest grades first, so a ranking that puts
    partially-relevant (1) hotels above fully-relevant (2) ones is penalised.
    """
    if not gains:
        return 0.0
    dcg = 0.0
    for i, x in enumerate(ranked[:k], start=1):
        g = gains.get(x, 0.0)
        if g > 0:
            dcg += g / math.log2(i + 1)
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: Set[str]) -> float:
    for i, x in enumerate(ranked, start=1):
        if x in relevant:
            return 1.0 / i
    return 0.0


def evaluate_ranking(
    ranked: Sequence[str],
    relevant: Set[str],
    k: int,
    gains: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    ndcg = (
        ndcg_at_k_graded(ranked, gains, k)
        if gains is not None
        else ndcg_at_k(ranked, relevant, k)
    )
    return {
        f"P@{k}": precision_at_k(ranked, relevant, k),
        f"R@{k}": recall_at_k(ranked, relevant, k),
        f"nDCG@{k}": ndcg,
        "MRR": reciprocal_rank(ranked, relevant),
    }


def mean_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(r[key] for r in rows) / len(rows) for key in keys}
