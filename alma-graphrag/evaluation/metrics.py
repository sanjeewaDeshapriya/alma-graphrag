"""
Information-retrieval metrics for the GraphRAG evaluation harness.

Binary-relevance metrics over ranked hotel-id lists:
    Precision@K, Recall@K, nDCG@K, MRR.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Set


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


def reciprocal_rank(ranked: Sequence[str], relevant: Set[str]) -> float:
    for i, x in enumerate(ranked, start=1):
        if x in relevant:
            return 1.0 / i
    return 0.0


def evaluate_ranking(ranked: Sequence[str], relevant: Set[str], k: int) -> Dict[str, float]:
    return {
        f"P@{k}": precision_at_k(ranked, relevant, k),
        f"R@{k}": recall_at_k(ranked, relevant, k),
        f"nDCG@{k}": ndcg_at_k(ranked, relevant, k),
        "MRR": reciprocal_rank(ranked, relevant),
    }


def mean_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(r[key] for r in rows) / len(rows) for key in keys}
