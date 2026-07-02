"""
Measures hybrid retrieval (fusing graph-structured and vector/lexical retrieval)
against graph-only and the single baselines, to quantify the benefit of
combining the two retrieval paradigms.

Fusion method: Reciprocal Rank Fusion (RRF), the standard parameter-light way to
combine ranked lists. Also reports an intent-adaptive hybrid that leans on the
structured signal for single-attribute queries and on the graph for
multi-dimensional / accessibility queries.

Usage: python evaluation/run_hybrid_eval.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import logging
import statistics as st

from evaluation.baselines import FilterBaseline, VectorBaseline, WeightedGraphBaseline, fetch_city_hotels
from evaluation.gold import relevant_set
from evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from src.crag.query_parser import parse_query

logging.basicConfig(level=logging.WARNING)


def rrf(*rankings, K=60):
    score = {}
    for r in rankings:
        for rank, doc in enumerate(r, start=1):
            score[doc] = score.get(doc, 0.0) + 1.0 / (K + rank)
    return [d for d, _ in sorted(score.items(), key=lambda x: -x[1])]


def is_single_attribute(question, city):
    """A query is single-attribute if it constrains exactly one of price/rating/
    star and expresses no accessibility/multi-dimensional intent."""
    it = parse_query(question, default_city=city)
    dims = 0
    if it.max_price_lkr is not None or it.min_price_lkr is not None:
        dims += 1
    if it.min_rating is not None or it.min_star is not None:
        dims += 1
    accessibility = it.accessibility_priority == "high" or it.avoid_traffic or it.near_attractions
    return dims <= 1 and not accessibility


def main():
    spec = json.loads(Path("evaluation/queryset.json").read_text(encoding="utf-8"))
    city, k, queries = spec["city"], spec["k"], spec["queries"]
    pool = fetch_city_hotels(city)

    F, V, G = FilterBaseline(), VectorBaseline(), WeightedGraphBaseline()
    systems = ["Filter", "VectorRAG", "Graph", "Hybrid(G+V)", "Hybrid(G+F)", "Hybrid-Adaptive"]
    ndcg = {s: [] for s in systems}
    prec = {s: [] for s in systems}
    mrr = {s: [] for s in systems}

    for q in queries:
        rel = relevant_set(pool, q["gold"])
        g = G.retrieve(q["question"], city, k)
        v = V.retrieve(q["question"], city, k)
        f = F.retrieve(q["question"], city, k)

        adaptive = rrf(f, f, g) if is_single_attribute(q["question"], city) else rrf(g, g, f)

        ranked = {
            "Filter": f, "VectorRAG": v, "Graph": g,
            "Hybrid(G+V)": rrf(g, v), "Hybrid(G+F)": rrf(g, f),
            "Hybrid-Adaptive": adaptive,
        }
        for s in systems:
            ndcg[s].append(ndcg_at_k(ranked[s], rel, k))
            prec[s].append(precision_at_k(ranked[s], rel, k))
            mrr[s].append(reciprocal_rank(ranked[s], rel))

    base = st.mean(ndcg["Graph"])
    print(f"\nHybrid evaluation (city={city}, k={k}, queries={len(queries)})\n")
    print(f"{'System':<18s} {'nDCG@10':>9s} {'P@10':>7s} {'MRR':>7s}  {'vs graph-only':>14s}")
    print("-" * 60)
    for s in systems:
        m = st.mean(ndcg[s])
        delta = "" if s == "Graph" else f"{(m - base) / base * 100:+.1f}%"
        print(f"{s:<18s} {m:>9.3f} {st.mean(prec[s]):>7.3f} {st.mean(mrr[s]):>7.3f}  {delta:>14s}")
    print()


if __name__ == "__main__":
    main()
