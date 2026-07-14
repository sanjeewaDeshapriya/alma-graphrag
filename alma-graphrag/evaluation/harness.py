"""
Shared evaluation core — used by both the CLI (`run_eval.py`) and the API
(`src/api/eval_routes.py`) so they compute identical numbers.

Two entry points:
    run_evaluation(...)  -> aggregate metrics over the whole query set
                            (the structure written to evaluation/results.json).
    inspect_query(...)   -> a full per-query trace: every system's ranked list
                            with per-hotel relevance flags + metric breakdown,
                            plus the GraphRAG composite-score components. Powers
                            the step-by-step evaluation walkthrough in the UI.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from evaluation.baselines import all_baselines, fetch_city_hotels
from evaluation.gold import relevant_set
from evaluation.metrics import evaluate_ranking, mean_metrics
from src.crag.query_parser import parse_query
from src.graph.retriever import WeightedRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERYSET = PROJECT_ROOT / "evaluation" / "queryset.json"
DEFAULT_RESULTS = PROJECT_ROOT / "evaluation" / "results.json"
DEFAULT_GOLD_HUMAN = PROJECT_ROOT / "evaluation" / "gold_human.json"

GRAPH_SYSTEM = "WeightedGraphRAG"


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_spec(queryset_path: Path | str = DEFAULT_QUERYSET) -> Dict[str, Any]:
    return json.loads(Path(queryset_path).read_text(encoding="utf-8"))


def load_human_gold(
    gold_human_path: Path | str = DEFAULT_GOLD_HUMAN,
    no_human: bool = False,
) -> Dict[str, Any]:
    """Return the human-gold payload ({} when absent/ignored).

    Shape mirrors evaluation/annotation/aggregate.py output:
        {"relevant": {qid: [hotel_id, ...]}, "krippendorff_alpha_interval": ...}
    """
    path = Path(gold_human_path)
    if no_human or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def gold_for_query(
    pool: List[Dict[str, Any]],
    query: Dict[str, Any],
    human_relevant: Dict[str, List[str]],
) -> Tuple[Set[str], str]:
    """Human gold supersedes the rule-based bootstrap per query, matching
    run_eval.py's behaviour exactly."""
    if query["id"] in human_relevant:
        return set(human_relevant[query["id"]]), "human"
    return relevant_set(pool, query["gold"]), "rule"


def _travel_time(hotel: Dict[str, Any]) -> Optional[float]:
    tt = hotel.get("travel_time_traffic_min")
    if tt is None:
        tt = hotel.get("travel_time_min")
    return float(tt) if tt is not None else None


# ---------------------------------------------------------------------------
# Aggregate evaluation (the results.json producer)
# ---------------------------------------------------------------------------

def run_evaluation(
    queryset_path: Path | str = DEFAULT_QUERYSET,
    gold_human_path: Path | str = DEFAULT_GOLD_HUMAN,
    no_human: bool = False,
) -> Dict[str, Any]:
    spec = load_spec(queryset_path)
    city = spec["city"]
    k = int(spec.get("k", 10))
    queries = spec["queries"]

    human = load_human_gold(gold_human_path, no_human)
    human_relevant: Dict[str, List[str]] = human.get("relevant", {})

    pool = fetch_city_hotels(city)
    baselines = all_baselines()
    system_order = [b.name for b in baselines]

    results: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    cat_results: Dict[str, Dict[str, List[Dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    per_query: List[Dict[str, Any]] = []

    for q in queries:
        gold_set, gold_source = gold_for_query(pool, q, human_relevant)
        category = q.get("category", "general")
        row: Dict[str, Any] = {
            "id": q["id"], "question": q["question"], "category": category,
            "gold": q["gold"], "gold_source": gold_source,
            "n_relevant": len(gold_set), "scores": {},
        }
        for b in baselines:
            ranked = b.retrieve(q["question"], city, k)
            m = evaluate_ranking(ranked, gold_set, k)
            results[b.name].append(m)
            cat_results[category][b.name].append(m)
            row["scores"][b.name] = {kk: round(v, 4) for kk, v in m.items()}
        per_query.append(row)

    cats = sorted(cat_results.keys())
    overall = {name: mean_metrics(results[name]) for name in system_order}
    by_category = {
        c: {name: mean_metrics(cat_results[c][name]) for name in system_order}
        for c in cats
    }
    best = max(overall.items(), key=lambda kv: kv[1].get(f"nDCG@{k}", 0.0))

    return {
        "city": city,
        "k": k,
        "n_queries": len(queries),
        "pool_size": len(pool),
        "system_order": system_order,
        "gold_meta": {
            "used_human": bool(human_relevant),
            "human_queries": len(human_relevant),
            "alpha": human.get("krippendorff_alpha_interval"),
        },
        "overall": overall,
        "by_category": by_category,
        "best_system": best[0],
        "best_ndcg": round(best[1].get(f"nDCG@{k}", 0.0), 4),
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Per-query inspection (the walkthrough's "how it works" step)
# ---------------------------------------------------------------------------

def inspect_query(
    query_id: str,
    queryset_path: Path | str = DEFAULT_QUERYSET,
    gold_human_path: Path | str = DEFAULT_GOLD_HUMAN,
    no_human: bool = False,
) -> Dict[str, Any]:
    spec = load_spec(queryset_path)
    city = spec["city"]
    k = int(spec.get("k", 10))
    query = next((q for q in spec["queries"] if q["id"] == query_id), None)
    if query is None:
        raise KeyError(query_id)

    pool = fetch_city_hotels(city)
    by_id = {str(h["id"]): h for h in pool}
    human = load_human_gold(gold_human_path, no_human)
    gold_set, gold_source = gold_for_query(pool, query, human.get("relevant", {}))

    # Retrieve once with the weighted retriever to expose composite-score
    # components for the GraphRAG column (same ranking the baseline produces).
    intent = parse_query(query["question"], default_city=city)
    if not intent.city:
        intent.city = city
    graph_result = WeightedRetriever().retrieve(intent, limit=k)
    comp_by_id = {
        h.id: {
            "score": h.score,
            "components": h.components,
            "weighted_components": h.weighted_components,
            "reasons": h.reasons,
        }
        for h in graph_result.hotels
    }

    systems: List[Dict[str, Any]] = []
    for b in all_baselines():
        ranked = b.retrieve(query["question"], city, k)
        metrics = evaluate_ranking(ranked, gold_set, k)
        rows: List[Dict[str, Any]] = []
        for rank, hid in enumerate(ranked, start=1):
            h = by_id.get(hid, {})
            entry: Dict[str, Any] = {
                "rank": rank,
                "id": hid,
                "name": h.get("name") or hid,
                "price_lkr": h.get("price"),
                "rating": h.get("rating"),
                "star": h.get("star"),
                "travel_time_min": _travel_time(h),
                "relevant": hid in gold_set,
            }
            if b.name == GRAPH_SYSTEM and hid in comp_by_id:
                entry.update(comp_by_id[hid])
            rows.append(entry)
        systems.append({
            "name": b.name,
            "metrics": {kk: round(v, 4) for kk, v in metrics.items()},
            "ranked": rows,
        })

    return {
        "id": query["id"],
        "question": query["question"],
        "category": query.get("category", "general"),
        "city": city,
        "k": k,
        "gold": query["gold"],
        "gold_source": gold_source,
        "gold_ids": sorted(gold_set),
        "n_relevant": len(gold_set),
        "pool_size": len(pool),
        "intent": intent.to_dict(),
        "weights": graph_result.weights.to_dict(),
        "systems": systems,
    }
