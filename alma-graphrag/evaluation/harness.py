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

import logging

from evaluation.baselines import all_baselines, fetch_city_hotels
from evaluation.gold import graded_gold, relevant_set
from evaluation.metrics import evaluate_ranking, mean_metrics
from evaluation.stats import compare_systems
from src.crag.query_parser import parse_query
from src.graph.retriever import WeightedRetriever

logger = logging.getLogger("alma.eval.harness")

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
) -> Tuple[Set[str], Dict[str, int], str]:
    """Human gold supersedes the rule-based bootstrap per query.

    Returns (binary relevant set, graded {id: 1|2} gains for nDCG, source).
    Human annotations are binary lists, so human-relevant hotels all carry
    grade 2 (fully relevant).
    """
    if query["id"] in human_relevant:
        ids = set(human_relevant[query["id"]])
        return ids, {hid: 2 for hid in ids}, "human"
    return relevant_set(pool, query["gold"]), graded_gold(pool, query["gold"]), "rule"


def _vector_index_meta(city: str, pool_size: int) -> Dict[str, Any]:
    """Freshness check: the pgvector snapshot must match the live Neo4j pool,
    otherwise the text baselines search a different candidate universe."""
    from src.search import vector_store as vs
    try:
        indexed = vs.count(city)
    except Exception:
        return {"available": False, "indexed": 0, "stale": False}
    stale = indexed != pool_size
    if stale:
        logger.warning(
            "pgvector index for %s has %d rows but the Neo4j pool has %d — "
            "re-run scripts/build_vector_index.py before trusting the "
            "Keyword/SemanticVec/Hybrid numbers.", city, indexed, pool_size,
        )
    return {"available": indexed > 0, "indexed": indexed, "stale": stale}


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
        gold_set, gains, gold_source = gold_for_query(pool, q, human_relevant)
        category = q.get("category", "general")
        row: Dict[str, Any] = {
            "id": q["id"], "question": q["question"], "category": category,
            "gold": q["gold"], "gold_source": gold_source,
            "n_relevant": len(gold_set), "scores": {},
        }
        for b in baselines:
            ranked = b.retrieve(q["question"], city, k)
            m = evaluate_ranking(ranked, gold_set, k, gains=gains)
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

    # Paired significance vs the proposed system on the headline metric.
    significance: Dict[str, Any] = {}
    if GRAPH_SYSTEM in results and len(queries) >= 2:
        ndcg_key = f"nDCG@{k}"
        per_system = {name: [m[ndcg_key] for m in results[name]] for name in system_order}
        significance = {
            "reference": GRAPH_SYSTEM,
            "metric": ndcg_key,
            "tests": "paired bootstrap 95% CI + Wilcoxon signed-rank, Holm-corrected",
            "vs": {
                name: {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in block.items()}
                for name, block in compare_systems(per_system, GRAPH_SYSTEM).items()
            },
        }

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
            "ndcg_gains": "graded (2 full / 1 partial)",
        },
        "vector_index": _vector_index_meta(city, len(pool)),
        "overall": overall,
        "by_category": by_category,
        "best_system": best[0],
        "best_ndcg": round(best[1].get(f"nDCG@{k}", 0.0), 4),
        "significance": significance,
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
    gold_set, gains, gold_source = gold_for_query(pool, query, human.get("relevant", {}))

    # Retrieve once with the weighted retriever to expose composite-score
    # components for the GraphRAG column; the GraphRAG baseline reuses this
    # ranking below instead of retrieving a second time.
    intent = parse_query(query["question"], default_city=city)
    if not intent.city:
        intent.city = city
    graph_result = WeightedRetriever().retrieve(intent, limit=k)
    graph_ranked = [h.id for h in graph_result.hotels[:k]]
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
        ranked = graph_ranked if b.name == GRAPH_SYSTEM else b.retrieve(query["question"], city, k)
        metrics = evaluate_ranking(ranked, gold_set, k, gains=gains)
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
