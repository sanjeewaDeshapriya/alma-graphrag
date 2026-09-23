"""
Shared evaluation core — used by both the CLI (`run_eval.py`) and the API
(`src/api/eval_routes.py`) so they compute identical numbers.

Entry points:
    run_evaluation(...)    -> aggregate metrics over the whole query set
                              (the structure written to evaluation/results.json).
    inspect_query(...)     -> a full per-query trace: every system's ranked list
                              with per-hotel relevance flags + metric breakdown,
                              plus the GraphRAG composite-score components.
                              Powers the walkthrough in the UI.
    weight_sensitivity(...) -> how many queries can tell weight vectors apart.

Two properties this module is responsible for:

1. **One parse per query, shared by every system.** `parse_query` has a
   non-deterministic LLM slot-fill stage, so letting each baseline parse
   independently means systems can be answering different readings of the same
   question — and any measured gap then confounds retrieval quality with parser
   luck. The intent is resolved once here and threaded through.

2. **Reporting whether the benchmark can see the contribution.**
   `weight_sensitivity` measures the fraction of queries whose top-k actually
   changes when the weight vector changes. A benchmark where that fraction is
   low cannot measure a re-weighting method, no matter what its headline nDCG
   says, and the number belongs in the results file next to the metrics rather
   than in a footnote.
"""
from __future__ import annotations

import copy
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from evaluation.baselines import all_baselines, fetch_city_hotels
from evaluation.gold import DEFAULT_BANDS, ToleranceBands, graded_gold, relevant_set
from evaluation.metrics import evaluate_ranking, mean_metrics
from evaluation.stats import compare_systems
from src.config import SCORING_WEIGHTS_PROFILE
from src.crag.query_parser import parse_query
from src.graph.retriever import (
    DEFAULT_PRICE_POLICY,
    HISTORICAL_WEIGHT_PROFILES,
    WEIGHT_PROFILES,
    ScoringWeights,
    WeightedRetriever,
    apply_intent_adjustments,
)

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
    bands: ToleranceBands = DEFAULT_BANDS,
) -> Tuple[Set[str], Dict[str, int], str]:
    """Human gold supersedes the rule-based bootstrap per query.

    Returns (binary relevant set, graded {id: 1|2} gains for nDCG, source).
    Human annotations are binary lists, so human-relevant hotels all carry
    grade 2 (fully relevant).
    """
    if query["id"] in human_relevant:
        ids = set(human_relevant[query["id"]])
        return ids, {hid: 2 for hid in ids}, "human"
    return (relevant_set(pool, query["gold"], bands),
            graded_gold(pool, query["gold"], bands), "rule")


def resolve_intents(queries: List[Dict[str, Any]], city: str,
                    cache_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Parse every query ONCE. Returns {query_id: QueryIntent}.

    See the module docstring: sharing one parse across systems is what makes the
    comparison a comparison of retrieval rather than of parser draws.

    `cache_path` freezes the parse ACROSS runs as well. The LLM slot-filler is
    not deterministic even at temperature 0 (measured 2026-09-14: the same
    price-slice query parsed to different sort intents on consecutive calls,
    moving that slice's headline nDCG by 0.045 between runs). When the file
    exists, intents for questions it holds are read from it; any question it
    lacks is parsed live and the file is rewritten, so a results file can be
    reproduced exactly by re-running against the same intent file.
    """
    import json
    from src.crag.query_parser import QueryIntent

    cached: Dict[str, Dict[str, Any]] = {}
    path = Path(cache_path) if cache_path else None
    if path and path.exists():
        cached = json.loads(path.read_text(encoding="utf-8")).get("intents", {})

    intents: Dict[str, Any] = {}
    changed = False
    for q in queries:
        key = f"{city}::{q['question']}"
        if key in cached:
            intent = QueryIntent(**cached[key])
        else:
            intent = parse_query(q["question"], default_city=city)
            cached[key] = intent.to_dict()
            changed = True
        if not intent.city:
            intent.city = city
        intents[q["id"]] = intent
    if path and changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"note": "frozen query intents; delete to re-parse",
                                    "intents": cached}, indent=2), encoding="utf-8")
    return intents


def _retrieve(baseline: Any, query: Dict[str, Any], city: str, k: int,
              intent: Any) -> List[str]:
    """Call a baseline with the shared intent, and the query id if it wants one."""
    if getattr(baseline, "wants_query_id", False):
        return baseline.retrieve(query["question"], city, k, intent=intent,
                                 query_id=query["id"])
    return baseline.retrieve(query["question"], city, k, intent=intent)


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
# Benchmark diagnostic — can this query set see a weight change at all?
# ---------------------------------------------------------------------------

class _StaticWeightModel:
    """Non-serving counterfactual vector used only for sensitivity analysis."""

    def __init__(self, weights: ScoringWeights) -> None:
        self.weights = weights

    def predict(self, intent: Any, _candidates: Any) -> ScoringWeights:
        return apply_intent_adjustments(self.weights, intent)

def weight_sensitivity(
    queryset_path: Path | str = DEFAULT_QUERYSET,
    profiles: Sequence[str] = ("handset", "elicited", "blended", "balanced"),
    k: Optional[int] = None,
    intents: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fraction of queries whose top-k changes across weight vectors.

    Motivation: the proposed contribution is a *weighting*. If swapping the
    weight vector leaves the top-k identical for most queries, the benchmark is
    structurally incapable of measuring the contribution — every system will
    score the same and the differences that do appear are noise.

    Reports three things per query:
      changed        — did the top-k set change at all?
      order_changed  — did the ORDER change (a weaker but real sensitivity)?
      max_jaccard_gap— 1 - min pairwise Jaccard of the top-k sets

    A query set with a low `sensitive_fraction` should be regenerated with
    `evaluation/generate_queries.py --min-weight-sensitivity` rather than
    reported as-is.
    """
    spec = load_spec(queryset_path)
    city = spec["city"]
    kk = int(k or spec.get("k", 10))
    queries = spec["queries"]
    # Reuse the caller's parse when there is one: re-parsing would spend another
    # LLM call per query AND risk measuring sensitivity against a different
    # reading of the query than the metrics were computed on.
    intents = intents or resolve_intents(queries, city)

    vectors = {**HISTORICAL_WEIGHT_PROFILES, **WEIGHT_PROFILES}
    missing = [profile for profile in profiles if profile not in vectors]
    if missing:
        raise ValueError(f"unknown sensitivity profile(s): {', '.join(missing)}")
    retrievers = {
        profile: WeightedRetriever(
            weight_model=_StaticWeightModel(vectors[profile]), cache_candidates=True
        )
        for profile in profiles
    }

    rows: List[Dict[str, Any]] = []
    for q in queries:
        intent = intents[q["id"]]
        tops: Dict[str, List[str]] = {}
        for p, r in retrievers.items():
            # A fresh copy per profile: retrieve() may mutate the intent
            # (proximity_preference), and a shared object would leak that
            # mutation into the next profile's run.
            tops[p] = [h.id for h in r.retrieve(copy.deepcopy(intent), limit=kk).hotels]

        sets = {p: set(v) for p, v in tops.items()}
        names = list(profiles)
        jaccards = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                inter = len(sets[a] & sets[b])
                union = len(sets[a] | sets[b]) or 1
                jaccards.append(inter / union)
        changed = len({frozenset(s) for s in sets.values()}) > 1
        order_changed = len({tuple(v) for v in tops.values()}) > 1
        rows.append({
            "id": q["id"],
            "category": q.get("category", "general"),
            "changed": changed,
            "order_changed": order_changed,
            "max_jaccard_gap": round(1.0 - min(jaccards), 4) if jaccards else 0.0,
        })

    n = len(rows) or 1
    by_cat: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "changed": 0, "order_changed": 0}
    )
    for r in rows:
        c = by_cat[r["category"]]
        c["n"] += 1
        c["changed"] += 1 if r["changed"] else 0
        c["order_changed"] += 1 if r["order_changed"] else 0
    for c in by_cat.values():
        c["fraction"] = round(c["changed"] / c["n"], 4) if c["n"] else 0.0
        c["order_fraction"] = round(c["order_changed"] / c["n"], 4) if c["n"] else 0.0

    return {
        "profiles": list(profiles),
        "k": kk,
        "n_queries": len(rows),
        "sensitive": sum(1 for r in rows if r["changed"]),
        "sensitive_fraction": round(sum(1 for r in rows if r["changed"]) / n, 4),
        "order_sensitive": sum(1 for r in rows if r["order_changed"]),
        "order_sensitive_fraction": round(
            sum(1 for r in rows if r["order_changed"]) / n, 4),
        "by_category": dict(by_cat),
        "per_query": rows,
    }


def assess_reportability(
    *,
    query_count: int,
    human_query_count: int,
    vector_index: Dict[str, Any],
    by_category: Dict[str, Dict[str, Dict[str, float]]],
    sensitivity: Optional[Dict[str, Any]],
    ndcg_key: str,
) -> Dict[str, Any]:
    """State whether an evaluation can support comparative research claims."""
    blockers: List[str] = []
    warnings: List[str] = []
    if human_query_count < query_count:
        blockers.append("human relevance gold is incomplete; rule-based labels remain")
    if vector_index.get("stale"):
        blockers.append("vector index and graph candidate pools differ")
    if sensitivity:
        blind = [
            category for category, values in sensitivity["by_category"].items()
            if values["fraction"] == 0.0 and values["order_fraction"] == 0.0
        ]
        if blind:
            warnings.append("weight-blind categories: " + ", ".join(sorted(blind)))
    near_perfect = [
        category for category, values in by_category.items()
        if values.get(GRAPH_SYSTEM, {}).get(ndcg_key, 0.0) >= 0.99
    ]
    if near_perfect:
        warnings.append("near-perfect reference categories: " + ", ".join(sorted(near_perfect)))
    return {"reportable": not blockers, "blockers": blockers, "warnings": warnings}


# ---------------------------------------------------------------------------
# Aggregate evaluation (the results.json producer)
# ---------------------------------------------------------------------------

def run_evaluation(
    queryset_path: Path | str = DEFAULT_QUERYSET,
    gold_human_path: Path | str = DEFAULT_GOLD_HUMAN,
    no_human: bool = False,
    weight_profiles: Optional[List[str]] = None,
    price_policies: Optional[List[str]] = None,
    weight_policies: Optional[List[str]] = None,
    bands: ToleranceBands = DEFAULT_BANDS,
    include_sensitivity: bool = True,
    intent_cache: Optional[Path | str] = None,
    **baseline_opts: Any,
) -> Dict[str, Any]:
    spec = load_spec(queryset_path)
    city = spec["city"]
    k = int(spec.get("k", 10))
    queries = spec["queries"]

    human = load_human_gold(gold_human_path, no_human)
    human_relevant: Dict[str, List[str]] = human.get("relevant", {})

    pool = fetch_city_hotels(city)
    intents = resolve_intents(queries, city, intent_cache)
    available_profiles = [name for name in (weight_profiles or []) if name in WEIGHT_PROFILES]
    default_profile = (SCORING_WEIGHTS_PROFILE
                       if SCORING_WEIGHTS_PROFILE in WEIGHT_PROFILES else "handset")
    baselines = all_baselines(available_profiles, price_policies, weight_policies,
                              **baseline_opts)
    system_order = [b.name for b in baselines]

    # Gold is needed up front by any baseline that trains on it (LTR), so
    # compute it once and reuse for both training and scoring.
    gold_cache: Dict[str, Tuple[Set[str], Dict[str, int], str]] = {
        q["id"]: gold_for_query(pool, q, human_relevant, bands) for q in queries
    }
    for b in baselines:
        if hasattr(b, "fit"):
            # Shared intents go to the trainable baseline too, so it conditions
            # on exactly the query reading every other system received.
            b.fit(pool, queries, lambda q: gold_cache[q["id"]][1], intents=intents)

    results: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    cat_results: Dict[str, Dict[str, List[Dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    per_query: List[Dict[str, Any]] = []

    for q in queries:
        gold_set, gains, gold_source = gold_cache[q["id"]]
        category = q.get("category", "general")
        intent = intents[q["id"]]
        row: Dict[str, Any] = {
            "id": q["id"], "question": q["question"], "category": category,
            "gold": q["gold"], "gold_source": gold_source,
            "n_relevant": len(gold_set), "scores": {},
        }
        for b in baselines:
            ranked = _retrieve(b, q, city, k, intent)
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
    sensitivity = weight_sensitivity(queryset_path, k=k, intents=intents) if include_sensitivity else None
    vector_index = _vector_index_meta(city, len(pool))

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

    out: Dict[str, Any] = {
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
            # Recorded so a results file always states the band widths that
            # produced it; see evaluation/sensitivity.py for the sweep.
            "tolerance_bands": bands.to_dict(),
        },
        "vector_index": vector_index,
        "intent_cache": str(intent_cache) if intent_cache else None,
        "price_policy": {
            "default": DEFAULT_PRICE_POLICY,
            "compared": list(price_policies or []),
            "pool_missing_price": sum(1 for h in pool if not h.get("price")),
            "pool_missing_rate": round(
                sum(1 for h in pool if not h.get("price")) / len(pool), 4
            ) if pool else 0.0,
        },
        "weight_profiles": {
            "default": default_profile,
            "compared": available_profiles,
            "policies": list(weight_policies or []),
            "vectors": {
                name: WEIGHT_PROFILES[name].to_dict()
                for name in ([default_profile] + available_profiles)
                if name in WEIGHT_PROFILES
            },
        },
        "overall": overall,
        "by_category": by_category,
        "best_system": best[0],
        "best_ndcg": round(best[1].get(f"nDCG@{k}", 0.0), 4),
        "significance": significance,
        "per_query": per_query,
        "research_validity": assess_reportability(
            query_count=len(queries), human_query_count=len(human_relevant),
            vector_index=vector_index, by_category=by_category, sensitivity=sensitivity,
            ndcg_key=f"nDCG@{k}",
        ),
    }

    if sensitivity:
        out["weight_sensitivity"] = {
            kk: v for kk, v in sensitivity.items() if kk != "per_query"
        }
        if sensitivity["sensitive_fraction"] < 0.5:
            logger.warning(
                "Only %.0f%% of queries change their top-%d when the weight "
                "vector changes. This query set cannot measure a re-weighting "
                "method; regenerate with generate_queries.py "
                "--min-weight-sensitivity.",
                100 * sensitivity["sensitive_fraction"], k,
            )

    return out


# ---------------------------------------------------------------------------
# Per-query inspection (the walkthrough's "how it works" step)
# ---------------------------------------------------------------------------

def inspect_query(
    query_id: str,
    queryset_path: Path | str = DEFAULT_QUERYSET,
    gold_human_path: Path | str = DEFAULT_GOLD_HUMAN,
    no_human: bool = False,
    bands: ToleranceBands = DEFAULT_BANDS,
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
    gold_set, gains, gold_source = gold_for_query(
        pool, query, human.get("relevant", {}), bands
    )

    # One shared parse, same as the aggregate path.
    intent = resolve_intents([query], city)[query_id]

    # Retrieve once with the weighted retriever to expose composite-score
    # components for the GraphRAG column; the GraphRAG baseline reuses this
    # ranking below instead of retrieving a second time.
    import copy
    graph_result = WeightedRetriever().retrieve(copy.deepcopy(intent), limit=k)
    graph_ranked = [h.id for h in graph_result.hotels[:k]]
    comp_by_id = {
        h.id: {
            "score": h.score,
            "components": h.components,
            "weighted_components": h.weighted_components,
            "reasons": h.reasons,
            # Surfaces the multi-hop contribution per hotel so the walkthrough
            # can show what the neighbourhood traversal actually changed.
            "diffusion": h.raw.get("diffusion"),
        }
        for h in graph_result.hotels
    }

    systems: List[Dict[str, Any]] = []
    for b in all_baselines(include_ltr=False):
        ranked = (graph_ranked if b.name == GRAPH_SYSTEM
                  else _retrieve(b, query, city, k, intent))
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
        "tolerance_bands": bands.to_dict(),
        "systems": systems,
    }
