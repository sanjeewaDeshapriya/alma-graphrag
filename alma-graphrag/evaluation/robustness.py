"""
Robustness analyses for the journal version of the paper.

Three questions a reviewer asks of a weighted retriever whose weights are set by
hand, each answered offline from the frozen intents so every number reproduces:

1. **Ablation** — what does each part contribute? Each component is removed
   AFTER the intent ladder (so an intent cannot quietly re-add it), and the
   feasibility filter, the intent ladder and neighbourhood diffusion are each
   switched off in turn. Paired bootstrap + Wilcoxon + Holm against the full
   system, exactly as in the main table.

2. **Weight robustness** — is the headline an artefact of the chosen prior?
   Base vectors are drawn uniformly from the 5-simplex (Dirichlet(1,...,1)),
   the serving-time intent ladder is applied on top, and the distribution of
   nDCG@k is compared with filter-and-sort.

3. **Efficiency** — per-query wall-clock latency of every system on the same
   pool, warm, LLM baselines excluded. The proposed system is timed both with
   the graph round trip (as served) and from a cached pool (scoring only).

Usage
-----
    python evaluation/robustness.py                       # everything
    python evaluation/robustness.py --samples 500 --latency-reps 5
    python evaluation/robustness.py --skip-latency

Writes evaluation/results_robustness.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import copy
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from evaluation.baselines import FilterBaseline, WeightedGraphBaseline, all_baselines, fetch_city_hotels
from evaluation.gold import DEFAULT_BANDS
from evaluation.harness import gold_for_query, load_spec, resolve_intents
from evaluation.metrics import evaluate_ranking
from evaluation.stats import compare_systems
from src.graph.retriever import (
    ScoringWeights,
    WeightedRetriever,
    apply_intent_adjustments,
    base_weights,
    clear_candidate_cache,
)

logging.basicConfig(level=logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
COMPONENT_NAMES = ("spatial", "accessibility", "facility", "economic", "disruption")
QUERY_SETS = {
    "main": (ROOT / "evaluation" / "queryset.json", ROOT / "evaluation" / "intents_main.json"),
    "price": (ROOT / "evaluation" / "queryset_price.json", ROOT / "evaluation" / "intents_price.json"),
}


# ---------------------------------------------------------------------------
# Weight models (duck-typed on WeightedRetriever's `.predict(intent, pool)`)
# ---------------------------------------------------------------------------

@dataclass
class DropComponent:
    """Serving weights with one component zeroed after the intent ladder."""
    component: str

    def predict(self, intent: Any, _pool: Any) -> ScoringWeights:
        w = apply_intent_adjustments(base_weights("handset"), intent)
        setattr(w, self.component, 0.0)
        return w.normalised()


@dataclass
class OnlyComponent:
    """All mass on one component: the single-criterion ranking."""
    component: str

    def predict(self, _intent: Any, _pool: Any) -> ScoringWeights:
        w = ScoringWeights(0.0, 0.0, 0.0, 0.0, 0.0)
        setattr(w, self.component, 1.0)
        return w


class NoIntentLadder:
    """The hand-set prior as-is, ignoring what the query asked for."""

    def predict(self, _intent: Any, _pool: Any) -> ScoringWeights:
        return base_weights("handset").normalised()


@dataclass
class FixedBase:
    """An arbitrary base vector with the serving-time intent ladder on top."""
    base: ScoringWeights

    def predict(self, intent: Any, _pool: Any) -> ScoringWeights:
        return apply_intent_adjustments(copy.copy(self.base), intent)


class _NoFilterRetriever(WeightedRetriever):
    """Scores the whole pool: hard constraints become soft preferences only."""

    def _apply_filters(self, cands, intent):  # noqa: D401 - override
        return list(cands)


def _graph_system(label: str, model: Any = None, self_weight: float = 0.7,
                  no_filter: bool = False) -> WeightedGraphBaseline:
    b = WeightedGraphBaseline(label=label, self_weight=self_weight)
    if no_filter:
        b._retriever = _NoFilterRetriever(self_weight=self_weight, cache_candidates=True)
    if model is not None:
        b._retriever.weight_model = model
    return b


# ---------------------------------------------------------------------------
# Shared evaluation plumbing
# ---------------------------------------------------------------------------

class Bench:
    """One query set: pool, frozen intents and graded gold, loaded once."""

    def __init__(self, queryset: Path, intents: Path) -> None:
        spec = load_spec(queryset)
        self.city = spec["city"]
        self.k = int(spec.get("k", 10))
        self.queries = spec["queries"]
        self.pool = fetch_city_hotels(self.city)
        self.intents = resolve_intents(self.queries, self.city, intents)
        self.gold = {q["id"]: gold_for_query(self.pool, q, {}, DEFAULT_BANDS) for q in self.queries}

    def ndcg(self, system: Any) -> List[float]:
        out = []
        for q in self.queries:
            gold_set, gains, _ = self.gold[q["id"]]
            ranked = system.retrieve(q["question"], self.city, self.k,
                                     intent=copy.deepcopy(self.intents[q["id"]]))
            out.append(evaluate_ranking(ranked, gold_set, self.k, gains=gains)[f"nDCG@{self.k}"])
        return out


def _summ(xs: List[float]) -> float:
    return round(float(np.mean(xs)), 4)


def run_ablation(bench: Bench) -> Dict[str, Any]:
    systems: Dict[str, Any] = {"Full": _graph_system("Full")}
    for c in COMPONENT_NAMES:
        systems[f"w/o {c}"] = _graph_system(f"w/o {c}", DropComponent(c))
    systems["w/o intent ladder"] = _graph_system("w/o intent ladder", NoIntentLadder())
    systems["w/o feasibility filter"] = _graph_system("w/o feasibility filter", no_filter=True)
    systems["w/o neighbourhood diffusion"] = _graph_system("w/o diffusion", self_weight=1.0)
    for c in COMPONENT_NAMES:
        systems[f"only {c}"] = _graph_system(f"only {c}", OnlyComponent(c))
    systems["Filter"] = FilterBaseline()

    per = {name: bench.ndcg(s) for name, s in systems.items()}
    stats = compare_systems(per, "Full")
    rows = []
    for name in systems:
        row = {"system": name, "ndcg": _summ(per[name])}
        if name != "Full":
            s = stats[name]
            # compare_systems reports Full - system; flip so the row reads as
            # the change caused by the ablation.
            row.update({
                "delta": round(-s["mean_diff"], 4),
                "ci_low": round(-s["ci_high"], 4),
                "ci_high": round(-s["ci_low"], 4),
                "p_holm": round(s["p_holm"], 4),
                "significant": s["significant"],
            })
        rows.append(row)

    by_cat: Dict[str, Dict[str, float]] = {}
    for i, q in enumerate(bench.queries):
        by_cat.setdefault(q.get("category", "general"), {})
    for cat in by_cat:
        idx = [i for i, q in enumerate(bench.queries) if q.get("category", "general") == cat]
        by_cat[cat] = {name: round(float(np.mean([per[name][i] for i in idx])), 4) for name in systems}
    return {"rows": rows, "by_category": by_cat}


def run_simplex(bench: Bench, samples: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    filt = bench.ndcg(FilterBaseline())
    full = bench.ndcg(_graph_system("Full"))
    filt_mean, full_mean = float(np.mean(filt)), float(np.mean(full))

    probe = _graph_system("simplex")
    scores, vectors = [], []
    for _ in range(samples):
        v = rng.dirichlet(np.ones(len(COMPONENT_NAMES)))
        probe._retriever.weight_model = FixedBase(ScoringWeights(*[float(x) for x in v]))
        per = bench.ndcg(probe)
        scores.append(float(np.mean(per)))
        vectors.append([round(float(x), 4) for x in v])

    arr = np.array(scores)
    q = np.quantile(arr, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    # Which direction of the simplex matters: correlation of each base weight
    # with the resulting score over the samples.
    V = np.array(vectors)
    corr = {c: round(float(np.corrcoef(V[:, i], arr)[0, 1]), 3) for i, c in enumerate(COMPONENT_NAMES)}
    return {
        "samples": samples,
        "seed": seed,
        "distribution": "Dirichlet(1,1,1,1,1) base vector + serving intent ladder",
        "filter_ndcg": round(filt_mean, 4),
        "handset_ndcg": round(full_mean, 4),
        "quantiles": dict(zip(["min", "p05", "p25", "median", "p75", "p95", "max"],
                              [round(float(x), 4) for x in q])),
        "mean": round(float(arr.mean()), 4),
        "share_above_filter": round(float((arr > filt_mean).mean()), 4),
        "share_at_or_above_handset": round(float((arr >= full_mean - 1e-9).mean()), 4),
        "rank_of_handset_pct": round(float((arr < full_mean - 1e-9).mean()), 4),
        "corr_weight_score": corr,
        "scores": [round(s, 4) for s in scores],
    }


def run_latency(bench: Bench, reps: int) -> Dict[str, Any]:
    systems = all_baselines(include_llm=False, include_ablations=False)
    for b in systems:
        if hasattr(b, "fit"):
            b.fit(bench.pool, bench.queries,
                  lambda q: bench.gold[q["id"]][1], intents=bench.intents)
    served = WeightedGraphBaseline(label="WeightedGraphRAG (graph round trip)")
    served._retriever.cache_candidates = False
    systems.append(served)

    out: Dict[str, Any] = {}
    for b in systems:
        if b.name == "WeightedGraphRAG":
            clear_candidate_cache()
            bench.ndcg(b)          # fill the pool cache once: scoring-only timing
        else:
            bench.ndcg(b)          # warm-up (model load, JIT, connection pools)
        times: List[float] = []
        for _ in range(reps):
            for q in bench.queries:
                intent = copy.deepcopy(bench.intents[q["id"]])
                t0 = time.perf_counter()
                if getattr(b, "wants_query_id", False):
                    b.retrieve(q["question"], bench.city, bench.k, intent=intent, query_id=q["id"])
                else:
                    b.retrieve(q["question"], bench.city, bench.k, intent=intent)
                times.append((time.perf_counter() - t0) * 1000.0)
        a = np.array(times)
        name = "WeightedGraphRAG (cached pool)" if b.name == "WeightedGraphRAG" else b.name
        out[name] = {
            "n": int(a.size),
            "median_ms": round(float(np.median(a)), 2),
            "p95_ms": round(float(np.quantile(a, 0.95)), 2),
            "mean_ms": round(float(a.mean()), 2),
        }
    return out


def disruption_diagnostic(bench: Bench) -> Dict[str, Any]:
    """Can this benchmark see the disruption component at all?

    Two separate reasons it might not: the graph snapshot carries too little
    variation in route delay for the component to separate hotels, or the
    queries labelled "disruption" are graded on something else. Both are
    measured rather than asserted.
    """
    full = _graph_system("Full")
    no_dis = _graph_system("w/o disruption", DropComponent("disruption"))
    set_changed = order_changed = 0
    for q in bench.queries:
        intent = bench.intents[q["id"]]
        a = full.retrieve(q["question"], bench.city, bench.k, intent=copy.deepcopy(intent))
        b = no_dis.retrieve(q["question"], bench.city, bench.k, intent=copy.deepcopy(intent))
        set_changed += set(a) != set(b)
        order_changed += a != b

    from src.crag.query_parser import QueryIntent
    pool = WeightedRetriever(cache_candidates=True).retrieve(
        QueryIntent(city=bench.city), limit=len(bench.pool)).hotels
    dis = np.array([h.components["disruption"] for h in pool])
    delays = sorted({float(h.raw.get("max_eta_change_min") or 0.0) for h in pool})

    gold_keys: Dict[str, Dict[str, int]] = {}
    for q in bench.queries:
        key = "+".join(sorted(q["gold"].keys()))
        gold_keys.setdefault(q.get("category", "general"), {}).setdefault(key, 0)
        gold_keys[q.get("category", "general")][key] += 1
    delay_graded = sum(1 for q in bench.queries
                       if any("disrupt" in k or "delay" in k for k in q["gold"]))
    return {
        "queries": len(bench.queries),
        "topk_set_changed_without_disruption": set_changed,
        "topk_order_changed_without_disruption": order_changed,
        "pool_disruption_min": round(float(dis.min()), 3),
        "pool_disruption_max": round(float(dis.max()), 3),
        "pool_disruption_sd": round(float(dis.std()), 4),
        "pool_route_delay_values_min": delays,
        "queries_graded_on_delay": delay_graded,
        "gold_constraints_by_category": gold_keys,
    }


def personalisation_check(city: str = "Colombo", k: int = 5) -> Dict[str, Any]:
    """Same graph, same event, two profiles: how far apart are the top-k sets?

    The event is the console's default demo event (a street race at
    6.9244, 79.8487, 3 km radius, high severity). This demonstrates the
    mechanism deterministically; it is not an evaluation of personalisation.
    """
    from src.crag.query_parser import QueryIntent
    from src.crag.user_profile import ActiveEvent, get_profile
    from src.graph.retriever import _haversine_km

    event = ActiveEvent(name="F1 Street Race", lat=6.9244, lng=79.8487,
                        impact_radius_km=3.0, severity="high")
    retriever = WeightedRetriever(cache_candidates=True)
    out: Dict[str, Any] = {"event": event.to_dict(), "k": k}
    tops = {}
    for pid in ("event_seeker", "quiet_seeker"):
        res = retriever.retrieve(QueryIntent(city=city), limit=k, profile=get_profile(pid), event=event)
        dists = [_haversine_km(event.lat, event.lng, float(h.raw["lat"]), float(h.raw["lng"]))
                 for h in res.hotels]
        tops[pid] = {h.id for h in res.hotels}
        out[pid] = {"min_km": round(min(dists), 2), "max_km": round(max(dists), 2)}
    out["overlap"] = len(tops["event_seeker"] & tops["quiet_seeker"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--latency-reps", type=int, default=5)
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--diagnostic-only", action="store_true",
                    help="recompute only the disruption diagnostic and merge it into --out")
    ap.add_argument("--out", type=Path, default=ROOT / "evaluation" / "results_robustness.json")
    args = ap.parse_args()

    if args.diagnostic_only:
        result = json.loads(args.out.read_text(encoding="utf-8"))
        result["disruption_diagnostic"] = disruption_diagnostic(Bench(*QUERY_SETS["main"]))
        result["personalisation_check"] = personalisation_check()
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result["disruption_diagnostic"], indent=2))
        return

    result: Dict[str, Any] = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "sets": {}}
    for name, (qs, intents) in QUERY_SETS.items():
        t = time.time()
        bench = Bench(qs, intents)
        block: Dict[str, Any] = {"n_queries": len(bench.queries), "pool": len(bench.pool)}
        block["ablation"] = run_ablation(bench)
        print(f"[{name}] ablation done ({time.time() - t:.0f}s)", flush=True)
        block["simplex"] = run_simplex(bench, args.samples, args.seed)
        print(f"[{name}] simplex done ({time.time() - t:.0f}s)", flush=True)
        if name == "main" and not args.skip_latency:
            block["latency"] = run_latency(bench, args.latency_reps)
            print(f"[{name}] latency done ({time.time() - t:.0f}s)", flush=True)
        result["sets"][name] = block

    result["disruption_diagnostic"] = disruption_diagnostic(Bench(*QUERY_SETS["main"]))
    result["personalisation_check"] = personalisation_check()
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out.resolve().relative_to(ROOT)}")


if __name__ == "__main__":
    main()
