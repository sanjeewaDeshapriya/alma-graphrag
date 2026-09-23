"""
Controlled disruption scenarios: does the ranking react when congestion appears?

Why this exists
---------------
The live snapshot cannot test the disruption component. Every route signal is
"light", the largest delay is 0.2 min, and removing the component changes 0 of
60 top-10 sets (results_robustness.json). Collecting real rush-hour traffic is
one fix (scripts/collect_peak_traffic.py). This is the other: write a documented
SYNTHETIC disruption into the graph, rank the same queries before and after,
and remove it again.

Design
------
* Ground truth is the scenario, not the graph. Each scenario gives every hotel a
  TRUE added delay (linear decay from the centre to the radius), and the
  `max_added_delay_min` gold is graded on that truth. Grading on the signals the
  retriever reads would rebuild the circular gold the benchmark was fixed to
  remove.
* Systems see partial evidence. Route signals are written for a deterministic
  `observed_fraction` of affected hotels, like incomplete sensor coverage. A
  hotel without a signal can only be down-ranked through NEAR_HOTEL diffusion
  (or a linked Event), so the sensor-only scenarios test the graph itself.
* Travel times are not changed, so the accessibility component cannot react and
  the test isolates the disruption channel.
* Every injected node and edge carries `scenario_id`. Removal is DETACH DELETE on
  that property, runs in `finally`, and is verified to leave nothing behind.
  While a run is in progress the live API would also see the scenario.

Reported per system, per scenario
---------------------------------
* nDCG@10 against the scenario gold for the ranking served AFTER the scenario is
  in the graph, and for the STALE ranking served before it (reaction = difference)
* mean true delay across the top 10, in minutes, after and stale
* share of the top 10 inside the zone, split into hotels with and without a signal

This is a sensitivity test on synthetic scenarios. It shows whether the
mechanism responds; it does not show that real travellers are better served.

Usage
-----
    python evaluation/build_delay_queries.py                 # once, writes the query set
    python evaluation/disruption_scenarios.py                # coverage from the scenario file
    python evaluation/disruption_scenarios.py --coverage 0.25,0.5,1.0
    python evaluation/disruption_scenarios.py --scenario fort_closure
    python evaluation/disruption_scenarios.py --cleanup      # remove leftovers only

Writes evaluation/results_disruption.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import copy
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from src.ingest.traffic import classify_severity, haversine

logger = logging.getLogger("alma.eval.disruption_scenarios")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIOS = ROOT / "evaluation" / "scenarios" / "colombo_disruptions.json"
DEFAULT_QUERYSET = ROOT / "evaluation" / "queryset_disruption.json"
DEFAULT_INTENTS = ROOT / "evaluation" / "intents_disruption.json"
DEFAULT_OUT = ROOT / "evaluation" / "results_disruption.json"

SCENARIO_PREFIX = "scenario"
REFERENCE = "WeightedGraphRAG"

# Event impact at the centre by announced severity; decays linearly with distance.
EVENT_SEVERITY_SCALE = {"high": 1.0, "medium": 0.6, "low": 0.3}


# ---------------------------------------------------------------------------
# Scenario model (pure; unit-tested without a database)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    id: str
    name: str
    lat: float
    lng: float
    radius_km: float
    peak_delay_min: float
    observed_fraction: float = 0.5
    event_linked: bool = False
    event_severity: str = "medium"
    event_title: str = ""
    rationale: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Scenario":
        ev = d.get("event") or {}
        s = cls(
            id=str(d["id"]), name=str(d["name"]),
            lat=float(d["lat"]), lng=float(d["lng"]),
            radius_km=float(d["radius_km"]), peak_delay_min=float(d["peak_delay_min"]),
            observed_fraction=float(d.get("observed_fraction", 0.5)),
            event_linked=bool(ev.get("linked", False)),
            event_severity=str(ev.get("severity", "medium")),
            event_title=str(ev.get("title", d["name"])),
            rationale=str(d.get("rationale", "")),
        )
        if s.radius_km <= 0 or s.peak_delay_min <= 0:
            raise ValueError(f"scenario {s.id}: radius and peak delay must be positive")
        if not 0.0 <= s.observed_fraction <= 1.0:
            raise ValueError(f"scenario {s.id}: observed_fraction must be in [0, 1]")
        return s


def load_scenarios(path: Path = DEFAULT_SCENARIOS) -> Tuple[str, List[Scenario]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return raw["city"], [Scenario.from_dict(s) for s in raw["scenarios"]]


def true_delay_min(s: Scenario, lat: Any, lng: Any) -> float:
    """Added delay on the route to a hotel at (lat, lng): linear decay, 0 outside."""
    if lat is None or lng is None:
        return 0.0
    d = haversine(s.lat, s.lng, float(lat), float(lng))
    if d >= s.radius_km:
        return 0.0
    return round(s.peak_delay_min * (1.0 - d / s.radius_km), 2)


def route_severity(free_flow_min: Any, delay_min: float) -> str:
    """The Google ingest rule: ratio = free-flow / in-traffic time.

    Using the same rule keeps injected signals indistinguishable from real ones.
    An unknown free-flow time is treated as 5 minutes, a typical Colombo route.
    """
    if delay_min <= 0:
        return "light"
    base = float(free_flow_min) if free_flow_min else 5.0
    return classify_severity(base / (base + delay_min), 1.0)


def observed_subset(s: Scenario, affected_ids: Iterable[str],
                    fraction: Optional[float] = None) -> Set[str]:
    """Which affected hotels get a route signal. Deterministic per scenario.

    Hotels are ordered by md5(scenario_id:hotel_id) and the first
    round(fraction * n) are kept, so coverage levels are nested: every hotel
    observed at 25% is also observed at 50%.
    """
    f = s.observed_fraction if fraction is None else float(fraction)
    ids = sorted(affected_ids, key=lambda h: hashlib.md5(f"{s.id}:{h}".encode()).hexdigest())
    return set(ids[:int(round(f * len(ids)))])


@dataclass
class InjectionPlan:
    scenario_id: str
    truth: Dict[str, float]          # hotel id -> true added delay, affected hotels only
    observed: Set[str]               # affected hotels that receive a route signal
    signals: List[Dict[str, Any]]
    event: Optional[Dict[str, Any]]
    event_links: List[Dict[str, Any]]


def plan_injection(s: Scenario, hotels: List[Dict[str, Any]],
                   fraction: Optional[float] = None,
                   now_iso: Optional[str] = None) -> InjectionPlan:
    now_iso = now_iso or datetime.now(timezone.utc).isoformat()
    by_id = {str(h["id"]): h for h in hotels}
    truth: Dict[str, float] = {}
    dist: Dict[str, float] = {}
    for hid, h in by_id.items():
        delay = true_delay_min(s, h.get("lat"), h.get("lng"))
        if delay > 0:
            truth[hid] = delay
            dist[hid] = haversine(s.lat, s.lng, float(h["lat"]), float(h["lng"]))

    observed = observed_subset(s, truth, fraction)
    signals = [{
        "id": f"{SCENARIO_PREFIX}:{s.id}:{hid}",
        "hotel_id": hid,
        "timestamp": now_iso,
        "location_name": f"{s.name} -> {by_id[hid].get('name', '?')}",
        "severity": route_severity(by_id[hid].get("travel_time_min"), truth[hid]),
        "eta_change_min": truth[hid],
        "lat": float(by_id[hid].get("lat") or 0.0),
        "lng": float(by_id[hid].get("lng") or 0.0),
        "source": "scenario",
        "scenario_id": s.id,
    } for hid in sorted(observed)]

    event, links = None, []
    if s.event_linked:
        scale = EVENT_SEVERITY_SCALE.get(s.event_severity, 0.6)
        event = {
            "id": f"{SCENARIO_PREFIX}:{s.id}:event", "title": s.event_title,
            "type": "scenario", "severity": s.event_severity,
            "start_time": now_iso, "end_time": "", "source": "scenario",
            "scenario_id": s.id,
        }
        links = [{
            "hotel_id": hid,
            "impact_score": round(scale * (1.0 - dist[hid] / s.radius_km), 4),
            "distance_km": round(dist[hid], 3),
        } for hid in sorted(truth)]
    return InjectionPlan(s.id, truth, observed, signals, event, links)


def grid_scenarios(hotels: List[Dict[str, Any]], spacing_km: float, radius_km: float,
                   peak_delay_min: float, observed_fraction: float = 0.5,
                   min_affected: int = 3) -> List[Scenario]:
    """Scenario centres on a regular grid over the hotels' bounding box.

    The named scenarios were placed by hand, and the first run showed why that
    is fragile: a 1.2 km zone at Galle Face stopped just short of the hotels the
    retriever ranks highest, so it measured almost nothing. A grid removes the
    choice. Centres that would affect fewer than `min_affected` hotels are
    dropped because they cannot change a top-10 in any system.
    """
    pts = [(float(h["lat"]), float(h["lng"])) for h in hotels
           if h.get("lat") is not None and h.get("lng") is not None]
    if not pts:
        return []
    lat0, lat1 = min(p[0] for p in pts), max(p[0] for p in pts)
    lng0, lng1 = min(p[1] for p in pts), max(p[1] for p in pts)
    km_lat = 111.195
    km_lng = 111.195 * math.cos(math.radians((lat0 + lat1) / 2))
    n_lat = max(1, int(math.ceil((lat1 - lat0) * km_lat / spacing_km)) + 1)
    n_lng = max(1, int(math.ceil((lng1 - lng0) * km_lng / spacing_km)) + 1)

    out: List[Scenario] = []
    for i in range(n_lat):
        for j in range(n_lng):
            lat = lat0 + i * spacing_km / km_lat
            lng = lng0 + j * spacing_km / km_lng
            affected = sum(1 for p in pts if haversine(lat, lng, p[0], p[1]) < radius_km)
            if affected < min_affected:
                continue
            out.append(Scenario(
                id=f"grid_{i:02d}_{j:02d}", name=f"Grid closure {i},{j}",
                lat=round(lat, 5), lng=round(lng, 5), radius_km=radius_km,
                peak_delay_min=peak_delay_min, observed_fraction=observed_fraction,
                rationale="grid sweep: centre chosen by position, not by hand",
            ))
    return out


def gold_pool(pool: List[Dict[str, Any]], truth: Dict[str, float]) -> List[Dict[str, Any]]:
    """Copy of the pool whose route delay is the scenario TRUTH, for grading only."""
    out = []
    for h in pool:
        c = dict(h)
        hid = str(h["id"])
        if hid in truth:
            c["max_eta_change_min"] = max(float(h.get("max_eta_change_min") or 0.0), truth[hid])
        out.append(c)
    return out


def exposure(ranked: List[str], plan: InjectionPlan, k: int) -> Dict[str, float]:
    """How much of the served top-k sits in the disruption, and how badly."""
    top = ranked[:k]
    if not top:
        return {"mean_delay_min": 0.0, "share_affected": 0.0,
                "share_observed": 0.0, "share_unobserved": 0.0}
    n = len(top)
    affected = [h for h in top if h in plan.truth]
    return {
        "mean_delay_min": round(sum(plan.truth.get(h, 0.0) for h in top) / n, 3),
        "share_affected": round(len(affected) / n, 4),
        "share_observed": round(sum(1 for h in affected if h in plan.observed) / n, 4),
        "share_unobserved": round(sum(1 for h in affected if h not in plan.observed) / n, 4),
    }


# ---------------------------------------------------------------------------
# Graph writes
# ---------------------------------------------------------------------------

class GraphInjector:
    """Writes and removes scenario-tagged nodes. Nothing untagged is touched."""

    def __init__(self, driver: Any = None) -> None:
        if driver is None:
            from src.graph.query import _get_driver
            driver = _get_driver()
        self.driver = driver

    def inject(self, plan: InjectionPlan) -> None:
        with self.driver.session() as session:
            if plan.signals:
                session.run(
                    """
                    UNWIND $rows AS r
                    MATCH (h:Hotel {id: r.hotel_id})
                    MERGE (t:TrafficSignal {id: r.id})
                    SET t.timestamp = r.timestamp, t.location_name = r.location_name,
                        t.severity = r.severity, t.eta_change_min = r.eta_change_min,
                        t.lat = r.lat, t.lng = r.lng, t.source = r.source,
                        t.scenario_id = r.scenario_id
                    MERGE (h)-[e:HAS_SIGNAL]->(t)
                    SET e.severity = r.severity, e.scenario_id = r.scenario_id
                    """,
                    {"rows": plan.signals},
                )
            if plan.event:
                session.run("MERGE (e:Event {id: $id}) SET e += $props",
                            {"id": plan.event["id"], "props": plan.event})
                session.run(
                    """
                    UNWIND $rows AS r
                    MATCH (h:Hotel {id: r.hotel_id})
                    MATCH (e:Event {id: $event_id})
                    MERGE (h)-[a:AFFECTED_BY]->(e)
                    SET a.impact_score = r.impact_score, a.distance_km = r.distance_km,
                        a.impact_type = 'scenario', a.scenario_id = $sid
                    """,
                    {"rows": plan.event_links, "event_id": plan.event["id"], "sid": plan.scenario_id},
                )

    def remove(self, scenario_id: Optional[str] = None) -> int:
        where = "n.scenario_id = $sid" if scenario_id else "n.scenario_id IS NOT NULL"
        with self.driver.session() as session:
            rec = session.run(
                f"MATCH (n) WHERE {where} DETACH DELETE n RETURN count(n) AS n",
                {"sid": scenario_id},
            ).single()
        return int(rec["n"]) if rec else 0

    def leftovers(self, scenario_id: Optional[str] = None) -> int:
        where = "n.scenario_id = $sid" if scenario_id else "n.scenario_id IS NOT NULL"
        with self.driver.session() as session:
            nodes = session.run(f"MATCH (n) WHERE {where} RETURN count(n) AS n",
                                {"sid": scenario_id}).single()["n"]
            rels = session.run(
                "MATCH ()-[r]->() WHERE "
                + ("r.scenario_id = $sid" if scenario_id else "r.scenario_id IS NOT NULL")
                + " RETURN count(r) AS n",
                {"sid": scenario_id},
            ).single()["n"]
        return int(nodes) + int(rels)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def build_systems() -> List[Any]:
    from evaluation.baselines import (
        FilterBaseline, HybridBaseline, KeywordBaseline, PopularityBaseline,
        RandomBaseline, SemanticBaseline, WeightedGraphBaseline,
    )
    from evaluation.robustness import DropComponent
    from src.search import vector_store as vs

    systems: List[Any] = [RandomBaseline(), PopularityBaseline(), FilterBaseline()]
    if vs.is_available():
        systems += [KeywordBaseline(), SemanticBaseline(), HybridBaseline()]
    full = WeightedGraphBaseline(label=REFERENCE)
    no_dis = WeightedGraphBaseline(label="WeightedGraphRAG[w/o disruption]")
    no_dis._retriever.weight_model = DropComponent("disruption")
    no_diff = WeightedGraphBaseline(label="WeightedGraphRAG[w/o diffusion]", self_weight=1.0)
    return systems + [full, no_dis, no_diff]


def _rank_all(systems: List[Any], queries: List[Dict[str, Any]], intents: Dict[str, Any],
              city: str, k: int) -> Dict[str, Dict[str, List[str]]]:
    return {
        s.name: {q["id"]: s.retrieve(q["question"], city, k, intent=copy.deepcopy(intents[q["id"]]))
                 for q in queries}
        for s in systems
    }


def run_scenario(s: Scenario, systems: List[Any], queries: List[Dict[str, Any]],
                 intents: Dict[str, Any], city: str, k: int, fraction: Optional[float],
                 injector: GraphInjector) -> Dict[str, Any]:
    from evaluation.baselines import fetch_city_hotels
    from evaluation.gold import DEFAULT_BANDS
    from evaluation.harness import gold_for_query
    from evaluation.metrics import evaluate_ranking
    from src.graph.retriever import clear_candidate_cache

    clear_candidate_cache()
    base_pool = fetch_city_hotels(city)
    plan = plan_injection(s, base_pool, fraction)
    stale = _rank_all(systems, queries, intents, city, k)

    injector.inject(plan)
    clear_candidate_cache()
    try:
        live = _rank_all(systems, queries, intents, city, k)
    finally:
        removed = injector.remove(s.id)
        clear_candidate_cache()
    left = injector.leftovers(s.id)
    if left:
        raise RuntimeError(f"scenario {s.id}: {left} tagged nodes/edges left after removal")

    gpool = gold_pool(base_pool, plan.truth)
    per_query, skipped = [], []
    for q in queries:
        relevant, gains, _ = gold_for_query(gpool, q, {}, DEFAULT_BANDS)
        if not relevant or len(relevant) == len(gpool):
            skipped.append(q["id"])
            continue
        row: Dict[str, Any] = {"id": q["id"], "gold_size": len(relevant), "systems": {}}
        for name in live:
            after, before = live[name][q["id"]], stale[name][q["id"]]
            ex_after, ex_before = exposure(after, plan, k), exposure(before, plan, k)
            row["systems"][name] = {
                "ndcg": round(evaluate_ranking(after, relevant, k, gains=gains)[f"nDCG@{k}"], 4),
                "ndcg_stale": round(evaluate_ranking(before, relevant, k, gains=gains)[f"nDCG@{k}"], 4),
                **ex_after,
                "mean_delay_min_stale": ex_before["mean_delay_min"],
                "topk_changed": set(after) != set(before),
            }
        per_query.append(row)

    return {
        "scenario": s.id,
        "name": s.name,
        "coverage": s.observed_fraction if fraction is None else fraction,
        "affected_hotels": len(plan.truth),
        "observed_hotels": len(plan.observed),
        "signals_written": len(plan.signals),
        "event_links_written": len(plan.event_links),
        "nodes_removed": removed,
        "queries_scored": len(per_query),
        "queries_skipped_degenerate": skipped,
        "summary": summarise(per_query),
        "per_query": per_query,
    }


def summarise(per_query: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    if not per_query:
        return {}
    metrics = ("ndcg", "ndcg_stale", "mean_delay_min", "mean_delay_min_stale",
               "share_affected", "share_unobserved", "topk_changed")
    names = per_query[0]["systems"].keys()
    out: Dict[str, Dict[str, float]] = {}
    for name in names:
        rows = [r["systems"][name] for r in per_query]
        out[name] = {m: round(sum(float(r[m]) for r in rows) / len(rows), 4) for m in metrics}
        out[name]["reaction_ndcg"] = round(out[name]["ndcg"] - out[name]["ndcg_stale"], 4)
    return out


def pooled_tests(scenario_results: List[Dict[str, Any]], unit: str = "query") -> Dict[str, Any]:
    """Paired tests at one coverage level.

    unit="query" pairs every (scenario, query); the same query recurs across
    scenarios, so those pairs are not independent. unit="scenario" pairs
    per-scenario means instead, which is the honest unit for the grid sweep.
    """
    from evaluation.stats import compare_systems

    ndcg: Dict[str, List[float]] = {}
    neg_delay: Dict[str, List[float]] = {}
    reaction: Dict[str, List[float]] = {"live": [], "stale": []}
    for sc in scenario_results:
        if unit == "scenario":
            if not sc["summary"]:
                continue
            rows = [{"systems": sc["summary"]}]
        else:
            rows = sc["per_query"]
        for row in rows:
            for name, m in row["systems"].items():
                ndcg.setdefault(name, []).append(m["ndcg"])
                neg_delay.setdefault(name, []).append(-m["mean_delay_min"])
            reaction["live"].append(row["systems"][REFERENCE]["ndcg"])
            reaction["stale"].append(row["systems"][REFERENCE]["ndcg_stale"])
    if not reaction["live"]:
        return {"unit": unit, "n_pairs": 0}

    delay_tests = compare_systems(neg_delay, REFERENCE)
    for block in delay_tests.values():
        # compare_systems reports ref - system on the negated metric, which is
        # (system delay - reference delay): minutes the reference saves.
        block["minutes_saved_by_reference"] = round(block.pop("mean_diff"), 3)
    return {
        "unit": unit,
        "n_pairs": len(reaction["live"]),
        "ndcg_vs_reference": compare_systems(ndcg, REFERENCE),
        "top10_delay_vs_reference": delay_tests,
        "reference_live_vs_stale": compare_systems(reaction, "live")["stale"],
    }


def _print_table(label: str, results: List[Dict[str, Any]]) -> None:
    print(f"\n=== {label} ===")
    for sc in results:
        print(f"\n{sc['name']}  (affected {sc['affected_hotels']}, with signal "
              f"{sc['observed_hotels']}, queries {sc['queries_scored']}, "
              f"skipped {len(sc['queries_skipped_degenerate'])})")
        print(f"  {'system':34s} {'nDCG':>6s} {'stale':>6s} {'react':>6s} "
              f"{'delay':>6s} {'stale':>6s} {'unobs':>6s}")
        for name, m in sc["summary"].items():
            print(f"  {name:34s} {m['ndcg']:6.3f} {m['ndcg_stale']:6.3f} {m['reaction_ndcg']:+6.3f} "
                  f"{m['mean_delay_min']:6.2f} {m['mean_delay_min_stale']:6.2f} {m['share_unobserved']:6.3f}")


def _print_grid_summary(label: str, results: List[Dict[str, Any]]) -> None:
    scored = [sc for sc in results if sc["summary"]]
    print(f"\n=== grid, {label}: mean over {len(scored)} scenarios ===")
    print(f"  {'system':34s} {'nDCG':>6s} {'stale':>6s} {'react':>6s} "
          f"{'delay':>6s} {'stale':>6s} {'top10 chg':>9s}")
    for name in scored[0]["summary"]:
        m = {key: sum(sc["summary"][name][key] for sc in scored) / len(scored)
             for key in ("ndcg", "ndcg_stale", "reaction_ndcg", "mean_delay_min",
                         "mean_delay_min_stale", "topk_changed")}
        print(f"  {name:34s} {m['ndcg']:6.3f} {m['ndcg_stale']:6.3f} {m['reaction_ndcg']:+6.3f} "
              f"{m['mean_delay_min']:6.2f} {m['mean_delay_min_stale']:6.2f} {m['topk_changed']:9.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    ap.add_argument("--queryset", type=Path, default=DEFAULT_QUERYSET)
    ap.add_argument("--intent-cache", type=Path, default=DEFAULT_INTENTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--scenario", default="", help="comma-separated scenario ids (default all)")
    ap.add_argument("--coverage", default="",
                    help="comma-separated sensor coverage levels overriding the file, e.g. 0.25,0.5,1.0")
    ap.add_argument("--grid", type=float, default=0.0,
                    help="run a grid sweep with this centre spacing in km instead of the named scenarios")
    ap.add_argument("--grid-radius", type=float, default=1.0)
    ap.add_argument("--grid-peak", type=float, default=15.0)
    ap.add_argument("--cleanup", action="store_true", help="remove every scenario-tagged node and exit")
    args = ap.parse_args()
    if args.grid and args.out == DEFAULT_OUT:
        args.out = ROOT / "evaluation" / "results_disruption_grid.json"
    logging.basicConfig(level=logging.ERROR)

    injector = GraphInjector()
    if args.cleanup:
        print(f"removed {injector.remove()} scenario-tagged nodes")
        return
    stray = injector.leftovers()
    if stray:
        print(f"warning: removing {injector.remove()} leftover scenario nodes from an earlier run")

    from evaluation.harness import load_spec, resolve_intents

    city, scenarios = load_scenarios(args.scenarios)
    wanted = {x.strip() for x in args.scenario.split(",") if x.strip()}
    if wanted:
        scenarios = [s for s in scenarios if s.id in wanted]
    spec = load_spec(args.queryset)
    queries, k = spec["queries"], int(spec.get("k", 10))
    if spec["city"].lower() != city.lower():
        raise SystemExit(f"query set city {spec['city']} != scenario city {city}")
    intents = resolve_intents(queries, city, args.intent_cache)
    avoid = sum(1 for i in intents.values() if getattr(i, "avoid_traffic", False))

    systems = build_systems()
    levels: List[Optional[float]] = (
        [float(x) for x in args.coverage.split(",") if x.strip()] or [None]
    )

    out: Dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "city": city, "k": k,
        "queryset": str(args.queryset.relative_to(ROOT)),
        "n_queries": len(queries),
        "intents_with_avoid_traffic": avoid,
        "systems": [s.name for s in systems],
        "scenario_file": str(args.scenarios.relative_to(ROOT)),
        "note": ("Synthetic scenarios. Gold graded on true scenario delay; systems see "
                 "route signals for `coverage` of affected hotels. Travel times unchanged."),
        "runs": {},
    }
    if args.grid:
        from evaluation.baselines import fetch_city_hotels
        scenarios = grid_scenarios(fetch_city_hotels(city), args.grid, args.grid_radius, args.grid_peak)
        out["grid"] = {"spacing_km": args.grid, "radius_km": args.grid_radius,
                       "peak_delay_min": args.grid_peak, "centres": len(scenarios)}
        print(f"grid: {len(scenarios)} centres (spacing {args.grid} km, radius {args.grid_radius} km)")

    for level in levels:
        label = "file" if level is None else f"coverage={level:g}"
        results = []
        for n, s in enumerate(scenarios, 1):
            results.append(run_scenario(s, systems, queries, intents, city, k, level, injector))
            if args.grid:
                print(f"  [{label}] {n}/{len(scenarios)} {s.id}", flush=True)
        if args.grid:
            _print_grid_summary(label, results)
            out["runs"][label] = {"scenarios": results, "pooled": pooled_tests(results, unit="scenario")}
        else:
            _print_table(label, results)
            out["runs"][label] = {"scenarios": results, "pooled": pooled_tests(results)}

    if injector.leftovers():
        raise RuntimeError("scenario nodes remain in the graph after the run")
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out.relative_to(ROOT)} (graph clean: 0 scenario nodes)")


if __name__ == "__main__":
    main()
