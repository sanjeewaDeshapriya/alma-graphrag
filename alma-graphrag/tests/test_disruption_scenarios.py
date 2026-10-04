"""Scenario planning for evaluation/disruption_scenarios.py (no database)."""
import pytest

from evaluation.disruption_scenarios import (
    DEFAULT_SCENARIOS,
    SCENARIO_PREFIX,
    InjectionPlan,
    Scenario,
    exposure,
    gold_pool,
    grid_scenarios,
    load_scenarios,
    observed_subset,
    plan_injection,
    route_severity,
    true_delay_min,
)
from evaluation.gold import grade

CENTRE = (6.9244, 79.8487)


def _scenario(**kw):
    base = dict(id="s", name="Test closure", lat=CENTRE[0], lng=CENTRE[1],
                radius_km=1.0, peak_delay_min=20.0, observed_fraction=0.5)
    base.update(kw)
    return Scenario(**base)


def _pool():
    # Hotels due north of the centre at 0, 0.25, 0.5, 0.75 km (affected) and 2 km (not).
    km_per_deg = 111.195
    return [
        {"id": f"h{i}", "name": f"Hotel {i}", "lat": CENTRE[0] + d / km_per_deg, "lng": CENTRE[1],
         "travel_time_min": 5.0, "max_eta_change_min": 0.0, "price": 20000}
        for i, d in enumerate([0.0, 0.25, 0.5, 0.75, 2.0])
    ]


def test_delay_decays_linearly_and_is_zero_outside_radius():
    s = _scenario()
    pool = _pool()
    delays = [true_delay_min(s, h["lat"], h["lng"]) for h in pool]
    assert delays[0] == pytest.approx(20.0)
    assert delays[1] == pytest.approx(15.0, abs=0.05)
    assert delays[2] == pytest.approx(10.0, abs=0.05)
    assert delays[4] == 0.0
    assert true_delay_min(s, None, None) == 0.0


def test_route_severity_uses_the_ingest_ratio_rule():
    assert route_severity(5.0, 0.0) == "light"
    assert route_severity(10.0, 2.0) == "light"       # 10/12 = 0.83
    assert route_severity(10.0, 10.0) == "moderate"   # 10/20 = 0.50
    assert route_severity(2.0, 10.0) == "heavy"       # 2/12 = 0.17


def test_observed_subset_is_deterministic_and_nested():
    s = _scenario()
    ids = [f"h{i}" for i in range(20)]
    half = observed_subset(s, ids, 0.5)
    assert half == observed_subset(s, list(reversed(ids)), 0.5)
    assert len(half) == 10
    assert observed_subset(s, ids, 0.25) <= half <= observed_subset(s, ids, 1.0)
    assert observed_subset(s, ids, 0.0) == set()
    assert observed_subset(s, ids, 1.0) == set(ids)


def test_plan_writes_signals_only_for_observed_hotels_and_tags_everything():
    s = _scenario(observed_fraction=0.5)
    plan = plan_injection(s, _pool(), now_iso="2026-09-15T00:00:00+00:00")
    assert set(plan.truth) == {"h0", "h1", "h2", "h3"}
    assert len(plan.observed) == 2
    assert {sig["hotel_id"] for sig in plan.signals} == plan.observed
    for sig in plan.signals:
        assert sig["id"].startswith(f"{SCENARIO_PREFIX}:s:")
        assert sig["scenario_id"] == "s"
        assert sig["eta_change_min"] == plan.truth[sig["hotel_id"]]
    assert plan.event is None and plan.event_links == []


def test_linked_event_covers_every_affected_hotel_with_graded_impact():
    s = _scenario(event_linked=True, event_severity="high")
    plan = plan_injection(s, _pool())
    assert plan.event["scenario_id"] == "s"
    impacts = {l["hotel_id"]: l["impact_score"] for l in plan.event_links}
    assert set(impacts) == set(plan.truth)
    assert impacts["h0"] > impacts["h1"] > impacts["h2"] > impacts["h3"] > 0


def test_gold_is_graded_on_truth_not_on_what_systems_observe():
    s = _scenario(observed_fraction=0.0)   # no signals at all
    pool = _pool()
    plan = plan_injection(s, pool)
    assert plan.signals == []
    gpool = gold_pool(pool, plan.truth)
    gold = {"max_added_delay_min": 5.0}
    by_id = {h["id"]: h for h in gpool}
    assert grade(by_id["h0"], gold) == 0          # 20 min: fails even with no signal
    assert grade(by_id["h4"], gold) == 2          # outside the zone
    assert pool[0]["max_eta_change_min"] == 0.0   # the input pool is not mutated


def test_exposure_splits_observed_and_unobserved():
    plan = InjectionPlan("s", truth={"a": 10.0, "b": 4.0}, observed={"a"},
                         signals=[], event=None, event_links=[])
    ex = exposure(["a", "b", "c", "d"], plan, k=10)
    assert ex["mean_delay_min"] == pytest.approx(3.5)
    assert ex["share_affected"] == 0.5
    assert ex["share_observed"] == 0.25
    assert ex["share_unobserved"] == 0.25
    assert exposure([], plan, k=10)["mean_delay_min"] == 0.0


def test_grid_covers_the_pool_and_drops_empty_centres():
    pool = _pool()
    grid = grid_scenarios(pool, spacing_km=0.5, radius_km=0.6, peak_delay_min=10, min_affected=2)
    assert grid, "a pool spanning 2 km must yield grid centres"
    assert len({s.id for s in grid}) == len(grid)
    for s in grid:
        affected = [h for h in pool if true_delay_min(s, h["lat"], h["lng"]) > 0]
        assert len(affected) >= 2
    assert grid_scenarios([], 0.5, 0.6, 10) == []


def test_scenario_file_loads_and_validates():
    city, scenarios = load_scenarios(DEFAULT_SCENARIOS)
    assert city == "Colombo"
    assert len({s.id for s in scenarios}) == len(scenarios) >= 3
    assert any(s.event_linked for s in scenarios)
    assert any(not s.event_linked for s in scenarios)
    with pytest.raises(ValueError):
        Scenario.from_dict({"id": "x", "name": "x", "lat": 0, "lng": 0,
                            "radius_km": 1, "peak_delay_min": 5, "observed_fraction": 1.5})
