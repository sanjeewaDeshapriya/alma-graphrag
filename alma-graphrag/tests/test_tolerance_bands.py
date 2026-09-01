"""Tests for configurable gold tolerance bands and the disruption predicate.

The bands decide every relevance label in the benchmark. Making them a
parameter is only useful if the parameter genuinely flows through — a band
argument that is accepted and then ignored would be worse than a constant,
because the sensitivity sweep would report false stability.
"""
from __future__ import annotations

import pytest

from evaluation.gold import (
    DEFAULT_BANDS,
    FAIL,
    PARTIAL,
    PASS,
    ToleranceBands,
    _added_delay,
    grade,
    graded_gold,
    is_relevant,
    relevant_set,
)


def h(hid="h1", **kw):
    base = {"id": hid, "price": 10000.0, "rating": 4.0, "star": 4,
            "travel_time_min": 20.0, "travel_time_traffic_min": 22.0,
            "amenities": ["pool", "wifi"]}
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# Band plumbing
# ---------------------------------------------------------------------------

def test_module_constants_mirror_default_bands():
    """The legacy constants and the dataclass must not drift apart."""
    from evaluation import gold
    assert gold.PRICE_TOL == DEFAULT_BANDS.price
    assert gold.RATING_TOL == DEFAULT_BANDS.rating
    assert gold.STAR_TOL == DEFAULT_BANDS.star
    assert gold.TRAVEL_TOL_MIN == DEFAULT_BANDS.travel_min


def test_scaled_multiplies_every_band():
    b = DEFAULT_BANDS.scaled(2.0)
    assert b.price == pytest.approx(DEFAULT_BANDS.price * 2)
    assert b.rating == pytest.approx(DEFAULT_BANDS.rating * 2)
    assert b.star == pytest.approx(DEFAULT_BANDS.star * 2)
    assert b.travel_min == pytest.approx(DEFAULT_BANDS.travel_min * 2)
    assert b.disruption_min == pytest.approx(DEFAULT_BANDS.disruption_min * 2)


def test_bands_are_immutable():
    with pytest.raises(Exception):
        DEFAULT_BANDS.price = 0.99  # type: ignore[misc]


def test_to_dict_records_every_band():
    assert set(DEFAULT_BANDS.to_dict()) == {
        "price", "rating", "star", "travel_min", "disruption_min"
    }


# ---------------------------------------------------------------------------
# Bands actually change the verdict
# ---------------------------------------------------------------------------

def test_wider_price_band_promotes_a_fail_to_partial():
    gold = {"max_price": 10000.0}
    hotel = h(price=12000.0)  # 20% over — outside the default 15% band
    assert grade(hotel, gold) == FAIL
    assert grade(hotel, gold, DEFAULT_BANDS.scaled(2.0)) == PARTIAL


def test_zero_band_collapses_partial_to_fail():
    """scale 0.0 is strict pass/fail — the pre-graded gold, as an extreme."""
    gold = {"max_price": 10000.0}
    hotel = h(price=10500.0)  # 5% over: partial by default
    assert grade(hotel, gold) == PARTIAL
    assert grade(hotel, gold, DEFAULT_BANDS.scaled(0.0)) == FAIL


def test_exact_satisfaction_is_pass_at_any_band():
    gold = {"max_price": 10000.0}
    for scale in (0.0, 1.0, 5.0):
        assert grade(h(price=9000.0), gold, DEFAULT_BANDS.scaled(scale)) == PASS


def test_relevant_set_respects_bands():
    pool = [h("a", price=9000.0), h("b", price=12000.0)]
    gold = {"max_price": 10000.0}
    assert relevant_set(pool, gold) == {"a"}
    assert relevant_set(pool, gold, DEFAULT_BANDS.scaled(2.0)) == {"a", "b"}


def test_graded_gold_respects_bands():
    pool = [h("a", price=9000.0), h("b", price=12000.0)]
    gold = {"max_price": 10000.0}
    assert graded_gold(pool, gold) == {"a": PASS}
    assert graded_gold(pool, gold, DEFAULT_BANDS.scaled(2.0)) == {"a": PASS, "b": PARTIAL}


def test_custom_band_object_is_honoured():
    bands = ToleranceBands(price=0.5)
    assert grade(h(price=14000.0), {"max_price": 10000.0}, bands) == PARTIAL


# ---------------------------------------------------------------------------
# Disruption predicate
# ---------------------------------------------------------------------------

def test_added_delay_prefers_explicit_signal():
    assert _added_delay(h(max_eta_change_min=9.0)) == pytest.approx(9.0)


def test_added_delay_falls_back_to_travel_time_difference():
    assert _added_delay(h(travel_time_min=20.0,
                          travel_time_traffic_min=27.0)) == pytest.approx(7.0)


def test_added_delay_never_negative():
    """Off-peak sampling can make the traffic-aware time the FASTER one."""
    assert _added_delay(h(travel_time_min=25.0,
                          travel_time_traffic_min=20.0)) == pytest.approx(0.0)


def test_added_delay_is_none_without_evidence():
    assert _added_delay({"id": "x"}) is None


def test_disruption_constraint_grades_on_delay_not_travel_time():
    """The saturation fix.

    A hotel with a short journey but a large delay must FAIL a disruption
    query. Grading such a query by `max_travel_time` is what produced
    nDCG = 1.0000 for every system.
    """
    gold = {"max_added_delay_min": 5.0}
    calm = h("calm", max_eta_change_min=1.0)
    jammed = h("jammed", max_eta_change_min=25.0, travel_time_traffic_min=8.0)
    assert grade(calm, gold) == PASS
    assert grade(jammed, gold) == FAIL


def test_disruption_band_allows_slight_overshoot():
    gold = {"max_added_delay_min": 5.0}
    assert grade(h(max_eta_change_min=7.0), gold) == PARTIAL   # within +3
    assert grade(h(max_eta_change_min=12.0), gold) == FAIL


def test_missing_disruption_evidence_fails_a_disruption_query():
    """Relevance requires positive evidence, here as everywhere else."""
    assert grade({"id": "x"}, {"max_added_delay_min": 5.0}) == FAIL


def test_disruption_constraint_combines_with_others():
    gold = {"max_price": 10000.0, "max_added_delay_min": 5.0}
    assert grade(h(price=9000.0, max_eta_change_min=2.0), gold) == PASS
    # A hard fail on either constraint disqualifies.
    assert grade(h(price=9000.0, max_eta_change_min=30.0), gold) == FAIL
    assert grade(h(price=90000.0, max_eta_change_min=2.0), gold) == FAIL


def test_event_impact_constraint():
    gold = {"max_event_impact": 0.2}
    assert grade(h(event_impact=0.1), gold) == PASS
    assert grade(h(event_impact=0.9), gold) == FAIL


def test_is_relevant_binarises_at_partial():
    gold = {"max_price": 10000.0}
    assert is_relevant(h(price=10500.0), gold)     # partial counts
    assert not is_relevant(h(price=50000.0), gold)
