"""Price re-snapshot planning (src/ingest/price_backfill.py), no network or database."""
from datetime import date

import pytest

from src.ingest.price_backfill import (
    Match,
    checkin_dates,
    cheapest_nightly,
    match_hotel,
    normalise_name,
    plan_updates,
    slim_rate_entry,
    summarise_plans,
    token_overlap,
)

LAT, LNG = 6.9300, 79.8450
M = 1 / 111195.0  # degrees per metre of latitude


def _cat(id_, name, dm=0.0, deleted=None):
    return {"id": id_, "name": name, "latitude": LAT + dm * M, "longitude": LNG, "deletedAt": deleted}


def test_names_normalise_away_city_and_generic_words():
    assert normalise_name("Hilton Colombo") == normalise_name("The Hilton Hotel, Colombo")
    assert token_overlap("Cinnamon Red Colombo", "Cinnamon Red") == 1.0
    assert token_overlap("Cinnamon Red", "Cinnamon Grand") == 0.5


def test_liteapi_nodes_match_by_id():
    m = match_hotel({"id": "lp123", "source": "liteapi", "lat": LAT, "lng": LNG}, [])
    assert m == Match("lp123", "id", 0.0)


def test_exact_name_within_radius_beats_nearer_other_hotel():
    node = {"id": "ChIJx", "source": "google_places", "name": "Galle Face Hotel", "lat": LAT, "lng": LNG}
    cat = [_cat("lpA", "Some Other Hotel", 5), _cat("lpB", "Galle Face Hotel", 120)]
    m = match_hotel(node, cat)
    assert (m.liteapi_id, m.method) == ("lpB", "exact_name")
    assert m.distance_m == pytest.approx(120, abs=2)


def test_no_match_when_too_far_deleted_or_dissimilar():
    node = {"id": "ChIJx", "source": "google_places", "name": "Galle Face Hotel", "lat": LAT, "lng": LNG}
    assert match_hotel(node, [_cat("lpB", "Galle Face Hotel", 900)]) is None
    assert match_hotel(node, [_cat("lpB", "Galle Face Hotel", 10, deleted="2025-01-01")]) is None
    assert match_hotel(node, [_cat("lpC", "Kingsbury", 10)]) is None


def test_fuzzy_match_and_ambiguity():
    node = {"id": "g", "source": "google_places", "name": "Cinnamon Red Colombo", "lat": LAT, "lng": LNG}
    assert match_hotel(node, [_cat("lpR", "Cinnamon Red Hotel Colombo 3", 60)]).method in ("exact_name", "fuzzy_name")
    twin = [_cat("lp1", "Galle Face", 50), _cat("lp2", "Galle Face", 50.5)]
    node2 = {"id": "g2", "source": "google_places", "name": "Galle Face", "lat": LAT, "lng": LNG}
    assert match_hotel(node2, twin) is None


def test_cheapest_nightly_uses_minimum_offer_and_fallback():
    entry = {"roomTypes": [
        {"offerRetailRate": [{"amount": 30000}]},
        {"offerRetailRate": [], "rates": [{"retailRate": {"total": [{"amount": 24000}]}}]},
        {"offerRetailRate": [{"amount": 0}]},
    ]}
    assert cheapest_nightly(entry) == 24000
    assert cheapest_nightly(entry, nights=2) == 12000
    assert cheapest_nightly({"roomTypes": []}) is None


def test_slim_rate_entry_reproduces_price():
    entry = {"hotelId": "lp1", "extra": "x" * 1000, "roomTypes": [
        {"offerRetailRate": [], "offerId": "o", "rates": [
            {"name": "Deluxe", "boardName": "RO", "cancellationPolicies": {"x": 1},
             "retailRate": {"total": [{"amount": 18000}], "taxes": []}}]},
        {"offerRetailRate": [{"amount": 21000}], "rates": []},
    ]}
    assert cheapest_nightly(slim_rate_entry(entry)) == cheapest_nightly(entry) == 18000
    assert "extra" not in slim_rate_entry(entry)


def test_plan_takes_median_flags_duplicates_and_never_imputes():
    nodes = [
        {"id": "lpA", "name": "A", "price": 30000},
        {"id": "gA", "name": "A", "price": None},
        {"id": "gX", "name": "X", "price": None},
        {"id": "lpB", "name": "B", "price": 50000},
    ]
    matches = {"lpA": Match("lpA", "id", 0), "gA": Match("lpA", "exact_name", 12),
               "gX": None, "lpB": Match("lpB", "id", 0)}
    dates = ["d1", "d2", "d3"]
    nightly = {"lpA": {"d1": 20000, "d2": None, "d3": 26000}, "lpB": {}}
    plans = {p.hotel_id: p for p in plan_updates(nodes, matches, nightly, dates)}

    assert plans["lpA"].new_price_lkr == 23000 and plans["lpA"].n_dates == 2
    assert plans["gA"].new_price_lkr == 23000 and plans["gA"].duplicate_of == "lpA"
    assert plans["lpA"].duplicate_of is None
    assert plans["gX"].new_price_lkr is None and plans["gX"].liteapi_id is None
    assert plans["lpB"].new_price_lkr is None   # no availability: not imputed

    s = summarise_plans(list(plans.values()))
    assert (s["priced_before"], s["priced_after"], s["newly_priced"], s["lost_price"]) == (2, 2, 1, 1)
    assert s["duplicates"] == 1 and s["unmatched"] == 1


def test_checkin_dates():
    assert checkin_dates([14, 21], today=date(2026, 9, 15)) == ["2026-09-29", "2026-10-06"]
