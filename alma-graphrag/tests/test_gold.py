from evaluation.gold import PASS, PARTIAL, FAIL, grade, is_relevant, relevant_set


def h(**kw):
    base = {"id": "x", "price": None, "rating": None, "star": None,
            "travel_time_traffic_min": None, "amenities": []}
    base.update(kw)
    return base


def test_price_within_budget_is_full():
    assert grade(h(price=20000), {"max_price": 25000}) == PASS


def test_price_slightly_over_is_partial():
    # 27000 is within +15% of 25000 (=28750) -> partial, still relevant.
    assert grade(h(price=27000), {"max_price": 25000}) == PARTIAL
    assert is_relevant(h(price=27000), {"max_price": 25000}) is True


def test_price_well_over_is_not_relevant():
    assert grade(h(price=40000), {"max_price": 25000}) == FAIL


def test_missing_attribute_fails():
    assert grade(h(price=None), {"max_price": 25000}) == FAIL


def test_rating_tolerance_band():
    assert grade(h(rating=4.5), {"min_rating": 4.5}) == PASS
    assert grade(h(rating=4.35), {"min_rating": 4.5}) == PARTIAL   # within 0.2
    assert grade(h(rating=4.0), {"min_rating": 4.5}) == FAIL


def test_travel_time_tolerance():
    assert grade(h(travel_time_traffic_min=5), {"max_travel_time": 5.0}) == PASS
    assert grade(h(travel_time_traffic_min=6.5), {"max_travel_time": 5.0}) == PARTIAL  # +1.5 min
    assert grade(h(travel_time_traffic_min=9), {"max_travel_time": 5.0}) == FAIL


def test_multi_constraint_any_hard_fail_disqualifies():
    # Under budget (pass) but rating well below tolerance (fail): a hard fail
    # on any constraint means not relevant — otherwise multi-constraint gold
    # sets cover most of the pool and saturate the metrics.
    hotel = h(price=20000, rating=3.5)
    assert grade(hotel, {"max_price": 25000, "min_rating": 4.5}) == FAIL


def test_multi_constraint_band_hit_is_partial():
    # Both constraints at least band-satisfied, one only in the band -> partial.
    hotel = h(price=27000, rating=4.6)
    assert grade(hotel, {"max_price": 25000, "min_rating": 4.5}) == PARTIAL


def test_multi_constraint_two_fails_not_relevant():
    hotel = h(price=40000, rating=3.5)
    assert grade(hotel, {"max_price": 25000, "min_rating": 4.5}) == FAIL


def test_multi_constraint_all_pass_is_full():
    hotel = h(price=20000, rating=4.6)
    assert grade(hotel, {"max_price": 25000, "min_rating": 4.5}) == PASS


def test_amenities_partial_when_some_present():
    hotel = h(amenities=["Air Conditioning"])
    assert grade(hotel, {"required_amenities": ["air conditioning", "hot water"]}) == PARTIAL


def test_relevant_set_binarises():
    hotels = [h(id="a", price=20000), h(id="b", price=27000), h(id="c", price=50000)]
    for hh, hid in zip(hotels, ["a", "b", "c"]):
        hh["id"] = hid
    assert relevant_set(hotels, {"max_price": 25000}) == {"a", "b"}  # c excluded
