"""Regex-only intent parsing (LLM slot-fill disabled in conftest)."""
from src.crag.query_parser import parse_query


def test_default_city_used_when_question_names_none():
    intent = parse_query("cheap hotels", default_city="Colombo")
    assert intent.city == "Colombo"


def test_max_price_under():
    intent = parse_query("hotels under 20,000 LKR", default_city="Colombo")
    assert intent.max_price_lkr == 20000.0
    assert intent.min_price_lkr is None


def test_price_between():
    intent = parse_query("hotels between Rs 5000 and 12000", default_city="Colombo")
    assert intent.min_price_lkr == 5000.0
    assert intent.max_price_lkr == 12000.0


def test_star_and_rating():
    intent = parse_query("4 star hotels with rating above 4.2", default_city="Colombo")
    assert intent.min_star == 4
    assert intent.min_rating == 4.2


def test_top_rated_implies_min_rating():
    intent = parse_query("top-rated hotels", default_city="Colombo")
    assert intent.min_rating == 4.0
    assert intent.sort_intent == "highest_rated"


def test_cheapest_sort_intent():
    intent = parse_query("budget friendly hotels", default_city="Colombo")
    assert intent.sort_intent == "cheapest"


def test_quiet_query_sets_far_and_avoid_traffic():
    intent = parse_query("a quiet peaceful hotel", default_city="Colombo")
    assert intent.proximity_preference == "far"
    assert intent.avoid_traffic is True
    assert intent.sort_intent == "most_accessible"


def test_avoid_traffic_terms():
    intent = parse_query("hotels with easy access and stable eta", default_city="Colombo")
    assert intent.avoid_traffic is True
    assert intent.sort_intent == "most_accessible"


def test_accessibility_priority_terms():
    intent = parse_query("accessible hotels with good transport links", default_city="Colombo")
    assert intent.accessibility_priority == "high"


def test_amenity_extraction():
    intent = parse_query("hotel with pool and wifi", default_city="Colombo")
    assert "pool" in intent.required_amenities
    assert "wifi" in intent.required_amenities


def test_attraction_extraction():
    intent = parse_query("hotels near the beach", default_city="Colombo")
    assert "beach" in intent.near_attractions
