"""`economic` is a directional component, not a "cheap is good" constant.

The component is defined cheaper-is-better. Before `price_preference` existed,
"upmarket hotels in colombo" parsed to sort_intent="best_overall" and scored the
CHEAPEST hotels highest — and raising the economic weight (as the `balanced`
profile does) made that worse, not better. On the price-preference query slice
the premium queries were the only ones where `balanced` lost to `handset`.

These tests run offline — no Neo4j, no pgvector.
"""
import pytest

from src.crag.query_parser import _regex_intent


def _intent(q):
    return _regex_intent(q, "Colombo")


@pytest.mark.parametrize("q", [
    "upmarket hotels in colombo",
    "high end hotels with excellent ratings",
    "premium accommodation in colombo",
    "luxury stays worth splurging on",
    "upscale places to stay",
])
def test_premium_queries_set_price_preference_high(q):
    assert _intent(q).price_preference == "high"


@pytest.mark.parametrize("q", [
    "cheaper hotels in colombo",
    "budget friendly hotels near the city centre",
    "the most affordable places to stay",
    "wallet friendly accommodation in colombo",
    "good value stays in colombo",
    "economical hotels that are still well rated",
    "inexpensive rooms below 15000 a night",
])
def test_budget_queries_set_price_preference_low(q):
    assert _intent(q).price_preference == "low"


@pytest.mark.parametrize("q", [
    "hotels near galle face",
    "quiet hotels away from traffic",
    "hotels with a pool",
])
def test_neutral_queries_leave_price_preference_any(q):
    assert _intent(q).price_preference == "any"


def test_premium_is_checked_before_budget():
    """"luxury" is both a quality and a price signal; price must read it as high."""
    i = _intent("luxury stays worth splurging on")
    assert i.price_preference == "high"
    assert i.sort_intent == "highest_rated"


def test_upmarket_carries_no_rating_claim():
    """The gap this closed: 'upmarket' never matched any rule, so it fell through
    to best_overall with `economic` still pointing at the cheapest hotels."""
    i = _intent("upmarket hotels in colombo")
    assert i.price_preference == "high"
    assert i.sort_intent == "best_overall"


def test_price_preference_defaults_to_any():
    from src.crag.query_parser import QueryIntent
    assert QueryIntent().price_preference == "any"


def test_flip_is_symmetric_around_the_midpoint():
    """The retriever applies `economic = 1 - economic`; a median-priced hotel is
    unaffected, and the cheapest/dearest swap places."""
    for economic in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert (1.0 - (1.0 - economic)) == pytest.approx(economic)
    assert (1.0 - 0.5) == pytest.approx(0.5)
