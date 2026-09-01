"""Tests that one parsed intent is shared by every baseline.

Why this needs a test rather than a code review: `parse_query` runs an LLM
slot-fill stage whose output is not deterministic. If a baseline quietly
re-parses instead of using the intent it was handed, the bug is invisible in
normal runs — the two parses usually agree — and shows up only as unexplained
variance between otherwise identical evaluation runs. The test makes the
re-parse itself the failure.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

import evaluation.baselines as bl
from src.crag.query_parser import QueryIntent


POOL: List[Dict[str, Any]] = [
    {"id": "a", "name": "Alpha", "rating": 4.8, "star": 5, "price": 45000.0,
     "amenities": ["pool"], "attractions": [], "description": "Alpha hotel",
     "distance_km": 1.0, "travel_time_min": 10.0,
     "travel_time_traffic_min": 11.0, "max_eta_change_min": 0.0,
     "event_impact": 0.0},
    {"id": "b", "name": "Beta", "rating": 4.1, "star": 3, "price": 8000.0,
     "amenities": ["wifi"], "attractions": [], "description": "Beta hotel",
     "distance_km": 4.0, "travel_time_min": 25.0,
     "travel_time_traffic_min": 30.0, "max_eta_change_min": 0.0,
     "event_impact": 0.0},
]


@pytest.fixture
def no_db(monkeypatch):
    monkeypatch.setattr(bl, "fetch_city_hotels", lambda city: [dict(h) for h in POOL])


@pytest.fixture
def parse_spy(monkeypatch):
    """Make any call to parse_query explode, so a re-parse cannot hide."""
    calls: List[str] = []

    def boom(question, default_city=None):
        calls.append(question)
        raise AssertionError(
            "parse_query was called even though a shared intent was supplied"
        )

    monkeypatch.setattr(bl, "parse_query", boom)
    return calls


# ---------------------------------------------------------------------------
# The shared intent is used, not re-derived
# ---------------------------------------------------------------------------

def test_filter_uses_supplied_intent_without_reparsing(no_db, parse_spy):
    intent = QueryIntent(city="Colombo", max_price_lkr=10000.0)
    got = bl.FilterBaseline().retrieve("anything at all", "Colombo", 10, intent=intent)
    assert got == ["b"]          # only Beta is under 10,000
    assert parse_spy == []       # and no parse happened


def test_graph_baseline_passes_intent_through(no_db, parse_spy, monkeypatch):
    seen = {}

    class FakeRetriever:
        def __init__(self, *a, **kw):
            pass

        def retrieve(self, intent, limit=10, **kw):
            seen["intent"] = intent
            return type("R", (), {"hotels": []})()

    monkeypatch.setattr(bl, "WeightedRetriever", FakeRetriever)
    intent = QueryIntent(city="Colombo", sort_intent="cheapest")
    bl.WeightedGraphBaseline().retrieve("q", "Colombo", 10, intent=intent)
    assert seen["intent"] is intent


def test_baseline_still_parses_when_no_intent_supplied(no_db, monkeypatch):
    """Standalone use must keep working — `intent=None` falls back to parsing."""
    calls: List[str] = []

    def fake_parse(question, default_city=None):
        calls.append(question)
        return QueryIntent(city=default_city, max_price_lkr=10000.0)

    monkeypatch.setattr(bl, "parse_query", fake_parse)
    got = bl.FilterBaseline().retrieve("cheap hotel", "Colombo", 10)
    assert calls == ["cheap hotel"]
    assert got == ["b"]


def test_resolve_intent_fills_missing_city(monkeypatch):
    monkeypatch.setattr(bl, "parse_query",
                        lambda q, default_city=None: QueryIntent(city=None))
    assert bl._resolve_intent("q", "Colombo", None).city == "Colombo"


def test_resolve_intent_returns_the_same_object(monkeypatch):
    intent = QueryIntent(city="Kandy")
    assert bl._resolve_intent("q", "Colombo", intent) is intent


# ---------------------------------------------------------------------------
# Every baseline accepts the intent keyword
# ---------------------------------------------------------------------------

def test_all_baselines_accept_an_intent_keyword():
    """A baseline missing the parameter would raise at harness call time."""
    import inspect

    classes = [bl.RandomBaseline, bl.PopularityBaseline, bl.FilterBaseline,
               bl.KeywordBaseline, bl.SemanticBaseline, bl.HybridBaseline,
               bl.CrossEncoderBaseline, bl.LTRBaseline, bl.WeightedGraphBaseline,
               bl.LLMRerankerBaseline]
    for cls in classes:
        params = inspect.signature(cls.retrieve).parameters
        assert "intent" in params, f"{cls.__name__}.retrieve has no intent parameter"


def test_ltr_declares_it_wants_the_query_id():
    assert getattr(bl.LTRBaseline, "wants_query_id", False) is True


def test_other_baselines_do_not_want_the_query_id():
    for cls in (bl.FilterBaseline, bl.WeightedGraphBaseline, bl.RandomBaseline):
        assert getattr(cls, "wants_query_id", False) is False


# ---------------------------------------------------------------------------
# Floors behave as documented
# ---------------------------------------------------------------------------

def test_random_baseline_is_seeded_per_question(no_db):
    r = bl.RandomBaseline()
    assert r.retrieve("same question", "Colombo", 2) == \
           r.retrieve("same question", "Colombo", 2)


def test_popularity_ignores_the_query(no_db):
    p = bl.PopularityBaseline()
    assert (p.retrieve("cheap hotel under 5000", "Colombo", 2)
            == p.retrieve("luxury 5-star spa", "Colombo", 2) == ["a", "b"])


def test_ltr_returns_nothing_when_unfitted(no_db):
    """Refusing to rank is correct: any ranking would come from a model that
    trained on this query's gold."""
    assert bl.LTRBaseline().retrieve("q", "Colombo", 10, query_id="q01") == []


# ---------------------------------------------------------------------------
# Verbalisation feeds the text baselines the numbers they need
# ---------------------------------------------------------------------------

def test_verbalise_includes_numeric_attributes():
    text = bl._verbalise(POOL[0])
    assert "45000 LKR" in text
    assert "4.8/5" in text
    assert "5-star" in text


def test_verbalise_marks_missing_price_explicitly():
    assert "price not listed" in bl._verbalise({"name": "X", "price": None})
