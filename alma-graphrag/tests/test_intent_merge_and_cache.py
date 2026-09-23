"""LLM/regex intent merge rules, and frozen intents for reproducible evaluation.

The merge bug these tests pin down was measured on 2026-09-14: gemini-2.5-flash
answered price_preference="any" for "premium accommodation in colombo", the merge
let the LLM win, and the regex's correct "high" was erased.
"""
from __future__ import annotations

import json

import pytest

from evaluation.harness import resolve_intents
from src.crag import query_parser as qp


def _with_llm(monkeypatch, payload):
    monkeypatch.setattr(qp, "_llm_intent", lambda question: payload)


def test_neutral_llm_value_does_not_erase_a_specific_regex_value(monkeypatch):
    _with_llm(monkeypatch, {"price_preference": "any", "sort_intent": "best_overall",
                            "proximity_preference": "any", "accessibility_priority": "normal",
                            "avoid_traffic": False})
    intent = qp.parse_query("premium quiet hotels with easy access, cheapest first", "Colombo")
    regex = qp._regex_intent("premium quiet hotels with easy access, cheapest first", "Colombo")
    assert intent.price_preference == regex.price_preference != "any"
    assert intent.proximity_preference == regex.proximity_preference
    assert intent.avoid_traffic == regex.avoid_traffic
    assert intent.sort_intent == regex.sort_intent


def test_specific_llm_value_still_overrides(monkeypatch):
    _with_llm(monkeypatch, {"price_preference": "low", "sort_intent": "highest_rated",
                            "proximity_preference": "close", "avoid_traffic": True})
    intent = qp.parse_query("hotels in colombo", "Colombo")
    assert intent.price_preference == "low"
    assert intent.sort_intent == "highest_rated"
    assert intent.proximity_preference == "close"
    assert intent.avoid_traffic is True


def test_neutral_llm_value_applies_when_regex_was_also_neutral(monkeypatch):
    _with_llm(monkeypatch, {"price_preference": "any"})
    intent = qp.parse_query("hotels in colombo", "Colombo")
    assert intent.price_preference == "any"


def test_intent_cache_freezes_the_parse_across_runs(monkeypatch, tmp_path):
    calls = {"n": 0}
    real = qp.parse_query

    def counting(question, default_city=None):
        calls["n"] += 1
        return real(question, default_city)

    monkeypatch.setattr("evaluation.harness.parse_query", counting)
    queries = [{"id": "q1", "question": "cheap hotels"},
               {"id": "q2", "question": "luxury stays"}]
    cache = tmp_path / "intents.json"

    first = resolve_intents(queries, "Colombo", cache)
    assert calls["n"] == 2 and cache.exists()

    # Tamper with the frozen file: the next run must read it, not re-parse.
    data = json.loads(cache.read_text(encoding="utf-8"))
    data["intents"]["Colombo::cheap hotels"]["sort_intent"] = "most_accessible"
    cache.write_text(json.dumps(data), encoding="utf-8")

    second = resolve_intents(queries, "Colombo", cache)
    assert calls["n"] == 2
    assert second["q1"].sort_intent == "most_accessible"
    assert second["q2"].to_dict() == first["q2"].to_dict()

    resolve_intents(queries + [{"id": "q3", "question": "hotels near fort"}], "Colombo", cache)
    assert calls["n"] == 3
    assert "Colombo::hotels near fort" in json.loads(cache.read_text(encoding="utf-8"))["intents"]
