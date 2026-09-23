"""The ranked price slice must remain balanced and directionally parseable."""
import json
from pathlib import Path

from src.crag.query_parser import _regex_intent


QUERYSET = Path(__file__).resolve().parents[1] / "evaluation" / "queryset_price.json"


def test_price_slice_has_balanced_cheap_and_premium_coverage():
    queries = json.loads(QUERYSET.read_text(encoding="utf-8"))["queries"]
    cheap = [query for query in queries if "prefer_cheaper" in query["gold"]]
    premium = [query for query in queries if "prefer_premium" in query["gold"]]
    assert len(queries) == 20
    assert len(cheap) == len(premium) == 10
    assert {query["id"] for query in queries} == {f"p{i:04d}" for i in range(1, 21)}


def test_price_slice_uses_current_frozen_tiers_and_correct_directions():
    queries = json.loads(QUERYSET.read_text(encoding="utf-8"))["queries"]
    for query in queries:
        intent = _regex_intent(query["question"], "Colombo")
        if "prefer_cheaper" in query["gold"]:
            assert query["gold"]["prefer_cheaper"] == {"full": 17874.78, "partial": 25817.06}
            assert intent.price_preference == "low"
        else:
            assert query["gold"]["prefer_premium"] == {"full": 31612.4, "partial": 19350.81}
            assert intent.price_preference == "high"