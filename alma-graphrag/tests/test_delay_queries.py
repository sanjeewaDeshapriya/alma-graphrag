"""Delay-graded query set (evaluation/build_delay_queries.py)."""
import json
from pathlib import Path

from evaluation.build_delay_queries import THRESHOLDS, build_queries
from src.crag.query_parser import parse_query

QUERYSET = Path(__file__).resolve().parents[1] / "evaluation" / "queryset_disruption.json"


def test_every_query_is_graded_on_added_delay():
    queries = build_queries()
    assert len({q["id"] for q in queries}) == len(queries)
    assert all("max_added_delay_min" in q["gold"] for q in queries)
    assert all("max_travel_time" not in q["gold"] for q in queries)
    assert {q["gold"]["max_added_delay_min"] for q in queries} == {float(t) for t in THRESHOLDS}


def test_regex_parser_sets_avoid_traffic_and_reads_no_false_price():
    # conftest disables the LLM client, so this is the deterministic regex pass.
    for q in build_queries():
        intent = parse_query(q["question"], default_city="Colombo")
        assert intent.avoid_traffic, q["question"]
        assert intent.max_price_lkr == q["gold"].get("max_price"), q["question"]
        assert intent.min_rating == q["gold"].get("min_rating"), q["question"]


def test_committed_file_matches_the_builder():
    spec = json.loads(QUERYSET.read_text(encoding="utf-8"))
    assert spec["queries"] == build_queries()
