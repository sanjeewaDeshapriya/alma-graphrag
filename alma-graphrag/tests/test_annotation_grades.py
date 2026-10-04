"""Graded human judgements must survive aggregation and reach nDCG as gains.

The protocol asks annotators to separate "fully relevant" (2) from "reasonable
but flawed" (1). An earlier version binarised at aggregation and the harness
then gave every human-relevant hotel a gain of 2, so that distinction never
reached a metric. These tests keep it alive end to end.
"""
import pytest

from evaluation.annotation.agreement import aggregate_graded, consensus_grade
from evaluation.harness import gold_for_query

QUERY = {"id": "q1", "gold": {"max_price": 20000}}
POOL = [{"id": "h1", "price": 10000}, {"id": "h2", "price": 90000}]


def test_consensus_grade_is_the_lower_median():
    assert consensus_grade([2, 2, 2]) == 2
    assert consensus_grade([1, 2, 2]) == 2
    assert consensus_grade([1, 1, 2]) == 1
    assert consensus_grade([0, 1, 2]) == 1
    assert consensus_grade([1, 2]) == 1          # even split stays conservative
    assert consensus_grade([None, 2, None]) == 2
    assert consensus_grade([None, None]) is None


def test_graded_aggregation_keeps_partial_and_full_apart():
    labels = {"q1": {"full": [2, 2, 2], "partial": [1, 1, 2], "rejected": [0, 0, 1]}}
    graded = aggregate_graded(labels)["q1"]
    assert graded == {"full": 2, "partial": 1}   # rejected loses the majority vote


def test_a_hotel_one_annotator_loves_does_not_enter_on_that_alone():
    labels = {"q1": {"h": [0, 0, 2]}}
    assert aggregate_graded(labels)["q1"] == {}


def test_harness_uses_consensus_grades_as_gains():
    relevant = {"q1": ["h1", "h2"]}
    graded = {"q1": {"h1": 2, "h2": 1}}
    ids, gains, source = gold_for_query(POOL, QUERY, relevant, human_graded=graded)
    assert source == "human"
    assert ids == {"h1", "h2"}
    assert gains == {"h1": 2, "h2": 1}


def test_harness_falls_back_to_full_relevance_for_older_artifacts():
    """Artifacts written before grades were kept carry only the binary lists."""
    ids, gains, source = gold_for_query(POOL, QUERY, {"q1": ["h1", "h2"]})
    assert source == "human"
    assert gains == {"h1": 2, "h2": 2}


def test_rule_gold_still_applies_when_a_query_has_no_human_labels():
    ids, gains, source = gold_for_query(POOL, QUERY, {}, human_graded={})
    assert source == "rule"


def test_grades_change_the_ranking_metric():
    """The point of keeping them: nDCG must prefer the fully relevant hotel."""
    from evaluation.metrics import ndcg_at_k_graded
    gains = {"h1": 2, "h2": 1}
    assert ndcg_at_k_graded(["h1", "h2"], gains, 10) > ndcg_at_k_graded(["h2", "h1"], gains, 10)
    flat = {"h1": 2, "h2": 2}
    assert ndcg_at_k_graded(["h1", "h2"], flat, 10) == pytest.approx(
        ndcg_at_k_graded(["h2", "h1"], flat, 10))
