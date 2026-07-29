import math

from evaluation.metrics import (
    dcg_at_k,
    evaluate_ranking,
    mean_metrics,
    ndcg_at_k,
    ndcg_at_k_graded,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


RANKED = ["a", "b", "c", "d", "e"]


def test_perfect_ranking_scores_one():
    relevant = {"a", "b", "c"}
    assert precision_at_k(RANKED, relevant, 3) == 1.0
    assert recall_at_k(RANKED, relevant, 3) == 1.0
    assert ndcg_at_k(RANKED, relevant, 3) == 1.0
    assert reciprocal_rank(RANKED, relevant) == 1.0


def test_no_relevant_items_scores_zero():
    assert precision_at_k(RANKED, {"x"}, 5) == 0.0
    assert recall_at_k(RANKED, {"x"}, 5) == 0.0
    assert ndcg_at_k(RANKED, {"x"}, 5) == 0.0
    assert reciprocal_rank(RANKED, {"x"}) == 0.0


def test_empty_relevant_set_is_zero_not_error():
    assert recall_at_k(RANKED, set(), 5) == 0.0
    assert ndcg_at_k(RANKED, set(), 5) == 0.0


def test_partial_hits():
    relevant = {"b", "d"}
    assert precision_at_k(RANKED, relevant, 2) == 0.5
    assert recall_at_k(RANKED, relevant, 2) == 0.5
    assert reciprocal_rank(RANKED, relevant) == 0.5  # first hit at rank 2


def test_k_zero_and_empty_ranking():
    assert precision_at_k(RANKED, {"a"}, 0) == 0.0
    assert precision_at_k([], {"a"}, 5) == 0.0


def test_ndcg_rewards_earlier_hits():
    relevant = {"a"}
    early = ndcg_at_k(["a", "b", "c"], relevant, 3)
    late = ndcg_at_k(["b", "c", "a"], relevant, 3)
    assert early > late > 0.0


def test_dcg_monotonic_in_hits():
    assert dcg_at_k(["a", "b"], {"a", "b"}, 2) > dcg_at_k(["a", "x"], {"a", "b"}, 2)


def test_evaluate_ranking_keys():
    row = evaluate_ranking(RANKED, {"a"}, 3)
    assert set(row) == {"P@3", "R@3", "nDCG@3", "MRR"}


def test_mean_metrics():
    rows = [{"m": 1.0}, {"m": 0.0}]
    assert mean_metrics(rows) == {"m": 0.5}
    assert mean_metrics([]) == {}


# --- graded nDCG -----------------------------------------------------------

def test_graded_ndcg_perfect_order_is_one():
    gains = {"a": 2, "b": 1}
    assert ndcg_at_k_graded(["a", "b", "x"], gains, 3) == 1.0


def test_graded_ndcg_prefers_full_over_partial_first():
    gains = {"full": 2, "part": 1}
    good = ndcg_at_k_graded(["full", "part"], gains, 2)
    swapped = ndcg_at_k_graded(["part", "full"], gains, 2)
    assert good == 1.0
    assert swapped < good
    # binary nDCG cannot see the difference — that's the point of grading
    assert ndcg_at_k(["part", "full"], {"full", "part"}, 2) == 1.0


def test_graded_ndcg_empty_gains_is_zero():
    assert ndcg_at_k_graded(["a"], {}, 3) == 0.0


def test_graded_ndcg_hand_computed():
    # ranked: [a(2), x(0), b(1)]; dcg = 2/log2(2) + 1/log2(4) = 2 + 0.5
    # ideal: [2, 1] -> idcg = 2/log2(2) + 1/log2(3)
    gains = {"a": 2, "b": 1}
    expected = (2.0 + 1.0 / math.log2(4)) / (2.0 + 1.0 / math.log2(3))
    assert abs(ndcg_at_k_graded(["a", "x", "b"], gains, 3) - expected) < 1e-12


def test_evaluate_ranking_uses_gains_for_ndcg_only():
    gains = {"a": 2, "b": 1}
    row = evaluate_ranking(["b", "a"], {"a", "b"}, 2, gains=gains)
    assert row["P@2"] == 1.0          # binary: both relevant
    assert row["nDCG@2"] < 1.0        # graded: wrong order penalised
