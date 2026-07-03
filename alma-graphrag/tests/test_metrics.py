from evaluation.metrics import (
    dcg_at_k,
    evaluate_ranking,
    mean_metrics,
    ndcg_at_k,
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
