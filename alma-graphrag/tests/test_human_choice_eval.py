"""Share-graded metrics of evaluation/human_choice_eval.py (no data files needed)."""
import collections

import pytest

from evaluation.human_choice_eval import task_metrics


def test_share_graded_metrics():
    # 10 choices: A 5, B 3, C 2  ->  shares 0.5, 0.3, 0.2
    votes = collections.Counter({"A": 5, "B": 3, "C": 2})
    ranked = ["B", "X", "A"] + [f"n{i}" for i in range(20)] + ["C"]
    m = task_metrics(ranked, votes, k=10)
    assert m["P@10"] == pytest.approx(2 / 10)           # B and A in the top 10
    assert m["R@10"] == pytest.approx(2 / 3)
    assert m["coverage@10"] == pytest.approx(0.8)       # 50% + 30% of the choices
    assert m["MRR"] == 1.0 and m["top1_accuracy"] == 1.0
    assert m["top1_share"] == pytest.approx(0.3)
    assert m["favourite_rank"] == 3.0                   # A, the most chosen, is third


def test_perfect_order_scores_one_and_empty_scores_zero():
    votes = collections.Counter({"A": 5, "B": 3, "C": 2})
    assert task_metrics(["A", "B", "C", "D"], votes, k=10)["nDCG@10"] == pytest.approx(1.0)
    none = task_metrics([f"n{i}" for i in range(12)] + ["A", "B", "C"], votes, k=10)
    assert none["nDCG@10"] == 0.0 and none["coverage@10"] == 0.0
