import pytest

from evaluation.annotation.agreement import (
    aggregate_gold,
    binarize,
    krippendorff_alpha,
    majority_relevant,
)


def test_alpha_perfect_agreement():
    units = [[2, 2, 2], [0, 0, 0], [1, 1, 1]]
    assert krippendorff_alpha(units) == pytest.approx(1.0)


def test_alpha_degenerate_single_value():
    # Everyone says 1 for everything: no expected disagreement either.
    assert krippendorff_alpha([[1, 1], [1, 1]]) == pytest.approx(1.0)


def test_alpha_systematic_disagreement_is_low():
    units = [[0, 2], [2, 0], [0, 2], [2, 0]]
    assert krippendorff_alpha(units) < 0.0  # worse than chance


def test_alpha_mixed_agreement_between_bounds():
    units = [[2, 2, 2], [0, 0, 1], [1, 1, 2], [0, 0, 0]]
    a = krippendorff_alpha(units)
    assert 0.0 < a < 1.0


def test_alpha_ignores_missing_and_single_judgments():
    with_missing = [[2, 2, None], [0, None, 0], [1]]  # third unit unpairable
    assert krippendorff_alpha(with_missing) == pytest.approx(1.0)


def test_alpha_empty_input():
    assert krippendorff_alpha([]) == 1.0
    assert krippendorff_alpha([[None, None], [2]]) == 1.0


def test_binarize_threshold():
    assert binarize(0) == 0
    assert binarize(1) == 1
    assert binarize(2) == 1
    assert binarize(1, threshold=2) == 0


def test_majority_relevant():
    assert majority_relevant([2, 1, 0]) is True      # 2 of 3 votes relevant
    assert majority_relevant([0, 0, 2]) is False
    assert majority_relevant([1, 0]) is False        # tie -> conservative
    assert majority_relevant([None, None]) is None
    assert majority_relevant([None, 2]) is True


def test_aggregate_gold():
    labels = {
        "q1": {"h1": [2, 2, 1], "h2": [0, 0, 1], "h3": [None, 2, 2]},
        "q2": {"h1": [0, 0, 0]},
    }
    gold = aggregate_gold(labels)
    assert gold["q1"] == {"h1", "h3"}
    assert gold["q2"] == set()
