import pytest

from evaluation.stats import (
    compare_systems,
    holm_correction,
    paired_bootstrap_ci,
    wilcoxon_p,
)


def test_bootstrap_ci_contains_true_diff():
    a = [0.9, 0.8, 0.85, 0.95, 0.9, 0.88, 0.92, 0.87]
    b = [0.5, 0.4, 0.45, 0.55, 0.5, 0.48, 0.52, 0.47]
    out = paired_bootstrap_ci(a, b, n_boot=2000)
    assert out["ci_low"] <= out["mean_diff"] <= out["ci_high"]
    assert out["mean_diff"] == pytest.approx(0.4, abs=1e-9)
    assert out["ci_low"] > 0  # clearly separated systems -> CI excludes 0


def test_bootstrap_ci_identical_systems_centres_on_zero():
    a = [0.5, 0.6, 0.7, 0.8]
    out = paired_bootstrap_ci(a, a, n_boot=500)
    assert out["mean_diff"] == 0.0
    assert out["ci_low"] == 0.0 and out["ci_high"] == 0.0


def test_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0], [1.0, 2.0])


def test_wilcoxon_identical_is_one():
    a = [0.5, 0.6, 0.7]
    assert wilcoxon_p(a, a) == 1.0


def test_wilcoxon_separated_is_small():
    a = [0.9] * 10
    b = [0.1] * 10
    assert wilcoxon_p(a, b) < 0.05


def test_holm_correction_monotone_and_bounded():
    p = {"x": 0.01, "y": 0.04, "z": 0.9}
    adj = holm_correction(p)
    assert adj["x"] == pytest.approx(0.03)   # 3 * 0.01
    assert adj["y"] == pytest.approx(0.08)   # 2 * 0.04
    assert adj["z"] == pytest.approx(0.9)    # 1 * 0.9
    assert all(0.0 <= v <= 1.0 for v in adj.values())
    # adjusted p-values never below raw
    assert all(adj[k] >= p[k] for k in p)


def test_compare_systems_shape_and_direction():
    per_query = {
        "Ref":  [0.9, 0.85, 0.95, 0.9, 0.88, 0.92, 0.87, 0.91],
        "Weak": [0.3, 0.25, 0.35, 0.3, 0.28, 0.32, 0.27, 0.31],
        "Tied": [0.9, 0.85, 0.95, 0.9, 0.88, 0.92, 0.87, 0.91],
    }
    out = compare_systems(per_query, "Ref")
    assert set(out) == {"Weak", "Tied"}
    weak = out["Weak"]
    assert weak["mean_diff"] > 0          # reference better
    assert weak["significant"] is True
    tied = out["Tied"]
    assert tied["mean_diff"] == 0.0
    assert tied["p"] == 1.0
    assert tied["significant"] is False
