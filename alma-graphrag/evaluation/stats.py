"""
Statistical significance testing for the comparative evaluation.

Every system difference reported in results.json is a mean over per-query
scores, so the right tests are *paired* across queries:

  - paired bootstrap 95% CI of the mean difference (reference - system)
  - Wilcoxon signed-rank test on the per-query score pairs
  - Holm-Bonferroni correction across the systems compared against the
    reference (controls family-wise error over multiple comparisons)

`compare_systems` is the entry point used by evaluation/harness.py; it returns
a JSON-serialisable block stored under results["significance"].
"""
from __future__ import annotations

import random
from typing import Dict, List, Sequence

N_BOOTSTRAP = 10_000
CONFIDENCE = 0.95
SEED = 17


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = N_BOOTSTRAP,
    confidence: float = CONFIDENCE,
    seed: int = SEED,
) -> Dict[str, float]:
    """CI of mean(a) - mean(b) by resampling query indices with replacement."""
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be equal-length and non-empty")
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)
    )
    alpha = (1.0 - confidence) / 2.0
    lo = means[int(alpha * n_boot)]
    hi = means[min(int((1.0 - alpha) * n_boot), n_boot - 1)]
    return {"mean_diff": sum(diffs) / n, "ci_low": lo, "ci_high": hi}


def wilcoxon_p(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank p-value on paired scores.

    All-zero differences (systems identical on every query) carry no evidence
    against the null — return p=1.0 rather than letting scipy raise.
    """
    diffs = [x - y for x, y in zip(a, b)]
    if not any(d != 0 for d in diffs):
        return 1.0
    from scipy.stats import wilcoxon
    # zero_method="wilcox" drops zero-diff pairs (the classic treatment).
    return float(wilcoxon(a, b, zero_method="wilcox").pvalue)


def holm_correction(pvalues: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni step-down adjustment; preserves input keys."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: Dict[str, float] = {}
    running_max = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running_max = max(running_max, adj)  # enforce monotonicity
        adjusted[name] = running_max
    return adjusted


def compare_systems(
    per_query: Dict[str, List[float]],
    reference: str,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """Compare every system against `reference` on paired per-query scores.

    Returns {system: {mean_diff, ci_low, ci_high, p, p_holm, significant}}
    where mean_diff = mean(reference) - mean(system), so positive means the
    reference system is better.
    """
    ref = per_query[reference]
    raw_p: Dict[str, float] = {}
    out: Dict[str, Dict[str, float]] = {}
    for name, scores in per_query.items():
        if name == reference:
            continue
        boot = paired_bootstrap_ci(ref, scores)
        raw_p[name] = wilcoxon_p(ref, scores)
        out[name] = {**boot, "p": raw_p[name]}
    adjusted = holm_correction(raw_p)
    for name, block in out.items():
        block["p_holm"] = adjusted[name]
        block["significant"] = bool(adjusted[name] < alpha)
    return out
