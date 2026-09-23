"""
Inter-annotator agreement and label aggregation (pure functions, no I/O).

Krippendorff's alpha with the interval distance metric, which is appropriate
for the equally spaced ordinal relevance scale 0/1/2 used in the annotation
protocol (docs/annotation_protocol.md). Handles missing judgments (an
annotator may skip an item).

Interpretation (Krippendorff 2004): alpha >= 0.800 reliable;
0.667 <= alpha < 0.800 acceptable for tentative conclusions; below 0.667 the
labels should not be used without adjudication.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Set


def krippendorff_alpha(units: Sequence[Sequence[Optional[float]]]) -> float:
    """Alpha over `units`, one inner sequence of judgments per item.

    ``None`` marks a missing judgment. Items with fewer than two judgments are
    excluded (they carry no agreement information). Returns 1.0 when there is
    no observed *or* expected disagreement (degenerate single-value data).
    """
    # Keep only pairable values.
    pairable: List[List[float]] = []
    for unit in units:
        vals = [v for v in unit if v is not None]
        if len(vals) >= 2:
            pairable.append(vals)
    n_total = sum(len(vals) for vals in pairable)
    if n_total == 0:
        return 1.0

    def delta(a: float, b: float) -> float:
        return (a - b) ** 2  # interval metric

    # Observed disagreement.
    do = 0.0
    for vals in pairable:
        m = len(vals)
        do += sum(
            delta(vals[i], vals[j])
            for i in range(m)
            for j in range(m)
            if i != j
        ) / (m - 1)
    do /= n_total

    # Expected disagreement from the pooled value distribution.
    counts = Counter(v for vals in pairable for v in vals)
    de = 0.0
    for v1, c1 in counts.items():
        for v2, c2 in counts.items():
            de += c1 * c2 * delta(v1, v2)
    de /= n_total * (n_total - 1)

    if de == 0.0:
        return 1.0
    return 1.0 - do / de


def binarize(score: float, threshold: float = 1.0) -> int:
    """Graded 0/1/2 relevance -> binary (>= threshold is relevant)."""
    return 1 if score >= threshold else 0


def majority_relevant(
    judgments: Sequence[Optional[float]], threshold: float = 1.0
) -> Optional[bool]:
    """Majority vote over one item's judgments after binarization.

    Returns None when no judgments exist; ties count as NOT relevant
    (conservative — relevance requires positive evidence, matching
    evaluation/gold.py semantics).
    """
    votes = [binarize(v, threshold) for v in judgments if v is not None]
    if not votes:
        return None
    return sum(votes) > len(votes) / 2


def aggregate_gold(
    labels: Dict[str, Dict[str, List[Optional[float]]]], threshold: float = 1.0
) -> Dict[str, Set[str]]:
    """{query_id: {hotel_id: [judgments...]}} -> {query_id: relevant hotel ids}."""
    out: Dict[str, Set[str]] = {}
    for qid, per_hotel in labels.items():
        out[qid] = {
            hid
            for hid, judgments in per_hotel.items()
            if majority_relevant(judgments, threshold)
        }
    return out


def consensus_grade(judgments: Sequence[Optional[float]]) -> Optional[int]:
    """One 0/1/2 grade for an item: the lower median of its judgments.

    The protocol asks annotators to separate "fully relevant" (2) from
    "reasonable but flawed" (1), and nDCG is built to use that difference. An
    earlier version threw it away — it binarised here and the harness then gave
    every relevant hotel a gain of 2, so a hotel most annotators called partial
    counted the same as a perfect answer. The lower median keeps the
    distinction and stays conservative when an even number of annotators splits.
    """
    scored = sorted(v for v in judgments if v is not None)
    if not scored:
        return None
    return int(scored[(len(scored) - 1) // 2])


def aggregate_graded(
    labels: Dict[str, Dict[str, List[Optional[float]]]], threshold: float = 1.0
) -> Dict[str, Dict[str, int]]:
    """-> {query_id: {hotel_id: consensus grade}} for hotels judged relevant.

    Relevance is still the majority vote (a hotel two of three annotators call
    irrelevant does not enter the gold on the strength of one high grade); the
    grade that survives is the consensus one, and it becomes the nDCG gain.
    """
    out: Dict[str, Dict[str, int]] = {}
    for qid, per_hotel in labels.items():
        graded: Dict[str, int] = {}
        for hid, judgments in per_hotel.items():
            if not majority_relevant(judgments, threshold):
                continue
            grade = consensus_grade(judgments)
            if grade is not None and grade > 0:
                graded[hid] = grade
        out[qid] = graded
    return out
