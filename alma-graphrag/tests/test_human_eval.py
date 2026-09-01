"""Tests for the choice-based evaluation (evaluation/human_eval.py).

Covers the pure logic — cohort selection, ranking utilities, and the two metric
views — without touching the embedder, pgvector or Neo4j, so the suite stays
offline and fast.
"""
import collections

import numpy as np
import pytest

from evaluation.human_eval import (STALE_PERSONAS, _bm25, _complete, _per_choice,
                                   _per_task, _rrf, select_cohort)


# ---------------------------------------------------------------------------
# Cohort selection
# ---------------------------------------------------------------------------

def _rows(*people):
    """people: (participant_id, attention_pass, decision_ms) -> response rows."""
    out = []
    for pid, passed, ms in people:
        for task in ("t1", "t2"):
            out.append({"participant_id": pid, "task_id": task,
                        "decision_ms": str(ms), "attention_pass": "",
                        "is_attention_check": "false", "scenario_persona": "x",
                        "chosen_hotel_id": "h1"})
        out.append({"participant_id": pid, "task_id": "t_attn",
                    "decision_ms": str(ms), "attention_pass": str(passed).lower(),
                    "is_attention_check": "true", "scenario_persona": "attention",
                    "chosen_hotel_id": "h1"})
    return out


ROWS = _rows(("p_good", True, 20000), ("p_fail", False, 20000),
             ("p_fast", True, 1000), ("p_bad", False, 900))


def test_cohort_all_keeps_everyone():
    keep, stats = select_cohort(ROWS, "all")
    assert keep == {"p_good", "p_fail", "p_fast", "p_bad"}
    assert stats["participants_kept"] == 4


def test_cohort_attention_drops_failures_only():
    keep, _ = select_cohort(ROWS, "attention")
    assert keep == {"p_good", "p_fast"}      # speeders still allowed


def test_cohort_clean_drops_failures_and_speeders():
    keep, stats = select_cohort(ROWS, "clean", min_decision_ms=5000.0)
    assert keep == {"p_good"}
    assert stats["dropped_attention"] == 2   # p_fail, p_bad
    assert stats["dropped_speeder"] == 1     # p_fast


def test_cohort_stats_always_report_the_totals():
    """The filter must be visible in the output, never silently applied."""
    _keep, stats = select_cohort(ROWS, "clean")
    assert stats["participants_total"] == 4
    assert stats["participants_kept"] + stats["dropped_attention"] + \
        stats["dropped_speeder"] == 4


def test_participant_with_no_attention_row_is_not_clean():
    rows = [{"participant_id": "p", "task_id": "t1", "decision_ms": "20000",
             "attention_pass": "", "is_attention_check": "false",
             "scenario_persona": "x", "chosen_hotel_id": "h1"}]
    assert select_cohort(rows, "clean")[0] == set()
    assert select_cohort(rows, "all")[0] == {"p"}


# ---------------------------------------------------------------------------
# Ranking utilities
# ---------------------------------------------------------------------------

def test_complete_preserves_order_and_appends_the_rest():
    """Every system must return the full candidate set.

    A system returning a short list would otherwise get an easier denominator
    than one returning everything — the bug found in the rule-based harness.
    """
    ids = ["a", "b", "c", "d"]
    assert _complete(["c", "a"], ids) == ["c", "a", "b", "d"]


def test_complete_returns_every_candidate_exactly_once():
    ids = ["a", "b", "c"]
    out = _complete(["b"], ids)
    assert sorted(out) == sorted(ids)
    assert len(out) == len(set(out))


def test_complete_ignores_ids_outside_the_candidate_set():
    assert _complete(["zzz", "a"], ["a", "b"]) == ["a", "b"]


def test_bm25_ranks_the_matching_document_first():
    docs = ["quiet garden hotel near the lake", "busy city centre business tower",
            "beach resort with pool"]
    assert int(np.argmax(_bm25(docs, "quiet garden lake"))) == 0
    assert int(np.argmax(_bm25(docs, "beach pool resort"))) == 2


def test_bm25_scores_zero_when_nothing_matches():
    assert _bm25(["alpha beta", "gamma"], "nothing here").sum() == pytest.approx(0.0)


def test_rrf_rewards_agreement_between_rankings():
    """An item both lists rank highly must beat one only a single list likes."""
    fused = _rrf(["a", "b", "c"], ["a", "c", "b"])
    assert fused[0] == "a"
    assert set(fused) == {"a", "b", "c"}


def test_rrf_verdict_does_not_depend_on_argument_order():
    """Which list is passed first must not change a non-tied outcome.

    (Exact ties DO fall back to insertion order, because `sorted` is stable —
    that is deliberate and deterministic, not a property worth asserting.)
    """
    sem, kw = ["x", "a", "b", "c"], ["x", "c", "b", "a"]
    assert _rrf(sem, kw)[0] == "x"
    assert _rrf(kw, sem)[0] == "x"


def test_rrf_beats_a_single_list_top_hit_when_the_other_list_disagrees():
    """Agreement across lists is what RRF rewards — the point of fusing."""
    fused = _rrf(["solo", "agreed", "z"], ["agreed", "z", "solo"])
    assert fused.index("agreed") < fused.index("solo")


# ---------------------------------------------------------------------------
# Metric views
# ---------------------------------------------------------------------------

RANKED = {"t1": ["h1", "h2", "h3", "h4", "h5"]}


def test_per_choice_perfect_ranking():
    m = _per_choice(RANKED, [("p", "t1", "h1")], k=5)
    assert m["R@5"] == pytest.approx(1.0)     # single relevant item -> hit rate
    assert m["MRR"] == pytest.approx(1.0)
    assert m["mean_rank"] == pytest.approx(1.0)
    assert m["P@5"] == pytest.approx(0.2)     # capped at 1/k with one relevant


def test_per_choice_precision_is_capped_by_one_relevant_item():
    """With a single relevant item, P@k can never exceed 1/k — document it."""
    for k in (2, 5):
        m = _per_choice(RANKED, [("p", "t1", "h1")], k=k)
        assert m[f"P@{k}"] == pytest.approx(1.0 / k)


def test_per_choice_miss_outside_k_scores_zero_but_keeps_the_rank():
    m = _per_choice(RANKED, [("p", "t1", "h5")], k=2)
    assert m["R@2"] == pytest.approx(0.0)
    assert m["nDCG@2"] == pytest.approx(0.0)
    assert m["mean_rank"] == pytest.approx(5.0)   # rank is over the full list


def test_per_choice_averages_over_observations():
    obs = [("p1", "t1", "h1"), ("p2", "t1", "h3")]
    m = _per_choice(RANKED, obs, k=5)
    assert m["mean_rank"] == pytest.approx(2.0)   # (1 + 3) / 2
    assert m["MRR"] == pytest.approx((1.0 + 1 / 3) / 2)


def test_per_task_uses_the_vote_threshold():
    votes = {"t1": collections.Counter({"h1": 3, "h2": 1})}
    m = _per_task(RANKED, votes, k=5, min_votes=2)
    assert m["R@5"] == pytest.approx(1.0)         # gold = {h1} only
    m_all = _per_task(RANKED, votes, k=5, min_votes=1)
    assert m_all["R@5"] == pytest.approx(1.0)     # gold = {h1, h2}, both in top-5
    assert m_all["P@5"] == pytest.approx(0.4)     # 2 hits / 5


def test_per_task_skips_tasks_with_an_empty_gold_set():
    votes = {"t1": collections.Counter({"h1": 3}), "t2": collections.Counter({"x": 1})}
    ranked = {"t1": RANKED["t1"], "t2": ["a", "b"]}
    m = _per_task(ranked, votes, k=5, min_votes=2)
    assert m["MRR"] == pytest.approx(1.0)         # t2 excluded, not scored as 0


def test_per_task_returns_empty_when_no_task_has_gold():
    votes = {"t1": collections.Counter({"h1": 1})}
    assert _per_task(RANKED, votes, k=5, min_votes=5) == {}


def test_graded_ndcg_rewards_ranking_the_popular_hotel_higher():
    """Vote count is the gain, so the crowd favourite belongs at the top."""
    votes = {"t1": collections.Counter({"h1": 10, "h2": 2})}
    good = _per_task({"t1": ["h1", "h2", "h3"]}, votes, k=3, min_votes=2)
    bad = _per_task({"t1": ["h2", "h1", "h3"]}, votes, k=3, min_votes=2)
    assert good["gradedNDCG@3"] > bad["gradedNDCG@3"]
    # Binary nDCG cannot tell them apart — which is why the graded view exists.
    assert good["nDCG@3"] == pytest.approx(bad["nDCG@3"])


def test_stale_personas_are_declared():
    """Rows from the superseded material version must stay excluded."""
    assert "quiet_seeker" in STALE_PERSONAS
    assert "budget_backpacker" in STALE_PERSONAS
