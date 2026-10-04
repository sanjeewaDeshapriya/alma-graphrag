"""Research-validity metadata must prevent overclaiming from weak evaluations."""
from evaluation.harness import assess_reportability


def test_reportability_requires_complete_human_gold_and_index_parity():
    result = assess_reportability(
        query_count=60,
        human_query_count=0,
        vector_index={"stale": True},
        by_category={"disruption": {"WeightedGraphRAG": {"nDCG@10": 1.0}}},
        sensitivity={"by_category": {"economic": {"fraction": 0.0, "order_fraction": 0.0}}},
        ndcg_key="nDCG@10",
    )
    assert result["reportable"] is False
    assert len(result["blockers"]) == 2
    assert "weight-blind categories: economic" in result["warnings"]
    assert "near-perfect reference categories: disruption" in result["warnings"]


def test_reportability_accepts_complete_human_gold_and_aligned_index():
    result = assess_reportability(
        query_count=2,
        human_query_count=2,
        vector_index={"stale": False},
        by_category={"mixed": {"WeightedGraphRAG": {"nDCG@10": 0.8}}},
        sensitivity={"by_category": {"mixed": {"fraction": 0.5, "order_fraction": 1.0}}},
        ndcg_key="nDCG@10",
    )
    assert result == {"reportable": True, "blockers": [], "warnings": []}