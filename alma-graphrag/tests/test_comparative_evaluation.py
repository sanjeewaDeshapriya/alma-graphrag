"""Rule and human-choice metrics must remain separate in comparative output."""
from evaluation.comparative import build_matrix


def test_comparative_matrix_aligns_handset_names_without_pooling_metrics():
    rule = {
        "n_queries": 60,
        "pool_size": 77,
        "gold_meta": {"used_human": False},
        "research_validity": {"reportable": False},
        "system_order": ["Filter", "WeightedGraphRAG"],
        "overall": {"Filter": {"nDCG@10": 0.5}, "WeightedGraphRAG": {"nDCG@10": 0.8}},
    }
    human = {
        "material_version": "v4",
        "n_test_participants": 28,
        "n_test_choices": 280,
        "min_votes": 2,
        "selection_share_gold": {"tasks": {}},
        "system_order": ["Filter", "WeightedGraphRAG[handset]"],
        "per_choice": {"Filter": {"nDCG@10": 0.2}, "WeightedGraphRAG[handset]": {"nDCG@10": 0.6}},
        "per_task": {"Filter": {"nDCG@10": 0.3}, "WeightedGraphRAG[handset]": {"nDCG@10": 0.7}},
    }

    matrix = build_matrix(rule, human)

    graph = next(row for row in matrix["matrix"] if row["system"] == "WeightedGraphRAG")
    assert graph["rule_based"]["nDCG@10"] == 0.8
    assert graph["human_choice_per_choice"]["nDCG@10"] == 0.6
    assert "Do not average" in matrix["note"]