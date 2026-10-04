"""Present rule-based and held-out human-choice evaluation side by side."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULE_RESULTS = ROOT / "evaluation" / "results_final_77hotels.json"
DEFAULT_HUMAN_RESULTS = ROOT / "evaluation" / "results_human.json"
DEFAULT_COMPARATIVE_RESULTS = ROOT / "evaluation" / "results_comparative.json"


def build_matrix(rule_results: Dict[str, Any], human_results: Dict[str, Any]) -> Dict[str, Any]:
    """Keep separate gold sources in adjacent columns without pooling metrics."""
    rule_metrics = rule_results["overall"]
    human_choice = human_results["per_choice"]
    human_task = human_results["per_task"]
    system_order = list(rule_results["system_order"])
    for system in human_results["system_order"]:
        if system not in system_order:
            system_order.append(system)

    rows = []
    for system in system_order:
        human_system = "WeightedGraphRAG[handset]" if system == "WeightedGraphRAG" else system
        rows.append({
            "system": system,
            "rule_based": rule_metrics.get(system),
            "human_choice_per_choice": human_choice.get(human_system),
            "human_choice_per_task": human_task.get(human_system),
        })
    return {
        "note": (
            "Parallel evaluations only: rule-based relevance uses the benchmark query set; "
            "human-choice metrics use held-out study selections. Do not average or test them together."
        ),
        "rule_based": {
            "query_count": rule_results["n_queries"],
            "pool_size": rule_results["pool_size"],
            "gold_meta": rule_results["gold_meta"],
            "research_validity": rule_results.get("research_validity", {}),
        },
        "human_choice": {
            "material_version": human_results["material_version"],
            "test_participants": human_results["n_test_participants"],
            "test_choices": human_results["n_test_choices"],
            "min_votes": human_results["min_votes"],
            "selection_share_gold": human_results["selection_share_gold"],
        },
        "matrix": rows,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule-results", type=Path, default=DEFAULT_RULE_RESULTS)
    parser.add_argument("--human-results", type=Path, default=DEFAULT_HUMAN_RESULTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_COMPARATIVE_RESULTS)
    args = parser.parse_args()
    rule = json.loads(args.rule_results.read_text(encoding="utf-8"))
    human = json.loads(args.human_results.read_text(encoding="utf-8"))
    matrix = build_matrix(rule, human)
    args.out.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    print(f"wrote {args.out} ({len(matrix['matrix'])} systems)")


if __name__ == "__main__":
    main()