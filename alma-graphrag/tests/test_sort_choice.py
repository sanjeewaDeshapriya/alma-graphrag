"""Sort-switch extraction and the within-participant association test."""
from __future__ import annotations

import numpy as np

from weight_elicitation.sort_choice import association_test, extract_events, primed_lift


def response(pid, task, index, final, sorts=(), primary="economic"):
    return {"participantId": pid, "taskId": task, "primaryDimension": primary,
            "isAttentionCheck": False,
            "timing": {"task_index": index, "final_sort": final},
            "interactions": [{"kind": "sort", "at_ms": 5000, "value": s} for s in sorts]}


def test_switches_are_relative_to_the_previous_question_in_task_order():
    rows = [response("p", "t2", 1, "price_asc", ["price_asc"]),
            response("p", "t1", 0, "distance"),
            response("p", "t3", 2, "price_asc")]
    ev = {e["task"]: e for e in extract_events(rows)}
    assert ev["t1"]["previous_sort"] is None and not ev["t1"]["switched"]
    assert ev["t2"]["switched"] and ev["t2"]["previous_sort"] == "distance"
    assert not ev["t3"]["switched"]                      # carried over, not chosen again
    assert ev["t2"]["n_sort_events"] == 1


def test_attention_checks_and_missing_sorts_are_ignored():
    rows = [response("p", "t1", 0, "distance"),
            {**response("p", "t_attn", 1, "rating"), "isAttentionCheck": True},
            {"participantId": "p", "taskId": "t2", "timing": {"task_index": 2}}]
    assert [e["task"] for e in extract_events(rows)] == ["t1"]


def test_association_is_detected_when_a_question_steers_the_sort():
    rng = np.random.default_rng(0)
    rows = []
    for p in range(120):
        prev = "distance"
        rows.append(response(f"p{p}", "t1", 0, prev))
        for i, (task, target) in enumerate([("t2", "price_asc"), ("t3", "travel"), ("t4", "price_asc")],
                                           start=1):
            final = target if rng.random() < 0.7 else prev
            if final == prev:
                final = "rating" if prev != "rating" else "distance"
            rows.append(response(f"p{p}", task, i, final))
            prev = final
    events = extract_events(rows)
    sw = [e for e in events if e["switched"]]
    tasks = ["t2", "t3", "t4"]
    out = association_test(sw, tasks, n_perm=200, seed=1)
    assert out["p_within_participant_permutation"] < 0.05


def test_primed_lift_skips_the_first_question():
    rows = [response("a", "t1", 0, "distance", primary="economic"),
            response("a", "t2", 1, "price_asc", ["price_asc"], primary="economic"),
            response("b", "t1", 0, "distance", primary="economic"),
            response("b", "t2", 1, "distance", primary="economic")]
    events = extract_events(rows)
    lift = primed_lift([e for e in events if e["switched"]], events, n_boot=20, seed=1)
    assert "t1" not in lift and "t2" in lift
    assert lift["t2"]["rate_this_question"] == 0.5
