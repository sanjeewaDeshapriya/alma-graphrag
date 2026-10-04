"""Tests for the split-robustness check (evaluation/human_eval_seeds.py).

Built on a small synthetic study so the suite stays offline: no embedder, no
language model, no database.
"""
import pytest

from evaluation.human_eval import Study
from evaluation.human_eval_seeds import one_split, summarise

HOTELS = ["h1", "h2", "h3", "h4"]


def _material():
    # h1 is nearest the anchor and cheapest; people below always choose it, so a
    # fitted vector should rank it first on every split.
    spatial = {"h1": 1.0, "h2": 0.6, "h3": 0.3, "h4": 0.0}
    return {
        "hotels": {h: {"name": h, "attributes": {},
                       "components_global": {"facility": 0.5, "economic": 0.5,
                                             "disruption": 0.5}} for h in HOTELS},
        "anchor_components": {"a": {h: {"spatial": spatial[h],
                                        "accessibility": spatial[h]} for h in HOTELS}},
        "tasks": [{"id": "t1", "anchor_id": "a"}, {"id": "t2", "anchor_id": "a"}],
    }


def _rows(n_people=10):
    rows = []
    for i in range(n_people):
        for task in ("t1", "t2"):
            rows.append({"participant_id": f"p{i}", "task_id": task,
                         "chosen_hotel_id": "h1" if i % 5 else "h2",
                         "is_attention_check": "false", "scenario_persona": "x"})
    return rows


@pytest.fixture
def study():
    rows = _rows()
    return Study(_material(), rows, {r["participant_id"] for r in rows})


def test_one_split_reports_every_ranker_and_the_fitted_vector(study):
    out = one_split(study, seed=0, holdout=0.3, k=3, min_votes=1)
    rankers = set(out) - {"fitted_weights"}
    # `unweighted` is graph-only GraphRAG (equal weights), always reported.
    assert {"handset", "fitted-on-train", "unweighted"} <= rankers
    # `human` joins whenever the study fit is loadable, gated out or not.
    assert rankers <= {"handset", "fitted-on-train", "human", "unweighted"}
    for name in rankers:
        assert 0.0 <= out[name]["choice_ndcg"] <= 1.0
        assert out[name]["mean_rank"] >= 1.0
    assert abs(sum(out["fitted_weights"].values()) - 1.0) < 1e-3


def test_one_split_is_deterministic_for_a_seed(study):
    assert one_split(study, 3, 0.3, 3, 1) == one_split(study, 3, 0.3, 3, 1)


def test_fitted_vector_learns_the_dominant_choice(study):
    out = one_split(study, seed=1, holdout=0.3, k=3, min_votes=1)
    assert out["fitted-on-train"]["choice_ndcg"] >= out["handset"]["choice_ndcg"] - 1e-9


def test_summarise_reports_spread_not_just_a_mean():
    s = summarise([0.1, 0.2, 0.3, 0.4])
    assert s["mean"] == pytest.approx(0.25)
    assert s["min"] == 0.1 and s["max"] == 0.4
    assert s["sd"] > 0
