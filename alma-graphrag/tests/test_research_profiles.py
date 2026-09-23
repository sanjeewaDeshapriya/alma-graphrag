"""The research-profile boundary: evaluated, never served.

`human` is fitted from the choice study but fails its acceptance gates, so it
must be rankable by the evaluation harness and invisible to the serving path.
These tests pin both halves, because a regression in either direction is
silent: serving a gated vector breaks NFR-09, and hiding it from evaluation
would make the human-weighted configuration unmeasurable.
"""
import json

import pytest

from evaluation.baselines import WeightedGraphBaseline
from src.crag.query_parser import QueryIntent
from src.graph.retriever import (HANDSET_WEIGHTS, RESEARCH_WEIGHT_PROFILES,
                                 WEIGHT_PROFILES, base_weights,
                                 load_human_weights, load_research_human_weights,
                                 weights_for_intent)

VECTOR = {"spatial": 0.5, "accessibility": 0.2, "facility": 0.1,
          "economic": 0.15, "disruption": 0.05}


def _artifact(tmp_path, shippable):
    path = tmp_path / "human_weights.json"
    path.write_text(json.dumps({"shippable": shippable, "weights": VECTOR}),
                    encoding="utf-8")
    return path


def test_failed_gates_block_serving_but_not_research(tmp_path):
    path = _artifact(tmp_path, shippable=False)
    assert load_human_weights(path) is None
    assert load_research_human_weights(path).to_dict()["spatial"] == 0.5


def test_passing_gates_load_on_both_paths(tmp_path):
    path = _artifact(tmp_path, shippable=True)
    assert load_human_weights(path) is not None
    assert load_research_human_weights(path) is not None


def test_a_malformed_vector_is_refused_even_for_research(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"shippable": False,
                                "weights": dict(VECTOR, spatial=-0.5)}),
                    encoding="utf-8")
    assert load_research_human_weights(path) is None


def test_human_is_research_only_while_the_wave1_fit_fails():
    """The shipped state today: fitted, evaluated, not served."""
    if "human" in WEIGHT_PROFILES:            # a future passing fit
        pytest.skip("a human fit has passed its gates and is served")
    assert "human" in RESEARCH_WEIGHT_PROFILES


@pytest.mark.skipif("human" in WEIGHT_PROFILES, reason="human fit now passes gates")
def test_serving_lookup_falls_back_to_handset():
    assert base_weights("human").to_dict() == HANDSET_WEIGHTS.to_dict()


@pytest.mark.skipif("human" in WEIGHT_PROFILES, reason="human fit now passes gates")
def test_research_lookup_returns_the_fitted_vector():
    served = base_weights("human")
    research = base_weights("human", research=True)
    assert research.to_dict() != served.to_dict()
    assert research.to_dict() == RESEARCH_WEIGHT_PROFILES["human"].to_dict()


@pytest.mark.skipif("human" in WEIGHT_PROFILES, reason="human fit now passes gates")
def test_intent_ladder_applies_to_a_research_vector():
    """A research row must rank the way serving would if the vector were served."""
    intent = QueryIntent(city="Colombo", sort_intent="cheapest")
    plain = weights_for_intent(intent, "human")
    research = weights_for_intent(intent, "human", research=True)
    assert research.economic > research.facility      # ladder still applied
    assert plain.to_dict() != research.to_dict()


def test_baseline_only_reaches_research_vectors_when_asked(monkeypatch):
    calls = {}

    class _Retriever:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr("evaluation.baselines.WeightedRetriever", _Retriever)
    WeightedGraphBaseline(weight_profile="human")
    assert calls["research_profile"] is False
    WeightedGraphBaseline(weight_profile="human", research_profile=True)
    assert calls["research_profile"] is True
