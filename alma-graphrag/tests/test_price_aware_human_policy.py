"""The price-aware human policy: study weights only where the study could see.

Wave 1 identified the economic weight and nothing else, so the policy applies
the study-derived vector to price-led intents and the hand-set prior to
everything else. These tests pin that rule, because widening it silently is
exactly how the accessibility artifact (0.026) would reach travel-time queries.
"""
import pytest

from src.crag.query_parser import QueryIntent
from src.graph.retriever import HANDSET_WEIGHTS, RESEARCH_WEIGHT_PROFILES
from src.graph.weight_policy import PriceAwareHumanPolicy, get_policy

INFORMED = RESEARCH_WEIGHT_PROFILES["human_informed"]


@pytest.fixture
def policy():
    return PriceAwareHumanPolicy()


@pytest.mark.parametrize("intent", [
    QueryIntent(city="Colombo", price_preference="low"),
    QueryIntent(city="Colombo", price_preference="high"),
    QueryIntent(city="Colombo", sort_intent="cheapest"),
    QueryIntent(city="Colombo", max_price_lkr=25000),
])
def test_price_led_intents_use_the_study_weight(policy, intent):
    assert policy._price_led(intent) is True


@pytest.mark.parametrize("intent", [
    QueryIntent(city="Colombo"),
    QueryIntent(city="Colombo", avoid_traffic=True),
    QueryIntent(city="Colombo", sort_intent="highest_rated"),
    QueryIntent(city="Colombo", accessibility_priority="high"),
])
def test_other_intents_stay_on_the_hand_set_prior(policy, intent):
    assert policy._price_led(intent) is False


def test_the_two_branches_differ_only_by_the_base_vector(policy):
    """A travel-time query must rank exactly as the hand-set system does."""
    from src.graph.retriever import weights_for_intent
    intent = QueryIntent(city="Colombo", avoid_traffic=True)
    assert policy.predict(intent).to_dict() == weights_for_intent(intent, "handset").to_dict()


def test_a_price_query_carries_more_economic_mass_than_hand_set(policy):
    intent = QueryIntent(city="Colombo", price_preference="low")
    from src.graph.retriever import weights_for_intent
    assert policy.predict(intent).economic > weights_for_intent(intent, "handset").economic


def test_the_study_weight_never_leaks_accessibility(policy):
    """The dimension wave 1 could not identify must not reach any ranking."""
    intent = QueryIntent(city="Colombo", price_preference="low")
    # 0.026 was the unidentified fit; the informed vector keeps the prior ratio.
    assert policy.predict(intent).accessibility > 0.10


def test_policy_is_resolvable_by_name():
    assert isinstance(get_policy("human-price-aware"), PriceAwareHumanPolicy)
    assert isinstance(get_policy("human_price_aware"), PriceAwareHumanPolicy)


def test_describe_states_the_rule_and_both_vectors(policy):
    described = policy.describe()
    assert described["learned"] is False
    assert described["price_led_weights"] == INFORMED.to_dict()
    assert described["otherwise"] == HANDSET_WEIGHTS.to_dict()
