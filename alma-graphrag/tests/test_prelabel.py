"""Tests for model-assisted pre-labelling and its anchoring audit.

The pre-label is a suggestion; the label is what the annotator leaves behind.
These tests pin the parts of that arrangement that would silently turn model
output into "human" gold if they broke: the blind control share, the parsing
of model replies, and the verdict that decides how the labels may be described.
"""
import pytest

from evaluation.annotation.prelabel import hotel_line, parse_reply
from evaluation.annotation.prelabel_audit import (agreement, audit, split_rows,
                                                  verdict)

ROW = {"query_id": "q1", "hotel_id": "h1", "question": "cheap hotel near Fort",
       "hotel_name": "CityRest Fort", "price_lkr": "12000", "rating": "4.2",
       "star": "3", "travel_time_min": "6", "amenities": "wifi; air conditioning"}


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------

def test_hotel_line_carries_the_attributes_and_nothing_else():
    line = hotel_line(ROW, 4)
    assert line.startswith("[4] request: cheap hotel near Fort")
    for fragment in ("CityRest Fort", "12000 LKR", "4.2/5", "3-star", "6 min"):
        assert fragment in line


def test_hotel_line_never_leaks_retrieval_provenance():
    """The grader must not know which system found the hotel, or at what rank."""
    line = hotel_line(dict(ROW, rank="1", system="WeightedGraphRAG"), 0).lower()
    assert "rank" not in line and "graphrag" not in line


def test_missing_price_is_stated_not_hidden():
    assert "price not listed" in hotel_line(dict(ROW, price_lkr=""), 0)


# --------------------------------------------------------------------------
# Reply parsing
# --------------------------------------------------------------------------

def test_parse_reply_reads_id_equals_grade():
    assert parse_reply("0=2\n1=0\n2=1", [0, 1, 2]) == {0: 2, 1: 0, 2: 1}


def test_parse_reply_survives_chatty_models():
    text = "Sure! Here are the grades:\n- [0] = 2\n  1=1\nLet me know if..."
    assert parse_reply(text, [0, 1]) == {0: 2, 1: 1}


def test_parse_reply_drops_out_of_range_and_unknown_ids():
    assert parse_reply("0=5\n9=2\n1=1", [0, 1]) == {1: 1}


def test_parse_reply_on_empty_response():
    assert parse_reply("", [0, 1]) == {}


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------

def _rows(shown, blind):
    """shown/blind: lists of (human, model) grades."""
    out = []
    for human, model in shown:
        out.append({"relevance": str(human), "model_prelabel": str(model),
                    "prelabel_shown": "yes"})
    for human, model in blind:
        out.append({"relevance": str(human), "model_prelabel": str(model),
                    "prelabel_shown": "no"})
    return out


def test_split_separates_shown_from_blind_controls():
    shown, blind = split_rows(_rows([(2, 2), (1, 0)], [(0, 0)]))
    assert len(shown) == 2 and len(blind) == 1


def test_unlabelled_rows_are_ignored():
    rows = [{"relevance": "", "model_prelabel": "2", "prelabel_shown": "yes"}]
    assert split_rows(rows) == ([], [])


def test_agreement_is_exact_grade_match():
    assert agreement([(2, 2), (1, 2)]) == pytest.approx(0.5)


def test_rubber_stamping_is_caught():
    """Everything confirmed, nothing overridden: that is not human labelling."""
    report = audit(_rows([(2, 2)] * 50, [(2, 2)] * 10))
    assert report["override_rate"] == 0.0
    assert report["verdict"] == "model-driven"


def test_anchoring_shows_up_as_assisted():
    # Annotators agree with a shown grade far more often than a blind one.
    shown = [(2, 2)] * 45 + [(1, 2)] * 5
    blind = [(2, 2)] * 5 + [(1, 2)] * 15
    report = audit(shown and _rows(shown, blind))
    assert report["anchoring_gap"] > 0.10
    assert report["verdict"] == "assisted"


def test_independent_judgement_reads_as_human():
    shown = [(2, 2)] * 30 + [(0, 2)] * 20
    blind = [(2, 2)] * 12 + [(0, 2)] * 8
    report = audit(_rows(shown, blind))
    assert abs(report["anchoring_gap"]) <= 0.10
    assert report["verdict"] == "human"


def test_override_direction_is_reported():
    report = audit(_rows([(2, 1), (0, 1), (1, 1)], [(1, 1)]))
    assert report["overrides_up"] == 1 and report["overrides_down"] == 1


def test_verdict_thresholds_are_explicit():
    assert verdict(0.9, 0.5, 0.3, tolerance=0.10) == "assisted"
    assert verdict(0.9, 0.85, 0.3, tolerance=0.10) == "human"
    assert verdict(0.99, 0.97, 0.3) == "model-driven"      # blind near-total
