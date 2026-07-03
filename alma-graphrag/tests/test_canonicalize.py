import pytest

from src.ingest.canonicalize import canonical_city


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Colombo", "Colombo"),
        ("colombo", "Colombo"),
        ("Colombo 03", "Colombo"),
        ("colombo 7", "Colombo"),
        ("Colombo, 00300", "Colombo"),
        ("Galle 80000", "Galle"),
        ("  Piliyandala  ", "Piliyandala"),
        ("Mt Lavinia", "Mount Lavinia"),
        ("mount-lavinia", "Mount Lavinia"),
        ("nuwara-eliya", "Nuwara Eliya"),
        ("NUWARA ELIYA", "Nuwara Eliya"),
        ("Kandy City", "Kandy"),
        ("trinco", "Trincomalee"),
    ],
)
def test_canonical_city(raw, expected):
    assert canonical_city(raw) == expected


def test_none_and_empty_pass_through():
    assert canonical_city(None) is None
    assert canonical_city("") == ""


def test_unknown_city_is_titlecased_not_dropped():
    assert canonical_city("matara") == "Matara"
