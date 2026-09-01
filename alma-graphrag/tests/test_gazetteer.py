"""Tests for the place gazetteer used to geocode events.

The failure modes that matter are precision failures: a substring match that
turns "Comfort Inn" into the Fort district, or a bare "Colombo" that gets
treated as a point and links an event to every hotel in the pool — which is the
exact constant-feature problem the gazetteer exists to remove.
"""
from __future__ import annotations

from src.ingest.gazetteer import (
    CITY_LEVEL,
    PLACES,
    find_places,
    locate,
    precision,
)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def test_finds_a_simple_place():
    assert locate("Heavy traffic reported in Pettah this morning")[0] == "Pettah"


def test_returns_none_when_no_place_named():
    assert locate("Tourism arrivals rise ahead of the new season") is None


def test_empty_text_is_safe():
    assert find_places("") == []
    assert locate("") is None


def test_longest_match_wins():
    """"Colombo Fort" must not be split into "Colombo" + "Fort"."""
    hits = find_places("Colombo Fort station closed")
    assert hits[0][0] == "Colombo Fort"
    assert "Fort" not in [h[0] for h in hits]
    assert "Colombo" not in [h[0] for h in hits]


def test_word_boundaries_prevent_substring_false_positives():
    # "Fort" inside "Comfort", "Ella" inside "Stella", "Galle" inside "Gallery".
    assert locate("The Comfort Lodge reopened") is None
    assert locate("Stella Maris hosted the event") is None


def test_case_insensitive():
    assert locate("PROTEST IN PETTAH")[0] == "Pettah"
    assert locate("protest in pettah")[0] == "Pettah"


def test_respects_the_limit():
    text = "Delays from Pettah through Borella to Dehiwala and on to Kandy"
    assert len(find_places(text, limit=2)) == 2


def test_results_follow_text_order():
    names = [n for n, _, _ in find_places("From Kandy to Pettah", limit=2)]
    assert names == ["Kandy", "Pettah"]


def test_coordinates_are_returned():
    name, lat, lng = locate("Flooding near Wellawatte")
    assert name == "Wellawatte"
    assert 5.0 < lat < 10.0      # Sri Lanka's latitude range
    assert 79.0 < lng < 82.0     # and its longitude range


# ---------------------------------------------------------------------------
# Precision classification
# ---------------------------------------------------------------------------

def test_city_names_are_city_precision():
    for name in ("Colombo", "Kandy", "Galle"):
        assert precision(name) == "city"


def test_wards_and_landmarks_are_local_precision():
    for name in ("Pettah", "Galle Face", "Marine Drive", "Lotus Tower"):
        assert precision(name) == "local"


def test_every_city_level_entry_exists_in_places():
    """A CITY_LEVEL name absent from PLACES would never be matched, so the
    coarse-precision guard would silently never fire for it."""
    missing = CITY_LEVEL - set(PLACES)
    assert not missing, f"CITY_LEVEL names not in PLACES: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Table integrity
# ---------------------------------------------------------------------------

def test_all_coordinates_are_inside_sri_lanka():
    for name, (lat, lng) in PLACES.items():
        assert 5.8 <= lat <= 10.0, f"{name} latitude {lat} outside Sri Lanka"
        assert 79.5 <= lng <= 82.0, f"{name} longitude {lng} outside Sri Lanka"


def test_no_duplicate_coordinates_for_distinct_areas():
    """Aliases may share a point; genuinely different areas must not.

    Two distinct wards on the same coordinate would make an event equidistant
    from both, quietly destroying the discrimination the linking depends on.
    """
    known_aliases = {
        frozenset({"Kollupitiya", "Colpetty"}),
        frozenset({"Wellawatte", "Wellawatta"}),
        frozenset({"Sri Jayawardenepura Kotte", "Kotte", "Parliament Road"}),
        frozenset({"Colombo Fort", "Fort"}),
        frozenset({"Katunayake", "Bandaranaike International Airport"}),
        frozenset({"BMICH",
                   "Bandaranaike Memorial International Conference Hall"}),
        frozenset({"Galle Face", "Galle Face Green", "One Galle Face"}),
        frozenset({"Havelock Town", "Havelock Road"}),
        frozenset({"Bambalapitiya", "Galle Road"}),
        frozenset({"Lotus Tower", "Colombo"}),
    }
    by_point: dict = {}
    for name, point in PLACES.items():
        by_point.setdefault(point, set()).add(name)
    for point, names in by_point.items():
        if len(names) > 1:
            assert any(names <= alias for alias in known_aliases), (
                f"unexpected coordinate collision at {point}: {sorted(names)}"
            )
