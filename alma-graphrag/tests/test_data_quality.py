"""Conservative duplicate detection used by scripts/fix_data_quality.py."""
from scripts.fix_data_quality import _group_duplicates


def _hotel(name, *, source="liteapi", lat=6.93, lng=79.84, price=25000):
    return {"city": "Colombo", "id": name, "name": name, "source": source,
            "lat": lat, "lng": lng, "price": price}


def test_groups_exact_name_duplicates_without_coordinate_data():
    groups = _group_duplicates([
        _hotel("Cinnamon Grand Colombo", source="liteapi", lat=None, lng=None),
        _hotel("Cinnamon Grand Colombo", source="google_places", lat=None, lng=None),
    ])
    assert len(groups) == 1
    assert len(groups[0]["hs"]) == 2


def test_groups_same_location_same_price_name_variants():
    groups = _group_duplicates([
        _hotel("Fairway Colombo", source="google_places", lat=6.933924, lng=79.843916),
        _hotel("Fairway Colombo - Sri Lanka's First Hotel With Robot Technology",
               lat=6.933895, lng=79.843896),
    ])
    assert len(groups) == 1


def test_keeps_nearby_differently_priced_or_named_hotels_distinct():
    groups = _group_duplicates([
        _hotel("Hilton Colombo", price=43979),
        _hotel("Hilton Colombo Residence", lat=6.920345, lng=79.856151, price=41671),
    ])
    assert groups == []