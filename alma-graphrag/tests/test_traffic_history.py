"""Peak-hour traffic history (src/ingest/traffic_history.py), no network or database."""
from datetime import datetime, timezone

import pytest

from src.ingest.traffic_history import (
    append_history,
    history_rows,
    load_history,
    parse_times,
    slot_label,
    summarise,
)


def _utc(hh, mm):
    # Colombo is UTC+05:30, so 02:15 UTC is 07:45 local.
    return datetime(2026, 9, 15, hh, mm, tzinfo=timezone.utc)


def test_slot_label_uses_colombo_local_time():
    assert slot_label(_utc(2, 15)) == "am_peak"    # 07:45
    assert slot_label(_utc(7, 0)) == "off_peak"    # 12:30
    assert slot_label(_utc(12, 0)) == "pm_peak"    # 17:30
    assert slot_label(_utc(16, 0)) == "off_peak"   # 21:30


def test_parse_times():
    assert parse_times("07:45, 17:30") == [(7, 45), (17, 30)]
    with pytest.raises(ValueError):
        parse_times("25:00")
    with pytest.raises(ValueError):
        parse_times("")


def test_history_rows_compute_added_delay_and_severity():
    traffic = {"distances": [
        {"origin_name": "Colombo", "hotel_id": "h1", "hotel_name": "A",
         "distance_km": 3.0, "duration_min": 10.0, "duration_in_traffic_min": 22.0},
        {"origin_name": "Colombo", "hotel_id": "h2", "hotel_name": "B",
         "distance_km": 2.0, "duration_min": 8.0, "duration_in_traffic_min": 6.5},
        {"origin_name": "Colombo", "hotel_id": "h3", "hotel_name": "C",
         "distance_km": 2.0, "duration_min": 8.0, "duration_in_traffic_min": None},
    ]}
    rows = history_rows(traffic, _utc(2, 15))
    assert [r["added_delay_min"] for r in rows] == [12.0, 0.0, None]
    assert [r["severity"] for r in rows] == ["moderate", "light", "unknown"]
    assert rows[0]["local_time"] == "2026-09-15 07:45"
    assert rows[0]["slot"] == "am_peak"


def test_append_load_and_summarise(tmp_path):
    traffic = {"distances": [
        {"hotel_id": f"h{i}", "duration_min": 10.0, "duration_in_traffic_min": 10.0 + i}
        for i in range(5)
    ]}
    append_history(history_rows(traffic, _utc(2, 15)), tmp_path)
    append_history(history_rows(traffic, _utc(7, 0)), tmp_path)
    assert append_history([], tmp_path) is None

    rows = load_history(tmp_path)
    assert len(rows) == 10
    summary = summarise(rows)
    assert set(summary) == {"am_peak", "off_peak"}
    assert summary["am_peak"]["batches"] == 1
    assert summary["am_peak"]["routes"] == 5
    assert summary["am_peak"]["max_delay_min"] == 4.0
    assert summary["am_peak"]["median_delay_min"] == 2.0
