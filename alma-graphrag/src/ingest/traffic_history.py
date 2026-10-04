"""
Peak-hour traffic collection with an append-only history.

Why this exists
---------------
The graph keeps traffic signals for TRAFFIC_SIGNAL_TTL_HOURS and then deletes
them. That is right for serving (stale congestion must not rank hotels) and
wrong for research: the evaluation needs to know what rush hour actually looked
like on the days it was measured. The July snapshot was one off-peak batch in
which traffic time came out *below* free flow, so the disruption component had
nothing to react to (evaluation/results_robustness.json).

Each collection run therefore does two things:

1. writes the batch into the graph exactly as `scripts/run_ingest_traffic.py`
   does (so serving stays live), and
2. appends every city->hotel route observation to
   ``data/traffic_history/YYYY-MM-DD.jsonl`` (dates in Colombo time), which is
   never pruned.

Collection times default to Colombo's commuter peaks plus one midday control, so
the history can show peak against off-peak on the same routes.

Sri Lanka has no daylight saving, so a fixed UTC+05:30 offset is exact for local
dates; the scheduler itself is given the named zone.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.ingest.traffic import classify_severity

logger = logging.getLogger("alma.ingest.traffic_history")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_DIR = PROJECT_ROOT / "data" / "traffic_history"

COLOMBO_TZ_NAME = "Asia/Colombo"
COLOMBO_TZ = timezone(timedelta(hours=5, minutes=30), COLOMBO_TZ_NAME)

# Two commuter peaks and a midday control. Local Colombo time.
DEFAULT_COLLECTION_TIMES = ("07:45", "08:30", "12:30", "17:30", "18:15")

# Windows used to label an observation, independent of when it was scheduled,
# so a manual `--once` run at 08:10 is still recorded as am_peak.
PEAK_WINDOWS: Dict[str, Tuple[dtime, dtime]] = {
    "am_peak": (dtime(7, 0), dtime(9, 30)),
    "pm_peak": (dtime(16, 30), dtime(19, 30)),
}


def parse_times(spec: str | Sequence[str]) -> List[Tuple[int, int]]:
    """'07:45,17:30' -> [(7, 45), (17, 30)]. Rejects malformed entries loudly."""
    items = spec.split(",") if isinstance(spec, str) else list(spec)
    out: List[Tuple[int, int]] = []
    for raw in items:
        raw = raw.strip()
        if not raw:
            continue
        hh, _, mm = raw.partition(":")
        h, m = int(hh), int(mm or 0)
        if not (0 <= h < 24 and 0 <= m < 60):
            raise ValueError(f"invalid collection time {raw!r}")
        out.append((h, m))
    if not out:
        raise ValueError("no collection times given")
    return out


def slot_label(observed_at: datetime) -> str:
    """am_peak / pm_peak / off_peak for the Colombo-local time of an observation."""
    local = observed_at.astimezone(COLOMBO_TZ).time()
    for label, (start, end) in PEAK_WINDOWS.items():
        if start <= local <= end:
            return label
    return "off_peak"


def history_rows(traffic_data: Dict[str, Any], observed_at: datetime) -> List[Dict[str, Any]]:
    """One row per city->hotel route from a `fetch_all_traffic` result.

    `added_delay_min` is in-traffic minus free-flow time, floored at zero, and
    severity uses the same ratio rule as the Google ingest path, so history and
    graph can never disagree about what counted as congestion.
    """
    local = observed_at.astimezone(COLOMBO_TZ)
    slot = slot_label(observed_at)
    rows: List[Dict[str, Any]] = []
    for d in traffic_data.get("distances", []) or []:
        free = d.get("duration_min")
        in_traffic = d.get("duration_in_traffic_min")
        added: Optional[float] = None
        severity = "unknown"
        if free is not None and in_traffic:
            added = round(max(0.0, float(in_traffic) - float(free)), 1)
            severity = classify_severity(float(free) / float(in_traffic), 1.0)
        rows.append({
            "observed_at_utc": observed_at.astimezone(timezone.utc).isoformat(),
            "local_time": local.strftime("%Y-%m-%d %H:%M"),
            "weekday": local.strftime("%a"),
            "slot": slot,
            "origin": d.get("origin_name", ""),
            "hotel_id": d.get("hotel_id", ""),
            "hotel_name": d.get("hotel_name", ""),
            "distance_km": d.get("distance_km"),
            "free_flow_min": free,
            "in_traffic_min": in_traffic,
            "added_delay_min": added,
            "severity": severity,
            "source": d.get("source", ""),
        })
    return rows


def append_history(rows: Iterable[Dict[str, Any]],
                   directory: Path = DEFAULT_HISTORY_DIR) -> Optional[Path]:
    """Append rows to the Colombo-local date file. Returns the path, or None if empty."""
    rows = list(rows)
    if not rows:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    day = rows[0]["local_time"][:10]
    path = directory / f"{day}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def load_history(directory: Path = DEFAULT_HISTORY_DIR) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not directory.exists():
        return rows
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return rows


def _percentile(values: List[float], q: float) -> float:
    """Nearest-rank percentile; no numpy so the ingest path stays light."""
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def summarise(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per slot: batches, routes, and the added-delay distribution in minutes."""
    by_slot: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_slot.setdefault(r.get("slot", "off_peak"), []).append(r)
    out: Dict[str, Dict[str, Any]] = {}
    for slot in sorted(by_slot):
        group = by_slot[slot]
        delays = [float(r["added_delay_min"]) for r in group if r.get("added_delay_min") is not None]
        congested = sum(1 for r in group if r.get("severity") in ("moderate", "heavy"))
        out[slot] = {
            "batches": len({r["observed_at_utc"] for r in group}),
            "routes": len(group),
            "median_delay_min": round(_percentile(delays, 0.5), 1) if delays else None,
            "p90_delay_min": round(_percentile(delays, 0.9), 1) if delays else None,
            "max_delay_min": round(max(delays), 1) if delays else None,
            "share_moderate_or_heavy": round(congested / len(group), 3) if group else 0.0,
        }
    return out


def collect_once(city: str, provider: Optional[str] = None,
                 history_dir: Path = DEFAULT_HISTORY_DIR) -> Dict[str, Any]:
    """Fetch one traffic batch, archive it, then write it into the graph.

    History is written BEFORE the graph so an observation survives a failed
    graph write. Hotels are read from the graph by city name; note that hotels
    are stored under their canonical city ("Colombo"), not the suburb names in
    HOTELS_CITIES, which is why this takes an explicit city.
    """
    from src.config import TRAFFIC_PROVIDER
    from src.graph.query import _get_driver
    from src.ingest.traffic import fetch_all_traffic
    from src.ingest.traffic_linker import cleanup_stale_signals, link_traffic_to_hotels

    driver = _get_driver()
    with driver.session() as session:
        hotels = session.run(
            """
            MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)
            WHERE toLower(c.name) = toLower($city)
            RETURN h.id AS id, h.name AS name, h.lat AS lat, h.lng AS lng, c.name AS city_name
            """,
            {"city": city},
        ).data()
    if not hotels:
        raise RuntimeError(f"no hotels found under City {city!r}; check the canonical city name")

    lats = [h["lat"] for h in hotels if h.get("lat")]
    lngs = [h["lng"] for h in hotels if h.get("lng")]
    centre = [{"name": hotels[0]["city_name"], "lat": sum(lats) / len(lats), "lng": sum(lngs) / len(lngs)}]

    observed_at = datetime.now(timezone.utc)
    traffic = fetch_all_traffic(centre, hotels, provider=provider or TRAFFIC_PROVIDER)
    rows = history_rows(traffic, observed_at)
    path = append_history(rows, history_dir)
    counts = link_traffic_to_hotels(traffic, hotels)
    deleted = cleanup_stale_signals()

    delays = [r["added_delay_min"] for r in rows if r["added_delay_min"] is not None]
    result = {
        "observed_at_utc": observed_at.isoformat(),
        "slot": slot_label(observed_at),
        "hotels": len(hotels),
        "routes": len(rows),
        "max_delay_min": max(delays) if delays else None,
        "congested_routes": sum(1 for r in rows if r["severity"] in ("moderate", "heavy")),
        "history_file": str(path) if path else None,
        "graph_links": counts,
        "stale_signals_removed": deleted,
    }
    logger.info("Traffic collection: %s", result)
    return result
