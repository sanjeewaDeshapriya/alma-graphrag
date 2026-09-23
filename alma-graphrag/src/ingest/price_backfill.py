"""
Re-snapshot hotel prices from LiteAPI, filling the hotels that had none.

Why this exists
---------------
24 of the 58 Colombo pool hotels (41%) had no price. All 24 were Google Places
nodes: Google returns no nightly rate, and the ingest correctly stores null
rather than a placeholder. The retriever then scores them a neutral 0.5 on
`economic`, and every price-constrained gold spec fails them, so price queries
were partly decided by which source a hotel came from.

Design decisions
----------------
* ONE snapshot for every hotel, not a fill of the gaps. Rates move a lot between
  dates (Radisson Colombo: 35,463 LKR in the July snapshot, 19,064 LKR quoted in
  September), so filling only the missing 24 would compare prices taken two
  months apart. Previous values are kept on the node as `price_prev_lkr` and in
  a JSON backup that `restore_prices` puts back.
* Nightly price = median, over several check-in dates, of the cheapest offer
  available on that date (1 night, same occupancy and currency as the ingest).
  A date with no availability is skipped. A hotel with no availability on any
  date gets NO price; nothing is imputed.
* Google nodes are matched to LiteAPI ids by location and name: identical
  normalised name within 400 m, else token overlap >= 0.6 within 250 m. Two
  equally good candidates count as no match.
* A Google node that matches a LiteAPI id which already has its own node is the
  same hotel stored twice. It is priced identically and tagged `duplicate_of`;
  merging the pair is a separate decision because it changes the pool size.
"""
from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.ingest.traffic import haversine

logger = logging.getLogger("alma.ingest.price_backfill")

DEFAULT_OFFSETS_DAYS = (14, 21, 28)
EXACT_NAME_MAX_M = 400.0
FUZZY_NAME_MAX_M = 250.0
FUZZY_MIN_OVERLAP = 0.6
RATE_BATCH = 50

_STOP = {"hotel", "hotels", "colombo", "sri", "lanka", "the", "by", "and", "city", "a", "of", "at"}

PRICE_PROPS = ("price_per_night_lkr", "price_per_night", "price_currency", "price_estimated",
               "price_range", "price_source", "price_snapshot_at", "price_checkins",
               "price_n_dates", "liteapi_id", "duplicate_of", "price_prev_lkr")


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def name_tokens(name: str) -> List[str]:
    s = re.sub(r"[^a-z0-9 ]", " ", (name or "").lower())
    return [w for w in s.split() if w not in _STOP]


def normalise_name(name: str) -> str:
    return " ".join(name_tokens(name))


def token_overlap(a: str, b: str) -> float:
    """Overlap coefficient: shared tokens / tokens in the shorter name."""
    ta, tb = set(name_tokens(a)), set(name_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


@dataclass
class Match:
    liteapi_id: str
    method: str          # "id" | "exact_name" | "fuzzy_name"
    distance_m: float


def match_hotel(node: Dict[str, Any], catalogue: Sequence[Dict[str, Any]]) -> Optional[Match]:
    """Find the LiteAPI id for a graph hotel node, or None."""
    node_id = str(node.get("id") or "")
    if node.get("source") == "liteapi" or node_id.startswith("lp"):
        return Match(node_id, "id", 0.0)
    lat, lng = node.get("lat"), node.get("lng")
    if lat is None or lng is None:
        return None

    target = normalise_name(node.get("name", ""))
    exact: List[Tuple[float, str]] = []
    fuzzy: List[Tuple[float, float, str]] = []
    for c in catalogue:
        if c.get("deletedAt") or c.get("latitude") is None or c.get("longitude") is None:
            continue
        dist_m = haversine(float(lat), float(lng), float(c["latitude"]), float(c["longitude"])) * 1000
        if dist_m > EXACT_NAME_MAX_M:
            continue
        if target and normalise_name(c.get("name", "")) == target:
            exact.append((dist_m, str(c["id"])))
        elif dist_m <= FUZZY_NAME_MAX_M:
            ov = token_overlap(node.get("name", ""), c.get("name", ""))
            if ov >= FUZZY_MIN_OVERLAP:
                fuzzy.append((ov, dist_m, str(c["id"])))

    if exact:
        exact.sort()
        if len(exact) > 1 and abs(exact[0][0] - exact[1][0]) < 1.0:
            return None
        return Match(exact[0][1], "exact_name", round(exact[0][0], 1))
    if fuzzy:
        fuzzy.sort(key=lambda t: (-t[0], t[1]))
        if len(fuzzy) > 1 and fuzzy[0][0] == fuzzy[1][0] and abs(fuzzy[0][1] - fuzzy[1][1]) < 25:
            return None
        return Match(fuzzy[0][2], "fuzzy_name", round(fuzzy[0][1], 1))
    return None


def _amount(price_array: Any) -> Optional[float]:
    if isinstance(price_array, list) and price_array:
        try:
            v = float(price_array[0].get("amount"))
            return v if v > 0 else None
        except (TypeError, ValueError):
            return None
    return None


def cheapest_nightly(rate_entry: Dict[str, Any], nights: int = 1) -> Optional[float]:
    """Cheapest offer total across room types, per night. Mirrors LiteApiClient."""
    totals = []
    for rt in rate_entry.get("roomTypes") or []:
        total = _amount(rt.get("offerRetailRate"))
        if total is None:
            rates = rt.get("rates") or []
            total = _amount(((rates[0] if rates else {}).get("retailRate") or {}).get("total"))
        if total is not None:
            totals.append(total)
    return round(min(totals) / max(nights, 1), 2) if totals else None


def slim_rate_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only what `cheapest_nightly` reads, plus room and board names.

    A full rates response for 55 hotels over three dates is ~29 MB; this keeps
    provenance at ~2 MB and reproduces every nightly price exactly.
    """
    slim_types = []
    for rt in entry.get("roomTypes") or []:
        r0 = (rt.get("rates") or [{}])[0]
        slim_types.append({
            "offerRetailRate": rt.get("offerRetailRate"),
            "rates": [{"name": (r0.get("name") or "")[:80], "boardName": r0.get("boardName"),
                       "retailRate": {"total": (r0.get("retailRate") or {}).get("total")}}],
        })
    return {"hotelId": entry.get("hotelId"), "roomTypes": slim_types}


def price_range_label(price_lkr: Optional[float]) -> str:
    from src.ingest.liteapi import LiteApiClient
    return LiteApiClient._price_range(price_lkr or 0.0)


@dataclass
class PriceUpdate:
    hotel_id: str
    name: str
    liteapi_id: Optional[str]
    match_method: Optional[str]
    match_distance_m: Optional[float]
    duplicate_of: Optional[str]
    prev_price_lkr: Optional[float]
    new_price_lkr: Optional[float]
    nightly_by_checkin: Dict[str, Optional[float]] = field(default_factory=dict)

    @property
    def n_dates(self) -> int:
        return sum(1 for v in self.nightly_by_checkin.values() if v)


def plan_updates(nodes: Sequence[Dict[str, Any]], matches: Dict[str, Optional[Match]],
                 nightly: Dict[str, Dict[str, Optional[float]]],
                 checkins: Sequence[str]) -> List[PriceUpdate]:
    """Combine matches and per-date nightly prices into one update per node.

    `nightly` is {liteapi_id: {checkin: price or None}}.
    """
    node_ids = {str(n["id"]) for n in nodes}
    plans: List[PriceUpdate] = []
    for n in nodes:
        hid = str(n["id"])
        m = matches.get(hid)
        by_date = {ci: (nightly.get(m.liteapi_id, {}).get(ci) if m else None) for ci in checkins}
        got = [v for v in by_date.values() if v]
        dup = m.liteapi_id if (m and m.method != "id" and m.liteapi_id in node_ids) else None
        plans.append(PriceUpdate(
            hotel_id=hid, name=n.get("name", ""),
            liteapi_id=m.liteapi_id if m else None,
            match_method=m.method if m else None,
            match_distance_m=m.distance_m if m else None,
            duplicate_of=dup,
            prev_price_lkr=float(n["price"]) if n.get("price") else None,
            new_price_lkr=round(statistics.median(got), 2) if got else None,
            nightly_by_checkin=by_date,
        ))
    return plans


def checkin_dates(offsets: Iterable[int], today: Optional[date] = None) -> List[str]:
    today = today or date.today()
    return [(today + timedelta(days=int(o))).isoformat() for o in offsets]


# ---------------------------------------------------------------------------
# LiteAPI calls
# ---------------------------------------------------------------------------

def _client():
    import httpx
    from src.config import LITEAPI_KEY, LITEAPI_TIMEOUT
    if not LITEAPI_KEY:
        raise RuntimeError("LITEAPI_KEY is not set")
    return httpx.Client(timeout=LITEAPI_TIMEOUT + 50, headers={
        "X-API-Key": LITEAPI_KEY, "Accept": "application/json", "Content-Type": "application/json"})


def fetch_catalogue(city: str, country: str = "LK") -> Dict[str, Any]:
    from src.config import LITEAPI_BASE_URL
    with _client() as c:
        r = c.get(f"{LITEAPI_BASE_URL}/data/hotels",
                  params={"countryCode": country, "cityName": city, "limit": 5000})
    r.raise_for_status()
    return r.json()


def fetch_rates(hotel_ids: Sequence[str], checkin: str, nights: int = 1) -> List[Dict[str, Any]]:
    from src.config import (LITEAPI_ADULTS, LITEAPI_BASE_URL, LITEAPI_CURRENCY,
                            LITEAPI_GUEST_NATIONALITY, LITEAPI_TIMEOUT)
    checkout = (date.fromisoformat(checkin) + timedelta(days=nights)).isoformat()
    out: List[Dict[str, Any]] = []
    ids = list(dict.fromkeys(hotel_ids))
    with _client() as c:
        for i in range(0, len(ids), RATE_BATCH):
            payload = {
                "hotelIds": ids[i:i + RATE_BATCH], "checkin": checkin, "checkout": checkout,
                "currency": LITEAPI_CURRENCY, "guestNationality": LITEAPI_GUEST_NATIONALITY,
                "occupancies": [{"adults": LITEAPI_ADULTS}], "timeout": LITEAPI_TIMEOUT,
            }
            r = c.post(f"{LITEAPI_BASE_URL}/hotels/rates", json=payload)
            if r.status_code != 200:
                logger.warning("rates %s batch %d: HTTP %d %s", checkin, i, r.status_code, r.text[:200])
                continue
            out.extend(r.json().get("data") or [])
    return out


# ---------------------------------------------------------------------------
# Graph I/O
# ---------------------------------------------------------------------------

def read_pool(city: str) -> List[Dict[str, Any]]:
    from src.graph.query import _get_driver
    with _get_driver().session() as s:
        return s.run(
            """
            MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)
            WHERE toLower(c.name) = toLower($city)
            RETURN h.id AS id, h.name AS name, h.source AS source, h.lat AS lat, h.lng AS lng,
                   h.price_per_night_lkr AS price, properties(h) AS props
            ORDER BY h.id
            """, {"city": city}).data()


def backup_prices(nodes: Sequence[Dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "taken_at": datetime.now(timezone.utc).isoformat(),
        "props": list(PRICE_PROPS),
        "hotels": {str(n["id"]): {p: n["props"].get(p) for p in PRICE_PROPS} for n in nodes},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_updates(plans: Sequence[PriceUpdate], checkins: Sequence[str]) -> int:
    from src.graph.query import _get_driver
    snapshot_at = datetime.now(timezone.utc).isoformat()
    rows = [{
        "id": p.hotel_id,
        "price": p.new_price_lkr,
        "range": price_range_label(p.new_price_lkr) if p.new_price_lkr else None,
        "prev": p.prev_price_lkr,
        "liteapi_id": p.liteapi_id,
        "dup": p.duplicate_of,
        "n": p.n_dates,
    } for p in plans]
    with _get_driver().session() as s:
        rec = s.run(
            """
            UNWIND $rows AS r
            MATCH (h:Hotel {id: r.id})
            SET h.price_prev_lkr = r.prev,
                h.price_per_night_lkr = r.price,
                h.price_per_night = r.price,
                h.price_currency = CASE WHEN r.price IS NULL THEN h.price_currency ELSE 'LKR' END,
                h.price_estimated = r.price IS NULL,
                h.price_range = coalesce(r.range, h.price_range),
                h.price_source = CASE WHEN r.price IS NULL THEN 'unavailable' ELSE 'liteapi' END,
                h.price_snapshot_at = $snapshot_at,
                h.price_checkins = $checkins,
                h.price_n_dates = r.n,
                h.liteapi_id = r.liteapi_id,
                h.duplicate_of = r.dup
            RETURN count(h) AS n
            """, {"rows": rows, "snapshot_at": snapshot_at, "checkins": list(checkins)}).single()
    return int(rec["n"])


def restore_prices(path: Path) -> int:
    from src.graph.query import _get_driver
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [{"id": hid, "props": props} for hid, props in payload["hotels"].items()]
    with _get_driver().session() as s:
        rec = s.run(
            """
            UNWIND $rows AS r
            MATCH (h:Hotel {id: r.id})
            SET h.price_per_night_lkr = r.props.price_per_night_lkr,
                h.price_per_night = r.props.price_per_night,
                h.price_currency = r.props.price_currency,
                h.price_estimated = r.props.price_estimated,
                h.price_range = r.props.price_range,
                h.price_source = r.props.price_source,
                h.price_snapshot_at = r.props.price_snapshot_at,
                h.price_checkins = r.props.price_checkins,
                h.price_n_dates = r.props.price_n_dates,
                h.liteapi_id = r.props.liteapi_id,
                h.duplicate_of = r.props.duplicate_of,
                h.price_prev_lkr = r.props.price_prev_lkr
            RETURN count(h) AS n
            """, {"rows": rows}).single()
    return int(rec["n"])


def summarise_plans(plans: Sequence[PriceUpdate]) -> Dict[str, Any]:
    had = [p for p in plans if p.prev_price_lkr]
    changed = [p for p in had if p.new_price_lkr]
    pct = [100.0 * (p.new_price_lkr - p.prev_price_lkr) / p.prev_price_lkr for p in changed]
    return {
        "hotels": len(plans),
        "priced_before": len(had),
        "priced_after": sum(1 for p in plans if p.new_price_lkr),
        "missing_after": sum(1 for p in plans if not p.new_price_lkr),
        "newly_priced": sum(1 for p in plans if p.new_price_lkr and not p.prev_price_lkr),
        "lost_price": sum(1 for p in plans if p.prev_price_lkr and not p.new_price_lkr),
        "unmatched": sum(1 for p in plans if not p.liteapi_id),
        "duplicates": sum(1 for p in plans if p.duplicate_of),
        "match_methods": {m: sum(1 for p in plans if p.match_method == m)
                          for m in ("id", "exact_name", "fuzzy_name")},
        "median_change_pct_previously_priced": round(statistics.median(pct), 1) if pct else None,
        "median_abs_change_pct_previously_priced": round(statistics.median(abs(x) for x in pct), 1) if pct else None,
    }


def plans_to_json(plans: Sequence[PriceUpdate]) -> List[Dict[str, Any]]:
    return [{**asdict(p), "n_dates": p.n_dates} for p in plans]
