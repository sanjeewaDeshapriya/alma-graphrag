"""
Traffic data ingestion — Google Maps Directions/Distance Matrix (primary)
and TomTom Traffic Flow/Incidents (secondary).

Provider is selected via TRAFFIC_PROVIDER env var: google | tomtom | both.
"""
from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from math import radians, sin, cos, sqrt, atan2
from typing import Any, Dict, List, Optional

import httpx

from src.config import (
    GOOGLE_MAPS_API_KEY,
    TOMTOM_API_KEY,
    TRAFFIC_MAX_HOTELS_PER_BATCH,
    TRAFFIC_PROVIDER,
)

logger = logging.getLogger("alma.ingest.traffic")

GOOGLE_DISTANCE_MATRIX_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"
GOOGLE_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
TOMTOM_FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
TOMTOM_INCIDENTS_URL = "https://api.tomtom.com/traffic/services/5/incidentDetails"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id(*parts: str) -> str:
    raw = ":".join(str(p) for p in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_severity(current: float, free_flow: float) -> str:
    if free_flow <= 0:
        return "unknown"
    ratio = current / free_flow
    if ratio < 0.3:
        return "heavy"
    if ratio < 0.7:
        return "moderate"
    return "light"


def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    la1, lo1, la2, lo2 = map(radians, [lat1, lng1, lat2, lng2])
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = sin(dlat / 2) ** 2 + cos(la1) * cos(la2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ---------------------------------------------------------------------------
# Google Maps provider
# ---------------------------------------------------------------------------

def fetch_distance_matrix(
    origins: List[Dict[str, float]],
    destinations: List[Dict[str, float]],
    api_key: str | None = None,
) -> List[Dict[str, Any]]:
    """Batch city→hotel distances via Google Distance Matrix API.

    `origins` / `destinations` are lists of ``{"lat": ..., "lng": ..., "id": ..., "name": ...}``.
    Returns a flat list of distance records.
    """
    key = api_key or GOOGLE_MAPS_API_KEY
    if not key:
        logger.warning("GOOGLE_MAPS_API_KEY not set — skipping distance matrix")
        return []

    results: List[Dict[str, Any]] = []
    batch_size = TRAFFIC_MAX_HOTELS_PER_BATCH

    client = httpx.Client(timeout=30)
    try:
        for o_start in range(0, len(origins), batch_size):
            o_batch = origins[o_start : o_start + batch_size]
            for d_start in range(0, len(destinations), batch_size):
                d_batch = destinations[d_start : d_start + batch_size]

                o_str = "|".join(f"{o['lat']},{o['lng']}" for o in o_batch)
                d_str = "|".join(f"{d['lat']},{d['lng']}" for d in d_batch)

                params = {
                    "origins": o_str,
                    "destinations": d_str,
                    "mode": "driving",
                    "departure_time": "now",
                    "key": key,
                }
                resp = client.get(GOOGLE_DISTANCE_MATRIX_URL, params=params)
                data = resp.json()

                if data.get("status") != "OK":
                    logger.warning("Distance Matrix API error: %s", data.get("status"))
                    continue

                rows = data.get("rows", [])
                for oi, row in enumerate(rows):
                    origin = o_batch[oi]
                    for di, elem in enumerate(row.get("elements", [])):
                        dest = d_batch[di]
                        if elem.get("status") != "OK":
                            continue
                        distance_m = elem.get("distance", {}).get("value", 0)
                        duration_s = elem.get("duration", {}).get("value", 0)
                        duration_traffic_s = elem.get("duration_in_traffic", {}).get("value")

                        results.append({
                            "origin_id": origin.get("id", origin.get("name", "")),
                            "origin_name": origin.get("name", ""),
                            "hotel_id": dest.get("id", ""),
                            "hotel_name": dest.get("name", ""),
                            "distance_km": round(distance_m / 1000, 2),
                            "duration_min": round(duration_s / 60, 1),
                            "duration_in_traffic_min": round(duration_traffic_s / 60, 1) if duration_traffic_s else None,
                            "source": "google_maps",
                            "timestamp": _now_iso(),
                        })

                time.sleep(0.3)
    finally:
        client.close()

    logger.info("Google Distance Matrix: %d records fetched", len(results))
    return results


def fetch_directions(
    origin_lat: float,
    origin_lng: float,
    dest_lat: float,
    dest_lng: float,
    api_key: str | None = None,
) -> Optional[Dict[str, Any]]:
    """Single route via Google Directions API with traffic-aware duration."""
    key = api_key or GOOGLE_MAPS_API_KEY
    if not key:
        return None

    client = httpx.Client(timeout=30)
    try:
        params = {
            "origin": f"{origin_lat},{origin_lng}",
            "destination": f"{dest_lat},{dest_lng}",
            "mode": "driving",
            "departure_time": "now",
            "key": key,
        }
        resp = client.get(GOOGLE_DIRECTIONS_URL, params=params)
        data = resp.json()

        if data.get("status") != "OK":
            logger.warning("Directions API error: %s", data.get("status"))
            return None

        route = data.get("routes", [{}])[0]
        leg = route.get("legs", [{}])[0]

        distance_m = leg.get("distance", {}).get("value", 0)
        duration_s = leg.get("duration", {}).get("value", 0)
        duration_traffic_s = leg.get("duration_in_traffic", {}).get("value")
        polyline = route.get("overview_polyline", {}).get("points", "")
        warnings = route.get("warnings", [])

        return {
            "distance_km": round(distance_m / 1000, 2),
            "duration_min": round(duration_s / 60, 1),
            "duration_in_traffic_min": round(duration_traffic_s / 60, 1) if duration_traffic_s else None,
            "polyline": polyline,
            "warnings": warnings,
            "source": "google_maps",
            "timestamp": _now_iso(),
        }
    finally:
        client.close()


# ---------------------------------------------------------------------------
# TomTom provider
# ---------------------------------------------------------------------------

def fetch_traffic_flow(
    lat: float,
    lng: float,
    api_key: str | None = None,
) -> Optional[Dict[str, Any]]:
    """Real-time congestion at a point via TomTom Flow Segment Data."""
    key = api_key or TOMTOM_API_KEY
    if not key:
        return None

    client = httpx.Client(timeout=15)
    try:
        params = {"point": f"{lat},{lng}", "key": key, "unit": "KMPH"}
        resp = client.get(TOMTOM_FLOW_URL, params=params)
        if resp.status_code != 200:
            logger.warning("TomTom Flow API %d: %s", resp.status_code, resp.text[:200])
            return None

        data = resp.json()
        flow = data.get("flowSegmentData", {})
        current_speed = flow.get("currentSpeed", 0)
        free_flow_speed = flow.get("freeFlowSpeed", 0)
        confidence = flow.get("confidence", 0)

        severity = classify_severity(current_speed, free_flow_speed)
        congestion = round(current_speed / free_flow_speed, 3) if free_flow_speed > 0 else 1.0

        eta_change = 0.0
        if free_flow_speed > 0 and current_speed > 0:
            eta_change = round((1.0 / current_speed - 1.0 / free_flow_speed) * 60, 1)

        return {
            "id": _make_id("tomtom_flow", str(lat), str(lng), _now_iso()[:13]),
            "timestamp": _now_iso(),
            "location_name": f"{lat:.4f},{lng:.4f}",
            "severity": severity,
            "eta_change_min": max(0, eta_change),
            "lat": lat,
            "lng": lng,
            "source": "tomtom",
            "congestion_ratio": congestion,
            "current_speed": current_speed,
            "free_flow_speed": free_flow_speed,
            "confidence": confidence,
        }
    finally:
        client.close()


def fetch_traffic_incidents(
    min_lat: float,
    min_lng: float,
    max_lat: float,
    max_lng: float,
    api_key: str | None = None,
) -> List[Dict[str, Any]]:
    """Traffic incidents in a bounding box via TomTom Incident Details."""
    key = api_key or TOMTOM_API_KEY
    if not key:
        logger.warning("TOMTOM_API_KEY not set — skipping incidents")
        return []

    bbox = f"{min_lat},{min_lng},{max_lat},{max_lng}"
    client = httpx.Client(timeout=15)
    try:
        params = {
            "bbox": bbox,
            "key": key,
            "fields": "{incidents{type,geometry{type,coordinates},properties{id,iconCategory,magnitudeOfDelay,events{description,code},startTime,endTime,from,to,length,delay,roadNumbers}}}",
            "language": "en-US",
            "categoryFilter": "1,2,3,4,5,6,7,8,9,10,11,14",
        }
        resp = client.get(TOMTOM_INCIDENTS_URL, params=params)
        if resp.status_code != 200:
            logger.warning("TomTom Incidents API %d: %s", resp.status_code, resp.text[:200])
            return []

        data = resp.json()
        raw_incidents = data.get("incidents", [])

        CATEGORY_MAP = {
            0: "UNKNOWN", 1: "ACCIDENT", 2: "FOG", 3: "DANGEROUS_CONDITIONS",
            4: "RAIN", 5: "ICE", 6: "JAM", 7: "LANE_CLOSED", 8: "ROAD_CLOSURE",
            9: "ROAD_WORKS", 10: "WIND", 11: "FLOODING", 14: "BROKEN_DOWN_VEHICLE",
        }

        results: List[Dict[str, Any]] = []
        for inc in raw_incidents:
            props = inc.get("properties", {})
            geom = inc.get("geometry", {})
            coords = geom.get("coordinates", [])

            inc_lat, inc_lng = 0.0, 0.0
            if coords:
                first = coords[0] if geom.get("type") == "LineString" else coords
                if isinstance(first, list) and len(first) >= 2:
                    inc_lng, inc_lat = float(first[0]), float(first[1])

            category = props.get("iconCategory", 0)
            incident_type = CATEGORY_MAP.get(category, "UNKNOWN")
            events = props.get("events", [])
            description = events[0].get("description", "") if events else ""
            delay = props.get("delay", 0) or 0
            magnitude = props.get("magnitudeOfDelay", 0) or 0
            severity_map = {0: "unknown", 1: "minor", 2: "moderate", 3: "major", 4: "undefined"}
            severity = severity_map.get(magnitude, "unknown")
            from_road = props.get("from", "")
            to_road = props.get("to", "")

            inc_id = props.get("id") or _make_id("tomtom_inc", str(inc_lat), str(inc_lng), description[:50])

            results.append({
                "id": str(inc_id),
                "title": f"{incident_type}: {from_road} to {to_road}".strip(": "),
                "type": "traffic_incident",
                "incident_type": incident_type,
                "severity": severity,
                "description": description[:500],
                "lat": inc_lat,
                "lng": inc_lng,
                "start_time": props.get("startTime", _now_iso()),
                "end_time": props.get("endTime", ""),
                "delay_seconds": delay,
                "source": "tomtom",
            })

        logger.info("TomTom Incidents: %d incidents in bbox %s", len(results), bbox)
        return results
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def fetch_all_traffic(
    cities: List[Dict[str, Any]],
    hotels: List[Dict[str, Any]],
    provider: str | None = None,
) -> Dict[str, Any]:
    """Fetch traffic data from configured providers.

    Parameters
    ----------
    cities : list of dicts with keys ``id/name``, ``lat``, ``lng``
    hotels : list of dicts with keys ``id``, ``name``, ``lat``, ``lng``, ``city_name``
    provider : override for TRAFFIC_PROVIDER config

    Returns
    -------
    dict with keys ``distances``, ``signals``, ``incidents``
    """
    prov = (provider or TRAFFIC_PROVIDER).lower()
    result: Dict[str, Any] = {"distances": [], "signals": [], "incidents": []}

    use_google = prov in ("google", "both")
    use_tomtom = prov in ("tomtom", "both")

    # --- Google: Distance Matrix for city→hotel travel times ---------------
    if use_google and GOOGLE_MAPS_API_KEY:
        origins = [{"id": c.get("name", ""), "name": c.get("name", ""), "lat": c["lat"], "lng": c["lng"]} for c in cities if c.get("lat") and c.get("lng")]
        dests = [{"id": h["id"], "name": h.get("name", ""), "lat": h["lat"], "lng": h["lng"]} for h in hotels if h.get("lat") and h.get("lng")]

        if origins and dests:
            result["distances"] = fetch_distance_matrix(origins, dests)

    # --- TomTom: Flow for each hotel location ------------------------------
    if use_tomtom and TOMTOM_API_KEY:
        for h in hotels:
            lat, lng = h.get("lat"), h.get("lng")
            if not lat or not lng:
                continue
            flow = fetch_traffic_flow(lat, lng)
            if flow:
                result["signals"].append(flow)
            time.sleep(0.2)

        # Incidents: build bounding box over all hotels
        hotel_lats = [h["lat"] for h in hotels if h.get("lat")]
        hotel_lngs = [h["lng"] for h in hotels if h.get("lng")]
        if hotel_lats and hotel_lngs:
            margin = 0.05
            result["incidents"] = fetch_traffic_incidents(
                min(hotel_lats) - margin,
                min(hotel_lngs) - margin,
                max(hotel_lats) + margin,
                max(hotel_lngs) + margin,
            )

    # --- Google-only: generate synthetic traffic signals from duration diff -
    if use_google and not use_tomtom:
        hotel_coords = {h["id"]: (h.get("lat", 0.0), h.get("lng", 0.0)) for h in hotels if h.get("id")}
        for d in result["distances"]:
            traffic_min = d.get("duration_in_traffic_min")
            normal_min = d.get("duration_min")
            if traffic_min and normal_min and traffic_min > 0:
                ratio = normal_min / traffic_min
                severity = classify_severity(ratio, 1.0)
                eta_diff = max(0, traffic_min - normal_min)
                h_lat, h_lng = hotel_coords.get(d.get("hotel_id", ""), (0.0, 0.0))
                result["signals"].append({
                    "id": _make_id("google_traffic", d.get("hotel_id", ""), d.get("origin_name", ""), _now_iso()[:13]),
                    # Google signals are per-route (city -> this hotel), so carry
                    # the destination hotel id for a 1:1 link in the linker.
                    "hotel_id": d.get("hotel_id", ""),
                    "timestamp": _now_iso(),
                    "location_name": f"{d.get('origin_name', '?')} -> {d.get('hotel_name', '?')}",
                    "severity": severity,
                    "eta_change_min": round(eta_diff, 1),
                    "lat": h_lat or 0.0,
                    "lng": h_lng or 0.0,
                    "source": "google_maps",
                    "congestion_ratio": round(ratio, 3),
                })

    total = len(result["distances"]) + len(result["signals"]) + len(result["incidents"])
    logger.info(
        "Traffic fetch complete (provider=%s): %d distances, %d signals, %d incidents",
        prov, len(result["distances"]), len(result["signals"]), len(result["incidents"]),
    )
    return result
