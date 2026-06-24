"""
Links traffic data (distances, signals, incidents) to hotels in the graph.

Mirrors the pattern of ``event_linker.py`` — receives normalised traffic data
and creates/updates Neo4j nodes and relationships via the GraphLoader.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.config import TRAFFIC_RADIUS_KM, TRAFFIC_SIGNAL_TTL_HOURS
from src.graph.loader import GraphLoader
from src.ingest.traffic import haversine

logger = logging.getLogger("alma.ingest.traffic_linker")


def link_traffic_to_hotels(
    traffic_data: Dict[str, Any],
    hotels: List[Dict[str, Any]],
) -> Dict[str, int]:
    """Link traffic distances, signals, and incidents to hotel nodes.

    Parameters
    ----------
    traffic_data : dict with keys ``distances``, ``signals``, ``incidents``
    hotels : list of hotel dicts with ``id``, ``lat``, ``lng``, ``city_name``

    Returns
    -------
    dict with counts of linked items
    """
    loader = GraphLoader()
    counts = {"distances": 0, "signals": 0, "incidents": 0}

    try:
        # 1. Distance records → enrich LOCATED_IN edges
        for dist in traffic_data.get("distances", []):
            hotel_id = dist.get("hotel_id")
            city_name = dist.get("origin_name")
            if not hotel_id or not city_name:
                continue
            loader.update_located_in_travel_time(
                hotel_id=hotel_id,
                city_name=city_name,
                distance_km=dist.get("distance_km", 0.0),
                travel_time_min=dist.get("duration_min", 0.0),
                travel_time_traffic_min=dist.get("duration_in_traffic_min"),
            )
            counts["distances"] += 1

        # 2. Traffic signals → TrafficSignal nodes linked to nearby hotels
        for signal in traffic_data.get("signals", []):
            loader.upsert_traffic_signal(signal)

            sig_lat = signal.get("lat", 0.0)
            sig_lng = signal.get("lng", 0.0)

            for h in hotels:
                h_lat, h_lng = h.get("lat"), h.get("lng")
                if not h_lat or not h_lng:
                    continue
                if sig_lat == 0.0 and sig_lng == 0.0:
                    # Google-derived signal without coords — match by hotel_id in signal metadata
                    if signal.get("location_name", "").endswith(h.get("name", "__none__")):
                        loader.link_hotel_traffic_signal(
                            h["id"], signal["id"], signal.get("severity", "unknown")
                        )
                        counts["signals"] += 1
                    continue

                dist = haversine(sig_lat, sig_lng, h_lat, h_lng)
                if dist <= TRAFFIC_RADIUS_KM:
                    loader.link_hotel_traffic_signal(
                        h["id"], signal["id"], signal.get("severity", "unknown")
                    )
                    counts["signals"] += 1

        # 3. Incidents → Event nodes linked to nearby hotels via AFFECTED_BY
        for incident in traffic_data.get("incidents", []):
            event = {
                "id": incident["id"],
                "title": incident.get("title", "Traffic Incident")[:200],
                "type": "traffic_incident",
                "start_time": incident.get("start_time", ""),
                "end_time": incident.get("end_time", ""),
                "severity": incident.get("severity", "unknown"),
                "source": incident.get("source", "tomtom"),
            }
            loader.upsert_event(event)

            inc_lat = incident.get("lat", 0.0)
            inc_lng = incident.get("lng", 0.0)
            if inc_lat == 0.0 and inc_lng == 0.0:
                continue

            for h in hotels:
                h_lat, h_lng = h.get("lat"), h.get("lng")
                if not h_lat or not h_lng:
                    continue
                dist = haversine(inc_lat, inc_lng, h_lat, h_lng)
                if dist <= TRAFFIC_RADIUS_KM:
                    loader.link_hotel_event(h["id"], incident["id"], impact_type="traffic")
                    counts["incidents"] += 1

    finally:
        loader.close()

    logger.info(
        "Traffic linking complete: %d distances, %d signal links, %d incident links",
        counts["distances"], counts["signals"], counts["incidents"],
    )
    return counts


def cleanup_stale_signals() -> int:
    loader = GraphLoader()
    try:
        deleted = loader.cleanup_stale_traffic(TRAFFIC_SIGNAL_TTL_HOURS)
        if deleted:
            logger.info("Cleaned up %d stale TrafficSignal nodes", deleted)
        return deleted
    finally:
        loader.close()
