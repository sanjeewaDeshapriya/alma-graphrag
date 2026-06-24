"""
Script: Ingest traffic data from configured providers (Google Maps / TomTom).

Usage:
    python scripts/run_ingest_traffic.py
    python scripts/run_ingest_traffic.py --provider google
    python scripts/run_ingest_traffic.py --provider tomtom
    python scripts/run_ingest_traffic.py --city Piliyandala
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import logging

from neo4j import GraphDatabase
from src.config import (
    HOTELS_CITIES,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    TRAFFIC_PROVIDER,
)
from src.ingest.traffic import fetch_all_traffic
from src.ingest.traffic_linker import link_traffic_to_hotels, cleanup_stale_signals

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alma.scripts.traffic")


def _fetch_graph_data(cities: list[str]) -> tuple[list[dict], list[dict]]:
    """Read city coords and hotel coords from Neo4j."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    city_data: list[dict] = []
    hotel_data: list[dict] = []

    with driver.session() as session:
        for city_name in cities:
            # City center coords (use average of hotel coords if city node has none)
            city_row = session.run(
                """
                MATCH (c:City {name: $name})
                RETURN c.name AS name, c.lat AS lat, c.lng AS lng
                """,
                {"name": city_name},
            ).single()

            city_lat = city_row["lat"] if city_row and city_row["lat"] else None
            city_lng = city_row["lng"] if city_row and city_row["lng"] else None

            # Hotels in this city
            hotel_rows = session.run(
                """
                MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name: $city})
                RETURN h.id AS id, h.name AS name, h.lat AS lat, h.lng AS lng, c.name AS city_name
                """,
                {"city": city_name},
            ).data()

            if not city_lat and hotel_rows:
                lats = [r["lat"] for r in hotel_rows if r.get("lat")]
                lngs = [r["lng"] for r in hotel_rows if r.get("lng")]
                if lats and lngs:
                    city_lat = sum(lats) / len(lats)
                    city_lng = sum(lngs) / len(lngs)

            if city_lat and city_lng:
                city_data.append({"name": city_name, "lat": city_lat, "lng": city_lng})

            hotel_data.extend(hotel_rows)

    driver.close()
    return city_data, hotel_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest traffic data into the ALMA knowledge graph")
    parser.add_argument("--city", type=str, default=None, help="Single city to process (default: all configured)")
    parser.add_argument("--provider", type=str, default=None, help="Provider override: google | tomtom | both")
    parser.add_argument("--cleanup", action="store_true", help="Only run stale signal cleanup")
    args = parser.parse_args()

    if args.cleanup:
        deleted = cleanup_stale_signals()
        print(f"Cleaned up {deleted} stale traffic signals.")
        return

    cities = [args.city] if args.city else list(HOTELS_CITIES)
    provider = args.provider or TRAFFIC_PROVIDER

    print(f"\n=== Traffic Ingestion ===")
    print(f"Cities: {', '.join(cities)}")
    print(f"Provider: {provider}\n")

    city_data, hotel_data = _fetch_graph_data(cities)
    print(f"Found {len(city_data)} cities with coords, {len(hotel_data)} hotels in graph")

    if not hotel_data:
        print("No hotels found in graph — run hotel ingestion first.")
        return

    traffic_data = fetch_all_traffic(city_data, hotel_data, provider=provider)

    print(f"\nFetched:")
    print(f"  Distances: {len(traffic_data.get('distances', []))}")
    print(f"  Signals:   {len(traffic_data.get('signals', []))}")
    print(f"  Incidents: {len(traffic_data.get('incidents', []))}")

    counts = link_traffic_to_hotels(traffic_data, hotel_data)

    print(f"\nLinked to graph:")
    print(f"  Distance updates: {counts['distances']}")
    print(f"  Signal links:     {counts['signals']}")
    print(f"  Incident links:   {counts['incidents']}")

    # Cleanup old signals
    deleted = cleanup_stale_signals()
    if deleted:
        print(f"  Stale cleanup:    {deleted} removed")

    total = sum(counts.values())
    print(f"\nTotal traffic items ingested: {total}")


if __name__ == "__main__":
    main()
