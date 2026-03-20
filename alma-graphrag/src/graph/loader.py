from __future__ import annotations

from typing import Iterable, Dict, Any, List
from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


class GraphLoader:
    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def close(self) -> None:
        self.driver.close()

    def upsert_hotel(self, hotel: Dict[str, Any]) -> None:
        with self.driver.session() as session:
            session.run(
                """
                MERGE (h:Hotel {id: $id})
                SET h.name = $name,
                    h.description = $description,
                    h.rating = $rating,
                    h.price_range = $price_range,
                    h.price_per_night_lkr = $price_per_night_lkr,
                    h.address = $address,
                    h.city_name = $city_name,
                    h.lat = $lat,
                    h.lng = $lng,
                    h.phone = $phone,
                    h.website = $website,
                    h.source = $source,
                    h.last_updated = $last_updated
                """,
                hotel,
            )

            session.run(
                """
                MATCH (h:Hotel {id: $id})
                MATCH (c:City {name: $city_name})
                MERGE (h)-[r:LOCATED_IN]->(c)
                SET r.distance_from_center_km = $distance_from_center_km,
                    r.confidence = $confidence
                """,
                {
                    "id": hotel["id"],
                    "city_name": hotel["city_name"],
                    "distance_from_center_km": hotel.get(
                        "distance_from_center_km", 0.5
                    ),
                    "confidence": 0.9,
                },
            )

    def upsert_amenities(self, hotel_id: str, amenities: Iterable[str]) -> None:
        with self.driver.session() as session:
            for name in amenities:
                session.run(
                    """
                    MERGE (a:Amenity {name: $name})
                    ON CREATE SET a.category = $category
                    """,
                    {"name": name, "category": "General"},
                )
                session.run(
                    """
                    MATCH (h:Hotel {id: $hotel_id})
                    MATCH (a:Amenity {name: $name})
                    MERGE (h)-[r:HAS_AMENITY]->(a)
                    SET r.availability = true, r.confidence = 0.8
                    """,
                    {"hotel_id": hotel_id, "name": name},
                )

    def upsert_locations(self, hotel_id: str, locations: List[Dict[str, Any]]) -> None:
        with self.driver.session() as session:
            for loc in locations:
                session.run(
                    """
                    MERGE (l:Location {name: $name})
                    SET l.type = $type,
                        l.lat = $lat,
                        l.lng = $lng
                    """,
                    {
                        "name": loc.get("name"),
                        "type": loc.get("type", "landmark"),
                        "lat": loc.get("lat", 0.0),
                        "lng": loc.get("lng", 0.0),
                    },
                )
                session.run(
                    """
                    MATCH (h:Hotel {id: $hotel_id})
                    MATCH (l:Location {name: $name})
                    MERGE (h)-[r:NEAR]->(l)
                    SET r.distance_km = $distance_km,
                        r.walk_time_min = $walk_time_min,
                        r.transport_cost_lkr = $transport_cost_lkr
                    """,
                    {
                        "hotel_id": hotel_id,
                        "name": loc.get("name"),
                        "distance_km": loc.get("distance_km", 1.0),
                        "walk_time_min": loc.get("walk_time_min"),
                        "transport_cost_lkr": loc.get("transport_cost_lkr"),
                    },
                )

    def upsert_news_signal(self, news: Dict[str, Any]) -> None:
        with self.driver.session() as session:
            session.run(
                """
                MERGE (n:NewsSignal {url: $url})
                SET n.id = $id,
                    n.title = $title,
                    n.summary = $summary,
                    n.published_at = $published_at,
                    n.source = $source
                """,
                news,
            )

    def upsert_event(self, event: Dict[str, Any]) -> None:
        with self.driver.session() as session:
            session.run(
                """
                MERGE (e:Event {id: $id})
                SET e.title = $title,
                    e.type = $type,
                    e.start_time = $start_time,
                    e.end_time = $end_time,
                    e.severity = $severity,
                    e.source = $source
                """,
                event,
            )

    def link_event_news(self, event_id: str, news_url: str) -> None:
        with self.driver.session() as session:
            session.run(
                """
                MATCH (e:Event {id: $event_id})
                MATCH (n:NewsSignal {url: $news_url})
                MERGE (e)-[r:MENTIONED_IN]->(n)
                SET r.extract_confidence = 0.7
                """,
                {"event_id": event_id, "news_url": news_url},
            )

    def link_hotel_event(self, hotel_id: str, event_id: str) -> None:
        with self.driver.session() as session:
            session.run(
                """
                MATCH (h:Hotel {id: $hotel_id})
                MATCH (e:Event {id: $event_id})
                MERGE (h)-[r:AFFECTED_BY]->(e)
                SET r.impact_score = 0.5, r.impact_type = 'news'
                """,
                {"hotel_id": hotel_id, "event_id": event_id},
            )
