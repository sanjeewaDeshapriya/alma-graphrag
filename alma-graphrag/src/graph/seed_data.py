from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


@dataclass
class CitySeed:
    name: str
    district: str
    province: str
    lat: float
    lng: float


def _seed_districts(session) -> None:
    districts = [
        ("Colombo", "Western"),
        ("Kalutara", "Western"),
        ("Gampaha", "Western"),
    ]
    for name, province in districts:
        session.run(
            "MERGE (d:District {name: $name}) SET d.province = $province",
            {"name": name, "province": province},
        )


def _seed_cities(session, cities: Iterable[CitySeed]) -> None:
    for city in cities:
        session.run(
            """
            MERGE (c:City {name: $name})
            SET c.district = $district,
                c.province = $province,
                c.lat = $lat,
                c.lng = $lng
            """,
            city.__dict__,
        )
        session.run(
            """
            MATCH (c:City {name: $city})
            MATCH (d:District {name: $district})
            MERGE (c)-[:IN_DISTRICT]->(d)
            """,
            {"city": city.name, "district": city.district},
        )


def _seed_amenities(session) -> None:
    amenity_groups = {
        "Connectivity": ["Free WiFi", "Business Center"],
        "Recreation": ["Swimming Pool", "Fitness Center", "Spa"],
        "Dining": ["Restaurant", "Bar", "Room Service"],
        "Transport": ["Free Parking", "Airport Shuttle"],
        "Comfort": ["Air Conditioning", "Hot Water", "24h Front Desk"],
        "Safety": ["CCTV", "Security Guard"],
    }
    for category, names in amenity_groups.items():
        for name in names:
            session.run(
                "MERGE (a:Amenity {name: $name}) SET a.category = $category",
                {"name": name, "category": category},
            )


def seed_all() -> None:
    cities = [
        CitySeed("Piliyandala", "Colombo", "Western", 6.8011, 79.9169),
        CitySeed("Maharagama", "Colombo", "Western", 6.8483, 79.9260),
        CitySeed("Homagama", "Colombo", "Western", 6.8424, 80.0023),
        CitySeed("Colombo", "Colombo", "Western", 6.9271, 79.8612),
        CitySeed("Kesbewa", "Colombo", "Western", 6.7833, 79.9333),
        CitySeed("Moratuwa", "Colombo", "Western", 6.7728, 79.8820),
        CitySeed("Dehiwala", "Colombo", "Western", 6.8516, 79.8715),
    ]

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        _seed_districts(session)
        _seed_cities(session, cities)
        _seed_amenities(session)
    driver.close()
