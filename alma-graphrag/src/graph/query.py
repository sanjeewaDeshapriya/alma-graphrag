from __future__ import annotations

from typing import List
from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


def build_graph_context(city: str, limit: int = 30) -> str:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        records = session.run(
            """
            MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name: $city})
            OPTIONAL MATCH (h)-[:HAS_AMENITY]->(a:Amenity)
                 OPTIONAL MATCH (h)-[:AFFECTED_BY]->(e:Event)
                 OPTIONAL MATCH (e)-[:MENTIONED_IN]->(n:NewsSignal)
                 OPTIONAL MATCH (h)-[:NEAR]->(l:Location)
            RETURN h, collect(DISTINCT a.name) AS amenities,
                     collect(DISTINCT e.title) AS events,
                     collect(DISTINCT n.title) AS news,
                     collect(DISTINCT l.name) AS locations
            ORDER BY h.rating DESC
            LIMIT $limit
            """,
            {"city": city, "limit": limit},
        ).data()

    lines: List[str] = []
    lines.append(f"City: {city}")
    for row in records:
        h = row["h"]
        amenities = ", ".join(row["amenities"]) if row["amenities"] else "None"
        events = ", ".join(row["events"]) if row["events"] else "None"
        news = ", ".join(row["news"]) if row["news"] else "None"
        locations = (
            ", ".join(row["locations"]) if row.get("locations") else "None"
        )
        lines.append(
            " | ".join(
                [
                    f"Hotel: {h.get('name')}",
                    f"Rating: {h.get('rating')}",
                    f"Price: {h.get('price_range')}",
                    f"Amenities: {amenities}",
                    f"Near: {locations}",
                    f"Events: {events}",
                    f"News: {news}",
                ]
            )
        )

    driver.close()
    return "\n".join(lines)
