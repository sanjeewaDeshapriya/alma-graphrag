from __future__ import annotations

from typing import List
import logging
from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

logger = logging.getLogger("alma.graph")


def _available_relationships(session) -> List[str]:
    res = session.run("CALL db.relationshipTypes()")
    return [row["relationshipType"] for row in res]


def build_graph_context(city: str, limit: int = 30) -> str:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        rels = _available_relationships(session)

        # Build a query that only uses relationship types that exist in the DB
        query_parts = [
            "MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name: $city})",
            "OPTIONAL MATCH (h)-[:HAS_AMENITY]->(a:Amenity)",
        ]

        if "AFFECTED_BY" in rels:
            query_parts.append("OPTIONAL MATCH (h)-[:AFFECTED_BY]->(e:Event)")
        if "MENTIONED_IN" in rels:
            # only add this if events are present; it is harmless otherwise
            query_parts.append("OPTIONAL MATCH (e)-[:MENTIONED_IN]->(n:NewsSignal)")
        if "NEAR" in rels:
            query_parts.append("OPTIONAL MATCH (h)-[:NEAR]->(l:Location)")

        events_expr = "collect(DISTINCT e.title) AS events" if "AFFECTED_BY" in rels else "[] AS events"
        news_expr = "collect(DISTINCT n.title) AS news" if "MENTIONED_IN" in rels else "[] AS news"
        locations_expr = "collect(DISTINCT l.name) AS locations" if "NEAR" in rels else "[] AS locations"

        query = "\n".join(query_parts)
        query += (
            "\nRETURN h, collect(DISTINCT a.name) AS amenities,"
            f"\n         {events_expr},"
            f"\n         {news_expr},"
            f"\n         {locations_expr}"
            "\nORDER BY h.rating DESC"
            "\nLIMIT $limit"
        )

        logger.debug("Built graph query:\n%s", query)

        records = session.run(query, {"city": city, "limit": limit}).data()

    lines: List[str] = []
    lines.append(f"City: {city}")
    for row in records:
        h = row["h"]
        amenities = ", ".join(row.get("amenities") or []) or "None"
        events = ", ".join(row.get("events") or []) or "None"
        news = ", ".join(row.get("news") or []) or "None"
        locations = ", ".join(row.get("locations") or []) or "None"
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
