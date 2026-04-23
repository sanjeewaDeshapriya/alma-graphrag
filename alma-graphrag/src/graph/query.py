"""
Graph query — builds structured context from the Neo4j knowledge graph.

The context string is fed into the CRAG pipeline for LLM grading + generation.
"""
from __future__ import annotations

from typing import List
import logging
from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

logger = logging.getLogger("alma.graph")

# Module-level driver (reused across calls instead of creating per-call)
_driver = None


def _get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver


def _available_relationships(session) -> List[str]:
    res = session.run("CALL db.relationshipTypes()")
    return [row["relationshipType"] for row in res]


def _available_labels(session) -> List[str]:
    res = session.run("CALL db.labels()")
    return [row["label"] for row in res]


def build_graph_context(city: str, limit: int = 30) -> str:
    """
    Build a rich text context from the knowledge graph for a given city.
    Includes hotels, amenities, locations, events, and news signals.
    """
    driver = _get_driver()
    with driver.session() as session:
        rels = _available_relationships(session)
        labels = _available_labels(session)

        # Build a query that only uses relationship types that exist in the DB
        query_parts = [
            "MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name: $city})",
            "OPTIONAL MATCH (h)-[:HAS_AMENITY]->(a:Amenity)",
        ]

        if "AFFECTED_BY" in rels:
            query_parts.append("OPTIONAL MATCH (h)-[:AFFECTED_BY]->(e:Event)")
        if "MENTIONED_IN" in rels:
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

        # Also gather city-level stats
        city_stats = session.run(
            """
            MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name: $city})
            RETURN count(h) AS hotel_count,
                   avg(h.rating) AS avg_rating,
                   min(h.price_per_night_lkr) AS min_price,
                   max(h.price_per_night_lkr) AS max_price,
                   collect(DISTINCT h.price_range) AS price_ranges
            """,
            {"city": city},
        ).single()

        # Gather recent news/events if they exist
        recent_news = []
        if "NewsSignal" in labels:
            news_records = session.run(
                """
                MATCH (n:NewsSignal)
                RETURN n.title AS title, n.source AS source, n.published_at AS published
                ORDER BY n.published_at DESC
                LIMIT 10
                """
            ).data()
            recent_news = news_records

    # --- Format context ---
    lines: List[str] = []

    # City summary
    if city_stats:
        lines.append(f"=== City: {city} ===")
        lines.append(
            f"Total hotels: {city_stats.get('hotel_count', 0)} | "
            f"Avg rating: {city_stats.get('avg_rating', 0):.1f} | "
            f"Price range: {city_stats.get('min_price', '?')}-{city_stats.get('max_price', '?')} LKR | "
            f"Categories: {', '.join(city_stats.get('price_ranges', []))}"
        )
        lines.append("")

    # Hotel details
    for row in records:
        h = row["h"]
        amenities = ", ".join(row.get("amenities") or []) or "None listed"
        events = ", ".join(row.get("events") or []) or "None"
        news = ", ".join(row.get("news") or []) or "None"
        locations = ", ".join(row.get("locations") or []) or "None"

        hotel_line = (
            f"Hotel: {h.get('name')} | "
            f"Rating: {h.get('rating', 'N/A')}/5 | "
            f"Price: {h.get('price_range', 'N/A')} ({h.get('price_per_night_lkr', 'N/A')} LKR/night) | "
            f"Address: {h.get('address', 'N/A')} | "
            f"Amenities: [{amenities}] | "
            f"Near: [{locations}] | "
            f"Events: [{events}] | "
            f"News: [{news}]"
        )
        lines.append(hotel_line)

    # Recent news section
    if recent_news:
        lines.append("")
        lines.append("=== Recent News & Events ===")
        for n in recent_news:
            lines.append(f"- {n.get('title', 'N/A')} (source: {n.get('source', '?')}, date: {n.get('published', '?')})")

    context = "\n".join(lines)
    logger.info("Graph context built: %d hotels, %d chars", len(records), len(context))
    return context
