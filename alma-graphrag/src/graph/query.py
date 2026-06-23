"""
Graph query — builds structured context from the Neo4j knowledge graph.

The context string is fed into the CRAG pipeline for LLM grading + generation.
"""
from __future__ import annotations

from typing import Any, Dict, List
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
            "MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)",
            "WHERE toLower(c.name) = toLower($city)",
            "OPTIONAL MATCH (h)-[:HAS_AMENITY]->(a:Amenity)",
        ]

        if "AFFECTED_BY" in rels:
            query_parts.append("OPTIONAL MATCH (h)-[:AFFECTED_BY]->(e:Event)")
        if "MENTIONED_IN" in rels:
            query_parts.append("OPTIONAL MATCH (e)-[:MENTIONED_IN]->(n:NewsSignal)")
        if "NEAR" in rels:
            query_parts.append("OPTIONAL MATCH (h)-[:NEAR]->(l:Location)")
        if "HAS_ROOM" in rels:
            query_parts.append("OPTIONAL MATCH (h)-[:HAS_ROOM]->(rt:RoomType)")
        if "OFFERS_BOARD" in rels:
            query_parts.append("OPTIONAL MATCH (h)-[:OFFERS_BOARD]->(b:BoardType)")

        events_expr = "collect(DISTINCT e.title) AS events" if "AFFECTED_BY" in rels else "[] AS events"
        news_expr = "collect(DISTINCT n.title) AS news" if "MENTIONED_IN" in rels else "[] AS news"
        locations_expr = "collect(DISTINCT l.name) AS locations" if "NEAR" in rels else "[] AS locations"
        rooms_expr = (
            "collect(DISTINCT {name: rt.name, price: rt.price, currency: rt.currency, "
            "board: rt.board_name, refundable: rt.refundable}) AS rooms"
            if "HAS_ROOM" in rels
            else "[] AS rooms"
        )
        boards_expr = "collect(DISTINCT b.name) AS boards" if "OFFERS_BOARD" in rels else "[] AS boards"

        query = "\n".join(query_parts)
        query += (
            "\nRETURN h, collect(DISTINCT a.name) AS amenities,"
            f"\n         {events_expr},"
            f"\n         {news_expr},"
            f"\n         {locations_expr},"
            f"\n         {rooms_expr},"
            f"\n         {boards_expr}"
            "\nORDER BY h.rating DESC"
            "\nLIMIT $limit"
        )

        logger.debug("Built graph query:\n%s", query)

        records = session.run(query, {"city": city, "limit": limit}).data()

        # Also gather city-level stats
        city_stats = session.run(
            """
                 MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)
                 WHERE toLower(c.name) = toLower($city)
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
        avg_rating = city_stats.get("avg_rating") or 0.0
        lines.append(f"=== City: {city} ===")
        lines.append(
            f"Total hotels: {city_stats.get('hotel_count', 0) or 0} | "
            f"Avg rating: {avg_rating:.1f} | "
            f"Price range: {city_stats.get('min_price') or '?'}-{city_stats.get('max_price') or '?'} LKR | "
            f"Categories: {', '.join([p for p in (city_stats.get('price_ranges') or []) if p])}"
        )
        lines.append("")

    # Hotel details
    for row in records:
        h = row["h"]
        amenities = ", ".join(row.get("amenities") or []) or "None listed"
        events = ", ".join(row.get("events") or []) or "None"
        news = ", ".join(row.get("news") or []) or "None"
        locations = ", ".join(row.get("locations") or []) or "None"
        boards = ", ".join([b for b in (row.get("boards") or []) if b]) or "None"

        # Format LiteAPI room/rate plans, if any.
        rooms_raw = [r for r in (row.get("rooms") or []) if r and r.get("name")]
        if rooms_raw:
            room_strs = []
            for r in rooms_raw:
                price = r.get("price")
                cur = r.get("currency") or ""
                refund = "refundable" if r.get("refundable") else "non-refundable"
                price_str = f"{price:.0f} {cur}".strip() if price else "price N/A"
                room_strs.append(
                    f"{r.get('name')} ({price_str}, {r.get('board') or 'RO'}, {refund})"
                )
            rooms = "; ".join(room_strs)
        else:
            rooms = "None"

        star = h.get("star_rating")
        star_str = f"{star:.0f}-star | " if star else ""
        source = h.get("source", "?")

        hotel_line = (
            f"Hotel: {h.get('name')} [{source}] | "
            f"{star_str}"
            f"Rating: {h.get('rating', 'N/A')}/5 | "
            f"Price: {h.get('price_range', 'N/A')} ({h.get('price_per_night_lkr', 'N/A')} LKR/night) | "
            f"Address: {h.get('address', 'N/A')} | "
            f"Amenities: [{amenities}] | "
            f"Rooms: [{rooms}] | "
            f"Board: [{boards}] | "
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


def _safe_primitive(value: Any) -> Any:
    """Keep API payload JSON-safe for frontend rendering."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _sanitize_properties(props: Dict[str, Any]) -> Dict[str, Any]:
    """Remove oversized fields and keep values JSON-safe for UI payloads."""
    cleaned: Dict[str, Any] = {}
    for key, value in props.items():
        if key == "embedding":
            continue
        cleaned[key] = _safe_primitive(value)
    return cleaned


def get_graph_overview(city: str | None = None) -> Dict[str, Any]:
    """Return label/relationship counts and key city-level stats for the dashboard."""
    driver = _get_driver()
    with driver.session() as session:
        label_rows = session.run(
            """
            MATCH (n)
            UNWIND labels(n) AS label
            RETURN label, count(*) AS count
            ORDER BY count DESC
            """
        ).data()
        rel_rows = session.run(
            """
            MATCH ()-[r]->()
            RETURN type(r) AS type, count(*) AS count
            ORDER BY count DESC
            """
        ).data()

        city_stats = None
        if city:
            city_stats = session.run(
                """
                MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)
                WHERE toLower(c.name) = toLower($city)
                RETURN count(h) AS hotel_count,
                       avg(h.rating) AS avg_rating,
                       min(h.price_per_night_lkr) AS min_price,
                       max(h.price_per_night_lkr) AS max_price
                """,
                {"city": city},
            ).single()

    return {
        "labels": [
            {"label": row["label"], "count": int(row["count"])} for row in label_rows
        ],
        "relationships": [
            {"type": row["type"], "count": int(row["count"])} for row in rel_rows
        ],
        "city": city,
        "city_stats": {
            "hotel_count": int(city_stats["hotel_count"] or 0),
            "avg_rating": float(city_stats["avg_rating"] or 0),
            "min_price": _safe_primitive(city_stats["min_price"]),
            "max_price": _safe_primitive(city_stats["max_price"]),
        }
        if city_stats
        else None,
    }


def get_graph_network(city: str | None = None, limit: int = 180) -> Dict[str, Any]:
    """Return graph nodes/edges for interactive visualization."""
    driver = _get_driver()
    with driver.session() as session:
        if city:
            rows = list(session.run(
                """
                MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)
                WHERE toLower(c.name) = toLower($city)
                WITH h LIMIT $limit
                OPTIONAL MATCH (h)-[r]-(m)
                RETURN h AS a, r AS rel, m AS b
                """,
                {"city": city, "limit": limit},
            ))
        else:
            rows = list(session.run(
                """
                MATCH (a)-[r]->(b)
                RETURN a AS a, r AS rel, b AS b
                LIMIT $limit
                """,
                {"limit": limit},
            ))

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    def _node_id(node: Any) -> str | None:
        if node is None:
            return None
        element_id = getattr(node, "element_id", None)
        if element_id:
            return str(element_id)
        if isinstance(node, dict):
            return str(node.get("element_id") or node.get("id") or node.get("name") or "") or None
        return None

    def _node_labels(node: Any) -> List[str]:
        if node is None:
            return []
        labels = getattr(node, "labels", None)
        if labels is not None:
            return list(labels)
        if isinstance(node, dict):
            raw = node.get("labels")
            if isinstance(raw, list):
                return [str(x) for x in raw]
        return ["Node"]

    def _node_props(node: Any) -> Dict[str, Any]:
        if node is None:
            return {}
        if isinstance(node, dict):
            base = node
        else:
            base = dict(node)
        return _sanitize_properties(base)

    def _rel_type(rel: Any) -> str:
        rel_type = getattr(rel, "type", None)
        if rel_type:
            return str(rel_type)
        if isinstance(rel, dict):
            return str(rel.get("type") or "RELATED")
        return "RELATED"

    def _rel_props(rel: Any) -> Dict[str, Any]:
        if rel is None:
            return {}
        if isinstance(rel, dict):
            base = rel
        else:
            base = dict(rel)
        return _sanitize_properties(base)

    for row in rows:
        a = row.get("a")
        b = row.get("b")
        rel = row.get("rel")
        if a is None:
            continue

        a_id = _node_id(a)
        if not a_id:
            continue

        if a_id not in nodes:
            nodes[a_id] = {
                "id": a_id,
                "labels": _node_labels(a),
                "properties": _node_props(a),
            }

        if b is not None:
            b_id = _node_id(b)
            if not b_id:
                continue

            if b_id not in nodes:
                nodes[b_id] = {
                    "id": b_id,
                    "labels": _node_labels(b),
                    "properties": _node_props(b),
                }

            if rel is not None:
                source = getattr(getattr(rel, "start_node", None), "element_id", None) or a_id
                target = getattr(getattr(rel, "end_node", None), "element_id", None) or b_id
                edge_id = getattr(rel, "element_id", None) or f"{source}-{_rel_type(rel)}-{target}"
                edges.append(
                    {
                        "id": str(edge_id),
                        "source": str(source),
                        "target": str(target),
                        "type": _rel_type(rel),
                        "properties": _rel_props(rel),
                    }
                )

    return {
        "city": city,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": list(nodes.values()),
        "edges": edges,
    }


def get_node_details(node_id: str, neighbor_limit: int = 40) -> Dict[str, Any] | None:
    """Return one node and its neighboring edges/nodes for side-panel inspection."""
    driver = _get_driver()
    with driver.session() as session:
        row = session.run(
            """
            MATCH (n)
            WHERE elementId(n) = $node_id
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN n,
                   collect({
                     edgeId: CASE WHEN r IS NULL THEN NULL ELSE elementId(r) END,
                     relType: CASE WHEN r IS NULL THEN NULL ELSE type(r) END,
                     direction: CASE
                       WHEN r IS NULL THEN NULL
                       WHEN startNode(r) = n THEN 'out'
                       ELSE 'in'
                     END,
                     otherId: CASE WHEN m IS NULL THEN NULL ELSE elementId(m) END,
                     otherLabels: CASE WHEN m IS NULL THEN [] ELSE labels(m) END,
                     otherProps: CASE WHEN m IS NULL THEN {} ELSE properties(m) END
                   }) AS neighbors
            """,
            {"node_id": node_id},
        ).single()

    if not row:
        return None

    n = row["n"]
    all_neighbors = row["neighbors"] or []
    cleaned_neighbors = []
    for item in all_neighbors:
        if not item.get("edgeId"):
            continue
        cleaned_neighbors.append(
            {
                "edge_id": item.get("edgeId"),
                "relationship": item.get("relType"),
                "direction": item.get("direction"),
                "other_node": {
                    "id": item.get("otherId"),
                    "labels": item.get("otherLabels") or [],
                    "properties": _sanitize_properties(item.get("otherProps") or {}),
                },
            }
        )

    return {
        "id": n.element_id,
        "labels": list(n.labels),
        "properties": _sanitize_properties(dict(n)),
        "neighbors": cleaned_neighbors[:neighbor_limit],
        "total_neighbors": len(cleaned_neighbors),
    }


def clear_graph_data() -> Dict[str, int]:
    """Delete all nodes and relationships in the active Neo4j database."""
    driver = _get_driver()
    with driver.session() as session:
        counts = session.run(
            """
            MATCH (n)
            OPTIONAL MATCH (n)-[r]-()
            RETURN count(DISTINCT n) AS nodes, count(DISTINCT r) AS relationships
            """
        ).single()
        session.run("MATCH (n) DETACH DELETE n")

    return {
        "deleted_nodes": int((counts or {}).get("nodes") or 0),
        "deleted_relationships": int((counts or {}).get("relationships") or 0),
    }
