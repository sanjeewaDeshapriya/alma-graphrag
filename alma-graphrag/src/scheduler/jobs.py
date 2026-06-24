from __future__ import annotations

import logging
from typing import Iterable
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from src.config import (
    HOTEL_MAX_RESULTS,
    LITEAPI_ENABLED,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    NEWS_API_KEY,
    GNEWS_API_KEY,
    TRAFFIC_ENABLED,
    TRAFFIC_PROVIDER,
    TRAFFIC_REFRESH_MINUTES,
    TRAFFIC_SIGNAL_TTL_HOURS,
)
from src.ingest.hotel_ingest import ingest_hotels
from src.ingest.news_api import fetch_all_news
from src.ingest.news_rss import fetch_news
from src.ingest.event_linker import link_events_to_hotels
from src.ingest.traffic import fetch_all_traffic
from src.ingest.traffic_linker import link_traffic_to_hotels, cleanup_stale_signals

logger = logging.getLogger("alma.scheduler")


def start_scheduler(cities: Iterable[str]) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler()

    hotel_source = "both" if LITEAPI_ENABLED else "google"
    scheduler.add_job(
        ingest_hotels,
        "interval",
        hours=24,
        args=[list(cities), HOTEL_MAX_RESULTS, False, hotel_source],
        id="ingest_hotels",
    )

    scheduler.add_job(
        _ingest_news,
        "interval",
        minutes=15,
        args=[list(cities)],
        id="ingest_news",
    )

    if TRAFFIC_ENABLED:
        scheduler.add_job(
            _ingest_traffic,
            "interval",
            minutes=TRAFFIC_REFRESH_MINUTES,
            args=[list(cities)],
            id="ingest_traffic",
        )
        scheduler.add_job(
            _cleanup_traffic,
            "interval",
            hours=1,
            id="cleanup_traffic",
        )

    scheduler.start()
    return scheduler


def _ingest_traffic(cities: Iterable[str]) -> None:
    """Fetch traffic data and link to hotels."""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    city_data: list[dict] = []
    hotel_data: list[dict] = []

    with driver.session() as session:
        for city_name in cities:
            rows = session.run(
                """
                MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name: $city})
                RETURN h.id AS id, h.name AS name, h.lat AS lat, h.lng AS lng, c.name AS city_name
                """,
                {"city": city_name},
            ).data()

            lats = [r["lat"] for r in rows if r.get("lat")]
            lngs = [r["lng"] for r in rows if r.get("lng")]
            if lats and lngs:
                city_data.append({"name": city_name, "lat": sum(lats) / len(lats), "lng": sum(lngs) / len(lngs)})
            hotel_data.extend(rows)
    driver.close()

    if not hotel_data:
        logger.info("Scheduled traffic: no hotels found")
        return

    traffic_data = fetch_all_traffic(city_data, hotel_data, provider=TRAFFIC_PROVIDER)
    counts = link_traffic_to_hotels(traffic_data, hotel_data)
    logger.info("Scheduled traffic ingestion: %s", counts)


def _cleanup_traffic() -> None:
    deleted = cleanup_stale_signals()
    if deleted:
        logger.info("Scheduled traffic cleanup: %d removed", deleted)


def _ingest_news(cities: Iterable[str]) -> None:
    """Fetch news from APIs first, fall back to RSS, then link to hotels."""
    news_items = fetch_all_news(
        newsapi_key=NEWS_API_KEY,
        gnews_key=GNEWS_API_KEY,
    )
    if not news_items:
        logger.info("No API news — falling back to RSS")
        news_items = fetch_news()
    link_events_to_hotels(news_items, list(cities))
    logger.info("Scheduled news ingestion: %d items", len(news_items))
