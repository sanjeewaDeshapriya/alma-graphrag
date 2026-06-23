from __future__ import annotations

import logging
from typing import Iterable
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from src.config import (
    HOTEL_MAX_RESULTS,
    LITEAPI_ENABLED,
    NEWS_API_KEY,
    GNEWS_API_KEY,
)
from src.ingest.hotel_ingest import ingest_hotels
from src.ingest.news_api import fetch_all_news
from src.ingest.news_rss import fetch_news
from src.ingest.event_linker import link_events_to_hotels

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

    scheduler.start()
    return scheduler


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
