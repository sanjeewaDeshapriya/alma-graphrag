from __future__ import annotations

from typing import Iterable
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from src.config import HOTEL_MAX_RESULTS
from src.ingest.hotel_ingest import ingest_hotels
from src.ingest.news_rss import fetch_news
from src.ingest.event_linker import link_events_to_hotels


def start_scheduler(cities: Iterable[str]) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler()

    scheduler.add_job(
        ingest_hotels,
        "interval",
        hours=24,
        args=[list(cities), HOTEL_MAX_RESULTS, False],
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
    news_items = fetch_news()
    link_events_to_hotels(news_items, list(cities))
