from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Dict, Iterable, List

from src.graph.loader import GraphLoader


def _event_from_news(news: Dict) -> Dict:
    title = news.get("title", "")
    event_id = hashlib.md5(title.encode("utf-8")).hexdigest()
    return {
        "id": event_id,
        "title": title[:200],
        "type": "news",
        "start_time": news.get("published_at", datetime.utcnow().isoformat()),
        "end_time": news.get("published_at", datetime.utcnow().isoformat()),
        "severity": "medium",
        "source": news.get("source", "rss"),
    }


def link_events_to_hotels(news_items: Iterable[Dict], city_keywords: List[str]) -> None:
    loader = GraphLoader()
    try:
        for news in news_items:
            loader.upsert_news_signal(news)
            event = _event_from_news(news)
            loader.upsert_event(event)
            loader.link_event_news(event["id"], news["url"])

            summary = (news.get("summary") or "").lower()
            title = (news.get("title") or "").lower()
            combined = f"{title} {summary}"
            if any(k.lower() in combined for k in city_keywords):
                # Link all hotels in those cities to the event.
                _link_city_hotels_to_event(loader, city_keywords, event["id"])
    finally:
        loader.close()


def _link_city_hotels_to_event(loader: GraphLoader, cities: List[str], event_id: str) -> None:
    with loader.driver.session() as session:
        for city in cities:
            records = session.run(
                """
                MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name: $city})
                RETURN h.id AS id
                """,
                {"city": city},
            ).data()
            for row in records:
                loader.link_hotel_event(row["id"], event_id)
