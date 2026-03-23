from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Dict, List
import feedparser

from src.config import NEWS_MAX_ITEMS
import logging

logger = logging.getLogger("alma.news")

NEWS_FEEDS = [
    "https://www.newsfirst.lk/feed/",
    "https://www.hirunews.lk/rss.xml",
    "https://www.lankadeepa.lk/rss.xml",
    "https://sltda.gov.lk/rss",
]


def fetch_news() -> List[Dict]:
    items: List[Dict] = []
    logger.info("Fetching news from %d feeds, NEWS_MAX_ITEMS=%s", len(NEWS_FEEDS), NEWS_MAX_ITEMS)
    for url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            logger.warning("Failed to parse feed %s: %s", url, e)
            continue

        entry_count = len(getattr(feed, 'entries', []))
        logger.info("Feed %s parsed: entries=%d, bozo=%s", url, entry_count, getattr(feed, 'bozo', False))
        if getattr(feed, "bozo", False):
            logger.debug("Feed %s bozo_exception: %s", url, getattr(feed, "bozo_exception", None))

        for entry in getattr(feed, 'entries', []):
            title = entry.get("title", "")
            link = entry.get("link", "")
            summary = entry.get("summary", "")
            published = entry.get("published", datetime.utcnow().isoformat())
            if not link:
                logger.debug("Skipping entry without link in feed %s: %s", url, title)
                continue
            news_id = hashlib.md5(link.encode("utf-8")).hexdigest()
            items.append(
                {
                    "id": news_id,
                    "title": title,
                    "summary": summary,
                    "published_at": published,
                    "url": link,
                    "source": url,
                }
            )

    logger.info("Total news items fetched: %d (returning up to %s)", len(items), NEWS_MAX_ITEMS)
    return items[:NEWS_MAX_ITEMS]
