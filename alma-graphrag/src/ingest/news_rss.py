from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Dict, List
import feedparser

from src.config import NEWS_MAX_ITEMS

NEWS_FEEDS = [
    "https://www.newsfirst.lk/feed/",
    "https://www.hirunews.lk/rss.xml",
    "https://www.lankadeepa.lk/rss.xml",
    "https://sltda.gov.lk/rss",
]


def fetch_news() -> List[Dict]:
    items: List[Dict] = []
    for url in NEWS_FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = entry.get("title", "")
            link = entry.get("link", "")
            summary = entry.get("summary", "")
            published = entry.get("published", datetime.utcnow().isoformat())
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
    return items[:NEWS_MAX_ITEMS]
