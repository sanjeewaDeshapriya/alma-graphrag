"""
RSS-based news ingestion with reliable Google News RSS feeds as primary source.

The original Sri Lankan news feeds (newsfirst, hirunews, lankadeepa, sltda) are
frequently offline or have broken RSS. Google News RSS is always available and
returns current results for any search query.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Dict, List
import feedparser
import logging

from src.config import NEWS_MAX_ITEMS

logger = logging.getLogger("alma.news")

# --- Reliable RSS feeds ---
# Google News RSS feeds (always available, search-based)
GOOGLE_NEWS_FEEDS = [
    "https://news.google.com/rss/search?q=Sri+Lanka+tourism+hotels&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Sri+Lanka+travel&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Colombo+hotels&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Sri+Lanka+weather+travel+advisory&hl=en-US&gl=US&ceid=US:en",
]

# Sri Lankan news outlets (may be unreliable — kept as secondary fallback)
LOCAL_FEEDS = [
    "https://www.newsfirst.lk/feed/",
    "https://www.hirunews.lk/rss.xml",
    "https://www.dailymirror.lk/RSS_Feeds/travel/295",
]

# International travel news (usually reliable)
INTL_FEEDS = [
    "https://www.traveldailynews.com/feed/",
    "https://skift.com/feed/",
]

# Priority order: Google News first (always works), then international, then local
NEWS_FEEDS = GOOGLE_NEWS_FEEDS + INTL_FEEDS + LOCAL_FEEDS


def _parse_single_feed(url: str) -> List[Dict]:
    """Parse a single RSS feed URL. Returns list of normalized news items."""
    items: List[Dict] = []
    try:
        feed = feedparser.parse(url)
    except Exception as e:
        logger.warning("Failed to parse feed %s: %s", url, e)
        return items

    entry_count = len(getattr(feed, "entries", []))
    is_bozo = getattr(feed, "bozo", False)
    logger.info("Feed %s parsed: entries=%d, bozo=%s", url, entry_count, is_bozo)

    if is_bozo and entry_count == 0:
        logger.debug(
            "Feed %s bozo_exception: %s",
            url,
            getattr(feed, "bozo_exception", None),
        )
        return items

    for entry in getattr(feed, "entries", []):
        title = entry.get("title", "")
        link = entry.get("link", "")
        summary = entry.get("summary", entry.get("description", ""))
        published = entry.get("published", datetime.utcnow().isoformat())
        if not link:
            continue
        news_id = hashlib.md5(link.encode("utf-8")).hexdigest()
        items.append(
            {
                "id": news_id,
                "title": title,
                "summary": summary[:500] if summary else "",
                "published_at": published,
                "url": link,
                "source": f"rss:{url.split('/')[2]}",
            }
        )
    return items


def fetch_news(max_items: int = NEWS_MAX_ITEMS) -> List[Dict]:
    """
    Fetch news from all configured RSS feeds.
    Google News feeds are tried first (most reliable), then others.
    Deduplicates by URL.
    """
    all_items: List[Dict] = []
    seen_urls: set = set()

    logger.info(
        "Fetching news from %d feeds, max_items=%d", len(NEWS_FEEDS), max_items
    )

    for url in NEWS_FEEDS:
        if len(all_items) >= max_items:
            break
        for item in _parse_single_feed(url):
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                all_items.append(item)
                if len(all_items) >= max_items:
                    break

    logger.info(
        "Total RSS news items fetched: %d (returning up to %d)",
        len(all_items),
        max_items,
    )
    return all_items[:max_items]
