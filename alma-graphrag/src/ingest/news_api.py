"""
News API integration — dual-provider (NewsAPI.org + GNews) with RSS fallback.

Provider priority:
  1. NewsAPI.org  (100 req/day free, broad coverage)
  2. GNews.io     (100 req/day free, Google News rankings)
  3. RSS feeds    (unlimited, no key required — original fallback)

Set NEWS_API_KEY and/or GNEWS_API_KEY in .env to enable the respective providers.
If neither key is set the system falls back to RSS-only mode automatically.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import httpx

from src.config import NEWS_MAX_ITEMS

logger = logging.getLogger("alma.news_api")

# ---------------------------------------------------------------------------
# Provider: NewsAPI.org
# ---------------------------------------------------------------------------

NEWSAPI_BASE = "https://newsapi.org/v2/everything"

# Search terms designed to pull tourism/travel/hotel news relevant to the KG
NEWSAPI_QUERIES = [
    "Sri Lanka tourism",
    "Sri Lanka hotels",
    "Sri Lanka travel",
    "Colombo hotels tourism",
    "Sri Lanka weather travel",
]


def fetch_newsapi(
    api_key: str,
    queries: Optional[List[str]] = None,
    max_items: int = NEWS_MAX_ITEMS,
) -> List[Dict]:
    """Fetch articles from NewsAPI.org /v2/everything endpoint."""
    if not api_key:
        logger.debug("NewsAPI key not set — skipping")
        return []

    queries = queries or NEWSAPI_QUERIES
    items: List[Dict] = []
    seen_urls: set = set()

    from_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

    client = httpx.Client(timeout=20)
    try:
        for q in queries:
            if len(items) >= max_items:
                break
            try:
                resp = client.get(
                    NEWSAPI_BASE,
                    params={
                        "q": q,
                        "from": from_date,
                        "sortBy": "publishedAt",
                        "language": "en",
                        "pageSize": min(20, max_items - len(items)),
                        "apiKey": api_key,
                    },
                )
                data = resp.json()
                if data.get("status") != "ok":
                    logger.warning(
                        "NewsAPI error for query '%s': %s",
                        q,
                        data.get("message", data.get("code")),
                    )
                    continue

                for article in data.get("articles", []):
                    url = article.get("url", "")
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    items.append(
                        {
                            "id": hashlib.md5(url.encode("utf-8")).hexdigest(),
                            "title": article.get("title", ""),
                            "summary": (article.get("description") or "")[:500],
                            "published_at": article.get("publishedAt", datetime.utcnow().isoformat()),
                            "url": url,
                            "source": f"newsapi:{article.get('source', {}).get('name', 'unknown')}",
                        }
                    )
            except Exception as exc:
                logger.warning("NewsAPI request failed for query '%s': %s", q, exc)
    finally:
        client.close()

    logger.info("NewsAPI fetched %d articles", len(items))
    return items[:max_items]


# ---------------------------------------------------------------------------
# Provider: GNews.io
# ---------------------------------------------------------------------------

GNEWS_BASE = "https://gnews.io/api/v4"

GNEWS_QUERIES = [
    "Sri Lanka tourism hotels",
    "Sri Lanka travel",
    "Colombo hotel",
]


def fetch_gnews(
    api_key: str,
    queries: Optional[List[str]] = None,
    max_items: int = NEWS_MAX_ITEMS,
) -> List[Dict]:
    """Fetch articles from GNews.io search + top-headlines endpoints."""
    if not api_key:
        logger.debug("GNews key not set — skipping")
        return []

    queries = queries or GNEWS_QUERIES
    items: List[Dict] = []
    seen_urls: set = set()

    client = httpx.Client(timeout=20)
    try:
        # --- top headlines for Sri Lanka travel/general ---
        try:
            resp = client.get(
                f"{GNEWS_BASE}/top-headlines",
                params={
                    "category": "general",
                    "lang": "en",
                    "country": "lk",
                    "max": min(10, max_items),
                    "apikey": api_key,
                },
            )
            data = resp.json()
            for article in data.get("articles", []):
                url = article.get("url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                items.append(
                    {
                        "id": hashlib.md5(url.encode("utf-8")).hexdigest(),
                        "title": article.get("title", ""),
                        "summary": (article.get("description") or "")[:500],
                        "published_at": article.get("publishedAt", datetime.utcnow().isoformat()),
                        "url": url,
                        "source": f"gnews:{article.get('source', {}).get('name', 'unknown')}",
                    }
                )
        except Exception as exc:
            logger.warning("GNews top-headlines failed: %s", exc)

        # --- keyword search ---
        for q in queries:
            if len(items) >= max_items:
                break
            try:
                resp = client.get(
                    f"{GNEWS_BASE}/search",
                    params={
                        "q": q,
                        "lang": "en",
                        "max": min(10, max_items - len(items)),
                        "apikey": api_key,
                    },
                )
                data = resp.json()
                for article in data.get("articles", []):
                    url = article.get("url", "")
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    items.append(
                        {
                            "id": hashlib.md5(url.encode("utf-8")).hexdigest(),
                            "title": article.get("title", ""),
                            "summary": (article.get("description") or "")[:500],
                            "published_at": article.get("publishedAt", datetime.utcnow().isoformat()),
                            "url": url,
                            "source": f"gnews:{article.get('source', {}).get('name', 'unknown')}",
                        }
                    )
            except Exception as exc:
                logger.warning("GNews search failed for '%s': %s", q, exc)
    finally:
        client.close()

    logger.info("GNews fetched %d articles", len(items))
    return items[:max_items]


# ---------------------------------------------------------------------------
# Unified fetcher — tries all providers, deduplicates by URL
# ---------------------------------------------------------------------------


def fetch_all_news(
    newsapi_key: str = "",
    gnews_key: str = "",
    max_items: int = NEWS_MAX_ITEMS,
) -> List[Dict]:
    """
    Aggregate news from all configured providers.

    Falls back to RSS (see news_rss.py) if both API keys are empty.
    The caller (ingest pipeline / scheduler) should handle the RSS fallback.
    """
    all_items: List[Dict] = []
    seen_urls: set = set()

    # 1) NewsAPI.org
    for item in fetch_newsapi(newsapi_key, max_items=max_items):
        if item["url"] not in seen_urls:
            seen_urls.add(item["url"])
            all_items.append(item)

    # 2) GNews.io
    remaining = max_items - len(all_items)
    if remaining > 0:
        for item in fetch_gnews(gnews_key, max_items=remaining):
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                all_items.append(item)

    logger.info(
        "Total news from APIs: %d (newsapi=%s, gnews=%s)",
        len(all_items),
        "configured" if newsapi_key else "skipped",
        "configured" if gnews_key else "skipped",
    )
    return all_items[:max_items]
