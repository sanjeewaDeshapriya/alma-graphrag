"""
Script: Ingest news from API providers (NewsAPI + GNews) with RSS fallback.

Usage:
    python scripts/run_ingest_news.py
    python scripts/run_ingest_news.py --rss-only   # skip API, use RSS feeds only
"""
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `src` imports work when running scripts directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import logging

from src.config import HOTELS_CITIES, NEWS_API_KEY, GNEWS_API_KEY
from src.ingest.news_api import fetch_all_news
from src.ingest.news_rss import fetch_news
from src.ingest.event_linker import link_events_to_hotels

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alma.scripts.news")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest news into the ALMA knowledge graph")
    parser.add_argument(
        "--rss-only",
        action="store_true",
        help="Skip API providers and use RSS feeds only",
    )
    args = parser.parse_args()

    # Fetch news
    news_items = []
    if not args.rss_only:
        news_items = fetch_all_news(
            newsapi_key=NEWS_API_KEY,
            gnews_key=GNEWS_API_KEY,
        )
        if news_items:
            print(f"\n=== News from APIs: {len(news_items)} articles ===")
        else:
            print("\nNo articles from APIs — falling back to RSS feeds.")

    if not news_items:
        news_items = fetch_news()
        print(f"\n=== News from RSS: {len(news_items)} items ===")

    # Show sample
    if news_items:
        print("\nSample news items:")
        for i, it in enumerate(news_items[:5], start=1):
            print(f"  {i}. [{it.get('source', '?')}] {it.get('title')}")
            print(f"     URL: {it.get('url')}")
    else:
        print("\nNo news items fetched. Check feed/API availability or network.")

    # Link to knowledge graph
    link_events_to_hotels(news_items, HOTELS_CITIES)
    print(f"\nIngested {len(news_items)} news items into the knowledge graph.")


if __name__ == "__main__":
    main()
