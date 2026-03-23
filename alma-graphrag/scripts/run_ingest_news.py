import sys
from pathlib import Path

# Ensure repo root is on sys.path so `src` imports work when running scripts directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingest.news_rss import fetch_news
from src.ingest.event_linker import link_events_to_hotels
from src.config import HOTELS_CITIES
import logging

logging.basicConfig(level=logging.INFO)



if __name__ == "__main__":
    news_items = fetch_news()
    # show a quick sample of fetched items for debugging
    if news_items:
        print("Sample news items:")
        for i, it in enumerate(news_items[:5], start=1):
            print(f"{i}. {it.get('title')} -> {it.get('url')}")
    else:
        print("No news items fetched from feeds. Check feed availability or network.")
    link_events_to_hotels(news_items, HOTELS_CITIES)
    print(f"Ingested {len(news_items)} news items.")
