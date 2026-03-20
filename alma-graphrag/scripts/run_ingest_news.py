from src.ingest.news_rss import fetch_news
from src.ingest.event_linker import link_events_to_hotels
from src.config import HOTELS_CITIES


if __name__ == "__main__":
    news_items = fetch_news()
    link_events_to_hotels(news_items, HOTELS_CITIES)
    print(f"Ingested {len(news_items)} news items.")
