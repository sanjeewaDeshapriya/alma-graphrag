from __future__ import annotations

from typing import Iterable, List

from src.config import HOTEL_MAX_RESULTS, LLM_EXTRACT_ENABLED
from src.graph.loader import GraphLoader
from src.ingest.google_places import GooglePlacesClient
from src.ingest.llm_extractor import LLMExtractor


def ingest_hotels(
    cities: Iterable[str],
    max_results: int = HOTEL_MAX_RESULTS,
    use_llm_extract: bool = LLM_EXTRACT_ENABLED,
) -> int:
    client = GooglePlacesClient()
    loader = GraphLoader()
    extractor = LLMExtractor() if use_llm_extract else None

    total = 0
    try:
        for city in cities:
            hotels = client.scrape_city(city, max_results=max_results)
            for hotel in hotels:
                loader.upsert_hotel(hotel)
                loader.upsert_amenities(hotel["id"], hotel.get("amenities", []))

                if extractor:
                    enriched = extractor.extract(hotel)
                    if enriched.get("amenities"):
                        loader.upsert_amenities(hotel["id"], enriched["amenities"])
                    if enriched.get("locations"):
                        loader.upsert_locations(hotel["id"], enriched["locations"])

            total += len(hotels)
    finally:
        loader.close()
        client.close()

    return total
