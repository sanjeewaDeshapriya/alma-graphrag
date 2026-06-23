from __future__ import annotations

import logging
from typing import Iterable, Optional

from src.config import HOTEL_MAX_RESULTS, LLM_EXTRACT_ENABLED
from src.graph.loader import GraphLoader
from src.ingest.google_places import GooglePlacesClient
from src.ingest.liteapi import LiteApiClient
from src.ingest.llm_extractor import LLMExtractor
from src.ingest.ner_extractor import NERExtractor

logger = logging.getLogger("alma.hotel_ingest")


def ingest_hotels(
    cities: Iterable[str],
    max_results: int = HOTEL_MAX_RESULTS,
    use_llm_extract: bool = LLM_EXTRACT_ENABLED,
    source: str = "both",
) -> int:
    """
    Ingest hotels into the knowledge graph from one or more sources.

    source:
      - "google"  : Google Places only (preserves existing behaviour)
      - "liteapi" : Lite API rates/content only
      - "both"    : run both sources (default)
    """
    cities = list(cities)
    loader = GraphLoader()
    extractor = LLMExtractor() if use_llm_extract else None
    ner = NERExtractor()

    total = 0
    try:
        if source in ("google", "both"):
            try:
                total += _ingest_google(cities, max_results, loader, extractor, ner)
            except Exception:
                logger.exception("Google Places ingest failed; continuing")
        if source in ("liteapi", "both"):
            try:
                total += _ingest_liteapi(cities, max_results, loader, extractor, ner)
            except Exception:
                logger.exception("Lite API ingest failed; continuing")
    finally:
        loader.close()

    return total


def _ingest_google(
    cities: Iterable[str],
    max_results: int,
    loader: GraphLoader,
    extractor: Optional[LLMExtractor],
    ner: NERExtractor,
) -> int:
    logger.info("=== Google Places ingest started for cities: %s ===", list(cities) if not isinstance(cities, list) else cities)
    client = GooglePlacesClient()
    total = 0
    try:
        for city in cities:
            logger.info("Google Places → fetching hotels for city=%s max=%d", city, max_results)
            hotels = client.scrape_city(city, max_results=max_results)
            logger.info("Google Places ← received %d hotels for city=%s", len(hotels), city)
            for i, hotel in enumerate(hotels, 1):
                logger.info(
                    "Google Places: writing hotel %d/%d id=%s name=%s",
                    i, len(hotels), hotel["id"], hotel.get("name", "?"),
                )
                loader.upsert_hotel(hotel)
                loader.upsert_amenities(hotel["id"], hotel.get("amenities", []))
                _ner_enrich(ner, loader, hotel)
                _llm_enrich(extractor, loader, hotel)
            total += len(hotels)
            logger.info("Google Places: ingested %d hotels for %s", len(hotels), city)
    finally:
        client.close()
    logger.info("=== Google Places ingest done: %d total hotels ===", total)
    return total


def _ingest_liteapi(
    cities: Iterable[str],
    max_results: int,
    loader: GraphLoader,
    extractor: Optional[LLMExtractor],
    ner: NERExtractor,
) -> int:
    logger.info("=== Lite API ingest started for cities: %s ===", list(cities) if not isinstance(cities, list) else cities)
    client = LiteApiClient()
    total = 0
    try:
        for city in cities:
            hotels = client.scrape_city(city, max_results=max_results)
            logger.info("Lite API: writing %d hotels to graph for city=%s", len(hotels), city)
            for i, hotel in enumerate(hotels, 1):
                logger.info(
                    "Lite API: writing hotel %d/%d id=%s name=%s price=%s %s",
                    i, len(hotels), hotel["id"], hotel.get("name", "?"),
                    hotel.get("price_per_night", 0), hotel.get("price_currency", ""),
                )
                loader.upsert_hotel(hotel)
                loader.upsert_hotel_extras(hotel["id"], hotel)
                loader.upsert_amenities(hotel["id"], hotel.get("amenities", []))
                loader.upsert_room_types(hotel["id"], hotel.get("room_types", []))
                loader.upsert_board_types(hotel["id"], hotel.get("board_types", []))
                _ner_enrich(ner, loader, hotel)
                _llm_enrich(extractor, loader, hotel)
            total += len(hotels)
            logger.info("Lite API: ingested %d hotels for %s", len(hotels), city)
    finally:
        client.close()
    logger.info("=== Lite API ingest done: %d total hotels ===", total)
    return total


def _ner_enrich(ner: NERExtractor, loader: GraphLoader, hotel: dict) -> None:
    """NER-based entity extraction — runs on every hotel, no API cost."""
    try:
        result = ner.extract(hotel)
    except Exception:
        logger.exception("NER extraction failed for hotel %s", hotel.get("id"))
        return

    hid = hotel["id"]
    nb = result.get("neighborhoods", [])
    lm = result.get("landmarks", [])
    at = result.get("attraction_types", [])
    logger.info(
        "NER hotel=%s → %d neighborhood(s), %d landmark(s), %d attraction type(s): %s",
        hid, len(nb), len(lm), len(at), at or "-",
    )
    if nb:
        loader.upsert_neighborhoods(hid, nb)
    if lm:
        loader.upsert_ner_landmarks(hid, lm)
    if at:
        loader.upsert_attraction_types(hid, at)


def _llm_enrich(
    extractor: Optional[LLMExtractor],
    loader: GraphLoader,
    hotel: dict,
) -> None:
    """Optional LLM attribute extraction shared across sources."""
    if not extractor:
        return
    enriched = extractor.extract(hotel)
    if enriched.get("amenities"):
        loader.upsert_amenities(hotel["id"], enriched["amenities"])
    if enriched.get("locations"):
        loader.upsert_locations(hotel["id"], enriched["locations"])
