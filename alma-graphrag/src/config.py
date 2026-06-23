from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "alma_password123")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")

# --- LLM provider switch (openai | gemini) ----------------------------------
# Gemini is reached through its OpenAI-compatible endpoint, so the same
# `openai` SDK / `langchain_openai` clients work for both providers — only the
# api_key, model, and base_url change.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
GEMINI_OPENAI_BASE_URL = os.getenv(
    "GEMINI_OPENAI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
)

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
GOOGLE_PLACES_BASE_URL = os.getenv(
    "GOOGLE_PLACES_BASE_URL",
    "https://maps.googleapis.com/maps/api/place",
)

# LiteAPI (hotel rates / content)
LITEAPI_KEY = os.getenv("LITEAPI_KEY", "")
LITEAPI_BASE_URL = os.getenv("LITEAPI_BASE_URL", "https://api.liteapi.travel/v3.0")
LITEAPI_ENABLED = os.getenv("LITEAPI_ENABLED", "false").lower() == "true"
# Search currency — keep LKR so rates align with the rest of the graph.
LITEAPI_CURRENCY = os.getenv("LITEAPI_CURRENCY", "LKR")
LITEAPI_GUEST_NATIONALITY = os.getenv("LITEAPI_GUEST_NATIONALITY", "LK")
# Days from "today" to use as the check-in date when sampling live rates.
LITEAPI_CHECKIN_OFFSET_DAYS = int(os.getenv("LITEAPI_CHECKIN_OFFSET_DAYS", "14"))
# Length of stay (nights) for the rate sample used to derive nightly prices.
LITEAPI_LOS_NIGHTS = int(os.getenv("LITEAPI_LOS_NIGHTS", "1"))
LITEAPI_ADULTS = int(os.getenv("LITEAPI_ADULTS", "2"))
# How many cheapest rate plans to keep per hotel as RoomType nodes.
LITEAPI_MAX_RATES_PER_HOTEL = int(os.getenv("LITEAPI_MAX_RATES_PER_HOTEL", "5"))
LITEAPI_TIMEOUT = int(os.getenv("LITEAPI_TIMEOUT", "10"))

DEFAULT_CITY = os.getenv("DEFAULT_CITY", "Piliyandala")
DEFAULT_COUNTRY = os.getenv("DEFAULT_COUNTRY", "Sri Lanka")
DEFAULT_COUNTRY_CODE = os.getenv("DEFAULT_COUNTRY_CODE", "LK")
HOTEL_MAX_RESULTS = int(os.getenv("HOTEL_MAX_RESULTS", "40"))
NEWS_MAX_ITEMS = int(os.getenv("NEWS_MAX_ITEMS", "30"))

HOTELS_CITIES = [
    c.strip()
    for c in os.getenv("HOTELS_CITIES", DEFAULT_CITY).split(",")
    if c.strip()
]

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

CRAG_MIN_SCORE = float(os.getenv("CRAG_MIN_SCORE", "0.6"))
CRAG_MAX_RETRIES = int(os.getenv("CRAG_MAX_RETRIES", "1"))

LLM_EXTRACT_ENABLED = os.getenv("LLM_EXTRACT_ENABLED", "false").lower() == "true"

# News API providers (free tier, optional — falls back to RSS if empty)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "")

# Embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
HOTEL_VECTOR_INDEX = os.getenv("HOTEL_VECTOR_INDEX", "hotel_embeddings")

# --- Resolved active LLM settings -------------------------------------------
# All agents (CRAG, LLM extractor) and embedding pipelines read these instead
# of the raw OPENAI_*/GEMINI_* vars, so flipping LLM_PROVIDER switches the
# whole stack with no code changes.
if LLM_PROVIDER == "gemini":
    LLM_API_KEY = GEMINI_API_KEY or OPENAI_API_KEY
    LLM_MODEL = GEMINI_MODEL
    LLM_BASE_URL = OPENAI_BASE_URL or GEMINI_OPENAI_BASE_URL
    ACTIVE_EMBEDDING_MODEL = GEMINI_EMBEDDING_MODEL
    ACTIVE_EMBEDDING_API_KEY = GEMINI_API_KEY or OPENAI_API_KEY
    ACTIVE_EMBEDDING_BASE_URL = OPENAI_BASE_URL or GEMINI_OPENAI_BASE_URL
else:
    LLM_API_KEY = OPENAI_API_KEY
    LLM_MODEL = OPENAI_MODEL
    LLM_BASE_URL = OPENAI_BASE_URL or None
    ACTIVE_EMBEDDING_MODEL = EMBEDDING_MODEL
    ACTIVE_EMBEDDING_API_KEY = OPENAI_API_KEY
    ACTIVE_EMBEDDING_BASE_URL = OPENAI_BASE_URL or None
