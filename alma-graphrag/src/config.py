from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "alma_password123")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
GOOGLE_PLACES_BASE_URL = os.getenv(
    "GOOGLE_PLACES_BASE_URL",
    "https://maps.googleapis.com/maps/api/place",
)

DEFAULT_CITY = os.getenv("DEFAULT_CITY", "Piliyandala")
DEFAULT_COUNTRY = os.getenv("DEFAULT_COUNTRY", "Sri Lanka")
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
