from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# Clean up empty string env vars from os.environ so libraries (like openai)
# don't interpret them as active blank values (e.g. empty OPENAI_BASE_URL).
for k, v in list(os.environ.items()):
    if v == "":
        os.environ.pop(k, None)

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

# Composite-score weight profile for the weighted retriever:
#   handset  — the original hand-tuned prior (default; preserves published results)
#   elicited — conditional-logit estimate from the discrete-choice study
#              (studies/weight-elicitation); facility/economic clip to zero
#   blended  — elicited proportions for spatial/accessibility/disruption, hand-set
#              prior retained for facility/economic
# See src/graph/retriever.py WEIGHT_PROFILES and
# studies/weight-elicitation/analysis/DATA_AUDIT.md
SCORING_WEIGHTS_PROFILE = os.getenv("SCORING_WEIGHTS_PROFILE", "handset").lower()

CRAG_MIN_SCORE = float(os.getenv("CRAG_MIN_SCORE", "0.6"))
CRAG_MAX_RETRIES = int(os.getenv("CRAG_MAX_RETRIES", "1"))

LLM_EXTRACT_ENABLED = os.getenv("LLM_EXTRACT_ENABLED", "false").lower() == "true"

# News API providers (free tier, optional — falls back to RSS if empty)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "")

# Traffic API integration
TRAFFIC_PROVIDER = os.getenv("TRAFFIC_PROVIDER", "google").lower()  # google | tomtom | both
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "")
TRAFFIC_ENABLED = os.getenv("TRAFFIC_ENABLED", "false").lower() == "true"
TRAFFIC_REFRESH_MINUTES = int(os.getenv("TRAFFIC_REFRESH_MINUTES", "30"))
TRAFFIC_SIGNAL_TTL_HOURS = int(os.getenv("TRAFFIC_SIGNAL_TTL_HOURS", "6"))
TRAFFIC_RADIUS_KM = float(os.getenv("TRAFFIC_RADIUS_KM", "5.0"))
TRAFFIC_MAX_HOTELS_PER_BATCH = int(os.getenv("TRAFFIC_MAX_HOTELS_PER_BATCH", "25"))

# Embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
HOTEL_VECTOR_INDEX = os.getenv("HOTEL_VECTOR_INDEX", "hotel_embeddings")

# --- pgvector keyword + semantic search -------------------------------------
# Local sentence-transformers model used for dense semantic search (offline,
# reproducible). all-MiniLM-L6-v2 emits 384-dim vectors.
ST_EMBEDDING_MODEL = os.getenv("ST_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
ST_EMBEDDING_DIM = int(os.getenv("ST_EMBEDDING_DIM", "384"))

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5433"))
PG_DB = os.getenv("PG_DB", "alma_vectors")
PG_USER = os.getenv("PG_USER", "alma")
PG_PASSWORD = os.getenv("PG_PASSWORD", "alma_password123")

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
