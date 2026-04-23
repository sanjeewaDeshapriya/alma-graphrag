"""
ALMA-GraphRAG API — FastAPI application.

Endpoints:
  POST /query          — ask a question against the knowledge graph
  POST /ingest/trigger — manually trigger hotel + news ingestion
  POST /ingest/news    — trigger news-only ingestion (API + RSS)
  GET  /health         — health / readiness check
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import (
    DEFAULT_CITY,
    HOTEL_MAX_RESULTS,
    HOTELS_CITIES,
    NEWS_API_KEY,
    GNEWS_API_KEY,
)
from src.crag.graph import run_crag
from src.ingest.hotel_ingest import ingest_hotels
from src.ingest.news_rss import fetch_news
from src.ingest.news_api import fetch_all_news
from src.ingest.event_linker import link_events_to_hotels

logger = logging.getLogger("alma.api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="ALMA GraphRAG Phase 1",
    description="Graph-based hotel recommendation engine with CRAG orchestration.",
    version="1.1.0",
)

# --- CORS middleware ----------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request / Response models ------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, description="Natural-language question")
    city: Optional[str] = Field(None, max_length=100, description="Target city (defaults to DEFAULT_CITY)")


class QueryResponse(BaseModel):
    answer: str
    context: str


# --- Endpoints ----------------------------------------------------------------

@app.get("/health")
def health_check() -> dict:
    """Readiness probe — returns basic service status."""
    return {"status": "ok", "service": "alma-graphrag", "version": "1.1.0"}


@app.post("/query", response_model=QueryResponse)
def query_graph(req: QueryRequest) -> QueryResponse:
    """Ask a question against the hotel knowledge graph using CRAG."""
    city = req.city or DEFAULT_CITY
    logger.info("API /query: question=%s, city=%s", req.question, city)
    try:
        result = run_crag(question=req.question, city=city)
    except Exception as exc:
        logger.exception("CRAG pipeline failed")
        raise HTTPException(status_code=500, detail=f"Query processing error: {exc}")
    logger.info(
        "API /query result: answer_len=%d, context_len=%d",
        len(result.get("answer", "")),
        len(result.get("context", "")),
    )
    return QueryResponse(answer=result["answer"], context=result["context"])


@app.post("/ingest/trigger")
def manual_ingest(city: Optional[str] = None) -> dict:
    """Trigger full ingestion (hotels + news) for one or all configured cities."""
    cities = [city] if city else HOTELS_CITIES
    logger.info("API /ingest/trigger: cities=%s", cities)
    try:
        hotel_count = ingest_hotels(cities, max_results=HOTEL_MAX_RESULTS)
    except Exception as exc:
        logger.exception("Hotel ingestion failed")
        raise HTTPException(status_code=500, detail=f"Hotel ingestion error: {exc}")

    # Fetch news from APIs first, fall back to RSS
    news_items = _fetch_news_combined()
    link_events_to_hotels(news_items, list(cities))

    return {
        "status": "ok",
        "hotels_ingested": hotel_count,
        "news_ingested": len(news_items),
    }


@app.post("/ingest/news")
def ingest_news_only() -> dict:
    """Trigger news-only ingestion from all configured providers."""
    news_items = _fetch_news_combined()
    link_events_to_hotels(news_items, list(HOTELS_CITIES))
    return {"status": "ok", "news_ingested": len(news_items)}


# --- Helpers ------------------------------------------------------------------

def _fetch_news_combined() -> list:
    """
    Fetch news from API providers (NewsAPI + GNews) with RSS fallback.
    If both API keys are missing, falls back to RSS-only.
    """
    news_items = fetch_all_news(
        newsapi_key=NEWS_API_KEY,
        gnews_key=GNEWS_API_KEY,
    )
    if not news_items:
        logger.info("No API news — falling back to RSS feeds")
        news_items = fetch_news()
    return news_items
