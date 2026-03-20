from __future__ import annotations

from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from src.config import DEFAULT_CITY, HOTEL_MAX_RESULTS, HOTELS_CITIES
from crag.graph import run_crag
from src.ingest.hotel_ingest import ingest_hotels
from src.ingest.news_rss import fetch_news
from src.ingest.event_linker import link_events_to_hotels

app = FastAPI(title="ALMA GraphRAG Phase 1")


class QueryRequest(BaseModel):
    question: str
    city: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    context: str


@app.post("/query", response_model=QueryResponse)
def query_graph(req: QueryRequest) -> QueryResponse:
    city = req.city or DEFAULT_CITY
    result = run_crag(question=f"{req.question} near {city}")
    return QueryResponse(answer=result["answer"], context=result["context"])


@app.post("/ingest/trigger")
def manual_ingest(city: Optional[str] = None) -> dict:
    cities = [city] if city else HOTELS_CITIES
    hotel_count = ingest_hotels(cities, max_results=HOTEL_MAX_RESULTS)
    news_items = fetch_news()
    link_events_to_hotels(news_items, list(cities))
    return {"status": "ok", "hotels_ingested": hotel_count}
