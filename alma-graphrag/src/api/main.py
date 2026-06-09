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
from pathlib import Path
import re
import subprocess
import sys
from threading import Lock
from time import time
from typing import Dict, Optional
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import (
    DEFAULT_CITY,
    HOTEL_MAX_RESULTS,
    HOTELS_CITIES,
    NEWS_API_KEY,
    GNEWS_API_KEY,
)
from src.crag.graph import run_crag
from src.graph.query import get_graph_network, get_graph_overview, get_node_details
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


class GraphFilterRequest(BaseModel):
    city: Optional[str] = Field(None, max_length=100)
    limit: int = Field(180, ge=20, le=600)


class IngestStartRequest(BaseModel):
    city: Optional[str] = Field(None, max_length=100)


_ingest_jobs: Dict[str, dict] = {}
_ingest_jobs_lock = Lock()
_repo_root = Path(__file__).resolve().parents[2]


def _append_job_log(job_id: str, message: str) -> None:
    with _ingest_jobs_lock:
        job = _ingest_jobs.get(job_id)
        if not job:
            return
        logs = job.setdefault("logs", [])
        logs.append({"ts": int(time()), "message": message})
        if len(logs) > 120:
            del logs[:-120]


def _eta_seconds(started_at: float, progress: int) -> Optional[int]:
    if progress <= 0:
        return None
    elapsed = max(0.0, time() - started_at)
    remaining = elapsed * ((100 - progress) / progress)
    return max(0, int(round(remaining)))


def _extract_count(lines: list[str], pattern: str) -> int:
    rx = re.compile(pattern, re.IGNORECASE)
    for line in lines:
        match = rx.search(line)
        if match:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                return 0
    return 0


def _run_script_with_logs(job_id: str, args: list[str]) -> list[str]:
    """Run a python script command and stream all output to app logs + job logs."""
    command = [sys.executable, *args]
    command_text = " ".join(command)
    logger.info("Ingest job %s running command: %s", job_id, command_text)
    _append_job_log(job_id, f"run: {command_text}")

    lines: list[str] = []
    process = subprocess.Popen(
        command,
        cwd=str(_repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if process.stdout is not None:
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if not line:
                continue
            lines.append(line)
            logger.info("Ingest job %s | %s", job_id, line)
            _append_job_log(job_id, line)

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed ({return_code}): {command_text}")

    return lines


def _set_job_state(job_id: str, **updates) -> None:
    log_line = None
    with _ingest_jobs_lock:
        if job_id in _ingest_jobs:
            prev_status = _ingest_jobs[job_id].get("status")
            prev_step = _ingest_jobs[job_id].get("step")
            prev_progress = _ingest_jobs[job_id].get("progress")
            _ingest_jobs[job_id].update(updates)
            now_ts = time()
            _ingest_jobs[job_id]["updated_at"] = now_ts
            started_at = _ingest_jobs[job_id].get("started_at")
            if started_at:
                _ingest_jobs[job_id]["elapsed_seconds"] = max(
                    0, int(round(now_ts - float(started_at)))
                )
                progress = int(_ingest_jobs[job_id].get("progress") or 0)
                status = _ingest_jobs[job_id].get("status")
                _ingest_jobs[job_id]["eta_seconds"] = (
                    0 if status in {"completed", "failed"} else _eta_seconds(float(started_at), progress)
                )
            status = _ingest_jobs[job_id].get("status")
            step = _ingest_jobs[job_id].get("step")
            progress = _ingest_jobs[job_id].get("progress")
            if status != prev_status or step != prev_step or progress != prev_progress:
                log_line = f"status={status} step={step} progress={progress}%"

    if log_line:
        _append_job_log(job_id, log_line)
        logger.info("Ingest job %s %s", job_id, log_line)


def _run_ingest_job(job_id: str, city: str) -> None:
    """Background ingestion pipeline that runs ingest scripts with live console logs."""
    _set_job_state(
        job_id,
        status="running",
        progress=10,
        step="Starting ingestion",
        step_index=1,
        step_total=3,
        step_progress=100,
    )
    try:
        _set_job_state(
            job_id,
            progress=25,
            step="Running hotel ingest script",
            step_index=2,
            step_total=3,
            step_progress=15,
        )
        hotel_lines = _run_script_with_logs(
            job_id,
            ["scripts/run_ingest_hotels.py", "--city", city],
        )
        hotels_ingested = _extract_count(hotel_lines, r"total\s+hotels\s+ingested:\s*(\d+)")

        _set_job_state(
            job_id,
            progress=65,
            step="Running news ingest script",
            step_index=3,
            step_total=3,
            step_progress=50,
        )
        news_lines = _run_script_with_logs(
            job_id,
            ["scripts/run_ingest_news.py"],
        )
        news_ingested = _extract_count(news_lines, r"ingested\s+(\d+)\s+news\s+items")

        _set_job_state(
            job_id,
            status="completed",
            progress=100,
            step="Ingestion completed",
            step_index=3,
            step_total=3,
            step_progress=100,
            finished_at=time(),
            result={
                "city": city,
                "hotels_ingested": hotels_ingested,
                "news_ingested": news_ingested,
            },
        )
    except Exception as exc:  # noqa: BLE001 - capture any failure and expose via job status
        logger.exception("Background ingestion job failed")
        _set_job_state(
            job_id,
            status="failed",
            progress=100,
            step="Ingestion failed",
            step_index=3,
            step_total=3,
            step_progress=100,
            finished_at=time(),
            error=str(exc),
        )


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
        raise HTTPException(status_code=500, detail=f"Query processing error: {exc}") from exc
    logger.info(
        "API /query result: answer_len=%d, context_len=%d",
        len(result.get("answer", "")),
        len(result.get("context", "")),
    )
    return QueryResponse(answer=result["answer"], context=result["context"])


@app.get("/graph/overview")
def graph_overview(city: Optional[str] = None) -> dict:
    """Return graph summary for dashboard cards/charts."""
    try:
        return get_graph_overview(city=city)
    except Exception as exc:
        logger.exception("Graph overview query failed")
        raise HTTPException(status_code=500, detail=f"Graph overview error: {exc}") from exc


@app.post("/graph/network")
def graph_network(req: GraphFilterRequest) -> dict:
    """Return a graph sub-network for visualization."""
    try:
        return get_graph_network(city=req.city, limit=req.limit)
    except Exception as exc:
        logger.exception("Graph network query failed")
        raise HTTPException(status_code=500, detail=f"Graph network error: {exc}") from exc


@app.get("/graph/node/{node_id}")
def graph_node_details(node_id: str, neighbor_limit: int = 40) -> dict:
    """Return one node with connected neighbors and relationship metadata."""
    try:
        data = get_node_details(node_id=node_id, neighbor_limit=neighbor_limit)
    except Exception as exc:
        logger.exception("Graph node query failed")
        raise HTTPException(status_code=500, detail=f"Graph node error: {exc}") from exc

    if data is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return data


@app.post("/ingest/start")
def start_ingest_job(req: IngestStartRequest, background_tasks: BackgroundTasks) -> dict:
    """Start async ingestion for a city and return a trackable job id."""
    city = req.city or DEFAULT_CITY
    job_id = str(uuid4())
    logger.info("Creating ingest job: city=%s, job_id=%s", city, job_id)
    with _ingest_jobs_lock:
        _ingest_jobs[job_id] = {
            "job_id": job_id,
            "city": city,
            "status": "queued",
            "progress": 0,
            "step": "Queued",
            "step_index": 0,
            "step_total": 3,
            "step_progress": 0,
            "result": None,
            "error": None,
            "created_at": time(),
            "started_at": None,
            "updated_at": time(),
            "finished_at": None,
            "elapsed_seconds": 0,
            "eta_seconds": None,
            "logs": [{"ts": int(time()), "message": "Job created"}],
        }
    _set_job_state(job_id, started_at=time())
    background_tasks.add_task(_run_ingest_job, job_id, city)
    return _ingest_jobs[job_id]


@app.get("/ingest/status/{job_id}")
def get_ingest_status(job_id: str) -> dict:
    """Get status/progress for a previously started ingestion job."""
    with _ingest_jobs_lock:
        job = _ingest_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    logger.info(
        "Ingest status polled: job_id=%s status=%s progress=%s",
        job_id,
        job.get("status"),
        job.get("progress"),
    )
    return job


@app.post("/ingest/trigger")
def manual_ingest(city: Optional[str] = None) -> dict:
    """Trigger full ingestion (hotels + news) for one or all configured cities."""
    cities = [city] if city else HOTELS_CITIES
    logger.info("API /ingest/trigger: cities=%s", cities)
    try:
        hotel_count = ingest_hotels(cities, max_results=HOTEL_MAX_RESULTS)
    except Exception as exc:
        logger.exception("Hotel ingestion failed")
        raise HTTPException(status_code=500, detail=f"Hotel ingestion error: {exc}") from exc

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


_static_dir = Path(__file__).resolve().parent / "static"
if _static_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(_static_dir), html=True), name="ui")


@app.get("/")
def root_redirect() -> RedirectResponse:
    """Send browser users to the web UI."""
    return RedirectResponse(url="/ui")
