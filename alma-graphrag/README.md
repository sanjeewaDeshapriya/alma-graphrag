# ALMA GraphRAG Phase 1 (Graph-only)

Phase 1 MVP: full GraphRAG stack with Neo4j schema, embeddings, and CRAG orchestration. Graph reasoning remains the primary signal, with vector retrieval as a supporting layer.

## Prereqs

- Docker Desktop (for Neo4j + Redis)
- Python 3.11+

## Quick start

1) Start Neo4j + Redis:

```
docker-compose up -d
```

2) Create a local env file:

```
copy .env.example .env
```

3) Install dependencies:

```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

4) Initialize schema + seed data:

```
python scripts\run_schema.py
python scripts\run_seed.py
```

5) Ingest hotels (city is optional):

```
python scripts\run_ingest_hotels.py --city Piliyandala

# Multiple nearby cities
python scripts\run_ingest_hotels.py --cities Piliyandala,Maharagama,Homagama
```

6) Ingest news (optional, RSS):

```
python scripts\run_ingest_news.py
```

7) Run API:

```
uvicorn src.api.main:app --reload
```

8) Example query:

```
POST http://127.0.0.1:8000/query
{
  "question": "Find quiet hotels with parking",
  "city": "Piliyandala"
}

# Manual ingest trigger (API)
POST http://127.0.0.1:8000/ingest/trigger

9) Optional: start scheduler (daily hotel refresh, 15-min news refresh):

```
python scripts\run_scheduler.py

10) (Optional) Run KG construction + embeddings pipeline:

```
python main.py
```
```
```

## Notes

- GraphRAG is the primary reasoning layer, with vector retrieval used for support.
- CRAG loop uses self-grading and query rewrite (no web search fallback).
- Change city at ingest or query time.
- Redis is optional cache for CRAG responses.
- Set LLM_EXTRACT_ENABLED=true to enrich amenities/locations from hotel text.
