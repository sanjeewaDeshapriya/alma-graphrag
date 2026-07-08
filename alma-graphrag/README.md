# ALMA GraphRAG Phase 1 (Graph-only)

Phase 1 MVP: full GraphRAG stack with Neo4j schema, embeddings, and CRAG orchestration. Graph reasoning remains the primary signal, with vector retrieval as a supporting layer.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  FastAPI  (/query, /ingest/trigger, /ingest/news, /health)  │
└───────────┬────────────────────────┬────────────────────────┘
            │                        │
    ┌───────▼─────────┐      ┌───────▼──────────┐
    │   CRAG Pipeline  │      │  Ingest Pipeline  │
    │  retrieve→grade  │      │  Hotels + News    │
    │  →rewrite→gen    │      │  + Event Linker   │
    └───────┬──────────┘      └───────┬───────────┘
            │                         │
    ┌───────▼─────────────────────────▼───────┐
    │           Neo4j Knowledge Graph          │
    │  Hotel, City, Amenity, Location, Event,  │
    │  NewsSignal + relationships              │
    └──────────────────┬──────────────────────┘
                       │
    ┌──────────────────▼──────────────────────┐
    │           Redis (optional cache)         │
    └─────────────────────────────────────────┘
```

## News Providers

The system supports **three news sources** in priority order:

| Provider | Free Tier | Key Required | Notes |
|----------|-----------|-------------|-------|
| [NewsAPI.org](https://newsapi.org/register) | 100 req/day | `NEWS_API_KEY` | Broad coverage, 24h delay |
| [GNews.io](https://gnews.io/) | 100 req/day | `GNEWS_API_KEY` | Google News rankings |
| RSS feeds | Unlimited | None | Sri Lankan news outlets (fallback) |

If no API keys are set, the system automatically falls back to RSS-only mode.

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

6) Ingest news (APIs + RSS fallback):

```
python scripts\run_ingest_news.py

# RSS-only mode (skip API providers)
python scripts\run_ingest_news.py --rss-only
```

7) Run API:

```
uvicorn src.api.main:app --reload
```

8) Health check:

```
GET http://127.0.0.1:8000/health
```

9) Open web UI:

```
http://127.0.0.1:8000/ui
```

10) Optional: start scheduler (daily hotel refresh, 15-min news refresh):

```
python scripts\run_scheduler.py
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health / readiness check |
| `POST` | `/query` | Ask a question against the knowledge graph |
| `GET` | `/graph/overview` | Get graph summary counts + optional city stats |
| `POST` | `/graph/network` | Get graph nodes/edges for interactive visualization |
| `GET` | `/graph/node/{node_id}` | Get one node with neighbor details |
| `POST` | `/graph/clear` | Clear all graph nodes + relationships (requires confirmation) |
| `GET` | `/client/config` | Client-safe UI runtime config (map key, defaults) |
| `POST` | `/ingest/start` | Start async full ingest by running hotel + news scripts |
| `GET` | `/ingest/status/{job_id}` | Check async ingestion progress/status |
| `POST` | `/ingest/trigger` | Trigger full hotel + news ingestion |
| `POST` | `/ingest/news` | Trigger news-only ingestion (APIs + RSS) |

`GET /ingest/status/{job_id}` includes live UI-friendly fields:
- `progress` (0-100 overall)
- `step` (current stage name)
- `step_index`, `step_total`, `step_progress`
- `elapsed_seconds`, `eta_seconds`

`POST /ingest/start` runs script commands in order:
- `python scripts/run_ingest_hotels.py --city <city>`
- `python scripts/run_ingest_news.py`

## Example Test Queries

### 🏨 Hotel Discovery

```bash
# Top-rated luxury hotels
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the top-rated luxury hotels in Piliyandala with swimming pools?", "city": "Piliyandala"}'

# Budget hotels
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Find budget-friendly hotels under 5000 LKR per night near Maharagama", "city": "Maharagama"}'

# Multi-amenity filter
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Which hotels in Colombo have both free parking and a restaurant?", "city": "Colombo"}'
```

### 🏖️ Amenity-Based Search

```bash
# Spa + fitness
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "List hotels with spa and fitness center facilities", "city": "Piliyandala"}'

# Business traveler needs
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Hotels suitable for business travelers with WiFi, business center, and room service", "city": "Colombo"}'
```

### 📍 Location-Aware

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Hotels near major landmarks in Dehiwala with good transport access", "city": "Dehiwala"}'
```

### 📰 News & Event-Aware

```bash
# Events affecting hotels
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Are there any recent events or news affecting hotels in Piliyandala?", "city": "Piliyandala"}'

# Tourism development impact
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Which hotels are impacted by recent tourism development news?", "city": "Colombo"}'
```

### 🔄 Comparative

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Compare hotels in Piliyandala vs Maharagama by rating and price", "city": "Piliyandala"}'
```

### 🧭 Complex Multi-Criteria

```bash
# Family-friendly
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Recommend a family-friendly hotel with pool, restaurant, and good security", "city": "Colombo"}'

# Best value
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Best value hotel with high rating, low price, and multiple amenities in Maharagama", "city": "Maharagama"}'

# Safety-focused
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Safe, well-reviewed hotels with CCTV and security guards near Homagama center", "city": "Homagama"}'
```

### 🔧 Manual Ingest Trigger

```bash
# Full ingest (hotels + news)
curl -X POST http://127.0.0.1:8000/ingest/trigger

# News-only ingest
curl -X POST http://127.0.0.1:8000/ingest/news
```

## Notes

- GraphRAG is the primary reasoning layer, with vector retrieval used for support.
- CRAG loop uses self-grading and query rewrite (no web search fallback).
- Change city at ingest or query time.
- Redis is optional cache for CRAG responses.
- Set `LLM_EXTRACT_ENABLED=true` to enrich amenities/locations from hotel text.
- News ingestion tries NewsAPI → GNews → RSS in order; set API keys in `.env`.
- Set `GOOGLE_MAPS_API_KEY` in `.env` to enable hotel map with zoom/pan (Google Maps + Places library).
