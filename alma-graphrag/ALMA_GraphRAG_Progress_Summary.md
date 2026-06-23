# ALMA-GraphRAG — Research Progress Summary

**Student:** Dilshan Sanjeewa Deshapriya
**Date:** June 2026
**Topic:** Graph-based RAG System for Sri Lanka Hotel Recommendations

---

## What Was Built

### 1. Knowledge Graph (Neo4j)

Designed and implemented a property graph database with the following node types and relationships:

| Node Types | Relationships |
|---|---|
| Hotel, City, Amenity | LOCATED_IN, HAS_AMENITY |
| Event, NewsSignal | AFFECTED_BY, MENTIONED_IN |
| Location | NEAR |
| RoomType, BoardType | HAS_ROOM, OFFERS_BOARD |

The graph stores real hotel data including ratings, price ranges (LKR), amenities, GPS coordinates, and linked news events.

---

### 2. Data Ingestion Pipeline

Four independent data sources are integrated and automatically scheduled:

- **Google Places API** — Scrapes hotel data (name, rating, address, coordinates, amenities) for configured Sri Lanka cities. Runs automatically every 24 hours via a background scheduler.
- **LiteAPI (`/hotels/rates`)** — Searches bookable hotels by city/country and ingests live rate plans plus hotel content (name, address, coordinates, star rating, facilities). Each rate plan becomes a `RoomType` node (price, board, refundability) linked via `HAS_ROOM`, and meal plans become `BoardType` nodes via `OFFERS_BOARD`. Runs on the same 24h schedule when enabled (`LITEAPI_ENABLED=true`).
- **NewsAPI.org + GNews.io** — Dual-provider news ingestion focused on Sri Lanka tourism/travel topics, with automatic fallback to RSS feeds. Refreshed every 15 minutes.
- **LLM Entity Extractor** — Uses GPT-4o-mini to extract additional amenities and nearby location names from unstructured hotel descriptions and links them into the graph.

---

### 3. CRAG Reasoning Pipeline

Implemented a **Corrective RAG (CRAG)** pipeline using LangGraph with the following sequential flow:

```
Retrieve (Neo4j)  →  Grade (LLM scores 0–1)  →  [if low score] Rewrite Query  →  Re-Retrieve  →  Generate (GPT)
```

- **Context retrieval** — Pulls rich structured context from the graph including hotels, amenities, events, news signals, and city-level statistics.
- **Self-correction** — LLM grades context relevance; if below threshold, rewrites the query and retries (configurable maximum retries).
- **Grounded generation** — Final prompt instructs GPT to cite specific hotel names, ratings, price ranges, amenities, and linked events.
- **Caching** — Results cached in Redis for 15 minutes to reduce API cost and latency.

---

### 4. REST API (FastAPI)

A production-grade REST API exposing the full system:

| Endpoint | Purpose |
|---|---|
| `POST /query` | Natural language hotel query via CRAG pipeline |
| `GET  /graph/overview` | Graph statistics — node/edge counts, city stats |
| `POST /graph/network` | Interactive graph sub-network for visualization |
| `GET  /graph/node/{id}` | Node details with all neighbors and relationships |
| `POST /ingest/start` | Start async ingestion job with live progress tracking |
| `GET  /ingest/status/{job_id}` | Poll ingestion progress and ETA |
| `POST /ingest/trigger` | Manual full ingestion trigger |
| `POST /ingest/news` | News-only ingestion from all providers |
| `POST /graph/clear` | Delete all graph data (requires confirmation text) |
| `GET  /health` | Readiness probe |

---

### 5. Web UI — Graph Intelligence Console

A browser-based management console served directly from the API:

- **Overview Panel** — Live graph statistics (node/edge counts, city averages, price range categories).
- **Interactive Graph** — Cytoscape.js visualization with zoom/pan, colour-coded node labels, and typed edges.
- **Node Explorer** — Click any graph node to inspect its properties and all connected neighbor relationships.
- **Hotel Map** — Google Maps integration displaying hotel locations as interactive markers.
- **Ingestion Console** — Start ingestion jobs with real-time progress bar, ETA estimate, and live log stream.

---

## System Architecture

```
[Google Places API] ──┐
[NewsAPI / GNews]   ──┼──► [Ingestion Pipeline] ──► [Neo4j Knowledge Graph]
[RSS Feeds]         ──┘                                       │
                                                              ▼
[User Query] ──► [FastAPI] ──► [CRAG Pipeline] ──► [GPT-4o-mini] ──► Answer
                                      │
                               [Redis Cache]
```

**Tech Stack:** Python · FastAPI · LangGraph · Neo4j · OpenAI (GPT-4o-mini) · Google Places API · Google Maps API · NewsAPI.org · GNews.io · APScheduler · Redis · Cytoscape.js

---

## Current Status

| Area | Status |
|---|---|
| End-to-end CRAG pipeline (ingest → graph → query → answer) | ✅ Complete |
| Google Places hotel ingestion for Sri Lanka cities | ✅ Complete |
| Dual news provider ingestion (NewsAPI + GNews + RSS fallback) | ✅ Complete |
| LLM-based entity extraction and graph enrichment | ✅ Complete |
| Background scheduler (24h hotels, 15min news) | ✅ Complete |
| REST API with all endpoints and error handling | ✅ Complete |
| Web UI with graph, map, and ingestion console | ✅ Complete |
| Redis caching for CRAG results | ✅ Complete |
| Research evaluation framework and benchmarks | 🔄 In Progress |
| Security hardening (API key rotation, endpoint auth) | 📋 Planned |
| Ablation studies and baseline comparisons | 📋 Planned |

---

## Research Contribution

The core novel contribution of this project is the **CRAG + Knowledge Graph hybrid architecture**: combining structured graph context (relationships between hotels, events, news, and locations) with LLM self-correction to produce grounded, evidence-backed hotel recommendations.

Unlike standard RAG, the system evaluates context quality before generation and iteratively refines its retrieval strategy, reducing hallucination risk while leveraging real-time news signals linked directly to hotels in the knowledge graph.

---

*ALMA-GraphRAG · Final Year Research · 2026*
