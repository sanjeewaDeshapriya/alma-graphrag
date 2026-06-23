# ALMA-GraphRAG Final Year Research Implementation Plan

## 1. Document analyzed
Primary reference: project_audit.md

This plan cross-checks the audit findings against the current codebase and defines a practical roadmap for completing the research project to submission quality.

## 2. Current implementation status (audit cross-check)

### 2.1 Critical bug/fix plan status

| Audit item | Status | Evidence | Notes |
|---|---|---|---|
| Fix missing import in CRAG | Done | src/crag/graph.py | os import is present. |
| API imports CRAG from src | Done | src/api/main.py | Uses src.crag.graph. |
| Remove duplicate root crag package | Partial | crag/ and src/crag/ both exist | Runtime path is fixed, but duplicate package still exists and should be removed/archived. |
| Add NewsAPI + GNews fallback | Done | src/ingest/news_api.py, src/api/main.py | API providers are integrated with RSS fallback. |
| Add /health endpoint | Done | src/api/main.py | Health endpoint available. |
| Add query error handling | Done | src/api/main.py | Query endpoint catches and returns handled 500 errors. |
| Add CORS middleware | Done | src/api/main.py | CORS enabled; currently open wildcard policy. |
| Improve tourism/travel news query coverage | Done | src/ingest/news_api.py | Multiple tourism/travel focused queries implemented. |
| Expand README queries | Mostly done | README.md | Many test queries are present; can further expand for research evaluation set. |

### 2.2 Additional implementation already completed

- Graph UI endpoints are available:
  - GET /graph/overview
  - POST /graph/network
  - GET /graph/node/{node_id}
- Async ingestion job with progress/status:
  - POST /ingest/start
  - GET /ingest/status/{job_id}
- Graph maintenance endpoint:
  - POST /graph/clear (confirmation protected)
- Professional web UI with:
  - graph overview
  - interactive graph
  - node explorer/details
  - ingestion controls and progress logs
  - hotel map (Google Maps) with zoom/pan markers

## 3. Key gaps to close for final-year research quality

### 3.1 High priority

1. Remove duplicate CRAG code path
- Decide one canonical module (src/crag).
- Delete or archive root crag package to avoid accidental imports.

2. Security hardening
- Rotate all exposed keys immediately (OpenAI, Google, News API, GNews).
- Restrict Google Maps API key by HTTP referrer.
- Restrict backend endpoints with auth for destructive operations:
  - /graph/clear
  - /ingest/start
  - /ingest/trigger
- Replace allow_origins=["*"] with environment-based allowlist.

3. Research evaluation framework
- Build benchmark dataset of representative user queries (100-300).
- Add objective metrics: grounding, relevance, factuality, latency.
- Add baseline comparisons and ablation experiments.

4. Test coverage
- Add unit tests for core graph query functions and ingestion transforms.
- Add API integration tests for /query, /graph/network, /ingest/start, /graph/clear.
- Add smoke test for full pipeline in CI.

### 3.2 Medium priority

1. Scheduler correctness
- Current AsyncIOScheduler usage with sync jobs should be reviewed.
- Either move jobs async or switch to scheduler type better aligned with sync tasks.

2. Dead code cleanup
- build_crag_app path is currently mostly reference code.
- Keep only if needed for experiments; otherwise remove or gate via feature flag.

3. Data model quality
- Add confidence and provenance metadata for generated recommendations.
- Improve event-hotel linking scoring and explainability.

## 4. Proposed phased roadmap (10-week plan)

## Phase 1 (Week 1-2): Stabilize and secure

Goals:
- Make system safe and deterministic for research runs.

Tasks:
- Rotate keys and sanitize .env handling.
- Add auth guard for write/destructive endpoints.
- Remove duplicate root crag package.
- Add environment profiles (dev/staging/prod).
- Add structured logging with request/job correlation ids.

Deliverables:
- Security hardening checklist complete.
- Clean module structure with one CRAG path.

## Phase 2 (Week 3-4): Data and retrieval quality

Goals:
- Improve KG quality and retrieval reliability.

Tasks:
- Improve entity extraction normalization (hotel, city, location aliases).
- Add ingestion data quality checks (missing coordinates, duplicates, stale events).
- Add fallback and retry policy for external APIs.
- Add city normalization mapping (case, aliases, spelling variants).

Deliverables:
- Data quality report with before/after stats.
- Improved graph consistency metrics.

## Phase 3 (Week 5-6): Research contributions in reasoning

Goals:
- Implement novel/defensible methods for final-year research novelty.

Candidate contributions:
- Graph path-aware context compression (shortest evidence paths).
- CRAG self-evaluation with confidence calibration.
- Hybrid reranking (graph score + semantic score + event impact score).
- Explainable response block (which nodes/edges support each recommendation).

Deliverables:
- Contribution design note and implementation.
- Ablation-ready feature flags.

## Phase 4 (Week 7-8): Evaluation and experiments

Goals:
- Produce measurable, publication-style results.

Tasks:
- Build fixed evaluation query set and expected evidence annotations.
- Compare baselines:
  - keyword or simple retrieval baseline
  - graph-only retrieval baseline
  - graph + CRAG (current)
  - graph + CRAG + new contribution
- Track metrics:
  - answer relevance (human and/or rubric LLM)
  - factual grounding rate
  - hallucination rate
  - latency p50/p95
  - ingest freshness impact

Deliverables:
- Experiment table and charts.
- Reproducible scripts to rerun experiments.

## Phase 5 (Week 9-10): Productization and thesis packaging

Goals:
- Convert technical work into submission quality outputs.

Tasks:
- Final demo flow (ingest, query, graph evidence, map visualization).
- Finalize architecture diagrams and methodology chapter inputs.
- Add operations guide and troubleshooting guide.
- Freeze release tag for thesis submission.

Deliverables:
- Thesis-ready architecture + results package.
- Demo video script and reproducibility guide.

## 5. Immediate next implementation checklist (next 7 days)

1. Security first
- Rotate all keys.
- Add endpoint token auth for clear/ingest endpoints.

2. Repository cleanup
- Remove duplicate root crag package.
- Add a migration note in README.

3. Testing foundation
- Add pytest and first 10 tests for graph endpoints and query pipeline.

4. Evaluation prep
- Create evaluation_queries.json with at least 50 diverse queries.
- Add script to run all queries and capture answer/context/latency.

## 6. Suggested markdown artifacts to create next

- docs/research_methodology.md
- docs/evaluation_protocol.md
- docs/ablation_plan.md
- docs/experiment_results_template.md
- docs/threats_to_validity.md

## 7. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| API key leakage or abuse | High | Rotate keys, restrict keys, never expose server secrets, add endpoint auth. |
| Low quality or sparse city data | Medium | Better ingestion validation, city normalization, fallback to all-cities with clear UI labeling. |
| Hallucinated recommendations | High | Grounded evidence extraction and confidence calibration. |
| Unreproducible results | High | Fixed datasets, seed controls, scripted experiment runner. |

## 8. Acceptance criteria for research completion

- End-to-end pipeline stable for at least 7 consecutive daily runs.
- At least one clear research contribution implemented and evaluated against baselines.
- Evaluation metrics and experiment scripts reproducible from repository.
- Security and reliability checklist completed for demo environment.
