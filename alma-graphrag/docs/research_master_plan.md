# ALMA-GraphRAG → PhD-Level Research Master Plan

*Author: D. S. Deshapriya, with analysis assistance. Last updated: 2026-07-03.*
*Companion documents: [rectour_paper_outline.md](rectour_paper_outline.md) (first paper), project_audit.md (code audit).*

---

## Part 1 — Where we are (honest assessment)

### What is genuinely built and working
| Component | Status |
|---|---|
| Live ingestion (Google Places, LiteAPI, news APIs/RSS, Google Distance Matrix traffic) | Working, scheduled |
| Neo4j KG (Hotel/City/Amenity/AttractionType/TrafficSignal/Event; per-hotel traffic edges) | Working, ~40 hotels, 1 city |
| Safe NL→intent→parameterised Cypher (regex + optional LLM slot-fill) | Working |
| Weighted multi-hop retriever, 5-dimension composite score, intent-adaptive weights | Working, verified live |
| Deterministic persona personalisation + ActiveEvent geo-zones ("same event → opposite rankings") | Working, verified end-to-end |
| Evaluation harness: P@K/R@K/nDCG/MRR, 3 baselines + RRF hybrids, 10 queries | Working |
| Unit tests (52) + CI | Working (added 2026-07-03) |

### Honest weaknesses (ranked by how badly they block publication)
1. **Circular gold labels** — rule-based relevance uses the same dimensions the retriever scores. Blocks *any* strong empirical claim.
2. **Scale** — 40 hotels, 1 city, 10 queries. Nothing generalises.
3. **No temporal semantics** — the graph stores only current state; "dynamic KG" is refresh, not time-aware reasoning.
4. **No learned components** — all ranking weights are hand-set; the proposal's RL/LTR claims are unimplemented.
5. **No positioning against literature** — never evaluated on or compared with any published system/benchmark.
6. **Disruption never fired on real data** — traffic sampled off-peak; demos use injected congestion.
7. **No human evaluation** — zero annotators, zero users.

---

## Part 2 — State of the art (researched July 2026)

### Research landscape
**GraphRAG methods:** Microsoft GraphRAG (community summarisation), LightRAG (dual-level retrieval, ~60% cheaper indexing), HippoRAG/HippoRAG-2 (personalised PageRank as associative memory), G-Retriever, GFM-RAG (graph foundation models). Meta-analyses now ask *when* graphs actually beat vector RAG — any new paper must answer that question for its domain.

**Temporal GraphRAG (the emerging frontier):** T-GRAG (time-stamped graphs + temporal query decomposition), TS-Retriever (event dynamics), IA-RAG (Allen interval-algebra temporal reasoning), entity-event KGs for temporal-causal consistency. **Nobody has applied this line to recommendation over a live operational graph.**

**Travel planning agents:** TravelPlanner is solved (>97% via LLM+symbolic hybrids). The field moved to exactly our setting: Flex-TravelPlanner (constraints changing mid-plan), TP-RAG (spatiotemporal RAG), TripTailor (real-world personalisation), GroupTravelBench, VeriTrip, TravelEval, COMPASS (constrained optimisation). Winning recipe everywhere: **LLM for language, symbolic/structured layer for correctness** — which is exactly our NL→intent→Cypher architecture. We are architecturally on-trend; we lack scale and rigour.

**Recommender-systems methods relevant to us:** KG-based recommenders (KGAT, path-based RL recommenders like CADRL), learning-to-rank (LambdaMART still SOTA for tabular ranking), contextual bandits and offline RL with off-policy evaluation (IPS/DR estimators), temporal graph networks (TGN/TGAT) for dynamic-graph representation learning.

### Product landscape
- **Expedia Romie** — proactive trip monitoring + disruption handling; proves commercial demand for exactly our thesis.
- **Booking.com AI Trip Planner, Mindtrip, Google Gemini travel** — discovery-focused planners.
- **2026 trend:** agentic booking (AI takes actions, adapts itineraries as conditions change). Market projection ~$1.3B (2026) → ~$5.8B (2035).
- **What none of them publish:** how to rank under live disruption with verifiable graph grounding. That is our lane.

### The gap we own
> **Disruption-aware recommendation over a temporal knowledge graph, with a public benchmark.**
> Products do it proprietarily; temporal-GraphRAG papers have no recommendation testbed; travel benchmarks have no live graph. We sit at the intersection with a working system already in hand.

---

## Part 3 — Research vision and questions (PhD framing)

**Thesis statement:** *Recommendation systems operating in dynamic physical environments (traffic, events, closures) require temporally-versioned knowledge graphs and learned, disruption-aware ranking; retrieval-augmented generation grounded in such graphs outperforms static, vector-only, and snapshot-graph approaches, and the gap widens as environment volatility increases.*

- **RQ1.** How should facts with real-world volatility (travel times, events, prices) be represented so retrieval can reason *as-of* a time point and *ahead* of one (forecast)? → bitemporal KG + temporal retrieval operators.
- **RQ2.** Does temporal-graph grounding measurably improve recommendation quality over static-graph, vector, and hybrid baselines — and under what volatility conditions? → benchmark + controlled disruption replay.
- **RQ3.** Can ranking weights be *learned* (LTR → bandits → offline RL) to beat hand-tuned and persona-preset weights, safely, from logged/simulated feedback? → learned ranking with off-policy evaluation.
- **RQ4.** Does an agentic multi-step graph retrieval loop (plan → traverse → self-grade → refine) beat one-shot retrieval, and at what latency/cost? → agentic vs one-shot ablation.

---

## Part 4 — Technical plan (the four pillars, in detail)

### Pillar A — Temporal knowledge graph (RQ1)
**A1. Bitemporal fact model.** Every volatile relationship/property gets `valid_from`, `valid_to` (world time) and `recorded_at` (system time). Concretely in Neo4j: reify volatile state as nodes, e.g. `(:Hotel)-[:HAS_STATE]->(:HotelState {price, valid_from, valid_to, recorded_at})`, `(:TrafficObservation {eta_min, delta_min, observed_at})-[:ON_ROUTE_TO]->(:Hotel)`. Current-state edges kept as a materialised "now" view so existing retriever queries stay fast.
**A2. As-of retrieval operators.** Extend the retriever's Cypher templates with `AS OF $t` semantics (filter on validity intervals). Add *trend features*: ETA slope over last N observations, event proximity-in-time.
**A3. Snapshot & replay infrastructure.** Nightly graph snapshots (Neo4j dumps) + append-only observation log → enables *disruption replay*: re-run any query at any historical timestamp. This is the machinery that makes both the benchmark (Pillar B) and offline RL (Pillar C) possible. **Fix first:** the traffic scheduler must sample peak hours (7–9 am, 5–7 pm Colombo) so real congestion enters the log — right now all real data is off-peak.
**A4 (stretch, model training).** Train a **temporal graph network (TGN/TGAT-style)** on the observation stream to produce time-aware hotel embeddings; compare vs static embeddings for retrieval. This is a publishable component by itself.

### Pillar B — Public benchmark: "DisruptRec" (RQ2) — *the highest-impact artifact*
**B1. Scale the graph.** 3+ Sri Lankan cities (Colombo, Kandy, Galle) → 500–2,000 properties via LiteAPI + OpenStreetMap POIs (free, global) + GTFS transit where available. The ingestion pipeline is already city-agnostic.
**B2. Query set.** ≥150 queries stratified across economic / quality / accessibility / multi-dimensional / disruption-conditioned categories; author half manually, generate half with an LLM and *manually verify all*.
**B3. Disruption scenarios.** Two tracks: (a) *replayed real* disruptions from the observation log (peak-hour traffic, real events from news ingestion); (b) *scripted synthetic* disruptions (road closure, stadium event, flood zone) injected with known ground truth — synthetic gives controlled difficulty, real gives validity.
**B4. Human gold labels.** ≥3 annotators per query (tourism students / hotel staff / crowdsourcing), graded relevance 0–3, with a written annotation guideline. Report inter-annotator agreement (Krippendorff's α ≥ 0.6 target). This *permanently* kills the circular-gold problem.
**B5. Release.** Time-stamped graph snapshots + queries + labels + evaluation script on GitHub/Zenodo with a DOI and leaderboard README. Benchmarks are the most-cited artifact class a small lab can produce.

### Pillar C — Learned ranking (RQ3): the model-training program
Train in escalating order of sophistication; each stage is a paper section and a skill milestone.

**C1. Feature store.** Per (query, hotel) pair extract ~30 features: the 5 existing component scores + raw attributes (price, rating, stars, amenity overlap), spatial (distance to city centre / named attraction / event), temporal (current ETA delta, ETA slope, event time-distance), intent flags. Log every retrieval through the API to build an interaction dataset.
**C2. Learning-to-rank.** LambdaMART via XGBoost/LightGBM (`rank:ndcg`) on the human-labelled benchmark: train/val/test split *by query*, 5-fold CV, compare vs hand-weighted retriever, filter, vector, RRF hybrids. Deliverable: does learned ranking beat `weights_for_intent`? Feature-importance analysis tells you *which graph signals matter* — that is the scientific finding. Hardware: laptop CPU is enough.
**C3. Contextual bandits for personalisation.** Replace fixed persona presets: context = (intent vector, persona, disruption state), arms = weight configurations (or direct slate ranking), reward = simulated click model calibrated on annotations first, later real UI clicks. LinUCB / Thompson sampling. Evaluate with **off-policy estimators (IPS, doubly-robust)** — this is the "Safe RL" claim done properly and defensibly without a live user base.
**C4. Offline RL (stretch).** CQL/IQL on logged trajectories for session-level re-ranking. Only attempt after C3 works; frame as "safe policy improvement with OPE guarantees".
**C5. LLM components worth training/fine-tuning.**
   - **Text2Cypher:** fine-tune a small open model (Llama-3.x-8B / Qwen-7B, QLoRA on Colab/Kaggle GPU) on the Neo4j text2cypher public dataset + ~500 synthetic examples from our schema; compare against the regex+slot-fill parser (accuracy, safety, latency). Keeps the safe parameterised executor; the model only emits the intent/typed slots.
   - **Retrieval grader:** distil GPT-4o-class grading judgments into a small cross-encoder so the CRAG self-correction loop is cheap enough to actually run per-query.
   - Rigor: hold-out schema entities to test generalisation; report exact-match + execution accuracy.

### Pillar D — Agentic retrieval loop (RQ4)
Make the CRAG skeleton real: plan (decompose query) → retrieve (weighted traversal) → grade (trained grader from C5) → refine (re-traverse with adjusted intent) → generate with **citations to graph node IDs** (verifiable grounding; measure faithfulness/hallucination rate with an LLM-judge + human spot-check). Ablate: one-shot vs agentic vs hybrid-RRF on DisruptRec + on TP-RAG/TripTailor for external comparability. Report quality *and* latency/cost — reviewers respect honest cost curves.

---

## Part 5 — Timeline (18 months, part-time realistic)

| Phase | Months | Work | Deliverable / Gate |
|---|---|---|---|
| 0. Rigor floor | 0–1 | ✅ code cleanup, tests, CI (done); peak-hour traffic sampling; retrieval logging; annotation guideline v1 | CI green; real congestion in log |
| 1. First paper | 1–3 | Expand to 50 queries, 3 annotators, significance tests; re-run all baselines | **RecTour/RecSys-LBR workshop submission** |
| 2. Temporal KG | 3–7 | Bitemporal schema, as-of retrieval, snapshot/replay infra | Replay any query at any timestamp |
| 3. Benchmark | 5–10 | 3 cities, 150+ queries, disruption tracks, full annotation, public release | **DisruptRec v1 + full paper (RecSys/CIKM/ECIR)** |
| 4. Learned ranking | 8–13 | Feature store, LambdaMART, bandits + OPE; text2cypher fine-tune | **Second full paper**; learned vs hand-tuned verdict |
| 5. Agentic + synthesis | 12–16 | Agentic loop, faithfulness eval, external benchmarks; small user study (15–20 users, SUS + task success) | **Third paper or journal (IP&M / UMUAI)** |
| 6. Thesis/portfolio | 15–18 | Consolidate; PhD applications with 2–3 publications in hand | PhD admission portfolio |

**Parallel track — PhD applications:** shortlist supervisors working on GraphRAG/temporal KGs/recsys (look at who authors the papers above); email with the workshop paper attached from Month 4; a published benchmark makes you a *desirable* applicant, not a hopeful one.

---

## Part 6 — Senior-researcher training regimen (how you level up while doing this)

1. **Reading discipline:** 2 papers/week from Part 2's lists; write a 5-line structured summary each (problem / method / evidence / flaw / steal-able idea). 6 months ≈ 50 papers = field fluency.
2. **Experiment discipline:** every experiment = one script + pinned data snapshot + seed + one command; results auto-written to `evaluation/results/` with config hash. Dated lab-notebook entry per experiment (`docs/labnotes/`).
3. **Pre-registration habit:** write the hypothesis and expected outcome *before* running; negative results get recorded, not deleted (your "filter wins single-attribute queries" finding is exactly the honesty reviewers reward).
4. **Statistics floor:** paired bootstrap / randomisation tests for every comparison table; never claim "better" without a p-value and effect size again.
5. **Community entry:** submit to a workshop early (deadline pressure teaches more than any course); volunteer as a sub-reviewer once you have one publication; present at local university seminars.
6. **Writing cadence:** one page of the current paper per week from Month 1 — papers written alongside experiments are 10× easier than papers written after.

---

## Part 7 — Business model (detailed)

**Core insight:** the defensible asset is not the model or the code — it is the **live local knowledge graph** (Sri Lankan traffic patterns, events, closures, small properties) that global OTAs will not build for a small market, plus the disruption-ranking layer on top.

**Customer segments & value proposition**
| Segment | Pain | Our offer |
|---|---|---|
| Local OTAs / DMCs (destination management companies) | Generic rankings ignore live conditions; manual re-planning when disruptions hit | Disruption-aware recommendation API + proactive re-recommendation alerts |
| Hotel chains (3–5 properties) | Invisible when conditions favour them (e.g., "quiet during event weekend") | Context-aware placement + demand signals from disruption forecasts |
| Tourism board / provincial councils | No live picture of accessibility & visitor experience | White-label trip assistant + analytics dashboard |
| Corporate travel desks (Colombo) | Traffic makes airport/meeting logistics unreliable | ETA-aware hotel/venue selection API |

**Revenue lines (in order of realism):**
1. **B2B API SaaS** — tiered by request volume (e.g., $99/$399/$999 per month); the recommendation + disruption-alert endpoints you already have are the MVP.
2. **Booking affiliate** — LiteAPI is already integrated; commission per completed booking flows through existing code.
3. **White-label licence** — annual contract with a tourism board or a large DMC (one contract can fund the research).
4. **Data/insights reports** — quarterly "accessibility & disruption index" for hotel investors (byproduct of the benchmark infrastructure).

**Go-to-market:** get **one design partner** (a boutique DMC or 3-hotel group) within 3 months — free pilot in exchange for logged interactions (which feed Pillar C's bandit training: *the business and the research feed each other*). Expand city coverage only when a partner pays for it.

**Moat:** (a) longitudinal observation log — every month of collected traffic/event history is history a competitor cannot backfill; (b) published benchmark = credibility no local competitor has; (c) switching cost once a partner's rankings are bandit-tuned on their users.

**Costs & risks:** Google API costs dominate (cap via caching + peak-hour-only sampling); Neo4j Community + one VPS ≈ $40–80/month. Risks: Google/Expedia ship local disruption features (mitigate: stay B2B infrastructure, not consumer app); API price hikes (mitigate: OSM + crowdsourced signals as fallback); single-founder bandwidth (mitigate: research-first, business only after design partner traction).

---

## Part 8 — Immediate next actions (this month)
1. Switch traffic scheduler to peak-hour sampling windows (unblocks everything temporal).
2. Add retrieval logging (query, intent, ranking, clicked result) to the API — future training data starts accruing *now*.
3. Write annotation guideline v1 + recruit 3 annotators; expand queryset to 50.
4. Draft the workshop paper from [rectour_paper_outline.md](rectour_paper_outline.md); check the RecTour CFP deadline.
5. Email 3 potential design partners (DMCs) for a pilot conversation.
