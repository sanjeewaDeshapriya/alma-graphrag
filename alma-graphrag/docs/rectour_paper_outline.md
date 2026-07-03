# Workshop Paper Outline — RecTour @ RecSys (or similar tourism-RecSys venue)

**Working title:** *Disruption-Aware Hotel Recommendation over a Live Knowledge Graph:
Weighted Multi-Hop GraphRAG vs. Filter and Vector Baselines*

**Authors:** D. S. Deshapriya, R. Ranaweera (supervisor)

**Target length:** 4–6 pages (workshop format, ACM 2-column)

---

## 1. Introduction (0.75 p)
- Problem: hotel recommendation quality depends on *live* context — traffic,
  events, disruptions — which static recommenders and vector RAG cannot see.
- Products (Expedia Romie) prove commercial demand for disruption handling;
  academic benchmarks (TravelPlanner → Flex-TravelPlanner, TP-RAG) are moving
  toward dynamic constraints, but no public system studies recommendation over
  a *continuously updated* knowledge graph.
- Contributions:
  1. A live KG for Colombo hotels fusing Google Places/LiteAPI, news/events,
     and per-hotel real-time travel-time edges (Google Distance Matrix).
  2. A weighted multi-hop retriever with intent-adaptive scoring
     (spatial / accessibility / facility / economic / disruption).
  3. Deterministic persona-based personalisation: the same live event produces
     *opposite* rankings for contrasting personas (event-seeker vs quiet-seeker).
  4. Comparative evaluation vs filter, vector (TF-IDF), and RRF-hybrid
     baselines with an honest error analysis.

## 2. Related work (0.5 p)
- GraphRAG line: MS GraphRAG, LightRAG, HippoRAG/2; temporal GraphRAG
  (T-GRAG, TS-Retriever, IA-RAG).
- Travel planning agents/benchmarks: TravelPlanner (solved), Flex-TravelPlanner,
  TP-RAG, TripTailor.
- Position: we study the *retrieval/ranking* layer over a dynamic graph, not
  itinerary planning; complementary to those benchmarks.

## 3. System (1.25 p)
- KG schema figure: Hotel, City, Amenity, AttractionType, TrafficSignal, Event
  (+ LOCATED_IN.travel_time_traffic_min as accessibility weight).
- Ingestion pipeline + refresh cadence; entity canonicalisation
  (city aliases, placeholder-price hygiene) — brief but include, reviewers
  reward data honesty.
- Safe NL→intent→parameterised-Cypher (why not free-form text2cypher).
- Scoring: composite score with per-component breakdown; weights_for_intent;
  persona presets + ActiveEvent geographic impact zone.

## 4. Evaluation (1.5 p)
- Setup: 40-hotel Colombo pool, 10-query set across economic / quality /
  accessibility / multi-dimensional categories, k=10.
  **TODO before submission: expand to ≥50 queries; replace rule-based gold with
  ≥3 human annotators; report Krippendorff's alpha + significance tests
  (paired bootstrap).**
- Systems: Filter, Vector(TF-IDF), Graph (ours), Hybrid RRF(G+V), RRF(G+F),
  intent-adaptive hybrid.
- Existing results (to be re-run after annotation): overall nDCG Graph 0.760 /
  Filter 0.741 / Vector 0.267; accessibility category Graph 1.000 vs Filter
  0.064 — the killer result (travel-time lives in graph edges, invisible to
  filter/vector); single-attribute queries: Filter optimal (honest negative
  result — keep it, it motivates the adaptive hybrid).
- Personalisation case study: F1 Street Race @ Galle Face — event_seeker top-5
  all within 1.4 km of the event; quiet_seeker top-5 all ≥3 km away; disjoint
  sets from identical graph state. Map figure.

## 5. Limitations & future work (0.5 p)
- Single city, modest pool; rule-based gold (if not yet replaced) is
  structurally favourable to attribute-aware systems — state this plainly.
- Real traffic sampled off-peak → disruption dimension demonstrated via
  injected congestion; live validation pending.
- Future: bitemporal fact versioning + as-of retrieval; learned ranking
  (LTR → offline RL with off-policy evaluation) to replace preset persona
  weights; public benchmark release (time-stamped graph snapshots +
  disruption scenarios).

## 6. Checklist before submission
- [ ] ≥50 queries, ≥3 annotators, agreement + significance reported
- [ ] Re-run all baselines on annotated gold (results may drop — that is fine)
- [ ] Ablation: each scoring component removed once
- [ ] Reproducibility: pinned graph snapshot (Neo4j dump) + seed + one command
- [ ] Code + anonymised data released on GitHub
- [ ] Check RecTour CFP deadline and page limit; fallback venues: ACM RecSys
      LBR track, CIKM short, ECIR short
