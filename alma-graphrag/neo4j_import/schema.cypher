// =============================================
// ALMA-GraphRAG Hotel Schema — Phase 1 (Full Spec)
// =============================================

// --- UNIQUENESS CONSTRAINTS ---
CREATE CONSTRAINT hotel_id IF NOT EXISTS
  FOR (h:Hotel) REQUIRE h.id IS UNIQUE;

CREATE CONSTRAINT city_name IF NOT EXISTS
  FOR (c:City) REQUIRE c.name IS UNIQUE;

CREATE CONSTRAINT district_name IF NOT EXISTS
  FOR (d:District) REQUIRE d.name IS UNIQUE;

CREATE CONSTRAINT amenity_name IF NOT EXISTS
  FOR (a:Amenity) REQUIRE a.name IS UNIQUE;

CREATE CONSTRAINT location_name IF NOT EXISTS
  FOR (l:Location) REQUIRE l.name IS UNIQUE;

CREATE CONSTRAINT price_range_label IF NOT EXISTS
  FOR (p:PriceRange) REQUIRE p.label IS UNIQUE;

CREATE CONSTRAINT road_name IF NOT EXISTS
  FOR (r:RoadSegment) REQUIRE r.name IS UNIQUE;

CREATE CONSTRAINT transport_name IF NOT EXISTS
  FOR (t:TransportMode) REQUIRE t.name IS UNIQUE;

CREATE CONSTRAINT event_id IF NOT EXISTS
  FOR (e:Event) REQUIRE e.id IS UNIQUE;

CREATE CONSTRAINT news_url IF NOT EXISTS
  FOR (n:NewsSignal) REQUIRE n.url IS UNIQUE;

CREATE CONSTRAINT user_id IF NOT EXISTS
  FOR (u:UserProfile) REQUIRE u.id IS UNIQUE;

CREATE CONSTRAINT chunk_id IF NOT EXISTS
  FOR (c:EmbeddingChunk) REQUIRE c.id IS UNIQUE;

// --- STANDARD INDEXES ---
CREATE INDEX hotel_name IF NOT EXISTS
  FOR (h:Hotel) ON (h.name);

CREATE INDEX hotel_rating IF NOT EXISTS
  FOR (h:Hotel) ON (h.rating);

CREATE INDEX hotel_city IF NOT EXISTS
  FOR (h:Hotel) ON (h.city_name);

CREATE INDEX event_type IF NOT EXISTS
  FOR (e:Event) ON (e.type);

CREATE INDEX event_start IF NOT EXISTS
  FOR (e:Event) ON (e.start_time);

CREATE INDEX news_published IF NOT EXISTS
  FOR (n:NewsSignal) ON (n.published_at);

// --- FULLTEXT INDEXES ---
CREATE FULLTEXT INDEX hotel_fulltext IF NOT EXISTS
  FOR (h:Hotel) ON EACH [h.name, h.description, h.address];

CREATE FULLTEXT INDEX news_fulltext IF NOT EXISTS
  FOR (n:NewsSignal) ON EACH [n.title, n.summary];

// --- VECTOR INDEXES ---
CALL db.index.vector.createNodeIndex(
  'hotel_embeddings',
  'Hotel',
  'embedding',
  1536,
  'cosine'
);

CALL db.index.vector.createNodeIndex(
  'event_embeddings',
  'Event',
  'embedding',
  1536,
  'cosine'
);

CALL db.index.vector.createNodeIndex(
  'news_embeddings',
  'NewsSignal',
  'embedding',
  1536,
  'cosine'
);
