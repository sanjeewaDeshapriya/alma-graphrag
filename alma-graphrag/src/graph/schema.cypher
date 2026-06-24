// =============================================
// ALMA-GraphRAG Phase 1 Schema (Graph-only)
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

CREATE CONSTRAINT roomtype_id IF NOT EXISTS
  FOR (rt:RoomType) REQUIRE rt.id IS UNIQUE;

CREATE CONSTRAINT boardtype_name IF NOT EXISTS
  FOR (b:BoardType) REQUIRE b.name IS UNIQUE;

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

CREATE CONSTRAINT traffic_signal_id IF NOT EXISTS
  FOR (t:TrafficSignal) REQUIRE t.id IS UNIQUE;

// --- STANDARD INDEXES ---
CREATE INDEX hotel_name IF NOT EXISTS
  FOR (h:Hotel) ON (h.name);

CREATE INDEX hotel_city IF NOT EXISTS
  FOR (h:Hotel) ON (h.city_name);

CREATE INDEX hotel_source IF NOT EXISTS
  FOR (h:Hotel) ON (h.source);

CREATE INDEX event_type IF NOT EXISTS
  FOR (e:Event) ON (e.type);

CREATE INDEX news_published IF NOT EXISTS
  FOR (n:NewsSignal) ON (n.published_at);

CREATE INDEX traffic_signal_timestamp IF NOT EXISTS
  FOR (t:TrafficSignal) ON (t.timestamp);

CREATE INDEX traffic_signal_severity IF NOT EXISTS
  FOR (t:TrafficSignal) ON (t.severity);

// --- FULLTEXT INDEXES ---
CREATE FULLTEXT INDEX hotel_fulltext IF NOT EXISTS
  FOR (h:Hotel) ON EACH [h.name, h.description, h.address];

CREATE FULLTEXT INDEX news_fulltext IF NOT EXISTS
  FOR (n:NewsSignal) ON EACH [n.title, n.summary];
