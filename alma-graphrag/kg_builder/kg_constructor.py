import os
from typing import List
from dotenv import load_dotenv
from neo4j import GraphDatabase

from kg_builder.node_models import (
    HotelNode, CityNode, DistrictNode, AmenityNode,
    LocationNode, RoadSegmentNode, EventNode,
    NewsSignalNode, TrafficSignalNode, WeatherSignalNode,
    EmbeddingChunkNode, UserProfileNode
)
from kg_builder.relationship_models import (
    LocatedIn, InDistrict, HasAmenity, Near,
    AccessibleVia, AffectedBy, MentionedIn,
    HasSignal, SimilarTo, HasDescriptionChunk, DerivedFrom
)

load_dotenv()


class KGConstructor:
    """
    Core ALMA-GraphRAG knowledge graph builder.
    Handles all node creation and relationship wiring for Phase 1.
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "alma_password123"),
            ),
        )

    def close(self):
        self.driver.close()

    # -----------------------------------------
    # NODE CREATION
    # -----------------------------------------

    def create_hotel(self, hotel: HotelNode):
        with self.driver.session() as s:
            s.run(
                """
                MERGE (h:Hotel {id: $id})
                SET h.name              = $name,
                    h.description       = $description,
                    h.rating            = $rating,
                    h.price_range       = $price_range,
                    h.price_per_night_lkr = $price_per_night_lkr,
                    h.address           = $address,
                    h.city_name         = $city_name,
                    h.lat               = $lat,
                    h.lng               = $lng,
                    h.phone             = $phone,
                    h.website           = $website,
                    h.source            = $source,
                    h.last_updated      = $last_updated
                """,
                hotel.to_cypher_props(),
            )

    def create_city(self, city: CityNode):
        with self.driver.session() as s:
            s.run(
                """
                MERGE (c:City {name: $name})
                SET c.district = $district,
                    c.province = $province,
                    c.lat      = $lat,
                    c.lng      = $lng
                """,
                city.__dict__,
            )

    def create_district(self, district: DistrictNode):
        with self.driver.session() as s:
            s.run(
                """
                MERGE (d:District {name: $name})
                SET d.province = $province
                """,
                district.__dict__,
            )

    def create_amenity(self, amenity: AmenityNode):
        with self.driver.session() as s:
            s.run(
                """
                MERGE (a:Amenity {name: $name})
                SET a.category = $category
                """,
                amenity.__dict__,
            )

    def create_location(self, location: LocationNode):
        with self.driver.session() as s:
            s.run(
                """
                MERGE (l:Location {name: $name})
                SET l.type        = $type,
                    l.lat         = $lat,
                    l.lng         = $lng,
                    l.description = $description
                """,
                location.__dict__,
            )

    def create_event(self, event: EventNode):
        with self.driver.session() as s:
            s.run(
                """
                MERGE (e:Event {id: $id})
                SET e.title             = $title,
                    e.type              = $type,
                    e.start_time        = $start_time,
                    e.end_time          = $end_time,
                    e.impact_radius_km  = $impact_radius_km,
                    e.severity          = $severity,
                    e.source            = $source,
                    e.lat               = $lat,
                    e.lng               = $lng,
                    e.description       = $description
                """,
                {k: v for k, v in event.__dict__.items() if k != "embedding"},
            )

    def create_news_signal(self, news: NewsSignalNode):
        with self.driver.session() as s:
            s.run(
                """
                MERGE (n:NewsSignal {url: $url})
                SET n.id                = $id,
                    n.title             = $title,
                    n.summary           = $summary,
                    n.published_at      = $published_at,
                    n.source            = $source,
                    n.credibility_score = $credibility_score
                """,
                {k: v for k, v in news.__dict__.items() if k != "embedding"},
            )

    def create_traffic_signal(self, signal: TrafficSignalNode):
        with self.driver.session() as s:
            s.run(
                """
                MERGE (t:TrafficSignal {id: $id})
                SET t.timestamp       = $timestamp,
                    t.location_name   = $location_name,
                    t.severity        = $severity,
                    t.eta_change_min  = $eta_change_min,
                    t.lat             = $lat,
                    t.lng             = $lng
                """,
                signal.__dict__,
            )

    def create_embedding_chunk(self, chunk: EmbeddingChunkNode):
        with self.driver.session() as s:
            s.run(
                """
                MERGE (c:EmbeddingChunk {id: $id})
                SET c.text        = $text,
                    c.source_id   = $source_id,
                    c.source_type = $source_type,
                    c.chunk_index = $chunk_index
                """,
                {k: v for k, v in chunk.__dict__.items() if k != "embedding"},
            )

    # -----------------------------------------
    # RELATIONSHIP CREATION
    # -----------------------------------------

    def link_hotel_to_city(self, rel: LocatedIn):
        with self.driver.session() as s:
            s.run(
                """
                MATCH (h:Hotel {id: $hotel_id})
                MATCH (c:City  {name: $city_name})
                MERGE (h)-[r:LOCATED_IN]->(c)
                SET r.distance_from_center_km = $distance_from_center_km,
                    r.confidence              = $confidence
                """,
                rel.__dict__,
            )

    def link_city_to_district(self, rel: InDistrict):
        with self.driver.session() as s:
            s.run(
                """
                MATCH (c:City     {name: $city_name})
                MATCH (d:District {name: $district_name})
                MERGE (c)-[:IN_DISTRICT]->(d)
                """,
                rel.__dict__,
            )

    def link_hotel_amenity(self, rel: HasAmenity):
        with self.driver.session() as s:
            s.run(
                """
                MATCH (h:Hotel   {id: $hotel_id})
                MATCH (a:Amenity {name: $amenity_name})
                MERGE (h)-[r:HAS_AMENITY]->(a)
                SET r.available  = $available,
                    r.confidence = $confidence
                """,
                rel.__dict__,
            )

    def link_hotel_location(self, rel: Near):
        with self.driver.session() as s:
            s.run(
                """
                MATCH (h:Hotel    {id: $hotel_id})
                MATCH (l:Location {name: $location_name})
                MERGE (h)-[r:NEAR]->(l)
                SET r.distance_km         = $distance_km,
                    r.walk_time_min       = $walk_time_min,
                    r.transport_cost_lkr  = $transport_cost_lkr
                """,
                rel.__dict__,
            )

    def link_hotel_road(self, rel: AccessibleVia):
        with self.driver.session() as s:
            s.run(
                """
                MATCH (h:Hotel       {id: $hotel_id})
                MATCH (r:RoadSegment {name: $road_name})
                MERGE (h)-[rel:ACCESSIBLE_VIA]->(r)
                SET rel.travel_time_min = $travel_time_min,
                    rel.road_quality    = $road_quality
                """,
                rel.__dict__,
            )

    def link_hotel_event(self, rel: AffectedBy):
        with self.driver.session() as s:
            s.run(
                """
                MATCH (h:Hotel {id: $hotel_id})
                MATCH (e:Event {id: $event_id})
                MERGE (h)-[r:AFFECTED_BY]->(e)
                SET r.impact_score = $impact_score,
                    r.impact_type  = $impact_type
                """,
                rel.__dict__,
            )

    def link_event_news(self, rel: MentionedIn):
        with self.driver.session() as s:
            s.run(
                """
                MATCH (e:Event     {id:  $source_id})
                MATCH (n:NewsSignal {url: $news_id})
                MERGE (e)-[r:MENTIONED_IN]->(n)
                SET r.extract_confidence = $extract_confidence
                """,
                rel.__dict__,
            )

    def link_hotel_signal(self, rel: HasSignal):
        with self.driver.session() as s:
            label = "TrafficSignal" if rel.signal_type == "traffic" else "WeatherSignal"
            s.run(
                f"""
                MATCH (h:Hotel {{id: $hotel_id}})
                MATCH (sig:{label} {{id: $signal_id}})
                MERGE (h)-[r:HAS_SIGNAL]->(sig)
                SET r.timestamp = $timestamp,
                    r.severity  = $severity
                """,
                rel.__dict__,
            )

    def link_similar_hotels(self, rel: SimilarTo):
        with self.driver.session() as s:
            s.run(
                """
                MATCH (a:Hotel {id: $hotel_id_a})
                MATCH (b:Hotel {id: $hotel_id_b})
                MERGE (a)-[r:SIMILAR_TO]->(b)
                SET r.similarity_score = $similarity_score
                """,
                rel.__dict__,
            )

    def link_hotel_chunk(self, rel: HasDescriptionChunk):
        with self.driver.session() as s:
            s.run(
                """
                MATCH (h:Hotel          {id: $hotel_id})
                MATCH (c:EmbeddingChunk {id: $chunk_id})
                MERGE (h)-[r:HAS_DESCRIPTION_CHUNK]->(c)
                SET r.source_type = $source_type
                """,
                rel.__dict__,
            )

    # -----------------------------------------
    # BATCH LOADER — used by scraper pipeline
    # -----------------------------------------

    def build_hotel_subgraph(
        self,
        hotel: HotelNode,
        amenities: List[str],
        nearby_locations: List[dict],
        city: CityNode,
        district: DistrictNode,
    ):
        """
        Full hotel subgraph in one call.
        Creates: Hotel + City + District + Amenities + Locations
        Wires:   LOCATED_IN + IN_DISTRICT + HAS_AMENITY + NEAR
        """
        self.create_district(district)
        self.create_city(city)
        self.create_hotel(hotel)

        self.link_city_to_district(
            InDistrict(city_name=city.name, district_name=district.name)
        )
        self.link_hotel_to_city(
            LocatedIn(
                hotel_id=hotel.id,
                city_name=city.name,
                distance_from_center_km=0.5,
            )
        )

        for amenity_name in amenities:
            self.create_amenity(AmenityNode(name=amenity_name, category="General"))
            self.link_hotel_amenity(
                HasAmenity(hotel_id=hotel.id, amenity_name=amenity_name)
            )

        for loc in nearby_locations:
            self.create_location(
                LocationNode(
                    name=loc["name"],
                    type=loc.get("type", "landmark"),
                    lat=loc.get("lat", 0.0),
                    lng=loc.get("lng", 0.0),
                )
            )
            self.link_hotel_location(
                Near(
                    hotel_id=hotel.id,
                    location_name=loc["name"],
                    distance_km=loc.get("distance_km", 1.0),
                    walk_time_min=loc.get("walk_time_min"),
                )
            )
