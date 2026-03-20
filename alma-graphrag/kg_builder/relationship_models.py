from dataclasses import dataclass
from typing import Optional


# -----------------------------------------
# STRUCTURAL RELATIONSHIPS
# -----------------------------------------

@dataclass
class LocatedIn:
    """Hotel -> City: hotel belongs to this city."""
    hotel_id: str
    city_name: str
    distance_from_center_km: float
    confidence: float = 1.0


@dataclass
class InDistrict:
    """City -> District: city is inside this district."""
    city_name: str
    district_name: str


@dataclass
class HasAmenity:
    """Hotel -> Amenity: hotel provides this amenity."""
    hotel_id: str
    amenity_name: str
    available: bool = True
    confidence: float = 1.0


@dataclass
class Near:
    """Hotel -> Location: hotel is near this place."""
    hotel_id: str
    location_name: str
    distance_km: float
    walk_time_min: Optional[float] = None
    transport_cost_lkr: Optional[float] = None


@dataclass
class AccessibleVia:
    """Hotel -> RoadSegment: access route to the hotel."""
    hotel_id: str
    road_name: str
    travel_time_min: float
    road_quality: str


@dataclass
class ServicedBy:
    """Hotel -> TransportMode: transport option available near hotel."""
    hotel_id: str
    transport_name: str
    cost_estimate_lkr: float
    reliability: str


# -----------------------------------------
# LIVE SIGNAL RELATIONSHIPS
# -----------------------------------------

@dataclass
class AffectedBy:
    """Hotel -> Event: hotel is impacted by a live event."""
    hotel_id: str
    event_id: str
    impact_score: float
    impact_type: str


@dataclass
class MentionedIn:
    """Event/Hotel -> NewsSignal: source evidence link."""
    source_id: str
    source_type: str
    news_id: str
    extract_confidence: float = 0.8


@dataclass
class HasSignal:
    """Hotel -> TrafficSignal / WeatherSignal: live operational signal."""
    hotel_id: str
    signal_id: str
    signal_type: str
    timestamp: str
    severity: str


# -----------------------------------------
# SEMANTIC RELATIONSHIPS
# -----------------------------------------

@dataclass
class SimilarTo:
    """Hotel -> Hotel: semantically similar — computed from embeddings."""
    hotel_id_a: str
    hotel_id_b: str
    similarity_score: float


@dataclass
class MatchesPreference:
    """Hotel -> UserProfile: hotel fits user preferences."""
    hotel_id: str
    user_id: str
    score: float
    reason: str


@dataclass
class DerivedFrom:
    """EmbeddingChunk -> NewsSignal/Hotel: chunk came from this source."""
    chunk_id: str
    source_id: str
    source_type: str


@dataclass
class HasDescriptionChunk:
    """Hotel -> EmbeddingChunk: text pieces from this hotel for retrieval."""
    hotel_id: str
    chunk_id: str
    source_type: str = "description"
