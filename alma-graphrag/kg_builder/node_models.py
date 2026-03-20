from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime


# -----------------------------------------
# CORE ENTITIES (Static)
# -----------------------------------------

@dataclass
class HotelNode:
    """Primary recommendation entity — anchor of the entire graph."""
    id: str
    name: str
    description: str
    rating: float
    price_range: str
    price_per_night_lkr: float
    address: str
    city_name: str
    lat: float
    lng: float
    phone: Optional[str] = None
    website: Optional[str] = None
    source: str = "scraped"
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    embedding: Optional[List[float]] = None

    def to_cypher_props(self) -> dict:
        return {
            k: v
            for k, v in self.__dict__.items()
            if v is not None and k != "embedding"
        }


@dataclass
class CityNode:
    """Geographic target city — e.g. Piliyandala, Maharagama."""
    name: str
    district: str
    province: str
    lat: float
    lng: float


@dataclass
class DistrictNode:
    """Administrative grouping — e.g. Colombo District."""
    name: str
    province: str


@dataclass
class AmenityNode:
    """Hotel facility or feature."""
    name: str
    category: str


@dataclass
class LocationNode:
    """Nearby landmark, attraction, or transport point."""
    name: str
    type: str
    lat: float
    lng: float
    description: Optional[str] = None


@dataclass
class RoadSegmentNode:
    """Access and mobility context — key for disruption-aware routing."""
    name: str
    road_type: str
    surface: str
    condition: str
    avg_travel_time_min: float
    last_checked: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class TransportModeNode:
    """Travel mode available near a hotel."""
    name: str
    cost_factor: float
    speed_factor: float
    reliability: str


# -----------------------------------------
# SIGNAL ENTITIES (Live / Dynamic)
# -----------------------------------------

@dataclass
class EventNode:
    """Live event — concert, festival, road closure, match, ceremony."""
    id: str
    title: str
    type: str
    start_time: str
    end_time: str
    impact_radius_km: float
    severity: str
    source: str
    lat: Optional[float] = None
    lng: Optional[float] = None
    description: Optional[str] = None
    embedding: Optional[List[float]] = None


@dataclass
class NewsSignalNode:
    """Scraped news article providing evidence for events/disruptions."""
    id: str
    title: str
    summary: str
    published_at: str
    url: str
    source: str
    credibility_score: float = 0.8
    embedding: Optional[List[float]] = None


@dataclass
class TrafficSignalNode:
    """Live congestion data from Google Maps or OpenStreetMap."""
    id: str
    timestamp: str
    location_name: str
    severity: str
    eta_change_min: float
    lat: float
    lng: float


@dataclass
class WeatherSignalNode:
    """Weather disruption — flooding, heavy rain, storm."""
    id: str
    timestamp: str
    condition: str
    impact_level: str
    area_name: str


# -----------------------------------------
# CONTEXT ENTITIES (Semantic / Preference)
# -----------------------------------------

@dataclass
class UserProfileNode:
    """User preference model — drives CRAG personalization."""
    id: str
    preferences: List[str]
    budget: str
    noise_tolerance: str
    mobility_needs: str
    travel_purpose: str


@dataclass
class EmbeddingChunkNode:
    """Text chunk from hotel/news/event — used by vector retriever."""
    id: str
    text: str
    source_id: str
    source_type: str
    chunk_index: int
    embedding: Optional[List[float]] = None
