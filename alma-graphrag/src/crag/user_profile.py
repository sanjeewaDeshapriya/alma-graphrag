"""
User profiles and active-event modelling for personalised recommendation (P3).

This is the *deterministic* form of the proposal's disruption-aware
personalisation (the presentation's "same live event -> opposite
recommendations" demo) — no reinforcement learning. A UserProfile supplies a
fixed set of scoring weights plus an event preference; the weighted retriever
(src/graph/retriever.py) consumes them. RL re-ranking is left as future work; it
would *learn* these weights instead of using presets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from src.graph.retriever import ScoringWeights


@dataclass
class ActiveEvent:
    """A live event with a geographic impact zone (e.g. an F1 race, concert)."""
    name: str
    lat: float
    lng: float
    impact_radius_km: float = 3.0
    severity: str = "high"  # high | medium | low — scales the in-zone disruption

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "lat": self.lat,
            "lng": self.lng,
            "impact_radius_km": self.impact_radius_km,
            "severity": self.severity,
        }


@dataclass
class UserProfile:
    """A traveller persona that personalises ranking via fixed weights."""
    id: str
    name: str
    weights: ScoringWeights
    event_preference: str = "neutral"     # seek | avoid | neutral
    proximity_preference: str = "any"     # close | far | any (applied to intent)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "weights": self.weights.to_dict(),
            "event_preference": self.event_preference,
            "proximity_preference": self.proximity_preference,
            "description": self.description,
        }


# Preset personas. Weights are intentionally contrasting so personalisation is
# visible; they are normalised at scoring time.
PRESETS: Dict[str, UserProfile] = {
    "event_seeker": UserProfile(
        id="event_seeker",
        name="Event Seeker (e.g. F1 Fan)",
        weights=ScoringWeights(spatial=0.30, accessibility=0.15, facility=0.30, economic=0.10, disruption=0.15),
        event_preference="seek",
        proximity_preference="close",
        description="Wants to be in the action — close to the event zone, walkable, "
                    "tolerant of crowds and traffic.",
    ),
    "quiet_seeker": UserProfile(
        id="quiet_seeker",
        name="Quiet Seeker",
        weights=ScoringWeights(spatial=0.10, accessibility=0.25, facility=0.20, economic=0.10, disruption=0.35),
        event_preference="avoid",
        proximity_preference="far",
        description="Wants calm — away from the event impact zone, low noise/traffic, "
                    "stable ETA.",
    ),
    "budget_traveler": UserProfile(
        id="budget_traveler",
        name="Budget Traveller",
        weights=ScoringWeights(spatial=0.15, accessibility=0.15, facility=0.15, economic=0.45, disruption=0.10),
        event_preference="neutral",
        proximity_preference="any",
        description="Price-first; maximises value for money.",
    ),
    "luxury_traveler": UserProfile(
        id="luxury_traveler",
        name="Luxury Traveller",
        weights=ScoringWeights(spatial=0.20, accessibility=0.15, facility=0.45, economic=0.05, disruption=0.15),
        event_preference="neutral",
        proximity_preference="any",
        description="Quality-first; prioritises ratings, stars and amenities.",
    ),
}


def get_profile(profile_id: Optional[str]) -> Optional[UserProfile]:
    if not profile_id:
        return None
    return PRESETS.get(profile_id.lower())


def list_profiles() -> list[dict]:
    return [p.to_dict() for p in PRESETS.values()]
