from __future__ import annotations

import time
import uuid
from typing import Dict, List
import httpx
from datetime import datetime

from src.config import (
    GOOGLE_MAPS_API_KEY,
    GOOGLE_PLACES_BASE_URL,
    DEFAULT_COUNTRY,
    HOTEL_MAX_RESULTS,
)


class GooglePlacesClient:
    def __init__(self) -> None:
        if not GOOGLE_MAPS_API_KEY:
            raise ValueError("GOOGLE_MAPS_API_KEY is not set")
        self.client = httpx.Client(timeout=30)

    def close(self) -> None:
        self.client.close()

    def search_hotels(self, city: str, max_results: int) -> List[Dict]:
        results: List[Dict] = []
        params = {
            "query": f"hotels in {city} {DEFAULT_COUNTRY}",
            "type": "lodging",
            "key": GOOGLE_MAPS_API_KEY,
        }
        while len(results) < max_results:
            resp = self.client.get(
                f"{GOOGLE_PLACES_BASE_URL}/textsearch/json", params=params
            )
            data = resp.json()
            status = data.get("status")
            if status not in ("OK", "ZERO_RESULTS"):
                break

            results.extend(data.get("results", []))
            next_token = data.get("next_page_token")
            if not next_token or len(results) >= max_results:
                break
            time.sleep(2)
            params = {"pagetoken": next_token, "key": GOOGLE_MAPS_API_KEY}

        return results[:max_results]

    def place_details(self, place_id: str) -> Dict:
        params = {
            "place_id": place_id,
            "fields": ",".join(
                [
                    "place_id",
                    "name",
                    "formatted_address",
                    "formatted_phone_number",
                    "website",
                    "rating",
                    "user_ratings_total",
                    "price_level",
                    "geometry",
                    "types",
                    "editorial_summary",
                    "reviews",
                ]
            ),
            "key": GOOGLE_MAPS_API_KEY,
        }
        resp = self.client.get(
            f"{GOOGLE_PLACES_BASE_URL}/details/json", params=params
        )
        return resp.json().get("result", {})

    def normalize_hotel(self, raw: Dict, city: str) -> Dict:
        price_map = {0: "Budget", 1: "Budget", 2: "Mid-Range", 3: "Luxury", 4: "Luxury"}
        price_lkr_map = {0: 2000, 1: 3500, 2: 9000, 3: 20000, 4: 40000}
        price_level = raw.get("price_level", 1)

        summary = raw.get("editorial_summary", {}).get("overview", "")
        reviews = raw.get("reviews", [])
        top_review = reviews[0].get("text", "") if reviews else ""
        description = f"{summary} {top_review}".strip() or f"Hotel in {city}."

        geo = raw.get("geometry", {}).get("location", {})

        return {
            "id": raw.get("place_id", str(uuid.uuid4())),
            "name": raw.get("name", "Unknown Hotel"),
            "description": description[:1000],
            "rating": float(raw.get("rating", 0.0)),
            "price_range": price_map.get(price_level, "Mid-Range"),
            "price_per_night_lkr": price_lkr_map.get(price_level, 8000),
            "address": raw.get("formatted_address", ""),
            "city_name": city,
            "lat": float(geo.get("lat", 0.0)),
            "lng": float(geo.get("lng", 0.0)),
            "phone": raw.get("formatted_phone_number"),
            "website": raw.get("website"),
            "source": "google_places",
            "last_updated": datetime.utcnow().isoformat(),
            "amenities": self._extract_amenities(raw),
        }

    def _extract_amenities(self, raw: Dict) -> List[str]:
        types = raw.get("types", [])
        amenity_map = {
            "lodging": ["24h Front Desk"],
            "restaurant": ["Restaurant"],
            "spa": ["Spa"],
            "parking": ["Free Parking"],
            "gym": ["Fitness Center"],
            "swimming_pool": ["Swimming Pool"],
            "wifi": ["Free WiFi"],
        }
        amenities = set()
        for t in types:
            amenities.update(amenity_map.get(t, []))
        amenities.update(["Air Conditioning", "Hot Water", "24h Front Desk"])
        return list(amenities)

    def scrape_city(self, city: str, max_results: int = HOTEL_MAX_RESULTS) -> List[Dict]:
        basic = self.search_hotels(city, max_results=max_results)
        hotels: List[Dict] = []
        for item in basic:
            place_id = item.get("place_id")
            if not place_id:
                continue
            details = self.place_details(place_id)
            hotels.append(self.normalize_hotel(details, city))
            time.sleep(0.3)
        return hotels
