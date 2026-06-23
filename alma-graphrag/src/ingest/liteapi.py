from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx

from src.config import (
    DEFAULT_COUNTRY_CODE,
    LITEAPI_ADULTS,
    LITEAPI_BASE_URL,
    LITEAPI_CHECKIN_OFFSET_DAYS,
    LITEAPI_CURRENCY,
    LITEAPI_GUEST_NATIONALITY,
    LITEAPI_KEY,
    LITEAPI_LOS_NIGHTS,
    LITEAPI_MAX_RATES_PER_HOTEL,
    LITEAPI_TIMEOUT,
)

logger = logging.getLogger("alma.liteapi")


class LiteApiClient:
    """
    Client for the LiteAPI `/hotels/rates` endpoint.

    Searches hotels by city/country, returning live rates plus hotel content
    (name, address, coordinates, rating, facilities), which we normalise into
    the same hotel shape used by the Google Places ingestion path so the graph
    loader can consume both sources uniformly.
    """

    def __init__(self) -> None:
        if not LITEAPI_KEY:
            raise ValueError("LITEAPI_KEY is not set")
        self.client = httpx.Client(
            timeout=LITEAPI_TIMEOUT + 10,
            headers={
                "X-API-Key": LITEAPI_KEY,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

    def close(self) -> None:
        self.client.close()

    # ------------------------------------------------------------------ #
    # Fetching
    # ------------------------------------------------------------------ #
    def search_rates(
        self,
        city: str,
        country_code: str = DEFAULT_COUNTRY_CODE,
        max_results: int = 40,
        checkin: Optional[str] = None,
        checkout: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST /hotels/rates filtered by city + country, with hotel data included."""
        checkin = checkin or self._default_checkin()
        checkout = checkout or self._default_checkout(checkin)

        payload: Dict[str, Any] = {
            "cityName": city,
            "countryCode": country_code,
            "checkin": checkin,
            "checkout": checkout,
            "currency": LITEAPI_CURRENCY,
            "guestNationality": LITEAPI_GUEST_NATIONALITY,
            "occupancies": [{"adults": LITEAPI_ADULTS}],
            "limit": max_results,
            "maxRatesPerHotel": LITEAPI_MAX_RATES_PER_HOTEL,
            "includeHotelData": True,
            "roomMapping": True,
            "timeout": LITEAPI_TIMEOUT,
        }

        url = f"{LITEAPI_BASE_URL}/hotels/rates"
        logger.info(
            "Lite API → POST %s | city=%s country=%s checkin=%s checkout=%s "
            "currency=%s guests=%d limit=%d maxRates=%d",
            url, city, country_code, checkin, checkout,
            LITEAPI_CURRENCY, LITEAPI_ADULTS, max_results, LITEAPI_MAX_RATES_PER_HOTEL,
        )

        try:
            resp = self.client.post(url, json=payload)
        except httpx.RequestError as exc:
            logger.warning("Lite API request failed for %s: %s", city, exc)
            return {"data": [], "hotels": []}

        if resp.status_code != 200:
            logger.warning(
                "Lite API ← %d FAILED for %s: %s",
                resp.status_code, city, resp.text[:300],
            )
            return {"data": [], "hotels": []}

        body = resp.json()
        print("Lite API returned result:", body)
        logger.info("Lite API returned result: %s", body)
        n_rates = len(body.get("data") or [])
        n_meta = len(body.get("hotels") or [])
        logger.info(
            "Lite API ← 200 OK | city=%s | %d rate entries, %d hotel metadata records",
            city, n_rates, n_meta,
        )
        return body

    def scrape_city(self, city: str, max_results: int = 40) -> List[Dict[str, Any]]:
        """Return a list of normalised hotel dicts for a city."""
        logger.info("Lite API: scraping city=%s max=%d", city, max_results)
        body = self.search_rates(city, max_results=max_results)
        data = body.get("data") or []
        hotels_meta = {h.get("id"): h for h in (body.get("hotels") or []) if h.get("id")}

        hotels: List[Dict[str, Any]] = []
        skipped = 0
        for entry in data:
            hotel_id = entry.get("hotelId")
            if not hotel_id:
                skipped += 1
                continue
            meta = hotels_meta.get(hotel_id, {})
            normalized = self.normalize_hotel(entry, meta, city)
            if normalized:
                hotels.append(normalized)
            else:
                skipped += 1

        logger.info(
            "Lite API: normalised %d hotels for %s (%d skipped, no hotelId or bad data)",
            len(hotels), city, skipped,
        )
        return hotels

    # ------------------------------------------------------------------ #
    # Normalisation / attribute extraction
    # ------------------------------------------------------------------ #
    def normalize_hotel(
        self, entry: Dict[str, Any], meta: Dict[str, Any], city: str
    ) -> Optional[Dict[str, Any]]:
        hotel_id = entry.get("hotelId")
        if not hotel_id:
            return None

        room_types = self._extract_room_types(entry.get("roomTypes") or [])
        nights = max(LITEAPI_LOS_NIGHTS, 1)

        # Cheapest offer total across all room types (already in LITEAPI_CURRENCY).
        offer_totals = [rt["price"] for rt in room_types if rt.get("price")]
        min_total = min(offer_totals) if offer_totals else 0.0
        price_per_night = round(min_total / nights, 2) if min_total else 0.0

        geo_lat = self._coerce_float(meta.get("latitude"))
        geo_lng = self._coerce_float(meta.get("longitude"))
        # LiteAPI review scores are on a 0-10 scale; normalise to 0-5 so they
        # combine consistently with Google Places ratings in the graph.
        raw_rating = self._coerce_float(meta.get("rating"))
        rating = round(raw_rating / 2, 1) if raw_rating > 5 else raw_rating
        star_rating = self._coerce_float(meta.get("stars") or meta.get("starRating"))

        description = (
            meta.get("hotelDescription")
            or meta.get("description")
            or f"Hotel in {city}."
        )

        board_types = sorted({rt["board_name"] for rt in room_types if rt.get("board_name")})
        refundable_available = any(rt.get("refundable") for rt in room_types)

        return {
            "id": str(hotel_id),
            "name": meta.get("name", "Unknown Hotel"),
            "description": str(description)[:1000],
            "rating": rating,
            "price_range": self._price_range(price_per_night),
            "price_per_night_lkr": price_per_night
            if LITEAPI_CURRENCY == "LKR"
            else 0.0,
            "price_currency": LITEAPI_CURRENCY,
            "price_per_night": price_per_night,
            "address": meta.get("address", ""),
            "city_name": city,
            "lat": geo_lat,
            "lng": geo_lng,
            "phone": meta.get("phone"),
            "website": meta.get("website"),
            "source": "liteapi",
            "star_rating": star_rating,
            "refundable_available": refundable_available,
            "board_types": board_types,
            "last_updated": datetime.utcnow().isoformat(),
            "amenities": self._extract_amenities(meta),
            "room_types": room_types,
        }

    def _extract_room_types(self, room_types: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flatten LiteAPI roomTypes into compact rate-plan dicts for the graph."""
        plans: List[Dict[str, Any]] = []
        for rt in room_types:
            offer_total = self._first_amount(rt.get("offerRetailRate"))
            rates = rt.get("rates") or []
            first_rate = rates[0] if rates else {}

            board_name = first_rate.get("boardName") or first_rate.get("boardType") or ""
            refundable_tag = (
                (first_rate.get("cancellationPolicies") or {}).get("refundableTag")
            )
            retail = first_rate.get("retailRate") or {}
            rate_total = self._first_amount(retail.get("total"))

            price = offer_total or rate_total
            if not price:
                continue

            plans.append(
                {
                    "room_type_id": str(rt.get("roomTypeId") or rt.get("offerId") or ""),
                    "name": (rt.get("name") or first_rate.get("name") or "Room")[:200],
                    "price": float(price),
                    "currency": LITEAPI_CURRENCY,
                    "board_name": board_name,
                    "board_type": first_rate.get("boardType") or "",
                    "refundable": refundable_tag == "RFN",
                    "rate_type": rt.get("rateType") or "standard",
                }
            )

        plans.sort(key=lambda p: p["price"])
        return plans[:LITEAPI_MAX_RATES_PER_HOTEL]

    def _extract_amenities(self, meta: Dict[str, Any]) -> List[str]:
        """Pull amenity/facility names from hotel metadata (best-effort)."""
        amenities: set[str] = set()
        candidates = (
            meta.get("amenities")
            or meta.get("hotelFacilities")
            or meta.get("facilities")
            or []
        )
        for item in candidates:
            if isinstance(item, str):
                amenities.add(item.strip())
            elif isinstance(item, dict):
                name = item.get("name") or item.get("facility")
                if name:
                    amenities.add(str(name).strip())
        amenities.update(["Air Conditioning", "24h Front Desk"])
        return [a for a in amenities if a]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _first_amount(price_array: Any) -> float:
        if isinstance(price_array, list) and price_array:
            amount = price_array[0].get("amount")
            try:
                return float(amount)
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    @staticmethod
    def _coerce_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _price_range(price_per_night_lkr: float) -> str:
        if price_per_night_lkr <= 0:
            return "Mid-Range"
        if price_per_night_lkr < 6000:
            return "Budget"
        if price_per_night_lkr < 18000:
            return "Mid-Range"
        return "Luxury"

    @staticmethod
    def _default_checkin() -> str:
        return (datetime.utcnow() + timedelta(days=LITEAPI_CHECKIN_OFFSET_DAYS)).strftime(
            "%Y-%m-%d"
        )

    @staticmethod
    def _default_checkout(checkin: str) -> str:
        dt = datetime.strptime(checkin, "%Y-%m-%d") + timedelta(
            days=max(LITEAPI_LOS_NIGHTS, 1)
        )
        return dt.strftime("%Y-%m-%d")
