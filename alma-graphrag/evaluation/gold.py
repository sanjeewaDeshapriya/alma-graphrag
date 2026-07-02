"""
Rule-based gold relevance for the evaluation harness.

Ground truth is derived from objective hotel attributes via per-query predicates
(max_price, min_rating, min_star, max_travel_time, required_amenities). This is a
*bootstrap* gold standard — fully reproducible and bias-free for a pilot — and is
designed to be swappable for expert annotations later (the proposal's Phase 5).

A hotel is relevant to a query iff it satisfies ALL specified gold constraints.
Constraints that reference a missing attribute (e.g. max_price on an unpriced
hotel) make the hotel non-relevant: relevance requires positive evidence.
"""
from __future__ import annotations

from typing import Any, Dict, List, Set


def _amenity_present(hotel: Dict[str, Any], needle: str) -> bool:
    needle = needle.lower()
    return any(needle in a.lower() for a in (hotel.get("amenities") or []))


def is_relevant(hotel: Dict[str, Any], gold: Dict[str, Any]) -> bool:
    price = hotel.get("price")
    rating = hotel.get("rating")
    star = hotel.get("star")
    tt = hotel.get("travel_time_traffic_min") or hotel.get("travel_time_min")

    if "max_price" in gold:
        if price is None or float(price) > gold["max_price"]:
            return False
    if "min_price" in gold:
        if price is None or float(price) < gold["min_price"]:
            return False
    if "min_rating" in gold:
        if rating is None or float(rating) < gold["min_rating"]:
            return False
    if "min_star" in gold:
        if star is None or float(star) < gold["min_star"]:
            return False
    if "max_travel_time" in gold:
        if tt is None or float(tt) > gold["max_travel_time"]:
            return False
    if "required_amenities" in gold:
        if not all(_amenity_present(hotel, a) for a in gold["required_amenities"]):
            return False
    return True


def relevant_set(hotels: List[Dict[str, Any]], gold: Dict[str, Any]) -> Set[str]:
    return {str(h["id"]) for h in hotels if is_relevant(h, gold)}
