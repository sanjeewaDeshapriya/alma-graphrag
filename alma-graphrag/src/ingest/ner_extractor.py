from __future__ import annotations

import re
import logging
from typing import Any, Dict, List

logger = logging.getLogger("alma.ner_extractor")

# Maps lowercase keyword → canonical AttractionType label
_ATTRACTION_KEYWORDS: Dict[str, str] = {
    "beach": "Beach",
    "lake": "Lake",
    "river": "River",
    "park": "Park",
    "temple": "Temple",
    "church": "Church",
    "mosque": "Mosque",
    "museum": "Museum",
    "mall": "Shopping Mall",
    "market": "Market",
    "airport": "Airport",
    "railway station": "Train Station",
    "train station": "Train Station",
    "bus station": "Bus Station",
    "bus stand": "Bus Station",
    "hospital": "Hospital",
    "university": "University",
    "stadium": "Stadium",
    "fort": "Fort",
    "palace": "Palace",
    "shrine": "Shrine",
    "harbour": "Harbour",
    "harbor": "Harbour",
    "port": "Port",
    "waterfall": "Waterfall",
    "mountain": "Mountain",
    "hill": "Hill",
    "zoo": "Zoo",
    "resort": "Resort Area",
    "spa": "Spa",
    "casino": "Casino",
    "golf": "Golf Course",
    "botanical": "Botanical Garden",
}

# Patterns that pull a named location from descriptive text
_NEAR_PATTERNS = [
    # "near the Galle Fort", "close to Gangaramaya Temple"
    r"(?:near|close\s+to|next\s+to|adjacent\s+to|opposite)\s+(?:the\s+)?([A-Z][A-Za-z\s]{2,35}?)(?=[,.\n]|$)",
    # "located in the Pettah district / area / zone"
    r"(?:located\s+in|situated\s+in|in)\s+(?:the\s+)?([A-Z][A-Za-z\s]{2,30}?)\s+(?:district|area|zone|neighbourhood|neighborhood|region|quarter)",
    # "walking distance from X"
    r"walking\s+distance\s+from\s+(?:the\s+)?([A-Z][A-Za-z\s]{2,35}?)(?=[,.\n]|$)",
    # "minutes from X"
    r"\d+\s+minutes?\s+(?:from|to)\s+(?:the\s+)?([A-Z][A-Za-z\s]{2,35}?)(?=[,.\n]|$)",
]

# Country/region names to skip when splitting addresses into neighborhoods
_SKIP_TERMS = {
    "sri lanka", "india", "thailand", "malaysia", "singapore",
    "maldives", "indonesia", "vietnam", "cambodia", "nepal",
}


class NERExtractor:
    """
    Lightweight, dependency-free NER for hotel data.
    Uses regex patterns + keyword matching to extract:
      - neighborhoods / districts (from address)
      - nearby landmarks (from description text)
      - attraction type tags (keyword scan of description)
    """

    def extract(self, hotel: Dict[str, Any]) -> Dict[str, Any]:
        text = " ".join(filter(None, [
            hotel.get("name", ""),
            hotel.get("description", ""),
        ]))
        return {
            "neighborhoods": self._neighborhoods(hotel),
            "landmarks": self._landmarks(text),
            "attraction_types": self._attraction_types(text),
        }

    # ------------------------------------------------------------------ #
    # Neighborhoods / districts from address
    # ------------------------------------------------------------------ #
    def _neighborhoods(self, hotel: Dict[str, Any]) -> List[Dict[str, str]]:
        address = hotel.get("address", "")
        city = (hotel.get("city_name") or "").lower().strip()
        if not address:
            return []

        results: List[Dict[str, str]] = []
        seen: set = set()
        parts = [p.strip() for p in re.split(r"[,/|]", address) if p.strip()]

        for part in parts:
            # Drop pure numbers, postcodes, and overly short strings
            if re.fullmatch(r"[\d\s\-]+", part) or len(part) < 4:
                continue
            lower = part.lower()
            if lower == city or lower in _SKIP_TERMS or lower in seen:
                continue
            # Drop if it's just a street abbreviation like "Rd", "Ave", "St"
            if re.fullmatch(r"[A-Za-z]{1,3}\.?", part):
                continue
            seen.add(lower)
            results.append({"name": part, "type": "district"})
            if len(results) >= 2:
                break

        return results

    # ------------------------------------------------------------------ #
    # Nearby landmarks from description text
    # ------------------------------------------------------------------ #
    def _landmarks(self, text: str) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        seen: set = set()

        for pattern in _NEAR_PATTERNS:
            for m in re.finditer(pattern, text):
                name = m.group(1).strip().rstrip(".,;")
                # Skip noise: very short, all-lower, or generic words
                if len(name) < 4 or name.lower() in seen:
                    continue
                if not re.search(r"[A-Z]", name):
                    continue
                seen.add(name.lower())
                results.append({"name": name, "type": "landmark"})
                if len(results) >= 6:
                    return results

        return results

    # ------------------------------------------------------------------ #
    # Attraction type tags from keyword scan
    # ------------------------------------------------------------------ #
    def _attraction_types(self, text: str) -> List[str]:
        text_lower = text.lower()
        found: List[str] = []
        for keyword, label in _ATTRACTION_KEYWORDS.items():
            if keyword in text_lower and label not in found:
                found.append(label)
        return found
