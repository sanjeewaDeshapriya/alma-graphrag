"""
Natural-language → structured query intent (safe NL→Cypher translation).

Instead of asking an LLM to emit free-form Cypher (unsafe, non-reproducible,
injection-prone), we translate the question into a *constrained* QueryIntent
schema. A parameterised Cypher template in ``src/graph/retriever.py`` then turns
that intent into an actual graph traversal. This is the production-correct form
of "NL→Cypher" used by text2cypher systems and keeps the graph layer safe.

Pipeline:
    question --(regex slots)--> base intent
             --(LLM slot-fill)--> enriched intent (proximity, sort, priorities)
             --> merged QueryIntent

The regex pass is deterministic and always runs (fallback if no LLM key); the
LLM pass fills higher-level intent slots the regex can't infer reliably.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from openai import OpenAI

from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

logger = logging.getLogger("alma.crag.query_parser")

_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL) if LLM_API_KEY else None


# ---------------------------------------------------------------------------
# Intent schema
# ---------------------------------------------------------------------------

@dataclass
class QueryIntent:
    """Structured representation of a hotel-search question."""

    city: Optional[str] = None
    required_amenities: List[str] = field(default_factory=list)
    near_attractions: List[str] = field(default_factory=list)
    max_price_lkr: Optional[float] = None
    min_price_lkr: Optional[float] = None
    min_rating: Optional[float] = None
    min_star: Optional[int] = None
    board_preference: Optional[str] = None

    # Higher-level intent (drives dynamic scoring weights)
    proximity_preference: str = "any"      # close | far | any
    accessibility_priority: str = "normal"  # high | normal
    avoid_traffic: bool = False
    sort_intent: str = "best_overall"       # best_overall | cheapest | highest_rated | most_accessible

    raw_keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Deterministic regex extraction
# ---------------------------------------------------------------------------

# Canonical amenity → mention synonyms (lowercase). Matching is substring-based.
_AMENITY_SYNONYMS: Dict[str, List[str]] = {
    "pool": ["pool", "swimming"],
    "wifi": ["wifi", "wi-fi", "internet"],
    "spa": ["spa", "wellness"],
    "gym": ["gym", "fitness"],
    "parking": ["parking", "car park"],
    "restaurant": ["restaurant", "dining"],
    "bar": ["bar", "lounge"],
    "air conditioning": ["air conditioning", "ac ", "a/c", "air-conditioned"],
    "beach access": ["beachfront", "beach access", "private beach"],
    "pet friendly": ["pet friendly", "pet-friendly", "pets allowed"],
    "airport shuttle": ["airport shuttle", "shuttle"],
    "breakfast": ["breakfast included", "free breakfast"],
}

# Attraction keywords (align with NERExtractor attraction labels, lowercase).
_ATTRACTION_KEYWORDS = [
    "beach", "lake", "river", "park", "temple", "church", "mosque", "museum",
    "mall", "market", "airport", "train station", "railway station",
    "bus station", "hospital", "university", "stadium", "fort", "palace",
    "waterfall", "mountain", "hill", "zoo", "spa", "casino", "golf",
    "botanical", "harbour", "port", "city center", "city centre",
]

_QUIET_TERMS = ["quiet", "calm", "peaceful", "secluded", "tranquil", "relaxing", "away from"]
_CLOSE_TERMS = ["walkable", "walking distance", "close to", "near the", "next to", "in the heart"]
_TRAFFIC_TERMS = ["avoid traffic", "low traffic", "no traffic", "easy access", "good road", "stable eta", "quick access", "fast access"]
_ACCESS_TERMS = ["accessible", "accessibility", "easy to reach", "good access", "transport"]

_STOPWORDS = {
    "the", "a", "an", "in", "near", "with", "and", "or", "for", "hotels",
    "hotel", "good", "best", "top", "show", "find", "me", "that", "have",
    "has", "are", "is", "to", "of", "which", "what", "where", "i", "want",
    "looking", "recommend", "please", "some", "any", "around",
}


def _extract_price(text: str) -> Dict[str, Optional[float]]:
    """Pull budget bounds from phrases like 'under 20000 LKR', 'below Rs 15,000'."""
    out: Dict[str, Optional[float]] = {"max_price_lkr": None, "min_price_lkr": None}
    t = text.lower().replace(",", "")

    # "under/below/less than/max X"
    m = re.search(r"(?:under|below|less than|cheaper than|max|up to|within)\s*(?:rs\.?|lkr|\$)?\s*(\d{3,7})", t)
    if m:
        out["max_price_lkr"] = float(m.group(1))

    # "over/above/more than/min X"
    m = re.search(r"(?:over|above|more than|at least|min|minimum)\s*(?:rs\.?|lkr|\$)?\s*(\d{3,7})", t)
    if m:
        out["min_price_lkr"] = float(m.group(1))

    # "between X and Y"
    m = re.search(r"between\s*(?:rs\.?|lkr|\$)?\s*(\d{3,7})\s*(?:and|-|to)\s*(?:rs\.?|lkr|\$)?\s*(\d{3,7})", t)
    if m:
        lo, hi = sorted([float(m.group(1)), float(m.group(2))])
        out["min_price_lkr"], out["max_price_lkr"] = lo, hi

    return out


def _extract_rating(text: str) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {"min_rating": None, "min_star": None}
    t = text.lower()

    m = re.search(r"(\d(?:\.\d)?)\s*\+?\s*(?:star|stars)", t)
    if m:
        out["min_star"] = int(float(m.group(1)))

    m = re.search(r"rating\s*(?:above|over|of|>=?)?\s*(\d(?:\.\d)?)", t)
    if m:
        out["min_rating"] = float(m.group(1))

    if any(p in t for p in ["top-rated", "top rated", "highly rated", "best rated", "highest rated"]):
        out["min_rating"] = out["min_rating"] or 4.0

    return out


def _regex_intent(question: str, default_city: Optional[str]) -> QueryIntent:
    t = question.lower()
    intent = QueryIntent(city=default_city)

    # Amenities
    for canonical, syns in _AMENITY_SYNONYMS.items():
        if any(s in t for s in syns):
            intent.required_amenities.append(canonical)

    # Attractions
    for kw in _ATTRACTION_KEYWORDS:
        if kw in t:
            intent.near_attractions.append(kw)

    # Price / rating
    price = _extract_price(question)
    intent.max_price_lkr = price["max_price_lkr"]
    intent.min_price_lkr = price["min_price_lkr"]
    rating = _extract_rating(question)
    intent.min_rating = rating["min_rating"]
    intent.min_star = rating["min_star"]

    # Proximity / disruption signals
    if any(p in t for p in _QUIET_TERMS):
        intent.proximity_preference = "far"
        intent.avoid_traffic = True
    elif any(p in t for p in _CLOSE_TERMS):
        intent.proximity_preference = "close"

    if any(p in t for p in _TRAFFIC_TERMS):
        intent.avoid_traffic = True
    if any(p in t for p in _ACCESS_TERMS):
        intent.accessibility_priority = "high"

    # Sort intent
    if any(p in t for p in ["cheap", "budget", "affordable", "lowest price", "least expensive"]):
        intent.sort_intent = "cheapest"
    elif any(p in t for p in ["top-rated", "highest rated", "best rated", "luxury", "5 star", "five star"]):
        intent.sort_intent = "highest_rated"
    elif intent.accessibility_priority == "high" or intent.avoid_traffic:
        intent.sort_intent = "most_accessible"

    # Raw keywords (for fulltext-style fallback / logging)
    intent.raw_keywords = [
        w for w in re.findall(r"[a-zA-Z]{3,}", t) if w not in _STOPWORDS
    ][:12]

    return intent


# ---------------------------------------------------------------------------
# LLM slot-filling (enriches the higher-level intent slots)
# ---------------------------------------------------------------------------

_LLM_SLOT_PROMPT = """You are a query understanding module for a hotel search engine.
Extract the user's intent into STRICT JSON (no markdown, no prose). Schema:
{{
  "city": string or null,
  "required_amenities": [string],
  "near_attractions": [string],
  "max_price_lkr": number or null,
  "min_price_lkr": number or null,
  "min_rating": number or null,
  "min_star": integer or null,
  "proximity_preference": "close" | "far" | "any",
  "accessibility_priority": "high" | "normal",
  "avoid_traffic": boolean,
  "sort_intent": "best_overall" | "cheapest" | "highest_rated" | "most_accessible"
}}
Rules:
- "quiet/calm/peaceful/away from crowds" => proximity_preference="far", avoid_traffic=true.
- "walkable/close/in the heart" => proximity_preference="close".
- "easy access/good roads/avoid traffic/stable ETA" => avoid_traffic=true, accessibility_priority="high".
- Only include amenities/attractions the user actually mentions.
- Do NOT invent a city if none is named; return null.

Question: {question}
JSON:"""


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _llm_intent(question: str) -> Optional[Dict]:
    if _client is None:
        return None
    try:
        resp = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": _LLM_SLOT_PROMPT.format(question=question)}],
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        return json.loads(_strip_fences(raw))
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM slot-fill failed, using regex intent only: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def parse_query(question: str, default_city: Optional[str] = None) -> QueryIntent:
    """Translate a natural-language question into a structured QueryIntent.

    Regex extraction always runs; LLM output (if available) overrides/enriches
    the higher-level slots and the city. The merge is conservative — LLM values
    win for scalar intent slots, and list slots are unioned.
    """
    intent = _regex_intent(question, default_city)
    llm = _llm_intent(question)

    if llm:
        # City: prefer an explicit LLM-detected city, else keep default.
        if llm.get("city"):
            intent.city = str(llm["city"]).strip()

        # Union list slots (dedupe, lowercase).
        for key in ("required_amenities", "near_attractions"):
            vals = llm.get(key) or []
            if isinstance(vals, list):
                merged = {a.lower().strip() for a in intent.__dict__[key]}
                merged |= {str(v).lower().strip() for v in vals if v}
                setattr(intent, key, sorted(merged))

        # Scalar slots: LLM wins when present.
        for key in ("max_price_lkr", "min_price_lkr", "min_rating", "min_star"):
            if llm.get(key) is not None:
                setattr(intent, key, llm[key])

        if llm.get("proximity_preference") in ("close", "far", "any"):
            intent.proximity_preference = llm["proximity_preference"]
        if llm.get("accessibility_priority") in ("high", "normal"):
            intent.accessibility_priority = llm["accessibility_priority"]
        if isinstance(llm.get("avoid_traffic"), bool):
            intent.avoid_traffic = llm["avoid_traffic"]
        if llm.get("sort_intent") in ("best_overall", "cheapest", "highest_rated", "most_accessible"):
            intent.sort_intent = llm["sort_intent"]

    logger.info(
        "Parsed intent: city=%s amenities=%s attractions=%s price<=%s rating>=%s prox=%s avoid_traffic=%s sort=%s",
        intent.city, intent.required_amenities, intent.near_attractions,
        intent.max_price_lkr, intent.min_rating, intent.proximity_preference,
        intent.avoid_traffic, intent.sort_intent,
    )
    return intent
