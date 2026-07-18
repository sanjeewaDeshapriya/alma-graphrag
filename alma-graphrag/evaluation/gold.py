"""
Graded gold relevance for the evaluation harness.

Replaces the earlier hard pass/fail gold — which mirrored the FilterBaseline's own
decision rule and made the comparison circular — with GRADED relevance on a 0/1/2
scale, binarised at >= 1 for the binary IR metrics. A hotel that is slightly over
budget or slightly slower to reach is now *partially relevant* (1) instead of
irrelevant (0). This matches the human annotation protocol
(docs/annotation_protocol.md) and, crucially, decouples gold from any single
system's rule: the Filter baseline's hard cut-offs are no longer a perfect oracle,
so the comparison becomes fair.

Grade:
  2  fully relevant     — satisfies every main constraint within strict bounds
  1  partially relevant — satisfies within a tolerance band, or (multi-constraint)
                          fails exactly one main constraint while meeting the rest
  0  not relevant       — fails beyond tolerance (single-constraint), or fails >= 2

Tolerance bands (mirror the protocol's "partially relevant"):
  price       : within budget = pass; up to +15% over = partial
  rating      : >= target = pass; within 0.2 below = partial
  star        : >= target = pass; exactly 1 below = partial
  travel time : <= target = pass; up to +2 min over = partial
  amenities   : all present = pass; some present = partial; none = fail

A missing constrained attribute counts as a fail for that constraint
(conservative — relevance requires positive evidence).
"""
from __future__ import annotations

from typing import Any, Dict, List, Set

# Grade levels
PASS, PARTIAL, FAIL = 2, 1, 0

# Tolerance bands
PRICE_TOL = 0.15       # fractional (15% over budget / under floor)
RATING_TOL = 0.2       # absolute rating points
STAR_TOL = 1           # absolute stars
TRAVEL_TOL_MIN = 2.0   # absolute minutes


def _v_ceiling_frac(value: Any, limit: float, tol_frac: float) -> int:
    """Lower-is-better ceiling constraint (e.g. max_price)."""
    if value is None:
        return FAIL
    v = float(value)
    if v <= limit:
        return PASS
    if v <= limit * (1 + tol_frac):
        return PARTIAL
    return FAIL


def _v_floor_frac(value: Any, limit: float, tol_frac: float) -> int:
    """Higher-is-better floor constraint (e.g. min_price / 'upscale above X')."""
    if value is None:
        return FAIL
    v = float(value)
    if v >= limit:
        return PASS
    if v >= limit * (1 - tol_frac):
        return PARTIAL
    return FAIL


def _v_floor_abs(value: Any, limit: float, tol_abs: float) -> int:
    """Higher-is-better floor with absolute tolerance (rating / star)."""
    if value is None:
        return FAIL
    v = float(value)
    if v >= limit:
        return PASS
    if v >= limit - tol_abs:
        return PARTIAL
    return FAIL


def _v_ceiling_abs(value: Any, limit: float, tol_abs: float) -> int:
    """Lower-is-better ceiling with absolute tolerance (travel time)."""
    if value is None:
        return FAIL
    v = float(value)
    if v <= limit:
        return PASS
    if v <= limit + tol_abs:
        return PARTIAL
    return FAIL


def _v_amenities(hotel: Dict[str, Any], needles: List[str]) -> int:
    have = [a.lower() for a in (hotel.get("amenities") or [])]
    hits = sum(1 for n in needles if any(n.lower() in a for a in have))
    if hits == len(needles):
        return PASS
    if hits > 0:
        return PARTIAL
    return FAIL


def _verdicts(hotel: Dict[str, Any], gold: Dict[str, Any]) -> List[int]:
    price = hotel.get("price")
    rating = hotel.get("rating")
    star = hotel.get("star")
    tt = hotel.get("travel_time_traffic_min")
    if tt is None:
        tt = hotel.get("travel_time_min")

    vs: List[int] = []
    if "max_price" in gold:
        vs.append(_v_ceiling_frac(price, gold["max_price"], PRICE_TOL))
    if "min_price" in gold:
        vs.append(_v_floor_frac(price, gold["min_price"], PRICE_TOL))
    if "min_rating" in gold:
        vs.append(_v_floor_abs(rating, gold["min_rating"], RATING_TOL))
    if "min_star" in gold:
        vs.append(_v_floor_abs(star, gold["min_star"], STAR_TOL))
    if "max_travel_time" in gold:
        vs.append(_v_ceiling_abs(tt, gold["max_travel_time"], TRAVEL_TOL_MIN))
    if "required_amenities" in gold:
        vs.append(_v_amenities(hotel, gold["required_amenities"]))
    return vs


def grade(hotel: Dict[str, Any], gold: Dict[str, Any]) -> int:
    """Graded relevance 0/1/2 of a hotel to a query's gold spec."""
    vs = _verdicts(hotel, gold)
    if not vs:
        return FAIL
    fails = sum(1 for v in vs if v == FAIL)
    partials = sum(1 for v in vs if v == PARTIAL)

    # Single-constraint query: the verdict is the grade.
    if len(vs) == 1:
        return vs[0]

    # Multi-constraint: two+ hard fails => not relevant; exactly one fail caps at
    # partial; otherwise partial if any tolerance-band hit, else fully relevant.
    if fails >= 2:
        return FAIL
    if fails == 1:
        return PARTIAL
    return PARTIAL if partials else PASS


def is_relevant(hotel: Dict[str, Any], gold: Dict[str, Any]) -> bool:
    """Binarised relevance for the binary IR metrics (grade >= 1)."""
    return grade(hotel, gold) >= PARTIAL


def relevant_set(hotels: List[Dict[str, Any]], gold: Dict[str, Any]) -> Set[str]:
    return {str(h["id"]) for h in hotels if is_relevant(h, gold)}


def graded_gold(hotels: List[Dict[str, Any]], gold: Dict[str, Any]) -> Dict[str, int]:
    """Full graded map {hotel_id: 1|2} for graded-gain metrics (nDCG). Excludes 0s."""
    out: Dict[str, int] = {}
    for h in hotels:
        g = grade(h, gold)
        if g > 0:
            out[str(h["id"])] = g
    return out
