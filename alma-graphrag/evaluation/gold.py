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
  1  partially relevant — satisfies every constraint at least within its
                          tolerance band (one or more only band-satisfied)
  0  not relevant       — fails any constraint beyond its tolerance band

A hard fail on ANY constraint is disqualifying: a hotel twice over budget is not
relevant to a budget query no matter how well rated it is. (An earlier rule let
one hard fail through as "partial", which made multi-constraint gold sets cover
80-95% of the pool and saturated every metric.)

A missing constrained attribute counts as a fail for that constraint
(conservative — relevance requires positive evidence).

--------------------------------------------------------------------------
Tolerance bands are a PARAMETER, not a constant
--------------------------------------------------------------------------
The band widths below decide every gold label in the benchmark, and they were
originally four unexplained magic numbers. A reviewer is entitled to ask whether
the reported system ordering is an artefact of that choice.

They are therefore packaged in `ToleranceBands`, threaded through every grading
function as an optional argument, and swept by `evaluation/sensitivity.py`, which
re-grades the whole query set across a grid of band settings and reports whether
the ranking of systems is stable. Cite that table instead of defending the
defaults.

Defaults (the values used for all previously reported results, kept so numbers
remain reproducible):
  price       : within budget = pass; up to +15% over = partial
  rating      : >= target = pass; within 0.2 below = partial
  star        : >= target = pass; exactly 1 below = partial
  travel time : <= target = pass; up to +2 min over = partial
  disruption  : <= target delay = pass; up to +3 min over = partial
  amenities   : all present = pass; some present = partial; none = fail

--------------------------------------------------------------------------
Preference ladders (`prefer_cheaper` / `prefer_premium`)
--------------------------------------------------------------------------
Threshold constraints answer "is this hotel acceptable?". Preference ladders
answer "is this hotel BETTER?" — needed because a threshold grades everything
inside the budget as equally relevant, which makes the composite score's
`economic` weight unobservable. See `_v_ladder_lower`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Set

# Grade levels
PASS, PARTIAL, FAIL = 2, 1, 0


@dataclass(frozen=True)
class ToleranceBands:
    """Width of the "partially relevant" band for each constraint type.

    Frozen so a band set can be safely shared across a whole evaluation run and
    recorded verbatim in results.json for provenance.
    """
    price: float = 0.15        # fractional (15% over budget / under floor)
    rating: float = 0.2        # absolute rating points
    star: float = 1.0          # absolute stars
    travel_min: float = 2.0    # absolute minutes
    disruption_min: float = 3.0  # absolute minutes of added delay

    def scaled(self, factor: float) -> "ToleranceBands":
        """Uniformly widen (factor > 1) or tighten (factor < 1) every band.

        Used by the sensitivity sweep to trace the whole family of band choices
        with a single scalar, which keeps the resulting table readable.
        """
        return replace(
            self,
            price=self.price * factor,
            rating=self.rating * factor,
            star=self.star * factor,
            travel_min=self.travel_min * factor,
            disruption_min=self.disruption_min * factor,
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "price": round(self.price, 4),
            "rating": round(self.rating, 4),
            "star": round(self.star, 4),
            "travel_min": round(self.travel_min, 4),
            "disruption_min": round(self.disruption_min, 4),
        }


DEFAULT_BANDS = ToleranceBands()

# Backwards-compatible module constants (referenced by tests/test_gold.py and by
# the report generators). They mirror DEFAULT_BANDS and must not drift from it.
PRICE_TOL = DEFAULT_BANDS.price
RATING_TOL = DEFAULT_BANDS.rating
STAR_TOL = DEFAULT_BANDS.star
TRAVEL_TOL_MIN = DEFAULT_BANDS.travel_min
DISRUPTION_TOL_MIN = DEFAULT_BANDS.disruption_min


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
    """Lower-is-better ceiling with absolute tolerance (travel time, delay)."""
    if value is None:
        return FAIL
    v = float(value)
    if v <= limit:
        return PASS
    if v <= limit + tol_abs:
        return PARTIAL
    return FAIL


def _v_ladder_lower(value: Any, spec: Dict[str, float],
                    bands: ToleranceBands = DEFAULT_BANDS) -> int:
    """Graded PREFERENCE ladder for "cheaper is better" queries.

    Threshold gold (`max_price`) grades every hotel inside the budget as a 2, so
    they all tie and nDCG cannot see the order they were returned in. That is
    why the economic weight was invisible to the benchmark: on the 60-query set
    a profile with economic = 0.000 and one with 0.200 scored identically
    (0.6844, exactly the Filter baseline) on all ten economic queries.

    A ladder splits the pool into three bands instead of two, so a system that
    puts the cheapest hotels ABOVE the merely-affordable ones scores higher:

        price <= full     -> 2   fully relevant
        price <= partial  -> 1   partially relevant
        otherwise         -> 0

    The cut points are absolute LKR values FROZEN IN THE QUERY FILE, deliberately
    not percentiles of the live candidate pool. Pool percentiles are exactly what
    the retriever's own `economic` component computes, so grading against them
    would rebuild the circularity this module was written to remove — the gold
    would be scoring the system against its own scoring rule.

    Only the partial tier responds to the tolerance sweep, and it is expressed
    relative to the default band so that a sweep factor of 1.0 reproduces the
    stated cut point exactly.
    """
    if value is None:
        return FAIL
    v = float(value)
    if v <= float(spec["full"]):
        return PASS
    widened = float(spec["partial"]) * (1.0 + bands.price) / (1.0 + DEFAULT_BANDS.price)
    return PARTIAL if v <= widened else FAIL


def _v_ladder_higher(value: Any, spec: Dict[str, float],
                     bands: ToleranceBands = DEFAULT_BANDS) -> int:
    """Graded preference ladder for "more expensive is better" (premium) queries."""
    if value is None:
        return FAIL
    v = float(value)
    if v >= float(spec["full"]):
        return PASS
    narrowed = float(spec["partial"]) * (1.0 + DEFAULT_BANDS.price) / (1.0 + bands.price)
    return PARTIAL if v >= narrowed else FAIL


def _v_amenities(hotel: Dict[str, Any], needles: List[str]) -> int:
    have = [a.lower() for a in (hotel.get("amenities") or [])]
    hits = sum(1 for n in needles if any(n.lower() in a for a in have))
    if hits == len(needles):
        return PASS
    if hits > 0:
        return PARTIAL
    return FAIL


def _added_delay(hotel: Dict[str, Any]) -> Optional[float]:
    """Minutes of traffic/event delay on the route to this hotel.

    Prefers the explicit worst-case signal delay; falls back to the difference
    between the traffic-aware and free-flow travel times. Returns None when the
    graph carries no disruption evidence for the hotel, which grades as FAIL for
    a disruption constraint (relevance requires positive evidence).
    """
    eta = hotel.get("max_eta_change_min")
    if eta is not None:
        return float(eta)
    tt_traffic = hotel.get("travel_time_traffic_min")
    tt_free = hotel.get("travel_time_min")
    if tt_traffic is not None and tt_free is not None:
        return max(0.0, float(tt_traffic) - float(tt_free))
    return None


def _verdicts(hotel: Dict[str, Any], gold: Dict[str, Any],
              bands: ToleranceBands = DEFAULT_BANDS) -> List[int]:
    price = hotel.get("price")
    rating = hotel.get("rating")
    star = hotel.get("star")
    tt = hotel.get("travel_time_traffic_min")
    if tt is None:
        tt = hotel.get("travel_time_min")

    vs: List[int] = []
    if "max_price" in gold:
        vs.append(_v_ceiling_frac(price, gold["max_price"], bands.price))
    if "min_price" in gold:
        vs.append(_v_floor_frac(price, gold["min_price"], bands.price))
    if "prefer_cheaper" in gold:
        vs.append(_v_ladder_lower(price, gold["prefer_cheaper"], bands))
    if "prefer_premium" in gold:
        vs.append(_v_ladder_higher(price, gold["prefer_premium"], bands))
    if "min_rating" in gold:
        vs.append(_v_floor_abs(rating, gold["min_rating"], bands.rating))
    if "min_star" in gold:
        vs.append(_v_floor_abs(star, gold["min_star"], bands.star))
    if "max_travel_time" in gold:
        vs.append(_v_ceiling_abs(tt, gold["max_travel_time"], bands.travel_min))
    if "required_amenities" in gold:
        vs.append(_v_amenities(hotel, gold["required_amenities"]))
    # Disruption constraint — graded on ACTUAL added delay rather than on the
    # free-flow travel time. Without this a "avoid congestion" query is graded
    # by `max_travel_time`, which every system satisfies, producing a perfect
    # nDCG for all systems (a saturation signature, not a result).
    if "max_added_delay_min" in gold:
        vs.append(_v_ceiling_abs(
            _added_delay(hotel), gold["max_added_delay_min"], bands.disruption_min
        ))
    if "max_event_impact" in gold:
        vs.append(_v_ceiling_abs(
            hotel.get("event_impact"), gold["max_event_impact"], 0.1
        ))
    return vs


def grade(hotel: Dict[str, Any], gold: Dict[str, Any],
          bands: ToleranceBands = DEFAULT_BANDS) -> int:
    """Graded relevance 0/1/2 of a hotel to a query's gold spec."""
    vs = _verdicts(hotel, gold, bands)
    if not vs:
        return FAIL
    # Any hard fail is disqualifying; otherwise the weakest verdict wins
    # (all-pass => fully relevant, any band-hit => partially relevant).
    return min(vs)


def is_relevant(hotel: Dict[str, Any], gold: Dict[str, Any],
                bands: ToleranceBands = DEFAULT_BANDS) -> bool:
    """Binarised relevance for the binary IR metrics (grade >= 1)."""
    return grade(hotel, gold, bands) >= PARTIAL


def relevant_set(hotels: List[Dict[str, Any]], gold: Dict[str, Any],
                 bands: ToleranceBands = DEFAULT_BANDS) -> Set[str]:
    return {str(h["id"]) for h in hotels if is_relevant(h, gold, bands)}


def graded_gold(hotels: List[Dict[str, Any]], gold: Dict[str, Any],
                bands: ToleranceBands = DEFAULT_BANDS) -> Dict[str, int]:
    """Full graded map {hotel_id: 1|2} for graded-gain metrics (nDCG). Excludes 0s."""
    out: Dict[str, int] = {}
    for h in hotels:
        g = grade(h, gold, bands)
        if g > 0:
            out[str(h["id"])] = g
    return out
