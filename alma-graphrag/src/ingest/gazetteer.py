"""
Static gazetteer of Sri Lankan places, used to give Events a geography.

Why this exists
---------------
Events are ingested from news (`src/ingest/news_*.py`) and carry only
``id, title, severity, start_time, end_time, source, type`` — no coordinates.
Without a location an Event cannot be linked to hotels by anything except
"same city", which is what `event_linker._link_city_hotels_to_event` did: it
attached EVERY hotel in the city to the event with a constant impact_score of
0.5. A feature that takes the same value for every candidate cannot change a
ranking, so the disruption component was inert even on the rare occasions the
link fired.

Resolving the place named in a headline to a coordinate makes event impact a
*distance* — which does discriminate, and which is what the thesis actually
claims ("hotels near the disruption are down-ranked").

Why a static list rather than a geocoding API
---------------------------------------------
Deliberate, and it is a limitation to state in the write-up:

  * Reproducibility — a frozen list gives byte-identical results on re-run.
    A live geocoder does not, which would make the published numbers
    unreproducible by a reviewer.
  * No key, no quota, no network — the evaluation runs offline.
  * Precision over recall — a curated list of places that actually appear in
    Sri Lankan travel/traffic news yields few false positives. A general
    geocoder happily resolves "Fort" to a dozen countries.

The cost is recall: a headline naming a place absent from this list gets no
location and therefore no hotel links. `scripts/build_graph_topology.py`
reports that miss rate so the gap is measured rather than assumed. Extend the
list rather than swapping in an API if you need coverage — the table is the
documented, versioned input.

Coordinates are approximate centroids (~100 m) taken from public sources;
that is well inside the multi-kilometre impact radii they are used with.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# name -> (lat, lng). Ordered longest-first at match time so "Colombo Fort"
# wins over "Colombo".
PLACES: Dict[str, Tuple[float, float]] = {
    # --- Colombo city and its wards -------------------------------------
    "Colombo Fort": (6.9344, 79.8428),
    "Colombo": (6.9271, 79.8612),
    "Fort": (6.9344, 79.8428),
    "Pettah": (6.9367, 79.8503),
    "Galle Face": (6.9271, 79.8428),
    "Galle Face Green": (6.9271, 79.8428),
    "One Galle Face": (6.9270, 79.8428),
    "Slave Island": (6.9231, 79.8494),
    "Kollupitiya": (6.9111, 79.8497),
    "Colpetty": (6.9111, 79.8497),
    "Bambalapitiya": (6.8892, 79.8567),
    "Wellawatte": (6.8722, 79.8608),
    "Wellawatta": (6.8722, 79.8608),
    "Cinnamon Gardens": (6.9061, 79.8653),
    "Borella": (6.9147, 79.8778),
    "Maradana": (6.9294, 79.8656),
    "Kotahena": (6.9439, 79.8617),
    "Grandpass": (6.9450, 79.8700),
    "Dematagoda": (6.9333, 79.8781),
    "Mattakkuliya": (6.9639, 79.8697),
    "Modara": (6.9583, 79.8639),
    "Narahenpita": (6.8933, 79.8781),
    "Kirulapone": (6.8797, 79.8781),
    "Havelock Town": (6.8875, 79.8672),
    "Thimbirigasyaya": (6.8950, 79.8697),

    # --- Colombo landmarks ----------------------------------------------
    "Lotus Tower": (6.9270, 79.8612),
    "Viharamahadevi Park": (6.9147, 79.8614),
    "Independence Square": (6.9019, 79.8686),
    "Gangaramaya": (6.9169, 79.8564),
    "Beira Lake": (6.9236, 79.8531),
    "Colombo National Museum": (6.9107, 79.8613),
    "Port City": (6.9350, 79.8330),
    "Colombo Port": (6.9500, 79.8400),
    "Bandaranaike Memorial International Conference Hall": (6.8994, 79.8681),
    "BMICH": (6.8994, 79.8681),

    # --- Major arteries (linear features, centroid is adequate here) -----
    "Galle Road": (6.8892, 79.8567),
    "Marine Drive": (6.8850, 79.8540),
    "Duplication Road": (6.8950, 79.8580),
    "Baseline Road": (6.9200, 79.8800),
    "High Level Road": (6.8500, 79.9200),
    "Kandy Road": (6.9800, 79.9200),
    "Negombo Road": (7.0200, 79.8900),
    "Havelock Road": (6.8875, 79.8672),
    "Union Place": (6.9180, 79.8580),
    "Ward Place": (6.9150, 79.8700),
    "Parliament Road": (6.8878, 79.9186),

    # --- Greater Colombo -------------------------------------------------
    "Dehiwala": (6.8511, 79.8656),
    "Mount Lavinia": (6.8389, 79.8653),
    "Ratmalana": (6.8210, 79.8860),
    "Moratuwa": (6.7730, 79.8816),
    "Panadura": (6.7133, 79.9042),
    "Nugegoda": (6.8649, 79.8997),
    "Rajagiriya": (6.9089, 79.8931),
    "Battaramulla": (6.8994, 79.9181),
    "Sri Jayawardenepura Kotte": (6.8878, 79.9186),
    "Kotte": (6.8878, 79.9186),
    "Malabe": (6.9061, 79.9711),
    "Kaduwela": (6.9333, 79.9847),
    "Kottawa": (6.8408, 79.9647),
    "Maharagama": (6.8481, 79.9264),
    "Piliyandala": (6.8014, 79.9222),
    "Kesbewa": (6.7950, 79.9394),
    "Homagama": (6.8442, 80.0025),
    "Kelaniya": (6.9553, 79.9219),
    "Wattala": (6.9897, 79.8917),
    "Ja-Ela": (7.0744, 79.8919),
    "Kadawatha": (7.0000, 79.9500),

    # --- National (travel news frequently references these) --------------
    "Katunayake": (7.1808, 79.8842),
    "Bandaranaike International Airport": (7.1808, 79.8842),
    "Negombo": (7.2083, 79.8358),
    "Kandy": (7.2906, 80.6337),
    "Galle": (6.0535, 80.2210),
    "Bentota": (6.4260, 79.9959),
    "Hikkaduwa": (6.1395, 80.1063),
    "Mirissa": (5.9483, 80.4589),
    "Sigiriya": (7.9570, 80.7603),
    "Dambulla": (7.8742, 80.6511),
    "Anuradhapura": (8.3114, 80.4037),
    "Polonnaruwa": (7.9403, 81.0188),
    "Trincomalee": (8.5874, 81.2152),
    "Batticaloa": (7.7102, 81.6924),
    "Jaffna": (9.6615, 80.0255),
    "Nuwara Eliya": (6.9497, 80.7891),
    "Ella": (6.8667, 81.0466),
    "Arugam Bay": (6.8400, 81.8360),
    "Yala": (6.3728, 81.5016),
    "Matara": (5.9549, 80.5550),
    "Kurunegala": (7.4863, 80.3623),
    "Ratnapura": (6.6828, 80.3992),
    "Badulla": (6.9934, 81.0550),
}

# Places that name a whole city/town rather than a point inside one.
#
# This distinction decides whether an event may be linked to hotels by
# distance. A headline that says only "Colombo" localises the event to a
# 37 km² city; every hotel in the pool sits within a few km of that centroid,
# so a radius link from it would attach the event to essentially everything —
# reintroducing the constant-feature problem this module exists to remove.
# Such events are recorded with geo_precision="city" and are deliberately NOT
# used for impact linking. Only ward-, road- and landmark-level matches
# ("Pettah", "Galle Face", "Marine Drive") localise tightly enough to
# discriminate between candidates.
CITY_LEVEL: set[str] = {
    "Colombo", "Kandy", "Galle", "Negombo", "Matara", "Jaffna", "Batticaloa",
    "Trincomalee", "Anuradhapura", "Polonnaruwa", "Kurunegala", "Ratnapura",
    "Badulla", "Nuwara Eliya", "Moratuwa", "Panadura", "Dehiwala", "Katunayake",
}


def precision(name: str) -> str:
    """"city" (too coarse for impact linking) or "local" (usable)."""
    return "city" if name in CITY_LEVEL else "local"


# Longest names first so a specific place beats the city that contains it.
_ORDERED: List[str] = sorted(PLACES, key=len, reverse=True)

# Pre-compiled word-boundary patterns. Word boundaries matter: without them
# "Ella" matches inside "Stella" and "Fort" inside "Comfort".
_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    (name, re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE))
    for name in _ORDERED
]


def find_places(text: str, limit: int = 3) -> List[Tuple[str, float, float]]:
    """Return up to `limit` gazetteer places named in `text`.

    Longest match wins and overlapping spans are suppressed, so
    "Colombo Fort closed" yields Colombo Fort once, not Colombo Fort + Fort +
    Colombo. Results keep the order the places appear in the text.
    """
    if not text:
        return []

    taken: List[Tuple[int, int]] = []
    hits: List[Tuple[int, str]] = []

    for name, pat in _PATTERNS:
        for m in pat.finditer(text):
            span = (m.start(), m.end())
            if any(span[0] < e and s < span[1] for s, e in taken):
                continue  # overlaps a longer name already matched
            taken.append(span)
            hits.append((span[0], name))
            break  # one hit per place is enough

    hits.sort()
    out: List[Tuple[str, float, float]] = []
    for _, name in hits[:limit]:
        lat, lng = PLACES[name]
        out.append((name, lat, lng))
    return out


def locate(text: str) -> Optional[Tuple[str, float, float]]:
    """Best single place named in `text`, or None."""
    places = find_places(text, limit=1)
    return places[0] if places else None
