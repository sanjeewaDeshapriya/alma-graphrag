"""
Build study material from LIVE LiteAPI data — location-anchored design.

Pulls the full Colombo hotel pool (images, facilities, description, price,
reviews, coordinates, guest sentiment AND bookable rooms) and assembles ten
choice tasks, each ANCHORED to a real Colombo location (Galle Face,
Battaramulla, Fort, ...). Every task shows the whole pool.

The choice is TWO-STAGE, as a real booking is: the participant picks a hotel,
then picks a room inside it.

    stage 1  hotel   35 alternatives, scored on the five ALMA components
    stage 2  room    2-5 offers within the chosen hotel, differing in price,
                     board basis, refundability, size, beds and occupancy

Stage 2 is not decoration. The five hotel weights are only identified up to
scale from stage-1 clicks alone; the room step contains an explicit price
attribute measured in rupees, so the trade-offs inside it (what a guest pays to
add breakfast, or to keep free cancellation) put the whole weight vector on a
monetary scale. It is the standard reason a discrete-choice design carries a
price attribute.

Why the components are split
----------------------------
Two of the five ALMA sub-scores depend on WHERE the task is anchored and three
do not:

    per-anchor : spatial       — straight-line proximity to the anchor
                 accessibility — traffic-adjusted travel time to the anchor
    global     : facility      — star / facility count / rating
                 economic      — price
                 disruption    — how calm the hotel's own area is

`spatial` is straight-line proximity; `accessibility` is real traffic-aware
driving time pulled live from the Distance Matrix API. They stay ~0.90
correlated in Colombo — that is the city, not a measurement fault. It matters
only if you try to fit five separate coefficients; the deliverable here is a
LABELLED RANKING DATASET (context -> 35 candidates -> the one chosen), which is
the standard learning-to-rank shape and tolerates correlated features. build()
prints the correlation matrix so the property is recorded, not hidden.

Output: studies/weight-elicitation/material/study_material_v1.json (versioned)
— written into the Next.js sub-project, which bundles it at build time.

Run (needs LITEAPI_KEY in the repo .env):
    python -m weight_elicitation.build_material
"""
from __future__ import annotations

import argparse
import html
import json
import math
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import httpx  # noqa: E402

from src.config import LITEAPI_BASE_URL, LITEAPI_KEY, LITEAPI_LOS_NIGHTS  # noqa: E402
from src.ingest.liteapi import LiteApiClient  # noqa: E402

# Written INTO the Next.js sub-project: `lib/material.ts` imports this file
# directly, so Next bundles it at build time and the deployed study needs no
# database, no Neo4j and no Python at runtime.
from weight_elicitation import MATERIAL as OUT_DEFAULT  # noqa: E402

# Colombo Fort / Galle Face — the reference "city centre" for the global scores.
CENTER_LAT, CENTER_LNG = 6.9271, 79.8612

# --------------------------------------------------------------------------- #
# Location anchors — every choice task is framed around one of these.
# --------------------------------------------------------------------------- #
ANCHORS: Dict[str, Dict[str, Any]] = {
    "galle_face":    {"name": "Galle Face Green", "lat": 6.9271, "lng": 79.8426},
    "battaramulla":  {"name": "Battaramulla",     "lat": 6.8994, "lng": 79.9186},
    "fort":          {"name": "Colombo Fort",     "lat": 6.9344, "lng": 79.8428},
    "bambalapitiya": {"name": "Bambalapitiya",    "lat": 6.8890, "lng": 79.8564},
    "nugegoda":      {"name": "Nugegoda",         "lat": 6.8649, "lng": 79.8997},
    "mount_lavinia": {"name": "Mount Lavinia",    "lat": 6.8389, "lng": 79.8653},
    "rajagiriya":    {"name": "Rajagiriya",       "lat": 6.9097, "lng": 79.8944},
    "kollupitiya":   {"name": "Kollupitiya",      "lat": 6.9101, "lng": 79.8494},
    "dehiwala":      {"name": "Dehiwala",         "lat": 6.8511, "lng": 79.8653},
}

# Room amenities we surface as chips on a room card (real LiteAPI names → label).
ROOM_AMENITY_CHIPS = [
    ("Air conditioning", "AC"), ("Balcony", "Balcony"), ("Terrace", "Terrace"),
    ("Sea view", "Sea view"), ("City view", "City view"), ("Pool view", "Pool view"),
    ("Free WiFi", "Free WiFi"), ("Flat-screen TV", "Flat-screen TV"),
    ("Minibar", "Minibar"), ("Safety deposit box", "Safe"), ("Safe", "Safe"),
    ("Bath", "Bathtub"), ("Bathtub", "Bathtub"), ("Shower", "Shower"),
    ("Coffee machine", "Coffee machine"), ("Electric kettle", "Kettle"),
    ("Desk", "Desk"), ("Soundproofing", "Soundproofed"),
    ("Kitchenette", "Kitchenette"), ("Refrigerator", "Fridge"),
    ("Wardrobe or closet", "Wardrobe"), ("Private bathroom", "Private bathroom"),
    ("Free toiletries", "Toiletries"), ("Slippers", "Slippers"),
    ("Sofa", "Sofa"), ("Bathrobe", "Bathrobe"),
]

# Facilities we surface as chips (real LiteAPI names → short label), in priority order.
FACILITY_CHIPS = [
    ("Swimming pool", "Pool"), ("Outdoor pool", "Pool"), ("Rooftop pool", "Rooftop pool"),
    ("Spa", "Spa"), ("Spa and wellness", "Spa"),
    ("Free WiFi", "Free WiFi"), ("WiFi available", "WiFi"),
    ("Fitness center", "Gym"), ("Fitness facilities", "Gym"),
    ("Restaurant", "Restaurant"), ("Bar", "Bar"), ("Room service", "Room service"),
    ("Free Parking", "Free parking"), ("Parking", "Parking"),
    ("Airport shuttle", "Airport shuttle"), ("Family rooms", "Family rooms"),
    ("Non-smoking rooms", "Non-smoking"), ("Breakfast", "Breakfast"),
    ("Air conditioning", "AC"), ("Beachfront", "Beachfront"), ("Pet friendly", "Pet friendly"),
]

# --------------------------------------------------------------------------- #
# The ten tasks. Each one merges a PRIMARY dimension with a SECONDARY one, so
# every scenario forces a trade-off rather than a single-attribute sort. Each of
# the five dimensions is primary in exactly two tasks and secondary in two, which
# keeps the labelled dataset balanced across the scoring matrix.
#
#   task  anchor          primary        secondary
#   t1    Galle Face      spatial        economic
#   t2    Battaramulla    accessibility  facility
#   t3    Fort            economic       accessibility
#   t4    Bambalapitiya   facility       spatial
#   t5    Nugegoda        disruption     economic
#   t6    Mount Lavinia   spatial        disruption
#   t7    Rajagiriya      economic       accessibility
#   t8    Kollupitiya     facility       economic
#   t9    Dehiwala        disruption     spatial
#   t10   Battaramulla    accessibility  facility   (repeat of t2)
# --------------------------------------------------------------------------- #
TASKS_SPEC: List[Dict[str, Any]] = [
    {
        "id": "t1", "anchor": "galle_face", "persona": "Evening walker",
        "primary": "spatial", "secondary": "economic",
        "context": "You are visiting Colombo and want to walk to Galle Face Green every "
                   "evening. Being close enough to reach it on foot matters most, but you "
                   "are also paying for the room yourself and cannot overspend. Which "
                   "hotel would you book?",
    },
    {
        "id": "t2", "anchor": "battaramulla", "persona": "Business traveller",
        "primary": "accessibility", "secondary": "facility",
        "context": "You have two days of back-to-back meetings at government offices in "
                   "Battaramulla. Reaching them quickly and predictably each morning "
                   "decides your booking, though you would still like somewhere "
                   "comfortable to work in the evening. Which hotel would you book?",
    },
    {
        "id": "t3", "anchor": "fort", "persona": "Self-funded worker",
        "primary": "economic", "secondary": "accessibility",
        "context": "You are working in Colombo Fort for a week and paying for the room "
                   "yourself, so cost comes first - but a long daily commute would eat "
                   "into your working day. Which hotel would you book?",
    },
    {
        "id": "t4", "anchor": "bambalapitiya", "persona": "Family on holiday",
        "primary": "facility", "secondary": "spatial",
        "context": "You are travelling with your family and two young children and "
                   "visiting relatives in Bambalapitiya. A pool, space and good on-site "
                   "facilities matter most, and being near the family helps. Which hotel "
                   "would you book?",
    },
    {
        "id": "t5", "anchor": "nugegoda", "persona": "Light sleeper",
        "primary": "disruption", "secondary": "economic",
        "context": "You are attending an event in Nugegoda but you are a light sleeper who "
                   "cannot rest with traffic noise and crowds. A calm area matters most, "
                   "though you do not want to pay a premium for it. Which hotel would you "
                   "book?",
    },
    {
        "id": "t6", "anchor": "mount_lavinia", "persona": "Beach weekender",
        "primary": "spatial", "secondary": "disruption",
        "context": "You want a weekend by the sea at Mount Lavinia. Being close to the "
                   "beach is the point of the trip, but you also want somewhere peaceful "
                   "rather than a busy strip. Which hotel would you book?",
    },
    {
        "id": "t7", "anchor": "rajagiriya", "persona": "Long-stay consultant",
        "primary": "economic", "secondary": "accessibility",
        "context": "You are on a three-week assignment with an office in Rajagiriya and a "
                   "fixed daily allowance, so the nightly rate matters a great deal - but "
                   "a slow commute every day would wear you down. Which hotel would you "
                   "book?",
    },
    {
        "id": "t8", "anchor": "kollupitiya", "persona": "Value-seeking couple",
        "primary": "facility", "secondary": "economic",
        "context": "You and your partner are staying near Kollupitiya for a few nights and "
                   "want a genuinely nice hotel - good rooms and facilities - at a price "
                   "that still feels like value rather than a splurge. Which hotel would "
                   "you book?",
    },
    {
        "id": "t9", "anchor": "dehiwala", "persona": "Remote worker",
        "primary": "disruption", "secondary": "spatial",
        "context": "You are working remotely for two weeks near Dehiwala. You need "
                   "somewhere quiet enough to take calls all day, but you also do not want "
                   "to be cut off from shops and transport. Which hotel would you book?",
    },
    {
        # Deliberate repeat of t2 - measures test-retest reliability per participant.
        "id": "t10", "anchor": "battaramulla", "persona": "Business traveller",
        "primary": "accessibility", "secondary": "facility", "repeat_of": "t2",
        "context": "You are again booking for two days of meetings in Battaramulla, with "
                   "short and predictable travel as your priority and a comfortable room "
                   "to return to. Which hotel would you book?",
    },
]

DIMENSIONS = ["spatial", "accessibility", "facility", "economic", "disruption"]
ANCHOR_DIMS = ["spatial", "accessibility"]
GLOBAL_DIMS = ["facility", "economic", "disruption"]


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fetch_details(hotel_id: str) -> Dict[str, Any]:
    try:
        with httpx.Client(timeout=20.0) as c:
            r = c.get(f"{LITEAPI_BASE_URL}/data/hotel",
                      params={"hotelId": hotel_id},
                      headers={"X-API-Key": LITEAPI_KEY})
            r.raise_for_status()
            return r.json().get("data") or {}
    except Exception:
        return {}


def strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def short_area(address: str) -> str:
    m = re.search(r"Colombo\s*\d{1,2}", address or "", re.I)
    if m:
        return m.group(0)
    return (address or "Colombo").split(",")[0][:28]


def pick_chips(facility_names: List[str], limit: int = 6,
               table: List[Tuple[str, str]] | None = None) -> List[str]:
    present = {f.strip().lower() for f in facility_names}
    chips: List[str] = []
    for real, label in (table or FACILITY_CHIPS):
        if real.lower() in present and label not in chips:
            chips.append(label)
        if len(chips) >= limit:
            break
    return chips


def _amount(price_array: Any) -> float:
    """First amount out of a LiteAPI money array, or 0."""
    if isinstance(price_array, list) and price_array:
        try:
            return float(price_array[0].get("amount"))
        except (TypeError, ValueError, AttributeError):
            return 0.0
    if isinstance(price_array, dict):
        try:
            return float(price_array.get("amount"))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def normalise_size_unit(unit: Any) -> str:
    u = str(unit or "").strip().lower()
    if u in ("sqft", "ft2", "sq ft", "square feet"):
        return "ft²"
    return "m²"


def bed_summary(bed_types: List[Dict[str, Any]]) -> str:
    """'1 x King bed' / '2 x Twin bed' — the phrasing a booking site would use."""
    parts = []
    for b in bed_types or []:
        name = str(b.get("bedType") or "Bed").strip()
        qty = int(b.get("quantity") or 1)
        parts.append(f"{qty} x {name}" if qty > 1 else name)
    return ", ".join(parts[:2])


def build_room_offers(entry: Dict[str, Any], catalogue: Dict[int, Dict[str, Any]],
                      nights: int, limit: int = 5) -> List[Dict[str, Any]]:
    """Bookable room offers for one hotel, merging the two LiteAPI surfaces.

    `/hotels/rates` gives the COMMERCIAL side of an offer — live price, board
    basis, refundability, cancellation deadline — but only a supplier room label
    ("TWIN STANDARD"). `/data/hotel` gives the CONTENT side — proper room name,
    floor area, bed configuration, occupancy, photos, in-room amenities — but no
    price. `rates[].mappedRoomId` is the join key onto `rooms[].id`, and in
    practice it resolves for essentially every live offer.

    One offer per (room, board basis, refundability): those three are what a
    guest actually trades off inside a hotel, and keeping them distinct is the
    point of the second stage — the price gap between "Room Only,
    non-refundable" and "Breakfast, free cancellation" prices those attributes
    in rupees, which is what puts the hotel-level weights on a monetary scale.
    """
    best: Dict[Tuple[Any, str, bool], Dict[str, Any]] = {}

    for rt in entry.get("roomTypes") or []:
        rates = rt.get("rates") or []
        if not rates:
            continue
        rate = rates[0]
        retail = rate.get("retailRate") or {}
        total = _amount(rt.get("offerRetailRate")) or _amount(retail.get("total"))
        if not total:
            continue
        per_night = total / max(nights, 1)

        cancel = rate.get("cancellationPolicies") or {}
        refundable = cancel.get("refundableTag") == "RFN"
        infos = cancel.get("cancelPolicyInfos") or []
        board_type = str(rate.get("boardType") or "").strip()
        board_name = str(rate.get("boardName") or "").strip()

        mapped = rate.get("mappedRoomId")
        room = catalogue.get(int(mapped)) if mapped is not None else None

        taxes = retail.get("taxesAndFees") or []
        taxes_included = all(t.get("included") for t in taxes) if taxes else True

        amenities = [
            a.get("name") if isinstance(a, dict) else a
            for a in ((room or {}).get("roomAmenities") or [])
        ]
        photos = [p.get("url") for p in ((room or {}).get("photos") or []) if p.get("url")]

        offer = {
            # Stable within a material version: the join key plus what makes this
            # offer distinct from the others on the same room.
            "id": f"{mapped or rt.get('roomTypeId')}-{board_type or 'NA'}-{'RFN' if refundable else 'NRF'}",
            "name": str((room or {}).get("roomName") or rate.get("name") or "Room")[:110],
            "price_lkr": round(per_night),
            "board_name": board_name or "Room Only",
            "board_type": board_type,
            "refundable": refundable,
            "cancel_before": (infos[0].get("cancelTime") if infos else None),
            "taxes_included": taxes_included,
            "max_occupancy": int((room or {}).get("maxOccupancy")
                                 or rate.get("maxOccupancy") or 2),
            "max_adults": int((room or {}).get("maxAdults") or rate.get("adultCount") or 2),
            "max_children": int((room or {}).get("maxChildren") or 0),
            "size_sqm": (room or {}).get("roomSizeSquare"),
            # LiteAPI returns square metres as either "sqm" or "m2" depending on
            # the supplier; both mean the same thing, so show one label.
            "size_unit": normalise_size_unit((room or {}).get("roomSizeUnit")),
            "beds": bed_summary((room or {}).get("bedTypes") or []),
            "amenities": pick_chips([a for a in amenities if a], 6, ROOM_AMENITY_CHIPS),
            "image": photos[0] if photos else None,
            "description": strip_html((room or {}).get("description") or "")[:200],
            "mapped_room_id": mapped,
            "from_catalogue": room is not None,
        }

        key = (mapped, board_type, refundable)
        if key not in best or per_night < best[key]["price_lkr"]:
            best[key] = offer

    offers = sorted(best.values(), key=lambda o: o["price_lkr"])[:limit]
    return offers


def fetch_pool(city: str, fetch_n: int, pool_n: int, min_rooms: int) -> List[Dict[str, Any]]:
    client = LiteApiClient()
    try:
        body = client.search_rates(city, max_results=fetch_n)
        meta = {h.get("id"): h for h in (body.get("hotels") or []) if h.get("id")}
        rows: List[Dict[str, Any]] = []
        for entry in body.get("data") or []:
            hid = entry.get("hotelId")
            m = meta.get(hid)
            if not hid or not m:
                continue
            plans = client._extract_room_types(entry.get("roomTypes") or [])
            price = min((p["price"] for p in plans if p.get("price")), default=0)
            lat, lng = m.get("latitude"), m.get("longitude")
            raw_rating = float(m.get("rating") or 0)
            # An unrated hotel must not enter the pool. LiteAPI returns 0 for a
            # missing guest score, which would render as "0.0" beside genuine
            # 4.5s and read as the worst hotel on the list rather than an
            # unknown one — biasing every task against it, and docking its
            # hidden `facility` component too.
            if (not price or lat is None or lng is None
                    or not m.get("main_photo") or raw_rating <= 0):
                continue
            rows.append({
                "id": hid,
                "name": m.get("name", "Hotel"),
                "price_lkr": round(price),
                "rating5": round(raw_rating / 2, 1) if raw_rating > 5 else round(raw_rating, 1),
                "star": int(m.get("stars") or 0),
                "review_count": int(m.get("review_count") or 0),
                "image": m.get("main_photo"),
                "address": m.get("address", ""),
                "lat": float(lat),
                "lng": float(lng),
                "_entry": entry,          # kept for room extraction, stripped below
            })
    finally:
        client.close()

    # De-dup by name, keep the better-reviewed; cap to pool_n most-reviewed.
    seen: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = row["name"].lower()
        if key not in seen or row["review_count"] > seen[key]["review_count"]:
            seen[key] = row
    candidates = sorted(seen.values(), key=lambda h: -h["review_count"])[:pool_n]

    nights = max(LITEAPI_LOS_NIGHTS, 1)
    pool: List[Dict[str, Any]] = []
    dropped_no_rooms = 0
    for h in candidates:
        d = fetch_details(h["id"])
        facs = [
            f.get("name") if isinstance(f, dict) else f
            for f in (d.get("hotelFacilities") or d.get("facilities") or [])
        ]
        facs = [str(f).strip() for f in facs if f]

        catalogue = {
            int(r["id"]): r for r in (d.get("rooms") or []) if r.get("id") is not None
        }
        rooms = build_room_offers(h.pop("_entry"), catalogue, nights)
        # A hotel with a single offer makes the room step a formality, and a
        # forced non-choice is not an observation. Drop it rather than record it.
        if len(rooms) < min_rooms:
            dropped_no_rooms += 1
            continue

        h["rooms"] = rooms
        # The headline price is the cheapest bookable offer, so what the card
        # promises is what the room list can actually deliver.
        h["price_lkr"] = min(r["price_lkr"] for r in rooms)
        h["facilities"] = pick_chips(facs, 6) or ["Free WiFi", "AC"]
        h["n_facilities"] = len(facs)
        h["description"] = strip_html(d.get("hotelDescription") or d.get("description") or "")[:170]
        h["distance_km"] = round(haversine_km(h["lat"], h["lng"], CENTER_LAT, CENTER_LNG), 1)
        h["area"] = short_area(h["address"])
        h["detail"] = build_hotel_detail(d, facs)
        pool.append(h)

    if dropped_no_rooms:
        print(f"  dropped {dropped_no_rooms} hotels with fewer than {min_rooms} live room offers")
    return pool


def build_hotel_detail(d: Dict[str, Any], facilities: List[str]) -> Dict[str, Any]:
    """Everything the detail panel shows beyond the summary card.

    All of it is real LiteAPI content. A participant who opens a hotel should be
    looking at what a booking site would show them, otherwise the choice we are
    modelling is not the choice people actually make.
    """
    images = [
        {"url": im.get("url"), "caption": strip_html(im.get("caption") or "")[:60]}
        for im in (d.get("hotelImages") or [])[:8]
        if im.get("url")
    ]
    times = d.get("checkinCheckoutTimes") or {}
    sentiment = d.get("sentiment_analysis") or {}
    categories = [
        {"name": c.get("name"), "rating": round(float(c.get("rating") or 0), 1)}
        for c in (sentiment.get("categories") or [])[:6]
        if c.get("name")
    ]
    return {
        "description_full": strip_html(
            d.get("hotelDescription") or d.get("description") or ""
        )[:900],
        "facilities_all": facilities[:48],
        "n_facilities": len(facilities),
        "images": images,
        "checkin": times.get("checkin_start") or None,
        "checkout": times.get("checkout") or None,
        "hotel_type": d.get("hotelType") or None,
        "chain": (d.get("chain") if d.get("chain") not in ("Not Available", "") else None),
        "pros": [str(p)[:60] for p in (sentiment.get("pros") or [])[:4]],
        "cons": [str(c)[:60] for c in (sentiment.get("cons") or [])[:4]],
        "review_categories": categories,
    }


# --------------------------------------------------------------------------- #
# Component derivation
# --------------------------------------------------------------------------- #
def pct_rank(x: float, arr: List[float]) -> float:
    if not arr:
        return 0.5
    return sum(1 for v in arr if v <= x) / len(arr)


def clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def local_density(pool: List[Dict[str, Any]], radius_km: float = 1.2) -> Dict[str, float]:
    """Hotels within `radius_km` of each hotel, normalised to [0,1].

    Used as a reproducible congestion / bustle proxy: dense clusters of hotels
    sit in the busy commercial strips, which are slower to drive through and
    noisier to sleep in. It is the only input that separates `accessibility`
    from `spatial` and `disruption` from both, so it carries real weight in the
    design — see the correlation report in build().
    """
    counts: Dict[str, int] = {}
    for a in pool:
        counts[a["id"]] = sum(
            1 for b in pool
            if b["id"] != a["id"]
            and haversine_km(a["lat"], a["lng"], b["lat"], b["lng"]) <= radius_km
        )
    hi = max(counts.values()) or 1
    return {hid: n / hi for hid, n in counts.items()}


def facility_component(h: Dict[str, Any],
                       stars: List[float],
                       n_facilities: List[float],
                       ratings: List[float]) -> float:
    """How well-appointed a hotel is, on [0,1].

    THIS DEFINITION DECIDES WHETHER THE `facility` WEIGHT CAN BE ESTIMATED AT ALL.

    A conditional logit identifies each weight from the component's spread WITHIN
    a choice set. The original definition,

        0.45 * (star / 5) + 0.35 * min(n_facilities / 40, 1.0) + 0.20 * (rating / 5)

    destroyed most of that spread. Facility counts in the Colombo pool run 22 to
    125, so `min(n / 40, 1.0)` saturated at 1.0 for 30 of 32 hotels: 35% of the
    formula contributed ZERO variance, and what remained was a rescaled star
    rating (27 of 32 hotels are 4- or 5-star) crushed into [0.85, 0.99].

        facility within-set sd = 0.0968     <- the regressor being estimated
        every other dimension  = 0.2886     <- 3x more identifying variation

    With a third of the variation of its competitors, `facility` fitted with a
    standard error wide enough to swallow its own point estimate; the sign landed
    negative on noise and clipped to zero on the simplex. That is the whole
    explanation for `ELICITED_WEIGHTS.facility == 0.000` in the retriever.

    A second, related problem: `star` is also the main driver of price, so it is
    what produced corr(facility, economic) = -0.699 and left the two dimensions
    fighting each other for the same variation.

    The fix keeps the three inputs and their weights but puts each on the
    percentile scale that `economic`, `spatial` and `accessibility` already use,
    which is both the largest repair available and the one that leaves the
    meaning of "facility" unchanged for the thesis:

        within-set sd   corr w/ economic   VIF    identifying power
        ------------------------------------------------------------------
        current (cap)      0.0969   -0.747   2.33   0.063
        rank facilities    0.1616   -0.621   1.69   0.124
        rank all three     0.2401   -0.745   2.29   0.159   <- used
        drop star          0.2170   -0.719   2.09   0.150

    ("Identifying power" is sd * sqrt(1 - R2), since a coefficient's standard
    error scales as its reciprocal. See DATA_AUDIT.md §3b.)

    A caveat worth carrying into the write-up: none of these repairs breaks
    corr(facility, economic) ~ -0.7. In Colombo the expensive hotels genuinely do
    have more facilities, so only adding cheap-but-well-equipped hotels to the
    pool decorrelates the two — no formula can.

    `python -m weight_elicitation.fit_weights --facility-def` reproduces all four rows above
    against the collected data.
    """
    return (0.45 * pct_rank(h["star"] or 0, stars)
            + 0.35 * pct_rank(h["n_facilities"], n_facilities)
            + 0.20 * pct_rank(h["rating5"], ratings))


def compute_global_components(pool: List[Dict[str, Any]],
                              density: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """facility / economic / disruption — independent of which anchor is asked."""
    prices = [h["price_lkr"] for h in pool]
    stars = [float(h["star"] or 0) for h in pool]
    n_facilities = [float(h["n_facilities"]) for h in pool]
    ratings = [float(h["rating5"]) for h in pool]
    out: Dict[str, Dict[str, float]] = {}
    for h in pool:
        econ = 1 - pct_rank(h["price_lkr"], prices)              # cheaper → higher
        star = (h["star"] or 0) / 5.0
        fac = clamp(facility_component(h, stars, n_facilities, ratings))
        # Calm = away from the dense commercial clusters, and better-appointed
        # hotels insulate better (glazing, set-back grounds).
        calm = clamp(0.75 * (1 - density[h["id"]]) + 0.25 * star)
        out[h["id"]] = {"facility": round(fac, 3),
                        "economic": round(econ, 3),
                        "disruption": round(calm, 3),
                        # WAVE 2 §2E. `facility` bundles three different
                        # constructs behind fixed 0.45/0.35/0.20 weights: an
                        # official classification, an amenity count and guest
                        # satisfaction. Recording them separately lets the fit
                        # ESTIMATE that blend instead of inheriting our guess,
                        # while the combined value above keeps wave-1 and
                        # wave-2 models comparable.
                        "facility_star": round(pct_rank(h["star"] or 0, stars), 3),
                        "facility_count": round(pct_rank(h["n_facilities"], n_facilities), 3),
                        "facility_rating": round(pct_rank(h["rating5"], ratings), 3)}
    return out


def fetch_road_travel_times(pool: List[Dict[str, Any]],
                            ) -> Dict[str, Dict[str, float]]:
    """Real traffic-aware driving minutes from every hotel to every anchor.

    This is the measurement that makes `accessibility` a genuinely separate
    construct from `spatial`. Straight-line distance answers "how close is it";
    the road network under live traffic answers "how long does it actually
    take", and the two diverge sharply in Colombo - a hotel 2.5 km out on an
    arterial reaches Battaramulla faster than one 1.6 km out across the centre.

    Returns {anchor_id: {hotel_id: minutes}}, or {} if the API is unavailable,
    in which case the caller falls back to the density proxy.

    Distance Matrix allows 100 elements per request, so the pool is chunked into
    groups of origins small enough that origins x anchors stays under that.
    """
    from src.config import GOOGLE_MAPS_API_KEY
    if not GOOGLE_MAPS_API_KEY:
        print("  GOOGLE_MAPS_API_KEY not set - falling back to the density proxy")
        return {}

    anchor_ids = list(ANCHORS.keys())
    dests = "|".join(f"{ANCHORS[a]['lat']},{ANCHORS[a]['lng']}" for a in anchor_ids)
    chunk = max(1, 100 // len(anchor_ids))
    out: Dict[str, Dict[str, float]] = {a: {} for a in anchor_ids}

    try:
        with httpx.Client(timeout=40.0) as client:
            for start in range(0, len(pool), chunk):
                group = pool[start:start + chunk]
                origins = "|".join(f"{h['lat']},{h['lng']}" for h in group)
                r = client.get(
                    "https://maps.googleapis.com/maps/api/distancematrix/json",
                    params={"origins": origins, "destinations": dests,
                            "departure_time": "now", "traffic_model": "best_guess",
                            "key": GOOGLE_MAPS_API_KEY},
                )
                r.raise_for_status()
                data = r.json()
                if data.get("status") != "OK":
                    raise RuntimeError(data.get("error_message") or data.get("status"))
                for h, row in zip(group, data.get("rows", [])):
                    for aid, el in zip(anchor_ids, row.get("elements", [])):
                        if el.get("status") != "OK":
                            continue
                        secs = (el.get("duration_in_traffic") or el.get("duration") or {}).get("value")
                        if secs:
                            out[aid][h["id"]] = secs / 60.0
    except Exception as exc:  # network, quota, billing - fall back rather than fail
        print(f"  distance matrix unavailable ({exc}) - falling back to the density proxy")
        return {}

    covered = min(len(v) for v in out.values()) if out else 0
    if covered < len(pool):
        print(f"  distance matrix covered only {covered}/{len(pool)} hotels "
              f"- falling back to the density proxy")
        return {}
    print(f"  real traffic-aware travel times for {len(pool)} hotels x "
          f"{len(anchor_ids)} anchors")
    return out


def route_congestion(pool: List[Dict[str, Any]], hotel: Dict[str, Any],
                     anchor: Dict[str, Any], radius_km: float = 1.2) -> float:
    """Congestion along the hotel->anchor route, not just around the hotel.

    Sampling the corridor rather than the origin is what makes accessibility a
    genuinely different construct from spatial: two hotels equidistant from the
    anchor score differently when one route crosses the dense central strips and
    the other runs down a clear arterial. A hotel-only congestion factor is the
    same number for every anchor, so it cannot break the distance correlation.
    """
    loads: List[float] = []
    for frac in (0.25, 0.5, 0.75):
        lat = hotel["lat"] + (anchor["lat"] - hotel["lat"]) * frac
        lng = hotel["lng"] + (anchor["lng"] - hotel["lng"]) * frac
        near = sum(1 for b in pool
                   if haversine_km(lat, lng, b["lat"], b["lng"]) <= radius_km)
        loads.append(near)
    hi = max(len(pool) * 0.5, 1.0)
    return min(sum(loads) / len(loads) / hi, 1.0)


def compute_anchor_components(pool: List[Dict[str, Any]], anchor: Dict[str, Any],
                              density: Dict[str, float],
                              road_minutes: Dict[str, float] | None = None,
                              ) -> Dict[str, Dict[str, float]]:
    """spatial / accessibility relative to one anchor, plus the raw distance.

    `spatial`       straight-line proximity - "how close is it".
    `accessibility` traffic-aware driving minutes - "how long does it take".

    With real `road_minutes` these are two independent measurements and their
    weights are separately identifiable. Without them we fall back to a density
    proxy - but note that every proxy tried here is itself a function of the
    same straight-line distance, so it cannot break the correlation (the
    route-corridor variant made it worse, r 0.84 -> 0.95, because longer routes
    necessarily cross more hotels). The correlation report in build() is the check.
    """
    dists = {h["id"]: haversine_km(h["lat"], h["lng"], anchor["lat"], anchor["lng"])
             for h in pool}
    if road_minutes:
        travel = {h["id"]: road_minutes[h["id"]] for h in pool}
    else:
        travel = {
            h["id"]: (dists[h["id"]] / 26.0) * 60.0 * (1.0 + 1.4 * density[h["id"]])
            for h in pool
        }
    dist_vals = list(dists.values())
    travel_vals = list(travel.values())
    out: Dict[str, Dict[str, float]] = {}
    for h in pool:
        hid = h["id"]
        out[hid] = {
            "spatial": round(1 - pct_rank(dists[hid], dist_vals), 3),
            "accessibility": round(1 - pct_rank(travel[hid], travel_vals), 3),
            "distance_km": round(dists[hid], 1),
            "travel_min": round(travel[hid]),
        }
    return out


# --------------------------------------------------------------------------- #
# Design diagnostics
# --------------------------------------------------------------------------- #
def pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else 0.0


def correlation_report(rows: List[Dict[str, float]]) -> Tuple[List[List[float]], float]:
    """Correlation matrix of the five component columns over every alternative
    that any participant will actually see. High |r| between two columns means
    those two weights cannot be told apart — the design, not the sample size,
    is what has to change."""
    cols = {d: [r[d] for r in rows] for d in DIMENSIONS}
    mat = [[pearson(cols[a], cols[b]) for b in DIMENSIONS] for a in DIMENSIONS]
    worst = max(
        (abs(mat[i][j]) for i in range(len(DIMENSIONS)) for j in range(len(DIMENSIONS)) if i != j),
        default=0.0,
    )
    return mat, worst


# --------------------------------------------------------------------------- #
def build(city: str, fetch_n: int, pool_n: int, seed: int, out: Path,
          min_rooms: int = 2, max_abs_r: float = 0.7) -> None:
    rng = random.Random(seed)
    print(f"LiteAPI · fetching up to {fetch_n} {city} hotels (pool cap {pool_n})…")
    pool = fetch_pool(city, fetch_n, pool_n, min_rooms)
    if len(pool) < 10:
        raise SystemExit(f"Only {len(pool)} usable hotels — need at least 10.")
    n_rooms = sum(len(h["rooms"]) for h in pool)
    n_mapped = sum(1 for h in pool for r in h["rooms"] if r["from_catalogue"])
    print(f"  pool of {len(pool)} hotels with real images + facilities")
    print(f"  {n_rooms} room offers ({n_mapped} joined to the room catalogue, "
          f"{n_rooms / len(pool):.1f} per hotel)")

    density = local_density(pool)
    road = fetch_road_travel_times(pool)
    global_comp = compute_global_components(pool, density)
    anchor_comp = {
        aid: compute_anchor_components(pool, a, density, road.get(aid))
        for aid, a in ANCHORS.items()
    }

    hotels_out = {
        h["id"]: {
            "name": h["name"],
            "attributes": {
                "price_lkr": h["price_lkr"],       # cheapest bookable room offer
                "rating": h["rating5"],
                "review_count": h["review_count"],
                "star": h["star"],
                "distance_km": h["distance_km"],   # to the city centre (context only)
                "amenities": h["facilities"],
                "area": h["area"],
                "image": h["image"],
                "description": h["description"],
                "lat": h["lat"],
                "lng": h["lng"],
            },
            "detail": h["detail"],
            "rooms": h["rooms"],
            "components_global": global_comp[h["id"]],
        }
        for h in pool
    }

    pool_ids = [h["id"] for h in pool]

    tasks: List[Dict[str, Any]] = []
    for spec in TASKS_SPEC:
        tasks.append({
            "id": spec["id"],
            "anchor_id": spec["anchor"],
            "persona": spec["persona"],
            "context": spec["context"],
            "primary_dimension": spec["primary"],
            "secondary_dimension": spec["secondary"],
            "repeat_of": spec.get("repeat_of"),
            "is_attention_check": False,
            "option_ids": pool_ids,          # every task shows the whole pool
        })

    # Attention check over the same pool, answer = a named hotel.
    answer = rng.choice(pool_ids)
    tasks.append({
        "id": "t_attn",
        "anchor_id": "fort",
        "persona": "Attention check",
        "context": 'This is an attention check. Please ignore your own preferences and '
                   f'simply select the hotel named "{hotels_out[answer]["name"]}".',
        "primary_dimension": None,
        "secondary_dimension": None,
        "repeat_of": None,
        "is_attention_check": True,
        "attention_answer_hotel_id": answer,
        "option_ids": pool_ids,
    })

    # --- design diagnostics over every alternative shown in every task -------
    seen_rows: List[Dict[str, float]] = []
    for t in tasks:
        ac = anchor_comp[t["anchor_id"]]
        for hid in t["option_ids"]:
            seen_rows.append({**global_comp[hid],
                              "spatial": ac[hid]["spatial"],
                              "accessibility": ac[hid]["accessibility"]})
    mat, worst = correlation_report(seen_rows)

    print(f"\n  Component correlation over {len(seen_rows)} presented alternatives")
    print("                  " + "".join(f"{d[:8]:>11}" for d in DIMENSIONS))
    for i, d in enumerate(DIMENSIONS):
        print(f"    {d:<14}" + "".join(f"{mat[i][j]:>11.2f}" for j in range(len(DIMENSIONS))))
    note = ("columns are near-independent" if worst < 0.5 else
            "two columns are correlated - fine for a labelled ranking dataset, "
            "but do not report their coefficients as separately identified")
    print(f"    largest off-diagonal |r| = {worst:.2f}  ->  {note}")

    # WAVE 2 §2F — a gate, not a warning.
    #
    # Wave 1 fielded a pool with spatial/accessibility at r = +0.907 and
    # facility/economic at -0.699, then spent 268 participants discovering that
    # four of the five weights were unidentifiable. The correlation was printed
    # at build time and nobody stopped. Printing is evidently not enough, so
    # this now refuses to write the material.
    #
    # Passing the gate is a pool-composition problem, not a formula one: pick
    # anchor/time-of-day pairs where distance and travel time diverge, and
    # oversample off-diagonal hotels (cheap-but-well-rated, expensive-but-plain).
    if worst > max_abs_r:
        worst_pair = max(
            ((abs(mat[i][j]), DIMENSIONS[i], DIMENSIONS[j])
             for i in range(len(DIMENSIONS)) for j in range(i + 1, len(DIMENSIONS))),
            key=lambda t: t[0])
        raise SystemExit(
            f"\n  REFUSING TO WRITE MATERIAL: {worst_pair[1]} and {worst_pair[2]} correlate at "
            f"|r| = {worst_pair[0]:.3f} (limit {max_abs_r}).\n"
            "  A wave fielded on this pool cannot identify those two "
            "weights separately -- which is exactly how wave 1 failed.\n"
            "  Fix the POOL (see docs/Weight_Elicitation_Wave2_Design.md 2F), "
            "or pass --max-abs-r to override deliberately.")

    material = {
        "version": f"v4-rooms-{datetime.utcnow().strftime('%Y%m%d')}",
        "city": city,
        "source": "liteapi",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "dimensions": DIMENSIONS,
        "anchor_dimensions": ANCHOR_DIMS,
        "global_dimensions": GLOBAL_DIMS,
        "design": {
            "n_hotels": len(pool),
            "n_tasks": len(tasks),
            "all_options_shown": True,
            "two_stage": True,
            "min_rooms_per_hotel": min_rooms,
            "n_room_offers": n_rooms,
            "n_room_offers_catalogue_joined": n_mapped,
            "stay": {"nights": max(LITEAPI_LOS_NIGHTS, 1), "adults": 2},
            "max_abs_correlation": round(worst, 3),
            "max_abs_correlation_limit": max_abs_r,
            "component_correlation": {
                d: {e: round(mat[i][j], 3) for j, e in enumerate(DIMENSIONS)}
                for i, d in enumerate(DIMENSIONS)
            },
        },
        "anchors": ANCHORS,
        "hotels": hotels_out,
        "anchor_components": anchor_comp,
        "tasks": tasks,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(material, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out}\n  version {material['version']}, {len(pool)} hotels, "
          f"{len(tasks)} tasks, {len(ANCHORS)} anchors")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="Colombo")
    ap.add_argument("--fetch", type=int, default=150, help="hotels to request from LiteAPI")
    ap.add_argument("--pool", type=int, default=60, help="max hotels to keep")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-rooms", type=int, default=2,
                    help="drop hotels offering fewer live room options than this")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    # WAVE 2 §2F. Wave 1 fielded a pool at |r| = 0.907 and could not identify
    # four of five weights. Raising this is a deliberate decision to collect
    # data that cannot answer the question — do it only with a reason.
    ap.add_argument("--max-abs-r", type=float, default=0.7,
                    help="refuse to write material whose components correlate "
                         "above this (default 0.7)")
    args = ap.parse_args()
    if not LITEAPI_KEY:
        raise SystemExit("LITEAPI_KEY not set in repo .env")
    build(args.city, args.fetch, args.pool, args.seed, args.out, args.min_rooms,
          args.max_abs_r)


if __name__ == "__main__":
    main()
