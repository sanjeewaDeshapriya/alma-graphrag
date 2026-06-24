"""
Entity canonicalisation helpers (research gap RQ1 — entity canonicalisation).

Currently focused on city names: ingestion sources return inconsistent city
strings ("Colombo 03", "colombo", "Colombo 7", "Piliyandala ") that fragment the
City node space. ``canonical_city`` normalises these to a single canonical form
so hotels group correctly and city-scoped retrieval/evaluation is stable.
"""
from __future__ import annotations

import re
from typing import Optional

# Explicit alias map (lowercase key -> canonical display name).
_CITY_ALIASES = {
    "col": "Colombo",
    "colombo city": "Colombo",
    "mt lavinia": "Mount Lavinia",
    "mt. lavinia": "Mount Lavinia",
    "mount-lavinia": "Mount Lavinia",
    "nuwaraeliya": "Nuwara Eliya",
    "nuwara-eliya": "Nuwara Eliya",
    "nuwara eliya city": "Nuwara Eliya",
    "kandy city": "Kandy",
    "galle city": "Galle",
    "negombo city": "Negombo",
    "bentota beach": "Bentota",
    "trinco": "Trincomalee",
}

# Cities that use numbered postal zones (e.g. "Colombo 03") which we collapse to
# the base city for city-level grouping.
_ZONED_CITIES = {"colombo", "dehiwala", "mount lavinia"}

# Multi-word city names that should keep specific capitalisation.
_SPECIAL_CASE = {
    "nuwara eliya": "Nuwara Eliya",
    "mount lavinia": "Mount Lavinia",
    "sri jayawardenepura kotte": "Sri Jayawardenepura Kotte",
}


def _titlecase(s: str) -> str:
    return " ".join(w.capitalize() for w in s.split())


def canonical_city(name: Optional[str]) -> Optional[str]:
    """Normalise a raw city string to a canonical City node name.

    - Trims/collapses whitespace.
    - Strips trailing postal codes ("Colombo, 00300" -> "Colombo").
    - Collapses numbered zones for known zoned cities ("Colombo 03" -> "Colombo").
    - Applies an alias map and consistent title-casing.

    Returns the input unchanged if it is empty/None.
    """
    if not name:
        return name

    s = re.sub(r"\s+", " ", str(name).strip())
    low = s.lower()

    # Strip a trailing postal code: ", 00300" / " 80000".
    low = re.sub(r"[,\s]+\d{4,6}$", "", low).strip()

    # Collapse a numbered zone ("colombo 03", "colombo 7") for zoned cities.
    m = re.match(r"^(.*?)[\s-]+\d{1,3}$", low)
    if m and m.group(1).strip() in _ZONED_CITIES:
        low = m.group(1).strip()

    if low in _CITY_ALIASES:
        return _CITY_ALIASES[low]
    if low in _SPECIAL_CASE:
        return _SPECIAL_CASE[low]
    return _titlecase(low)
