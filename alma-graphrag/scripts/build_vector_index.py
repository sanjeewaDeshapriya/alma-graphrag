"""
Build the pgvector search index from the Neo4j hotel graph.

Fetches each city's hotels, builds a text document, embeds it with the local
sentence-transformers model, and upserts keyword (tsvector) + semantic
(embedding) rows into Postgres/pgvector.

The document contains EVERYTHING the graph systems can see, verbalised: name,
description, amenities, attractions, AND the structured attributes (price,
rating, star class, travel time) as natural-language sentences. Without the
attribute sentences the text baselines are structurally blind to the very
constraints the gold standard grades on (price/rating/travel time), which
rigs the comparison — a keyword system scoring 0.0 on economic queries says
nothing about keyword search when price was never in its input.

Requires Neo4j (hotel data) and the pgvector container to be up.

Usage:
    python scripts/build_vector_index.py --cities Colombo
    python scripts/build_vector_index.py --cities Colombo,Piliyandala
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from typing import Any, Dict

from src.search import vector_store as vs
from src.search.embedder import embed


def _price_words(price: float) -> str:
    """Qualitative price band, so 'cheap'/'budget'/'luxury' queries can match."""
    if price < 20000:
        return "cheap budget affordable inexpensive"
    if price < 40000:
        return "affordable mid-range value"
    if price < 70000:
        return "mid-range comfortable"
    return "premium luxury upscale expensive"


def _attr_text(h: Dict[str, Any]) -> str:
    """Verbalise the structured attributes the gold standard grades on."""
    parts = []
    price = h.get("price")
    if price is not None:
        parts.append(
            f"Price {int(price)} rupees per night, {_price_words(float(price))}."
        )
    rating = h.get("rating")
    if rating is not None:
        r = float(rating)
        quality = "excellent highly rated top rated" if r >= 4.5 else (
            "well rated good reviews" if r >= 4.0 else "average rated")
        parts.append(f"Guest rating {r:.1f} out of 5, {quality}.")
    star = h.get("star")
    if star is not None:
        s = int(star)
        parts.append(f"{s} star hotel" + (", luxury class." if s >= 5 else "."))
    tt = h.get("travel_time_traffic_min")
    if tt is None:
        tt = h.get("travel_time_min")
    if tt is not None:
        t = float(tt)
        access = "very quick easy access short drive low travel time" if t <= 5 else (
            "reachable moderate drive" if t <= 10 else "long drive far from the centre")
        parts.append(f"About {int(round(t))} minutes from the city centre in traffic, {access}.")
    return " ".join(parts)


def _doc(h: Dict[str, Any]) -> str:
    parts = [
        h.get("name") or "",
        h.get("description") or "",
        " ".join(h.get("amenities") or []),
        " ".join(h.get("attractions") or []),
        _attr_text(h),
    ]
    return " ".join(p for p in parts if p).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build pgvector search index")
    ap.add_argument("--cities", default="Colombo")
    args = ap.parse_args()
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]

    # Imported here so Neo4j is only touched when actually building.
    from evaluation.baselines import fetch_city_hotels

    print(f"Model: {vs.__name__} · initialising schema…")
    vs.init_schema()

    total = 0
    for city in cities:
        hotels = fetch_city_hotels(city)
        if not hotels:
            print(f"  {city}: no hotels found in graph — skipping")
            continue
        docs = [_doc(h) for h in hotels]
        print(f"  {city}: embedding {len(hotels)} hotels…")
        embs = embed(docs)
        rows = [
            {
                "id": str(h["id"]),
                "city": city,
                "name": h.get("name") or str(h["id"]),
                "doc": docs[i],
                "embedding": embs[i],
            }
            for i, h in enumerate(hotels)
        ]
        vs.index_hotels(rows)
        total += len(rows)
        print(f"  {city}: indexed {len(rows)} hotels")

    print(f"\nDone. {total} hotels indexed in pgvector (table '{vs.TABLE}').")


if __name__ == "__main__":
    main()
