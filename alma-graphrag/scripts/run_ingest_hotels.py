import sys
from pathlib import Path

# Ensure repo root is on sys.path so `src` imports work when running scripts directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from src.config import DEFAULT_CITY, HOTEL_MAX_RESULTS, LITEAPI_ENABLED
from src.ingest.hotel_ingest import ingest_hotels, ingest_new_liteapi_hotels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default=DEFAULT_CITY)
    parser.add_argument(
        "--cities",
        help="Comma-separated list of cities. Overrides --city.",
        default="",
    )
    parser.add_argument("--max", type=int, default=HOTEL_MAX_RESULTS)
    parser.add_argument(
        "--new-liteapi",
        type=int,
        metavar="COUNT",
        help="discover and ingest up to COUNT LiteAPI catalogue hotels not already in the graph",
    )
    parser.add_argument(
        "--source",
        choices=["google", "liteapi", "both"],
        default="both" if LITEAPI_ENABLED else "google",
        help="Hotel data source to ingest from.",
    )
    args = parser.parse_args()

    city_list = [args.city]
    if args.cities:
        city_list = [c.strip() for c in args.cities.split(",") if c.strip()]

    if args.new_liteapi is not None:
        if len(city_list) != 1:
            parser.error("--new-liteapi requires exactly one city")
        total = ingest_new_liteapi_hotels(city_list[0], args.new_liteapi)
        print(f"New LiteAPI hotels ingested for {city_list[0]}: {total}.")
    else:
        total = ingest_hotels(city_list, max_results=args.max, source=args.source)
        print(f"Total hotels ingested from '{args.source}': {total}.")


if __name__ == "__main__":
    main()
