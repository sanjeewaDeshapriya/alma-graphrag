import sys
from pathlib import Path

# Ensure repo root is on sys.path so `src` imports work when running scripts directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from src.config import DEFAULT_CITY, HOTEL_MAX_RESULTS
from src.ingest.hotel_ingest import ingest_hotels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default=DEFAULT_CITY)
    parser.add_argument(
        "--cities",
        help="Comma-separated list of cities. Overrides --city.",
        default="",
    )
    parser.add_argument("--max", type=int, default=HOTEL_MAX_RESULTS)
    args = parser.parse_args()

    city_list = [args.city]
    if args.cities:
        city_list = [c.strip() for c in args.cities.split(",") if c.strip()]

    total = ingest_hotels(city_list, max_results=args.max)
    print(f"Total hotels ingested: {total}.")


if __name__ == "__main__":
    main()
