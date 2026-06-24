"""
Script: One-shot data-quality fixes for the existing knowledge graph.

Cleans up two issues that corrupt retrieval/evaluation on already-ingested data:

  1. City canonicalisation — merges fragmented/aliased City nodes
     ("Colombo 03", "colombo", "Colombo 7") into one canonical node and rewires
     the LOCATED_IN edges.
  2. Placeholder prices — Google-Places hotels carry a fabricated uniform
     price (legacy default of 3,500 LKR from an absent price_level). These are
     nulled and flagged so the economic ranking signal is honest (the retriever
     then treats unknown prices as neutral instead of "cheapest").

It also reports duplicate hotels (same name within a city, typically a Google
place_id node + a LiteAPI hotelId node) as a diagnostic — these are NOT merged
automatically (merging hotel relationships is risky); review them manually.

Usage:
    python scripts/fix_data_quality.py                 # apply all fixes
    python scripts/fix_data_quality.py --dry-run       # report only, no writes
    python scripts/fix_data_quality.py --no-null-google-prices  # keep est. prices
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import logging
from collections import defaultdict

from neo4j import GraphDatabase

from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from src.ingest.canonicalize import canonical_city

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alma.scripts.fix_data")

# Legacy Google price_level -> LKR defaults that were fabricated for hotels with
# no real price_level. Used to detect & null placeholder prices.
_GOOGLE_PLACEHOLDER_PRICES = {2000.0, 3500.0, 8000.0, 9000.0, 20000.0, 40000.0}


def merge_city_aliases(session, dry_run: bool) -> int:
    """Merge City nodes that share a canonical name; rewire LOCATED_IN edges."""
    names = [r["name"] for r in session.run("MATCH (c:City) RETURN c.name AS name")]
    groups: dict[str, list[str]] = defaultdict(list)
    for name in names:
        groups[canonical_city(name)].append(name)

    merged = 0
    for canon, variants in groups.items():
        for old in variants:
            if old == canon:
                continue
            print(f"  merge City '{old}' -> '{canon}'")
            merged += 1
            if dry_run:
                continue
            # Rewire every LOCATED_IN edge from old city to the canonical city,
            # preserving edge properties, then drop the orphaned alias node.
            session.run(
                """
                MATCH (h:Hotel)-[r:LOCATED_IN]->(old:City {name: $old})
                MERGE (canon:City {name: $canon})
                MERGE (h)-[nr:LOCATED_IN]->(canon)
                SET nr += properties(r)
                DELETE r
                """,
                {"old": old, "canon": canon},
            )
            session.run(
                """
                MATCH (old:City {name: $old})
                WHERE NOT (old)<-[:LOCATED_IN]-()
                DETACH DELETE old
                """,
                {"old": old},
            )
    return merged


def fix_placeholder_prices(session, dry_run: bool, null_google: bool) -> int:
    """Flag Google prices as estimated and (optionally) null placeholder values."""
    # Mark all Google-Places hotels' prices as estimated.
    if not dry_run:
        session.run(
            "MATCH (h:Hotel) WHERE h.source = 'google_places' SET h.price_estimated = true"
        )

    # Count hotels carrying a fabricated placeholder price.
    rows = session.run(
        """
        MATCH (h:Hotel)
        WHERE h.source = 'google_places' AND h.price_per_night_lkr IN $vals
        RETURN count(h) AS n
        """,
        {"vals": list(_GOOGLE_PLACEHOLDER_PRICES)},
    ).single()
    n = int((rows or {}).get("n", 0))

    if null_google and not dry_run and n:
        session.run(
            """
            MATCH (h:Hotel)
            WHERE h.source = 'google_places' AND h.price_per_night_lkr IN $vals
            SET h.price_per_night_lkr = null, h.price_range = null, h.price_estimated = true
            """,
            {"vals": list(_GOOGLE_PLACEHOLDER_PRICES)},
        )
    return n


def _duplicate_groups(session) -> list[dict]:
    return session.run(
        """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)
        WITH toLower(trim(h.name)) AS nm, c.name AS city,
             collect({id: h.id, source: h.source, price: h.price_per_night_lkr}) AS hs
        WHERE size(hs) > 1
        RETURN nm AS name, city, hs
        ORDER BY city, name
        """
    ).data()


def report_duplicates(session) -> int:
    """Report hotels with the same name in the same city (multi-source dupes)."""
    rows = _duplicate_groups(session)
    for r in rows:
        sources = ", ".join(f"{h['source']}({h['price']})" for h in r["hs"])
        print(f"  DUP '{r['name']}' in {r['city']}: {sources}")
    return len(rows)


def dedupe_hotels(session, dry_run: bool) -> int:
    """For each duplicate name/city group, keep the best node and delete the rest.

    "Best" = the node with a real (non-null) price, preferring source='liteapi'
    (live rates + room/board data) over 'google_places' (coarse estimate). The
    losing duplicates are DETACH DELETEd. Conservative: only acts on exact
    name+city matches.
    """
    groups = _duplicate_groups(session)
    deleted = 0
    for g in groups:
        hs = g["hs"]

        def rank(h: dict) -> tuple:
            has_price = 1 if h.get("price") else 0
            is_lite = 1 if h.get("source") == "liteapi" else 0
            return (has_price, is_lite)

        keep = max(hs, key=rank)
        losers = [h for h in hs if h["id"] != keep["id"]]
        print(f"  dedupe '{g['name']}' in {g['city']}: keep {keep['source']}({keep['price']}), "
              f"delete {[h['source'] for h in losers]}")
        deleted += len(losers)
        if dry_run:
            continue
        session.run(
            "MATCH (h:Hotel) WHERE h.id IN $ids DETACH DELETE h",
            {"ids": [h["id"] for h in losers]},
        )
    return deleted


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix data-quality issues in the ALMA graph")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    parser.add_argument(
        "--no-null-google-prices",
        action="store_true",
        help="Keep estimated Google prices instead of nulling placeholders",
    )
    parser.add_argument(
        "--dedupe-hotels",
        action="store_true",
        help="Delete duplicate hotel nodes (same name+city), keeping the real-priced one",
    )
    args = parser.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            print("\n=== 1. City canonicalisation ===")
            merged = merge_city_aliases(session, args.dry_run)
            print(f"  -> {merged} city node(s) merged")

            print("\n=== 2. Placeholder prices ===")
            n = fix_placeholder_prices(
                session, args.dry_run, null_google=not args.no_null_google_prices
            )
            action = "would null" if (args.dry_run or args.no_null_google_prices) else "nulled"
            print(f"  -> {n} Google placeholder price(s) {action}; all Google prices flagged estimated")

            if args.dedupe_hotels:
                print("\n=== 3. Deduplicate hotels (keep real-priced node) ===")
                deleted = dedupe_hotels(session, args.dry_run)
                print(f"  -> {deleted} duplicate hotel node(s) deleted")
            else:
                print("\n=== 3. Duplicate hotels (diagnostic only; use --dedupe-hotels) ===")
                dups = report_duplicates(session)
                print(f"  -> {dups} duplicate name/city group(s) found")

            print("\n=== Resulting City nodes ===")
            for r in session.run("MATCH (h:Hotel)-[:LOCATED_IN]->(c:City) RETURN c.name AS city, count(h) AS n ORDER BY n DESC").data():
                print(f"  {r['city']}: {r['n']} hotels")

        if args.dry_run:
            print("\n(DRY RUN — no changes written)")
        else:
            print("\nData-quality fixes applied.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
