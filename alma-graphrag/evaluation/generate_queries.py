"""
Synthetic query generator for training the learned weight policy (Phase 1).

The hand-written evaluation/queryset.json has only 50 queries — far too few to
train a neural policy without memorising. This builds a large, varied query pool
by crossing category templates with slot values and multiple paraphrases, each
carrying a gold spec (the same constraint dict evaluation/gold.py grades against).

The curated queryset.json stays the reported test set; this file produces the
TRAINING pool (queryset_synth.json). Deterministic given --seed.

Usage:
    python evaluation/generate_queries.py --n 500
    python evaluation/generate_queries.py --n 800 --seed 7 --out evaluation/queryset_synth.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

Query = Tuple[str, Dict[str, Any], str]  # (question, gold, category)

# --- Slot values -----------------------------------------------------------
MAX_PRICES = [15000, 18000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 60000, 70000]
PRICE_RANGES = [(20000, 40000), (30000, 60000), (40000, 70000), (25000, 50000)]
MIN_PRICES = [70000, 80000, 100000]
RATINGS = [3.8, 4.0, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8]
STARS = [4, 5]
TIMES = [3, 4, 5, 6, 7, 8]
AMENITIES = ["air conditioning", "hot water", "front desk"]


def _economic() -> List[Query]:
    out: List[Query] = []
    max_phr = [
        "cheap hotels under {p} lkr",
        "budget stays below {p} rupees",
        "affordable places under rs {p}",
        "somewhere cheap to stay, less than {p}",
        "hotels costing at most {p} lkr",
        "value for money hotels under {p}",
        "inexpensive rooms below {p} a night",
        "wallet friendly hotels below {p}",
        "hotels i can afford under {p} lkr",
    ]
    for p in MAX_PRICES:
        for t in max_phr:
            out.append((t.format(p=p), {"max_price": p}, "economic"))
    for lo, hi in PRICE_RANGES:
        for t in ["mid-range hotels between {lo} and {hi} lkr",
                  "hotels priced from {lo} to {hi} rupees"]:
            out.append((t.format(lo=lo, hi=hi), {"min_price": lo, "max_price": hi}, "economic"))
    for p in MIN_PRICES:
        for t in ["upscale hotels above {p} a night", "premium hotels over {p} lkr"]:
            out.append((t.format(p=p), {"min_price": p}, "economic"))
    return out


def _quality() -> List[Query]:
    out: List[Query] = []
    rating_phr = [
        "top-rated hotels above {r}",
        "hotels rated at least {r}",
        "excellent hotels rated {r} or better",
        "well reviewed places rated over {r}",
        "highly rated hotels over {r}",
        "guest favourite hotels rated {r} or higher",
    ]
    for r in RATINGS:
        for t in rating_phr:
            out.append((t.format(r=r), {"min_rating": r}, "quality"))
    star_phr = [
        "{s} star hotels",
        "luxury {s} star properties",
        "{s}-star or better hotels",
    ]
    for s in STARS:
        for t in star_phr:
            out.append((t.format(s=s), {"min_star": s}, "quality"))
    return out


def _accessibility() -> List[Query]:
    out: List[Query] = []
    phr = [
        "hotels within {t} minutes drive from the city",
        "quick access hotels under {t} min commute",
        "easily reachable hotels within {t} minutes",
        "hotels with short {t} minute drive from the centre",
        "hotels close to the centre, at most {t} minutes away",
        "minimal commute hotels under {t} minutes",
    ]
    for t in TIMES:
        for ph in phr:
            out.append((ph.format(t=t), {"max_travel_time": float(t)}, "accessibility"))
    # phrasing without an explicit number (defaults to a tight bound)
    for ph in ["hotels with quick easy access and low travel time",
               "places with the shortest drive from the centre",
               "hotels that are effortless to get to"]:
        out.append((ph, {"max_travel_time": 5.0}, "accessibility"))
    return out


def _amenity() -> List[Query]:
    out: List[Query] = []
    for a in AMENITIES:
        for t in ["hotels with {a}", "{a} hotels", "places that have {a}",
                  "hotels offering {a}", "need a room with {a}",
                  "stays where {a} is included"]:
            out.append((t.format(a=a), {"required_amenities": [a]}, "amenity"))
        for p in [30000, 40000, 50000]:
            for t in ["{a} hotels under {p}", "hotels with {a} below {p} lkr"]:
                out.append((t.format(a=a, p=p),
                            {"required_amenities": [a], "max_price": p}, "amenity"))
    return out


def _disruption() -> List[Query]:
    out: List[Query] = []
    phr = [
        "hotels that stay reachable in heavy traffic under {t} min",
        "avoid congestion, hotels easy to get to within {t} minutes",
        "hotels with a stable eta under traffic below {t} min",
        "hotels reachable despite rush hour in {t} minutes or less",
        "hotels with reliable travel time under {t} min even with roadworks",
        "hotels not slowed by traffic, within {t} minutes",
    ]
    for t in [5, 6, 7, 8]:
        for ph in phr:
            out.append((ph.format(t=t), {"max_travel_time": float(t)}, "disruption"))
    for ph in ["hotels not stuck in congestion", "quiet hotels away from traffic jams",
               "hotels where the drive does not blow up in rush hour",
               "stays unaffected by road closures and jams"]:
        out.append((ph, {"max_travel_time": 6.0}, "disruption"))
    return out


def _multi() -> List[Query]:
    out: List[Query] = []
    # price + rating
    for p in [25000, 30000, 40000, 45000, 60000]:
        for r in [4.0, 4.3, 4.4]:
            for t in ["affordable hotels under {p} rated above {r}",
                      "good hotels below {p} with rating {r} or higher"]:
                out.append((t.format(p=p, r=r), {"max_price": p, "min_rating": r}, "multi_dimensional"))
    # price + travel
    for p in [25000, 30000, 35000, 45000]:
        for tt in [6, 7]:
            for t in ["budget hotels below {p} easy to reach within {tt} minutes",
                      "cheap hotels under {p} with quick {tt} min access"]:
                out.append((t.format(p=p, tt=tt), {"max_price": p, "max_travel_time": float(tt)}, "multi_dimensional"))
    # rating + travel
    for r in [4.2, 4.3, 4.5]:
        for tt in [5, 6, 7]:
            for t in ["top rated hotels above {r} within {tt} minutes drive",
                      "well rated hotels rated {r}+ with fast {tt} min access"]:
                out.append((t.format(r=r, tt=tt), {"min_rating": r, "max_travel_time": float(tt)}, "multi_dimensional"))
    # star + travel
    for s in STARS:
        for tt in [6, 7]:
            out.append((f"luxury {s} star hotels with quick access under {tt} minutes",
                        {"min_star": s, "max_travel_time": float(tt)}, "multi_dimensional"))
    # price + rating + travel
    for p in [45000, 60000]:
        for r in [4.2, 4.4]:
            for tt in [7, 8]:
                out.append((f"good hotels under {p} rated at least {r} within {tt} minutes",
                            {"max_price": p, "min_rating": r, "max_travel_time": float(tt)}, "multi_dimensional"))
    # rating + amenity
    for r in [4.2, 4.4]:
        out.append((f"well rated air conditioned hotels above {r}",
                    {"min_rating": r, "required_amenities": ["air conditioning"]}, "multi_dimensional"))
    return out


def _location_variants(q: Query) -> List[Query]:
    """Add a natural ' in colombo' variant for extra lexical variety — but only
    where the phrasing doesn't already reference the city/centre (avoids awkward
    'from the city in colombo')."""
    question, gold, cat = q
    variants = [q]
    if not any(w in question for w in ("colombo", "city", "centre", "center")):
        variants.append((f"{question} in colombo", gold, cat))
    return variants


def build_pool() -> List[Query]:
    pool: List[Query] = []
    for fn in (_economic, _quality, _accessibility, _amenity, _disruption, _multi):
        for q in fn():
            pool.extend(_location_variants(q))
    # Dedupe by question text (first occurrence wins).
    seen = set()
    unique: List[Query] = []
    for q in pool:
        if q[0] in seen:
            continue
        seen.add(q[0])
        unique.append(q)
    return unique


def _gold_sizes(pool: List[Query], city: str) -> Dict[str, int]:
    """Rule-gold set size per query question over the LIVE hotel pool.
    Small gold sets make hard, discriminative queries — when most of the pool
    is relevant, every system saturates P@K/nDCG and differences become noise.
    Requires Neo4j."""
    from evaluation.baselines import fetch_city_hotels
    from evaluation.gold import relevant_set

    hotels = fetch_city_hotels(city)
    if not hotels:
        raise SystemExit(f"--filter-gold needs the live pool but {city} has no hotels in Neo4j")
    return {q[0]: len(relevant_set(hotels, q[1])) for q in pool}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic query pool")
    ap.add_argument("--n", type=int, default=500, help="number of queries to sample")
    ap.add_argument("--per-cat", type=int, default=0,
                    help="exact queries per category (balanced eval-set mode; overrides --n)")
    ap.add_argument("--filter-gold", default=None, metavar="LO,HI",
                    help="keep only queries with LO..HI relevant hotels against the "
                         "live Neo4j pool (harder, discriminative queries)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--city", default="Colombo")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--id-prefix", default="s")
    ap.add_argument("--out", default="evaluation/queryset_synth.json")
    args = ap.parse_args()

    pool = build_pool()
    rng = random.Random(args.seed)

    sizes: Dict[str, int] = {}
    lo, hi = 1, 10**9
    if args.filter_gold:
        lo, hi = (int(x) for x in args.filter_gold.split(","))
        sizes = _gold_sizes(pool, args.city)
        n0 = len(pool)
        pool = [q for q in pool if sizes[q[0]] >= lo]  # drop degenerate golds
        print(f"Gold-size floor >= {lo}: {n0} -> {len(pool)} candidates "
              f"(target band [{lo},{hi}])")

    by_cat: Dict[str, List[Query]] = defaultdict(list)
    for q in pool:
        by_cat[q[2]].append(q)
    for lst in by_cat.values():
        rng.shuffle(lst)
    cats = sorted(by_cat)

    if args.per_cat:
        # Balanced eval-set mode: exactly per_cat queries from every category.
        # Prefer queries inside the target gold-size band; when a category
        # cannot fill from the band (pool attributes too homogeneous), fall
        # back to its hardest (smallest-gold) remaining queries. Sampling from
        # the hardest 2x window keeps slot/phrasing diversity.
        chosen = []
        for c in cats:
            cands = by_cat[c]
            if len(cands) < args.per_cat:
                raise SystemExit(
                    f"category '{c}' has only {len(cands)} candidates after "
                    f"filtering — cannot draw {args.per_cat}; relax --filter-gold "
                    f"or add templates"
                )
            if sizes:
                cands = sorted(cands, key=lambda q: (sizes[q[0]] > hi, sizes[q[0]]))
                window = cands[: max(args.per_cat * 2, args.per_cat)]
                picked = rng.sample(window, args.per_cat)
                n_in_band = sum(1 for q in picked if sizes[q[0]] <= hi)
                if n_in_band < args.per_cat:
                    print(f"  note: {c} filled {args.per_cat - n_in_band} queries "
                          f"outside the [{lo},{hi}] band (hardest available)")
            else:
                picked = cands[: args.per_cat]
            chosen.extend(picked)
        rng.shuffle(chosen)
    else:
        # Stratified round-robin sample so no single category dominates the
        # training pool (economic templates are otherwise ~40% of it). Small
        # categories exhaust first, so tails skew towards large ones.
        cursors = {c: 0 for c in cats}
        chosen = []
        while len(chosen) < args.n and any(cursors[c] < len(by_cat[c]) for c in cats):
            for c in cats:
                if cursors[c] < len(by_cat[c]):
                    chosen.append(by_cat[c][cursors[c]])
                    cursors[c] += 1
                    if len(chosen) >= args.n:
                        break

    queries = [
        {"id": f"{args.id_prefix}{ i+1 :04d}", "question": q, "gold": g, "category": c}
        for i, (q, g, c) in enumerate(chosen)
    ]
    mode = (
        f"balanced eval set ({args.per_cat}/category"
        + (f", gold-size {args.filter_gold}" if args.filter_gold else "")
        + ")"
        if args.per_cat
        else "training pool"
    )
    spec = {
        "description": (
            f"Synthetic {mode}: {len(queries)} queries, seed={args.seed}. "
            "Templated category x slot x paraphrase; gold is graded by evaluation/gold.py."
        ),
        "city": args.city,
        "k": args.k,
        "queries": queries,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    counts = Counter(c for _, _, c in chosen)
    print(f"Pool built: {len(pool)} unique templates; sampled {len(queries)} (seed={args.seed})")
    print("By category:")
    for cat, n in sorted(counts.items()):
        print(f"  {cat:<18s} {n}")
    print(f"\nWritten to {out_path}")
    print("Samples:")
    for q in queries[:5]:
        print(f"  [{q['category']}] {q['question']}  -> {q['gold']}")


if __name__ == "__main__":
    main()
