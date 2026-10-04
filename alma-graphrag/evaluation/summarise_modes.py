"""Side-by-side summary of graph-only (unweighted) and weighted GraphRAG.

Reads the result artifacts written by the SCORING_MODE runs and writes
evaluation/results_modes_summary.json and .md, the single source for the two
papers (paper 1: unweighted GraphRAG; paper 2: weighted GraphRAG).

    python -m evaluation.summarise_modes

Inputs (all on the 77-hotel Colombo pool):
    results_unweighted_main.json / _price.json   (run_eval --compare-modes)
    results_robustness_unweighted.json / results_robustness_77_weighted.json
    results_human_unweighted.json, results_human.json
    results_human_seeds_unweighted.json
    results_disruption_grid_unweighted.json / results_disruption_grid_77_weighted.json
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

EV = Path(__file__).resolve().parent
U, W, F = "GraphRAG[unweighted]", "WeightedGraphRAG", "Filter"
TEXT = ["Keyword", "SemanticVec", "Hybrid", "CrossEncoder"]


def _load(name):
    p = EV / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _wtl(pq, a, b):
    nd = lambda q, s: q["scores"][s]["nDCG@10"]
    w = sum(nd(q, a) > nd(q, b) + 1e-9 for q in pq)
    lo = sum(nd(q, a) < nd(q, b) - 1e-9 for q in pq)
    diff = [nd(q, a) - nd(q, b) for q in pq]
    sd = st.pstdev(diff)
    return {"wins": w, "ties": len(pq) - w - lo, "losses": lo,
            "d_z": round(st.mean(diff) / sd, 2) if sd else None}


def benchmark(res):
    ov, sig, pq = res["overall"], res["significance"]["vs"], res["per_query"]
    rows = {}
    for s in res["system_order"]:
        row = {m: round(ov[s][m], 3) for m in ("P@10", "R@10", "nDCG@10", "MRR")}
        row["top1"] = round(sum(q["scores"][s]["MRR"] == 1 for q in pq) / len(pq), 3)
        if s in sig:
            v = sig[s]
            row["vs_unweighted"] = {"diff": round(v["mean_diff"], 3),
                                    "ci": [round(v["ci_low"], 3), round(v["ci_high"], 3)],
                                    "p_holm": round(v["p_holm"], 4),
                                    "significant": v["significant"]}
        rows[s] = row
    comps = {b: {**_wtl(pq, U, b),
                 "relative_gain_pct": round(100 * (ov[U]["nDCG@10"] / ov[b]["nDCG@10"] - 1), 1)}
             for b in [F] + TEXT + [W] if b in ov}
    by_cat = {c: {s: round(v[s]["nDCG@10"], 3) for s in (F, "Keyword", U, W) if s in v}
              for c, v in res.get("by_category", {}).items()}
    return {"n_queries": res["n_queries"], "pool": res["pool_size"], "systems": rows,
            "unweighted_vs": comps, "weighted_vs_unweighted": _wtl(pq, W, U),
            "by_category": by_cat}


def ablation(res):
    if not res:
        return None
    out = {}
    for name, block in res["sets"].items():
        out[name] = {r["system"]: {"ndcg": r["ndcg"], "delta": r.get("delta"),
                                   "significant": r.get("significant")}
                     for r in block["ablation"]["rows"]}
        simplex = block.get("simplex", {})
        if "quantiles" in simplex:
            out[name]["_simplex"] = {"median": simplex["quantiles"]["median"],
                                     "share_above_filter": simplex.get("share_above_filter"),
                                     "handset_percentile": simplex.get("rank_of_handset_pct")}
    return out


def disruption(res):
    if not res:
        return None
    out = {}
    for level, run in res["runs"].items():
        pooled = run.get("pooled", {})
        live = pooled.get("reference_live_vs_stale", {})
        vs = pooled.get("ndcg_vs_reference", {})
        out[level] = {
            "live_minus_stale": round(live.get("mean_diff", 0.0), 4) if live else None,
            "live_vs_stale_p": live.get("p"),
            "vs_filter": ({"diff": round(vs[F]["mean_diff"], 4), "p_holm": vs[F].get("p_holm"),
                           "significant": vs[F].get("significant")} if F in vs else None),
        }
    return {"reference": res.get("reference"), "levels": out}


def human(res_u, res_w, seeds):
    out = {}
    if res_u:
        pc = res_u["per_choice"]
        out["unweighted_split"] = {s: {"choice_ndcg": round(v["nDCG@10"], 3),
                                       "hit10": round(v["R@10"], 3),
                                       "mean_rank": round(v["mean_rank"], 2)}
                                   for s, v in pc.items()}
    if res_w:
        v = res_w["per_choice"].get("WeightedGraphRAG[handset]")
        if v:
            out["weighted_handset_split"] = {"choice_ndcg": round(v["nDCG@10"], 3),
                                             "hit10": round(v["R@10"], 3),
                                             "mean_rank": round(v["mean_rank"], 2)}
    if seeds:
        out["fifty_splits"] = {s: {"choice_ndcg": seeds["choice_ndcg"][s]["mean"],
                                   "sd": seeds["choice_ndcg"][s]["sd"],
                                   "mean_rank": seeds["mean_rank"][s]["mean"]}
                               for s in seeds["systems"]}
        out["fifty_splits"]["unweighted_minus_handset"] = seeds.get(
            "unweighted_minus_handset_choice_ndcg")
    return out


def main():
    main_res, price_res = _load("results_unweighted_main.json"), _load("results_unweighted_price.json")
    summary = {
        "main": benchmark(main_res) if main_res else None,
        "price": benchmark(price_res) if price_res else None,
        "ablation": {"unweighted": ablation(_load("results_robustness_unweighted.json")),
                     "weighted": ablation(_load("results_robustness_77_weighted.json"))},
        "disruption": {"unweighted": disruption(_load("results_disruption_grid_unweighted.json")),
                       "weighted": disruption(_load("results_disruption_grid_77_weighted.json"))},
        "human_choice": human(_load("results_human_unweighted.json"), _load("results_human.json"),
                              _load("results_human_seeds_unweighted.json")),
    }
    (EV / "results_modes_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["# Unweighted vs weighted GraphRAG (77-hotel Colombo pool)", ""]
    for key, title in (("main", "Main benchmark (60 queries)"), ("price", "Price slice (20 queries)")):
        b = summary[key]
        if not b:
            continue
        lines += [f"## {title}", "", "| System | P@10 | R@10 | nDCG@10 | MRR | Top-1 |",
                  "|---|---|---|---|---|---|"]
        for s, r in b["systems"].items():
            lines.append(f"| {s} | {r['P@10']:.3f} | {r['R@10']:.3f} | {r['nDCG@10']:.3f} | "
                         f"{r['MRR']:.3f} | {r['top1']:.3f} |")
        lines += ["", "| Unweighted vs | W/T/L | d_z | Relative gain | Diff [95% CI], Holm p |",
                  "|---|---|---|---|---|"]
        for s, c in b["unweighted_vs"].items():
            v = b["systems"][s].get("vs_unweighted", {})
            # significance.vs reports reference (unweighted) minus system
            ci = f"{v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}], p={v['p_holm']}" if v else "-"
            lines.append(f"| {s} | {c['wins']}/{c['ties']}/{c['losses']} | {c['d_z']} | "
                         f"{c['relative_gain_pct']:+.1f}% | {ci} |")
        wv = b["weighted_vs_unweighted"]
        lines += ["", f"Weighted vs unweighted, per query: {wv['wins']} wins / {wv['ties']} "
                      f"ties / {wv['losses']} losses (d_z {wv['d_z']}).", ""]
        if b["by_category"]:
            lines += ["| Category | Filter | Keyword | Unweighted | Weighted |", "|---|---|---|---|---|"]
            for c, v in b["by_category"].items():
                lines.append(f"| {c} | " + " | ".join(f"{v.get(s, float('nan')):.3f}"
                                                        for s in (F, "Keyword", U, W)) + " |")
            lines.append("")
    lines += ["## Ablations (nDCG@10)", ""]
    for mode, abl in summary["ablation"].items():
        if not abl:
            continue
        for set_name, rows in abl.items():
            lines.append(f"**{mode}, {set_name}:** " + "; ".join(
                f"{k} {v['ndcg']:.3f}" + (f" ({v['delta']:+.3f}{'*' if v['significant'] else ''})"
                                          if v.get("delta") is not None else "")
                for k, v in rows.items() if not k.startswith("_")))
            if "_simplex" in rows:
                lines.append(f"  simplex: {rows['_simplex']}")
            lines.append("")
    grid = (_load("results_disruption_grid_unweighted.json") or {}).get("grid", {})
    lines += [f"## Controlled disruption ({grid.get('centres', '?')} closures, "
              f"{grid.get('spacing_km', '?')} km grid)", ""]
    for mode, dis in summary["disruption"].items():
        if dis:
            lines.append(f"**{mode}** (reference {dis['reference']}): {json.dumps(dis['levels'])}")
            lines.append("")
    lines += ["## Held-out booking choices", "", "```", json.dumps(summary["human_choice"], indent=2),
              "```", "", "`*` = significant after Holm correction. Paired bootstrap, 5,000 resamples."]
    (EV / "results_modes_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
