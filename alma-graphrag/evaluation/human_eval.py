"""
Choice-based evaluation — ground truth from real people, not from rules.

WHY THIS EXISTS ALONGSIDE harness.py
------------------------------------
`harness.py` grades against evaluation/gold.py: rules the project author wrote.
That evaluation answers "does the system satisfy the stated constraints?" — but
it is nearly blind to ranking. Hard filtering in `_apply_filters` prunes on the
same predicates the gold grades on, so 45 of 60 queries score identically across
every composite-weight configuration, and the profile spread (0.004 nDCG) sits
below the harness's own noise floor.

This module grades against what 247 participants in the discrete-choice study
(studies/weight-elicitation) actually booked. It answers a different question —
"does the ranking predict human choice?" — and, unlike the rule gold, it
separates the weight profiles clearly (+0.111 nDCG, p < 0.001).

Neither evaluation supersedes the other. Report both, and state each one's blind
spot: the rule gold cannot see ranking quality; the choice data cannot see price
sensitivity (hypothetical bias) and covers only ten scenario prompts.

TWO VIEWS OF THE SAME CHOICES
-----------------------------
per_choice  ground truth is the ONE hotel that participant booked. Exactly one
            relevant item, so P@K is capped at 1/K and R@K equals the hit rate;
            MRR and mean rank are the informative numbers here.
per_task    ground truth is every hotel that >= `min_votes` of the held-out
            group booked for that scenario. A multi-item gold set, which is what
            P/R/nDCG were designed for. Graded nDCG uses the vote count as gain.

THE ANCHOR-FAIRNESS PROBLEM
---------------------------
Every study task is anchored to a real Colombo location ("two days of meetings
in Battaramulla"), and human choice is dominated by proximity to that anchor.
The weighted retriever scores `spatial` / `accessibility` RELATIVE TO THE ANCHOR.
The deployed pgvector index does not: `scripts/build_vector_index.py` verbalises
travel time to the CITY CENTRE. Scored that way the text baselines are
structurally blind to the one variable that decides the answer, and they land at
or below random — a rigged comparison, and the same information asymmetry this
project already criticises elsewhere.

`anchor_fair=True` (default) therefore rebuilds each baseline's documents PER
TASK with anchor-relative travel time, using the project's own `_doc()` builder
and embedder, so every system sees the same information. `anchor_fair=False`
reproduces the deployed behaviour and is kept for comparison — the gap between
the two modes is itself a reportable result.
"""
from __future__ import annotations

import collections
import csv
import json
import logging
import math
import re
import statistics as st
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from evaluation.metrics import (ndcg_at_k, ndcg_at_k_graded, precision_at_k,
                                recall_at_k, reciprocal_rank)
from src.crag.query_parser import parse_query
from src.graph.retriever import WEIGHT_PROFILES

logger = logging.getLogger("alma.eval.human")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STUDY = PROJECT_ROOT / "studies" / "weight-elicitation"
DEFAULT_RESPONSES = STUDY / "data" / "responses_v4-rooms-20260818.csv"
DEFAULT_MATERIAL = STUDY / "material" / "study_material_v1_minmax.json"
DEFAULT_HUMAN_RESULTS = PROJECT_ROOT / "evaluation" / "results_human.json"

DIMS = ["spatial", "accessibility", "facility", "economic", "disruption"]

# Persona ids from a superseded material version (3 participants, Aug 3-14).
# Their choices were made against a different hotel pool.
STALE_PERSONAS = {"budget_backpacker", "business_accessibility", "family_facility",
                  "quiet_seeker", "sightseer", "attention"}

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_material(path: Path | str = DEFAULT_MATERIAL) -> Dict[str, Any]:
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    if m.get("normalisation") != "minmax":
        logger.warning(
            "Material %s is percentile-rank encoded; components will not match the "
            "retriever's min-max scale. Run studies/weight-elicitation/analysis/"
            "recode_components.py.", m.get("version"))
    return m


def load_responses(path: Path | str = DEFAULT_RESPONSES) -> List[Dict[str, str]]:
    with Path(path).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Cohort selection
# ---------------------------------------------------------------------------

def select_cohort(rows: List[Dict[str, str]], mode: str = "clean",
                  min_decision_ms: float = 5000.0) -> Tuple[Set[str], Dict[str, int]]:
    """Participant filter. Always reported, never silently applied.

    all       every participant
    attention passed the attention-check task
    clean     passed AND is not a speeder (median decision time >= min_decision_ms)

    Note the cohort barely moves the estimates — see DATA_AUDIT.md. It is
    reported so a reader can see the choice was made, not to reach a number.
    """
    by: Dict[str, Dict[str, Dict[str, str]]] = collections.defaultdict(dict)
    for r in rows:
        by[r["participant_id"]][r["task_id"]] = r

    def median_ms(p: str) -> float:
        vals: List[float] = []
        for r in by[p].values():
            try:
                vals.append(float(r["decision_ms"]))
            except (TypeError, ValueError):
                pass
        return st.median(vals) if vals else 0.0

    def passed(p: str) -> Optional[bool]:
        r = by[p].get("t_attn")
        return None if not r else str(r.get("attention_pass", "")).lower() == "true"

    keep: Set[str] = set()
    stats: Dict[str, int] = collections.defaultdict(int)
    for p in by:
        stats["participants_total"] += 1
        att, fast = passed(p), median_ms(p) < min_decision_ms
        if mode == "all":
            keep.add(p)
        elif mode == "attention":
            if att is True:
                keep.add(p)
            else:
                stats["dropped_attention"] += 1
        else:  # clean
            if att is True and not fast:
                keep.add(p)
            elif att is not True:
                stats["dropped_attention"] += 1
            else:
                stats["dropped_speeder"] += 1
    stats["participants_kept"] = len(keep)
    return keep, dict(stats)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

class Study:
    """The study material plus the usable observations drawn from it."""

    def __init__(self, material: Dict[str, Any], rows: List[Dict[str, str]],
                 keep: Set[str], restrict_to: Optional[Set[str]] = None) -> None:
        self.material = material
        self.hotels = material["hotels"]
        self.anchor_components = material["anchor_components"]
        self.tasks = {t["id"]: t for t in material["tasks"]
                      if t.get("anchor_id") and not t.get("is_attention_check")}
        self._restrict = restrict_to
        self.observations: List[Tuple[str, str, str]] = []   # (participant, task, hotel)
        self.skipped: Dict[str, int] = collections.defaultdict(int)

        for r in rows:
            if r["participant_id"] not in keep:
                self.skipped["cohort"] += 1
                continue
            if str(r.get("is_attention_check", "")).lower() == "true":
                self.skipped["attention_task"] += 1
                continue
            if r.get("scenario_persona") in STALE_PERSONAS:
                self.skipped["stale_material"] += 1
                continue
            if r["task_id"] not in self.tasks:
                self.skipped["unknown_task"] += 1
                continue
            if r["chosen_hotel_id"] not in self.pool(self.tasks[r["task_id"]]["anchor_id"]):
                self.skipped["chosen_outside_candidate_set"] += 1
                continue
            self.observations.append(
                (r["participant_id"], r["task_id"], r["chosen_hotel_id"]))

    def pool(self, anchor: str) -> List[str]:
        """Candidate set for an anchor — stable order, identical for every system."""
        ids = [h for h in self.anchor_components[anchor] if h in self.hotels]
        if self._restrict is not None:
            ids = [h for h in ids if h in self._restrict]
        return ids

    def features(self, anchor: str, ids: Sequence[str]) -> np.ndarray:
        ac, H = self.anchor_components[anchor], self.hotels
        return np.array([[ac[h]["spatial"], ac[h]["accessibility"],
                          H[h]["components_global"]["facility"],
                          H[h]["components_global"]["economic"],
                          H[h]["components_global"]["disruption"]] for h in ids])

    def participants(self) -> List[str]:
        return sorted({p for p, _, _ in self.observations})


# ---------------------------------------------------------------------------
# Conditional logit — the weights the choices imply
# ---------------------------------------------------------------------------

def _neg_loglik(b: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    u = X @ b
    u -= u.max(axis=1, keepdims=True)
    return float(-(u[np.arange(len(y)), y] - np.log(np.exp(u).sum(axis=1))).sum())


def fit_weights(study: Study, obs: Sequence[Tuple[str, str, str]]) -> np.ndarray:
    """Simplex weights implied by a set of choices (negatives clipped to zero)."""
    from scipy.optimize import minimize
    X, y = [], []
    for _p, t, hid in obs:
        ids = study.pool(study.tasks[t]["anchor_id"])
        X.append(study.features(study.tasks[t]["anchor_id"], ids))
        y.append(ids.index(hid))
    res = minimize(_neg_loglik, np.zeros(len(DIMS)),
                   args=(np.array(X), np.array(y)), method="BFGS")
    w = np.clip(res.x, 0.0, None)
    return w / w.sum() if w.sum() > 0 else np.full(len(DIMS), 1.0 / len(DIMS))


# ---------------------------------------------------------------------------
# Rankers
# ---------------------------------------------------------------------------

def _tokens(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def _bm25(docs: Sequence[str], query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """Lexical relevance, standing in for Postgres ts_rank when documents are
    rebuilt in memory (anchor-fair mode)."""
    tokenised = [_tokens(d) for d in docs]
    n = len(tokenised)
    avg = sum(len(d) for d in tokenised) / n if n else 0.0
    df: collections.Counter = collections.Counter()
    for d in tokenised:
        df.update(set(d))
    out = []
    for d in tokenised:
        tf = collections.Counter(d)
        score = 0.0
        for term in set(_tokens(query)):
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * len(d) / avg))
        out.append(score)
    return np.array(out)


def _rrf(*rankings: Sequence[str], k: int = 60) -> List[str]:
    """Reciprocal Rank Fusion — same constant as src/search/vector_store.py."""
    score: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            score[doc] = score.get(doc, 0.0) + 1.0 / (k + rank)
    return [doc for doc, _ in sorted(score.items(), key=lambda kv: -kv[1])]


def _complete(ranked: Sequence[str], ids: Sequence[str]) -> List[str]:
    """Keep the ranking's order, then append anything it left out.

    Every system must return the full candidate set so that rank-based metrics
    are comparable; a system that silently returns fewer items would otherwise
    get an easier denominator.
    """
    seen = set(ranked)
    return [h for h in ranked if h in set(ids)] + [h for h in ids if h not in seen]


class AnchorFairIndex:
    """Per-anchor documents + embeddings, built with the project's own `_doc()`.

    Travel time is verbalised relative to the TASK ANCHOR rather than the city
    centre, so the text baselines see the same location signal the weighted
    retriever does.
    """

    def __init__(self, study: Study) -> None:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from build_vector_index import _doc  # noqa: E402  (project's builder)
        from src.search.embedder import embed  # noqa: E402

        self.docs: Dict[str, List[str]] = {}
        self.emb: Dict[str, np.ndarray] = {}
        self.ids: Dict[str, List[str]] = {}
        for anchor in sorted({t["anchor_id"] for t in study.tasks.values()}):
            ids = study.pool(anchor)
            self.ids[anchor] = ids
            self.docs[anchor] = [_doc(self._hotel(study, h, anchor)) for h in ids]
            self.emb[anchor] = np.array(embed(self.docs[anchor]))

    @staticmethod
    def _hotel(study: Study, hid: str, anchor: str) -> Dict[str, Any]:
        a = study.hotels[hid]["attributes"]
        detail = study.hotels[hid].get("detail", {})
        return {
            "name": study.hotels[hid]["name"],
            "description": detail.get("description_full") or a.get("description", ""),
            "amenities": detail.get("facilities_all") or a.get("amenities", []),
            "attractions": [],
            "price": a.get("price_lkr"),
            "rating": a.get("rating"),
            "star": a.get("star"),
            "travel_time_traffic_min": study.anchor_components[anchor][hid]["travel_min"],
        }


def build_rankers(study: Study, anchor_fair: bool,
                  weight_vectors: Dict[str, np.ndarray]) -> Dict[str, Callable[[str, str], List[str]]]:
    """One ranking function per system: (anchor, question) -> ranked hotel ids."""
    rankers: Dict[str, Callable[[str, str], List[str]]] = {}

    # --- Filter: structured constraints, then rating (FilterBaseline's rule) ---
    def filter_rank(anchor: str, question: str) -> List[str]:
        ids = study.pool(anchor)
        intent = parse_query(question, default_city="Colombo")
        kept = []
        for h in ids:
            a = study.hotels[h]["attributes"]
            price, rating, star = a.get("price_lkr"), a.get("rating"), a.get("star")
            if intent.max_price_lkr is not None and (price is None or price > intent.max_price_lkr):
                continue
            if intent.min_price_lkr is not None and (price is None or price < intent.min_price_lkr):
                continue
            if intent.min_rating is not None and (rating is None or rating < intent.min_rating):
                continue
            if intent.min_star is not None and (star is None or star < intent.min_star):
                continue
            kept.append(h)
        kept.sort(key=lambda h: (-(study.hotels[h]["attributes"].get("rating") or 0),
                                 study.hotels[h]["attributes"].get("price_lkr") or float("inf")))
        return _complete(kept, ids)

    rankers["Filter"] = filter_rank

    if anchor_fair:
        index = AnchorFairIndex(study)
        from src.search.embedder import embed_one

        def keyword_rank(anchor: str, question: str) -> List[str]:
            ids = index.ids[anchor]
            order = np.argsort(-_bm25(index.docs[anchor], question))
            return _complete([ids[i] for i in order], ids)

        def semantic_rank(anchor: str, question: str) -> List[str]:
            ids, E = index.ids[anchor], index.emb[anchor]
            v = np.array(embed_one(question))
            sim = (E @ v) / (np.linalg.norm(E, axis=1) * np.linalg.norm(v) + 1e-12)
            return _complete([ids[i] for i in np.argsort(-sim)], ids)

        def hybrid_rank(anchor: str, question: str) -> List[str]:
            return _complete(_rrf(semantic_rank(anchor, question),
                                  keyword_rank(anchor, question)), index.ids[anchor])
    else:
        from src.search import vector_store as vs
        from src.search.embedder import embed_one
        DEEP = 200

        def keyword_rank(anchor: str, question: str) -> List[str]:
            return _complete(vs.keyword_search("Colombo", question, DEEP), study.pool(anchor))

        def semantic_rank(anchor: str, question: str) -> List[str]:
            return _complete(vs.semantic_search("Colombo", embed_one(question), DEEP),
                             study.pool(anchor))

        def hybrid_rank(anchor: str, question: str) -> List[str]:
            return _complete(vs.hybrid_search("Colombo", question, embed_one(question), DEEP),
                             study.pool(anchor))

    rankers["Keyword"] = keyword_rank
    rankers["SemanticVec"] = semantic_rank
    rankers["Hybrid"] = hybrid_rank

    for label, vec in weight_vectors.items():
        def make(v: np.ndarray) -> Callable[[str, str], List[str]]:
            def rank(anchor: str, _question: str) -> List[str]:
                ids = study.pool(anchor)
                return [ids[i] for i in np.argsort(-(study.features(anchor, ids) @ v))]
            return rank
        rankers[label] = make(vec)

    return rankers


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _per_choice(rank_lists: Dict[str, List[str]],
                obs: Sequence[Tuple[str, str, str]], k: int) -> Dict[str, float]:
    p = r = n = mrr = 0.0
    ranks: List[int] = []
    for _pid, task, hid in obs:
        ranked = rank_lists[task]
        truth = {hid}
        p += precision_at_k(ranked, truth, k)
        r += recall_at_k(ranked, truth, k)
        n += ndcg_at_k(ranked, truth, k)
        mrr += reciprocal_rank(ranked, truth)
        ranks.append(ranked.index(hid) + 1)
    m = len(obs)
    return {f"P@{k}": p / m, f"R@{k}": r / m, f"nDCG@{k}": n / m,
            "MRR": mrr / m, "mean_rank": st.mean(ranks)}


def _per_task(rank_lists: Dict[str, List[str]],
              votes: Dict[str, collections.Counter], k: int,
              min_votes: int) -> Dict[str, float]:
    acc: Dict[str, float] = collections.defaultdict(float)
    n = 0
    for task, counter in sorted(votes.items()):
        gold = {h for h, v in counter.items() if v >= min_votes}
        if not gold:
            continue
        ranked = rank_lists[task]
        gains = {h: float(v) for h, v in counter.items() if v >= min_votes}
        acc[f"P@{k}"] += precision_at_k(ranked, gold, k)
        acc[f"R@{k}"] += recall_at_k(ranked, gold, k)
        acc[f"nDCG@{k}"] += ndcg_at_k(ranked, gold, k)
        acc[f"gradedNDCG@{k}"] += ndcg_at_k_graded(ranked, gains, k)
        acc["MRR"] += reciprocal_rank(ranked, gold)
        n += 1
    return {key: val / n for key, val in acc.items()} if n else {}


def _bootstrap_vs(per_query: Dict[str, np.ndarray], reference: str,
                  reps: int = 5000, seed: int = 7) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    out: Dict[str, Any] = {}
    ref = per_query[reference]
    for name, vals in per_query.items():
        if name == reference:
            continue
        diff = ref - vals
        boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean()
                          for _ in range(reps)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p = 2 * min(float((boots <= 0).mean()), float((boots >= 0).mean()))
        out[name] = {"mean_diff": round(float(diff.mean()), 4),
                     "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
                     "p": round(p, 4), "significant": bool(lo * hi > 0)}
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_human_evaluation(
    responses_path: Path | str = DEFAULT_RESPONSES,
    material_path: Path | str = DEFAULT_MATERIAL,
    k: int = 10,
    cohort: str = "clean",
    holdout: float = 0.3,
    min_votes: int = 2,
    seed: int = 42,
    anchor_fair: bool = True,
    restrict_to_indexed: bool = False,
) -> Dict[str, Any]:
    """Evaluate every system against real booking choices.

    Weights are fitted on TRAIN participants and scored on TEST participants —
    splitting by participant, never by row, because one person's ten choices are
    correlated and a row split would leak them across the boundary.
    """
    material = load_material(material_path)
    rows = load_responses(responses_path)
    keep, cohort_stats = select_cohort(rows, cohort)

    restrict: Optional[Set[str]] = None
    if restrict_to_indexed:
        from src.search import vector_store as vs
        try:
            restrict = {r[0] for r in vs._query(
                "SELECT id FROM hotel_search WHERE lower(city)=lower(%s)", ("Colombo",))}
        except Exception as exc:
            logger.warning("Could not read the pgvector index (%s); not restricting.", exc)

    study = Study(material, rows, keep, restrict_to=restrict)
    if not study.observations:
        raise RuntimeError("No usable observations — check the responses file and material.")

    rng = np.random.default_rng(seed)
    people = study.participants()
    shuffled = list(people)
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * holdout))
    test_p = set(shuffled[:n_test])
    train_obs = [o for o in study.observations if o[0] not in test_p]
    test_obs = [o for o in study.observations if o[0] in test_p]

    fitted = fit_weights(study, train_obs)
    weight_vectors: Dict[str, np.ndarray] = {
        f"WeightedGraphRAG[{name}]": np.array([getattr(w, d) for d in DIMS])
        for name, w in WEIGHT_PROFILES.items()
    }
    weight_vectors["WeightedGraphRAG[fitted-on-train]"] = fitted

    rankers = build_rankers(study, anchor_fair, weight_vectors)
    system_order = list(rankers)

    # One ranking per (system, task) — tasks are the queries here.
    rank_lists: Dict[str, Dict[str, List[str]]] = {}
    for name, fn in rankers.items():
        rank_lists[name] = {t: fn(spec["anchor_id"], spec["context"])
                            for t, spec in study.tasks.items()}

    votes: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for _p, task, hid in test_obs:
        votes[task][hid] += 1

    per_choice = {n: _per_choice(rank_lists[n], test_obs, k) for n in system_order}
    per_task = {n: _per_task(rank_lists[n], votes, k, min_votes) for n in system_order}

    ndcg_vectors = {
        n: np.array([ndcg_at_k(rank_lists[n][t], {h}, k) for _p, t, h in test_obs])
        for n in system_order
    }
    reference = "WeightedGraphRAG[fitted-on-train]"
    best = max(per_choice.items(), key=lambda kv: kv[1].get(f"nDCG@{k}", 0.0))

    pool_sizes = {t: len(study.pool(s["anchor_id"])) for t, s in study.tasks.items()}
    return {
        "available": True,
        "evaluation": "choice-based (human ground truth)",
        "material_version": material.get("version"),
        "normalisation": material.get("normalisation", "pct_rank"),
        "anchor_fair": anchor_fair,
        "k": k,
        "cohort": cohort,
        "cohort_stats": cohort_stats,
        "min_votes": min_votes,
        "candidate_set_size": max(pool_sizes.values()) if pool_sizes else 0,
        "restricted_to_indexed": bool(restrict),
        "n_observations": len(study.observations),
        "n_train_participants": len(people) - n_test,
        "n_test_participants": n_test,
        "n_train_choices": len(train_obs),
        "n_test_choices": len(test_obs),
        "skipped": dict(study.skipped),
        "system_order": system_order,
        "weights": {n: {d: round(float(v), 4) for d, v in zip(DIMS, vec)}
                    for n, vec in weight_vectors.items()},
        "per_choice": {n: {kk: round(v, 4) for kk, v in m.items()} for n, m in per_choice.items()},
        "per_task": {n: {kk: round(v, 4) for kk, v in m.items()} for n, m in per_task.items()},
        "random_baseline": {
            f"P@{k}": round(1.0 / max(pool_sizes.values(), default=k), 4),
            f"R@{k}": round(k / max(pool_sizes.values(), default=k), 4),
            "mean_rank": round((max(pool_sizes.values(), default=k) + 1) / 2, 2),
        },
        "significance": {
            "reference": reference,
            "metric": f"nDCG@{k}",
            "tests": "paired bootstrap over held-out choices, 5000 resamples",
            "vs": _bootstrap_vs(ndcg_vectors, reference),
        },
        "tasks": [
            {
                "id": t,
                "persona": spec.get("persona"),
                "anchor": spec.get("anchor_id"),
                "context": spec.get("context"),
                "primary_dimension": spec.get("primary_dimension"),
                "n_test_choices": sum(votes[t].values()),
                "gold_size": len({h for h, v in votes[t].items() if v >= min_votes}),
                "top_pick": (lambda c: {"hotel": study.hotels[c[0][0]]["name"], "votes": c[0][1]}
                             if c else None)(votes[t].most_common(1)),
            }
            for t, spec in sorted(study.tasks.items())
        ],
        "best_system": best[0],
        "best_ndcg": round(best[1].get(f"nDCG@{k}", 0.0), 4),
    }
