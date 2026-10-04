"""
Fit the retriever's weights from the study's PER-QUESTION CHOICE SHARES.

    python -m weight_elicitation.fit_share_weights
    python -m weight_elicitation.fit_share_weights --grain display --bootstrap 400
    python -m weight_elicitation.fit_share_weights --top 8 --emit-profile
    python -m weight_elicitation.fit_share_weights --check-equivalence

Where the other two fitters consume one row per PARTICIPANT-CHOICE, this one
consumes one row per QUESTION x HOTEL, carrying the percentage of participants
who picked that hotel on that question:

    task  hotel                      share    chosen/shown
    t1    Shangri-La Colombo         27.9%       69/247
    t1    The Kingsbury              12.1%       30/247
    t1    Galle Face Hotel            9.7%       24/247
    ...   (every hotel in the pool, including the ones nobody picked)

Ten questions x 32 hotels = 320 rows, and the whole estimate is computed from
them. Nothing is filtered away first: a hotel shown 247 times and chosen 0 times
is a row, and it is exactly as informative as a popular one.

Why this is not a coarser version of `fit_weights.py`
----------------------------------------------------
It is not coarser at all; it is the same likelihood written over its sufficient
statistic. Two facts about the material make that true, and both are checked at
run time (the first raises, the second is reported by `--check-equivalence`):

1. The component vector of a hotel is CONSTANT within a question. All 352
   (task, hotel) cells in the dump carry exactly one vector, because the anchor
   -- and therefore `spatial` and `accessibility` -- is a property of the
   scenario, not of the participant.

2. Every participant saw the SAME 32-hotel pool. 2,225 of 2,232 sets show the
   full pool; the 7 that do not are handled exactly by an availability term
   (see `design()` below) rather than by dropping them.

When the alternatives are identical for everyone answering a question, the
multinomial logit likelihood depends on the data only through the counts. So
summing 247 individual likelihood terms and weighting one row by 247 give the
same value, the same gradient and the same estimate. The percentages ARE the
data, not a summary of it.

What the aggregation buys
-------------------------
* EVERY hotel appears, including the 0% ones. In the individual formulation a
  rejected alternative is implicit in the denominator; here it is a visible row
  with a measured selection rate, which is what makes the correlation analysis
  below possible at all.
* Correlation coefficients become well defined. Choice share is a continuous
  per-hotel outcome, so `corr(component, share)` over the 32 hotels of a
  question is an ordinary Pearson/Spearman coefficient with an honest p-value.
  A one-row-per-choice dataset has no such column to correlate against.
* Each question can be fitted ON ITS OWN and the ten vectors averaged, so a
  scenario cannot dominate by having attracted more participants.

Position bias
-------------
Rank still has to be controlled -- 46% of participants took the first hotel on
the list. Each row therefore carries the mean of -log(displayed_position) over
the participants who saw it, and that column is fitted alongside the components
and then DISCARDED, exactly as in `fit_weights.py`: at retrieval time there is
no pre-existing position, because the retriever is what creates the ordering.

At `--grain display` the rows split by the sort the participant chose, where
position is deterministic (1,551 of 1,600 cells carry a single value) and the
control is exact rather than averaged. At `--grain question` the ten questions
stay whole and the control is an average over sorts.

Audit result (2026-09-13) - read before quoting anything this prints
--------------------------------------------------------------------
The shares are the right thing to REPORT and the wrong thing to ship weights
from. On wave 1 the per-question macro estimator fails the placebo test: fed
choices driven by list position alone it returns economic .233 and facility
.289, more than the real data gives, because clipping ten noisy fits at zero
and averaging the corners cannot produce zero. With a position control that
matches the data (`dummies:3`, the new default) the components add nothing
beyond position (chi2(5) = 8.7, p = .12), and every top hotel's share equals
the share its list positions predict (Shangri-La on t1: 32.1% vs 31.4%).
So `shippable` is false unless `--gates` passes, and `--emit-profile` refuses
otherwise. See docs/Weight_Elicitation_Share_Audit.md.

The CSV now carries, per question x hotel, a Wilson 95% interval on the share
and the share EXPECTED FROM POSITION ALONE: the empirical P(choose | rank, sort
mode) from the same responses, applied to the ranks this hotel was actually
shown at. `position_z` is (chosen - expected) / sd under that baseline, so a
hotel preferred beyond where it sat shows up as a large positive z.

Outputs `out/choice_shares.csv` (the row-by-row table), `out/share_weights.json`
(weights, correlations, intervals, gates) and a report on stdout.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import minimize

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from weight_elicitation import DUMPS, MATERIAL, OUT, REPO, latest_dump
from weight_elicitation.choice_model import DEFAULT_POSITION, PositionSpec, wilson_interval
from weight_elicitation.fit_weights import (
    DIMS,
    HANDSET,
    apply_position_spec,
    build_choice_sets,
    facility_scores,
    failed_attention,
    fit_mnl,
    load_dump,
    load_material,
    macro_by_sort,
    ndcg_at_k,
    to_simplex,
    top1,
)

SORT_MODES = ("distance", "travel", "rating", "price_asc", "price_desc")


def _rel(path: Path) -> str:
    """Repo-relative when possible; an output written elsewhere keeps its path."""
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _task_order(task_id: str) -> int:
    return int(task_id[1:]) if task_id[1:].isdigit() else 99


# --------------------------------------------------------------------------- #
# The panel: per-response indicators, aggregated on demand
# --------------------------------------------------------------------------- #
class SharePanel:
    """Response-level indicator arrays plus the machinery to count them.

    Everything is parsed out of the dump ONE TIME and then aggregated to shares
    by group. A clustered bootstrap therefore only re-runs the aggregation --
    three `np.add.at` calls -- instead of re-walking 2,232 responses x 32
    options, which is what lets a few hundred replicates finish in seconds.

    exposed[j, h]    hotel h was in participant j's choice set (on screen, and
                     inside the top k for a `topk` position spec)
    rank[j, h]       the position it occupied (0 where not on screen)
    posfeat[j, h, p] the position terms of the spec (0 where not exposed)
    chosen[j]        index of the hotel they booked
    group[j]         which group (question, or question x sort) the row joins
    sorts[j]         the sort the participant had applied
    """

    def __init__(self, hotels: List[str], names: Dict[str, str],
                 groups: List[tuple], features: np.ndarray,
                 exposed: np.ndarray, rank: np.ndarray, posfeat: np.ndarray,
                 pos_names: List[str], position: PositionSpec,
                 chosen: np.ndarray, group: np.ndarray,
                 participants: np.ndarray, sorts: np.ndarray,
                 task_meta: Dict[str, dict], grain: str):
        self.hotels, self.names = hotels, names
        self.groups, self.features = groups, features          # features (G,H,5)
        self.exposed, self.rank = exposed, rank
        self.posfeat, self.pos_names, self.position = posfeat, pos_names, position
        self.neglogpos = np.where(exposed, -np.log(np.maximum(rank, 1)), 0.0)
        self.chosen, self.group = chosen, group
        self.participants, self.sorts = participants, sorts
        self.task_meta, self.grain = task_meta, grain

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def n_hotels(self) -> int:
        return len(self.hotels)

    def tasks(self) -> List[str]:
        return sorted({g[0] for g in self.groups}, key=_task_order)

    def rows_for_task(self, task_id: str) -> np.ndarray:
        return np.array([i for i, g in enumerate(self.groups) if g[0] == task_id])

    def counts(self, rows: Optional[np.ndarray] = None) -> "ShareCounts":
        """Aggregate a set of responses into the per-group share table."""
        if rows is None:
            rows = np.arange(len(self.chosen))
        G, H, Pn = self.n_groups, self.n_hotels, self.posfeat.shape[2]
        n_exposed = np.zeros((G, H))
        sum_nlp = np.zeros((G, H))
        sum_pos = np.zeros((G, H, Pn))
        n_chosen = np.zeros((G, H))
        n_resp = np.zeros(G)
        g = self.group[rows]
        np.add.at(n_exposed, g, self.exposed[rows].astype(float))
        np.add.at(sum_nlp, g, self.neglogpos[rows])
        np.add.at(sum_pos, g, self.posfeat[rows])
        np.add.at(n_chosen, (g, self.chosen[rows]), 1.0)
        np.add.at(n_resp, g, 1.0)
        exp_cnt, exp_var = self.position_baseline(rows)
        return ShareCounts(self, n_exposed, sum_nlp, n_chosen, n_resp, sum_pos,
                           exp_cnt, exp_var)

    def position_baseline(self, rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Expected choices per (group, hotel) if people chose by position ONLY.

        P(choose | rank r, sort s) is the empirical share of the sort-s responses
        whose chosen hotel sat at rank r, renormalised over the hotels each
        response could choose from. No functional form is assumed, which is the
        point: the -log(rank) curve over-predicts rank 1 by ~17 points on wave 1
        and turns Shangri-La's 32% into an apparent under-selection.
        Returns expected counts and the Poisson-binomial variance.
        """
        G, H = self.n_groups, self.n_hotels
        rk, ex, ch = self.rank[rows], self.exposed[rows], self.chosen[rows]
        srt = self.sorts[rows]
        max_r = int(self.rank.max()) + 1
        prob = np.zeros((len(rows), H))
        for mode in np.unique(srt):
            sel = srt == mode
            chosen_rank = rk[sel][np.arange(sel.sum()), ch[sel]]
            q = np.bincount(chosen_rank, minlength=max_r).astype(float) / max(sel.sum(), 1)
            prob[sel] = np.where(ex[sel], q[rk[sel]], 0.0)
        tot = prob.sum(axis=1, keepdims=True)
        prob = np.where(tot > 0, prob / np.maximum(tot, 1e-300), 0.0)
        exp_cnt, exp_var = np.zeros((G, H)), np.zeros((G, H))
        g = self.group[rows]
        np.add.at(exp_cnt, g, prob)
        np.add.at(exp_var, g, prob * (1.0 - prob))
        return exp_cnt, exp_var


class ShareCounts:
    """The row-by-row percentage table, in matrix form.

    share[g, h] = n_chosen[g, h] / n_exposed[g, h] -- the number the report
    prints as "27.9% picked Shangri-La on t1".
    """

    def __init__(self, panel: SharePanel, n_exposed: np.ndarray,
                 sum_nlp: np.ndarray, n_chosen: np.ndarray, n_resp: np.ndarray,
                 sum_pos: np.ndarray, expected: np.ndarray, expected_var: np.ndarray):
        self.panel = panel
        self.n_exposed, self.n_chosen, self.n_resp = n_exposed, n_chosen, n_resp
        denom = np.maximum(n_exposed, 1.0)
        self.share = np.where(n_exposed > 0, n_chosen / denom, 0.0)
        # Mean -log(rank) over the people who actually saw the hotel (reported).
        self.mean_nlp = np.where(n_exposed > 0, sum_nlp / denom, 0.0)
        # Mean of each position term: the exact control when rank is fixed
        # within a group, which it is at display grain.
        self.mean_pos = np.where((n_exposed > 0)[:, :, None],
                                 sum_pos / denom[:, :, None], 0.0)
        self.expected, self.expected_var = expected, expected_var
        self.mask = n_exposed > 0

    def design(self, use_position: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Feature tensor (G, H, K) and the log-availability offset (G, H).

        `log(n_exposed / n_resp)` is the availability correction: a hotel only
        two thirds of the group could see must not be scored as though all of
        them rejected it. With an unfiltered pool the term is log(1) = 0 and
        changes nothing, which is the case for 2,225 of 2,232 sets here -- it
        exists so the seven filtered sets stay in the estimate instead of being
        thrown away or quietly biasing it.
        """
        X = self.panel.features
        if use_position:
            X = np.concatenate([X, self.mean_pos], axis=2)
        avail = np.where(self.mask,
                         self.n_exposed / np.maximum(self.n_resp[:, None], 1.0), 1.0)
        logavail = np.where(avail > 0, np.log(np.maximum(avail, 1e-12)), 0.0)
        return X, logavail


def build_panel(responses: List[dict], material: dict, facility: Dict[str, float],
                pool_size: int, grain: str, drop_failed: bool,
                failed: set, position: PositionSpec = DEFAULT_POSITION) -> SharePanel:
    hotels = list(material["hotels"])
    names = {h: material["hotels"][h].get("name", h) for h in hotels}
    hidx = {h: i for i, h in enumerate(hotels)}
    task_meta = {t["id"]: t for t in material.get("tasks", [])}

    # Component vectors are a property of (task, hotel). Collect them once and
    # verify the invariance the whole aggregation rests on, rather than assuming
    # it: if a future wave randomises components per participant, the shares
    # stop being sufficient and this must fail loudly instead of averaging.
    comps: Dict[Tuple[str, str], np.ndarray] = {}
    for r in responses:
        if r.get("isAttentionCheck") or len(r.get("options") or []) != pool_size:
            continue
        for o in r["options"]:
            c = o.get("components")
            if not c:
                continue
            key = (r["taskId"], o["hotel_id"])
            v = np.array([c["spatial"], c["accessibility"],
                          facility.get(o["hotel_id"], c["facility"]),
                          c["economic"], c["disruption"]], dtype=float)
            if key in comps and not np.allclose(comps[key], v):
                raise RuntimeError(
                    f"components vary within {key}: the per-question share "
                    f"table is not a sufficient statistic for this dump")
            comps[key] = v

    variants: Dict[tuple, Dict[tuple, int]] = {}

    def group_key(r: dict) -> Optional[tuple]:
        if grain == "question":
            return (r["taskId"],)
        sort = (r.get("timing") or {}).get("final_sort")
        if sort not in SORT_MODES:
            return None
        # The display group is (question, sort, EXACT ORDER SHOWN). Within
        # (question, sort) rank is usually fixed, but ties in the sort key and
        # filters leave some hotels at rank 3 for one participant and rank 4
        # for another. Averaging those two position indicators is not the same
        # likelihood, so the grouped fit would stop being exact the moment the
        # position terms are indicators rather than a smooth curve. Keying on
        # the order keeps every group's positions identical, and the identity
        # with the individual-level fit exact.
        order = tuple(sorted((o["hotel_id"], o.get("displayed_position"))
                             for o in r["options"]
                             if o.get("displayed_position") is not None))
        slot = variants.setdefault((r["taskId"], sort), {})
        return (r["taskId"], sort, slot.setdefault(order, len(slot)))

    keep: List[dict] = []
    keys: List[tuple] = []
    for r in responses:
        if r.get("isAttentionCheck") or len(r.get("options") or []) != pool_size:
            continue
        if drop_failed and r["participantId"] in failed:
            continue
        shown = [o for o in r["options"]
                 if o.get("displayed_position") is not None and o.get("components")]
        if len(shown) < 2 or not any(o.get("chosen") for o in shown):
            continue
        k = group_key(r)
        if k is None:
            continue
        keep.append(r)
        keys.append(k)
    if not keep:
        raise RuntimeError("no usable responses at this pool size / grain")

    groups = sorted(set(keys), key=lambda k: (_task_order(k[0]),) + k[1:])
    gidx = {k: i for i, k in enumerate(groups)}

    n, H = len(keep), len(hotels)
    shown = np.zeros((n, H), dtype=bool)
    rank = np.zeros((n, H), dtype=int)
    chosen = np.zeros(n, dtype=int)
    group = np.zeros(n, dtype=int)
    for j, (r, k) in enumerate(zip(keep, keys)):
        group[j] = gidx[k]
        for o in r["options"]:
            p = o.get("displayed_position")
            if p is None or not o.get("components"):
                continue
            i = hidx[o["hotel_id"]]
            shown[j, i] = True
            rank[j, i] = int(max(int(p), 1))
            if o.get("chosen"):
                chosen[j] = i

    # Position terms. A `topk` spec shrinks the choice set to ranks 1..k and
    # drops the responses whose chosen hotel sat below it.
    posfeat, exposed, pos_names = position.columns(rank, shown)
    inside = exposed[np.arange(n), chosen]
    if not inside.all():
        sel = np.where(inside)[0]
        keep = [keep[i] for i in sel]
        exposed, rank, posfeat = exposed[sel], rank[sel], posfeat[sel]
        chosen, group = chosen[sel], group[sel]
        n = len(sel)
    live = [c for c in range(posfeat.shape[2]) if np.any(posfeat[:, :, c][exposed])]
    posfeat = posfeat[:, :, live]
    pos_names = [pos_names[c] for c in live]

    features = np.zeros((len(groups), H, 5))
    for gi, k in enumerate(groups):
        for hi, h in enumerate(hotels):
            v = comps.get((k[0], h))
            if v is not None:
                features[gi, hi] = v

    return SharePanel(hotels, names, groups, features, exposed,
                      np.where(exposed, rank, 0), posfeat, pos_names, position,
                      chosen, group,
                      np.array([r["participantId"] for r in keep]),
                      np.array([(r.get("timing") or {}).get("final_sort") or "unknown"
                                for r in keep]),
                      task_meta, grain)


# --------------------------------------------------------------------------- #
# Grouped conditional logit on the shares
# --------------------------------------------------------------------------- #
def grouped_objective(beta: np.ndarray, X: np.ndarray, logavail: np.ndarray,
                      counts: np.ndarray, mask: np.ndarray, l2: float
                      ) -> Tuple[float, np.ndarray]:
    """Negative log-likelihood of the counts, with its analytic gradient.

    Equal, up to a constant, to summing the individual likelihood over every
    participant in the group -- see the module docstring. The gradient is
    written out rather than left to finite differences because the bootstrap
    refits this a few hundred times.
    """
    u = np.where(mask, X @ beta + logavail, -np.inf)
    mx = np.max(np.where(mask, u, -np.inf), axis=1, keepdims=True)
    # A group can be entirely empty -- a (question, sort) cell that no resampled
    # or training participant landed in. Its max is then -inf, and -inf minus
    # -inf is NaN, which would poison the whole likelihood rather than just
    # contributing nothing. Anchor those rows at 0; they carry no counts, so
    # they add exactly zero either way.
    mx = np.where(np.isfinite(mx), mx, 0.0)
    e = np.where(mask, np.exp(u - mx), 0.0)
    s = np.maximum(e.sum(axis=1, keepdims=True), 1e-300)
    logsum = np.log(s) + mx
    ll = float(np.where(mask, counts * (np.where(mask, u, 0.0) - logsum), 0.0).sum())

    p = e / s
    n_g = counts.sum(axis=1, keepdims=True)
    resid = np.where(mask, counts - n_g * p, 0.0)
    grad_ll = np.einsum("gh,ghk->k", resid, X)

    b5 = beta[:5]
    gpen = np.zeros_like(beta)
    gpen[:5] = 2.0 * l2 * b5
    return -ll + l2 * float(b5 @ b5), -grad_ll + gpen


def fit_grouped(sc: ShareCounts, *, use_position: bool = True,
                non_negative: bool = True, l2: float = 0.0,
                rows: Optional[np.ndarray] = None) -> np.ndarray:
    """Fit on all groups, or only on the groups indexed by `rows`."""
    X, logavail = sc.design(use_position)
    counts, mask = sc.n_chosen, sc.mask
    if rows is not None:
        X, logavail, counts, mask = X[rows], logavail[rows], counts[rows], mask[rows]
    k = X.shape[2]
    if non_negative:
        # Components are "higher is better" and the retriever cannot use a
        # negative weight; the position nuisance is left free.
        bounds = [(0.0, None)] * 5 + [(None, None)] * (k - 5)
    else:
        bounds = [(None, None)] * k
    res = minimize(grouped_objective, np.zeros(k),
                   args=(X, logavail, counts, mask, l2),
                   jac=True, method="L-BFGS-B", bounds=bounds)
    return res.x


def combine_questions(per: Dict[str, np.ndarray]) -> np.ndarray:
    """Combine the ten per-question vectors into the one that ships.

    THE DECISION POINT of this estimator, kept in one function so it can be
    changed without touching the likelihood.

    A plain mean is what is implemented. It is the honest default -- every
    scenario counts once -- but it is worth knowing what it is averaging: with
    no regularisation, several questions land exactly on a simplex CORNER
    (t2 and t10 both return accessibility = 1.000). Those are not statements
    that the scenario cared about one component only; they are the optimiser
    picking a winner between components that correlate at r = 0.9. The mean of
    ten corners is a reasonable ensemble, but it inherits their instability, and
    a coordinate-wise median or a trimmed mean would resist a single question
    swinging the vector. See `--l2` for the other lever on the same problem.
    """
    return np.mean(np.stack(list(per.values())), axis=0)


def fit_per_question(sc: ShareCounts, *, l2: float = 0.0, min_choices: int = 30,
                     **kw) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str]]:
    """One fit per QUESTION, then a plain average of the resulting simplices.

    Each scenario gets one vote. t1 drew 247 participants and t10 drew 203; a
    pooled fit lets that turnout difference matter, when what the design varies
    is the scenario, not the sample size. Averaging also stops one framing --
    the two economic scenarios behave very differently from the proximity ones
    -- from setting the whole vector because it happened to be answered more.

    A question whose fit collapses to all-zero components is DROPPED, not
    averaged in. `to_simplex` turns a zero vector into a flat 0.2 each, and a
    flat vector looks like a considered finding ("this scenario weighted
    everything equally") when it is really the optimiser saying the log-rank
    term explained the whole question and no component earned any mass. Silently
    averaging that in would drag every estimate toward uniform. The dropped
    question ids are returned so the report can name them.
    """
    per: Dict[str, np.ndarray] = {}
    degenerate: List[str] = []
    for t in sc.panel.tasks():
        rows = sc.panel.rows_for_task(t)
        if sc.n_chosen[rows].sum() < min_choices:
            degenerate.append(t)
            continue
        beta = fit_grouped(sc, l2=l2, rows=rows, **kw)
        if np.clip(beta[:5], 0.0, None).sum() <= 1e-8:
            degenerate.append(t)
            continue
        per[t] = to_simplex(beta)
    if not per:
        raise RuntimeError("every question fitted degenerately; nothing to average")
    return per, combine_questions(per), degenerate


def estimate(sc: ShareCounts, kind: str, l2: float) -> np.ndarray:
    """The three weight vectors this module can report, behind one name.

    Used by the regularisation search and the bootstrap so that both describe
    the SAME estimator the report ships. A bootstrap of a different estimator
    is an interval around a number nobody is quoting.
    """
    if kind == "per_question":
        return fit_per_question(sc, l2=l2)[1]
    if kind == "correlation":
        return correlation_weights(question_correlations(sc))
    return to_simplex(fit_grouped(sc, l2=l2))


def select_l2(panel: SharePanel, cs, grid: Sequence[float], kind: str,
              test_frac: float, seed: int) -> Tuple[float, List[Tuple[float, float, np.ndarray]]]:
    """Choose the ridge strength on HELD-OUT PARTICIPANTS.

    The shares are recounted from the training people only, so the held-out
    people contribute nothing to the vector being scored. Splitting on people
    rather than on rows matters more here than usual: every participant
    contributes to ten different share cells, so a row-wise split would leave
    the same person on both sides of the boundary ten times over.

    Scored with the same macro-averaged nDCG@10 the other two fitters use, which
    is what makes the three sets of numbers comparable at all.
    """
    people = np.unique(panel.participants)
    rng = np.random.default_rng(seed)
    rng.shuffle(people)
    cut = max(1, int(round(test_frac * len(people))))
    held = set(people[:cut].tolist())
    train_rows = np.array([i for i, p in enumerate(panel.participants)
                           if p not in held])
    test_cs = cs.subset(np.array([i for i, p in enumerate(cs.participants)
                                  if p in held]))
    sc_tr = panel.counts(train_rows)
    table = []
    for l2 in grid:
        try:
            w = estimate(sc_tr, kind, l2)
        except RuntimeError:
            continue
        score, _ = macro_by_sort(test_cs, w, ndcg_at_k)
        table.append((l2, float(score), w))
    if not table:
        return 0.0, []
    best = max(table, key=lambda t: t[1])
    return best[0], table


# --------------------------------------------------------------------------- #
# Correlation analysis
# --------------------------------------------------------------------------- #
def _task_share_and_features(sc: ShareCounts, task_id: str
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse a question to one share and one feature row per hotel."""
    rows = sc.panel.rows_for_task(task_id)
    exposed = sc.n_exposed[rows].sum(axis=0)
    chosen = sc.n_chosen[rows].sum(axis=0)
    keep = exposed > 0
    y = (chosen[keep] / exposed[keep])
    X = sc.panel.features[rows[0]][keep]
    return y, X


def question_correlations(sc: ShareCounts) -> Dict[str, Dict[str, dict]]:
    """Pearson and Spearman of each component against share, per question.

    Computed over the 32 hotels of a question, so n = 32 and the p-value is the
    ordinary test on r with 30 df. This is DESCRIPTIVE, not the estimator: a
    component can correlate with share purely because it correlates with the one
    that drove the choice (`spatial` and `accessibility` sit near r = 0.9 in
    this material). The logit is what separates them; these coefficients say
    what the raw percentages look like before anything is separated.
    """
    out: Dict[str, Dict[str, dict]] = {}
    for t in sc.panel.tasks():
        y, X = _task_share_and_features(sc, t)
        d: Dict[str, dict] = {}
        for i, dim in enumerate(DIMS):
            x = X[:, i]
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                continue
            r, pr = stats.pearsonr(x, y)
            rho, ps = stats.spearmanr(x, y)
            d[dim] = {"pearson_r": float(r), "pearson_p": float(pr),
                      "spearman_rho": float(rho), "spearman_p": float(ps),
                      "n_hotels": int(len(y))}
        out[t] = d
    return out


def pooled_correlations(sc: ShareCounts) -> Dict[str, dict]:
    """Correlation with share after removing each question's own mean.

    Pooling the 320 rows raw would confound two different things: how hotels
    differ INSIDE a question (the preference) and how questions differ from each
    other (the scenario). Centring both share and component within question
    leaves only the first, which is the quantity the weights describe.
    """
    ys, xs = [], []
    for t in sc.panel.tasks():
        y, X = _task_share_and_features(sc, t)
        ys.append(y - y.mean())
        xs.append(X - X.mean(axis=0))
    y = np.concatenate(ys)
    X = np.vstack(xs)
    out: Dict[str, dict] = {}
    for i, dim in enumerate(DIMS):
        r, p = stats.pearsonr(X[:, i], y)
        rho, ps = stats.spearmanr(X[:, i], y)
        out[dim] = {"pearson_r": float(r), "pearson_p": float(p),
                    "spearman_rho": float(rho), "spearman_p": float(ps),
                    "n_rows": int(len(y))}
    cm = np.corrcoef(X, rowvar=False)
    out["_component_correlation_matrix"] = {
        d: {e: float(cm[i, j]) for j, e in enumerate(DIMS)}
        for i, d in enumerate(DIMS)}
    return out


def correlation_weights(per_q: Dict[str, Dict[str, dict]]) -> np.ndarray:
    """Weights read straight off the correlation coefficients.

    Average each component's Pearson r across questions, clip negatives to zero,
    normalise to the simplex. This is a MODEL-FREE reference point, not a rival
    estimator: it ignores both the collinearity between components and where the
    hotel sat on the page, which the logit handles. It is reported because it is
    computed directly from the percentages a reader can see in the table, so a
    large gap between it and the fitted vector localises how much work those two
    nuisances are actually doing.
    """
    means = []
    for dim in DIMS:
        rs = [q[dim]["pearson_r"] for q in per_q.values() if dim in q]
        means.append(float(np.mean(rs)) if rs else 0.0)
    w = np.clip(np.array(means), 0.0, None)
    return w / w.sum() if w.sum() > 0 else np.full(5, 0.2)


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_ci(panel: SharePanel, n: int, seed: int, l2: float, estimator: str
                 ) -> Dict[str, List[float]]:
    """Resample PARTICIPANTS, recount the shares, refit.

    The percentages are estimates too, and their sampling unit is the person:
    each participant answered ~10 questions, so resampling rows of the share
    table would treat one person's ten decisions as ten independent facts and
    shrink the interval to fiction.
    """
    if n <= 0:
        return {}
    rng = np.random.default_rng(seed)
    people = np.unique(panel.participants)
    by = {p: np.where(panel.participants == p)[0] for p in people}
    draws = []
    for _ in range(n):
        pick = rng.choice(people, size=len(people), replace=True)
        rows = np.concatenate([by[p] for p in pick])
        try:
            draws.append(estimate(panel.counts(rows), estimator, l2))
        except Exception:               # a degenerate resample: skip it
            continue
    if not draws:
        return {}
    arr = np.stack(draws)
    ci: Dict[str, List[float]] = {
        d: [float(np.percentile(arr[:, i], 2.5)),
            float(np.percentile(arr[:, i], 97.5))] for i, d in enumerate(DIMS)}
    loc = arr[:, 0] + arr[:, 1]
    ci["spatial_plus_accessibility"] = [float(np.percentile(loc, 2.5)),
                                        float(np.percentile(loc, 97.5))]
    return ci


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def fmt(w: Sequence[float]) -> str:
    return "  ".join(f"{d[:5]}={v:.3f}" for d, v in zip(DIMS, w))


def iter_cells(sc: ShareCounts, by_display: bool):
    """Walk the share table one cell at a time, at either grain.

    The estimator may run on display-split rows, but the table a human reads is
    always the per-question one -- "60% picked Shangri-La on t1" is a statement
    about the question, not about the participants who happened to sort by
    distance. So the two grains are separated here rather than in the caller:
    collapsing display rows back to their question is a plain sum of counts,
    because the groups partition the responses.
    """
    panel = sc.panel
    if by_display:
        # Order variants of one (question, sort) are estimation detail; the
        # table a person reads merges them back.
        merged: Dict[tuple, List[int]] = {}
        for gi, key in enumerate(panel.groups):
            merged.setdefault(tuple(key[:2]), []).append(gi)
        blocks = [(rows, key) for key, rows in merged.items()]
    else:
        blocks = [(list(panel.rows_for_task(t)), (t,)) for t in panel.tasks()]
    for rows, key in blocks:
        exposed = sc.n_exposed[rows].sum(axis=0)
        chosen = sc.n_chosen[rows].sum(axis=0)
        sum_nlp = (sc.mean_nlp[rows] * sc.n_exposed[rows]).sum(axis=0)
        n_resp = float(sc.n_resp[rows].sum())
        share = np.where(exposed > 0, chosen / np.maximum(exposed, 1.0), 0.0)
        mean_nlp = np.where(exposed > 0, sum_nlp / np.maximum(exposed, 1.0), 0.0)
        order = np.argsort(-share)
        rank = np.empty(panel.n_hotels, dtype=int)
        rank[order] = np.arange(1, panel.n_hotels + 1)
        expected = sc.expected[rows].sum(axis=0)
        exp_var = sc.expected_var[rows].sum(axis=0)
        yield (key, n_resp, exposed, chosen, share, mean_nlp, rank, rows[0],
               expected, exp_var)


def write_rows_csv(sc: ShareCounts, path: Path, material: dict,
                   by_display: bool = False) -> int:
    """The artifact this whole module exists to produce: one row per cell."""
    panel = sc.panel
    hotels = material["hotels"]
    cols = ["task_id", "persona", "primary_dimension", "secondary_dimension"]
    if by_display:
        cols.append("sort_mode")
    cols += ["hotel_id", "hotel_name", "n_respondents", "n_shown", "n_chosen",
             "share", "share_pct", "share_ci95_lo", "share_ci95_hi",
             "expected_share_by_position", "position_lift", "position_z",
             "rank_in_group", "mean_neg_log_position",
             *DIMS, "price_lkr", "rating", "star", "distance_km"]
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for key, n_resp, exposed, chosen, share, mean_nlp, rank, gi, expected, exp_var in \
                iter_cells(sc, by_display):
            meta = panel.task_meta.get(key[0], {})
            for hi, hid in enumerate(panel.hotels):
                if exposed[hi] <= 0:
                    continue
                a = hotels.get(hid, {}).get("attributes", {})
                row = [key[0], meta.get("persona", ""),
                       meta.get("primary_dimension", ""),
                       meta.get("secondary_dimension", "")]
                if by_display:
                    row.append(key[1])
                lo, hi_ = wilson_interval(float(chosen[hi]), float(exposed[hi]))
                sd = float(np.sqrt(exp_var[hi]))
                row += [hid, panel.names.get(hid, hid),
                        int(n_resp), int(exposed[hi]), int(chosen[hi]),
                        round(float(share[hi]), 6),
                        round(100.0 * float(share[hi]), 2),
                        round(lo, 6), round(hi_, 6),
                        round(float(expected[hi] / max(n_resp, 1.0)), 6),
                        (round(float(chosen[hi] / expected[hi]), 4)
                         if expected[hi] > 0 else ""),
                        (round(float((chosen[hi] - expected[hi]) / sd), 3)
                         if sd > 0 else ""),
                        int(rank[hi]), round(float(mean_nlp[hi]), 4),
                        *[round(float(v), 4) for v in panel.features[gi, hi]],
                        a.get("price_lkr"), a.get("rating"), a.get("star"),
                        a.get("distance_km")]
                w.writerow(row)
                n += 1
    return n


def print_share_table(sc: ShareCounts, top: int) -> None:
    panel = sc.panel
    # Same walk as the CSV, so the table on screen and the table on disk can
    # never disagree about what a percentage means.
    for key, n_resp_f, exposed, chosen, share, _mnlp, _rank, gi, expected, exp_var in \
            iter_cells(sc, by_display=False):
        t, n_resp = key[0], int(n_resp_f)
        meta = panel.task_meta.get(t, {})
        print(f"\n{t}  {meta.get('persona', '')}  "
              f"[{meta.get('primary_dimension')}/{meta.get('secondary_dimension')}]"
              f"   n={n_resp}")
        shown = 0
        for hi in np.argsort(-share):
            if chosen[hi] == 0 or shown >= top:
                break
            f = panel.features[gi, hi]
            lo, hi_ = wilson_interval(float(chosen[hi]), float(exposed[hi]))
            sd = float(np.sqrt(exp_var[hi]))
            z = (chosen[hi] - expected[hi]) / sd if sd > 0 else float("nan")
            print(f"   {100.0 * share[hi]:5.1f}% [{100 * lo:4.1f}-{100 * hi_:4.1f}]  "
                  f"{int(chosen[hi]):>3}/{int(exposed[hi]):<3}  "
                  f"{panel.names[panel.hotels[hi]][:30]:30s} "
                  f"pos-expected {100.0 * expected[hi] / max(n_resp, 1):5.1f}%  z={z:+5.1f}  "
                  + " ".join(f"{d[:3]}={v:.2f}" for d, v in zip(DIMS, f)))
            shown += 1
        picked = int((chosen > 0).sum())
        never = int(((chosen == 0) & (exposed > 0)).sum())
        print(f"   ... {picked - shown} more hotels chosen at least once, "
              f"{never} shown but never chosen")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, default=None)
    ap.add_argument("--material", type=Path, default=MATERIAL)
    ap.add_argument("--out", type=Path, default=OUT / "share_weights.json")
    ap.add_argument("--csv", type=Path, default=OUT / "choice_shares.csv",
                    help="row-by-row table, one row per question x hotel")
    ap.add_argument("--csv-display", type=Path,
                    default=OUT / "choice_shares_by_display.csv",
                    help="the same table split by the sort the participant "
                         "chose (written only at --grain display)")
    ap.add_argument("--grain", default="display", choices=["question", "display"],
                    help="rows per question x sort-mode x hotel (default; the "
                         "position control is then exact and the fit is "
                         "identical to the individual-level one), or per "
                         "question x hotel, where position can only be averaged")
    ap.add_argument("--facility-def", default="all_ranks",
                    choices=["current", "rank_facilities", "all_ranks", "no_star"])
    ap.add_argument("--pool-size", type=int, default=32)
    ap.add_argument("--l2", default="auto",
                    help="ridge on the components: a number, or 'auto' to "
                         "select it on held-out participants")
    ap.add_argument("--test-frac", type=float, default=0.30,
                    help="participants held out for the --l2 auto search")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--top", type=int, default=6,
                    help="hotels to print per question in the share table")
    ap.add_argument("--drop-failed-attention", action="store_true")
    ap.add_argument("--shipped", default="per_question",
                    choices=["per_question", "pooled", "correlation"],
                    help="which vector is reported as THE estimate and printed "
                         "by --emit-profile")
    ap.add_argument("--position-spec", type=PositionSpec.parse, default=DEFAULT_POSITION,
                    help="dummies:K (default dummies:3), topk:K, neglog (reproduces "
                         "the 2026-09-07 share weights), none")
    ap.add_argument("--gates", action="store_true",
                    help="run the acceptance gates on the shipped estimator")
    ap.add_argument("--placebo-reps", type=int, default=200)
    ap.add_argument("--emit-profile", action="store_true",
                    help="print a ScoringWeights profile (refused unless --gates passed)")
    ap.add_argument("--allow-ungated", action="store_true")
    ap.add_argument("--check-equivalence", action="store_true",
                    help="refit the individual-level logit and report the gap")
    args = ap.parse_args()

    dump = args.dump or latest_dump()
    if not dump.exists():
        dump = DUMPS / dump.name
    participants, responses, version = load_dump(dump)
    material = load_material(args.material)
    facility = facility_scores(material, args.facility_def)
    failed = failed_attention(responses)

    panel = build_panel(responses, material, facility, args.pool_size,
                        args.grain, args.drop_failed_attention, failed,
                        position=args.position_spec)
    sc = panel.counts()
    never = int(((sc.n_chosen.sum(axis=0) == 0)
                 & (sc.n_exposed.sum(axis=0) > 0)).sum())

    print("=" * 78)
    print(f"ALMA-GraphRAG - weights from per-question CHOICE SHARES  "
          f"(material {version})")
    print("=" * 78)
    print(f"source dump   : {dump.name}")
    print(f"grain         : {panel.grain}  "
          f"({panel.n_groups} groups x {panel.n_hotels} hotels)")
    print(f"responses     : {len(panel.chosen)} from "
          f"{len(np.unique(panel.participants))} participants")
    print(f"facility def  : {args.facility_def}")
    print(f"position      : {args.position_spec.label()}  "
          f"(terms: {', '.join(panel.pos_names) or 'none'})")
    print(f"share rows    : {int(sc.mask.sum())}   "
          f"(hotels never chosen anywhere: {never} of {panel.n_hotels})")
    if panel.grain == "question":
        print("NOTE: at --grain question the rank control is an AVERAGE over "
              "sort modes, so\n      the fit is an approximation of the "
              "individual-level one, not an identity.\n      Use --grain "
              "display for the exact estimate; this grain is for reading.")

    print("\n" + "-" * 78)
    print(f"SELECTION SHARE BY QUESTION - top {args.top} of {panel.n_hotels}")
    print("-" * 78)
    print_share_table(sc, args.top)

    # ---- correlations ------------------------------------------------------- #
    per_q = question_correlations(sc)
    pooled = pooled_correlations(sc)
    print("\n" + "-" * 78)
    print("CORRELATION OF COMPONENT WITH SELECTION SHARE (Pearson r over the "
          "32 hotels)")
    print("-" * 78)
    print(f"{'question':10s} " + "  ".join(f"{d[:6]:>7s}" for d in DIMS)
          + "   primary dimension")
    for t, d in per_q.items():
        cells = []
        for dim in DIMS:
            if dim not in d:
                cells.append(f"{'--':>7s}")
                continue
            cells.append(f"{d[dim]['pearson_r']:+6.2f}"
                         + ("*" if d[dim]["pearson_p"] < 0.05 else " "))
        print(f"{t:10s} " + "  ".join(cells) + "   "
              + str(panel.task_meta.get(t, {}).get("primary_dimension", "")))
    print(f"\n{'pooled r':10s} " + "  ".join(
        f"{pooled[d]['pearson_r']:+6.2f}"
        + ("*" if pooled[d]["pearson_p"] < 0.05 else " ") for d in DIMS)
        + f"   (question-centred, n={pooled[DIMS[0]]['n_rows']})")
    print(f"{'spearman':10s} " + "  ".join(
        f"{pooled[d]['spearman_rho']:+6.2f}"
        + ("*" if pooled[d]["spearman_p"] < 0.05 else " ") for d in DIMS))
    print("  * p < 0.05")

    print("\ncomponent x component correlation (question-centred) - read the "
          "off-diagonals\nbefore reading any single weight as a preference:")
    cm = pooled["_component_correlation_matrix"]
    print(f"{'':14s}" + "".join(f"{d[:6]:>9s}" for d in DIMS))
    for d in DIMS:
        print(f"{d:14s}" + "".join(f"{cm[d][e]:+9.3f}" for e in DIMS))

    # ---- estimates ---------------------------------------------------------- #
    # ---- regularisation ----------------------------------------------------- #
    cs = build_choice_sets(responses, facility, args.pool_size,
                           args.drop_failed_attention, failed)
    l2_table: List[Tuple[float, float, np.ndarray]] = []
    if str(args.l2).lower() == "auto":
        grid = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        l2, l2_table = select_l2(panel, cs, grid, args.shipped,
                                 args.test_frac, args.seed)
        print("\n" + "-" * 78)
        print(f"RIDGE SELECTION on held-out participants "
              f"({int(round(100 * args.test_frac))}% of people, "
              f"estimator={args.shipped})")
        print("-" * 78)
        for lam, score, w in l2_table:
            print(f"  l2={lam:6.2f}  held-out macro nDCG@10 {score:.4f}   "
                  f"{fmt(w)}" + ("   <- selected" if lam == l2 else ""))
    else:
        l2 = float(args.l2)

    print("\n" + "-" * 78)
    print("WEIGHTS")
    print("-" * 78)
    per_q_w, w_macro, degenerate = fit_per_question(sc, l2=l2)
    print(f"\nper-question fits (each question's own logit over its 32 rows, "
          f"l2={l2:g}):")
    for t, w in per_q_w.items():
        print(f"  {t:4s} {fmt(w)}   <- "
              f"{panel.task_meta.get(t, {}).get('primary_dimension', '')}")
    if degenerate:
        print(f"  dropped (all components pinned at zero; the rank term "
              f"explained the question): {', '.join(degenerate)}")
    print(f"\n  MACRO-AVERAGE OVER {len(per_q_w)} QUESTIONS  {fmt(w_macro)}")

    b_pooled = fit_grouped(sc, l2=l2)
    w_pooled = to_simplex(b_pooled)
    print(f"  pooled over all rows           {fmt(w_pooled)}")
    print("  position terms                 "
          + "  ".join(f"{nm}={v:+.3f}" for nm, v in zip(panel.pos_names, b_pooled[5:]))
          + "   (fitted, then discarded)")

    b_nopos = fit_grouped(sc, use_position=False, l2=l2)
    print(f"  pooled, no position control    {fmt(to_simplex(b_nopos))}   "
          f"<- what the shares say if rank is ignored")

    w_corr = correlation_weights(per_q)
    print(f"  from correlation coefficients  {fmt(w_corr)}   <- model-free")
    print(f"  hand-set prior (shipped)       {fmt(HANDSET)}")

    shipped = {"per_question": w_macro, "pooled": w_pooled,
               "correlation": w_corr}[args.shipped]
    print(f"\n  reported estimate ({args.shipped}): {fmt(shipped)}")
    print(f"  location total (spatial+accessibility) = "
          f"{shipped[0] + shipped[1]:.3f}")

    # ---- bootstrap ---------------------------------------------------------- #
    ci = bootstrap_ci(panel, args.bootstrap, args.seed, l2, args.shipped)
    if ci:
        print(f"\nbootstrap 95% CI ({args.bootstrap} replicates, resampled by "
              f"participant, estimator={args.shipped}):")
        for i, d in enumerate(DIMS):
            lo, hi = ci[d]
            flag = "   includes ~0" if lo <= 0.02 else ""
            print(f"  {d:16s} {shipped[i]:.3f}  [{lo:.3f}, {hi:.3f}]{flag}")
        lo, hi = ci["spatial_plus_accessibility"]
        print(f"  {'spatial+access':16s} {shipped[0] + shipped[1]:.3f}  "
              f"[{lo:.3f}, {hi:.3f}]   <- the identified quantity")

    # ---- how well each vector reproduces the observed percentages ----------- #
    print("\n" + "-" * 78)
    print("FIT TO THE OBSERVED SHARES  (correlation between the composite score")
    print("and the measured selection percentage, averaged over questions)")
    print("-" * 78)
    cands = {"per_question macro": w_macro, "pooled": w_pooled,
             "correlation": w_corr, "handset prior": HANDSET}
    fitq: Dict[str, dict] = {}
    print(f"{'weights':22s} {'r(share,score)':>15s} {'rho':>8s} "
          f"{'nDCG@10':>9s} {'top-1':>8s}")
    for name, w in cands.items():
        rs, rhos = [], []
        for t in panel.tasks():
            y, X = _task_share_and_features(sc, t)
            s = X @ w
            if np.std(y) > 1e-12 and np.std(s) > 1e-12:
                rs.append(stats.pearsonr(s, y)[0])
                rhos.append(stats.spearmanr(s, y)[0])
        nd, _ = macro_by_sort(cs, w, ndcg_at_k)
        t1, _ = macro_by_sort(cs, w, top1)
        fitq[name] = {"share_pearson_r": float(np.mean(rs)),
                      "share_spearman_rho": float(np.mean(rhos)),
                      "macro_ndcg10": nd, "macro_top1": t1}
        print(f"{name:22s} {np.mean(rs):15.3f} {np.mean(rhos):8.3f} "
              f"{nd:9.3f} {t1:8.3f}")

    # ---- the sufficiency claim, checked rather than asserted ---------------- #
    equiv = None
    if args.check_equivalence:
        b_ind = fit_mnl(apply_position_spec(cs, args.position_spec),
                        use_position=True, non_negative=True, l2=l2)
        w_ind = to_simplex(b_ind)
        gap = float(np.max(np.abs(w_ind - w_pooled)))
        equiv = {"individual_level": {d: float(v) for d, v in zip(DIMS, w_ind)},
                 "aggregated_shares": {d: float(v) for d, v in zip(DIMS, w_pooled)},
                 "max_abs_difference": gap}
        print("\n" + "-" * 78)
        print("EQUIVALENCE CHECK - one likelihood, two representations")
        print("-" * 78)
        print(f"  individual choice sets ({len(cs)} rows) : {fmt(w_ind)}")
        print(f"  aggregated shares ({int(sc.mask.sum())} rows)      : {fmt(w_pooled)}")
        print(f"  max abs difference: {gap:.2e}"
              + ("   OK" if gap < 5e-3 else "   <- LARGER THAN EXPECTED"))

    # ---- concentration per question (A3) ----------------------------------- #
    question_summary: Dict[str, dict] = {}
    for key, n_resp_f, exposed, chosen, share_c, *_rest in iter_cells(sc, by_display=False):
        sh = chosen / max(n_resp_f, 1.0)
        hhi = float((sh ** 2).sum())
        question_summary[key[0]] = {
            "n_respondents": int(n_resp_f),
            "hotels_chosen_at_least_once": int((chosen > 0).sum()),
            "hotels_shown_never_chosen": int(((chosen == 0) & (exposed > 0)).sum()),
            "herfindahl": round(hhi, 4),
            "effective_number_of_hotels": round(1.0 / hhi, 2) if hhi > 0 else None,
            "top_share": round(float(share_c.max()), 4),
        }

    # ---- acceptance gates --------------------------------------------------- #
    gate_report = None
    if args.gates:
        from weight_elicitation.choice_model import choice_data_from_choice_sets
        from weight_elicitation.gates import GateConfig, evaluate_gates, format_report
        est_name = {"per_question": "per_task_macro", "pooled": "pooled_nonneg"}.get(args.shipped)
        if est_name is None:
            print("\n(gates skipped: the correlation estimator is not a choice model)")
        else:
            gate_report = evaluate_gates(
                choice_data_from_choice_sets(cs), est_name, estimator_kwargs={"l2": l2},
                spec=args.position_spec,
                config=GateConfig(placebo_reps=args.placebo_reps, seed=args.seed),
                name=f"fit_share_weights {args.shipped} ({est_name}, l2={l2:g})",
                progress=lambda m: print(f"  running {m}"))
            print("\n" + format_report(gate_report))
    shippable = bool(gate_report and gate_report["shippable"])

    # ---- persist ------------------------------------------------------------ #
    n_rows = write_rows_csv(sc, args.csv, material, by_display=False)
    n_disp = (write_rows_csv(sc, args.csv_display, material, by_display=True)
              if panel.grain == "display" else 0)
    payload = {
        "source_dump": dump.name,
        "material_version": version,
        "estimator": ("grouped (aggregate) conditional logit on per-question "
                      f"selection shares, {args.position_spec.label()} position terms, "
                      f"non-negative, macro-averaged over questions; grain={panel.grain}"),
        "position_spec": args.position_spec.label(),
        "position_terms": panel.pos_names,
        "shippable": shippable,
        "shippable_note": ("passed every acceptance gate" if shippable else
                           "NOT SHIPPABLE: " + ("failed the acceptance gates"
                                                if gate_report else
                                                "acceptance gates not run (--gates)")
                           + ". Report the shares; do not deploy these weights."),
        "gates": gate_report if gate_report else "not run (use --gates)",
        "question_summary": question_summary,
        "grain": panel.grain,
        "seed": args.seed,
        "l2": float(l2),
        "l2_selection": ([{"l2": float(lam), "held_out_macro_ndcg10": float(sc_),
                           "weights": dict(zip(DIMS, [round(float(x), 4) for x in w]))}
                          for lam, sc_, w in l2_table] or None),
        "questions_dropped_degenerate": degenerate,
        "facility_definition": args.facility_def,
        "cohort": {"responses": int(len(panel.chosen)),
                   "participants": int(len(np.unique(panel.participants))),
                   "groups": panel.n_groups,
                   "hotels": panel.n_hotels,
                   "share_rows": int(sc.mask.sum())},
        "shipped_estimator": args.shipped,
        "weights": dict(zip(DIMS, [round(float(x), 4) for x in shipped])),
        "weights_per_question": {t: dict(zip(DIMS, [round(float(x), 4) for x in w]))
                                 for t, w in per_q_w.items()},
        "weights_macro_over_questions": dict(
            zip(DIMS, [round(float(x), 4) for x in w_macro])),
        "weights_pooled": dict(zip(DIMS, [round(float(x), 4) for x in w_pooled])),
        "weights_pooled_no_position": dict(
            zip(DIMS, [round(float(x), 4) for x in to_simplex(b_nopos)])),
        "weights_from_correlation": dict(
            zip(DIMS, [round(float(x), 4) for x in w_corr])),
        "position_coefficients": dict(zip(panel.pos_names, map(float, b_pooled[5:]))),
        "correlations_per_question": per_q,
        "correlations_pooled": pooled,
        "bootstrap_ci_95": ci,
        "fit_quality": fitq,
        "equivalence_check": equiv,
        "shares_csv": _rel(args.csv),
        "shares_by_display_csv": (_rel(args.csv_display)
                                  if n_disp else None),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {_rel(args.csv)}  ({n_rows} rows, "
          f"one per question x hotel)")
    if n_disp:
        print(f"wrote {_rel(args.csv_display)}  ({n_disp} rows, "
              f"split by sort mode)")
    print(f"wrote {_rel(args.out)}")

    if args.emit_profile:
        if not shippable and not args.allow_ungated:
            raise SystemExit("\nREFUSING --emit-profile: share weights have not passed "
                             "the acceptance gates (see shippable_note in the artifact).")
        if not shippable:
            print("\n# UNGATED - failed or skipped the acceptance gates; do not ship")
        print("\n# paste into src/graph/retriever.py")
        print("SHARE_WEIGHTS = ScoringWeights(")
        print("    " + ", ".join(f"{d}={v:.3f}" for d, v in zip(DIMS, shipped)) + ",")
        print(")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
