"""The per-question share fit must be the choice model, not a summary of it.

`weight_elicitation/fit_share_weights.py` reads the study as one row per
QUESTION x HOTEL carrying the percentage of participants who picked that hotel,
and fits the composite weights from those percentages. That is only legitimate
if the aggregation is lossless, so the central test here is an EQUIVALENCE test:
the grouped fit on shares and the individual fit on choice sets must return the
same vector, not merely a similar one.

The synthetic tests run offline on a generated dump with a known true weight
vector, so a reviewer can check the estimator without the 19 MB study export.
The one test that needs the real dump skips when it is absent.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from weight_elicitation import DUMPS, MATERIAL
from weight_elicitation.fit_weights import DIMS, build_choice_sets, fit_mnl, to_simplex
from weight_elicitation.fit_share_weights import (
    build_panel,
    correlation_weights,
    fit_grouped,
    fit_per_question,
    question_correlations,
    select_l2,
)

TRUE = np.array([0.35, 0.10, 0.25, 0.20, 0.10])
SORTS = ["distance", "travel", "rating"]


# --------------------------------------------------------------------------- #
# A synthetic study with a known answer
# --------------------------------------------------------------------------- #
def make_study(n_participants=400, n_hotels=10, n_tasks=4, scale=6.0,
               gamma=0.0, seed=7):
    """Generate a dump whose choices really were made by `TRUE`.

    Components are drawn per (task, hotel) and never per participant, which is
    the invariance the aggregation rests on. Position is a fixed function of
    (task, sort mode, hotel), as it is in the real study, so the individual and
    grouped representations see the same rank column and any difference between
    them must come from the estimator rather than from the data.
    """
    rng = np.random.default_rng(seed)
    hotels = [f"h{i}" for i in range(n_hotels)]
    tasks = [f"t{i + 1}" for i in range(n_tasks)]
    comps = {t: rng.uniform(0.05, 0.95, size=(n_hotels, 5)) for t in tasks}
    # Position: each sort mode orders the pool by a different component, which
    # is what makes rank correlate with the components in the real study.
    order = {(t, s): np.argsort(-comps[t][:, i % 5])
             for t in tasks for i, s in enumerate(SORTS)}
    position = {}
    for key, idx in order.items():
        pos = np.empty(n_hotels, dtype=int)
        pos[idx] = np.arange(1, n_hotels + 1)
        position[key] = pos

    responses = []
    for p in range(n_participants):
        pid = f"p{p}"
        sort = SORTS[p % len(SORTS)]
        for t in tasks:
            pos = position[(t, sort)]
            u = scale * (comps[t] @ TRUE) + gamma * (-np.log(pos))
            prob = np.exp(u - u.max())
            prob /= prob.sum()
            pick = int(rng.choice(n_hotels, p=prob))
            responses.append({
                "participantId": pid,
                "taskId": t,
                "isAttentionCheck": False,
                "timing": {"final_sort": sort},
                "options": [
                    {"hotel_id": hotels[h],
                     "components": dict(zip(DIMS, comps[t][h].tolist())),
                     "displayed_position": int(pos[h]),
                     "chosen": h == pick}
                    for h in range(n_hotels)],
            })
    material = {
        "hotels": {h: {"name": h.upper(), "attributes": {}} for h in hotels},
        "tasks": [{"id": t, "persona": t} for t in tasks],
    }
    return responses, material, comps


@pytest.fixture(scope="module")
def study():
    return make_study()


def panel_for(responses, material, grain, pool_size=10):
    return build_panel(responses, material, {}, pool_size, grain, False, set())


# --------------------------------------------------------------------------- #
# The share table itself
# --------------------------------------------------------------------------- #
def test_every_hotel_gets_a_row_including_the_unchosen(study):
    """A hotel nobody picked is a measured 0%, not a missing row.

    This is the whole reason the table is built at question grain: the rejected
    alternatives carry the information, and dropping them would leave the fit
    looking only at winners.
    """
    responses, material, _ = study
    sc = panel_for(responses, material, "question").counts()
    assert sc.mask.all()                         # 4 questions x 10 hotels
    assert int(sc.mask.sum()) == 40
    assert (sc.n_exposed > 0).all()


def test_shares_are_proportions_that_account_for_everyone(study):
    """Counts must reconcile: every respondent picked exactly one hotel."""
    responses, material, _ = study
    sc = panel_for(responses, material, "question").counts()
    assert np.allclose(sc.n_chosen.sum(axis=1), sc.n_resp)
    assert ((sc.share >= 0) & (sc.share <= 1)).all()
    # With an unfiltered pool everyone saw everything, so the availability
    # offset must be exactly zero rather than merely small.
    _, logavail = sc.design(use_position=True)
    assert np.allclose(logavail, 0.0)


def test_components_varying_within_a_question_is_refused(study):
    """If the alternatives differ per participant, shares stop being sufficient.

    Better to fail loudly than to average two different hotels into one row.
    """
    responses, material, _ = study
    corrupted = [dict(r) for r in responses]
    bad = dict(corrupted[0])
    opts = [dict(o) for o in bad["options"]]
    opts[0] = dict(opts[0], components=dict(opts[0]["components"], spatial=0.123))
    bad["options"] = opts
    corrupted[0] = bad
    with pytest.raises(RuntimeError, match="sufficient statistic"):
        panel_for(corrupted, material, "question")


# --------------------------------------------------------------------------- #
# The equivalence claim
# --------------------------------------------------------------------------- #
def test_grouped_fit_equals_individual_fit_on_synthetic_data(study):
    """Same likelihood, two representations: 40 share rows vs 1,600 choices."""
    responses, material, _ = study
    sc = panel_for(responses, material, "display").counts()
    w_grouped = to_simplex(fit_grouped(sc, use_position=True, non_negative=True))
    cs = build_choice_sets(responses, {}, 10, False, set())
    w_individual = to_simplex(
        fit_mnl(cs, use_position=True, non_negative=True))
    assert np.max(np.abs(w_grouped - w_individual)) < 1e-4


@pytest.mark.skipif(not (DUMPS / "study_data_v4-rooms-20260818.json").exists()
                    or not MATERIAL.exists(),
                    reason="the study dump is not checked out here")
def test_grouped_fit_equals_individual_fit_on_the_real_study():
    """The same identity, on the data the thesis actually reports."""
    from weight_elicitation.fit_weights import (facility_scores, failed_attention,
                                                load_dump, load_material)
    _, responses, _ = load_dump(DUMPS / "study_data_v4-rooms-20260818.json")
    material = load_material(MATERIAL)
    facility = facility_scores(material, "all_ranks")
    failed = failed_attention(responses)
    sc = build_panel(responses, material, facility, 32, "display", False,
                     failed).counts()
    w_grouped = to_simplex(fit_grouped(sc))
    cs = build_choice_sets(responses, facility, 32, False, failed)
    w_individual = to_simplex(fit_mnl(cs, use_position=True, non_negative=True))
    assert np.max(np.abs(w_grouped - w_individual)) < 5e-3


# --------------------------------------------------------------------------- #
# Recovery and the guards
# --------------------------------------------------------------------------- #
def test_the_fit_recovers_the_weights_that_generated_the_choices(study):
    """An estimator that cannot recover a known answer cannot be trusted here."""
    responses, material, _ = study
    sc = panel_for(responses, material, "display").counts()
    w = to_simplex(fit_grouped(sc, use_position=True, non_negative=True))
    assert np.max(np.abs(w - TRUE)) < 0.08
    assert np.argmax(w) == int(np.argmax(TRUE))


def test_position_bias_is_absorbed_rather_than_booked_as_preference():
    """With rank driving the choices, the components must NOT soak it up.

    The generated data has a strong position effect and the ordering is by
    `spatial`, so an uncontrolled fit inflates `spatial`. The controlled fit is
    the one that must stay near the truth; this is the failure mode the log-rank
    column exists to prevent.
    """
    responses, material, _ = make_study(gamma=3.0, seed=11)
    sc = panel_for(responses, material, "display").counts()
    controlled = to_simplex(fit_grouped(sc, use_position=True, non_negative=True))
    naive = to_simplex(fit_grouped(sc, use_position=False, non_negative=True))
    err_controlled = np.max(np.abs(controlled - TRUE))
    err_naive = np.max(np.abs(naive - TRUE))
    assert err_controlled < err_naive
    assert err_controlled < 0.10


def test_a_degenerate_question_is_dropped_not_averaged_as_uniform():
    """A collapsed fit must not enter the average disguised as 0.2 each.

    `to_simplex` maps an all-zero coefficient vector to a flat simplex, which
    reads like a finding ("this scenario valued everything equally") when it is
    really the optimiser declining to give any component mass. The question with
    identical components carries no information at all, so it must be named in
    the dropped list instead.
    """
    # Its own study: this test flattens a question in place, and the module
    # fixture is shared with every test after it.
    responses, material, _ = make_study(seed=23)
    for r in [r for r in responses if r["taskId"] == "t1"]:
        r["options"] = [dict(o, components=dict(zip(DIMS, [0.5] * 5)))
                        for o in r["options"]]
    sc = panel_for(responses, material, "display").counts()
    per, macro, dropped = fit_per_question(sc)
    assert "t1" in dropped
    assert "t1" not in per
    assert not np.allclose(macro, 0.2)
    assert np.isclose(macro.sum(), 1.0)


def test_correlation_weights_are_a_simplex(study):
    responses, material, _ = study
    sc = panel_for(responses, material, "question").counts()
    w = correlation_weights(question_correlations(sc))
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()


def test_correlations_carry_a_coefficient_and_a_p_value(study):
    responses, material, _ = study
    sc = panel_for(responses, material, "question").counts()
    per_q = question_correlations(sc)
    assert set(per_q) == {"t1", "t2", "t3", "t4"}
    for stats_by_dim in per_q.values():
        for dim in DIMS:
            cell = stats_by_dim[dim]
            assert -1.0 <= cell["pearson_r"] <= 1.0
            assert 0.0 <= cell["pearson_p"] <= 1.0
            assert cell["n_hotels"] == 10


def test_l2_is_selected_on_held_out_people_and_is_reproducible(study):
    """Two calls with the same seed must choose the same ridge.

    The split is over participants, so a drifting seed would silently change
    both the selected value and the vector that ships.
    """
    responses, material, _ = study
    panel = panel_for(responses, material, "display")
    cs = build_choice_sets(responses, {}, 10, False, set())
    grid = [0.0, 1.0, 4.0]
    first, table = select_l2(panel, cs, grid, "pooled", 0.3, 20260907)
    second, _ = select_l2(panel, cs, grid, "pooled", 0.3, 20260907)
    assert first == second
    assert len(table) == len(grid)
    held_out_people = {p for p in panel.participants}
    assert len(held_out_people) == 400
