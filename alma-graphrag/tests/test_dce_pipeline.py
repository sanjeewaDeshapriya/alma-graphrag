"""Wave-2 DCE pipeline: design -> simulated study -> fit -> gates -> profile.

Runs on a small synthetic material (no LiteAPI, no study app) shaped exactly
like the real one: hotels with `components_global`, anchor components, tasks.
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from weight_elicitation.choice_model import DIMS, PositionSpec, choice_data_from_responses, fit_model
from weight_elicitation.design_choice_sets import (
    apply_to_material,
    build_design,
    dominated_pairs,
    noise_level,
    set_information,
)
from weight_elicitation.fit_weights import facility_scores
from weight_elicitation.check_identifiability import load_design
from weight_elicitation.power_analysis import simulate_study


def make_material(n_hotels=16, n_tasks=6, seed=3):
    rng = np.random.default_rng(seed)
    hotels, anchor = {}, {}
    for i in range(n_hotels):
        hid = f"h{i}"
        hotels[hid] = {
            "name": f"Hotel {i}",
            "attributes": {"price_lkr": int(rng.integers(10_000, 90_000)),
                           "star": int(rng.integers(3, 6)), "rating": float(rng.uniform(3.5, 4.9))},
            "detail": {"n_facilities": int(rng.integers(10, 90))},
            "rooms": [{"id": f"{hid}-r1", "price_lkr": 20000}],
            "components_global": {"facility": float(rng.uniform()),
                                  "economic": float(rng.uniform()),
                                  "disruption": float(rng.uniform())},
        }
        anchor[hid] = {"spatial": float(rng.uniform()), "accessibility": float(rng.uniform()),
                       "distance_km": 1.0, "travel_min": 5}
    tasks = [{"id": f"t{t + 1}", "anchor_id": "a", "persona": "p", "context": "c",
              "primary_dimension": DIMS[t % 5], "secondary_dimension": None,
              "is_attention_check": False, "option_ids": list(hotels)} for t in range(n_tasks)]
    tasks.append({"id": "t_attn", "anchor_id": "a", "persona": "Attention", "context": "pick h0",
                  "primary_dimension": None, "secondary_dimension": None,
                  "is_attention_check": True, "attention_answer_hotel_id": "h0",
                  "option_ids": list(hotels)})
    return {"version": "vtest", "design": {}, "hotels": hotels,
            "anchor_components": {"a": anchor}, "tasks": tasks}


@pytest.fixture(scope="module")
def designed():
    material = make_material()
    design, diag, tasks = build_design(material, set_size=4, n_blocks=3, restarts=2, seed=1,
                                       max_passes=6, random_baseline=10)
    return material, design, diag, tasks, apply_to_material(material, design, tasks, diag,
                                                            seed=1, facility_def="all_ranks")


def test_sets_have_the_right_size_and_no_duplicates(designed):
    _, design, _, tasks, out = designed
    for t in out["tasks"]:
        assert len(t["choice_sets"]) == 3
        for s in t["choice_sets"]:
            assert len(s) == 4 and len(set(s)) == 4
        assert set(t["option_ids"]) == {h for s in t["choice_sets"] for h in s}


def test_attention_sets_contain_the_answer(designed):
    out = designed[4]
    attn = next(t for t in out["tasks"] if t["is_attention_check"])
    assert all("h0" in s for s in attn["choice_sets"])


def test_design_is_more_efficient_than_random(designed):
    diag = designed[2]
    assert diag["d_error"] < diag["random_design_d_error_mean"]
    assert diag["dominated_pairs"] == 0


def test_material_records_the_design_and_readable_noise_labels(designed):
    out = designed[4]
    assert out["design"]["mode"] == "dce"
    assert out["design"]["set_size"] == 4 and out["design"]["n_blocks"] == 3
    assert "hashSeed" in out["design"]["block_assignment"]
    assert all(h["attributes"]["noise_level"] for h in out["hotels"].values())
    assert out["version"].endswith("+dce4x3")


def test_identifiability_reads_each_presented_dce_choice_set(designed, tmp_path):
    material = tmp_path / "material_dce.json"
    material.write_text(json.dumps(designed[4]), encoding="utf-8")
    designs, labels = load_design(material)
    assert len(designs) == 18  # 6 non-attention tasks x 3 blocks
    assert all(matrix.shape == (4, 5) for matrix in designs)
    assert all("block" in label for label in labels)


def test_information_and_dominance_helpers():
    X = np.array([[1.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0], [0.5, 0.5, 0, 0, 0]])
    info = set_information(X, np.zeros(5))
    assert np.allclose(info, info.T) and np.all(np.linalg.eigvalsh(info) >= -1e-12)
    assert dominated_pairs(np.array([[1.0] * 5, [0.5] * 5])) == 1
    assert noise_level(0.9) == "Quiet street" and noise_level(0.1) == "Busy, noisy area"


def test_simulated_study_recovers_known_coefficients(designed):
    out = designed[4]
    beta = np.array([1.5, 0.0, 1.0, 0.8, 0.0])
    rng = np.random.default_rng(4)
    resp = simulate_study(out, beta, [-0.3, -0.6, -0.9], 400, rng, facility_scores(out, "all_ranks"))
    data = choice_data_from_responses(resp)
    assert set(np.unique([len(r["options"]) for r in resp])) == {4}
    f = fit_model(data, PositionSpec("dummies", 4))
    assert np.all(np.abs(f.components - beta) < 4 * f.se[:5])
    assert f.components[0] / f.se[0] > 3


def test_fit_dce_end_to_end_emits_a_declared_profile(designed, tmp_path):
    out = designed[4]
    mpath = tmp_path / "material_dce.json"
    mpath.write_text(json.dumps(out), encoding="utf-8")
    rng = np.random.default_rng(9)
    resp = simulate_study(out, np.array([1.6, 0.0, 1.2, 1.0, 0.0]), [-0.3, -0.6, -0.9], 300, rng,
                          facility_scores(out, "all_ranks"))
    dump = tmp_path / "study_data_sim.json"
    dump.write_text(json.dumps({"participants": [], "responses": resp, "version": out["version"]}),
                    encoding="utf-8")
    result = tmp_path / "dce_weights.json"
    proc = subprocess.run([sys.executable, "-m", "weight_elicitation.fit_dce", "--dump", str(dump),
                           "--material", str(mpath), "--bootstrap", "30", "--placebo-reps", "25",
                           "--out", str(result), "--emit-profile"],
                          capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr[-2000:]
    payload = json.loads(result.read_text(encoding="utf-8"))
    ident = payload["gates"]["identified_dimensions"]
    assert "spatial" in ident and "facility" in ident
    assert "accessibility" not in ident and "disruption" not in ident
    assert payload["shippable"] is False
    assert "DECLARED from the hand-set prior" in proc.stdout
    assert abs(sum(payload["deployable"]["weights"].values()) - 1) < 1e-9


def test_fit_dce_refuses_browse_material(tmp_path):
    material = make_material()
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(material), encoding="utf-8")
    dump = tmp_path / "d.json"
    dump.write_text(json.dumps({"participants": [], "responses": [], "version": "v"}), encoding="utf-8")
    proc = subprocess.run([sys.executable, "-m", "weight_elicitation.fit_dce", "--dump", str(dump),
                           "--material", str(mpath)], capture_output=True, text=True)
    assert proc.returncode != 0 and "not a DCE design" in (proc.stderr + proc.stdout)
