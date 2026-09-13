"""
Weight elicitation — the analysis half of the discrete-choice experiment.

This package owns everything that *reasons about* the study: generating its
material, checking the design is identifiable, exporting the labelled ranking
dataset, and fitting the retriever's composite-score weights.

    build_material.py         builds the frozen material from live LiteAPI data
    check_identifiability.py  can this design recover the weights? (run BEFORE fielding)
    export_dataset.py         responses -> learning-to-rank dataset
    fit_weights.py            responses -> ScoringWeights for src/graph/retriever.py
    fit_human_weights.py      the same, per display condition, study data only
    fit_share_weights.py      per-question choice SHARES -> weights + correlations

The seam
--------
`studies/weight-elicitation/` is a SEPARATE deployable sub-project: a Next.js app
whose only job is to show the frozen material to participants and record what
they choose. It is deliberately not a Python project, and nothing here is
importable from it.

Traffic across the seam runs in exactly two directions, both through files:

    build_material.py  ---(writes)--->  the app's material/study_material_v1.json
    the hosted app     ---(exports)-->  data/study_data_*.json  --->  fit_weights.py

That is why the material file lives inside the app rather than here: `lib/material.ts`
imports it directly, so Next bundles it at build time and the deployed study
never needs this repo, a database, Neo4j or pgvector at runtime.

Fitted weights land in `src/graph/retriever.py` as named profiles; the findings
they rest on are in `docs/Weight_Elicitation_Data_Audit.md`.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# --- the Next.js data-collection sub-project (not a Python package) ---------- #
STUDY_APP = REPO / "studies" / "weight-elicitation"

#: Frozen study material. Lives in the app because Next bundles it at build time.
MATERIAL = STUDY_APP / "material" / "study_material_v1.json"

#: The app's local JSONL fallback, written only when DATABASE_URL is unset.
#: Production responses go to Postgres and arrive here via the admin export.
LOCAL_CAPTURE = STUDY_APP / "data"

# --- this package ----------------------------------------------------------- #
#: Raw `?format=raw` exports downloaded from the hosted study.
DUMPS = HERE / "data"

#: Fitted weights, datasets and reports.
OUT = HERE / "out"


def latest_dump() -> Path:
    """Newest `study_data_*.json` in `data/`.

    The dump is named for the material version it was collected under, so
    picking the newest keeps a refit pointed at the current wave without anyone
    having to paste a filename. Raises if there is none — silently fitting on
    nothing would be worse than failing.
    """
    candidates = sorted(DUMPS.glob("study_data_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"no study_data_*.json in {DUMPS}. Download one from the hosted "
            f"study's admin page (Download JSON / ?format=raw) and put it there."
        )
    return candidates[-1]


__all__ = ["HERE", "REPO", "STUDY_APP", "MATERIAL", "LOCAL_CAPTURE",
           "DUMPS", "OUT", "latest_dump"]
