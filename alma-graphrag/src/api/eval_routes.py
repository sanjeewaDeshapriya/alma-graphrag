"""
Evaluation API — serves the comparative-evaluation walkthrough UI (eval.html).

  GET  /eval/queryset          — the 60-query evaluation set + gold predicates
  GET  /eval/results?run=…     — aggregate results. run=thesis (default) is the
                                 frozen run the thesis reports
                                 (results_final_77hotels.json), run=price the
                                 20-query price set, run=latest the last live
                                 run (results.json)
  GET  /eval/inspect/{qid}     — live per-query trace: each system's ranked list
                                 with relevance flags, metrics, and the GraphRAG
                                 composite-score components
  POST /eval/run               — re-run the full evaluation and refresh results.json
  GET  /eval/human/seeds       — held-out choice results over 50 person splits
  GET  /eval/weights           — weights estimated from the choice study + gates

Endpoints that touch Neo4j return 503 with a readable hint when the graph is
unreachable or empty, so the UI can degrade gracefully.
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Query

from evaluation.harness import (
    DEFAULT_GOLD_HUMAN,
    DEFAULT_QUERYSET,
    DEFAULT_RESULTS,
    inspect_query,
    load_spec,
    run_evaluation,
)
from evaluation.human_eval import (
    DEFAULT_HUMAN_RESULTS,
    DEFAULT_RESPONSES,
    run_human_evaluation,
)
from evaluation.comparative import DEFAULT_COMPARATIVE_RESULTS


logger = logging.getLogger("alma.eval")

router = APIRouter(prefix="/eval", tags=["evaluation"])


@router.get("/queryset")
def get_queryset() -> dict:
    """The evaluation query set (questions, categories, rule-based gold)."""
    try:
        return load_spec(DEFAULT_QUERYSET)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="queryset.json not found") from exc


EVAL_DIR = DEFAULT_RESULTS.parent
# The thesis reports frozen runs, not whatever the last button press produced:
# the live graph keeps changing (traffic, prices), so a fresh run drifts a few
# thousandths from the numbers quoted in Chapter 5.
RESULT_RUNS = {
    "thesis": EVAL_DIR / "results_final_77hotels.json",
    "price": EVAL_DIR / "results_price_77hotels.json",
    "latest": DEFAULT_RESULTS,
}
DEFAULT_SEEDS_RESULTS = EVAL_DIR / "results_human_seeds.json"
DEFAULT_WEIGHTS = EVAL_DIR.parent / "weight_elicitation" / "out" / "human_weights.json"


def _read_json(path) -> dict:
    if not path.exists():
        return {"available": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    data["available"] = True
    data["source_file"] = path.name
    return data


@router.get("/results")
def get_results(run: str = Query("thesis", pattern="^(thesis|price|latest)$")) -> dict:
    """Cached aggregate results for one run. Returns {"available": false} when
    that run has not been produced (so the UI can prompt for a live run)."""
    data = _read_json(RESULT_RUNS[run])
    data["run"] = run
    return data


@router.get("/inspect/{query_id}")
def get_inspection(query_id: str) -> dict:
    """Live per-query trace across all systems."""
    try:
        return inspect_query(query_id, DEFAULT_QUERYSET, DEFAULT_GOLD_HUMAN)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown query id: {query_id}") from exc
    except Exception as exc:  # Neo4j down / empty pool / retrieval error
        logger.exception("Query inspection failed")
        raise HTTPException(
            status_code=503,
            detail=f"Evaluation retrieval failed (is Neo4j up and seeded?): {exc}",
        ) from exc


@router.post("/run")
def post_run() -> dict:
    """Re-run the full evaluation, refresh results.json, and return the summary."""
    try:
        # The LLM re-ranker is excluded from the UI-triggered run: it bills a
        # request per query and this endpoint is one button click away. Run it
        # deliberately from the CLI instead:
        #     python evaluation/run_eval.py
        out = run_evaluation(DEFAULT_QUERYSET, DEFAULT_GOLD_HUMAN,
                             include_llm=False)
    except Exception as exc:
        logger.exception("Evaluation run failed")
        raise HTTPException(
            status_code=503,
            detail=f"Evaluation run failed (is Neo4j up and seeded?): {exc}",
        ) from exc
    DEFAULT_RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    out["available"] = True
    return out


# ---------------------------------------------------------------------------
# Choice-based evaluation (human ground truth)
# ---------------------------------------------------------------------------
#
# The rule-based evaluation above grades against evaluation/gold.py — predicates
# the author wrote. These endpoints grade against what 247 discrete-choice study
# participants actually booked. The two answer different questions and are shown
# side by side in the UI; neither supersedes the other.

@router.get("/human/results")
def get_human_results() -> dict:
    """Latest cached choice-based results, or {"available": false}."""
    if not DEFAULT_HUMAN_RESULTS.exists():
        return {"available": False}
    return json.loads(DEFAULT_HUMAN_RESULTS.read_text(encoding="utf-8"))


@router.get("/human/seeds")
def get_human_seeds() -> dict:
    """Held-out choice results repeated over 50 random person splits."""
    return _read_json(DEFAULT_SEEDS_RESULTS)


@router.get("/weights")
def get_weights() -> dict:
    """Weights estimated from the choice study, per sort order, with the five
    acceptance gates (G1-G5) they were tested against."""
    return _read_json(DEFAULT_WEIGHTS)


@router.get("/comparative/results")
def get_comparative_results() -> dict:
    """Rule-based and held-out human-choice metrics in separate columns."""
    if not DEFAULT_COMPARATIVE_RESULTS.exists():
        return {"available": False}
    data = json.loads(DEFAULT_COMPARATIVE_RESULTS.read_text(encoding="utf-8"))
    data["available"] = True
    return data


@router.post("/human/run")
def post_human_run(anchor_fair: bool = True, cohort: str = "clean") -> dict:
    """Re-run the choice-based evaluation and refresh results_human.json.

    `anchor_fair=false` scores the text baselines through the deployed pgvector
    index instead of rebuilding anchor-relative documents — production-faithful,
    but the baselines then cannot see the task anchor.
    """
    try:
        out = run_human_evaluation(anchor_fair=anchor_fair, cohort=cohort)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Study data missing (expected {DEFAULT_RESPONSES}): {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Choice-based evaluation failed")
        raise HTTPException(
            status_code=503, detail=f"Choice-based evaluation failed: {exc}") from exc
    DEFAULT_HUMAN_RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
