"""
Evaluation API — serves the comparative-evaluation walkthrough UI (eval.html).

  GET  /eval/queryset          — the 50-query evaluation set + gold predicates
  GET  /eval/results           — latest aggregate results (cached results.json)
  GET  /eval/inspect/{qid}     — live per-query trace: each system's ranked list
                                 with relevance flags, metrics, and the GraphRAG
                                 composite-score components
  POST /eval/run               — re-run the full evaluation and refresh results.json

Endpoints that touch Neo4j return 503 with a readable hint when the graph is
unreachable or empty, so the UI can degrade gracefully.
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException

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

logger = logging.getLogger("alma.eval")

router = APIRouter(prefix="/eval", tags=["evaluation"])


@router.get("/queryset")
def get_queryset() -> dict:
    """The evaluation query set (questions, categories, rule-based gold)."""
    try:
        return load_spec(DEFAULT_QUERYSET)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="queryset.json not found") from exc


@router.get("/results")
def get_results() -> dict:
    """Latest cached aggregate results. Returns {"available": false} when the
    harness has not been run yet (so the UI can prompt for a live run)."""
    if not DEFAULT_RESULTS.exists():
        return {"available": False}
    data = json.loads(DEFAULT_RESULTS.read_text(encoding="utf-8"))
    data["available"] = True
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
