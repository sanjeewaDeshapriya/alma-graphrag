"""
CRAG (Corrective RAG) pipeline for ALMA-GraphRAG.

Implements a simple sequential CRAG loop:
  retrieve → grade → (optional rewrite+retrieve) → generate

The LangGraph StateGraph definition is kept for reference/future use
but the production `run_crag()` uses a direct sequential flow to avoid
LangGraph recursion-limit issues.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Dict, TypedDict

from openai import OpenAI

from src.config import (
    CRAG_MAX_RETRIES,
    CRAG_MIN_SCORE,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
)
from src.crag.cache import Cache
from src.crag.query_parser import parse_query
from src.graph.query import build_graph_context
from src.graph.retriever import WeightedRetriever, format_retrieval_context

logger = logging.getLogger("alma.crag")

# Reusable LLM client (OpenAI or Gemini via OpenAI-compatible endpoint)
_client = (
    OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL) if LLM_API_KEY else None
)

# Feasibility-first weighted GraphRAG retriever (proposal Algorithm 1)
_retriever = WeightedRetriever()


def _extract_json(text: str) -> str:
    """Strip markdown code fences (```json ... ```) from LLM output.

    gpt-4o-mini often wraps JSON in markdown fences which breaks json.loads().
    """
    import re
    # Match ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class GraphState(TypedDict, total=False):
    question: str
    city: str
    context: str
    score: float
    answer: str
    retries: int
    ranked_ids: list  # ordered hotel ids from weighted retrieval (for evaluation)


# ---------------------------------------------------------------------------
# CRAG node functions
# ---------------------------------------------------------------------------

def _retrieve(state: GraphState) -> GraphState:
    """Feasibility-first weighted multi-hop retrieval (proposal Algorithm 1).

    1. Parse the natural-language question into a structured QueryIntent
       (safe NL->Cypher slot-filling).
    2. Run the weighted multi-hop retriever to rank hotels by a composite of
       spatial / accessibility / facility / economic / disruption scores.
    3. Format the ranked, score-annotated hotels as LLM context.

    Falls back to the legacy context-dump if weighted retrieval yields nothing.
    """
    question = state.get("question", "")
    city = state.get("city", "")
    context = ""
    ranked_ids: list = []

    try:
        intent = parse_query(question, default_city=city or None)
        # Keep the city the caller asked for if the parser didn't find one.
        if not intent.city:
            intent.city = city or None
        result = _retriever.retrieve(intent, limit=10)
        if result.hotels:
            context = format_retrieval_context(result)
            ranked_ids = [h.id for h in result.hotels]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Weighted retrieval failed for city=%s: %s", city, exc)

    # Fallback: legacy full-context dump if the weighted path produced nothing.
    if not context:
        try:
            context = build_graph_context(city=city)
            logger.info("CRAG retrieve: used legacy context-dump fallback")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Legacy graph retrieval also failed for city=%s: %s", city, exc)

    logger.info(
        "CRAG retrieve: question=%s, city=%s, ranked=%d, context_len=%d",
        question, city, len(ranked_ids), len(context),
    )
    return {**state, "context": context, "ranked_ids": ranked_ids}


def _grade(state: GraphState) -> GraphState:
    """Score relevance of retrieved context against the question."""
    if _client is None:
        return {**state, "score": 0.5}
    prompt = (
        "You are a relevance grader. Score how well the graph context answers the question.\n"
        "Return ONLY raw JSON (no markdown, no code fences): {\"score\": <number between 0 and 1>}\n\n"
        f"Question: {state['question']}\n\n"
        f"Context:\n{state.get('context', '')[:2000]}\n"
    )
    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        cleaned = _extract_json(content)
        parsed = json.loads(cleaned)
        score = float(parsed.get("score", 0.5))
    except (json.JSONDecodeError, ValueError, TypeError, Exception) as exc:
        logger.warning("Grading failed (raw response: %s), defaulting to 0.5: %s", content[:200] if 'content' in dir() else '?', exc)
        score = 0.5
    logger.info("CRAG grade: score=%.2f", score)
    return {**state, "score": score}


def _transform_query(state: GraphState) -> GraphState:
    """Rewrite the question for better graph-retrieval alignment."""
    if _client is None:
        return {**state, "retries": state.get("retries", 0) + 1}
    prompt = (
        "Rewrite the question to better match hotel and city graph fields. "
        "Return only the rewritten question.\n\n"
        f"Question: {state['question']}\n"
    )
    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        rewritten = response.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("Query transform failed: %s", exc)
        rewritten = state["question"]
    return {
        **state,
        "question": rewritten,
        "retries": state.get("retries", 0) + 1,
    }


def _generate(state: GraphState) -> GraphState:
    """Generate the final answer using the LLM."""
    if _client is None:
        return {**state, "answer": "LLM not configured (no API key for the selected provider)."}
    prompt = (
        "You are a GraphRAG assistant for Sri Lanka hotel recommendations.\n"
        "The context contains hotels already RANKED by a feasibility-first weighted "
        "retriever, each with a composite score, a per-component score breakdown "
        "(spatial / accessibility / facility / economic / disruption) and a 'Why' line.\n"
        "- Respect the ranking order; present the top matches first.\n"
        "- Explain WHY each recommended hotel ranks where it does, using its score "
        "breakdown and 'Why' reasons (e.g. fast access in traffic, within budget, "
        "low disruption, matched amenities).\n"
        "- Mention specific hotel names, ratings, and price ranges.\n"
        "- Include travel times and traffic conditions when available.\n"
        "- Warn about heavy congestion or nearby events if flagged in the breakdown.\n"
        "- Do not invent hotels or attributes not present in the context.\n"
        "- If the context notes filters were relaxed, say so honestly.\n\n"
        f"Question: {state['question']}\n\n"
        f"Graph context:\n{state.get('context', '')}\n"
    )
    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        answer = f"Sorry, I couldn't generate an answer: {exc}"
    return {**state, "answer": answer}


# ---------------------------------------------------------------------------
# LangGraph definition (kept for reference / future state-machine needs)
# ---------------------------------------------------------------------------

def _should_rewrite(state: GraphState) -> str:
    retries = state.get("retries", 0)
    if not state.get("context"):
        return "transform_query" if retries < CRAG_MAX_RETRIES else "generate"
    score = state.get("score", 0.0)
    if score >= CRAG_MIN_SCORE or retries >= CRAG_MAX_RETRIES:
        return "generate"
    return "transform_query"


def build_crag_app():
    # Lazy import: langgraph is optional and only needed for this reference
    # state-machine. Production uses the sequential run_crag() below.
    from langgraph.graph import StateGraph, END, START

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", _retrieve)
    workflow.add_node("grade", _grade)
    workflow.add_node("transform_query", _transform_query)
    workflow.add_node("generate", _generate)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges("grade", _should_rewrite)
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", END)
    return workflow.compile()


# ---------------------------------------------------------------------------
# Production entry point — sequential CRAG (avoids LangGraph recursion bugs)
# ---------------------------------------------------------------------------

def run_crag(question: str, city: str) -> Dict[str, str]:
    """
    Run the CRAG pipeline and return {"answer": ..., "context": ...}.

    Steps:
      1. Retrieve graph context
      2. If empty → rewrite query once → re-retrieve
      3. Grade relevance
      4. If score low and retries remain → rewrite → retrieve → re-grade
      5. Generate answer
    Results are cached in Redis for 15 minutes.
    """
    cache = Cache()
    key = hashlib.md5(f"{city}:{question}".encode("utf-8")).hexdigest()
    cached = cache.get(key)
    if cached:
        logger.info("CRAG cache hit for key=%s", key)
        return cached

    min_score = CRAG_MIN_SCORE
    max_retries = CRAG_MAX_RETRIES

    state: GraphState = {"question": question, "city": city, "retries": 0}

    # Step 1: initial retrieve
    state = _retrieve(state)

    # Step 2: if empty context, try one rewrite + re-retrieve
    if not state.get("context") and state.get("retries", 0) < max_retries:
        state = _transform_query(state)
        state = _retrieve(state)

    # Step 3: grade relevance
    state = _grade(state)

    # Step 4: if score too low and retries remain, try again
    if state.get("score", 0.0) < min_score and state.get("retries", 0) < max_retries:
        state = _transform_query(state)
        state = _retrieve(state)
        state = _grade(state)

    # Step 5: generate final answer
    state = _generate(state)

    result = {
        "answer": state.get("answer", ""),
        "context": state.get("context", ""),
        "ranked_ids": state.get("ranked_ids", []),
    }
    cache.set(key, result)
    return result
