from __future__ import annotations

import hashlib
import json
from typing import Dict, TypedDict
from langgraph.graph import StateGraph, END, START
from openai import OpenAI

from src.config import CRAG_MAX_RETRIES, CRAG_MIN_SCORE, OPENAI_API_KEY, OPENAI_MODEL
from src.crag.cache import Cache
from src.graph.query import build_graph_context


class GraphState(TypedDict, total=False):
    question: str
    city: str
    context: str
    score: float
    answer: str
    retries: int


def _retrieve(state: GraphState) -> GraphState:
    context = build_graph_context(city=state["city"])
    return {**state, "context": context}


def _grade(state: GraphState) -> GraphState:
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = (
        "Score the relevance of the graph context to the question from 0 to 1. "
        "Return JSON: {\"score\": number}.\n\n"
        f"Question: {state['question']}\n\n"
        f"Context:\n{state.get('context', '')}\n"
    )
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    content = response.choices[0].message.content.strip()
    score = 0.5
    try:
        parsed = json.loads(content)
        score = float(parsed.get("score", score))
    except (json.JSONDecodeError, ValueError, TypeError):
        score = 0.5
    return {**state, "score": score}


def _transform_query(state: GraphState) -> GraphState:
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = (
        "Rewrite the question to better match hotel and city graph fields. "
        "Return only the rewritten question.\n\n"
        f"Question: {state['question']}\n"
    )
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    rewritten = response.choices[0].message.content.strip()
    return {
        **state,
        "question": rewritten,
        "retries": state.get("retries", 0) + 1,
    }


def _generate(state: GraphState) -> GraphState:
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = (
        "You are a GraphRAG assistant. Use only the provided graph context. "
        "If the context is insufficient, say so and suggest next ingestion steps.\n\n"
        f"Question: {state['question']}\n\n"
        f"Graph context:\n{state.get('context', '')}\n"
    )
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = response.choices[0].message.content.strip()
    return {**state, "answer": answer}


def _should_rewrite(state: GraphState) -> str:
    # If there's no retrieved context, prefer rewriting the query only if
    # we haven't exhausted retry attempts; otherwise generate an answer.
    retries = state.get("retries", 0)
    if not state.get("context"):
        if retries < CRAG_MAX_RETRIES:
            return "transform_query"
        return "generate"

    score = state.get("score", 0.0)
    if score >= CRAG_MIN_SCORE or retries >= CRAG_MAX_RETRIES:
        return "generate"
    return "transform_query"


def build_crag_app():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", _retrieve)
    workflow.add_node("grade", _grade)
    workflow.add_node("transform_query", _transform_query)
    workflow.add_node("generate", _generate)

    # Entry point: start the graph by calling `retrieve`
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges("grade", _should_rewrite)
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()


def run_crag(question: str, city: str) -> Dict[str, str]:
    cache = Cache()
    key = hashlib.md5(f"{city}:{question}".encode("utf-8")).hexdigest()
    cached = cache.get(key)
    if cached:
        return cached

    # Simple sequential CRAG flow (no StateGraph) to avoid recursion issues
    min_score = float(os.getenv("CRAG_MIN_SCORE", "0.6"))
    max_retries = int(os.getenv("CRAG_MAX_RETRIES", "1"))

    state = {"question": question, "city": city, "retries": 0}

    # initial retrieve
    state = _retrieve(state)

    # if empty, try one rewrite + retrieve
    if not state.get("context") and state.get("retries", 0) < max_retries:
        state = _transform_query(state)
        state = _retrieve(state)

    state = _grade(state)

    if state.get("score", 0.0) < min_score and state.get("retries", 0) < max_retries:
        state = _transform_query(state)
        state = _retrieve(state)
        state = _grade(state)

    state = _generate(state)

    result = {"answer": state.get("answer", ""), "context": state.get("context", "")}
    cache.set(key, result)
    return result
