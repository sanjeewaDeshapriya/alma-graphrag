from __future__ import annotations

import hashlib
import json
from typing import Dict, TypedDict
from langgraph.graph import StateGraph, END
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
    score = state.get("score", 0.0)
    retries = state.get("retries", 0)
    if score >= CRAG_MIN_SCORE or retries >= CRAG_MAX_RETRIES:
        return "generate"
    return "transform_query"


def build_crag_app():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", _retrieve)
    workflow.add_node("grade", _grade)
    workflow.add_node("transform_query", _transform_query)
    workflow.add_node("generate", _generate)

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

    app = build_crag_app()
    final_state = app.invoke({"question": question, "city": city, "retries": 0})

    result = {
        "answer": final_state.get("answer", ""),
        "context": final_state.get("context", ""),
    }
    cache.set(key, result)
    return result
