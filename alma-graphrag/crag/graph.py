import os
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START

from crag.nodes import retrieve, grade_documents, transform_query, generate

load_dotenv()


class GraphState(TypedDict, total=False):
    question: str
    city: str
    context: str
    documents: list
    score: float
    answer: str
    retries: int


def _decide(state: GraphState) -> str:
    min_score = float(os.getenv("CRAG_MIN_SCORE", "0.6"))
    max_retries = int(os.getenv("CRAG_MAX_RETRIES", "1"))
    retries = state.get("retries", 0)

    # If no context was retrieved, prefer rewriting the query only if we
    # haven't exhausted retry attempts — otherwise generate an answer.
    if not state.get("context"):
        if retries < max_retries:
            return "transform_query"
        return "generate"

    if state.get("score", 0.0) >= min_score:
        return "generate"
    if retries >= max_retries:
        return "generate"
    return "transform_query"


def build_crag_app():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("generate", generate)

    # Entry point: start the graph by calling `retrieve`
    workflow.add_edge(START, "retrieve")

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", _decide)
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()


def run_crag(question: str, city: str) -> dict:
    """
    Simple, non-recursive CRAG pipeline to avoid langgraph recursion issues.
    Steps:
      1. retrieve
      2. if no context -> transform_query once and retrieve again
      3. grade
      4. optionally one more transform+retrieve if score low and retries remain
      5. generate
    """
    min_score = float(os.getenv("CRAG_MIN_SCORE", "0.6"))
    max_retries = int(os.getenv("CRAG_MAX_RETRIES", "1"))

    state = {"question": question, "city": city, "retries": 0}

    # initial retrieve
    state = retrieve(state)

    # if empty context, try one rewrite+retrieve
    if not state.get("context") and state.get("retries", 0) < max_retries:
        state = transform_query(state)
        state = retrieve(state)

    state = grade_documents(state)

    # if score still low and retries remain, attempt one more rewrite cycle
    if state.get("score", 0.0) < min_score and state.get("retries", 0) < max_retries:
        state = transform_query(state)
        state = retrieve(state)
        state = grade_documents(state)

    state = generate(state)
    return {"answer": state.get("answer", ""), "context": state.get("context", "")}
