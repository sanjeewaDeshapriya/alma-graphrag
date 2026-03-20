import os
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from crag.nodes import retrieve, grade_documents, transform_query, generate

load_dotenv()


class GraphState(TypedDict, total=False):
    question: str
    context: str
    documents: list
    score: float
    answer: str
    retries: int


def _decide(state: GraphState) -> str:
    min_score = float(os.getenv("CRAG_MIN_SCORE", "0.6"))
    max_retries = int(os.getenv("CRAG_MAX_RETRIES", "1"))
    if state.get("score", 0.0) >= min_score:
        return "generate"
    if state.get("retries", 0) >= max_retries:
        return "generate"
    return "transform_query"


def build_crag_app():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("generate", generate)

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", _decide)
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()


def run_crag(question: str) -> dict:
    app = build_crag_app()
    state = app.invoke({"question": question, "retries": 0})
    return {
        "answer": state.get("answer", ""),
        "context": state.get("context", ""),
    }
