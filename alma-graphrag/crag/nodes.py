import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

from crag.retriever import build_retriever
from crag.grader import CRAGGrader
try:
    from src.graph.query import build_graph_context
except Exception:
    build_graph_context = None

load_dotenv()

logger = logging.getLogger("alma.crag")
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def retrieve(state: dict) -> dict:
    retriever = build_retriever()
    docs = retriever.invoke(state["question"])
    context = "\n\n".join([d.page_content for d in docs])
    logger.info("CRAG retrieve: question=%s, docs=%d, context_len=%d", state.get("question"), len(docs), len(context))
    # Fallback: if vector retriever returned no docs, try graph-based context
    if len(docs) == 0 and build_graph_context is not None:
        try:
            city = state.get("city")
            if not city:
                logger.warning("CRAG retrieve fallback skipped: missing city in state")
                return {**state, "documents": docs, "context": context}
            graph_ctx = build_graph_context(city=city)
            if graph_ctx:
                logger.info("CRAG retrieve: fallback graph context found, len=%d", len(graph_ctx))
                return {**state, "documents": [], "context": graph_ctx}
        except Exception as e:
            logger.warning("CRAG retrieve fallback failed: %s", e)
    return {**state, "documents": docs, "context": context}


def grade_documents(state: dict) -> dict:
    grader = CRAGGrader()
    score = grader.score(state["question"], state.get("context", ""))
    logger.info("CRAG grade: question=%s, score=%s", state.get("question"), score)
    return {**state, "score": score}


def transform_query(state: dict) -> dict:
    prompt = (
        "Rewrite the query for hotel graph retrieval. Return only the query.\n\n"
        f"Query: {state['question']}"
    )
    response = _client.chat.completions.create(
        model=_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    rewritten = response.choices[0].message.content.strip()
    return {**state, "question": rewritten, "retries": state.get("retries", 0) + 1}


def generate(state: dict) -> dict:
    prompt = (
        "Answer using only the provided context. If insufficient, say so.\n\n"
        f"Question: {state['question']}\n\n"
        f"Context:\n{state.get('context', '')}\n"
    )
    response = _client.chat.completions.create(
        model=_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = response.choices[0].message.content.strip()
    logger.info("CRAG generate: question=%s, answer_len=%d", state.get("question"), len(answer))
    return {**state, "answer": answer}
