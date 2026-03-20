import os
from dotenv import load_dotenv
from openai import OpenAI

from crag.retriever import build_retriever
from crag.grader import CRAGGrader

load_dotenv()

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def retrieve(state: dict) -> dict:
    retriever = build_retriever()
    docs = retriever.invoke(state["question"])
    context = "\n\n".join([d.page_content for d in docs])
    return {**state, "documents": docs, "context": context}


def grade_documents(state: dict) -> dict:
    grader = CRAGGrader()
    score = grader.score(state["question"], state.get("context", ""))
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
    return {**state, "answer": answer}
