"""Offline tests for the production CRAG orchestration path."""
import src.crag.graph as crag


class _Cache:
    def __init__(self, cached=None):
        self.cached = cached
        self.stored = []

    def get(self, _key):
        return self.cached

    def set(self, key, value):
        self.stored.append((key, value))


def test_run_crag_returns_cached_result_without_retrieving(monkeypatch):
    cached = {"answer": "cached", "context": "context", "ranked_ids": ["h1"]}
    cache = _Cache(cached)
    monkeypatch.setattr(crag, "Cache", lambda: cache)
    monkeypatch.setattr(crag, "_retrieve", lambda _: (_ for _ in ()).throw(AssertionError()))

    assert crag.run_crag("hotel", "Colombo") == cached
    assert cache.stored == []


def test_empty_context_rewrites_once_then_generates(monkeypatch):
    cache, retrieved, transformed = _Cache(), [], []
    monkeypatch.setattr(crag, "Cache", lambda: cache)

    def retrieve(state):
        retrieved.append(state["question"])
        context = "" if len(retrieved) == 1 else "#1 Hotel [score=0.9]"
        return {**state, "context": context, "ranked_ids": ["h1"] if context else []}

    def transform(state):
        transformed.append(state["question"])
        return {**state, "question": "rewritten", "retries": state["retries"] + 1}

    monkeypatch.setattr(crag, "_retrieve", retrieve)
    monkeypatch.setattr(crag, "_transform_query", transform)
    monkeypatch.setattr(crag, "_grade", lambda state: {**state, "score": 0.9})
    monkeypatch.setattr(crag, "_generate", lambda state: {**state, "answer": "answer"})

    result = crag.run_crag("original", "Colombo")
    assert retrieved == ["original", "rewritten"]
    assert transformed == ["original"]
    assert result == {"answer": "answer", "context": "#1 Hotel [score=0.9]", "ranked_ids": ["h1"]}
    assert cache.stored


def test_low_grade_uses_at_most_one_rewrite(monkeypatch):
    cache, grades, transformed = _Cache(), [], []
    monkeypatch.setattr(crag, "Cache", lambda: cache)
    monkeypatch.setattr(crag, "CRAG_MAX_RETRIES", 1)
    monkeypatch.setattr(crag, "_retrieve", lambda state: {**state, "context": "context", "ranked_ids": ["h1"]})

    def grade(state):
        grades.append(state["retries"])
        return {**state, "score": 0.2}

    def transform(state):
        transformed.append(state["retries"])
        return {**state, "question": "rewritten", "retries": state["retries"] + 1}

    monkeypatch.setattr(crag, "_grade", grade)
    monkeypatch.setattr(crag, "_transform_query", transform)
    monkeypatch.setattr(crag, "_generate", lambda state: {**state, "answer": "answer"})

    crag.run_crag("original", "Colombo")
    assert grades == [0, 1]
    assert transformed == [0]


def test_json_and_fallback_helpers_are_safe_without_an_llm():
    assert crag._extract_json('```json\n{"score": 0.8}\n```') == '{"score": 0.8}'
    answer = crag._fallback_answer({"context": "#1 Hotel One [score=0.9]"})
    assert "Hotel One" in answer
    assert "LLM service was unavailable" in answer