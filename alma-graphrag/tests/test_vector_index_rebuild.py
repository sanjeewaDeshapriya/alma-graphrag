"""A city index rebuild must remove stale rows before inserting its snapshot."""
from src.search import vector_store as vs


class _Cursor:
    def __init__(self):
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def execute(self, query, params):
        self.executed.append((query, params))


class _Connection:
    def __init__(self):
        self.cursor_instance = _Cursor()
        self.committed = False
        self.closed = False

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


def test_replace_city_hotels_deletes_stale_rows_before_upserting(monkeypatch):
    connection = _Connection()
    monkeypatch.setattr(vs, "_connect", lambda **_: connection)
    rows = [{
        "id": "h1", "city": "Colombo", "name": "Hotel One",
        "doc": "hotel one", "embedding": [0.1, 0.2],
    }]

    assert vs.replace_city_hotels("Colombo", rows) == 1
    delete_query, delete_params = connection.cursor_instance.executed[0]
    assert "DELETE FROM hotel_search" in delete_query
    assert delete_params == ("Colombo",)
    assert "INSERT INTO hotel_search" in connection.cursor_instance.executed[1][0]
    assert connection.committed and connection.closed