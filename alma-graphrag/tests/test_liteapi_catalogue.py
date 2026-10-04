"""Catalogue discovery must retain metadata without inventing availability."""
from src.ingest.liteapi import LiteApiClient


def _catalogue_hotel(hotel_id, name):
    return {
        "id": hotel_id,
        "name": name,
        "latitude": 6.9,
        "longitude": 79.8,
        "rating": 8.0,
        "stars": 4,
        "address": "Colombo",
    }


def test_catalogue_discovery_selects_unseen_hotels_and_keeps_unpriced_metadata(monkeypatch):
    client = LiteApiClient.__new__(LiteApiClient)
    pages = {
        0: {"total": 4, "data": [
            _catalogue_hotel("known", "Known"),
            _catalogue_hotel("new-1", "New One"),
        ]},
        2: {"total": 4, "data": [
            _catalogue_hotel("new-2", "New Two"),
            _catalogue_hotel("new-3", "New Three"),
        ]},
    }
    monkeypatch.setattr(client, "list_hotels", lambda _city, *, limit, offset: pages[offset])
    monkeypatch.setattr(client, "search_rates", lambda _city, **kwargs: {
        "data": [{"hotelId": "new-1", "roomTypes": [{
            "roomTypeId": "r1", "name": "Room", "offerRetailRate": [{"amount": 25000}], "rates": []
        }]}],
        "hotels": [],
    })

    hotels = client.scrape_catalogue_city("Colombo", existing_ids={"known"}, target=3, page_size=2)

    assert [hotel["id"] for hotel in hotels] == ["new-1", "new-2", "new-3"]
    assert hotels[0]["price_per_night_lkr"] == 25000
    assert hotels[1]["price_per_night_lkr"] is None
    assert hotels[2]["room_types"] == []


def test_rate_request_uses_explicit_hotel_ids_when_provided():
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"data": [], "hotels": []}

    class Http:
        payload = None

        def post(self, _url, json):
            self.payload = json
            return Response()

    client = LiteApiClient.__new__(LiteApiClient)
    client.client = Http()
    client.search_rates("Colombo", checkin="2026-10-01", checkout="2026-10-02", hotel_ids=["a", "b"])

    assert client.client.payload["hotelIds"] == ["a", "b"]
    assert "cityName" not in client.client.payload