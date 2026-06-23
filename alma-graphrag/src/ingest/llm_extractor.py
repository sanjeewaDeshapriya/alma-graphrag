from __future__ import annotations

import json
from typing import Any, Dict, List
from openai import OpenAI

from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_PROVIDER


class LLMExtractor:
    def __init__(self) -> None:
        if not LLM_API_KEY:
            raise ValueError(f"No API key set for LLM provider '{LLM_PROVIDER}'")
        self.client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    def extract(self, hotel: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            "Extract amenities and nearby places from the hotel text. "
            "Return JSON with keys: amenities (list of strings), "
            "locations (list of objects with name, type, distance_km).\n\n"
            f"Hotel Name: {hotel.get('name', '')}\n"
            f"Address: {hotel.get('address', '')}\n"
            f"Description: {hotel.get('description', '')}\n"
        )

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = response.choices[0].message.content.strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return {"amenities": [], "locations": []}

        amenities = data.get("amenities", [])
        locations = data.get("locations", [])
        if not isinstance(amenities, list):
            amenities = []
        if not isinstance(locations, list):
            locations = []

        return {"amenities": amenities, "locations": locations}
