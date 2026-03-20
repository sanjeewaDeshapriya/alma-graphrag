import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class CRAGGrader:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def score(self, question: str, context: str) -> float:
        prompt = (
            "Score relevance from 0 to 1 as JSON {\"score\": number}.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}\n"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(content)
            return float(parsed.get("score", 0.5))
        except (json.JSONDecodeError, ValueError, TypeError):
            return 0.5
