import os
import uuid
from typing import List
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from neo4j import GraphDatabase
from kg_builder.node_models import EmbeddingChunkNode
from src.config import (
    ACTIVE_EMBEDDING_API_KEY,
    ACTIVE_EMBEDDING_BASE_URL,
    ACTIVE_EMBEDDING_MODEL,
)

load_dotenv()


class EmbeddingPipeline:
    """
    Generates embeddings for Hotel, Event, and NewsSignal nodes.
    Stores vectors on nodes and optionally as EmbeddingChunk nodes.

    Provider (OpenAI / Gemini) is resolved from LLM_PROVIDER in src.config;
    Gemini is reached via its OpenAI-compatible embeddings endpoint.
    """

    def __init__(self):
        self.embedder = OpenAIEmbeddings(
            model=ACTIVE_EMBEDDING_MODEL,
            openai_api_key=ACTIVE_EMBEDDING_API_KEY,
            base_url=ACTIVE_EMBEDDING_BASE_URL,
        )
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "alma_password123"),
            ),
        )

    def _chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            chunk = " ".join(words[start : start + chunk_size])
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def embed_hotels(self) -> None:
        with self.driver.session() as s:
            hotels = s.run(
                "MATCH (h:Hotel) WHERE h.description IS NOT NULL "
                "RETURN h.id AS id, h.name AS name, h.description AS description, "
                "h.address AS address"
            ).data()

        for hotel in hotels:
            text = (
                f"Hotel: {hotel['name']}. "
                f"Address: {hotel['address']}. "
                f"Description: {hotel['description']}"
            )
            vector = self.embedder.embed_query(text)
            with self.driver.session() as s:
                s.run(
                    "MATCH (h:Hotel {id: $id}) SET h.embedding = $embedding",
                    {"id": hotel["id"], "embedding": vector},
                )

    def embed_news_signals(self) -> None:
        with self.driver.session() as s:
            news_items = s.run(
                "MATCH (n:NewsSignal) WHERE n.summary IS NOT NULL "
                "RETURN n.id AS id, n.title AS title, n.summary AS summary"
            ).data()

        for news in news_items:
            text = f"{news['title']}. {news['summary']}"
            vector = self.embedder.embed_query(text)
            with self.driver.session() as s:
                s.run(
                    "MATCH (n:NewsSignal {id: $id}) SET n.embedding = $embedding",
                    {"id": news["id"], "embedding": vector},
                )

    def embed_events(self) -> None:
        with self.driver.session() as s:
            events = s.run(
                "MATCH (e:Event) WHERE e.description IS NOT NULL "
                "RETURN e.id AS id, e.title AS title, e.description AS description"
            ).data()

        for event in events:
            text = f"{event['title']}. {event['description']}"
            vector = self.embedder.embed_query(text)
            with self.driver.session() as s:
                s.run(
                    "MATCH (e:Event {id: $id}) SET e.embedding = $embedding",
                    {"id": event["id"], "embedding": vector},
                )

    def build_embedding_chunks(self, label: str, id_field: str, text_field: str) -> None:
        query = f"MATCH (n:{label}) WHERE n.{text_field} IS NOT NULL "
        query += f"RETURN n.{id_field} AS id, n.{text_field} AS text"
        with self.driver.session() as s:
            items = s.run(query).data()

        for item in items:
            chunks = self._chunk_text(item["text"], chunk_size=150)
            embeddings = self.embedder.embed_documents(chunks)
            with self.driver.session() as s:
                for idx, (chunk_text, vector) in enumerate(zip(chunks, embeddings)):
                    chunk_id = str(uuid.uuid4())
                    s.run(
                        """
                        MERGE (c:EmbeddingChunk {id: $id})
                        SET c.text = $text,
                            c.source_id = $source_id,
                            c.source_type = $source_type,
                            c.chunk_index = $chunk_index,
                            c.embedding = $embedding
                        """,
                        {
                            "id": chunk_id,
                            "text": chunk_text,
                            "source_id": item["id"],
                            "source_type": label.lower(),
                            "chunk_index": idx,
                            "embedding": vector,
                        },
                    )

    def close(self) -> None:
        self.driver.close()
