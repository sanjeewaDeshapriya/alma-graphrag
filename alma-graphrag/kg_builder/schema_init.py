from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "alma_password123")


def run_schema_file(filepath: str) -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    schema_path = Path(filepath)
    statements = [
        stmt.strip()
        for stmt in schema_path.read_text(encoding="utf-8").split(";")
        if stmt.strip()
    ]

    with driver.session() as session:
        for stmt in statements:
            try:
                session.run(stmt)
            except Exception:
                # Ignore if already exists or unsupported in current Neo4j
                pass

    driver.close()
