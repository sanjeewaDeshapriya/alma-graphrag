from __future__ import annotations

from pathlib import Path
from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


def run_schema_file(schema_path: Path) -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        statements = [
            stmt.strip()
            for stmt in schema_path.read_text(encoding="utf-8").split(";")
            if stmt.strip()
        ]
        for stmt in statements:
            session.run(stmt)
    driver.close()
