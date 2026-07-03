import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.graph.schema_init import run_schema_file

if __name__ == "__main__":
    run_schema_file(Path(__file__).resolve().parents[1] / "src" / "graph" / "schema.cypher")
    print("Schema applied.")
