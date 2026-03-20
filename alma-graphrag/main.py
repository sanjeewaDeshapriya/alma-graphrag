from dotenv import load_dotenv
from kg_builder.schema_init import run_schema_file
from kg_builder.seed_data import seed_all
from kg_builder.embeddings import EmbeddingPipeline

load_dotenv()


def run_phase1_setup():
    print("\n==============================")
    print(" ALMA-GraphRAG — Phase 1 Setup ")
    print("==============================\n")

    print("[1/3] Applying Neo4j schema...")
    run_schema_file("neo4j_import/schema.cypher")

    print("[2/3] Seeding static reference data...")
    seed_all()

    print("[3/3] Generating embeddings...")
    pipeline = EmbeddingPipeline()
    pipeline.embed_hotels()
    pipeline.embed_news_signals()
    pipeline.embed_events()
    pipeline.close()

    print("\nPhase 1 KG construction complete.")
    print("Open http://localhost:7474 to explore.\n")


if __name__ == "__main__":
    run_phase1_setup()
