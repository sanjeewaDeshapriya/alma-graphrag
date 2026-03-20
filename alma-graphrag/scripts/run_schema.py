from kg_builder.schema_init import run_schema_file

if __name__ == "__main__":
    run_schema_file("neo4j_import/schema.cypher")
    print("Schema applied.")
