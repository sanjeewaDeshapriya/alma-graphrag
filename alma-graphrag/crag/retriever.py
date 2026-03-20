import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import Neo4jVector

load_dotenv()


def build_retriever():
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    vector = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "alma_password123"),
        index_name=os.getenv("HOTEL_VECTOR_INDEX", "hotel_embeddings"),
        node_label="Hotel",
        text_node_property="description",
    )

    return vector.as_retriever(search_kwargs={"k": 6})
