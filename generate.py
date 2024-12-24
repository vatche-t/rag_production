from ingestion.index import load_index
from ingestion.embedder import EmbedChunks
from ingestion.vector_store import VectorStore
from query.query_agent import QueryAgent
from pathlib import Path
from utils.logger import logger

COLLECTION_NAME = "rag_collection"


def generate_response(user_query):
    """
    Generate a response to the user's query using the QueryAgent.

    Args:
        user_query (str): The query provided by the user.
    """
    try:
        # Load index or build it if not already present
        logger.info("Loading index from Qdrant...")
        chunks = load_index(
            embedding_model_name="nomic-embed-text",
            chunk_size=500,
            chunk_overlap=50,
            docs_dir=Path("data/pdfs"),  # Path to PDF documents
            collection_name=COLLECTION_NAME,
        )
        logger.info(f"Loaded {len(chunks)} chunks from Qdrant.")

        # Initialize embedder, vector store, and query agent
        logger.info("Initializing embedder and vector store...")
        embedder = EmbedChunks(model_name="nomic-embed-text")
        vector_store = VectorStore(collection_name=COLLECTION_NAME)
        query_agent = QueryAgent(embedder=embedder, vector_store=vector_store)

        # Generate response
        logger.info("Generating response for the query...")
        response = query_agent.answer_query(user_query)

        # Display results
        print(f"User Query: {user_query}")
        print(f"Response: {response}")
    except Exception as e:
        logger.error(f"Error generating response: {e}")


if __name__ == "__main__":
    try:
        user_query = input("Enter your query: ")
        if not user_query.strip():
            raise ValueError("Query cannot be empty.")
        generate_response(user_query)
    except Exception as e:
        logger.error(f"Error in user query input: {e}")
