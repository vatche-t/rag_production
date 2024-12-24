from ingestion.index import load_index
from ingestion.embedder import EmbedChunks
from ingestion.vector_store import VectorStore
from query.query_agent import QueryAgent
from pathlib import Path
from loguru import logger
from langchain_ollama import OllamaLLM

COLLECTION_NAME = "rag_collection"


def generate_response(user_query):
    try:
        logger.info("Loading index from Qdrant...")
        chunks = load_index(
            embedding_model_name="nomic-embed-text",
            chunk_size=500,
            chunk_overlap=50,
            docs_dir=Path("data/pdfs"),
            collection_name=COLLECTION_NAME,
        )
        logger.info(f"Loaded {len(chunks)} chunks from Qdrant.")

        # Initialize components
        embedder = EmbedChunks(model_name="nomic-embed-text")
        vector_store = VectorStore(collection_name=COLLECTION_NAME)
        llm = OllamaLLM(model="llama3.2", base_url="http://localhost:11434")
        query_agent = QueryAgent(
            embedding_model=embedder,
            vector_store=vector_store,
            llm=llm,
        )

        # Generate response
        logger.info("Generating response for the query...")
        response = query_agent(query=user_query, num_chunks=5, lexical_search_k=1, rerank_k=7, stream=False)

        # Display results
        print(f"User Query: {user_query}")
        print(f"Answer: {response['answer']}")
        print(f"Sources: {response['sources']}")
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
