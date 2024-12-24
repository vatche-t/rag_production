import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get Qdrant configuration from environment variables
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")  # Default to localhost if not set
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))  # Default to 6333 if not set
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "rag_collection")  # Default collection name


def get_qdrant_client():
    """
    Initializes and returns a QdrantClient instance.

    Returns:
        QdrantClient: The client to interact with the Qdrant server.
    """
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection_exists(client: QdrantClient, collection_name: str, vector_size: int = 1536):
    """
    Ensures the specified collection exists in Qdrant. If not, creates it.

    Args:
        client (QdrantClient): The Qdrant client.
        collection_name (str): Name of the collection.
        vector_size (int): Dimensionality of the vectors (default is 1536).
    """
    try:
        if not client.get_collection(collection_name=collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config={"size": vector_size, "distance": "Cosine"},
            )
            print(f"Collection '{collection_name}' created.")
        else:
            print(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        print(f"Error ensuring collection exists: {e}")
        raise
