import uuid
from configs.qdrant_config import get_qdrant_client
from qdrant_client.http.models import VectorParams, Distance


class VectorStore:
    """
    A class to interact with a Qdrant vector database for storing and retrieving text embeddings.
    """

    def __init__(self, collection_name, vector_size=768, distance=Distance.COSINE):
        """
        Initialize the VectorStore with Qdrant client and collection name.

        Args:
            collection_name (str): The name of the Qdrant collection to use.
            vector_size (int): The size of the embedding vectors.
            distance (Distance): The distance metric for similarity search.
        """
        self.client = get_qdrant_client()
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance

        # Ensure the collection exists or create it
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """
        Ensure the Qdrant collection exists; create it if it doesn't.
        """
        try:
            collections = self.client.get_collections().collections
            if not any(c.name == self.collection_name for c in collections):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=self.distance),
                )
                print(f"Created Qdrant collection: {self.collection_name}")
            else:
                print(f"Qdrant collection '{self.collection_name}' already exists.")
        except Exception as e:
            raise RuntimeError(f"Error ensuring collection exists: {e}")

    def add_documents(self, batch):
        """
        Add documents to the Qdrant collection.

        Args:
            batch (list[dict]): A list of dictionaries with "text", "source", and "embedding" keys.
        """
        points = [
            {
                "id": str(uuid.uuid4()),  # Use unique UUIDs as IDs
                "vector": item["embedding"],
                "payload": {
                    "text": item["text"],
                    "source": item["source"],
                },
            }
            for item in batch
        ]
        try:
            # Upsert points into the Qdrant collection
            self.client.upsert(collection_name=self.collection_name, points=points)
        except Exception as e:
            raise RuntimeError(f"Error adding documents to Qdrant: {e}")

    def search(self, query_vector, top_k=5):
        """
        Perform a semantic search in the Qdrant collection.

        Args:
            query_vector (list[float]): The query embedding vector.
            top_k (int): Number of top results to return.

        Returns:
            list[dict]: List of search results with text and metadata.
        """
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
            )
            return [
                {
                    "id": result.id,
                    "text": result.payload["text"],
                    "source": result.payload.get("source", "Unknown"),
                    "score": result.score,
                }
                for result in results
            ]
        except Exception as e:
            raise RuntimeError(f"Error during search in Qdrant: {e}")
