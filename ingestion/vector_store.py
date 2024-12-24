from configs.qdrant_config import get_qdrant_client


class VectorStore:
    def __init__(self, collection_name):
        """
        Initialize the VectorStore with Qdrant client and collection name.

        Args:
            collection_name (str): The name of the Qdrant collection to use.
        """
        self.client = get_qdrant_client()
        self.collection_name = collection_name

    def add_documents(self, chunks, embeddings):
        """
        Add documents to the Qdrant collection.

        Args:
            chunks (list[str]): List of text chunks.
            embeddings (list[list[float]]): List of corresponding embeddings.
        """
        points = [
            {
                "id": str(i),  # Ensure IDs are strings to avoid conflicts
                "vector": embeddings[i],
                "payload": {
                    "text": chunks[i]["text"],
                    "source": chunks[i]["source"],  # Include source metadata
                },
            }
            for i in range(len(chunks))
        ]
        # Upsert points into the Qdrant collection
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector, top_k=5):
        """
        Perform a semantic search in the Qdrant collection.

        Args:
            query_vector (list[float]): The query embedding vector.
            top_k (int): Number of top results to return.

        Returns:
            list[dict]: List of search results with text and metadata.
        """
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
