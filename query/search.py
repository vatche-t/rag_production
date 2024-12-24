import numpy as np
from qdrant_client import QdrantClient
from configs.qdrant_config import get_qdrant_client
from ingestion.embedder import EmbedChunks


def semantic_search(query, embedding_model, qdrant_client, collection_name, k):
    """
    Perform semantic search using Qdrant.

    Args:
        query (str): User's query.
        embedding_model (EmbedChunks): Embedding model to generate query embedding.
        qdrant_client (QdrantClient): Qdrant client for vector search.
        collection_name (str): Name of the Qdrant collection to search.
        k (int): Number of top results to return.

    Returns:
        list[dict]: List of top-k matching documents with metadata.
    """
    query_embedding = np.array(embedding_model.embedding_model.embed_documents([query])[0])
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=k,
    )
    semantic_context = [
        {"id": result.id, "text": result.payload.get("text"), "source": result.payload.get("source")}
        for result in search_results
    ]
    return semantic_context


def lexical_search(index, query, chunks, k):
    """
    Perform lexical search using token matching or BM25-like scores.

    Args:
        index: Lexical search index.
        query (str): User's query.
        chunks (list): List of document chunks (id, text, source).
        k (int): Number of top results to return.

    Returns:
        list[dict]: List of top-k matching document chunks with scores.
    """
    query_tokens = query.lower().split()  # Preprocess query
    scores = index.get_scores(query_tokens)
    indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    lexical_context = [
        {"id": chunks[i][0], "text": chunks[i][1], "source": chunks[i][2], "score": scores[i]} for i in indices
    ]
    return lexical_context
