import os
from configs.ollama_config import OllamaEmbeddingsWrapper
from loguru import logger
from tqdm.asyncio import tqdm as tqdm_asyncio
from dotenv import load_dotenv

load_dotenv()


OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL")


class EmbedChunks:
    """
    A class to handle embedding generation for chunks of text using OllamaEmbeddingsWrapper.
    """

    def __init__(self, model_name, base_url="http://localhost:11434"):
        """
        Initialize the embedding generator with the specified model.

        Args:
            model_name (str): Name of the embedding model.
            base_url (str): Base URL of the embedding service.
        """
        self.embedding_model = OllamaEmbeddingsWrapper(model=model_name, base_url=base_url)

    async def generate_embeddings_async(self, texts, batch_size=64):
        """
        Generate embeddings asynchronously in batches.

        Args:
            texts (list of str): The texts to embed.
            batch_size (int): Number of texts to process in each batch.

        Returns:
            list: Embeddings corresponding to the input texts.
        """
        embeddings = []

        for batch_start in tqdm_asyncio(
            range(0, len(texts), batch_size),
            desc="Generating Embeddings",
        ):
            batch_texts = texts[batch_start : batch_start + batch_size]
            try:
                batch_embeddings = self.embedding_model.embed_documents(batch_texts)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {batch_start}-{batch_start + batch_size}: {e}")
                embeddings.extend([None] * len(batch_texts))  # Placeholder for failed embeddings

        return embeddings
