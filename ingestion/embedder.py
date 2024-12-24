from configs.ollama_config import OllamaEmbeddings


class EmbedChunks:
    """
    A class to handle embedding generation for chunks of text using OllamaEmbeddings.
    """

    def __init__(self, model_name):
        """
        Initialize the embedding generator with the specified model.

        Args:
            model_name (str): Name of the embedding model.
        """
        self.embedding_model = OllamaEmbeddings(model=model_name, base_url="http://localhost:11434")

    def __call__(self, batch):
        """
        Generate embeddings for a batch of text chunks.

        Args:
            batch (dict): A dictionary containing text and source keys.

        Returns:
            dict: The input batch with added embeddings.
        """
        embeddings = self.embedding_model.embed_documents(batch["text"])
        return {"text": batch["text"], "source": batch["source"], "embeddings": embeddings}
