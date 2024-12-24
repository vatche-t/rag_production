from langchain_ollama import OllamaLLM, OllamaEmbeddings
from loguru import logger
from tqdm.asyncio import tqdm as tqdm_asyncio


class OllamaLLMWrapper:
    """
    A class to handle text generation using the Ollama LLM.
    """

    def __init__(self, model="llama3.2", base_url="http://localhost:11434"):
        """
        Initializes the OllamaLLMWrapper class.

        Args:
            model (str): Name of the model.
            base_url (str): Base URL of the Ollama API.
        """
        self.model = model
        self.base_url = base_url
        self.client = OllamaLLM(model=self.model, base_url=self.base_url)

    def generate(self, prompt, max_tokens=256, temperature=0.7, stream=False):
        """
        Generate a response from the LLM.

        Args:
            prompt (str): Input text prompt.
            max_tokens (int): Maximum number of tokens in the output.
            temperature (float): Sampling temperature.
            stream (bool): Whether to stream the response.

        Returns:
            str: Generated response from the model.
        """
        try:
            if stream:
                # Stream response chunks
                for chunk in self.client.stream(prompt, max_tokens=max_tokens, temperature=temperature):
                    yield chunk
            else:
                # Return full response
                response = self.client(prompt, max_tokens=max_tokens, temperature=temperature)
                return response  # ["choices"][0]["text"]
        except Exception as e:
            raise RuntimeError(f"Error generating response: {e}")


class OllamaEmbeddingsWrapper:
    """
    A wrapper class for embedding generation using Ollama.
    """

    def __init__(self, model, base_url="http://localhost:11434"):
        """
        Initialize the embedding generator with the specified model.

        Args:
            model (str): Name of the embedding model.
            base_url (str): Base URL for the Ollama API.
        """
        self.model = model
        self.embedding_model = OllamaEmbeddings(model=model, base_url=base_url)

    def embed_documents(self, texts):
        """
        Generate embeddings for the given texts.

        Args:
            texts (list of str): The texts to embed.

        Returns:
            list: Embeddings corresponding to the input texts.
        """
        try:
            logger.debug(f"Generating embeddings for {len(texts)} documents...")
            return self.embedding_model.embed_documents(texts)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
