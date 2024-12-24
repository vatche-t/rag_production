import requests
import json


class OllamaLLM:
    def __init__(self, model="llama3.2", base_url="http://localhost:11434"):
        """
        Initializes the OllamaLLM class.

        Args:
            model (str): Name of the model (default: "llama3.2").
            base_url (str): Base URL of the Llama3.2 API (default: "http://localhost:11434").
        """
        self.model = model
        self.base_url = base_url

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
        endpoint = f"{self.base_url}/v1/generate"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        try:
            response = requests.post(endpoint, json=payload, headers=headers, stream=stream)
            response.raise_for_status()

            if stream:
                return self._stream_response(response)
            else:
                return response.json()["choices"][0]["text"]

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error communicating with Llama3.2 API: {e}")

    def _stream_response(self, response):
        """
        Handle streaming responses from the Llama3.2 API.

        Args:
            response (requests.Response): Response object from the API.

        Yields:
            str: Streamed content chunks.
        """
        for line in response.iter_lines(decode_unicode=True):
            if line.strip():
                data = json.loads(line)
                if "choices" in data and "text" in data["choices"][0]:
                    yield data["choices"][0]["text"]


class OllamaEmbeddings:
    def __init__(self, model, base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def embed_documents(self, documents):
        """
        Sends a request to the Ollama embedding API to generate embeddings for documents.

        Args:
            documents (list[str]): List of text documents to embed.

        Returns:
            list[list[float]]: List of embeddings for the input documents.
        """
        # Replace with actual API call to your Ollama server
        # Example:
        embeddings = []
        for doc in documents:
            embedding = self._api_call_to_ollama(doc)
            embeddings.append(embedding)
        return embeddings

    def _api_call_to_ollama(self, document):
        """
        Placeholder for the API call to Ollama server.
        Replace this with your actual implementation.
        """
        # Make an HTTP request to the Ollama embedding endpoint here
        raise NotImplementedError("API call logic to Ollama should be implemented.")
