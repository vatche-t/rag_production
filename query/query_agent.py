import json
import re
from pathlib import Path

from tqdm import tqdm
from ingestion.embedder import EmbedChunks
from ingestion.vector_store import VectorStore
from query.reranker import rerank_documents
from query.search import semantic_search, lexical_search
from utils.logger import logger
from configs.ollama_config import OllamaLLM


class QueryAgent:
    def __init__(
        self,
        embedding_model: EmbedChunks,
        vector_store: VectorStore,
        llm: OllamaLLM,
        lexical_index=None,
        reranker=None,
        temperature=0.0,
        max_context_length=4096,
        system_content="",
        assistant_content="",
    ):
        """
        Initialize the QueryAgent.

        Args:
            embedding_model (EmbedChunks): Model for generating embeddings.
            vector_store (VectorStore): Vector store client for retrieval.
            llm (OllamaLLM): LLM client for response generation.
            lexical_index: Lexical search index (optional).
            reranker: Reranking model (optional).
            temperature (float): Sampling temperature for the LLM.
            max_context_length (int): Maximum token length for input/output.
            system_content (str): System prompt for the LLM.
            assistant_content (str): Assistant's initial content for the conversation.
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm = llm
        self.lexical_index = lexical_index
        self.reranker = reranker
        self.temperature = temperature
        self.context_length = int(0.5 * max_context_length)  # Reserve 50% for the context
        self.max_tokens = int(0.5 * max_context_length)  # Maximum output tokens
        self.system_content = system_content
        self.assistant_content = assistant_content

    def __call__(
        self,
        query,
        num_chunks=5,
        lexical_search_k=1,
        rerank_k=7,
        stream=True,
    ):
        """
        Process a user query to generate a response.

        Args:
            query (str): User query.
            num_chunks (int): Number of top semantic results to retrieve.
            lexical_search_k (int): Number of top lexical results to include.
            rerank_k (int): Number of reranked results to use.
            stream (bool): Whether to stream the response from the LLM.

        Returns:
            dict: Query result with answer, sources, and document IDs.
        """
        logger.info(f"Processing query: {query}")

        # Perform semantic search
        context_results = semantic_search(
            query=query,
            embedding_model=self.embedding_model,
            qdrant_client=self.vector_store.client,
            collection_name=self.vector_store.collection_name,
            k=num_chunks,
        )

        # Add lexical search results
        if self.lexical_index:
            lexical_context = lexical_search(
                index=self.lexical_index,
                query=query,
                chunks=self.vector_store.get_chunks(),
                k=lexical_search_k,
            )
            context_results[lexical_search_k:lexical_search_k] = lexical_context

        # Rerank results
        if self.reranker:
            context_results = rerank_documents(
                documents=context_results,
                query_embedding=self.embedding_model.embedding_model.embed_documents([query])[0],
            )
            context_results = context_results[:rerank_k]

        # Prepare context for the LLM
        document_ids = [item["id"] for item in context_results]
        context = [item["text"] for item in context_results]
        sources = set([item["source"] for item in context_results])
        user_content = f"query: {query}, context: {context}"

        # Generate LLM response
        answer = self.llm.generate(
            prompt=user_content,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=stream,
        )

        # Prepare result
        result = {
            "question": query,
            "sources": list(sources),
            "document_ids": document_ids,
            "answer": answer,
        }
        return result


# Generate responses for a set of queries
def generate_responses(
    queries,
    embedding_model,
    vector_store,
    llm,
    output_path,
    num_chunks=5,
    lexical_search_k=1,
    rerank_k=7,
):
    """
    Generate responses for a list of queries and save them to a file.

    Args:
        queries (list[str]): List of user queries.
        embedding_model (EmbedChunks): Embedding model.
        vector_store (VectorStore): Vector store for retrieval.
        llm (OllamaLLM): LLM for response generation.
        output_path (str): Path to save the generated responses.
        num_chunks (int): Number of semantic search results.
        lexical_search_k (int): Number of lexical search results.
        rerank_k (int): Number of reranked results to use.
    """
    agent = QueryAgent(
        embedding_model=embedding_model,
        vector_store=vector_store,
        llm=llm,
    )

    results = []
    for query in tqdm(queries):
        result = agent(
            query=query,
            num_chunks=num_chunks,
            lexical_search_k=lexical_search_k,
            rerank_k=rerank_k,
            stream=False,
        )
        results.append(result)

    # Save results to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fp:
        json.dump(results, fp, indent=4)
    logger.info(f"Responses saved to {output_path}")
