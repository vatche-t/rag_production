import json
from pathlib import Path
from tqdm import tqdm
from ingestion.embedder import EmbedChunks
from ingestion.vector_store import VectorStore
from query.reranker import rerank_documents
from query.search import semantic_search, lexical_search
from loguru import logger

# from configs.ollama_config import OllamaLLMWrapper
from langchain_ollama import OllamaLLM


class QueryAgent:
    """
    A class to handle user queries by retrieving, reranking, and generating responses.
    """

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
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm = llm
        self.lexical_index = lexical_index
        self.reranker = reranker
        self.temperature = temperature
        self.context_length = int(0.5 * max_context_length)  # Reserve 50% for context
        self.max_tokens = int(0.5 * max_context_length)  # Reserve 50% for response
        self.system_content = system_content
        self.assistant_content = assistant_content

    def __call__(
        self,
        query: str,
        num_chunks: int = 5,
        lexical_search_k: int = 1,
        rerank_k: int = 7,
        stream: bool = False,
    ) -> dict:
        logger.info(f"Processing query: {query}")

        # Semantic search
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
                chunks=self.vector_store.get_all_documents(),
                k=lexical_search_k,
            )
            context_results[lexical_search_k:lexical_search_k] = lexical_context

        # Rerank documents
        if self.reranker:
            context_results = rerank_documents(
                documents=context_results,
                query_embedding=self.embedding_model.embedding_model.embed_documents([query])[0],
            )
            context_results = context_results[:rerank_k]

        # Prepare context and generate response
        document_ids = [item["id"] for item in context_results]
        context = "\n\n".join([item["text"] for item in context_results])
        sources = set([item["source"] for item in context_results])
        prompts = [f"Context:\n{context}\n\nUser Query:\n{query}"]

        # Generate response
        response = self.llm.generate(
            prompts=prompts,
            stop=None,
        )

        # Extract the text portion of the response
        answer = response.generations[0][0].text.strip()
        return {
            "question": query,
            "sources": list(sources),
            "document_ids": document_ids,
            "answer": answer,
        }


# Generate responses for a batch of queries
def generate_responses(
    queries: list,
    embedding_model: EmbedChunks,
    vector_store: VectorStore,
    llm: OllamaLLM,
    output_path: str,
    num_chunks: int = 5,
    lexical_search_k: int = 1,
    rerank_k: int = 7,
):
    """
    Generate responses for a batch of queries and save results.

    Args:
        queries (list): List of user queries.
        embedding_model (EmbedChunks): Embedding model instance.
        vector_store (VectorStore): Vector store instance.
        llm (OllamaLLMWrapper): LLM instance.
        output_path (str): Path to save the output results.
        num_chunks (int): Number of semantic search results.
        lexical_search_k (int): Number of lexical search results.
        rerank_k (int): Number of reranked results.
    """
    agent = QueryAgent(
        embedding_model=embedding_model,
        vector_store=vector_store,
        llm=llm,
    )

    results = []
    for query in tqdm(queries, desc="Processing queries"):
        result = agent(
            query=query,
            num_chunks=num_chunks,
            lexical_search_k=lexical_search_k,
            rerank_k=rerank_k,
            stream=False,
        )
        results.append(result)

    # Save results to a JSON file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as file:
        json.dump(results, file, indent=4)
    logger.info(f"Saved query responses to {output_path}")
