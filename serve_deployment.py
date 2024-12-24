import os
import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

import ray
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from ray import serve
from starlette.responses import StreamingResponse
from ingestion.index import load_index
from query.query_agent import QueryAgent
from rank_bm25 import BM25Okapi
from utils.logger import logger

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    query: str


class RayAssistantDeployment:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        num_chunks: int,
        embedding_model_name: str,
        use_lexical_search: bool,
        lexical_search_k: int,
        use_reranking: bool,
        rerank_threshold: float,
        rerank_k: int,
        llm: str,
        sql_dump_fp: Path,
    ):
        logger.info("Initializing RayAssistantDeployment...")

        # Load document chunks from the vector database
        chunks = load_index(
            embedding_model_name=embedding_model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            sql_dump_fp=sql_dump_fp,
        )

        # Lexical Search (BM25)
        lexical_index = None
        if use_lexical_search:
            logger.info("Setting up BM25 for lexical search...")
            texts = [re.sub(r"[^a-zA-Z0-9]", " ", chunk[1]).lower().split() for chunk in chunks]
            lexical_index = BM25Okapi(texts)

        # Reranker
        reranker = None
        if use_reranking:
            logger.info("Loading reranker model...")
            reranker_fp = Path(os.environ["RAY_ASSISTANT_RERANKER_MODEL"])
            with open(reranker_fp, "rb") as file:
                reranker = pickle.load(file)

        # Initialize QueryAgent
        logger.info("Initializing QueryAgent...")
        self.query_agent = QueryAgent(
            embedding_model_name=embedding_model_name,
            chunks=chunks,
            lexical_index=lexical_index,
            reranker=reranker,
            llm=llm,
        )

        # Save configurations
        self.num_chunks = num_chunks
        self.lexical_search_k = lexical_search_k
        self.rerank_threshold = rerank_threshold
        self.rerank_k = rerank_k

    def answer_query(self, query: Query) -> Dict[str, Any]:
        return self.query_agent(
            query=query.query,
            num_chunks=self.num_chunks,
            lexical_search_k=self.lexical_search_k,
            rerank_threshold=self.rerank_threshold,
            rerank_k=self.rerank_k,
        )

    @app.post("/query")
    def query(self, query: Query) -> Dict[str, Any]:
        return self.answer_query(query)

    @app.post("/stream")
    def stream_query(self, query: Query) -> StreamingResponse:
        result = self.answer_query(query)
        response = result["answer"]
        return StreamingResponse(iter(response), media_type="text/plain")


# Deploy the Ray Serve app
serve.start()
deployment = RayAssistantDeployment.bind(
    chunk_size=700,
    chunk_overlap=50,
    num_chunks=30,
    embedding_model_name=os.getenv("RAY_ASSISTANT_EMBEDDING_MODEL"),
    use_lexical_search=True,
    lexical_search_k=5,
    use_reranking=True,
    rerank_threshold=0.9,
    rerank_k=10,
    llm="llama-3.2",
    sql_dump_fp=Path(os.getenv("RAY_ASSISTANT_INDEX")),
)
