import os
from functools import partial
from pathlib import Path

import ray
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ray.data import ActorPoolStrategy

from ingestion.pdf_processor import process_pdf
from ingestion.embedder import EmbedChunks
from ingestion.vector_store import VectorStore
from loguru import logger


class StoreResults:
    """
    Store results in the Qdrant vector store.
    """

    def __init__(self, collection_name):
        self.vector_store = VectorStore(collection_name=collection_name)

    def __call__(self, batch):
        """
        Process and store the batch of chunks and embeddings in Qdrant.

        Args:
            batch (dict): A batch of text chunks, sources, and embeddings.
        """
        self.vector_store.add_documents(
            chunks=[{"text": text, "source": source} for text, source in zip(batch["text"], batch["source"])],
            embeddings=batch["embeddings"],
        )
        return {}


def chunk_section(section, chunk_size, chunk_overlap):
    """
    Split a section of text into smaller chunks with overlap.

    Args:
        section (dict): A dictionary with "text" and "source".
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap between consecutive chunks.

    Returns:
        list[dict]: List of chunked texts with metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.create_documents(texts=[section["text"]], metadatas=[{"source": section["source"]}])
    return [{"text": chunk.page_content, "source": chunk.metadata["source"]} for chunk in chunks]


def build_index(docs_dir, chunk_size, chunk_overlap, embedding_model_name, collection_name):
    """
    Build an index from documents and store the results in Qdrant.

    Args:
        docs_dir (Path): Directory containing documents to process.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap between consecutive chunks.
        embedding_model_name (str): Name of the embedding model.
        collection_name (str): Name of the Qdrant collection.
    """
    logger.info("Starting the index-building process...")

    ds = ray.data.from_items([{"path": path} for path in docs_dir.rglob("*.pdf") if not path.is_dir()])
    sections_ds = ds.flat_map(process_pdf)
    chunks_ds = sections_ds.flat_map(partial(chunk_section, chunk_size=chunk_size, chunk_overlap=chunk_overlap))

    embedded_chunks = chunks_ds.map_batches(
        EmbedChunks,
        fn_constructor_kwargs={"model_name": embedding_model_name},
        batch_size=100,
        num_gpus=1,
        compute=ActorPoolStrategy(size=1),
    )

    embedded_chunks.map_batches(
        StoreResults(collection_name=collection_name),
        batch_size=128,
        num_cpus=1,
        compute=ActorPoolStrategy(size=6),
    ).count()

    logger.info("Index built and stored in Qdrant successfully!")


def load_index(embedding_model_name, chunk_size, chunk_overlap, docs_dir=None, collection_name=None):
    """
    Load or build the index. This ensures the Qdrant collection is up-to-date.

    Args:
        embedding_model_name (str): Name of the embedding model.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap between consecutive chunks.
        docs_dir (Path): Directory containing documents to process (optional).
        collection_name (str): Name of the Qdrant collection.

    Returns:
        list[dict]: List of chunks stored in Qdrant.
    """
    logger.info("Loading index from Qdrant...")

    vector_store = VectorStore(collection_name=collection_name)
    vector_store._ensure_collection_exists()

    # If no documents exist, build the index
    if not vector_store.collection_exists() or not vector_store.get_all_documents():
        logger.info("Qdrant collection is empty. Building a new index...")
        if not docs_dir:
            raise ValueError("No documents directory provided for building the index.")
        build_index(
            docs_dir=docs_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model_name=embedding_model_name,
            collection_name=collection_name,
        )

    # Fetch all chunks from the vector store
    return vector_store.get_all_documents()
