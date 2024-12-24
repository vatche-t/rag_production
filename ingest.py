import asyncio
import os
from ingestion.pdf_processor import process_pdf
from ingestion.embedder import EmbedChunks
from ingestion.vector_store import VectorStore
from loguru import logger

# Directory containing PDFs to process
PDF_DIR = "data/pdfs/"
COLLECTION_NAME = "rag_collection"


async def ingest_pdfs():
    """
    Process PDFs, generate embeddings, and store them in the Qdrant vector store asynchronously.
    """
    embedder = EmbedChunks(model_name="nomic-embed-text")
    vector_store = VectorStore(collection_name=COLLECTION_NAME)

    for pdf_file in os.listdir(PDF_DIR):
        if not pdf_file.endswith(".pdf"):
            logger.warning(f"Skipping non-PDF file: {pdf_file}")
            continue

        pdf_path = os.path.join(PDF_DIR, pdf_file)
        logger.info(f"Processing {pdf_file}...")

        try:
            chunks = process_pdf(pdf_path)
            if not chunks:
                logger.warning(f"No text extracted from {pdf_file}. Skipping.")
                continue

            logger.debug(f"Extracted {len(chunks)} chunks from {pdf_file}.")

            # Generate embeddings asynchronously
            texts = [chunk for chunk in chunks]
            logger.info(f"Generating embeddings for {pdf_file}...")
            embeddings = await embedder.generate_embeddings_async(texts)

            # Filter out invalid embeddings
            valid_data = [
                {"text": texts[i], "source": pdf_file, "embedding": embeddings[i]}
                for i in range(len(embeddings))
                if embeddings[i] is not None
            ]

            if not valid_data:
                logger.warning(f"No valid embeddings generated for {pdf_file}. Skipping.")
                continue

            # Add documents to Qdrant
            logger.info(f"Storing embeddings for {pdf_file} in Qdrant...")
            vector_store.add_documents(valid_data)

            logger.info(f"Successfully processed and stored {pdf_file}.")
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {e}")


if __name__ == "__main__":
    if not os.path.exists(PDF_DIR):
        logger.error(f"PDF directory not found: {PDF_DIR}")
    else:
        asyncio.run(ingest_pdfs())
