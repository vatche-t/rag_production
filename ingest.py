import os
from ingestion.pdf_processor import process_pdf
from ingestion.embedder import EmbedChunks
from ingestion.vector_store import VectorStore
from utils.logger import logger

# Directory containing PDFs to process
PDF_DIR = "data/pdfs/"
# Name of the Qdrant collection
COLLECTION_NAME = "rag_collection"


def ingest_pdfs():
    """
    Process PDFs, generate embeddings, and store them in the Qdrant vector store.
    """
    # Initialize embedder and vector store
    embedder = EmbedChunks(model_name="nomic-embed-text")
    vector_store = VectorStore(collection_name=COLLECTION_NAME)

    for pdf_file in os.listdir(PDF_DIR):
        # Skip non-PDF files
        if not pdf_file.endswith(".pdf"):
            logger.warning(f"Skipping non-PDF file: {pdf_file}")
            continue

        pdf_path = os.path.join(PDF_DIR, pdf_file)
        logger.info(f"Processing {pdf_file}...")

        try:
            # Extract and chunk text from the PDF
            chunks = process_pdf(pdf_path)
            if not chunks:
                logger.warning(f"No text extracted from {pdf_file}. Skipping.")
                continue

            # Prepare data for embedding
            batch = [{"text": chunk, "source": pdf_file} for chunk in chunks]

            # Generate embeddings for the text chunks
            logger.info(f"Generating embeddings for {pdf_file}...")
            embeddings = embedder(batch)

            # Store embeddings and metadata in Qdrant
            logger.info(f"Storing embeddings for {pdf_file} in Qdrant...")
            vector_store.add_documents(batch, embeddings)

            logger.info(f"Successfully processed and stored {pdf_file}.")
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {e}")


if __name__ == "__main__":
    # Ensure PDF directory exists
    if not os.path.exists(PDF_DIR):
        logger.error(f"PDF directory not found: {PDF_DIR}")
    else:
        ingest_pdfs()
