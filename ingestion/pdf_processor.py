from PyPDF2 import PdfReader


def process_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    # Chunk text
    chunks = chunk_text(text)
    return chunks


def chunk_text(text, chunk_size=1000, overlap=50):
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]
