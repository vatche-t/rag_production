
# **Custom RAG-Based LLM System**

This project implements a custom **RAG-based (Retrieval-Augmented Generation)** system using a locally hosted Llama model, Qdrant as the vector database, and PyPDF for processing PDF documents. The system allows users to upload documents, embed and store their content, and query the stored data using natural language.

---

## **Features**

- **PDF Ingestion**: Extracts text from local PDF files, chunks the data, and embeds it using `nomic-embed-text`.
- **Vector Database**: Uses Qdrant to store and retrieve vectorized document chunks.
- **Custom Query Agent**: Processes user queries, retrieves relevant chunks, re-ranks results, and generates a response using Llama 3.2.
- **FastAPI Integration**: Provides API endpoints for interacting with the system.
- **Streamlit UI**: A user-friendly interface for uploading documents and querying data.
- **Pre-trained Models**:
  - Embedding: `nomic-embed-text`
  - LLM: Local Llama 3.2 model via Ollama API.

---

## **Project Structure**

```plaintext
.
├── configs/                 # Configuration files for Qdrant and Ollama
├── data/                    # Folder to store PDF files and intermediate data
├── ingestion/               # Handles data ingestion and vectorization
│   ├── embedder.py          # Embedding generation for document chunks
│   ├── vector_store.py      # Interface with Qdrant for storage and retrieval
│   ├── pdf_processor.py     # PDF text extraction and chunking
│   ├── index.py             # Builds and loads vector database index
├── query/                   # Query agent and search functionalities
│   ├── query_agent.py       # Handles query processing and LLM response generation
│   ├── reranker.py          # Custom re-ranking logic for retrieved chunks
│   ├── search.py            # Lexical and semantic search logic
├── utils/                   # Utility functions and tools
│   ├── logger.py            # Logging setup
│   ├── config.py            # Environment variable management
│   ├── utils.py             # General helper functions
├── app.py                   # Streamlit app for user interaction
├── ingest.py                # Script to process and ingest PDFs
├── generate.py              # Generates responses to user queries
├── .env                     # Environment variables
├── .gitignore               # Git ignore file
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## **Setup**

### 1. Clone the Repository
```bash
git clone <repository_url>
cd <repository_name>
```

### 2. Install Dependencies
Make sure you have Python 3.9+ installed. Then run:
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the project root with the following content:
```plaintext
# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=rag_collection

# Ollama (LLM API)
OLLAMA_API_BASE=http://localhost:11434
OLLAMA_API_KEY=<your_api_key>
```

### 4. Start Qdrant
Ensure Qdrant is running locally:
```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
```

---

## **Usage**

### 1. Ingest PDFs
Run the `ingest.py` script to process PDFs, embed chunks, and store them in Qdrant:
```bash
python ingest.py
```
### 2. Query Data
Use `generate.py` to query the stored data:
```bash
python generate.py
```
### 3. API Usage 
Start the FastAPI server and interact with the system via REST API:
```bash
uvicorn api.main:app --reload
```

---

## **Key Components**

### **Ingestion Pipeline**
1. Extracts text from PDFs using PyPDF2.
2. Splits text into manageable chunks using LangChain’s `RecursiveCharacterTextSplitter`.
3. Embeds the chunks using `nomic-embed-text`.
4. Stores vectors and metadata in Qdrant.

### **Query Processing**
1. Embeds the user query.
2. Retrieves top-k relevant chunks from Qdrant.
3. Re-ranks results using custom logic.
4. Generates a response using the Llama 3.2 model.

---

## **Dependencies**

See the full list in `requirements.txt`. Key libraries include:
- **LangChain**: Text processing and embeddings.
- **Qdrant**: Vector database.
- **Ollama**: Local Llama API client.
- **Streamlit**: Interactive web app.
- **PyPDF2**: PDF text extraction.
- **Ray**: Distributed processing for ingestion.

---

## **Roadmap**

- Add support for additional document formats (e.g., Word, HTML).
- Enhance the re-ranking logic with deep learning models.
- Integrate a Slack bot for real-time interaction.
- Add support for larger models with distributed GPU inference.

---
