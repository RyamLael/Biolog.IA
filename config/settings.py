import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBDqzxBaudGDRmiTDTrorXgrF_2CNu_HEo")
PDF_PATH = os.getenv("PDF_PATH", "data/LivroBiologia.pdf")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")
CHROMADB_PERSIST_DIRECTORY = os.getenv("CHROMADB_PERSIST_DIRECTORY", "data/processed/chroma_db")
CHROMADB_COLLECTION_NAME = os.getenv("CHROMADB_COLLECTION_NAME", "docs_base")
HUGGINGFACE_EMBEDDING_MODEL = os.getenv("HUGGINGFACE_EMBEDDING_MODEL","sentence-transformers/all-mpnet-base-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "90"))
TOP_K_INITIAL_FOR_RERANKER = int(os.getenv("TOP_K_INITIAL_FOR_RERANKER", "20"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))