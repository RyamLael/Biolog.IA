import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_core.documents import Document
from config.settings import CHROMADB_PERSIST_DIRECTORY, CHROMADB_COLLECTION_NAME, TOP_K_RETRIEVAL, HUGGINGFACE_EMBEDDING_MODEL
import torch

def get_embedding_function():
    """Retorna a função de embedding usando HuggingFaceEmbeddings com o modelo especificado."""
    model_name = HUGGINGFACE_EMBEDDING_MODEL 
    
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available(): 
        device = 'mps'
    else:
        device = 'cpu'

    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': device})

def get_chroma_vector_store(embedding_function) -> Chroma:
    """Retorna uma instância de ChromaDB (persistente) com a função de embedding."""
    client = chromadb.PersistentClient(path=CHROMADB_PERSIST_DIRECTORY)
    return Chroma(
        client=client,
        collection_name=CHROMADB_COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=CHROMADB_PERSIST_DIRECTORY
    )

def ingest_documents_into_vector_store(chunks: list[Document]):
    """Adiciona chunks ao banco de dados vetorial."""
    embeddings = get_embedding_function()
    vector_store = get_chroma_vector_store(embeddings)

    print(f"Adicionando {len(chunks)} chunks à coleção ChromaDB '{CHROMADB_COLLECTION_NAME}'...")
    vector_store.add_documents(chunks)
    print("Ingestão concluída.")

def get_retriever(k: int = None):
    """
    Retorna um retriever simples para PDFs (sem imagens).
    - k: número de chunks a retornar
    """
    embeddings = get_embedding_function()
    vector_store = get_chroma_vector_store(embeddings)
    actual_k = k if k is not None else TOP_K_RETRIEVAL

    retriever = vector_store.as_retriever(  
        search_type="mmr",
        search_kwargs={
            "k": actual_k,
            "fetch_k": max(actual_k * 3, 30),
            "lambda_mult": 0.1,
            "filter": {"file_type": "pdf"}  # mantém apenas PDFs
        }
    )

    return retriever
