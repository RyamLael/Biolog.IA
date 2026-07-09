import warnings
import logging
from src.processing.document_processing import load_and_chunk_documents
from src.embedding.vector_store_manager import ingest_documents_into_vector_store
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, PDF_PATH

# Suprimir avisos desnecessários
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="gradio")

# Configurar logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

def run_ingestion():
    """
    Executa o pipeline de ingestão do livro fixo de Biologia.
    O usuário não pode adicionar novos documentos.
    """
    print("📘 Iniciando ingestão do livro de Biologia...")

    # Verificar se o PDF existe
    try:
        with open(PDF_PATH, "rb"):
            pass
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {PDF_PATH}")
        print("Coloque o livro 'biologia.pdf' na pasta 'data/'.")
        return

    # Carregar e dividir o PDF em chunks
    chunks = load_and_chunk_documents(PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP)

    # Ingerir os chunks no banco vetorial
    ingest_documents_into_vector_store(chunks)

    print("✅ Ingestão concluída com sucesso!")

if __name__ == "__main__":
    run_ingestion()
