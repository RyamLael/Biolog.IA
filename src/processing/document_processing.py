import fitz
import os
import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extrai texto nativo de um PDF (texto selecionável).
    """
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"Erro ao extrair texto de {pdf_path}: {e}")
        return None
    return text.strip()

def save_extracted_text_to_file(text_content: str, pdf_path: str):
    """
    Salva o texto extraído em um arquivo .txt.
    """
    output_dir = "data/processed/ocr_text"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"text_extracted_from_{base_name}_{timestamp}.txt"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text_content)

    print(f"Texto extraído salvo em: {output_path}")

def load_and_chunk_documents(pdf_path: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """
    Extrai texto de um PDF de biologia e divide em chunks usando LangChain.
    """
    text_content = extract_text_from_pdf(pdf_path)
    if not text_content:
        print("Nenhum texto extraído do PDF.")
        return []

    save_extracted_text_to_file(text_content, pdf_path)

    doc = Document(page_content=text_content, metadata={"source": pdf_path, "file_type": "pdf"})

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )

    chunks = text_splitter.split_documents([doc])
    print(f"{len(chunks)} chunks criados a partir do PDF.")

    return chunks
