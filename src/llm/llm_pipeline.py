from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders.huggingface import HuggingFaceCrossEncoder
import requests

from config.settings import GEMINI_API_KEY

from sentence_transformers import CrossEncoder
import torch
import google.generativeai as genai

from config.settings import (
    TOP_K_RETRIEVAL,
    TOP_K_INITIAL_FOR_RERANKER
)
from src.embedding.vector_store_manager import get_retriever

def call_gemini_flash(prompt: str, model="gemini-2.0-flash") -> str:
    api_key = GEMINI_API_KEY
    if not api_key:
        raise EnvironmentError("Defina GEMINI_API_KEY no ambiente")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"Content-Type": "application/json", "X-goog-api-key": api_key}
    body = {"contents":[{"parts":[{"text": prompt}]}]}
    res = requests.post(url, headers=headers, json=body)
    res.raise_for_status()
    data = res.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return res.text


# Wrapper para Gemini via HTTP compatível com LangChain
class GeminiLLM:
    def __init__(self, model="gemini-2.0-flash"):
        self.model = model

    def __call__(self, prompt, **kwargs):
        """
        Chamada compatível com LangChain usando HTTP direto.
        Converte ChatPromptValue em string antes de enviar.
        """
        # Se o prompt for um ChatPromptValue, transforme em string
        if hasattr(prompt, "to_string"):
            prompt_text = prompt.to_string()
        else:
            prompt_text = str(prompt)

        return call_gemini_flash(prompt_text, model=self.model)


def get_rag_chain():
    """Configura e retorna a cadeia RAG do Biolog.ia, com re-ranking e busca no PDF de Biologia."""

    # 1️⃣ LLM principal (Gemini via HTTP)
    llm = GeminiLLM(model="gemini-2.0-flash")

    # 2️⃣ Retriever baseado no PDF indexado
    base_retriever = get_retriever(k=TOP_K_INITIAL_FOR_RERANKER)

    # 3️⃣ Seleciona o dispositivo automaticamente
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # 4️⃣ Cria o modelo de re-ranking (CrossEncoder)
    hf_cross_encoder = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        model_kwargs={"device": device}
    )

    reranker = CrossEncoderReranker(
        model=hf_cross_encoder,
        top_n=TOP_K_RETRIEVAL
    )

    # 5️⃣ Junta retriever e reranker
    compressed_retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=reranker
    )

    # 6️⃣ Prompt para respostas baseadas no livro
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é o **Biolog.ia**, um assistente de Biologia treinado com um livro completo da disciplina.

Siga as instruções cuidadosamente:
1. Use **somente as informações do livro** para responder.
2. Sempre que possível, indique **o capítulo e a página** de onde a informação foi retirada.
3.  Se a resposta não puder ser inferida ou encontrada **DIRETAMENTE** no Livro, responda:
         "Desculpe, não encontrei informações suficientes no livro fornecido para responder a essa pergunta." ou "Não consigo responder a essa pergunta com base nas informações disponíveis.".
4. Seja claro, didático e direto.
5. Caso o usuário peça por perguntas sobre um tema, gere perguntas relacionadas e avalie as respostas do usuário conforme o conteúdo do livro.
6. Se o usuário pedir por um roadmap ou ajuda em um capítulo específico, envie temas e subtemas do capítulo para estudo.

---
**Livro de referência (contexto):**
{context}
"""),
        ("human", "{input}")
    ])

    # 7️⃣ Cria a cadeia de documentos (stuffing)
    document_chain = create_stuff_documents_chain(llm, prompt)

    # 8️⃣ Monta a RAG final
    retrieval_chain = create_retrieval_chain(
        compressed_retriever,
        document_chain
    )

    return retrieval_chain
