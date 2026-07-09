import os
from src.llm.llm_pipeline import get_rag_chain
from config.settings import OLLAMA_MODEL
from langchain_core.messages import HumanMessage, AIMessage
from src.embedding.vector_store_manager import get_chroma_vector_store, get_embedding_function, get_retriever 
from config.settings import TOP_K_INITIAL_FOR_RERANKER
import warnings
import logging

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='gradio')
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

def run_chat_interface():
    print(f"Bem-vindo ao assistente de RAG (alimentado por {OLLAMA_MODEL})!")
    print("Digite sua pergunta ou 'sair' para encerrar.")

    rag_chain = get_rag_chain()
    chat_history = []


    while True:
        user_input = input("\nSua pergunta: ")
        if user_input.lower() == 'sair':
            print("Encerrando o chat. Até mais!")
            break

        try:
            response = rag_chain.invoke({"input": user_input})
            print("\nResposta do Assistente:")
            print(response["answer"])

        except Exception as e:
            print(f"Ocorreu um erro: {e}")
            print("Por favor, tente novamente.")

if __name__ == "__main__":
    run_chat_interface()
