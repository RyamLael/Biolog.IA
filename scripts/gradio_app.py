import os
import gradio as gr
from src.llm.llm_pipeline import get_rag_chain
from src.embedding.vector_store_manager import get_embedding_function
import warnings
import logging
from config.settings import CHROMADB_PERSIST_DIRECTORY
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='gradio')
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

ROOT_DIR = Path(__file__).resolve().parent.parent
rag_chain = get_rag_chain()
embeddings_func = get_embedding_function() 
LOGO_PATH = str(ROOT_DIR / "assets" / "simbolo.svg")
PDF_PATH = str(ROOT_DIR / "assets" / "LivroBiologia.pdf")

def respond(message, history):
    try:
        response = rag_chain.invoke({"input": message})
        answer = response.get("answer", "")
        
        sources_info = "\n\n**Fontes:**\n"
        seen_sources = set()
        if "context" in response and response["context"]:
            for doc in response["context"]:
                source_name = doc.metadata.get('source', 'Desconhecida')
                if source_name not in seen_sources and source_name != 'N/A':
                    sources_info += f"- {os.path.basename(source_name)}\n"
                    seen_sources.add(source_name)
        
        if seen_sources:
            answer += sources_info
        else:
            answer += "\n\n(Nenhuma fonte específica encontrada para esta resposta no contexto imediato.)"

        return answer
    except Exception as e:
        return f"Ocorreu um erro ao processar sua pergunta: {e}"

# Função para gerar HTML com PDF embutido
def show_pdf_embed():
    pdf_path = "assets/LivroBiologia.pdf"
    if not os.path.exists(pdf_path):
        return "<p>PDF não encontrado.</p>"
    # Gradio entende "file=" como arquivo local
    return f'<iframe src="file={pdf_path}" width="100%" height="600px" style="border:none;"></iframe>'

with gr.Blocks(theme=gr.themes.Ocean(),elem_classes="main", title="Biolog.ia: Um tutor virtual de Biologia") as demo:

    # Cabeçalho com logo e título
    with gr.Row(variant="panel", equal_height=False, elem_classes=["header-row"]):
        with gr.Column(scale=1):
            if os.path.exists(LOGO_PATH):
                gr.Image(
                    value=LOGO_PATH,
                    show_label=False,
                    height=80,
                    container=False,
                    scale=0,
                    interactive=False
                )
            
        with gr.Column(scale=7):
            gr.Markdown(
                """
                # Biolog.ia: Um tutor virtual de Biologia
                Tire dúvidas, pratique quizzes e veja onde encontrar as respostas no livro.
                """
            )

    demo.css = """
    /* --- CONFIGURAÇÕES GERAIS --- */
    /* Define um tamanho de fonte base maior para o container principal do Gradio */
    .gradio-container {
        font-size: 1.1em !important; 
        max-width: 1200px;
        margin-left: 20%;
        margin-right: 20%;
        padding: 20px !important;
    }
    body {
        font-size: 1.1em !important; 
    }
    .message {
        font-size: 1.05em; 
    }
    .header-row {
        align-items: center; 
        justify-content: space-between; 
        margin-bottom: 20px; 
    }
    .send-button {
        height: 100%; 
    }

    /* --- NOVOS ESTILOS PARA CORES DE FUNDO --- */
    
    /* 1. Fundo do Chatbot (Área de Mensagens) */
    .gradio-chatbot {
        background-color: #f0f0f0 !important; /* Um cinza bem claro para contraste */
        border: 3px solid #429bf5 !important; /* Opcional: Adicionar uma borda para delimitar */
    }

    /* 2. Campo de Texto (Input) */
    /* Apontamos para o elemento <textarea> dentro do textbox */
    .gradio-textbox textarea {
        background-color: #8f8f8f !important; /* Branco puro */
        color: #333333 !important; /* Cor do texto digitado */
    }
    
    /* Opcional: Estiliza o placeholder também */
    .gradio-textbox textarea::placeholder {
        color: #aaaaaa !important;
    }
    
    /* Para o tema */
    :root {
        color-scheme: light dark !important;
    }
    """

    # Função para atualizar o tema em tempo real
    # def update_theme(selected_theme):
    #     theme = gr.themes.Default() if selected_theme == "Claro" else gr.themes.Dark()
    #     return gr.update(theme=theme)

    # theme_selector.change(update_theme, inputs=[theme_selector], outputs=[demo])

    # Chat
    with gr.Tab("🐸 Converse com a g.ia"):
        chatbot = gr.ChatInterface(
            respond,
            chatbot=gr.Chatbot(
                height=400,
                type='messages',
                value=[{"role": "assistant", "content": "Olá, eu sou g.ia! Como posso lhe ajudar?"}]
            ), 
            textbox=gr.Textbox(
                placeholder="Digite sua pergunta aqui...",
                container=False,
                scale=7
            ),
        )

    # Leitor de PDF fixo embutido
    with gr.Tab("📖 Livro de referência"):
        if os.path.exists(PDF_PATH):
            gr.File(
                value=PDF_PATH,
                label="Livro de Biologia (Visualização)",
                interactive=False,
                height=600,
                show_label=True,
            )
        else:
             gr.Markdown(f"<p style='text-align:center; padding: 20px; color:red;'>⚠️ **Erro:** O arquivo PDF 'LivroBiologia.pdf' não foi encontrado no caminho: <code>{PDF_PATH}</code></p>")

    # Footer
    with gr.Row():
        gr.Markdown(
            """
            <hr>
                <div style="text-align:center; font-size:0.8em;">
                    <p>v.0.01</p>
                    <h3>Desenvolvido por Ryam Lael para o projeto Pet Saúde Digital</h3>
                </div>
            """
        )

if __name__ == "__main__":
    os.makedirs(CHROMADB_PERSIST_DIRECTORY, exist_ok=True)
    print("\nAcesse a interface do Biolog.ia com Gradio:")
    demo.launch()
