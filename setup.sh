if [ -z "$BASH_VERSION" ] && [ -z "$ZSH_VERSION" ]; then
  printf "\033[0;31m✗ ERRO: Este script requer Bash ou Zsh para ser executado corretamente.\n"
  printf "Por favor, execute-o usando: \033[0;32mbash ./setup.sh\033[0m ou \033[0;32mzsh ./setup.sh\033[0m\n\n\033[0m"
  exit 1
fi

set -euo pipefail
IFS=$'\n\t'

# --------------------------------------------------------
#  setup.sh — Configura o ambiente, venv, dependências e Docker
# --------------------------------------------------------

# Cores para mensagens
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' 


VENV_DIR=".venv"
REQUIREMENTS_FILE="requirements.txt"
GRADIO_APP_SCRIPT="scripts.gradio_app"
CHAT_INTERFACE_SCRIPT="scripts.chat_interface"
INGEST_DATA_SCRIPT="scripts.ingest_data"
DATA_RAW_DIR="data/raw" 
# -------------------------
#  Funções de utilitário
# -------------------------
function echo_err {
  printf "${RED}[ERRO]${NC} %s\n" "$1"
  exit 1 # Sai do script em caso de erro crítico
}
function echo_ok {
  printf "${GREEN}[OK]${NC} %s\n" "$1"
}
function echo_warn {
  printf "${YELLOW}[AVISO]${NC} %s\n" "$1"
}

# Checa se o comando existe
function check_cmd {
  if ! command -v "$1" &>/dev/null; then
    echo_err "Comando '$1' não encontrado. Por favor, instale-o e rode o script novamente."
  fi
}

# -------------------------
#  Início do Setup
# -------------------------
printf "${GREEN}==============================================\n"
printf "  Iniciando Configuração do Projeto RAG       \n"
printf "==============================================${NC}\n"

# -------------------------
# 1) Detectar sistema
# -------------------------
printf "\n${YELLOW}Detectando sistema operacional...${NC}\n"
OS_TYPE="$(uname -s)"
PLATFORM=""
case "$OS_TYPE" in
  Linux*)  PLATFORM="linux" ;;
  Darwin*) PLATFORM="macos" ;;
  CYGWIN*|MINGW*|MSYS*) PLATFORM="windows" ;;
  *)
    echo_warn "Sistema '$OS_TYPE' não reconhecido. Assumindo Linux para comandos de ativação."
    PLATFORM="linux"
    ;;
esac
echo_ok "Plataforma detectada: $PLATFORM"

# -------------------------
# 2) Verificar Python & pip
# -------------------------
printf "\n${YELLOW}Verificando Python e pip...${NC}\n"
PYTHON_CMD=""
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
fi

if [ -z "$PYTHON_CMD" ]; then # Se PYTHON_CMD ainda estiver vazio
    echo_err "Python não encontrado. Por favor, instale Python (versão 3.8+ recomendada) e tente novamente."
fi

if ! "$PYTHON_CMD" -m pip --version &>/dev/null; then
  echo_err "pip não encontrado para '$PYTHON_CMD'. Por favor, verifique sua instalação do Python."
fi
echo_ok "Python e pip encontrados."

# -------------------------
# 3) Criar / ativar venv
# -------------------------
printf "\n${YELLOW}Verificando e criando ambiente virtual...${NC}\n"
if [ ! -d "$VENV_DIR" ]; then
  printf "${YELLOW}Criando ambiente virtual '$VENV_DIR'...${NC}\n"
  "$PYTHON_CMD" -m venv "$VENV_DIR" || echo_err "Falha ao criar o ambiente virtual. Verifique as permissões ou a instalação do Python."
  echo_ok "Ambiente virtual criado."
else
  echo_ok "Ambiente virtual '$VENV_DIR' já existe, pulando criação."
fi

# Ativa a venv para o resto do script
printf "${YELLOW}Ativando ambiente virtual...${NC}\n"
VENV_ACTIVATE_CMD=""
if [ "$PLATFORM" = "windows" ]; then
  VENV_ACTIVATE_CMD="source \"$VENV_DIR/Scripts/activate\""
  source "$VENV_DIR/Scripts/activate"
else
  VENV_ACTIVATE_CMD="source \"$VENV_DIR/bin/activate\""
  source "$VENV_DIR/bin/activate"
fi
echo_ok "Ambiente virtual ativado."

# Sanity check: verificar se o python do venv está sendo usado
if ! command -v python &>/dev/null || [[ "$(command -v python)" != *"$VENV_DIR"* ]]; then
    echo_warn "O ambiente virtual pode não ter sido ativado corretamente. As dependências podem ser instaladas globalmente."
    echo_warn "Recomendado ativar manualmente o ambiente virtual e re-executar o script."
    echo "  Comando para ativação manual: ${GREEN}${VENV_ACTIVATE_CMD}${NC}"
fi


# -------------------------
# 4) Instalar dependências
# -------------------------
printf "\n${YELLOW}Verificando e instalando dependências...${NC}\n"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo_err "Arquivo '$REQUIREMENTS_FILE' não encontrado. Certifique-se de que ele está na raiz do projeto."
fi

# Idempotência: Verificar se as dependências chave já estão instaladas no venv
if python -c "import gradio, langchain, chromadb, sentence_transformers" &>/dev/null; then
    echo_ok "Dependências principais já parecem instaladas no ambiente virtual. Pulando instalação."
else
    printf "${YELLOW}Instalando dependências de '$REQUIREMENTS_FILE'...${NC}\n"
    pip install --upgrade pip || echo_warn "Falha ao atualizar pip. Continuando..."
    pip install --no-cache-dir -r "$REQUIREMENTS_FILE" || echo_err "Falha ao instalar dependências. Verifique a conexão ou erros no '$REQUIREMENTS_FILE'."
    echo_ok "Dependências instaladas com sucesso."
fi

# -------------------------
# 5) Docker Compose (opcional)
# -------------------------
printf "\n${YELLOW}Configuração do Docker Compose${NC}\n"
read -p "Deseja buildar e subir os containers Docker (Ollama)? [y/N] " DOCKER_ANS
if [[ "${DOCKER_ANS,,}" == "y" ]]; then
  check_cmd docker
  # Verifica se o daemon Docker está rodando
  if ! docker info &>/dev/null; then
    echo_err "Docker Daemon não está rodando. Por favor, inicie o serviço Docker (ex: Docker Desktop ou 'sudo systemctl start docker')."
  fi
  printf "${YELLOW}Executando: docker compose up --build -d${NC}\n"
  # Usando 'docker compose' (sem hífen) para a sintaxe moderna
  docker compose up --build -d || echo_err "Falha ao executar 'docker compose up'. Verifique o 'docker-compose.yml'."
  echo_ok "Containers Docker rodando em segundo plano."
else
  echo_warn "Pulando etapa Docker Compose. Você pode iniciá-lo manualmente mais tarde se necessário."
fi

# -------------------------
# Conclusão e Próximos Passos
# -------------------------
printf "\n${GREEN}==============================================\n"
printf "  Setup Concluído com Sucesso!                \n"
printf "==============================================${NC}\n"

printf "\n${YELLOW}Próximos Passos Essenciais:${NC}\n"
printf "1. ${GREEN}Iniciar o servidor Ollama (para o modelo de linguagem):${NC}\n"
printf "     Se você pulou a etapa anterior, no diretório raiz do projeto, execute:\n"
printf "     ${GREEN}     docker compose up -d --build${NC}\n"
printf "     (Use '-d' para rodar em segundo plano. Remova '-d' para ver os logs do Ollama.)\n"

printf "\n2. ${GREEN}Preencher a base vetorial com seus documentos:${NC}\n"
printf "   Certifique-se de que seus arquivos (PDFs, imagens) estão na pasta '${DATA_RAW_DIR}'.\n"
printf "   Execute o script de ingestão (isso pode levar tempo, especialmente com OCR):\n"
printf "   ${GREEN}   python -m $INGEST_DATA_SCRIPT${NC}\n"

printf "\n3. ${GREEN}Escolha como interagir com o assistente RAG:${NC}\n"
printf "   ${YELLOW}Opção A (Interface Gráfica - Gradio):${NC}\n"
printf "     Execute:\n"
printf "     ${GREEN}     python -m $GRADIO_APP_SCRIPT${NC}\n"
printf "     Acesse o link fornecido no seu navegador (geralmente http://127.0.0.1:7860).\n"
printf "   ${YELLOW}Opção B (Terminal Interativo):${NC}\n"
printf "     Execute:\n"
printf "     ${GREEN}     python -m $CHAT_INTERFACE_SCRIPT${NC}\n"