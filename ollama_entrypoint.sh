
echo "Iniciando o servidor Ollama em segundo plano..."
ollama serve &
pid=$! 

echo "Aguardando 5 segundos para o Ollama subir..."
sleep 5

echo "🔴 Puxando o modelo mistral:7b..."
ollama pull mistral:7b
echo "🟢 Download do modelo mistral:7b concluído! Pronto para uso!"

wait $pid