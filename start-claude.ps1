# Script de automatización para Justin Franco
# Redirigiendo a Ollama local (GTX 1060)

# 1. Configurar variables de entorno para esta sesión
$env:ANTHROPIC_BASE_URL="http://localhost:11434/v1"
$env:ANTHROPIC_API_KEY="local"

# 2. Forzar la configuración persistente usando npx
npx @anthropic-ai/claude-code config set ANTHROPIC_BASE_URL http://localhost:11434/v1

Write-Host "--- Iniciando Claude Code en modo local ---" -ForegroundColor Cyan

# 3. Ejecutar Claude Code forzando tu modelo de Ollama
npx @anthropic-ai/claude-code --model qwen2.5-coder