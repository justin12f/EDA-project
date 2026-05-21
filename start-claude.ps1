

$env:ANTHROPIC_BASE_URL="http://localhost:11434/v1"
$env:ANTHROPIC_API_KEY="local"


npx @anthropic-ai/claude-code config set ANTHROPIC_BASE_URL http://localhost:11434/v1

Write-Host "--- Iniciando Claude Code en modo local ---" -ForegroundColor Cyan


npx @anthropic-ai/claude-code --model qwen2.5-coder