# start_server.ps1
# Place in: gcp_graphrag_complete\gcp_graphrag\
# Run with: powershell -ExecutionPolicy Bypass -File .\start_server.ps1

Write-Host "Starting GCP-GraphRAG server..." -ForegroundColor Cyan

$env:HF_HUB_OFFLINE                        = "1"
$env:TRANSFORMERS_NO_SAFETENSORS_CONVERSION = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING       = "1"
$env:TOKENIZERS_PARALLELISM                 = "false"
$env:USE_REAL_GENERATOR                     = "1"
$env:GENERATOR_BACKEND                      = "ollama"
$env:OLLAMA_MODEL                           = "phi3:mini"

Write-Host "  Embedding  : Sentence-Transformers [cached, offline]" -ForegroundColor Green
Write-Host "  Generation : phi3:mini via Ollama" -ForegroundColor Green
Write-Host "  Preset     : Balanced (sentence-transformers + phi3:mini)" -ForegroundColor Green
Write-Host ""

$ollamaRunning = $false
try {
    Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 | Out-Null
    $ollamaRunning = $true
} catch {
    $ollamaRunning = $false
}

if ($ollamaRunning) {
    Write-Host "  Ollama     : running" -ForegroundColor Green
} else {
    Write-Host "  Ollama     : starting..." -ForegroundColor Yellow
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
    Write-Host "  Ollama     : started" -ForegroundColor Green
}

Write-Host ""
Write-Host "API  : http://localhost:8000" -ForegroundColor Cyan
Write-Host "UI   : http://localhost:3000" -ForegroundColor Cyan
Write-Host "Docs : http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""

python -m uvicorn server:app --reload --port 8000
