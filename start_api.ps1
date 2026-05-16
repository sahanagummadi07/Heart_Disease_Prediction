# Run:  .\start_api.ps1
#       .\start_api.ps1 -Port 8001    (if 8000 is already in use)

param(
    [int]$Port = 8000
)

Set-Location $PSScriptRoot
Write-Host ""
Write-Host " Starting API: http://127.0.0.1:$Port" -ForegroundColor Cyan
Write-Host " Keep this window open while you use the site."
Write-Host " Press Ctrl+C to stop the server."
Write-Host ""
python -m uvicorn backend.main:app --host 127.0.0.1 --port $Port
