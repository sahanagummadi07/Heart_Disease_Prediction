Set-Location $PSScriptRoot
python scripts/fetch_data.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/train_models.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
