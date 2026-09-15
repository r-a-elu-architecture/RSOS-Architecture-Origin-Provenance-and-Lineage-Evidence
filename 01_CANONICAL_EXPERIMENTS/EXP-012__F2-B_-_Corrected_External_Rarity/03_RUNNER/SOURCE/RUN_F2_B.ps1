$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here
python -c "import numpy,pandas,scipy,sklearn,pyarrow,joblib; print('Dependency preflight: PASS')"
if ($LASTEXITCODE -ne 0) { throw "Missing Python dependency." }
python -u ".\RUN_F2_B.py" --input "C:\RSOS\Training data\F2_B_FRESH_WILDCHAT" --out "C:\RSOS\Training data\F2_B_CORRECTED_EXTERNAL_RARITY"
if ($LASTEXITCODE -ne 0) { throw "F2-B aborted. Do not reinterpret partial output." }
Write-Host ""
Write-Host "F2-B COMPLETE" -ForegroundColor Green
Write-Host "Result folder: C:\RSOS\Training data\F2_B_CORRECTED_EXTERNAL_RARITY"
Write-Host "Upload package: C:\RSOS\Training data\F2_B_CORRECTED_EXTERNAL_RARITY_RESULTS.zip"
