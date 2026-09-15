$ErrorActionPreference = "Stop"

$Root      = "C:\RSOS\Training data\F2_EXTERNAL_POPULATION_RARITY"
$RunnerZip = "C:\RSOS\Training data\F2_A_WILDCHAT_RARITY_RUNNER.zip"
$Py        = "$Root\RUN_F2_A_WILDCHAT_RARITY.py"
$FinalZip  = "$Root\F2_EXTERNAL_POPULATION_RARITY_COMPLETE.zip"
$TempZip   = Join-Path $env:TEMP "F2_EXTERNAL_POPULATION_RARITY_COMPLETE.zip"

if (!(Test-Path $Root)) {
    New-Item -ItemType Directory -Force -Path $Root | Out-Null
}

if (Test-Path $RunnerZip) {
    Copy-Item $RunnerZip "$Root\F2_A_WILDCHAT_RARITY_RUNNER_ORIGINAL.zip" -Force
}

Write-Host "Checking Python dependencies..." -ForegroundColor Yellow
python -c "import numpy,pandas,scipy,pyarrow" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing required local analysis packages..." -ForegroundColor Yellow
    python -m pip install --upgrade numpy pandas scipy pyarrow
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
}

Write-Host "Compiling F2-A runner..." -ForegroundColor Yellow
python -m py_compile $Py
if ($LASTEXITCODE -ne 0) { throw "F2-A compile failed." }

Write-Host "Running F2-A against the three frozen WildChat-4.8M shards..." -ForegroundColor Green
python -u $Py
if ($LASTEXITCODE -ne 0) { throw "F2-A failed with exit code $LASTEXITCODE" }

if (Test-Path $FinalZip) { Remove-Item $FinalZip -Force }
if (Test-Path $TempZip) { Remove-Item $TempZip -Force }

Write-Host "Packaging the complete one-folder F2 experiment locally..." -ForegroundColor Yellow
# Build outside the experiment folder to avoid the archive including itself.
tar.exe -a -c -f $TempZip -C $Root .
if ($LASTEXITCODE -ne 0) { throw "F2 final ZIP creation failed." }
Move-Item $TempZip $FinalZip -Force

$Hash = (Get-FileHash $FinalZip -Algorithm SHA256).Hash.ToLower()
$Hash | Set-Content "$Root\F2_EXTERNAL_POPULATION_RARITY_COMPLETE.zip.sha256.txt" -Encoding ASCII

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "F2 EXTERNAL POPULATION RARITY COMPLETE" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "ZIP: $FinalZip"
Write-Host "SHA256: $Hash"
Write-Host ""
Get-Content "$Root\10_FINAL_MANIFEST\F2_A_SUMMARY.txt"
