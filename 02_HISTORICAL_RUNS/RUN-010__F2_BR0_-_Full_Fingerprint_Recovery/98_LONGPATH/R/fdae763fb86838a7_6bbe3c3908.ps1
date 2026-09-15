$ErrorActionPreference = "Stop"

$Root = "C:\RSOS\Training data\F1_BR2_LMARENA_MATCHED_CONTROL"
$Py   = "$Root\RUN_F1_BR2_V2_EXACT_EXTRACTOR.py"
$Zip  = "$Root\F1_BR2_LMARENA_MATCHED_CONTROL_COMPLETE_V2.zip"

if (!(Test-Path $Py)) { throw "Missing V2 runner: $Py" }

Write-Host "Compiling corrected BR2 V2..." -ForegroundColor Yellow
python -m py_compile $Py
if ($LASTEXITCODE -ne 0) { throw "V2 compile failed." }

Write-Host "COMPILE PASS" -ForegroundColor Green
Write-Host "Running exact-extractor BR2 V2..." -ForegroundColor Green

python -u $Py
if ($LASTEXITCODE -ne 0) { throw "BR2 V2 failed with exit code $LASTEXITCODE" }

if (Test-Path $Zip) { Remove-Item $Zip -Force }

$Tmp = Join-Path $env:TEMP "F1_BR2_V2_PACKAGE"
if (Test-Path $Tmp) { Remove-Item $Tmp -Recurse -Force }
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null

Get-ChildItem $Root |
    Where-Object { $_.FullName -ne $Zip } |
    ForEach-Object { Copy-Item $_.FullName $Tmp -Recurse -Force }

Compress-Archive `
    -Path "$Tmp\*" `
    -DestinationPath $Zip `
    -CompressionLevel Optimal

Remove-Item $Tmp -Recurse -Force

$Hash = (Get-FileHash $Zip -Algorithm SHA256).Hash.ToLower()
$Hash | Set-Content "$Root\11_V2_MANIFEST\FINAL_V2_ZIP_SHA256.txt" -Encoding ASCII

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "F1-BR2 V2 COMPLETE" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "ZIP: $Zip"
Write-Host "SHA256: $Hash"
Write-Host ""
Get-Content "$Root\11_V2_MANIFEST\BR2_V2_SUMMARY.txt"
