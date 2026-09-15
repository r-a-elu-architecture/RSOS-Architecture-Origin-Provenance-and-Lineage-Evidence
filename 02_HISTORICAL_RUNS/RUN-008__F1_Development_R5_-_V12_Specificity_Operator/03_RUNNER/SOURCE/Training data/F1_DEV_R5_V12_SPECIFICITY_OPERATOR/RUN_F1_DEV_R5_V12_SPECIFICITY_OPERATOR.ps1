$ErrorActionPreference = "Stop"

$Root = "C:\RSOS\Training data\F1_DEV_R5_V12_SPECIFICITY_OPERATOR"
$Py = "$Root\RUN_F1_DEV_R5_V12_SPECIFICITY_OPERATOR.py"
$FinalZip = "$Root\F1_DEV_R5_V12_SPECIFICITY_OPERATOR_COMPLETE.zip"
$ManifestDir = "$Root\00_MANIFEST"

if (!(Test-Path $Py)) {
    throw "Python runner missing: $Py"
}

Write-Host ""
Write-Host "Compiling R5..." -ForegroundColor Yellow
python -m py_compile $Py
if ($LASTEXITCODE -ne 0) { throw "Python compilation failed." }

Write-Host "COMPILE PASS" -ForegroundColor Green
Write-Host ""
Write-Host "Starting R5..." -ForegroundColor Green
python -u $Py
if ($LASTEXITCODE -ne 0) { throw "R5 execution failed with exit code $LASTEXITCODE" }

$PsHash = (Get-FileHash $PSCommandPath -Algorithm SHA256).Hash.ToLower()
$PsHash | Set-Content "$ManifestDir\POWERSHELL_RUNNER_SHA256.txt" -Encoding ASCII

if (Test-Path $FinalZip) { Remove-Item $FinalZip -Force }

$Stage = Join-Path $env:TEMP "F1_DEV_R5_FINAL_STAGE"
if (Test-Path $Stage) { Remove-Item $Stage -Recurse -Force }
New-Item -ItemType Directory -Path $Stage -Force | Out-Null

Get-ChildItem $Root |
    Where-Object { $_.FullName -ne $FinalZip } |
    ForEach-Object { Copy-Item $_.FullName $Stage -Recurse -Force }

Compress-Archive -Path "$Stage\*" -DestinationPath $FinalZip -CompressionLevel Optimal
Remove-Item $Stage -Recurse -Force

$ZipHash = (Get-FileHash $FinalZip -Algorithm SHA256).Hash.ToLower()
$ZipHash | Set-Content "$ManifestDir\FINAL_ZIP_SHA256.txt" -Encoding ASCII

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "F1-DEV R5 COMPLETE - ONE FOLDER PACKAGE" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "FOLDER: $Root"
Write-Host "ZIP:    $FinalZip"
Write-Host "SHA256: $ZipHash"
Write-Host ""
Get-Content "$ManifestDir\R5_SUMMARY.txt"
