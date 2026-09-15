$ErrorActionPreference = "Stop"

$Root = "C:\RSOS\Training data\F1_DEV_R4_V12_GRAPH_CALIBRATION"
$Py = "$Root\RUN_F1_DEV_R4_V12_GRAPH_CALIBRATION.py"
$ThisPs1 = "$Root\RUN_F1_DEV_R4_V12_GRAPH_CALIBRATION.ps1"
$FinalZip = "$Root\F1_DEV_R4_V12_GRAPH_CALIBRATION_COMPLETE.zip"

if (!(Test-Path $Root)) {
    New-Item -ItemType Directory -Force -Path $Root | Out-Null
}

python -c "import pandas,numpy,networkx,sklearn,scipy,joblib,pyarrow; print('Dependencies PASS')"
if ($LASTEXITCODE -ne 0) {
    python -m pip install --upgrade pandas numpy networkx scikit-learn scipy joblib pyarrow
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
}

python -m py_compile $Py
if ($LASTEXITCODE -ne 0) { throw "Python compilation failed." }

python $Py
if ($LASTEXITCODE -ne 0) { throw "R4 execution failed." }

# Hash the PowerShell runner as provenance.
$PsHash = (Get-FileHash $ThisPs1 -Algorithm SHA256).Hash.ToLower()
$PsHash | Set-Content "$Root\00_MANIFEST\POWERSHELL_RUNNER_SHA256.txt" -Encoding ASCII

# Build final ZIP inside the SAME experiment folder, while avoiding self-recursion.
if (Test-Path $FinalZip) { Remove-Item $FinalZip -Force }

$Stage = Join-Path $env:TEMP "F1_DEV_R4_STAGE"
if (Test-Path $Stage) { Remove-Item $Stage -Recurse -Force }
New-Item -ItemType Directory -Path $Stage -Force | Out-Null

Get-ChildItem $Root |
    Where-Object { $_.FullName -ne $FinalZip } |
    ForEach-Object {
        Copy-Item $_.FullName $Stage -Recurse -Force
    }

Compress-Archive -Path "$Stage\*" -DestinationPath $FinalZip -CompressionLevel Optimal
Remove-Item $Stage -Recurse -Force

$ZipHash = (Get-FileHash $FinalZip -Algorithm SHA256).Hash.ToLower()
$ZipHash | Set-Content "$Root\00_MANIFEST\FINAL_ZIP_SHA256.txt" -Encoding ASCII

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "F1-DEV R4 COMPLETE — ONE-FOLDER PACKAGE" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "FOLDER: $Root"
Write-Host "ZIP:    $FinalZip"
Write-Host "SHA256: $ZipHash"
Write-Host ""
Get-Content "$Root\00_MANIFEST\R4_SUMMARY.txt"
