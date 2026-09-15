$ErrorActionPreference = "Stop"

$Py  = "C:\RSOS\Training data\RUN_F1_DEV_LMARENA_BLIND_R3_HARDENED.py"
$Out = "C:\RSOS\Training data\F1_DEV_LMARENA_BLIND_R3_HARDENED"
$Zip = "C:\RSOS\Training data\F1_DEV_LMARENA_BLIND_R3_HARDENED_RESULTS.zip"

python -c "import pandas,numpy,pyarrow,sklearn,scipy,joblib,networkx; print('Dependencies PASS')"
if ($LASTEXITCODE -ne 0) {
    python -m pip install --upgrade pandas numpy pyarrow scikit-learn scipy joblib networkx
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
}

python -m py_compile $Py
if ($LASTEXITCODE -ne 0) { throw "Python compilation failed." }

python $Py
if ($LASTEXITCODE -ne 0) { throw "R3 execution failed." }

if (Test-Path $Zip) { Remove-Item $Zip -Force }
Compress-Archive -Path "$Out\*" -DestinationPath $Zip -CompressionLevel Optimal

$Hash = (Get-FileHash $Zip -Algorithm SHA256).Hash.ToLower()

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "F1-DEV R3 HARDENED COMPLETE AND FROZEN" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "RESULTS: $Out"
Write-Host "ZIP:     $Zip"
Write-Host "SHA256:  $Hash"
Write-Host ""
Get-Content "$Out\00_MANIFEST\R3_SUMMARY.txt"
