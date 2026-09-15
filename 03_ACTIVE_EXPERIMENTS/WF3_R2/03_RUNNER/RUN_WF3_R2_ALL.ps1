$ErrorActionPreference = "Stop"

$Root = "C:\RSOS\RSOS_EXPERIMENTS_WF3_GENEALOGY_POPULATION_CONFIRMATION_R2"
$Runner = Join-Path $Root "03_RUNNER"

Write-Host "============================================================"
Write-Host " WF3-R2 SCIENTIFIC EXECUTION"
Write-Host "============================================================"
Write-Host ""

Write-Host "Executing V15..."
python (Join-Path $Runner "RUN_WF3_R2_V15_GENEALOGY.py")

if ($LASTEXITCODE -ne 0) {
    throw "V15 execution failed."
}

Write-Host ""
Write-Host "Executing V16..."
python (Join-Path $Runner "RUN_WF3_R2_V16_POPULATION.py")

if ($LASTEXITCODE -ne 0) {
    throw "V16 execution failed."
}

Write-Host ""
Write-Host "============================================================"
Write-Host " WF3-R2 EXECUTION COMPLETE"
Write-Host "============================================================"
