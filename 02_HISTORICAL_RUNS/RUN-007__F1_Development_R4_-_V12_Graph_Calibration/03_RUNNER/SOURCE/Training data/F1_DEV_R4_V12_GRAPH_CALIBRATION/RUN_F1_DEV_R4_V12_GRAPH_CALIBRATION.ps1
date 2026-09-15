$ErrorActionPreference = "Stop"

$Root = "C:\RSOS\Training data\F1_DEV_R4_V12_GRAPH_CALIBRATION"
$Py = "$Root\RUN_F1_DEV_R4_V12_GRAPH_CALIBRATION.py"
$FinalZip = "$Root\F1_DEV_R4_V12_GRAPH_CALIBRATION_COMPLETE.zip"
$Manifest = "$Root\00_MANIFEST\COMPLETE_PACKAGE_SHA256_MANIFEST.csv"

if (!(Test-Path $Py)) {
    throw "Python runner missing: $Py"
}

Write-Host ""
Write-Host "Checking dependencies..." -ForegroundColor Yellow

python -c "import pandas,numpy,networkx,sklearn,scipy,joblib,pyarrow; print('Dependencies PASS')"

if ($LASTEXITCODE -ne 0) {
    python -m pip install --upgrade pandas numpy networkx scikit-learn scipy joblib pyarrow
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed."
    }
}

Write-Host ""
Write-Host "Compiling Python runner..." -ForegroundColor Yellow

python -m py_compile $Py

if ($LASTEXITCODE -ne 0) {
    throw "Python compilation failed."
}

Write-Host "COMPILE PASS" -ForegroundColor Green

Write-Host ""
Write-Host "Starting F1-DEV R4..." -ForegroundColor Green
Write-Host ""

python $Py

if ($LASTEXITCODE -ne 0) {
    throw "R4 execution failed."
}

Write-Host ""
Write-Host "Creating complete provenance manifest..." -ForegroundColor Yellow

$PsHash = (Get-FileHash $PSCommandPath -Algorithm SHA256).Hash.ToLower()

$PsHash |
    Set-Content `
        "$Root\00_MANIFEST\POWERSHELL_RUNNER_SHA256.txt" `
        -Encoding ASCII

if (Test-Path $FinalZip) {
    Remove-Item $FinalZip -Force
}

if (Test-Path $Manifest) {
    Remove-Item $Manifest -Force
}

$Files = Get-ChildItem `
    $Root `
    -Recurse `
    -File |
    Where-Object {
        $_.FullName -ne $FinalZip -and
        $_.FullName -ne $Manifest
    }

$HashRows = foreach ($File in $Files) {

    [pscustomobject]@{
        RELATIVE_PATH = $File.FullName.Substring($Root.Length + 1)
        BYTES = $File.Length
        SHA256 = (
            Get-FileHash `
                $File.FullName `
                -Algorithm SHA256
        ).Hash.ToLower()
    }
}

$HashRows |
    Export-Csv `
        $Manifest `
        -NoTypeInformation `
        -Encoding UTF8

Write-Host ""
Write-Host "Creating one-folder final ZIP..." -ForegroundColor Yellow

$Stage = Join-Path $env:TEMP "F1_DEV_R4_PACKAGE_STAGE"

if (Test-Path $Stage) {
    Remove-Item $Stage -Recurse -Force
}

New-Item `
    -ItemType Directory `
    -Path $Stage `
    -Force | Out-Null

Get-ChildItem $Root |
    Where-Object {
        $_.FullName -ne $FinalZip
    } |
    ForEach-Object {
        Copy-Item `
            $_.FullName `
            $Stage `
            -Recurse `
            -Force
    }

Compress-Archive `
    -Path "$Stage\*" `
    -DestinationPath $FinalZip `
    -CompressionLevel Optimal

Remove-Item `
    $Stage `
    -Recurse `
    -Force

$ZipHash = (
    Get-FileHash `
        $FinalZip `
        -Algorithm SHA256
).Hash.ToLower()

$ZipHash |
    Set-Content `
        "$Root\00_MANIFEST\FINAL_ZIP_SHA256.txt" `
        -Encoding ASCII

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "F1-DEV R4 COMPLETE - ONE FOLDER PACKAGE" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

Write-Host ""
Write-Host "FOLDER:"
Write-Host $Root

Write-Host ""
Write-Host "ZIP:"
Write-Host $FinalZip

Write-Host ""
Write-Host "ZIP SHA256:"
Write-Host $ZipHash

Write-Host ""
Write-Host "SUMMARY:"
Get-Content "$Root\00_MANIFEST\R4_SUMMARY.txt"
