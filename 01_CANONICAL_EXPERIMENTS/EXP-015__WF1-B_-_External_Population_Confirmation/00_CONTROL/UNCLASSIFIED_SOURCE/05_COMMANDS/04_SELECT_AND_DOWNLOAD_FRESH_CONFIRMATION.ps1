$ErrorActionPreference = "Stop"

$ROOT   = "C:\RSOS\RSOS_EXPERIMENTS_WF1_A_V12_R_CORRECTIVE_CAUSAL_TOPOLOGY\WF1_B_EXTERNAL_POPULATION_PARQUET"
$FRESH  = "$ROOT\02_DATASETS\RAW_PARQUET\FRESH_CONFIRMATION"
$LOG    = "$ROOT\06_LOGS"
$PROTO  = "$ROOT\01_PROTOCOL"

$PREREG = "3ca9ab8b46aa852d951259fbd5f60c9427c4fbb0c9e24b1afc7cea40be61e547"

$USED = @(
    "train-00007-of-00086.parquet",
    "train-00022-of-00086.parquet",
    "train-00041-of-00086.parquet",
    "train-00049-of-00086.parquet",
    "train-00051-of-00086.parquet",
    "train-00052-of-00086.parquet",
    "train-00054-of-00086.parquet",
    "train-00080-of-00086.parquet",
    "train-00085-of-00086.parquet"
)

New-Item -ItemType Directory -Path $FRESH -Force | Out-Null
New-Item -ItemType Directory -Path $LOG   -Force | Out-Null

# Preserve any artifact from the failed 00010 download attempt.
Get-ChildItem -LiteralPath $FRESH -File -ErrorAction SilentlyContinue |
    ForEach-Object {
        Move-Item $_.FullName `
            "$LOG\FAILED_PREVIOUS_DOWNLOAD_$($_.Name)" `
            -Force
    }

$api = "https://huggingface.co/api/datasets/allenai/WildChat-4.8M/tree/main/data?recursive=false&expand=false&limit=1000"

$repo = Invoke-RestMethod -Uri $api -Method Get

$available = @(
    $repo |
    Where-Object {
        $_.path -match '^data/train-\d{5}-of-\d{5}\.parquet$'
    } |
    ForEach-Object {
        Split-Path $_.path -Leaf
    } |
    Sort-Object -Unique
)

if ($available.Count -lt 9) {
    throw "Could not retrieve enough real parquet shards."
}

function Get-Rank([string]$name) {
    $text = "$PREREG|$name"
    $bytes = [Text.Encoding]::UTF8.GetBytes($text)
    $sha = [Security.Cryptography.SHA256]::Create()

    try {
        return (
            [BitConverter]::ToString(
                $sha.ComputeHash($bytes)
            )
        ).Replace("-","").ToLower()
    }
    finally {
        $sha.Dispose()
    }
}

$candidates = @(
    $available |
    Where-Object { $_ -notin $USED } |
    ForEach-Object {
        [PSCustomObject]@{
            Name = $_
            Rank = Get-Rank $_
        }
    } |
    Sort-Object Rank
)

$selected = @($candidates | Select-Object -First 9)

$selected |
    Export-Csv `
        "$PROTO\WF1_B_FRESH_CONFIRMATION_SELECTED_SHARDS.csv" `
        -NoTypeInformation `
        -Encoding UTF8

@"
WF1-B FRESH CONFIRMATION SELECTION CORRECTION

Previous attempted filename train-00010-of-00086.parquet returned HTTP 404.
No scientific outcome data were inspected.

Corrected rule:
1. Query actual repository file list.
2. Exclude all previously used F2 shards.
3. Rank every remaining filename by:
   SHA256(PREREGISTRATION_SHA256 + "|" + filename)
4. Select the first 9.
5. Freeze selection before scientific scoring.

Preregistration SHA256:
$PREREG
"@ | Set-Content `
    "$PROTO\WF1_B_FRESH_SELECTION_CORRECTION.txt" `
    -Encoding UTF8

foreach ($x in $selected) {

    $name = $x.Name
    $url  = "https://huggingface.co/datasets/allenai/WildChat-4.8M/resolve/main/data/$name?download=true"
    $dst  = Join-Path $FRESH $name

    Write-Host "DOWNLOADING: $name"

    curl.exe -L --fail --retry 3 --retry-delay 5 `
        -o "$dst" `
        "$url"

    if ($LASTEXITCODE -ne 0) {
        throw "DOWNLOAD FAILED: $name"
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host " FRESH CONFIRMATION SHARDS READY"
Write-Host "============================================================"

Get-ChildItem -LiteralPath $FRESH -Filter "*.parquet" |
    Sort-Object Name |
    ForEach-Object {
        [PSCustomObject]@{
            Name   = $_.Name
            SizeMB = [math]::Round($_.Length / 1MB,2)
            SHA256 = (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower()
        }
    } |
    Format-Table -AutoSize

Write-Host ""
Write-Host "COUNT: $((Get-ChildItem $FRESH -Filter '*.parquet').Count)"
Write-Host "NO SCIENTIFIC SCORING PERFORMED."
