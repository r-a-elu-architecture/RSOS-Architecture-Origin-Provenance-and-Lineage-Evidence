$ErrorActionPreference = "Stop"

$WF3 = "C:\RSOS\RSOS_EXPERIMENTS_WF3_GENEALOGY_POPULATION_CONFIRMATION"
$Control = Join-Path $WF3 "00_CONTROL"

$ScanRoot = "C:\RSOS"

$KnownAuthorities = @(
    [pscustomobject]@{
        Authority = "WF2"
        Path = "C:\RSOS\RSOS_EXPERIMENTS_WF2_PROSPECTIVE_CROSS_SYSTEM_CONFIRMATION"
    },
    [pscustomobject]@{
        Authority = "WF1-P0"
        Path = "C:\RSOS\RSOS_EXPERIMENTS_WF1_P0_HISTORICAL_CORPUS_BRIDGE"
    },
    [pscustomobject]@{
        Authority = "F2-BR0"
        Path = "C:\RSOS\Training data\F2_BR0_FULL_FINGERPRINT_RECOVERY"
    },
    [pscustomobject]@{
        Authority = "F2-BR2-PACKAGE"
        Path = "C:\RSOS\Training data\F2_BR2_DETERMINISTIC_CORPUS_FINGERPRINT_REFREEZE_COMPLETE.zip"
    },
    [pscustomobject]@{
        Authority = "F2-EXTERNAL"
        Path = "C:\RSOS\Training data\F2_B_FRESH_WILDCHAT"
    }
)

# ------------------------------------------------------------
# Known-root existence table
# ------------------------------------------------------------

$RootInventory = foreach ($k in $KnownAuthorities) {

    $exists = Test-Path -LiteralPath $k.Path

    $type = if ($exists) {
        if ((Get-Item -LiteralPath $k.Path).PSIsContainer) {
            "DIRECTORY"
        } else {
            "FILE"
        }
    } else {
        "NOT_FOUND"
    }

    [pscustomobject]@{
        AUTHORITY = $k.Authority
        PATH      = $k.Path
        EXISTS    = $exists
        TYPE      = $type
    }
}

$RootInventory |
    Export-Csv `
        -NoTypeInformation `
        -Encoding UTF8 `
        -Path (Join-Path $Control "WF3_PHASE0_KNOWN_ROOTS.csv")

# ============================================================
# 3. METADATA-ONLY DISCOVERY
#    No scientific result content is opened.
# ============================================================

$scopeRegex = '(?i)(F1|F2|V11|V12|WF1|WF2|genealog|lineage|chronolog|temporal|ancestor|descendant|same.?author|non.?lineage|pre.?lineage|RSOS|RSSO|RSIA|RSX|wildchat|external|population|rarity|near.?hit|surface.?free|operator|relational|topolog|graph|edge|feature|extractor|detector|model|joblib|dedup|exclusion|inclusion|selection|metadata|timestamp|matched|control|fingerprint)'

$allFiles = Get-ChildItem `
    -LiteralPath $ScanRoot `
    -Recurse `
    -File `
    -ErrorAction SilentlyContinue |
    Where-Object {
        $_.FullName -notlike "$WF3*" -and
        (
            $_.Name -match $scopeRegex -or
            $_.DirectoryName -match '(?i)(F1|F2|V11|V12|WF1|WF2|Training data)'
        )
    }

function Get-Authority {
    param([string]$Path)

    switch -Regex ($Path) {
        '(?i)WF2'                      { return "WF2" }
        '(?i)WF1[_\-]?P0'              { return "WF1-P0" }
        '(?i)WF1[_\-]?A'               { return "WF1-A" }
        '(?i)WF1[_\-]?B'               { return "WF1-B" }
        '(?i)V12'                       { return "V12" }
        '(?i)V11'                       { return "V11" }
        '(?i)F2'                        { return "F2" }
        '(?i)F1'                        { return "F1" }
        default                         { return "UNRESOLVED_UPSTREAM" }
    }
}

function Get-Categories {
    param(
        [string]$Name,
        [string]$Path,
        [string]$Extension
    )

    $s = "$Name $Path"
    $cats = New-Object System.Collections.Generic.List[string]

    if ($s -match '(?i)(genealog|lineage|chronolog|ancestor|descendant|early|later|historical|matched)') {
        $cats.Add("01_HISTORICAL_GENEALOGY_REPRESENTATION")
    }

    if ($s -match '(?i)(same.?author|non.?lineage)') {
        $cats.Add("02_SAME_AUTHOR_CONTROL")
    }

    if ($s -match '(?i)(pre.?lineage|pre.?rsos|prelineage)') {
        $cats.Add("03_PRE_LINEAGE")
    }

    if ($s -match '(?i)(graph|topolog|relational|operator|edge|causal.?map|V11|V12)') {
        $cats.Add("04_RELATIONAL_TOPOLOGICAL")
    }

    if ($s -match '(?i)(wildchat|external|population|rarity|near.?hit|F2|WF1.?B)') {
        $cats.Add("05_EXTERNAL_POPULATION")
    }

    if ($s -match '(?i)(WF2|V13|V14|V17|prospective|cross.?system)') {
        $cats.Add("06_WF2_CROSS_SYSTEM")
    }

    if ($s -match '(?i)(extractor|feature.?set|feature.?schema|vectorizer|pipeline|scaler)') {
        $cats.Add("07_FEATURE_EXTRACTOR")
    }

    if (
        $Extension -match '(?i)\.(joblib|pkl|pickle)$' -or
        $s -match '(?i)(detector|classifier|model)'
    ) {
        $cats.Add("08_DETECTOR_MODEL")
    }

    if ($s -match '(?i)(temporal|chronolog|timestamp|metadata|created.?at|date)') {
        $cats.Add("09_TEMPORAL_METADATA")
    }

    if ($s -match '(?i)(dedup|duplicate|exclusion|exclude|inclusion|selection|missing|error|failure)') {
        $cats.Add("10_EXCLUSION_DEDUP")
    }

    if (
        $Extension -match '(?i)\.(parquet|jsonl|ndjson)$' -or
        $s -match '(?i)(wildchat|population)'
    ) {
        $cats.Add("11_POPULATION_DATASET")
    }

    if ($cats.Count -eq 0) {
        $cats.Add("99_RELEVANT_UNCLASSIFIED")
    }

    return ($cats -join ";")
}

function Get-ReuseDecision {
    param(
        [string]$Name,
        [string]$Categories,
        [string]$Extension
    )

    if ($Categories -match '11_POPULATION_DATASET') {
        return "POPULATION_INPUT_CANDIDATE"
    }

    if (
        $Categories -match '07_FEATURE_EXTRACTOR' -or
        $Categories -match '08_DETECTOR_MODEL'
    ) {
        return "DIRECT_REUSE_CANDIDATE"
    }

    if (
        $Categories -match '01_HISTORICAL_GENEALOGY_REPRESENTATION' -or
        $Categories -match '02_SAME_AUTHOR_CONTROL' -or
        $Categories -match '03_PRE_LINEAGE' -or
        $Categories -match '04_RELATIONAL_TOPOLOGICAL' -or
        $Categories -match '09_TEMPORAL_METADATA' -or
        $Categories -match '10_EXCLUSION_DEDUP'
    ) {
        return "REUSE_OR_REFERENCE_CANDIDATE"
    }

    if ($Name -match '(?i)(result|summary|adjudication|finding|final)') {
        return "REFERENCE_ONLY"
    }

    return "REVIEW_REQUIRED"
}

$Inventory = foreach ($f in $allFiles) {

    $authority = Get-Authority -Path $f.FullName

    $categories = Get-Categories `
        -Name $f.Name `
        -Path $f.FullName `
        -Extension $f.Extension

    $decision = Get-ReuseDecision `
        -Name $f.Name `
        -Categories $categories `
        -Extension $f.Extension

    [pscustomobject]@{
        AUTHORITY              = $authority
        CATEGORIES             = $categories
        REUSE_DECISION         = $decision
        FILE_NAME              = $f.Name
        EXTENSION              = $f.Extension
        SIZE_BYTES             = $f.Length
        SIZE_MB                = [math]::Round($f.Length / 1MB, 3)
        LAST_WRITE_UTC         = $f.LastWriteTimeUtc.ToString("o")
        FULL_PATH              = $f.FullName
        PHASE0_CONTENT_OPENED  = $false
        UPSTREAM_MUTATED       = $false
    }
}

$Inventory = $Inventory |
    Sort-Object AUTHORITY,CATEGORIES,FULL_PATH

$Inventory |
    Export-Csv `
        -NoTypeInformation `
        -Encoding UTF8 `
        -Path (Join-Path $Control "WF3_PHASE0_REUSE_INVENTORY.csv")

# ============================================================
# 4. TARGETED KNOWN FROZEN ASSET SEARCH
# ============================================================

$ImportantPatterns = @(
    "FROZEN_FEATURE_SETS.json",
    "F2_B_FROZEN_SPEC.json",
    "F2_BR2_FROZEN_SPEC.json",
    "*JOINT_MULTILAYER*",
    "*OPERATOR_RELATIONAL*",
    "*SURFACE_FREE*",
    "*R1_COMPATIBLE*",
    "*WINDOW8*",
    "*WINDOW12*",
    "*genealog*",
    "*lineage*",
    "*same*author*",
    "*non*lineage*",
    "*chronolog*",
    "*temporal*",
    "*topolog*",
    "*graph*",
    "*edge*",
    "*dedup*",
    "*exclusion*",
    "*wildchat*"
)

$Important = foreach ($p in $ImportantPatterns) {

    Get-ChildItem `
        -LiteralPath $ScanRoot `
        -Recurse `
        -File `
        -Filter $p `
        -ErrorAction SilentlyContinue |
        Where-Object {
            $_.FullName -notlike "$WF3*"
        } |
        ForEach-Object {
            [pscustomobject]@{
                SEARCH_PATTERN = $p
                FILE_NAME      = $_.Name
                SIZE_MB        = [math]::Round($_.Length / 1MB,3)
                FULL_PATH      = $_.FullName
            }
        }
}

$Important |
    Sort-Object SEARCH_PATTERN,FULL_PATH -Unique |
    Export-Csv `
        -NoTypeInformation `
        -Encoding UTF8 `
        -Path (Join-Path $Control "WF3_PHASE0_IMPORTANT_ASSETS.csv")

# ============================================================
# 5. EXTERNAL POPULATION CANDIDATES
# ============================================================

$PopulationCandidates = $Inventory |
    Where-Object {
        $_.CATEGORIES -match "11_POPULATION_DATASET" -or
        $_.CATEGORIES -match "05_EXTERNAL_POPULATION"
    } |
    Select-Object `
        AUTHORITY,
        FILE_NAME,
        EXTENSION,
        SIZE_BYTES,
        SIZE_MB,
        LAST_WRITE_UTC,
        FULL_PATH,
        REUSE_DECISION

$PopulationCandidates |
    Export-Csv `
        -NoTypeInformation `
        -Encoding UTF8 `
        -Path (Join-Path $Control "WF3_PHASE0_POPULATION_CANDIDATES.csv")

# ============================================================
# 6. CATEGORY COVERAGE
# ============================================================

$Required = @(
    "01_HISTORICAL_GENEALOGY_REPRESENTATION",
    "02_SAME_AUTHOR_CONTROL",
    "03_PRE_LINEAGE",
    "04_RELATIONAL_TOPOLOGICAL",
    "05_EXTERNAL_POPULATION",
    "06_WF2_CROSS_SYSTEM",
    "07_FEATURE_EXTRACTOR",
    "08_DETECTOR_MODEL",
    "09_TEMPORAL_METADATA",
    "10_EXCLUSION_DEDUP",
    "11_POPULATION_DATASET"
)

$Coverage = foreach ($r in $Required) {

    $hits = @(
        $Inventory |
        Where-Object {
            $_.CATEGORIES -match [regex]::Escape($r)
        }
    )

    [pscustomobject]@{
        CATEGORY = $r
        FILE_COUNT = $hits.Count
        STATUS = if ($hits.Count -gt 0) {
            "IDENTIFIED"
        } else {
            "NOT_IDENTIFIED_BY_METADATA_SCAN"
        }
    }
}

$Coverage |
    Export-Csv `
        -NoTypeInformation `
        -Encoding UTF8 `
        -Path (Join-Path $Control "WF3_PHASE0_CATEGORY_COVERAGE.csv")

# ============================================================
# 7. PHASE-0 SUMMARY
# ============================================================

$Summary = [ordered]@{
    experiment = "WF3 — Genealogy & Population Confirmation"
    modules = @(
        "V15 — Genealogy and Temporal Directionality",
        "V16 — Population-Scale External Confirmation"
    )
    phase = "PHASE 0 — FROZEN INPUT / REUSE INVENTORY"
    scientific_outcomes_inspected = $false
    upstream_mutation_performed = $false
    wf3_root = $WF3
    total_candidate_files = @($Inventory).Count
    population_candidate_files = @($PopulationCandidates).Count
    known_roots = $RootInventory
    category_coverage = $Coverage
    governing_rule = "Metadata-only inventory. No V15/V16 scientific observation."
    next_boundary = "BOUNDARY 1 — PREREGISTRATION"
}

$Summary |
    ConvertTo-Json -Depth 8 |
    Set-Content `
        -Encoding UTF8 `
        -Path (Join-Path $Control "WF3_PHASE0_SUMMARY.json")

# ============================================================
# 8. MACHINE-READABLE PHASE STATUS
# ============================================================

$Status = [ordered]@{
    WF3 = "IN_PROGRESS"
    phase = "PHASE_0"
    V15 = "NOT_EXECUTED"
    V16 = "NOT_EXECUTED"
    boundary_1 = "NOT_FROZEN"
    boundary_2 = "NOT_FROZEN"
    boundary_3 = "NOT_FROZEN"
}

$Status |
    ConvertTo-Json -Depth 4 |
    Set-Content `
        -Encoding UTF8 `
        -Path (Join-Path $Control "WF3_STATUS.json")

Write-Host ""
Write-Host "============================================================"
Write-Host " WF3 PHASE 0 COMPLETE"
Write-Host "============================================================"
Write-Host ""
Write-Host "Scientific outcomes inspected : NO"
Write-Host "Upstream files modified        : NO"
Write-Host "Candidate files found          : $(@($Inventory).Count)"
Write-Host "Population candidates found    : $(@($PopulationCandidates).Count)"
Write-Host ""
Write-Host "CATEGORY COVERAGE:"
$Coverage | Format-Table -AutoSize
Write-Host ""
Write-Host "OUTPUTS:"
Write-Host "  $Control\WF3_PHASE0_KNOWN_ROOTS.csv"
Write-Host "  $Control\WF3_PHASE0_REUSE_INVENTORY.csv"
Write-Host "  $Control\WF3_PHASE0_IMPORTANT_ASSETS.csv"
Write-Host "  $Control\WF3_PHASE0_POPULATION_CANDIDATES.csv"
Write-Host "  $Control\WF3_PHASE0_CATEGORY_COVERAGE.csv"
Write-Host "  $Control\WF3_PHASE0_SUMMARY.json"
Write-Host "  $Control\WF3_STATUS.json"
Write-Host ""
Write-Host "NEXT: BOUNDARY 1 — PREREGISTRATION"
Write-Host "============================================================"
