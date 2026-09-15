$ErrorActionPreference = "Stop"

$WF3     = "C:\RSOS\RSOS_EXPERIMENTS_WF3_GENEALOGY_POPULATION_CONFIRMATION"
$ROOT    = "C:\RSOS"

$Control = Join-Path $WF3 "00_CONTROL"
$Protocol= Join-Path $WF3 "01_PROTOCOL"
$Inputs  = Join-Path $WF3 "02_INPUTS"
$Runner  = Join-Path $WF3 "03_RUNNER"
$Diag    = Join-Path $WF3 "06_DIAGNOSTIC"
$Freeze  = Join-Path $WF3 "09_FREEZE"

foreach ($d in @($Control,$Protocol,$Inputs,$Runner,$Diag,$Freeze)) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

$P0 = Join-Path $Control "WF3_PHASE0_REUSE_INVENTORY.csv"

if (!(Test-Path $P0)) {
    throw "Phase-0 inventory missing: $P0"
}

$inv = Import-Csv $P0

Write-Host ""
Write-Host "============================================================"
Write-Host " WF3 BOUNDARY 1 — INPUT RESOLUTION"
Write-Host "============================================================"

function Hits {
    param([string]$Pattern)
    @(
        $inv | Where-Object {
            $_.FILE_NAME -match $Pattern -or
            $_.FULL_PATH -match $Pattern
        }
    )
}

$lineage    = Hits '(?i)(genealog|lineage|ancestor|descendant|chronolog)'
$sameAuthor = Hits '(?i)(same.?author|non.?lineage|nonlineage|author.?control)'
$preLineage = Hits '(?i)(pre.?lineage|pre.?rsos|before.?rsos|prelineage)'
$topology   = Hits '(?i)(V11|V12|topolog|relational|operator|edge.?causal|graph.?vector)'
$features   = Hits '(?i)(FROZEN_FEATURE_SETS|feature.?set|feature.?schema|extractor|vectorizer)'
$temporal   = Hits '(?i)(timestamp|temporal|chronolog|metadata|created.?at)'

$models = @(
    $inv | Where-Object {
        $_.EXTENSION -match '(?i)\.(joblib|pkl|pickle)$' -or
        $_.FILE_NAME -match '(?i)(JOINT_MULTILAYER|OPERATOR_RELATIONAL|SURFACE_FREE|R1_COMPATIBLE|WINDOW8|WINDOW12|detector|classifier)'
    }
)

$population = @(
    $inv | Where-Object {
        $_.FILE_NAME -match '(?i)^train-\d+-of-\d+\.parquet$' -or
        $_.FILE_NAME -match '(?i)^wildchat.*\.(parquet|jsonl|ndjson|json)$'
    }
)

# ZIP entry names only; no scientific outcome extraction.
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipRows = New-Object System.Collections.Generic.List[object]

$zips = Get-ChildItem $ROOT -Recurse -File -Filter "*.zip" -ErrorAction SilentlyContinue |
    Where-Object {
        $_.FullName -notlike "$WF3*" -and
        $_.FullName -match '(?i)(F1|F2|V11|V12|WF1|WF2|Training data)'
    }

foreach ($zfile in $zips) {
    try {
        $z = [System.IO.Compression.ZipFile]::OpenRead($zfile.FullName)
        foreach ($e in $z.Entries) {
            if ($e.Name) {
                $zipRows.Add([pscustomobject]@{
                    ARCHIVE = $zfile.FullName
                    ENTRY   = $e.FullName
                    SIZE    = $e.Length
                })
            }
        }
        $z.Dispose()
    } catch {
        Write-Warning "Unreadable ZIP: $($zfile.FullName)"
    }
}

$sameAuthorZip = @(
    $zipRows | Where-Object {
        $_.ENTRY -match '(?i)(same.?author|non.?lineage|nonlineage|author.?control)'
    }
)
$preLineageZip = @(
    $zipRows | Where-Object {
        $_.ENTRY -match '(?i)(pre.?lineage|pre.?rsos|before.?rsos|prelineage)'
    }
)
$lineageZip = @(
    $zipRows | Where-Object {
        $_.ENTRY -match '(?i)(genealog|lineage|ancestor|descendant|chronolog)'
    }
)
$topologyZip = @(
    $zipRows | Where-Object {
        $_.ENTRY -match '(?i)(V11|V12|topolog|relational|operator|edge.?causal|graph.?vector)'
    }
)

# Search only small upstream textual metadata/specification files for control labels.
$contentHits = New-Object System.Collections.Generic.List[object]

$scanFiles = Get-ChildItem $ROOT -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object {
        $_.FullName -notlike "$WF3*" -and
        $_.FullName -match '(?i)(F1|F2|WF1|WF2|V11|V12)' -and
        $_.Extension -match '(?i)\.(csv|json|txt|md|yaml|yml)$' -and
        $_.Length -le 20MB
    }

foreach ($f in $scanFiles) {
    try {
        $m = Select-String `
            -LiteralPath $f.FullName `
            -Pattern '(?i)(same.?author|non.?lineage|nonlineage|author.?control|pre.?lineage|pre.?rsos)' `
            -List `
            -ErrorAction SilentlyContinue

        if ($m) {
            $contentHits.Add([pscustomobject]@{
                FILE = $f.FullName
                SIZE_BYTES = $f.Length
            })
        }
    } catch {}
}

$contentSameAuthor = @(
    $contentHits | Where-Object {
        try {
            Select-String `
                -LiteralPath $_.FILE `
                -Pattern '(?i)(same.?author|non.?lineage|nonlineage|author.?control)' `
                -Quiet `
                -ErrorAction SilentlyContinue
        } catch { $false }
    }
)

$contentPreLineage = @(
    $contentHits | Where-Object {
        try {
            Select-String `
                -LiteralPath $_.FILE `
                -Pattern '(?i)(pre.?lineage|pre.?rsos)' `
                -Quiet `
                -ErrorAction SilentlyContinue
        } catch { $false }
    }
)

$roles = @(
    [pscustomobject]@{ROLE="HISTORICAL_LINEAGE";       DIRECT=$lineage.Count;    ZIP=$lineageZip.Count;    CONTENT=0;                       REQUIRED="YES"},
    [pscustomobject]@{ROLE="SAME_AUTHOR_NON_LINEAGE";  DIRECT=$sameAuthor.Count; ZIP=$sameAuthorZip.Count; CONTENT=$contentSameAuthor.Count; REQUIRED="YES"},
    [pscustomobject]@{ROLE="PRE_LINEAGE";              DIRECT=$preLineage.Count; ZIP=$preLineageZip.Count; CONTENT=$contentPreLineage.Count; REQUIRED="WHERE_VALID"},
    [pscustomobject]@{ROLE="RELATIONAL_TOPOLOGICAL";   DIRECT=$topology.Count;   ZIP=$topologyZip.Count;   CONTENT=0;                       REQUIRED="YES"},
    [pscustomobject]@{ROLE="FROZEN_FEATURES";          DIRECT=$features.Count;   ZIP=0;                    CONTENT=0;                       REQUIRED="YES"},
    [pscustomobject]@{ROLE="FROZEN_DETECTORS";         DIRECT=$models.Count;     ZIP=0;                    CONTENT=0;                       REQUIRED="YES"},
    [pscustomobject]@{ROLE="TEMPORAL_METADATA";        DIRECT=$temporal.Count;   ZIP=0;                    CONTENT=0;                       REQUIRED="YES"},
    [pscustomobject]@{ROLE="EXTERNAL_POPULATION";      DIRECT=$population.Count; ZIP=0;                    CONTENT=0;                       REQUIRED="YES"}
)

$RolePath = Join-Path $Control "WF3_BOUNDARY1_ROLE_RESOLUTION.csv"
$roles | Export-Csv -NoTypeInformation -Encoding UTF8 $RolePath

$sameAuthor |
    Select-Object FILE_NAME,FULL_PATH,SIZE_BYTES |
    Export-Csv -NoTypeInformation -Encoding UTF8 `
    (Join-Path $Control "WF3_SAME_AUTHOR_DIRECT_CANDIDATES.csv")

$sameAuthorZip |
    Export-Csv -NoTypeInformation -Encoding UTF8 `
    (Join-Path $Control "WF3_SAME_AUTHOR_ZIP_CANDIDATES.csv")

$contentSameAuthor |
    Export-Csv -NoTypeInformation -Encoding UTF8 `
    (Join-Path $Control "WF3_SAME_AUTHOR_CONTENT_REFERENCES.csv")

$preLineage |
    Select-Object FILE_NAME,FULL_PATH,SIZE_BYTES |
    Export-Csv -NoTypeInformation -Encoding UTF8 `
    (Join-Path $Control "WF3_PRELINEAGE_DIRECT_CANDIDATES.csv")

$preLineageZip |
    Export-Csv -NoTypeInformation -Encoding UTF8 `
    (Join-Path $Control "WF3_PRELINEAGE_ZIP_CANDIDATES.csv")

$PopulationManifestPath = Join-Path $Inputs "WF3_EXTERNAL_POPULATION_FILE_MANIFEST.csv"
$population |
    Select-Object FILE_NAME,FULL_PATH,SIZE_BYTES |
    Sort-Object FULL_PATH -Unique |
    Export-Csv -NoTypeInformation -Encoding UTF8 $PopulationManifestPath

$resolved = [ordered]@{
    historical_lineage      = (($lineage.Count + $lineageZip.Count) -gt 0)
    same_author_non_lineage = (($sameAuthor.Count + $sameAuthorZip.Count + $contentSameAuthor.Count) -gt 0)
    relational_topological  = (($topology.Count + $topologyZip.Count) -gt 0)
    frozen_features         = ($features.Count -gt 0)
    frozen_detectors        = ($models.Count -gt 0)
    temporal_metadata       = ($temporal.Count -gt 0)
    external_population     = ($population.Count -gt 0)
}

$blocking = @(
    $resolved.GetEnumerator() |
    Where-Object { !$_.Value } |
    ForEach-Object { $_.Key }
)

if ($blocking.Count -gt 0) {
    $blocked = [ordered]@{
        experiment = "WF3"
        boundary = "BOUNDARY 1"
        status = "BLOCKED_BEFORE_FREEZE"
        unresolved_mandatory_roles = $blocking
        phase0 = "COMPLETE"
        scientific_outcomes_inspected = $false
        upstream_modified = $false
        timestamp_utc = (Get-Date).ToUniversalTime().ToString("o")
    }

    $blocked |
        ConvertTo-Json -Depth 10 |
        Set-Content -Encoding UTF8 `
        (Join-Path $Diag "WF3_BOUNDARY1_BLOCKED.json")

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " WF3 BOUNDARY 1 — BLOCKED BEFORE FREEZE"
    Write-Host "============================================================"
    Write-Host "No scientific outcomes inspected."
    Write-Host "No upstream files modified."
    Write-Host ""
    Write-Host "UNRESOLVED:"
    foreach ($b in $blocking) { Write-Host "  - $b" }
    Write-Host ""
    $roles | Format-Table -AutoSize
    Write-Host ""
    Write-Host "Boundary 1 remains NOT FROZEN."
    Write-Host "============================================================"
    exit 20
}

# Preregistration.
$pre = [ordered]@{
    experiment = "WF3 — Genealogy & Population Confirmation"

    modules = [ordered]@{
        V15 = "Genealogy and Temporal Directionality"
        V16 = "Population-Scale External Confirmation"
    }

    immutable_upstream = @(
        "F1","F2","V11","V12","WF1-P0","WF1-A","WF1-B",
        "WF2-V13","WF2-V14","WF2-V17",
        "WF2 prospective active causal/topology replication"
    )

    representation_rule =
        "Reuse valid frozen upstream representations, extractors and detectors. No outcome-driven retraining or representation repair."

    V15 = [ordered]@{
        primary_question =
            "Does true historical chronology contain held-out developmental direction beyond authorship, vocabulary/topic, model era, length/complexity, temporal proximity, generic recursion and arbitrary ordering?"

        stage_model = @(
            "Pre-RSOS","Early RSOS","Mature RSOS","RSSO","RSIA","RSX"
        )

        branching_rule =
            "RSIA and RSX are not forced into a strict linear tree. Branching, convergence, mixed ancestry and unresolved relations remain admissible."

        split_rule =
            "Discovery versus confirmation assignment is deterministic from SHA256(unit_id + '|WF3_V15_SPLIT_2026')."

        primary_endpoint =
            "V15_PRIMARY_DIRECTIONAL_GATE"

        primary_components = @(
            "Feature-direction signs are learned only in the discovery split.",
            "Using those frozen signs, confirmation chronology must show positive nuisance-adjusted developmental score.",
            "True chronology must exceed its 10,000 within-model-era shuffled-history null at alpha <= 0.01.",
            "True chronology must outperform explicit later-to-early reversal using the same discovery-frozen directions.",
            "True chronology must exceed the 95th-percentile same-author non-lineage bootstrap control."
        )

        failure_rule =
            "Failure of any mandatory primary component means the preregistered genealogy primary endpoint does not pass."

        primary_alpha = 0.01

        nuisance_controls = @(
            "authorship","topic/vocabulary","model era","conversation length",
            "conversation complexity","temporal proximity",
            "generic recursive structure where frozen coordinates exist"
        )

        secondary_endpoints = @(
            "ancestor-to-descendant predictive advantage",
            "early-to-late generalization asymmetry",
            "directional graph compatibility",
            "conserved-versus-derived structure",
            "surface-free developmental direction",
            "operator/relational developmental direction",
            "RSSO-to-RSIA branch analysis",
            "RSSO-to-RSX branch analysis"
        )

        prohibited_inferences = @(
            "chronological order alone equals causality",
            "similarity alone equals ancestry",
            "classifier AUC alone establishes genealogy"
        )
    }

    V16 = [ordered]@{
        primary_question =
            "How prevalent are progressively richer frozen structural representations in the largest eligible independent external population available at Boundary 1?"

        inference_unit =
            "USER if stable user identifier covers >=95% of eligible records; otherwise SESSION if stable session identifier covers >=95%; otherwise CONVERSATION."

        repeated_unit_rule =
            "When multiple conversations belong to one primary population unit, classify that unit using the single eligible conversation with the smallest frozen joint distance to the lineage manifold. This is conservative against the rarity claim and prevents pseudoreplication."

        overlap_rule =
            "Observations identifiable as having been used for upstream F2/WF1-B evaluation are excluded from primary V16 confirmation and may appear only in separately labelled historical replication."

        deduplication = @(
            "collapse stable duplicate IDs",
            "collapse exact normalized-content SHA256 duplicates",
            "collapse duplicate records repeated across shards",
            "retain exclusion accounting",
            "technical failures are not scientific negatives"
        )

        compatibility_orientation =
            "All standardized compatibility coordinates must be oriented so higher means more lineage-compatible. Orientation is inherited from frozen upstream definitions and may not be flipped after external outcomes are inspected."

        levels = @(
            "PASSIVE_COORDINATE_NEAR_HIT",
            "MULTIDIMENSIONAL_LINEAGE_COMPATIBLE_MATCH",
            "FULL_ARCHITECTURE_LEVEL_MATCH"
        )

        component_rule =
            "Reuse compatible frozen upstream thresholds. If a component has no compatible frozen threshold, derive its acceptance boundary exclusively from frozen lineage references using the leave-one-out lower 5th-percentile lineage compatibility boundary before external prevalence is measured."

        joint_rule = [ordered]@{
            components = @(
                "passive compatibility",
                "surface-free compatibility",
                "operator/relational compatibility",
                "genealogy compatibility",
                "topology compatibility"
            )
            standardization =
                "Robust-standardize components against frozen lineage references using median and MAD."
            distance =
                "Euclidean distance in robust-standardized joint component space."
            k =
                "max(3, ceil(sqrt(N_lineage))), capped at N_lineage-1."
            lineage_boundary =
                "Empirical 95th percentile of leave-one-out lineage kNN distances."
            full_match =
                "All mandatory component boundaries pass AND external joint distance is no greater than the frozen lineage boundary."
            independence =
                "Marginal rarity estimates must never be multiplied as though coordinates were independent."
        }

        near_hit_rule =
            "Preserve every multidimensional/full match and the strongest passive strict-tail near-hits, including component scores, source identifier, nearest lineage neighbour and inclusion/exclusion reason."

        zero_match_rule =
            "If zero primary-population observations cross the preregistered full architecture boundary, report the exact 95% binomial upper bound and population resolution only; do not claim global uniqueness."

        forbidden_claims = @(
            "passive near-hit = RSOS",
            "rarity = literal uniqueness",
            "sampled population = global population"
        )
    }

    cross_module_rule =
        "V15 and V16 adjudicate independently. Neither can rescue failure of the other."

    created_utc = (Get-Date).ToUniversalTime().ToString("o")
}

$PrePath = Join-Path $Protocol "WF3_BOUNDARY1_PREREGISTRATION.json"
$pre | ConvertTo-Json -Depth 20 | Set-Content -Encoding UTF8 $PrePath

# Freeze input/reference manifest.
$manifest = New-Object System.Collections.Generic.List[object]

function AddDirect {
    param($role,$items)
    foreach ($x in $items) {
        if ($x.FULL_PATH) {
            $manifest.Add([pscustomobject]@{
                ROLE=$role; TYPE="FILE_REFERENCE"; PATH=$x.FULL_PATH; ARCHIVE=""; ENTRY=""
            })
        }
    }
}

AddDirect "HISTORICAL_LINEAGE" $lineage
AddDirect "SAME_AUTHOR_NON_LINEAGE" $sameAuthor
AddDirect "PRE_LINEAGE" $preLineage
AddDirect "RELATIONAL_TOPOLOGICAL" $topology
AddDirect "FROZEN_FEATURES" $features
AddDirect "FROZEN_DETECTORS" $models
AddDirect "TEMPORAL_METADATA" $temporal
AddDirect "EXTERNAL_POPULATION" $population

foreach ($x in $lineageZip) {
    $manifest.Add([pscustomobject]@{
        ROLE="HISTORICAL_LINEAGE"; TYPE="ZIP_ENTRY_REFERENCE"; PATH=""; ARCHIVE=$x.ARCHIVE; ENTRY=$x.ENTRY
    })
}
foreach ($x in $sameAuthorZip) {
    $manifest.Add([pscustomobject]@{
        ROLE="SAME_AUTHOR_NON_LINEAGE"; TYPE="ZIP_ENTRY_REFERENCE"; PATH=""; ARCHIVE=$x.ARCHIVE; ENTRY=$x.ENTRY
    })
}
foreach ($x in $preLineageZip) {
    $manifest.Add([pscustomobject]@{
        ROLE="PRE_LINEAGE"; TYPE="ZIP_ENTRY_REFERENCE"; PATH=""; ARCHIVE=$x.ARCHIVE; ENTRY=$x.ENTRY
    })
}
foreach ($x in $topologyZip) {
    $manifest.Add([pscustomobject]@{
        ROLE="RELATIONAL_TOPOLOGICAL"; TYPE="ZIP_ENTRY_REFERENCE"; PATH=""; ARCHIVE=$x.ARCHIVE; ENTRY=$x.ENTRY
    })
}
foreach ($x in $contentSameAuthor) {
    $manifest.Add([pscustomobject]@{
        ROLE="SAME_AUTHOR_NON_LINEAGE_REFERENCE"; TYPE="CONTENT_LOCATOR"; PATH=$x.FILE; ARCHIVE=""; ENTRY=""
    })
}

$ManifestPath = Join-Path $Inputs "WF3_BOUNDARY1_INPUT_REFERENCE_MANIFEST.csv"

$manifest |
    Sort-Object ROLE,TYPE,PATH,ARCHIVE,ENTRY -Unique |
    Export-Csv -NoTypeInformation -Encoding UTF8 $ManifestPath

$ContractPath = Join-Path $Protocol "WF3_BOUNDARY1_RUNNER_CONTRACT.json"
$contract = [ordered]@{
    V15_required_columns = @(
        "unit_id","stage_rank","is_lineage","is_same_author_control"
    )
    V15_optional_nuisance_columns = @(
        "model_era","topic_id","length","complexity","generic_recursion"
    )
    V16_required_compatibility_columns = @(
        "passive_compat","surface_free_compat","operator_relational_compat",
        "genealogy_compat","topology_compat"
    )
    V16_population_identifier_precedence = @(
        "user_id if >=95% coverage",
        "session_id if >=95% coverage",
        "conversation_id",
        "source_id"
    )
    V16_overlap_marker = "upstream_overlap=1 is excluded from primary confirmation"
    allowed_standardization = @(
        "deterministic column rename",
        "deterministic ID join",
        "deterministic chronology-to-stage mapping",
        "orientation inherited from frozen upstream definitions",
        "deduplication according to preregistration"
    )
    forbidden = @(
        "detector retraining",
        "post-outcome direction flipping",
        "post-outcome threshold tuning",
        "post-outcome genealogy redefinition",
        "discarding inconvenient external near-hits"
    )
}
$contract | ConvertTo-Json -Depth 12 | Set-Content -Encoding UTF8 $ContractPath

# Freeze the Boundary-1 runner itself as the executable artifact used to make this freeze.
$SelfPath = $MyInvocation.MyCommand.Path

$hashTargets = @(
    $PrePath,
    $SelfPath,
    $ContractPath,
    $ManifestPath,
    $RolePath,
    $PopulationManifestPath
)

$HashPath = Join-Path $Freeze "WF3_BOUNDARY1_CONTROL_SHA256.csv"

$hashes = foreach ($f in $hashTargets) {
    if (Test-Path $f) {
        $h = Get-FileHash -Algorithm SHA256 $f
        [pscustomobject]@{
            FILE=$f
            SHA256=$h.Hash
        }
    }
}

$hashes | Export-Csv -NoTypeInformation -Encoding UTF8 $HashPath

$freezeRecord = [ordered]@{
    experiment = "WF3 — Genealogy & Population Confirmation"
    boundary = "BOUNDARY 1 — PREREGISTRATION"
    status = "FROZEN"
    phase0 = "COMPLETE"
    V15 = "PREREGISTERED_NOT_EXECUTED"
    V16 = "PREREGISTERED_NOT_EXECUTED"
    scientific_outcomes_inspected = $false
    upstream_modified = $false
    primary_V15_endpoint = "V15_PRIMARY_DIRECTIONAL_GATE"
    primary_V16_rule = "Progressive compatibility classes plus frozen empirical joint lineage kNN boundary"
    same_author_control_resolved = $true
    pre_lineage_status = if (($preLineage.Count + $preLineageZip.Count + $contentPreLineage.Count) -gt 0) {
        "IDENTIFIED"
    } else {
        "NOT_IDENTIFIED_OPTIONAL_WHERE_VALID"
    }
    preregistration = $PrePath
    runner = $SelfPath
    runner_contract = $ContractPath
    input_manifest = $ManifestPath
    control_hash_manifest = $HashPath
    frozen_utc = (Get-Date).ToUniversalTime().ToString("o")
}

$FreezePath = Join-Path $Freeze "WF3_BOUNDARY1_FREEZE_RECORD.json"
$freezeRecord | ConvertTo-Json -Depth 12 | Set-Content -Encoding UTF8 $FreezePath

$freezeHash = Get-FileHash -Algorithm SHA256 $FreezePath
[pscustomobject]@{
    FILE=$FreezePath
    SHA256=$freezeHash.Hash
} | Export-Csv -NoTypeInformation -Encoding UTF8 `
    (Join-Path $Freeze "WF3_BOUNDARY1_FREEZE_RECORD_SHA256.csv")

$status = [ordered]@{
    WF3="IN_PROGRESS"
    phase_0="DONE"
    phase="BOUNDARY_1_FROZEN"
    V15="PREREGISTERED_NOT_EXECUTED"
    V16="PREREGISTERED_NOT_EXECUTED"
    boundary_1="FROZEN"
    boundary_2="NOT_FROZEN"
    boundary_3="NOT_FROZEN"
}

$status | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 `
    (Join-Path $Control "WF3_STATUS.json")

Write-Host ""
Write-Host "============================================================"
Write-Host " WF3 BOUNDARY 1 — FROZEN"
Write-Host "============================================================"
Write-Host "Scientific outcomes inspected : NO"
Write-Host "Upstream files modified        : NO"
Write-Host ""
Write-Host "ROLE RESOLUTION:"
$roles | Format-Table -AutoSize
Write-Host ""
Write-Host "V15 primary endpoint:"
Write-Host "  V15_PRIMARY_DIRECTIONAL_GATE"
Write-Host ""
Write-Host "V16 primary architecture rule:"
Write-Host "  Progressive components + empirical joint lineage boundary"
Write-Host ""
Write-Host "Freeze record:"
Write-Host "  $FreezePath"
Write-Host ""
Write-Host "NEXT:"
Write-Host "  WF3 EXECUTION PREPARATION — STANDARDIZE FROZEN INPUTS"
Write-Host "============================================================"
