import json
import math
import re
from pathlib import Path

import numpy as np

from WF3_R2_COMMON import (
    read_csv_rows,
    write_csv_rows,
    verify_reference_rows,
    named_numeric_from_file,
)

ROOT = Path(
    r"C:\RSOS\RSOS_EXPERIMENTS_WF3_GENEALOGY_POPULATION_CONFIRMATION_R2"
)

MANIFEST = (
    ROOT
    / "02_INPUTS"
    / "POPULATION_EXTERNAL"
    / "WF3_V16_CANONICAL_REFERENCE_MANIFEST.csv"
)

OUT = (
    ROOT
    / "05_OUTPUTS"
    / "V16_POPULATION_SCALE_EXTERNAL_CONFIRMATION"
)

DIAG = (
    ROOT
    / "06_DIAGNOSTIC"
    / "V16_POPULATION_SCALE_EXTERNAL_CONFIRMATION"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

DIAG.mkdir(
    parents=True,
    exist_ok=True,
)

PRIMARY_AUTHORITIES = [
    "EXP-011",
    "EXP-012",
    "EXP-015",
    "EXP-016",
]

HISTORICAL_NONFINAL = [
    "RUN-009",
    "RUN-010",
]

ALLOWED_BUCKETS = {
    "05_OUTPUTS",
    "07_ADJUDICATION",
    "09_FREEZE",
    "02_INPUTS",
}

METRIC_EXT = {
    ".csv",
    ".json",
    ".txt",
}

FILE_PATTERN = re.compile(
    r"(external|rarity|percentile|quantile|tail|auc|matched|"
    r"population|primary|summary|provider|cem|score|result)",
    re.I,
)

AUC_KEY = re.compile(
    r"(auc|roc_auc)",
    re.I,
)

AUC_CONTEXT = re.compile(
    r"(primary|external|matched|holdout|joint|surface|operator|"
    r"provider|result|summary)",
    re.I,
)

AUC_EXCLUDE = re.compile(
    r"(null|negative_control|diagnostic|debug|calibration_only)",
    re.I,
)

RARITY_KEY = re.compile(
    r"(rarity|tail|one[_ ]?in|frequency|prevalence)",
    re.I,
)

ONE_IN_TEXT = re.compile(
    r"\b1\s+in\s+([0-9][0-9,]*(?:\.[0-9]+)?)\b",
    re.I,
)


rows = verify_reference_rows(
    read_csv_rows(
        MANIFEST
    )
)


selection = []
metric_rows = []
parse_diag = []
rarity_rows = []
external_artifacts = []


for row in rows:
    p = Path(
        row[
            "CANONICAL_FILE"
        ]
    )

    entity = row[
        "ENTITY_ID"
    ]

    # External population artifact audit.
    if (
        p.suffix.lower()
        == ".parquet"
        and re.search(
            r"(wildchat|external)",
            str(p),
            re.I,
        )
    ):
        external_artifacts.append({
            "ENTITY_ID":
                entity,
            "CANONICAL_FILE":
                str(p),
            "SIZE_BYTES":
                p.stat().st_size,
            "SHA256":
                row[
                    "SHA256"
                ],
        })


    decision = ""

    if (
        row[
            "DESTINATION_BUCKET"
        ]
        not in ALLOWED_BUCKETS
    ):
        decision = "EXCLUDE_BUCKET"

    elif p.suffix.lower() not in METRIC_EXT:
        decision = "EXCLUDE_EXTENSION"

    elif p.stat().st_size > 52428800:
        decision = "EXCLUDE_SIZE"

    elif not FILE_PATTERN.search(
        p.name
    ):
        decision = "EXCLUDE_FILENAME"

    else:
        decision = "SELECT_METRIC_EVIDENCE"


    selection.append({
        **row,
        "V16_DECISION":
            decision,
    })


    if decision != "SELECT_METRIC_EVIDENCE":
        continue


    try:
        if p.suffix.lower() in {
            ".csv",
            ".json",
        }:
            values = named_numeric_from_file(
                p
            )

            for key, value in values:
                metric_rows.append({
                    "ENTITY_ID":
                        entity,
                    "CANONICAL_FILE":
                        str(p),
                    "METRIC_KEY":
                        key,
                    "VALUE":
                        float(value),
                })


        text = p.read_text(
            encoding="utf-8-sig",
            errors="replace",
        )


        for match in ONE_IN_TEXT.finditer(
            text
        ):
            value = float(
                match.group(1).replace(
                    ",",
                    "",
                )
            )

            rarity_rows.append({
                "ENTITY_ID":
                    entity,
                "CANONICAL_FILE":
                    str(p),
                "RARITY_TYPE":
                    "TEXT_ONE_IN_N",
                "KEY":
                    "1 in N",
                "VALUE":
                    value,
            })


        parse_diag.append({
            "ENTITY_ID":
                entity,
            "CANONICAL_FILE":
                str(p),
            "STATUS":
                "PARSED",
            "ERROR":
                "",
        })


    except Exception as exc:
        parse_diag.append({
            "ENTITY_ID":
                entity,
            "CANONICAL_FILE":
                str(p),
            "STATUS":
                "PARSE_ERROR",
            "ERROR":
                str(exc),
        })


write_csv_rows(
    OUT
    / "V16_FILE_SELECTION.csv",
    selection,
)

write_csv_rows(
    DIAG
    / "V16_PARSE_DIAGNOSTICS.csv",
    parse_diag,
)

write_csv_rows(
    OUT
    / "V16_METRIC_LEDGER.csv",
    metric_rows,
)

write_csv_rows(
    OUT
    / "V16_EXTERNAL_POPULATION_ARTIFACTS.csv",
    external_artifacts,
)


# Named numeric rarity metrics.
for row in metric_rows:
    if RARITY_KEY.search(
        row[
            "METRIC_KEY"
        ]
    ):
        rarity_rows.append({
            "ENTITY_ID":
                row[
                    "ENTITY_ID"
                ],
            "CANONICAL_FILE":
                row[
                    "CANONICAL_FILE"
                ],
            "RARITY_TYPE":
                "NAMED_NUMERIC",
            "KEY":
                row[
                    "METRIC_KEY"
                ],
            "VALUE":
                row[
                    "VALUE"
                ],
        })


write_csv_rows(
    OUT
    / "V16_RARITY_EVIDENCE.csv",
    rarity_rows,
)


# ============================================================
# PRIMARY AUTHORITY AUC EXTRACTION
# ============================================================

authority_values = {
    entity: []
    for entity in PRIMARY_AUTHORITIES
}


for row in metric_rows:
    entity = row[
        "ENTITY_ID"
    ]

    if entity not in authority_values:
        continue

    key = row[
        "METRIC_KEY"
    ]

    path = row[
        "CANONICAL_FILE"
    ]

    context = (
        path
        + " "
        + key
    )

    if not AUC_KEY.search(
        key
    ):
        continue

    if not AUC_CONTEXT.search(
        context
    ):
        continue

    if AUC_EXCLUDE.search(
        context
    ):
        continue

    value = float(
        row[
            "VALUE"
        ]
    )

    if (
        0.0
        <= value
        <= 1.0
    ):
        authority_values[
            entity
        ].append(
            value
        )


authority_rows = []

positive_count = 0
eligible_authorities = 0


for entity in PRIMARY_AUTHORITIES:
    values = authority_values[
        entity
    ]

    if values:
        median_auc = float(
            np.median(
                values
            )
        )

        eligible = True

        above = (
            median_auc
            > 0.5
        )

        eligible_authorities += 1

        if above:
            positive_count += 1

    else:
        median_auc = ""
        eligible = False
        above = ""


    authority_rows.append({
        "ENTITY_ID":
            entity,
        "ELIGIBLE_AUC_VALUES":
            len(values),
        "MEDIAN_AUC":
            median_auc,
        "ABOVE_CHANCE":
            above,
        "ELIGIBLE_AUTHORITY":
            eligible,
    })


write_csv_rows(
    OUT
    / "V16_AUTHORITY_AUC_SUMMARY.csv",
    authority_rows,
)


# ============================================================
# EXACT ONE-SIDED SIGN TEST
# ============================================================

sign_p = None

if eligible_authorities >= 1:
    numerator = 0.0

    for k in range(
        positive_count,
        eligible_authorities + 1,
    ):
        numerator += (
            math.comb(
                eligible_authorities,
                k,
            )
            * (
                0.5
                ** eligible_authorities
            )
        )

    sign_p = float(
        numerator
    )


corrected_rarity_rows = [
    r
    for r in rarity_rows
    if r[
        "ENTITY_ID"
    ]
    == "EXP-012"
]


population_artifacts_present = (
    len(
        external_artifacts
    )
    > 0
)


primary_gate_components = {
    "minimum_auc_authorities_met":
        eligible_authorities >= 3,

    "sign_test_computed":
        sign_p is not None,

    "corrected_rarity_evidence_present":
        len(
            corrected_rarity_rows
        )
        > 0,

    "external_population_artifacts_present":
        population_artifacts_present,
}


primary_gate_complete = all(
    primary_gate_components.values()
)


summary = {
    "module":
        "V16_POPULATION_SCALE_EXTERNAL_CONFIRMATION",

    "primary_authorities":
        PRIMARY_AUTHORITIES,

    "historical_nonfinal_entities":
        HISTORICAL_NONFINAL,

    "eligible_auc_authorities":
        eligible_authorities,

    "positive_above_chance_authorities":
        positive_count,

    "exact_one_sided_sign_test_p":
        sign_p,

    "corrected_rarity_evidence_rows_exp012":
        len(
            corrected_rarity_rows
        ),

    "external_population_artifacts":
        len(
            external_artifacts
        ),

    "primary_gate_components":
        primary_gate_components,

    "primary_gate_complete":
        primary_gate_complete,

    "alpha":
        0.05,

    "interpretation_constraints": [
        "RUN-009/F2-A remains incomplete historical evidence.",
        "RUN-010/F2-BR0 remains a recovery run.",
        "No claim of zero external reproduction is allowed.",
        "Rarity is conditional on tested representation and external population.",
        "Technical parse failures are not scientific negatives.",
    ],
}


(
    OUT
    / "V16_FINAL_SUMMARY.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


print(
    json.dumps(
        summary,
        indent=2,
    )
)
