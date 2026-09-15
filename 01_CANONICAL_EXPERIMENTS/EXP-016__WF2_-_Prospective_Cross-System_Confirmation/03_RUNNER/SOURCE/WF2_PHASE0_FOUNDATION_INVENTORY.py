from pathlib import Path
import csv
import json
import os
import re
import zipfile
from datetime import datetime, timezone


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_ROOT = Path(r"C:\RSOS")

WF2_ROOT = Path(
    r"C:\RSOS\RSOS_EXPERIMENTS_WF2_PROSPECTIVE_CROSS_SYSTEM_CONFIRMATION"
)

CONTROL = WF2_ROOT / "00_CONTROL"
TEMP = CONTROL / "TEMP"

CONTROL.mkdir(parents=True, exist_ok=True)
TEMP.mkdir(parents=True, exist_ok=True)


# ============================================================
# SAFETY RULE
#
# WF2 is excluded from the Foundation scan so the experiment
# never mistakes its own newly-created artifacts for historical
# Foundation evidence.
# ============================================================

EXCLUDE_DIR_NAMES = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "env"
}


# ============================================================
# HISTORICAL FOUNDATION FAMILIES
# ============================================================

SOURCE_FAMILY_PATTERNS = {

    "F1": [
        r"(^|[^A-Z0-9])F1([^A-Z0-9]|$)"
    ],

    "F2": [
        r"(^|[^A-Z0-9])F2([^A-Z0-9]|$)"
    ],

    "V11": [
        r"(^|[^A-Z0-9])V11([^A-Z0-9]|$)",
        r"RELATIONAL[_ -]?GRAPH[_ -]?FINGERPRINT"
    ],

    "V12": [
        r"(^|[^A-Z0-9])V12([^A-Z0-9]|$)",
        r"RELATIONAL[_ -]?TOPOLOGY",
        r"CAUSAL[_ -]?MAP"
    ],

    "WF1-P0": [
        r"WF1[_ -]?P0",
        r"HISTORICAL[_ -]?CORPUS[_ -]?BRIDGE"
    ],

    "WF1-A": [
        r"WF1[_ -]?A"
    ],

    "WF1-B": [
        r"WF1[_ -]?B"
    ]
}


# ============================================================
# REQUIRED ARTIFACT CLASSES
# ============================================================

CLASS_PATTERNS = {

    "FROZEN_DETECTOR": [
        r"DETECTOR",
        r"CLASSIFIER",
        r"MODEL\.PKL",
        r"MODEL\.PICKLE",
        r"MODEL\.JOBLIB",
        r"COEFFICIENT",
        r"WEIGHTS",
        r"DECISION[_ -]?FUNCTION"
    ],

    "FEATURE_CONTRACT": [
        r"FEATURE",
        r"SCALER",
        r"VECTORIZ",
        r"PREPROCESS",
        r"EXTRACTOR",
        r"GRAPH[_ -]?VECTOR",
        r"OPERATOR[_ -]?FIDELITY",
        r"RESPONSE[_ -]?GRAPH"
    ],

    "THRESHOLD_OR_SCORING_RULE": [
        r"THRESHOLD",
        r"CUTOFF",
        r"DECISION[_ -]?RULE",
        r"SCOR",
        r"PROBABIL"
    ],

    "MATCHED_SOURCE_SUBSET": [
        r"MATCHED",
        r"SOURCE[_ -]?SUBSET",
        r"HOLDOUT",
        r"SEED[_ -]?LEVEL",
        r"HISTORICAL[_ -]?SUBSET",
        r"CONTEXT[_ -]?SUBSET"
    ],

    "CONTROL_SUBSET": [
        r"CONTROL",
        r"NEGATIVE",
        r"NON[_ -]?LINEAGE",
        r"BASELINE",
        r"WILDCHAT"
    ],

    "CONTEXT_OR_ID_MAP": [
        r"CONTEXT",
        r"PROMPT",
        r"SEED",
        r"ID[_ -]?MAP",
        r"INDEX",
        r"CORPUS[_ -]?BRIDGE"
    ],

    "FROZEN_SCORER_OR_RUNNER": [
        r"RUNNER",
        r"SCORER",
        r"EVALUAT",
        r"PREDICT",
        r"INFERENCE"
    ],

    "FREEZE_OR_MANIFEST": [
        r"FREEZE",
        r"MANIFEST",
        r"SHA256",
        r"HASH",
        r"CUSTODY"
    ],

    "PREVIOUS_ADJUDICATION": [
        r"ADJUDICATION",
        r"FINAL[_ -]?SUMMARY",
        r"FINAL[_ -]?RESULT",
        r"AUTHORITY"
    ],

    "SURFACE_INVARIANCE_PRIOR": [
        r"SURFACE",
        r"TERMINOLOGY",
        r"PARAPHRASE",
        r"TOPIC",
        r"STYLE",
        r"DOMAIN",
        r"INVARIANCE"
    ],

    "PROVIDER_OR_MODEL_PRIOR": [
        r"MODEL[_ -]?NATIVE",
        r"PROVIDER",
        r"CROSS[_ -]?MODEL",
        r"MODEL[_ -]?SUMMARY"
    ]
}


# ============================================================
# FILE TYPES TO INSPECT
# ============================================================

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".py",
    ".ps1",
    ".csv"
}

CANDIDATE_EXTENSIONS = {
    ".txt",
    ".md",
    ".json",
    ".jsonl",
    ".csv",
    ".parquet",
    ".pkl",
    ".pickle",
    ".joblib",
    ".npz",
    ".npy",
    ".py",
    ".ps1",
    ".zip",
    ".yaml",
    ".yml",
    ".toml"
}


CONTENT_TERMS = [
    "detector",
    "classifier",
    "feature",
    "threshold",
    "holdout",
    "matched",
    "surface",
    "invariance",
    "operator",
    "graph vector",
    "adjudication",
    "prospective",
    "provider",
    "cross-model"
]


# ============================================================
# HELPERS
# ============================================================

def normalize(value):
    return str(value).replace("\\", "/").upper()


def source_family(text):

    t = normalize(text)

    found = []

    for family, patterns in SOURCE_FAMILY_PATTERNS.items():

        if any(
            re.search(pattern, t, re.I)
            for pattern in patterns
        ):
            found.append(family)

    return "|".join(found)


def classify(text):

    t = normalize(text)

    found = []

    for artifact_class, patterns in CLASS_PATTERNS.items():

        if any(
            re.search(pattern, t, re.I)
            for pattern in patterns
        ):
            found.append(artifact_class)

    return found


def inspect_small_text(path):

    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return []

    try:

        size = path.stat().st_size

        # Avoid loading very large text files during inventory.
        if size > 5 * 1024 * 1024:
            return []

        content = path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        content = content[:2_000_000].lower()

        return [
            term
            for term in CONTENT_TERMS
            if term in content
        ]

    except Exception:
        return []


# ============================================================
# SCAN FOUNDATION FILES
# ============================================================

file_rows = []
zip_rows = []


for current_root, dirs, files in os.walk(SOURCE_ROOT):

    current = Path(current_root)

    # --------------------------------------------------------
    # Exclude the active WF2 experiment completely.
    # --------------------------------------------------------

    try:

        current.relative_to(WF2_ROOT)

        dirs[:] = []

        continue

    except ValueError:
        pass


    # --------------------------------------------------------
    # Skip irrelevant technical directories.
    # --------------------------------------------------------

    dirs[:] = [
        d
        for d in dirs
        if d not in EXCLUDE_DIR_NAMES
    ]


    for filename in files:

        path = current / filename

        suffix = path.suffix.lower()

        if suffix not in CANDIDATE_EXTENSIONS:
            continue


        full_text = str(path)

        family = source_family(full_text)

        classes = classify(full_text)

        content_matches = inspect_small_text(path)


        # Ignore files with no meaningful relevance signal.
        if not family and not classes and not content_matches:
            continue


        try:

            stat = path.stat()

            size_bytes = stat.st_size

            modified_utc = datetime.fromtimestamp(
                stat.st_mtime,
                tz=timezone.utc
            ).isoformat()

        except Exception:

            size_bytes = None
            modified_utc = None


        score = (
            len(classes) * 3
            + len(content_matches)
            + (3 if family else 0)
        )


        file_rows.append({

            "source_family":
                family or "UNRESOLVED",

            "artifact_classes":
                "|".join(classes),

            "score":
                score,

            "size_bytes":
                size_bytes,

            "modified_utc":
                modified_utc,

            "content_terms":
                "|".join(content_matches),

            "path":
                str(path)
        })


        # ----------------------------------------------------
        # Inspect archive member names without extracting ZIPs.
        # ----------------------------------------------------

        if suffix == ".zip":

            try:

                with zipfile.ZipFile(path, "r") as archive:

                    for member in archive.namelist():

                        member_context = (
                            str(path)
                            + "/"
                            + member
                        )

                        member_family = source_family(
                            member_context
                        )

                        member_classes = classify(
                            member
                        )

                        if member_family or member_classes:

                            zip_rows.append({

                                "archive":
                                    str(path),

                                "source_family":
                                    member_family
                                    or family
                                    or "UNRESOLVED",

                                "artifact_classes":
                                    "|".join(
                                        member_classes
                                    ),

                                "member":
                                    member
                            })


            except Exception as exc:

                zip_rows.append({

                    "archive":
                        str(path),

                    "source_family":
                        family or "UNRESOLVED",

                    "artifact_classes":
                        "ZIP_READ_ERROR",

                    "member":
                        repr(exc)
                })


# ============================================================
# SORT RESULTS
# ============================================================

file_rows.sort(

    key=lambda row: (

        -row["score"],

        row["source_family"],

        row["path"].lower()
    )
)


zip_rows.sort(

    key=lambda row: (

        row["source_family"],

        row["archive"].lower(),

        row["member"].lower()
    )
)


# ============================================================
# WRITE FILE CANDIDATE INVENTORY
# ============================================================

file_csv = (
    CONTROL
    / "PHASE0_FOUNDATION_FILE_CANDIDATES.csv"
)

with file_csv.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "source_family",
            "artifact_classes",
            "score",
            "size_bytes",
            "modified_utc",
            "content_terms",
            "path"
        ]
    )

    writer.writeheader()

    writer.writerows(file_rows)


# ============================================================
# WRITE ZIP MEMBER INVENTORY
# ============================================================

zip_csv = (
    CONTROL
    / "PHASE0_FOUNDATION_ZIP_MEMBER_CANDIDATES.csv"
)

with zip_csv.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "archive",
            "source_family",
            "artifact_classes",
            "member"
        ]
    )

    writer.writeheader()

    writer.writerows(zip_rows)


# ============================================================
# HELPER FOR REQUIREMENT COUNTS
# ============================================================

def count_class(artifact_class):

    file_count = sum(

        artifact_class
        in row["artifact_classes"].split("|")

        for row in file_rows
    )

    zip_count = sum(

        artifact_class
        in row["artifact_classes"].split("|")

        for row in zip_rows
    )

    return file_count, zip_count


# ============================================================
# HISTORICAL REQUIREMENTS
# ============================================================

historical_requirements = [

    (
        "FROZEN_DETECTOR",
        "Exact previously frozen lineage detector/model",
        True
    ),

    (
        "FEATURE_CONTRACT",
        "Exact preprocessing and structural feature definition used by detector",
        True
    ),

    (
        "THRESHOLD_OR_SCORING_RULE",
        "Frozen scoring, threshold or decision rule where applicable",
        True
    ),

    (
        "MATCHED_SOURCE_SUBSET",
        "Historical contexts permitted to initialize prospective generation",
        True
    ),

    (
        "CONTROL_SUBSET",
        "Comparable frozen control contexts or source population",
        True
    ),

    (
        "CONTEXT_OR_ID_MAP",
        "Stable identifiers linking contexts to frozen Foundation evidence",
        True
    ),

    (
        "FROZEN_SCORER_OR_RUNNER",
        "Frozen scoring implementation or compatible reference implementation",
        True
    ),

    (
        "FREEZE_OR_MANIFEST",
        "Evidence that selected Foundation dependencies were previously frozen",
        True
    ),

    (
        "PREVIOUS_ADJUDICATION",
        "Authority/status records identifying valid Foundation artifacts",
        True
    ),

    (
        "SURFACE_INVARIANCE_PRIOR",
        "Historical perturbation evidence usable only to design WF2-C preregistration",
        False
    ),

    (
        "PROVIDER_OR_MODEL_PRIOR",
        "Historical cross-model evidence usable as context only",
        False
    )
]


requirement_rows = []


for artifact_class, purpose, required in historical_requirements:

    file_count, zip_count = count_class(
        artifact_class
    )

    total = (
        file_count
        + zip_count
    )


    requirement_rows.append({

        "requirement":
            artifact_class,

        "purpose":
            purpose,

        "required_before_boundary1":
            "YES"
            if required
            else "REFERENCE_ONLY",

        "file_candidates":
            file_count,

        "zip_member_candidates":
            zip_count,

        "phase0_status":
            "CANDIDATE_FOUND"
            if total
            else "NOT_FOUND"
    })


# ============================================================
# WF2-NATIVE PREREGISTRATION OBJECTS
# ============================================================

wf2_new_requirements = [

    (
        "WF2_PROSPECTIVE_SOURCE_LOCK",
        "Exact source/context rows selected before prospective generation"
    ),

    (
        "WF2_PROVIDER_MODEL_MATRIX",
        "Provider/model identities, roles and replication cells"
    ),

    (
        "WF2_GENERATION_MATRIX",
        "Prompts, turn counts, generation parameters and cell structure"
    ),

    (
        "WF2_PERTURBATION_MATRIX",
        "Predetermined terminology, topic, domain and style transformations"
    ),

    (
        "WF2_RANDOMIZATION_SEED_SCHEDULE",
        "Fixed seed/randomization allocation before observation"
    ),

    (
        "WF2_PRIMARY_ENDPOINTS",
        "Separate Module A, Module B and Module C outcome definitions"
    ),

    (
        "WF2_DECISION_RULES",
        "Positive, negative, mixed, null and technical-failure rules"
    ),

    (
        "WF2_FAILURE_LOG_SCHEMA",
        "Missing-cell and API/provider-failure preservation rules"
    ),

    (
        "WF2_INPUT_MANIFEST",
        "Selected Foundation dependencies and WF2 preregistration inputs"
    ),

    (
        "WF2_FROZEN_RUNNER",
        "Runner incapable of detector fitting/tuning after prospective observations"
    )
]


for name, purpose in wf2_new_requirements:

    requirement_rows.append({

        "requirement":
            name,

        "purpose":
            purpose,

        "required_before_boundary1":
            "YES",

        "file_candidates":
            0,

        "zip_member_candidates":
            0,

        "phase0_status":
            "TO_CREATE_WF2"
    })


# ============================================================
# WRITE REQUIREMENT MATRIX
# ============================================================

requirement_csv = (
    CONTROL
    / "PHASE0_REQUIREMENT_MATRIX.csv"
)

with requirement_csv.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "requirement",
            "purpose",
            "required_before_boundary1",
            "file_candidates",
            "zip_member_candidates",
            "phase0_status"
        ]
    )

    writer.writeheader()

    writer.writerows(
        requirement_rows
    )


# ============================================================
# IDENTIFY HISTORICAL REQUIREMENTS NOT YET LOCATED
# ============================================================

missing_historical = [

    row["requirement"]

    for row in requirement_rows

    if (
        row["required_before_boundary1"] == "YES"
        and
        row["phase0_status"] == "NOT_FOUND"
    )
]


# ============================================================
# WRITE MACHINE-READABLE SUMMARY
# ============================================================

summary = {

    "experiment":
        "WF2_PROSPECTIVE_CROSS_SYSTEM_CONFIRMATION",

    "phase":
        "PHASE_0_FOUNDATION_INVENTORY",

    "experiment_root":
        str(WF2_ROOT),

    "scientific_observations_created":
        False,

    "prospective_generation_performed":
        False,

    "detector_training_performed":
        False,

    "threshold_tuning_performed":
        False,

    "historical_artifacts_modified":
        False,

    "boundary_1_reached":
        False,

    "foundation_file_candidates":
        len(file_rows),

    "foundation_zip_member_candidates":
        len(zip_rows),

    "missing_historical_requirement_classes":
        missing_historical,

    "wf2_native_preregistration_objects_to_create":
        [
            item[0]
            for item
            in wf2_new_requirements
        ]
}


summary_json = (
    CONTROL
    / "PHASE0_SUMMARY.json"
)

summary_json.write_text(
    json.dumps(
        summary,
        indent=2
    ),
    encoding="utf-8"
)


# ============================================================
# WRITE HUMAN-READABLE REPORT
# ============================================================

report = (
    CONTROL
    / "PHASE0_REPORT.txt"
)


with report.open(
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "WF2 PHASE 0 — FOUNDATION INVENTORY\n"
    )

    f.write(
        "=" * 78
        + "\n\n"
    )

    f.write(
        f"WF2 ROOT:\n{WF2_ROOT}\n\n"
    )

    f.write(
        "SCIENTIFIC OBSERVATIONS CREATED: NO\n"
    )

    f.write(
        "PROSPECTIVE GENERATION PERFORMED: NO\n"
    )

    f.write(
        "DETECTOR TRAINING PERFORMED: NO\n"
    )

    f.write(
        "THRESHOLD TUNING PERFORMED: NO\n"
    )

    f.write(
        "FOUNDATION ARTIFACTS MODIFIED: NO\n"
    )

    f.write(
        "BOUNDARY 1 REACHED: NO\n\n"
    )

    f.write(
        f"FOUNDATION FILE CANDIDATES: "
        f"{len(file_rows)}\n"
    )

    f.write(
        f"FOUNDATION ZIP MEMBER CANDIDATES: "
        f"{len(zip_rows)}\n\n"
    )


    f.write(
        "HISTORICAL REQUIREMENT MATRIX\n"
    )

    f.write(
        "-" * 78
        + "\n"
    )


    for row in requirement_rows:

        f.write(

            f"{row['requirement']}: "
            f"{row['phase0_status']} "
            f"| files={row['file_candidates']} "
            f"| zip_members={row['zip_member_candidates']}"
            "\n"
        )


    f.write("\n")


    if missing_historical:

        f.write(
            "REQUIRED HISTORICAL CLASSES NOT YET LOCATED\n"
        )

        f.write(
            "-" * 78
            + "\n"
        )

        for item in missing_historical:

            f.write(
                f"- {item}\n"
            )

    else:

        f.write(
            "All required historical dependency classes "
            "have at least one candidate artifact.\n"
        )


    f.write("\n")


    f.write(
        "WF2-NATIVE OBJECTS REQUIRED BEFORE BOUNDARY 1\n"
    )

    f.write(
        "-" * 78
        + "\n"
    )


    for name, purpose in wf2_new_requirements:

        f.write(
            f"- {name}: {purpose}\n"
        )


    f.write("\n")


    f.write(
        "TOP 60 FOUNDATION CANDIDATES\n"
    )

    f.write(
        "-" * 78
        + "\n"
    )


    for row in file_rows[:60]:

        f.write(

            f"[score={row['score']}] "
            f"[{row['source_family']}] "
            f"[{row['artifact_classes']}] "
            f"{row['path']}\n"
        )


# ============================================================
# CREATE RETURN ARTIFACT
# ============================================================

return_file = (
    CONTROL
    / "PHASE0_RETURN_FOR_ADJUDICATION.txt"
)


with return_file.open(
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "===== WF2 PHASE 0 REPORT =====\n\n"
    )

    f.write(
        report.read_text(
            encoding="utf-8"
        )
    )

    f.write(
        "\n\n"
        "===== PHASE 0 REQUIREMENT MATRIX =====\n\n"
    )

    f.write(
        requirement_csv.read_text(
            encoding="utf-8-sig"
        )
    )


# ============================================================
# FINAL CONSOLE OUTPUT
# ============================================================

print()
print("=" * 78)
print("WF2 PHASE 0 COMPLETE")
print("=" * 78)

print()

print(
    report.read_text(
        encoding="utf-8"
    )
)

print()

print("=" * 78)
print("WF2 FILES CREATED")
print("=" * 78)

print(
    WF2_ROOT
    / "03_RUNNER"
    / "WF2_PHASE0_FOUNDATION_INVENTORY.py"
)

print(file_csv)

print(zip_csv)

print(requirement_csv)

print(summary_json)

print(report)

print(return_file)

print()

print(
    "BOUNDARY 1 REACHED: NO"
)

print(
    "NEXT SCIENTIFIC ACTION: "
    "select authoritative Foundation dependencies."
)
