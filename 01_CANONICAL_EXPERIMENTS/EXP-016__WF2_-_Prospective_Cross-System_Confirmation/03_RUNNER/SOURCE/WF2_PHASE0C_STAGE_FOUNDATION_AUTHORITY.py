from pathlib import Path
import csv
import hashlib
import json
import re
import shutil
import traceback


ROOT = Path(
    r"C:\RSOS\RSOS_EXPERIMENTS_WF2_PROSPECTIVE_CROSS_SYSTEM_CONFIRMATION"
)

CONTROL = ROOT / "00_CONTROL"
INPUTS = ROOT / "02_INPUTS"

LEGACY = (
    CONTROL
    / "PRECONSOLIDATION"
    / "RSOS_EXPERIMENTS_WF2_A_PROSPECTIVE_FROZEN_SIGNATURE"
)

STAGE = (
    INPUTS
    / "00_FOUNDATION_FROZEN"
)

SOURCE_PACKAGE = (
    LEGACY
    / "02_FROZEN_SIGNATURE"
    / "DETECTOR_PACKAGE"
)

SOURCE_MANIFEST = (
    LEGACY
    / "02_FROZEN_SIGNATURE"
    / "00_DETECTOR_MANIFEST.csv"
)

SOURCE_WF1B = (
    LEGACY
    / "02_FROZEN_SIGNATURE"
    / "WF1_B_FINAL_AUTHORITY_R3.zip"
)

SOURCE_PAIRSET = (
    LEGACY
    / "03_INPUTS"
    / "WF2_A_FROZEN_40_MATCHED_PAIRS.jsonl"
)

SOURCE_PREREG = (
    LEGACY
    / "01_PROTOCOL"
    / "WF2_A_PREREGISTRATION.txt"
)

SOURCE_SET_FREEZE = (
    LEGACY
    / "01_PROTOCOL"
    / "WF2_A_PROSPECTIVE_SET_FREEZE.txt"
)

SOURCE_BUILDER = (
    LEGACY
    / "04_RUNNER"
    / "01_BUILD_FROZEN_PROSPECTIVE_SET.py"
)

REPORT = (
    CONTROL
    / "PHASE0C_FOUNDATION_AUTHORITY_STAGING_REPORT.txt"
)

SUMMARY = (
    CONTROL
    / "PHASE0C_FOUNDATION_AUTHORITY_STAGING_SUMMARY.json"
)

ERRORLOG = (
    CONTROL
    / "PHASE0C_FOUNDATION_AUTHORITY_STAGING_ERROR.txt"
)


EXPECTED_WF1B_SHA256 = (
    "a5da5ffbb3c2ba71bbc2c17894fa97f8"
    "fec5b4c712e0714534517c5b54cedee3"
)

EXPECTED_PAIRSET_SHA256 = (
    "c43e54e5697fe6e35982ef4a559f909"
    "068f76e822edb7c6b172f536ef74090b5"
)


def sha256_file(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def extract_sha(text, heading):

    pattern = (
        re.escape(heading)
        + r".*?([a-fA-F0-9]{64})"
    )

    match = re.search(
        pattern,
        text,
        flags=re.S
    )

    return (
        match.group(1).lower()
        if match
        else None
    )


def copy_verified(src, dst):

    dst.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if dst.exists():

        src_hash = sha256_file(src)
        dst_hash = sha256_file(dst)

        if src_hash != dst_hash:

            raise RuntimeError(
                f"Existing staged file differs: {dst}"
            )

        return "ALREADY_PRESENT_IDENTICAL"

    shutil.copy2(
        src,
        dst
    )

    if sha256_file(src) != sha256_file(dst):

        raise RuntimeError(
            f"Post-copy SHA mismatch: {dst}"
        )

    return "COPIED_VERIFIED"


try:

    required = [
        SOURCE_PACKAGE,
        SOURCE_MANIFEST,
        SOURCE_WF1B,
        SOURCE_PAIRSET,
        SOURCE_PREREG,
        SOURCE_SET_FREEZE,
        SOURCE_BUILDER
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                str(path)
            )


    # ========================================================
    # 1. VERIFY UPSTREAM WF1-B AUTHORITY HASH
    # ========================================================

    actual_wf1b_sha = sha256_file(
        SOURCE_WF1B
    )

    if actual_wf1b_sha != EXPECTED_WF1B_SHA256:

        raise RuntimeError(
            "WF1-B final-authority ZIP SHA does not match "
            "the authority-resolution value."
        )


    prereg_text = SOURCE_PREREG.read_text(
        encoding="utf-8-sig",
        errors="ignore"
    )

    prereg_wf1b_sha = extract_sha(
        prereg_text,
        "UPSTREAM WF1-B FINAL AUTHORITY SHA256"
    )

    if prereg_wf1b_sha != actual_wf1b_sha:

        raise RuntimeError(
            "V13 preregistration does not bind to the actual "
            "WF1-B final-authority ZIP."
        )


    # ========================================================
    # 2. VERIFY PROSPECTIVE SOURCE-SET HASH
    # ========================================================

    actual_pairset_sha = sha256_file(
        SOURCE_PAIRSET
    )

    if actual_pairset_sha != EXPECTED_PAIRSET_SHA256:

        raise RuntimeError(
            "40-pair source-set SHA differs from Phase 0B."
        )


    freeze_text = SOURCE_SET_FREEZE.read_text(
        encoding="utf-8-sig",
        errors="ignore"
    )

    freeze_pair_sha = extract_sha(
        freeze_text,
        "INPUT SHA256"
    )

    if freeze_pair_sha != actual_pairset_sha:

        raise RuntimeError(
            "Prospective-set freeze SHA does not match "
            "the actual source set."
        )


    # ========================================================
    # 3. VERIFY EVERY DETECTOR PACKAGE FILE AGAINST MANIFEST
    # ========================================================

    manifest_rows = []

    with SOURCE_MANIFEST.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            manifest_rows.append({
                "relative_path":
                    row["RelativePath"],

                "size_bytes":
                    int(row["SizeBytes"]),

                "sha256":
                    row["SHA256"].lower()
            })


    manifest_paths = set()

    detector_verification = []

    for row in manifest_rows:

        rel = Path(
            row["relative_path"]
        )

        manifest_paths.add(
            str(rel).replace("/", "\\").lower()
        )

        src = SOURCE_PACKAGE / rel

        if not src.exists():

            raise FileNotFoundError(
                f"Manifest file missing: {src}"
            )

        actual_size = src.stat().st_size
        actual_sha = sha256_file(src)

        size_ok = (
            actual_size
            == row["size_bytes"]
        )

        sha_ok = (
            actual_sha
            == row["sha256"]
        )

        if not size_ok or not sha_ok:

            raise RuntimeError(
                f"Detector manifest mismatch: {rel}"
            )

        detector_verification.append({
            "relative_path":
                str(rel),

            "size_bytes":
                actual_size,

            "sha256":
                actual_sha,

            "status":
                "VERIFIED"
        })


    actual_package_files = [
        p
        for p in SOURCE_PACKAGE.rglob("*")
        if p.is_file()
    ]

    actual_paths = {
        str(
            p.relative_to(SOURCE_PACKAGE)
        ).replace("/", "\\").lower()

        for p in actual_package_files
    }


    missing_from_manifest = sorted(
        actual_paths - manifest_paths
    )

    missing_from_package = sorted(
        manifest_paths - actual_paths
    )


    if missing_from_manifest:

        raise RuntimeError(
            "Detector package contains unmanifested files: "
            + ", ".join(missing_from_manifest)
        )


    if missing_from_package:

        raise RuntimeError(
            "Detector manifest references missing files: "
            + ", ".join(missing_from_package)
        )


    # ========================================================
    # 4. VALIDATE SOURCE-SET STRUCTURE
    # ========================================================

    row_count = 0
    valid_count = 0

    labels = {}
    pair_indices = {}


    with SOURCE_PAIRSET.open(
        "r",
        encoding="utf-8",
        errors="strict"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            row_count += 1

            obj = json.loads(line)

            valid_count += 1

            label = str(
                obj.get("LABEL")
            )

            labels[label] = (
                labels.get(label, 0)
                + 1
            )

            pair = str(
                obj.get("PAIR_INDEX")
            )

            pair_indices[pair] = (
                pair_indices.get(pair, 0)
                + 1
            )


    if row_count != 80:

        raise RuntimeError(
            f"Expected 80 source rows, found {row_count}."
        )


    # ========================================================
    # 5. CREATE STAGING STRUCTURE
    # ========================================================

    STAGE.mkdir(
        parents=True,
        exist_ok=True
    )

    detector_dest = (
        STAGE
        / "DETECTOR_PACKAGE"
    )

    source_records_dest = (
        STAGE
        / "SOURCE_RECORDS"
    )

    source_records_dest.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # 6. COPY DETECTOR PACKAGE EXACTLY
    # ========================================================

    copy_records = []

    for row in manifest_rows:

        rel = Path(
            row["relative_path"]
        )

        src = (
            SOURCE_PACKAGE
            / rel
        )

        dst = (
            detector_dest
            / rel
        )

        action = copy_verified(
            src,
            dst
        )

        copy_records.append({
            "source":
                str(src),

            "destination":
                str(dst),

            "sha256":
                row["sha256"],

            "action":
                action
        })


    # ========================================================
    # 7. COPY AUTHORITY + SOURCE RECORDS
    # ========================================================

    direct_files = [

        (
            SOURCE_MANIFEST,
            STAGE / "00_DETECTOR_MANIFEST.csv"
        ),

        (
            SOURCE_WF1B,
            STAGE / "WF1_B_FINAL_AUTHORITY_R3.zip"
        ),

        (
            SOURCE_PAIRSET,
            STAGE / "WF2_A_FROZEN_40_MATCHED_PAIRS.jsonl"
        ),

        (
            SOURCE_PREREG,
            source_records_dest
            / "WF2_A_PREREGISTRATION.txt"
        ),

        (
            SOURCE_SET_FREEZE,
            source_records_dest
            / "WF2_A_PROSPECTIVE_SET_FREEZE.txt"
        ),

        (
            SOURCE_BUILDER,
            source_records_dest
            / "01_BUILD_FROZEN_PROSPECTIVE_SET.py"
        )
    ]


    for src, dst in direct_files:

        action = copy_verified(
            src,
            dst
        )

        copy_records.append({
            "source":
                str(src),

            "destination":
                str(dst),

            "sha256":
                sha256_file(src),

            "action":
                action
        })


    # ========================================================
    # 8. AUTHORITY DECISION
    # ========================================================

    authority = {

        "phase":
            "WF2_PHASE0C_FOUNDATION_AUTHORITY_STAGING",

        "wf1b_authority": {

            "status":
                "VALID_POSITIVE_CORE",

            "zip":
                "WF1_B_FINAL_AUTHORITY_R3.zip",

            "sha256":
                actual_wf1b_sha,

            "restriction":
                (
                    "FAILED SECONDARY CEM NUISANCE CONTROL "
                    "MUST NOT BE PROMOTED AS VALID "
                    "MATCHED-CONTROL EVIDENCE"
                )
        },

        "prospective_source_set": {

            "status":
                "FROZEN_PRE_PROSPECTIVE",

            "rows":
                row_count,

            "sha256":
                actual_pairset_sha,

            "labels":
                labels,

            "pair_index_multiplicity":
                pair_indices
        },

        "detector_package": {

            "manifest_rows":
                len(manifest_rows),

            "all_manifest_entries_verified":
                True,

            "unmanifested_files":
                [],

            "missing_manifest_files":
                []
        },

        "detector_roles": {

            "primary":
                "JOINT_MULTILAYER_V2",

            "confirmatory": [
                "OPERATOR_RELATIONAL_V2",
                "SURFACE_FREE",
                "R1_COMPATIBLE"
            ],

            "length_robustness": [
                "WINDOW8_DETERMINISTIC_V2",
                "WINDOW12_DETERMINISTIC_V2"
            ],

            "historical_derived_measures_not_promoted_to_primary": [
                "COMPOSITE_MEDIAN_Z",
                "COMPOSITE_MIN_Z",
                "MAHALANOBIS",
                "KNN5_DISTANCE",
                "CENTROID_COSINE"
            ]
        },

        "scientific_observations_created":
            False,

        "prospective_generation_performed":
            False,

        "detector_training_performed":
            False,

        "threshold_tuning_performed":
            False,

        "boundary_1_reached":
            False
    }


    # ========================================================
    # 9. WRITE STAGED AUTHORITY FILE
    # ========================================================

    authority_json = (
        STAGE
        / "WF2_FOUNDATION_AUTHORITY_DECISION.json"
    )

    authority_json.write_text(
        json.dumps(
            authority,
            indent=2,
            sort_keys=True
        ),
        encoding="utf-8"
    )


    # ========================================================
    # 10. CREATE STAGED INPUT MANIFEST
    # ========================================================

    staged_manifest = (
        STAGE
        / "WF2_FOUNDATION_STAGED_MANIFEST.csv"
    )

    staged_files = sorted(
        p
        for p in STAGE.rglob("*")
        if p.is_file()
        and p != staged_manifest
    )


    with staged_manifest.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "RelativePath",
                "SizeBytes",
                "SHA256"
            ]
        )

        writer.writeheader()

        for path in staged_files:

            writer.writerow({
                "RelativePath":
                    str(
                        path.relative_to(STAGE)
                    ),

                "SizeBytes":
                    path.stat().st_size,

                "SHA256":
                    sha256_file(path)
            })


    # ========================================================
    # 11. WRITE REPORT
    # ========================================================

    report_lines = []

    report_lines.extend([
        "=" * 88,
        "WF2 PHASE 0C — VERIFIED FOUNDATION AUTHORITY STAGING",
        "=" * 88,
        "",
        f"WF2 ROOT: {ROOT}",
        f"STAGED FOUNDATION ROOT: {STAGE}",
        "",
        "WF1-B FINAL AUTHORITY:",
        f"  SHA256: {actual_wf1b_sha}",
        "  SCIENTIFIC VALIDITY: VALID",
        "  PRIMARY CORE: VALID POSITIVE CORE",
        "",
        "IMPORTANT EXCLUSION:",
        "  CEM RESULT MUST NOT BE PROMOTED AS VALID MATCHED-CONTROL EVIDENCE.",
        "",
        "FROZEN DETECTOR PACKAGE:",
        f"  Manifest entries verified: {len(manifest_rows)}",
        "  Missing files: 0",
        "  Unmanifested files: 0",
        "",
        "PROSPECTIVE SOURCE SET:",
        f"  Rows: {row_count}",
        f"  Valid JSON rows: {valid_count}",
        f"  SHA256: {actual_pairset_sha}",
        f"  Labels: {labels}",
        "",
        "DETECTOR ROLE RESOLUTION:",
        "  PRIMARY: JOINT_MULTILAYER_V2",
        "  CONFIRMATORY: OPERATOR_RELATIONAL_V2",
        "  CONFIRMATORY: SURFACE_FREE",
        "  CONFIRMATORY: R1_COMPATIBLE",
        "  LENGTH ROBUSTNESS: WINDOW8_DETERMINISTIC_V2",
        "  LENGTH ROBUSTNESS: WINDOW12_DETERMINISTIC_V2",
        "",
        "DERIVED WF1-B MEASURES:",
        "  Preserved as historical/secondary evidence.",
        "  Not promoted to the primary prospective detector at this stage.",
        "",
        "PROSPECTIVE GENERATION PERFORMED: NO",
        "DETECTOR TRAINING PERFORMED: NO",
        "THRESHOLD TUNING PERFORMED: NO",
        "BOUNDARY 1 REACHED: NO",
        "",
        "NEXT:",
        "Construct unified WF2-A/B/C preregistration, provider matrix,",
        "generation matrix, perturbation matrix, decision rules and runner.",
    ])


    REPORT.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )


    SUMMARY.write_text(
        json.dumps(
            authority,
            indent=2,
            sort_keys=True
        ),
        encoding="utf-8"
    )


    if ERRORLOG.exists():
        ERRORLOG.unlink()


    print(
        REPORT.read_text(
            encoding="utf-8"
        )
    )


except Exception:

    err = traceback.format_exc()

    ERRORLOG.write_text(
        err,
        encoding="utf-8"
    )

    print("")
    print("=" * 88)
    print("WF2 PHASE 0C ERROR")
    print("=" * 88)
    print("")
    print(err)
    print("")
    print(f"ERROR LOG: {ERRORLOG}")

