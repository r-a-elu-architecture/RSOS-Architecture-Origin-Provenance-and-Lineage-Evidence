from pathlib import Path
import csv
import hashlib
import json
import traceback
import zipfile
import sys
from collections import defaultdict


ROOT = Path(
    r"C:\RSOS\RSOS_EXPERIMENTS_WF2_PROSPECTIVE_CROSS_SYSTEM_CONFIRMATION"
)

CONTROL = ROOT / "00_CONTROL"

LEGACY = (
    CONTROL
    / "PRECONSOLIDATION"
    / "RSOS_EXPERIMENTS_WF2_A_PROSPECTIVE_FROZEN_SIGNATURE"
)

ZIP_PATH = (
    LEGACY
    / "02_FROZEN_SIGNATURE"
    / "WF1_B_FINAL_AUTHORITY_R3.zip"
)

PACKAGE = (
    LEGACY
    / "02_FROZEN_SIGNATURE"
    / "DETECTOR_PACKAGE"
)

PAIRSET = (
    LEGACY
    / "03_INPUTS"
    / "WF2_A_FROZEN_40_MATCHED_PAIRS.jsonl"
)

PREREG = (
    LEGACY
    / "01_PROTOCOL"
    / "WF2_A_PREREGISTRATION.txt"
)

SETFREEZE = (
    LEGACY
    / "01_PROTOCOL"
    / "WF2_A_PROSPECTIVE_SET_FREEZE.txt"
)

DETECTOR_MANIFEST = (
    LEGACY
    / "02_FROZEN_SIGNATURE"
    / "00_DETECTOR_MANIFEST.csv"
)

OUT = (
    CONTROL
    / "PHASE0B_AUTHORITY_RESOLUTION_REPORT.txt"
)

JSONOUT = (
    CONTROL
    / "PHASE0B_AUTHORITY_RESOLUTION_SUMMARY.json"
)

ERRORLOG = (
    CONTROL
    / "PHASE0B_AUTHORITY_RESOLUTION_ERROR.txt"
)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def safe_text(path, limit=2_000_000):

    try:
        return path.read_text(
            encoding="utf-8",
            errors="ignore"
        )[:limit]

    except Exception:
        return ""


def is_relevant(name):

    u = name.upper()

    terms = (
        "FINAL",
        "AUTHORITY",
        "ADJUDICATION",
        "SUMMARY",
        "DETECTOR",
        "CLASSIFIER",
        "MODEL",
        "FEATURE",
        "FROZEN",
        "SPEC",
        "THRESHOLD",
        "SCOR",
        "HOLDOUT",
        "MATCHED",
        "WINDOW",
        "SURFACE",
        "OPERATOR",
        "JOINT",
        "R1",
        "REPLICATION",
        "RESULT"
    )

    return any(term in u for term in terms)


try:

    required = [
        ZIP_PATH,
        PACKAGE,
        PAIRSET,
        PREREG,
        SETFREEZE,
        DETECTOR_MANIFEST
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(str(path))


    lines = []

    lines.append("=" * 88)
    lines.append("WF2 PHASE 0B — FINAL WF1-B AUTHORITY RESOLUTION")
    lines.append("=" * 88)
    lines.append("")

    lines.append(f"WF2 ROOT: {ROOT}")
    lines.append("")
    lines.append("SCIENTIFIC OBSERVATIONS CREATED: NO")
    lines.append("PROSPECTIVE GENERATION PERFORMED: NO")
    lines.append("DETECTOR TRAINING PERFORMED: NO")
    lines.append("THRESHOLD TUNING PERFORMED: NO")
    lines.append("BOUNDARY 1 REACHED: NO")
    lines.append("")


    # ========================================================
    # A. WF1-B FINAL AUTHORITY ZIP
    # ========================================================

    lines.append("=" * 88)
    lines.append("A. WF1-B FINAL AUTHORITY PACKAGE")
    lines.append("=" * 88)
    lines.append("")

    lines.append(f"PATH: {ZIP_PATH}")
    lines.append(f"SIZE_BYTES: {ZIP_PATH.stat().st_size}")
    lines.append(f"SHA256: {sha256_file(ZIP_PATH)}")
    lines.append("")


    with zipfile.ZipFile(ZIP_PATH, "r") as z:

        names = z.namelist()

        lines.append(f"TOTAL ZIP MEMBERS: {len(names)}")
        lines.append("")

        relevant_names = [
            name for name in names
            if is_relevant(name)
        ]

        lines.append("RELEVANT ZIP MEMBERS")
        lines.append("-" * 88)

        for name in sorted(relevant_names):
            lines.append(name)

        lines.append("")


        # ----------------------------------------------------
        # Group ZIP entries by basename for loose-file matching.
        # ----------------------------------------------------

        by_basename = defaultdict(list)

        for info in z.infolist():

            if info.is_dir():
                continue

            basename = Path(info.filename).name

            by_basename[basename].append(info)


        # ----------------------------------------------------
        # Extract authority-relevant textual lines.
        # ----------------------------------------------------

        lines.append("=" * 88)
        lines.append("B. WF1-B AUTHORITY / ADJUDICATION EXCERPTS")
        lines.append("=" * 88)
        lines.append("")

        search_terms = (
            "final authority",
            "final_authority",
            "authoritative",
            "primary",
            "secondary",
            "sensitivity",
            "superseded",
            "corrective",
            "valid",
            "invalid",
            "joint_multilayer",
            "joint multilayer",
            "operator_relational",
            "operator relational",
            "surface_free",
            "surface-free",
            "r1_compatible",
            "r1 compatible",
            "window8",
            "window12",
            "auc",
            "holdout",
            "matched",
            "detector",
            "feature",
            "threshold"
        )

        text_extensions = (
            ".txt",
            ".md",
            ".json",
            ".csv",
            ".yaml",
            ".yml"
        )

        excerpt_files = 0

        for name in sorted(relevant_names):

            if not name.lower().endswith(text_extensions):
                continue

            try:
                raw = z.read(name)

                text = raw.decode(
                    "utf-8",
                    errors="ignore"
                )

            except Exception:
                continue

            matches = []

            for line in text.splitlines():

                low = line.lower()

                if any(term in low for term in search_terms):
                    matches.append(line)

            if matches:

                excerpt_files += 1

                lines.append(f"--- {name} ---")

                for line in matches[:250]:
                    lines.append(line)

                lines.append("")

        if excerpt_files == 0:
            lines.append(
                "No keyword-matched textual authority excerpts found."
            )
            lines.append("")


        # ====================================================
        # C. LOOSE FROZEN DETECTOR PACKAGE
        # ====================================================

        lines.append("=" * 88)
        lines.append("C. PRECONSOLIDATION FROZEN DETECTOR PACKAGE")
        lines.append("=" * 88)
        lines.append("")

        package_files = sorted(
            p for p in PACKAGE.rglob("*")
            if p.is_file()
        )

        loose_inventory = []

        for path in package_files:

            rel = path.relative_to(PACKAGE)

            row = {
                "relative_path": str(rel),
                "basename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path)
            }

            loose_inventory.append(row)

            lines.append(
                f"{row['relative_path']}\t"
                f"{row['size_bytes']}\t"
                f"{row['sha256']}"
            )

        lines.append("")
        lines.append(
            f"TOTAL LOOSE DETECTOR PACKAGE FILES: "
            f"{len(loose_inventory)}"
        )
        lines.append("")


        # ====================================================
        # D. MATCH LOOSE FILES AGAINST WF1-B ZIP
        # ====================================================

        lines.append("=" * 88)
        lines.append("D. LOOSE FILE ↔ WF1-B ZIP MATCHING")
        lines.append("=" * 88)
        lines.append("")

        exact_match_count = 0
        basename_only_no_hash_match = 0
        no_basename_match = 0

        comparison_rows = []

        for loose in loose_inventory:

            basename = loose["basename"]

            candidates = by_basename.get(
                basename,
                []
            )

            if not candidates:

                no_basename_match += 1

                comparison_rows.append({
                    "loose_relative_path":
                        loose["relative_path"],
                    "basename":
                        basename,
                    "loose_sha256":
                        loose["sha256"],
                    "zip_member":
                        "",
                    "zip_sha256":
                        "",
                    "status":
                        "NO_BASENAME_MATCH_IN_WF1B_ZIP"
                })

                continue


            matched = False

            for info in candidates:

                data = z.read(info.filename)

                zhash = sha256_bytes(data)

                if zhash == loose["sha256"]:

                    exact_match_count += 1
                    matched = True

                    comparison_rows.append({
                        "loose_relative_path":
                            loose["relative_path"],
                        "basename":
                            basename,
                        "loose_sha256":
                            loose["sha256"],
                        "zip_member":
                            info.filename,
                        "zip_sha256":
                            zhash,
                        "status":
                            "EXACT_SHA256_MATCH"
                    })

                    break


            if not matched:

                basename_only_no_hash_match += 1

                first = candidates[0]

                first_hash = sha256_bytes(
                    z.read(first.filename)
                )

                comparison_rows.append({
                    "loose_relative_path":
                        loose["relative_path"],
                    "basename":
                        basename,
                    "loose_sha256":
                        loose["sha256"],
                    "zip_member":
                        first.filename,
                    "zip_sha256":
                        first_hash,
                    "status":
                        "BASENAME_MATCH_HASH_DIFFERENT"
                })


        for row in comparison_rows:

            lines.append(
                f"{row['status']}\t"
                f"{row['loose_relative_path']}\t"
                f"{row['zip_member']}"
            )

        lines.append("")
        lines.append(
            f"EXACT_SHA256_MATCH: {exact_match_count}"
        )

        lines.append(
            "BASENAME_MATCH_HASH_DIFFERENT: "
            f"{basename_only_no_hash_match}"
        )

        lines.append(
            "NO_BASENAME_MATCH_IN_WF1B_ZIP: "
            f"{no_basename_match}"
        )

        lines.append("")


    # ========================================================
    # E. FROZEN MODEL INVENTORY
    # ========================================================

    lines.append("=" * 88)
    lines.append("E. FROZEN MODEL INVENTORY")
    lines.append("=" * 88)
    lines.append("")

    frozen_models_dir = (
        PACKAGE
        / "frozen_models"
    )

    frozen_models = []

    if frozen_models_dir.exists():

        for model in sorted(
            frozen_models_dir.glob("*.joblib")
        ):

            record = {
                "name": model.name,
                "size_bytes": model.stat().st_size,
                "sha256": sha256_file(model)
            }

            frozen_models.append(record)

            lines.append(
                f"{record['name']}\t"
                f"{record['size_bytes']}\t"
                f"{record['sha256']}"
            )

    lines.append("")
    lines.append(
        f"TOTAL FROZEN MODELS: {len(frozen_models)}"
    )

    lines.append("")


    # ========================================================
    # F. HISTORICAL V13 PREREGISTRATION
    # ========================================================

    lines.append("=" * 88)
    lines.append("F. PRECONSOLIDATION V13 PREREGISTRATION")
    lines.append("=" * 88)
    lines.append("")

    lines.append(
        safe_text(PREREG)
    )

    lines.append("")


    # ========================================================
    # G. PROSPECTIVE SET FREEZE
    # ========================================================

    lines.append("=" * 88)
    lines.append("G. PRECONSOLIDATION PROSPECTIVE-SET FREEZE")
    lines.append("=" * 88)
    lines.append("")

    lines.append(
        safe_text(SETFREEZE)
    )

    lines.append("")


    # ========================================================
    # H. DETECTOR MANIFEST
    # ========================================================

    lines.append("=" * 88)
    lines.append("H. DETECTOR MANIFEST")
    lines.append("=" * 88)
    lines.append("")

    lines.append(
        safe_text(DETECTOR_MANIFEST)
    )

    lines.append("")


    # ========================================================
    # I. FROZEN 40-MATCHED-PAIR SOURCE SET
    # ========================================================

    lines.append("=" * 88)
    lines.append("I. WF2-A FROZEN 40-MATCHED-PAIR SOURCE SET")
    lines.append("=" * 88)
    lines.append("")

    nonempty_rows = 0
    valid_json_rows = 0
    invalid_json_rows = 0

    top_level_keys = defaultdict(int)

    with PAIRSET.open(
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            nonempty_rows += 1

            try:
                obj = json.loads(line)

                valid_json_rows += 1

                if isinstance(obj, dict):

                    for key in obj:
                        top_level_keys[str(key)] += 1

            except Exception:
                invalid_json_rows += 1


    lines.append(
        f"PATH: {PAIRSET}"
    )

    lines.append(
        f"SIZE_BYTES: {PAIRSET.stat().st_size}"
    )

    lines.append(
        f"SHA256: {sha256_file(PAIRSET)}"
    )

    lines.append(
        f"NONEMPTY_ROWS: {nonempty_rows}"
    )

    lines.append(
        f"VALID_JSON_ROWS: {valid_json_rows}"
    )

    lines.append(
        f"INVALID_JSON_ROWS: {invalid_json_rows}"
    )

    lines.append("")

    lines.append(
        "TOP-LEVEL JSON KEYS:"
    )

    for key, count in sorted(
        top_level_keys.items(),
        key=lambda x: (-x[1], x[0])
    ):

        lines.append(
            f"{key}: {count}"
        )

    lines.append("")


    # ========================================================
    # J. AUTHORITY-RESOLUTION STATUS
    # ========================================================

    lines.append("=" * 88)
    lines.append("J. AUTHORITY-RESOLUTION STATUS")
    lines.append("=" * 88)
    lines.append("")

    lines.append(
        "The existence of a loose detector or prior V13 freeze does "
        "NOT automatically promote it to combined WF2 authority."
    )

    lines.append(
        "WF1-B final authority must determine the primary detector, "
        "secondary/sensitivity detectors, feature contract and "
        "permitted source/control material."
    )

    lines.append(
        "Combined WF2 Boundary 1 remains OPEN / NOT YET FROZEN."
    )

    lines.append(
        "No prospective WF2 observation has been created."
    )


    # ========================================================
    # WRITE OUTPUTS
    # ========================================================

    OUT.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


    summary = {
        "phase":
            "WF2_PHASE0B_AUTHORITY_RESOLUTION",

        "wf1b_zip":
            str(ZIP_PATH),

        "wf1b_zip_sha256":
            sha256_file(ZIP_PATH),

        "loose_detector_package_file_count":
            len(loose_inventory),

        "frozen_models":
            frozen_models,

        "exact_sha256_matches_to_wf1b_zip":
            exact_match_count,

        "basename_matches_with_different_hash":
            basename_only_no_hash_match,

        "loose_files_without_zip_basename_match":
            no_basename_match,

        "prospective_source_set": {
            "path":
                str(PAIRSET),

            "sha256":
                sha256_file(PAIRSET),

            "nonempty_rows":
                nonempty_rows,

            "valid_json_rows":
                valid_json_rows,

            "invalid_json_rows":
                invalid_json_rows
        },

        "scientific_observations_created":
            False,

        "detector_training_performed":
            False,

        "threshold_tuning_performed":
            False,

        "boundary_1_reached":
            False
    }


    JSONOUT.write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )


    if ERRORLOG.exists():
        ERRORLOG.unlink()


    print(
        OUT.read_text(
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
    print("WF2 PHASE 0B ERROR")
    print("=" * 88)
    print("")
    print(err)
    print("")
    print(f"ERROR LOG: {ERRORLOG}")

    sys.exit(2)
