import csv
import json
import re
from pathlib import Path

ROOT = Path(r"C:\RSOS")
WF3 = ROOT / "RSOS_EXPERIMENTS_WF3_GENEALOGY_POPULATION_CONFIRMATION"

CONTROL = WF3 / "00_CONTROL"
PROTOCOL = WF3 / "01_PROTOCOL"
DIAG = WF3 / "06_DIAGNOSTIC"

for d in [CONTROL, PROTOCOL, DIAG]:
    d.mkdir(parents=True, exist_ok=True)


anchors = []
exclusions = []


def add_anchor(family, role, path, authority):
    p = Path(path)

    anchors.append({
        "SOURCE_FAMILY": family,
        "ROLE": role,
        "AUTHORITY": authority,
        "PATH": str(p),
        "EXISTS": p.exists(),
        "TYPE": (
            "DIRECTORY"
            if p.exists() and p.is_dir()
            else "FILE"
            if p.exists()
            else "NOT_FOUND"
        ),
    })


def add_exclusion(family, pattern, reason):
    exclusions.append({
        "SOURCE_FAMILY": family,
        "EXCLUSION_PATTERN": pattern,
        "REASON": reason,
    })


# ============================================================
# 1. FIXED AUTHORITIES RESOLVED FROM EXISTING FREEZE RECORDS
# ============================================================

add_anchor(
    "F1",
    "FINAL_AUTHORITY_ROOT",
    r"C:\RSOS\F1_FINAL_LOCAL\F1_FINAL_FREEZE",
    "AUTHORITATIVE_REUSE",
)

add_anchor(
    "V11",
    "CLEAN_REPAIRED_AUTHORITY",
    r"C:\RSOS\V11_STRUCTURAL_METRIC_REPAIR\V6_CLEAN",
    "AUTHORITATIVE_REUSE",
)

add_anchor(
    "V12",
    "FINAL_FREEZE_ROOT",
    r"C:\RSOS\V12_FINAL_FREEZE",
    "AUTHORITATIVE_REUSE",
)

add_anchor(
    "WF1-P0",
    "COMPLETED_EXPERIMENT_ROOT",
    r"C:\RSOS\RSOS_EXPERIMENTS_WF1_P0_HISTORICAL_CORPUS_BRIDGE",
    "AUTHORITATIVE_REUSE",
)

add_anchor(
    "WF1-A",
    "CORRECTIVE_EXPERIMENT_ROOT",
    r"C:\RSOS\RSOS_EXPERIMENTS_WF1_A_V12_R_CORRECTIVE_CAUSAL_TOPOLOGY",
    "AUTHORITATIVE_REUSE_WITH_FINAL_BRANCH_FILTER",
)

add_anchor(
    "WF1-B",
    "FINAL_AUTHORITY_R3",
    r"C:\RSOS\RSOS_EXPERIMENTS_WF1_A_V12_R_CORRECTIVE_CAUSAL_TOPOLOGY\WF1_B_EXTERNAL_POPULATION_PARQUET\09_ADJUDICATION\WF1_B_FINAL_AUTHORITY_R3",
    "AUTHORITATIVE_REUSE",
)

add_anchor(
    "WF1-B",
    "FRESH_CONFIRMATION_R3",
    r"C:\RSOS\RSOS_EXPERIMENTS_WF1_A_V12_R_CORRECTIVE_CAUSAL_TOPOLOGY\WF1_B_EXTERNAL_POPULATION_PARQUET\07_OUTPUT\FRESH_SHARD14_CONFIRMATION_R3",
    "AUTHORITATIVE_REUSE_IF_REFERENCED_BY_FINAL_AUTHORITY",
)

add_anchor(
    "WF2",
    "FINAL_FREEZE_ROOT",
    r"C:\RSOS\RSOS_EXPERIMENTS_WF2_PROSPECTIVE_CROSS_SYSTEM_CONFIRMATION\08_FINAL_FREEZE",
    "AUTHORITATIVE_REUSE",
)


# ============================================================
# 2. EXPLICIT EXCLUSIONS
# ============================================================

add_exclusion(
    "V11",
    r"(?i)\\V6_ORIGINAL\\",
    "Superseded by V6_CLEAN structural metric repair.",
)

add_exclusion(
    "F2",
    r"(?i)F2.?A",
    "F2-A was methodologically incomplete and cannot be promoted to WF3 authority.",
)

add_exclusion(
    "ANY",
    r"(?i)(development.?only|audit.?only|dev.?only)",
    "Development/audit-only artifacts are prohibited from primary WF3 confirmation.",
)

add_exclusion(
    "ANY",
    r"(?i)(superseded|invalidated|methodological.?failure|incomplete|aborted)",
    "Invalidated or superseded artifacts are historical reference only.",
)

add_exclusion(
    "WF2",
    r"(?i)\\PRECONSOLIDATION\\",
    "Preconsolidation material cannot override WF2 final freeze.",
)


# ============================================================
# 3. AUTOMATIC F2 ROLE RESOLUTION
# ============================================================

f2_candidates = []

for p in ROOT.rglob("*"):

    try:
        if not p.is_file():
            continue

        if str(p).lower().startswith(str(WF3).lower()):
            continue

        s = str(p)

        if not re.search(r"(?i)F2", s):
            continue

        role = None
        priority = 0

        if re.search(
            r"(?i)F2_BR2_DETERMINISTIC_CORPUS_FINGERPRINT_REFREEZE_COMPLETE",
            s,
        ):
            role = "BR2_DETERMINISTIC_REFREEZE"
            priority = 100

        elif (
            re.search(r"(?i)BR2", s)
            and re.search(r"(?i)(REFREEZE|FROZEN|FINAL)", s)
        ):
            role = "BR2_FROZEN_REPRESENTATION"
            priority = 90

        elif (
            re.search(r"(?i)F2.?B", s)
            and re.search(r"(?i)(FINAL|CORRECTED|FREEZE|ADJUDIC)", s)
        ):
            role = "F2_B_CORRECTED_EXTERNAL_RARITY"
            priority = 80

        elif (
            re.search(r"(?i)F2.?R", s)
            and re.search(r"(?i)(FINAL|FREEZE|COMPLETE|ADJUDIC)", s)
        ):
            role = "F2_R_REBUILD"
            priority = 75

        elif (
            re.search(r"(?i)F2", s)
            and re.search(r"(?i)(FINAL|FREEZE|REFREEZE|CORRECTED)", s)
        ):
            role = "F2_OTHER_FINAL_CANDIDATE"
            priority = 50

        if role is None:
            continue

        # Never promote F2-A.
        if re.search(r"(?i)F2.?A", s):
            continue

        # Never promote explicit dev/audit-only assets.
        if re.search(
            r"(?i)(development.?only|audit.?only|dev.?only)",
            s,
        ):
            continue

        f2_candidates.append({
            "ROLE": role,
            "PRIORITY": priority,
            "FILE_NAME": p.name,
            "FULL_PATH": s,
            "SIZE_BYTES": p.stat().st_size,
        })

    except Exception:
        continue


f2_candidates.sort(
    key=lambda x: (
        -x["PRIORITY"],
        x["FULL_PATH"].lower(),
    )
)


# ============================================================
# 4. PROMOTE ONLY UNAMBIGUOUS F2 AUTHORITIES
# ============================================================

by_role = {}

for row in f2_candidates:
    by_role.setdefault(
        row["ROLE"],
        []
    ).append(row)


def promote_best(role, authority_name):
    candidates = by_role.get(role, [])

    if not candidates:
        return

    best = candidates[0]

    add_anchor(
        "F2",
        authority_name,
        best["FULL_PATH"],
        "AUTHORITATIVE_REUSE",
    )


promote_best(
    "BR2_DETERMINISTIC_REFREEZE",
    "DETERMINISTIC_REPRESENTATION_AUTHORITY",
)

# If exact BR2 archive was absent, allow the strongest frozen BR2 representation.
if not any(
    a["SOURCE_FAMILY"] == "F2"
    and a["ROLE"] == "DETERMINISTIC_REPRESENTATION_AUTHORITY"
    for a in anchors
):
    promote_best(
        "BR2_FROZEN_REPRESENTATION",
        "DETERMINISTIC_REPRESENTATION_AUTHORITY",
    )


promote_best(
    "F2_B_CORRECTED_EXTERNAL_RARITY",
    "CORRECTED_EXTERNAL_RARITY_AUTHORITY",
)

promote_best(
    "F2_R_REBUILD",
    "ERROR_CORRECTED_REBUILD_AUTHORITY",
)


# ============================================================
# 5. VALIDATE REQUIRED AUTHORITY FAMILIES
# ============================================================

required = {
    "F1",
    "F2",
    "V11",
    "V12",
    "WF1-P0",
    "WF1-A",
    "WF1-B",
    "WF2",
}

present = {
    a["SOURCE_FAMILY"]
    for a in anchors
    if a["EXISTS"]
}

missing = sorted(required - present)


# F2 specifically requires deterministic representation authority.
f2_rep = [
    a for a in anchors
    if (
        a["SOURCE_FAMILY"] == "F2"
        and a["ROLE"] == "DETERMINISTIC_REPRESENTATION_AUTHORITY"
        and a["EXISTS"]
    )
]

if not f2_rep:
    missing.append(
        "F2_DETERMINISTIC_REPRESENTATION_AUTHORITY"
    )


# ============================================================
# 6. WRITE REGISTRIES
# ============================================================

anchor_path = CONTROL / "WF3_AUTHORITATIVE_REUSE_ANCHORS.csv"

with anchor_path.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    fields = [
        "SOURCE_FAMILY",
        "ROLE",
        "AUTHORITY",
        "PATH",
        "EXISTS",
        "TYPE",
    ]

    w = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    w.writeheader()
    w.writerows(anchors)


exclusion_path = CONTROL / "WF3_AUTHORITY_EXCLUSION_RULES.csv"

with exclusion_path.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    fields = [
        "SOURCE_FAMILY",
        "EXCLUSION_PATTERN",
        "REASON",
    ]

    w = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    w.writeheader()
    w.writerows(exclusions)


f2_path = DIAG / "WF3_F2_AUTHORITY_CANDIDATES.csv"

with f2_path.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    fields = [
        "ROLE",
        "PRIORITY",
        "FILE_NAME",
        "FULL_PATH",
        "SIZE_BYTES",
    ]

    w = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    w.writeheader()
    w.writerows(f2_candidates)


# ============================================================
# 7. AUTHORITY ELIGIBILITY ADDENDUM
# ============================================================

addendum = {
    "experiment": "WF3",
    "document": "BOUNDARY_1_AUTHORITY_ELIGIBILITY_ADDENDUM",
    "boundary_1_original_preregistration_modified": False,
    "scientific_outcomes_inspected": False,

    "purpose": (
        "Narrow the Boundary-1 candidate input universe using "
        "pre-existing upstream validity and supersession status."
    ),

    "hard_rules": [
        "File existence does not imply scientific authority.",
        "Invalidated or superseded outputs cannot enter V15/V16.",
        "Development/audit-only outputs cannot enter primary confirmation.",
        "V11 V6_ORIGINAL is excluded in favor of V6_CLEAN.",
        "F2-A is excluded because it was methodologically incomplete.",
        "WF2 preconsolidation artifacts cannot override 08_FINAL_FREEZE.",
        "WF1-B is resolved independently even though its files are physically nested inside the WF1-A experiment estate.",
        "Only artifacts reachable from an authoritative anchor may enter the next standardization stage.",
    ],

    "authority_anchor_registry":
        str(anchor_path),

    "authority_exclusion_registry":
        str(exclusion_path),

    "status": (
        "READY_TO_FREEZE_AUTHORITY_ALLOWLIST"
        if not missing
        else "BLOCKED_BY_MISSING_AUTHORITY"
    ),

    "missing_authorities":
        missing,
}

addendum_path = (
    PROTOCOL
    / "WF3_BOUNDARY1_AUTHORITY_ELIGIBILITY_ADDENDUM.json"
)

addendum_path.write_text(
    json.dumps(
        addendum,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# 8. OUTPUT
# ============================================================

print()
print("============================================================")
print(" WF3 AUTHORITY ANCHOR RESOLUTION COMPLETE")
print("============================================================")
print("Scientific outcomes inspected : NO")
print("Boundary 1 rewritten           : NO")
print()

for a in anchors:
    print(
        f'{a["SOURCE_FAMILY"]:7s} | '
        f'{a["ROLE"]:42s} | '
        f'EXISTS={a["EXISTS"]}'
    )
    print(
        "          "
        + a["PATH"]
    )

print()
print("F2 CANDIDATES FOUND:", len(f2_candidates))

print()

if missing:
    print("MISSING REQUIRED AUTHORITIES:")
    for x in missing:
        print("  -", x)

    print()
    print("NEXT STATUS:")
    print("  AUTHORITY_ALLOWLIST_BLOCKED")

else:
    print("MISSING REQUIRED AUTHORITIES: NONE")
    print()
    print("NEXT STATUS:")
    print("  READY_TO_FREEZE_AUTHORITY_ALLOWLIST")

print()
print("ANCHORS:")
print(" ", anchor_path)
print()
print("EXCLUSIONS:")
print(" ", exclusion_path)
print()
print("F2 RESOLUTION:")
print(" ", f2_path)
print("============================================================")
