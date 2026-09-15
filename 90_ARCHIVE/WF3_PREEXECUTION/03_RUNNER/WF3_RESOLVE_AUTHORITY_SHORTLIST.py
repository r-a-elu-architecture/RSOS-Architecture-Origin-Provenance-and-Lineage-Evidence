import csv
import json
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"C:\RSOS")
WF3 = ROOT / "RSOS_EXPERIMENTS_WF3_GENEALOGY_POPULATION_CONFIRMATION"

CONTROL = WF3 / "00_CONTROL"
DIAG = WF3 / "06_DIAGNOSTIC"

REGISTRY = CONTROL / "WF3_UPSTREAM_AUTHORITY_REGISTRY.csv"

if not REGISTRY.exists():
    raise SystemExit(f"Missing authority registry: {REGISTRY}")


def read_csv(path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


rows = read_csv(REGISTRY)

# ------------------------------------------------------------
# AUTHORITY SCORE
# ------------------------------------------------------------

POSITIVE_NAME = [
    (r"(?i)\bFINAL\b", 8),
    (r"(?i)\bCOMPLETE\b", 8),
    (r"(?i)\bFREEZE\b", 7),
    (r"(?i)\bADJUDIC", 7),
    (r"(?i)\bCORRECTED\b", 6),
    (r"(?i)\bREFREEZE\b", 8),
    (r"(?i)\bRERUN\b", 5),
    (r"(?i)\bFROZEN\b", 7),
    (r"(?i)\bAUTHORIT", 6),
    (r"(?i)\bCANONICAL\b", 6),
    (r"(?i)\bSUMMARY\b", 2),
    (r"(?i)\bSTATUS\b", 2),
]

NEGATIVE_NAME = [
    (r"(?i)\bDEV\b", -12),
    (r"(?i)\bDEVELOPMENT\b", -12),
    (r"(?i)\bAUDIT\b", -8),
    (r"(?i)\bOLD\b", -8),
    (r"(?i)\bBACKUP\b", -6),
    (r"(?i)\bTEMP\b", -8),
    (r"(?i)\bFAILED\b", -20),
    (r"(?i)\bINVALID\b", -20),
    (r"(?i)\bSUPERSEDED\b", -20),
    (r"(?i)\bINCOMPLETE\b", -18),
    (r"(?i)\bABORT", -18),
    (r"(?i)\bDRAFT\b", -8),
]


def authority_score(row):
    text = (
        row.get("FILE_NAME", "")
        + " "
        + row.get("FULL_PATH", "")
    )

    score = 0

    for pattern, weight in POSITIVE_NAME:
        if re.search(pattern, text):
            score += weight

    for pattern, weight in NEGATIVE_NAME:
        if re.search(pattern, text):
            score += weight

    cls = row.get("CLASSIFICATION", "")

    if cls == "AUTHORITATIVE_CANDIDATE":
        score += 10

    elif cls == "INVALIDATED_OR_SUPERSEDED":
        score -= 30

    elif cls == "DEVELOPMENT_AUDIT_ONLY":
        score -= 20

    # Prefer deeper experiment-specific files over generic root summaries.
    parts = Path(row.get("FULL_PATH", "")).parts
    score += min(len(parts), 12) * 0.2

    return score


# ------------------------------------------------------------
# SHORTLIST BY FAMILY
# ------------------------------------------------------------

families = [
    "F1",
    "F2",
    "V11",
    "V12",
    "WF1-P0",
    "WF1-A",
    "WF1-B",
    "WF2",
]

grouped = defaultdict(list)

for row in rows:
    family = row.get("SOURCE_FAMILY", "")
    grouped[family].append(row)

shortlist = []

for family in families:

    candidates = [
        r for r in grouped.get(family, [])
        if r.get("CLASSIFICATION")
        == "AUTHORITATIVE_CANDIDATE"
    ]

    ranked = sorted(
        candidates,
        key=authority_score,
        reverse=True,
    )

    for rank, row in enumerate(ranked[:12], 1):

        shortlist.append(
            {
                "SOURCE_FAMILY": family,
                "RANK": rank,
                "SCORE": authority_score(row),
                "FILE_NAME": row.get("FILE_NAME", ""),
                "FULL_PATH": row.get("FULL_PATH", ""),
                "POSITIVE_EXCERPT": row.get(
                    "POSITIVE_SIGNAL_EXCERPT", ""
                ),
            }
        )

# ------------------------------------------------------------
# CONFLICT / EXCLUSION DOCUMENTS
# ------------------------------------------------------------

conflicts = []

for row in rows:

    if row.get("CLASSIFICATION") in {
        "INVALIDATED_OR_SUPERSEDED",
        "DEVELOPMENT_AUDIT_ONLY",
    }:
        conflicts.append(
            {
                "SOURCE_FAMILY":
                    row.get("SOURCE_FAMILY", ""),
                "CLASSIFICATION":
                    row.get("CLASSIFICATION", ""),
                "FILE_NAME":
                    row.get("FILE_NAME", ""),
                "FULL_PATH":
                    row.get("FULL_PATH", ""),
                "INVALID_EXCERPT":
                    row.get(
                        "INVALID_SIGNAL_EXCERPT", ""
                    ),
                "DEVELOPMENT_EXCERPT":
                    row.get(
                        "DEVELOPMENT_SIGNAL_EXCERPT", ""
                    ),
            }
        )

# ------------------------------------------------------------
# TARGETED WF1-B DISCOVERY
# ------------------------------------------------------------

wf1b_hits = []

patterns = [
    r"(?i)WF1[_\- ]?B",
    r"(?i)WF1.*EXTERNAL",
    r"(?i)EXTERNAL.*WF1",
    r"(?i)POPULATION.*WF1",
]

for p in ROOT.rglob("*"):

    try:
        if not p.is_file():
            continue

        if str(p).lower().startswith(str(WF3).lower()):
            continue

        s = str(p)

        if not any(
            re.search(pattern, s)
            for pattern in patterns
        ):
            continue

        wf1b_hits.append(
            {
                "FILE_NAME": p.name,
                "FULL_PATH": s,
                "SIZE_BYTES": p.stat().st_size,
            }
        )

    except Exception:
        continue

# ------------------------------------------------------------
# KNOWN CORRECTED / FINAL MARKER SEARCH
# ------------------------------------------------------------

marker_patterns = [
    r"(?i)BR2",
    r"(?i)DETERMINISTIC.*REFREEZE",
    r"(?i)CORRECTED",
    r"(?i)ERROR.?CORRECTED",
    r"(?i)FINAL_COMPLETE",
    r"(?i)COMPLETE_LOCAL",
    r"(?i)FREEZE_RECORD",
    r"(?i)FINAL_ADJUDICATION",
    r"(?i)FINAL_SUMMARY",
]

marker_hits = []

for p in ROOT.rglob("*"):

    try:
        if not p.is_file():
            continue

        if str(p).lower().startswith(str(WF3).lower()):
            continue

        s = str(p)

        matched = [
            pattern
            for pattern in marker_patterns
            if re.search(pattern, s)
        ]

        if matched:
            marker_hits.append(
                {
                    "MATCH":
                        "|".join(matched),
                    "FILE_NAME":
                        p.name,
                    "FULL_PATH":
                        s,
                    "SIZE_BYTES":
                        p.stat().st_size,
                }
            )

    except Exception:
        continue

# ------------------------------------------------------------
# WRITE OUTPUTS
# ------------------------------------------------------------

shortlist_path = CONTROL / "WF3_AUTHORITY_SHORTLIST.csv"

with shortlist_path.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    fields = [
        "SOURCE_FAMILY",
        "RANK",
        "SCORE",
        "FILE_NAME",
        "FULL_PATH",
        "POSITIVE_EXCERPT",
    ]

    w = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    w.writeheader()
    w.writerows(shortlist)


conflict_path = DIAG / "WF3_AUTHORITY_CONFLICTS.csv"

with conflict_path.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    fields = [
        "SOURCE_FAMILY",
        "CLASSIFICATION",
        "FILE_NAME",
        "FULL_PATH",
        "INVALID_EXCERPT",
        "DEVELOPMENT_EXCERPT",
    ]

    w = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    w.writeheader()
    w.writerows(conflicts)


wf1b_path = DIAG / "WF3_WF1B_TARGETED_DISCOVERY.csv"

with wf1b_path.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    fields = [
        "FILE_NAME",
        "FULL_PATH",
        "SIZE_BYTES",
    ]

    w = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    w.writeheader()
    w.writerows(wf1b_hits)


marker_path = DIAG / "WF3_CORRECTED_FINAL_MARKER_HITS.csv"

with marker_path.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    fields = [
        "MATCH",
        "FILE_NAME",
        "FULL_PATH",
        "SIZE_BYTES",
    ]

    w = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    w.writeheader()
    w.writerows(marker_hits)

# ------------------------------------------------------------
# PRINT COMPACT REVIEW
# ------------------------------------------------------------

print()
print("============================================================")
print(" WF3 AUTHORITY SHORTLIST COMPLETE")
print("============================================================")
print("Scientific outcomes inspected : NO")
print("Boundary 1 remains frozen      : YES")
print()

for family in families:

    subset = [
        r for r in shortlist
        if r["SOURCE_FAMILY"] == family
    ]

    print("------------------------------------------------------------")
    print(f"{family}")
    print("------------------------------------------------------------")

    if not subset:
        print("  NO AUTHORITATIVE CANDIDATE RESOLVED")
        continue

    for r in subset[:8]:

        print(
            f'  [{r["RANK"]}] score={r["SCORE"]:.1f}'
        )

        print(
            "      "
            + r["FULL_PATH"]
        )

print()
print("------------------------------------------------------------")
print("WF1-B TARGETED HITS:", len(wf1b_hits))
print("CONFLICT / EXCLUSION DOCS:", len(conflicts))
print("CORRECTED / FINAL MARKER HITS:", len(marker_hits))
print("------------------------------------------------------------")
print()
print("SHORTLIST:")
print(" ", shortlist_path)
print()
print("CONFLICTS:")
print(" ", conflict_path)
print()
print("WF1-B DISCOVERY:")
print(" ", wf1b_path)
print()
print("CORRECTED/FINAL MARKERS:")
print(" ", marker_path)
print()
print("NEXT STATUS:")
print("  MANUAL_AUTHORITY_ADJUDICATION_FROM_METADATA_REQUIRED")
print("============================================================")
