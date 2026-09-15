import csv
import json
import re
from pathlib import Path
from collections import Counter

ROOT = Path(r"C:\RSOS")
WF3 = ROOT / "RSOS_EXPERIMENTS_WF3_GENEALOGY_POPULATION_CONFIRMATION"

CONTROL = WF3 / "00_CONTROL"
PROTOCOL = WF3 / "01_PROTOCOL"
DIAG = WF3 / "06_DIAGNOSTIC"

for d in [CONTROL, PROTOCOL, DIAG]:
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# SCOPE
# ------------------------------------------------------------

UPSTREAM_SCOPE = re.compile(
    r"(?i)(F1|F2|V11|V12|WF1|WF2)"
)

AUTHORITY_FILENAME = re.compile(
    r"(?i)"
    r"(adjudic|freeze|status|final|complete|authority|canonical|"
    r"manifest|summary|prereg|protocol|valid|invalid|supersed|"
    r"failure|audit|development)"
)

ALLOWED_EXT = {
    ".json", ".txt", ".md", ".csv", ".yaml", ".yml"
}

MAX_SIZE = 10 * 1024 * 1024

# ------------------------------------------------------------
# CLASSIFICATION SIGNALS
# ------------------------------------------------------------

DEV_AUDIT = [
    r"\bdevelopment[\s_-]*only\b",
    r"\baudit[\s_-]*only\b",
    r"\bdevelopment[\s_-]*audit\b",
    r"\bdev[\s_-]*only\b",
    r"\bfor[\s_-]*audit\b",
]

INVALID = [
    r"\binvalidated\b",
    r"\bmethodological[\s_-]*failure\b",
    r"\bmethodology[\s_-]*failed\b",
    r"\bsuperseded\b",
    r"\bfailed[\s_-]*method",
    r"\bincomplete\b",
    r"\baborted\b",
    r"\bnot[\s_-]*authoritative\b",
    r"\bdo[\s_-]*not[\s_-]*use\b",
    r"\bblocked\b",
    r"\binvalid\b",
]

POSITIVE = [
    r"\bfrozen\b",
    r"\bfinal\b",
    r"\bcomplete\b",
    r"\bvalid\b",
    r"\bauthoritative\b",
    r"\bcanonical\b",
    r"\baccepted\b",
]

MIXED = [
    r"\bmixed\b",
    r"\bpartial\b",
    r"\bqualified\b",
]

DEV_RE = re.compile("|".join(DEV_AUDIT), re.I)
INVALID_RE = re.compile("|".join(INVALID), re.I)
POSITIVE_RE = re.compile("|".join(POSITIVE), re.I)
MIXED_RE = re.compile("|".join(MIXED), re.I)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def safe_read(path: Path):
    try:
        return path.read_text(
            encoding="utf-8-sig",
            errors="replace",
        )
    except Exception:
        return ""


def excerpt(text, pattern, radius=180):
    m = pattern.search(text)

    if not m:
        return ""

    lo = max(0, m.start() - radius)
    hi = min(len(text), m.end() + radius)

    s = text[lo:hi]
    s = re.sub(r"\s+", " ", s)

    return s[:500]


def source_family(path: Path):
    s = str(path)

    tests = [
        ("WF1-P0", r"(?i)WF1[_\-]?P0"),
        ("WF1-A",  r"(?i)WF1[_\-]?A"),
        ("WF1-B",  r"(?i)WF1[_\-]?B"),
        ("WF2",    r"(?i)WF2"),
        ("V12",    r"(?i)V12"),
        ("V11",    r"(?i)V11"),
        ("F2",     r"(?i)F2"),
        ("F1",     r"(?i)F1"),
    ]

    for name, pattern in tests:
        if re.search(pattern, s):
            return name

    return "UNRESOLVED_UPSTREAM"


def classify(path, text):
    combined = f"{path.name}\n{text}"

    dev = bool(DEV_RE.search(combined))
    invalid = bool(INVALID_RE.search(combined))
    positive_hits = len(POSITIVE_RE.findall(combined))
    mixed = bool(MIXED_RE.search(combined))

    # Development/audit artifacts are never primary authority.
    if dev:
        return (
            "DEVELOPMENT_AUDIT_ONLY",
            "development/audit-only signal found",
        )

    # Explicit failure/supersession outranks positive words.
    if invalid:
        return (
            "INVALIDATED_OR_SUPERSEDED",
            "failure/incomplete/superseded signal found",
        )

    # Strong positive authority signal.
    if positive_hits >= 2:
        return (
            "AUTHORITATIVE_CANDIDATE",
            f"{positive_hits} positive authority signals",
        )

    if positive_hits >= 1 and re.search(
        r"(?i)(freeze|adjudic|final|status|complete)",
        path.name,
    ):
        return (
            "AUTHORITATIVE_CANDIDATE",
            "authority filename plus positive status signal",
        )

    if mixed:
        return (
            "HISTORICAL_REFERENCE_ONLY",
            "mixed/qualified status without explicit invalidation",
        )

    return (
        "HISTORICAL_REFERENCE_ONLY",
        "insufficient evidence for scientific authority",
    )


# ------------------------------------------------------------
# DISCOVER ONLY AUTHORITY / STATUS DOCUMENTS
# ------------------------------------------------------------

candidate_files = []

for p in ROOT.rglob("*"):

    try:
        if not p.is_file():
            continue

        # Never inspect WF3 itself as upstream authority.
        if str(p).lower().startswith(str(WF3).lower()):
            continue

        if p.suffix.lower() not in ALLOWED_EXT:
            continue

        if p.stat().st_size > MAX_SIZE:
            continue

        full = str(p)

        if not UPSTREAM_SCOPE.search(full):
            continue

        if not AUTHORITY_FILENAME.search(p.name):
            continue

        candidate_files.append(p)

    except Exception:
        continue

# ------------------------------------------------------------
# CLASSIFY DOCUMENTS
# ------------------------------------------------------------

rows = []

for p in candidate_files:

    text = safe_read(p)

    status, reason = classify(p, text)

    invalid_excerpt = excerpt(text, INVALID_RE)
    dev_excerpt = excerpt(text, DEV_RE)
    positive_excerpt = excerpt(text, POSITIVE_RE)
    mixed_excerpt = excerpt(text, MIXED_RE)

    rows.append(
        {
            "SOURCE_FAMILY": source_family(p),
            "CLASSIFICATION": status,
            "REASON": reason,
            "FILE_NAME": p.name,
            "FULL_PATH": str(p),
            "SIZE_BYTES": p.stat().st_size,
            "INVALID_SIGNAL_EXCERPT": invalid_excerpt,
            "DEVELOPMENT_SIGNAL_EXCERPT": dev_excerpt,
            "POSITIVE_SIGNAL_EXCERPT": positive_excerpt,
            "MIXED_SIGNAL_EXCERPT": mixed_excerpt,
        }
    )

# ------------------------------------------------------------
# WRITE FULL REGISTRY
# ------------------------------------------------------------

registry_path = CONTROL / "WF3_UPSTREAM_AUTHORITY_REGISTRY.csv"

fieldnames = [
    "SOURCE_FAMILY",
    "CLASSIFICATION",
    "REASON",
    "FILE_NAME",
    "FULL_PATH",
    "SIZE_BYTES",
    "INVALID_SIGNAL_EXCERPT",
    "DEVELOPMENT_SIGNAL_EXCERPT",
    "POSITIVE_SIGNAL_EXCERPT",
    "MIXED_SIGNAL_EXCERPT",
]

with registry_path.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(rows)

# ------------------------------------------------------------
# FAMILY-LEVEL SUMMARY
# ------------------------------------------------------------

families = sorted(
    set(r["SOURCE_FAMILY"] for r in rows)
)

family_rows = []

for family in families:

    subset = [
        r for r in rows
        if r["SOURCE_FAMILY"] == family
    ]

    counts = Counter(
        r["CLASSIFICATION"]
        for r in subset
    )

    family_rows.append(
        {
            "SOURCE_FAMILY": family,
            "AUTHORITATIVE_CANDIDATE": counts.get(
                "AUTHORITATIVE_CANDIDATE", 0
            ),
            "HISTORICAL_REFERENCE_ONLY": counts.get(
                "HISTORICAL_REFERENCE_ONLY", 0
            ),
            "DEVELOPMENT_AUDIT_ONLY": counts.get(
                "DEVELOPMENT_AUDIT_ONLY", 0
            ),
            "INVALIDATED_OR_SUPERSEDED": counts.get(
                "INVALIDATED_OR_SUPERSEDED", 0
            ),
            "STATUS": (
                "REQUIRES_AUTHORITY_RESOLUTION"
                if (
                    counts.get(
                        "INVALIDATED_OR_SUPERSEDED", 0
                    ) > 0
                    or counts.get(
                        "DEVELOPMENT_AUDIT_ONLY", 0
                    ) > 0
                    or counts.get(
                        "AUTHORITATIVE_CANDIDATE", 0
                    ) == 0
                )
                else "AUTHORITY_CANDIDATES_PRESENT"
            ),
        }
    )

family_path = CONTROL / "WF3_UPSTREAM_AUTHORITY_FAMILY_SUMMARY.csv"

with family_path.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(family_rows[0].keys())
        if family_rows
        else ["SOURCE_FAMILY"],
    )

    writer.writeheader()

    if family_rows:
        writer.writerows(family_rows)

# ------------------------------------------------------------
# EXPLICITLY UNRESOLVED AUTHORITY CONFLICTS
# ------------------------------------------------------------

unresolved = []

for family in families:

    subset = [
        r for r in rows
        if r["SOURCE_FAMILY"] == family
    ]

    classes = set(
        r["CLASSIFICATION"]
        for r in subset
    )

    # Any family containing both authoritative-looking records and
    # invalid/superseded/development records MUST be resolved before reuse.
    if (
        "AUTHORITATIVE_CANDIDATE" in classes
        and (
            "INVALIDATED_OR_SUPERSEDED" in classes
            or "DEVELOPMENT_AUDIT_ONLY" in classes
        )
    ):
        for r in subset:
            unresolved.append(r)

    elif "AUTHORITATIVE_CANDIDATE" not in classes:
        for r in subset:
            unresolved.append(r)

unresolved_path = DIAG / "WF3_AUTHORITY_UNRESOLVED.csv"

with unresolved_path.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(unresolved)

# ------------------------------------------------------------
# BOUNDARY-1 AUTHORITY ADDENDUM DRAFT
#
# This does NOT rewrite Boundary 1.
# It constrains which already-existing upstream artifacts may be reused.
# ------------------------------------------------------------

addendum = {
    "experiment": "WF3",
    "document": "BOUNDARY_1_AUTHORITY_ELIGIBILITY_ADDENDUM_DRAFT",
    "boundary_1_remains_frozen": True,
    "scientific_outcomes_inspected": False,
    "purpose": (
        "Prevent invalidated, superseded, development-only, audit-only, "
        "or non-authoritative upstream runs from entering WF3 confirmatory calculations."
    ),
    "authority_classes": {
        "AUTHORITATIVE_REUSE": (
            "Eligible only after exact upstream run/path is resolved "
            "to a pre-existing final/frozen/adjudicated authority."
        ),
        "HISTORICAL_REFERENCE_ONLY": (
            "May be preserved for chronology/provenance but cannot enter "
            "primary V15/V16 calculation."
        ),
        "DEVELOPMENT_AUDIT_ONLY": (
            "May be used for debugging/history only; prohibited from primary confirmation."
        ),
        "INVALIDATED_OR_SUPERSEDED": (
            "Must never enter WF3 scientific inference."
        ),
    },
    "hard_rule": (
        "No file becomes AUTHORITATIVE_REUSE merely because it exists in an upstream folder."
    ),
    "current_registry": str(registry_path),
    "unresolved_registry": str(unresolved_path),
}

addendum_path = (
    PROTOCOL
    / "WF3_BOUNDARY1_AUTHORITY_ELIGIBILITY_ADDENDUM_DRAFT.json"
)

addendum_path.write_text(
    json.dumps(addendum, indent=2),
    encoding="utf-8",
)

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

class_counts = Counter(
    r["CLASSIFICATION"]
    for r in rows
)

summary = {
    "stage": "WF3_UPSTREAM_AUTHORITY_DISCOVERY",
    "scientific_outcomes_inspected": False,
    "boundary_1_remains_frozen": True,
    "authority_documents_found": len(rows),
    "class_counts": dict(class_counts),
    "source_families": families,
    "unresolved_documents": len(unresolved),
    "status": (
        "AUTHORITY_RESOLUTION_REQUIRED"
        if unresolved
        else "AUTHORITY_REGISTRY_CLEAN"
    ),
}

summary_path = DIAG / "WF3_AUTHORITY_DISCOVERY_SUMMARY.json"

summary_path.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print()
print("============================================================")
print(" WF3 UPSTREAM AUTHORITY DISCOVERY COMPLETE")
print("============================================================")
print("Scientific outcomes inspected : NO")
print("Boundary 1 remains frozen      : YES")
print()
print("Authority documents found      :", len(rows))
print(
    "Authoritative candidates      :",
    class_counts.get(
        "AUTHORITATIVE_CANDIDATE", 0
    ),
)
print(
    "Historical reference only     :",
    class_counts.get(
        "HISTORICAL_REFERENCE_ONLY", 0
    ),
)
print(
    "Development / audit only      :",
    class_counts.get(
        "DEVELOPMENT_AUDIT_ONLY", 0
    ),
)
print(
    "Invalidated / superseded      :",
    class_counts.get(
        "INVALIDATED_OR_SUPERSEDED", 0
    ),
)
print("Unresolved authority docs      :", len(unresolved))
print()
print("FAMILY SUMMARY:")
for r in family_rows:
    print(
        " ",
        r["SOURCE_FAMILY"],
        "| authoritative:",
        r["AUTHORITATIVE_CANDIDATE"],
        "| historical:",
        r["HISTORICAL_REFERENCE_ONLY"],
        "| dev/audit:",
        r["DEVELOPMENT_AUDIT_ONLY"],
        "| invalid/superseded:",
        r["INVALIDATED_OR_SUPERSEDED"],
        "|",
        r["STATUS"],
    )
print()
print("NEXT STATUS:")
print(" ", summary["status"])
print()
print("REGISTRY:")
print(" ", registry_path)
print()
print("UNRESOLVED:")
print(" ", unresolved_path)
print("============================================================")
