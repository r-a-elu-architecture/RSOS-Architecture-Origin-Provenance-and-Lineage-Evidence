from pathlib import Path
import hashlib
import zipfile
import json
import re

ROOT = Path(
    r"C:\RSOS\RSOS_EXPERIMENTS_WF2_PROSPECTIVE_CROSS_SYSTEM_CONFIRMATION"
)

LEGACY = (
    ROOT
    / "00_CONTROL"
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

OUT = (
    ROOT
    / "00_CONTROL"
    / "WF1B_VS_PRECONSOLIDATION_AUTHORITY_CHECK.txt"
)


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def relevant(name):
    u = name.upper()

    terms = [
        "FINAL",
        "AUTHORITY",
        "ADJUDICATION",
        "SUMMARY",
        "DETECTOR",
        "MODEL",
        "JOBLIB",
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
        "JOINT"
    ]

    return any(t in u for t in terms)


lines = []

lines.append("=" * 80)
lines.append("WF2 PHASE 0B — WF1-B AUTHORITY RESOLUTION")
lines.append("=" * 80)
lines.append("")

lines.append(f"WF1-B ZIP: {ZIP_PATH}")
lines.append(f"ZIP EXISTS: {ZIP_PATH.exists()}")
lines.append("")

if not ZIP_PATH.exists():
    raise SystemExit("WF1_B_FINAL_AUTHORITY_R3.zip not found.")


# ------------------------------------------------------------
# ZIP inventory
# ------------------------------------------------------------

with zipfile.ZipFile(ZIP_PATH, "r") as z:

    names = z.namelist()

    lines.append("===== RELEVANT WF1-B ZIP MEMBERS =====")
    lines.append("")

    for name in sorted(names):
        if relevant(name):
            lines.append(name)

    lines.append("")


    # --------------------------------------------------------
    # Search textual authority/adjudication/spec files
    # --------------------------------------------------------

    lines.append("===== AUTHORITY / ADJUDICATION TEXT EXCERPTS =====")
    lines.append("")

    candidate_text = []

    for name in names:

        lower = name.lower()

        if lower.endswith(
            (".txt", ".md", ".json", ".csv")
        ) and relevant(name):

            candidate_text.append(name)


    search_terms = [
        "final_authority",
        "final authority",
        "authoritative",
        "primary",
        "joint",
        "operator",
        "surface_free",
        "surface-free",
        "r1",
        "window8",
        "window12",
        "superseded",
        "valid",
        "invalid",
        "auc",
        "detector",
        "feature"
    ]


    for name in sorted(candidate_text):

        try:
            raw = z.read(name)

            text = raw.decode(
                "utf-8",
                errors="ignore"
            )

        except Exception:
            continue

        matching = []

        for line in text.splitlines():

            line_lower = line.lower()

            if any(
                term in line_lower
                for term in search_terms
            ):
                matching.append(line)

        if matching:

            lines.append("")
            lines.append(f"--- {name} ---")

            for line in matching[:120]:
                lines.append(line)


# ------------------------------------------------------------
# Loose detector package inventory
# ------------------------------------------------------------

lines.append("")
lines.append("")
lines.append("===== PRE-CONSOLIDATION DETECTOR PACKAGE =====")
lines.append("")

if PACKAGE.exists():

    for path in sorted(
        p for p in PACKAGE.rglob("*")
        if p.is_file()
    ):

        rel = path.relative_to(PACKAGE)

        if relevant(str(rel)):

            lines.append(
                f"{rel}\t"
                f"{path.stat().st_size}\t"
                f"{sha256_file(path)}"
            )

else:
    lines.append("DETECTOR_PACKAGE NOT FOUND")


# ------------------------------------------------------------
# Explicit frozen model list
# ------------------------------------------------------------

models = (
    PACKAGE
    / "frozen_models"
)

lines.append("")
lines.append("")
lines.append("===== FROZEN MODELS FOUND =====")
lines.append("")

if models.exists():

    for path in sorted(models.glob("*.joblib")):

        lines.append(
            f"{path.name}\t"
            f"{path.stat().st_size}\t"
            f"{sha256_file(path)}"
        )


# ------------------------------------------------------------
# Critical WF2 prospective input
# ------------------------------------------------------------

prospective = (
    LEGACY
    / "03_INPUTS"
    / "WF2_A_FROZEN_40_MATCHED_PAIRS.jsonl"
)

lines.append("")
lines.append("")
lines.append("===== PROSPECTIVE SOURCE SET =====")
lines.append("")

if prospective.exists():

    row_count = 0

    with prospective.open(
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        for line in f:
            if line.strip():
                row_count += 1

    lines.append(
        f"PATH: {prospective}"
    )

    lines.append(
        f"NONEMPTY JSONL ROWS: {row_count}"
    )

    lines.append(
        f"SHA256: {sha256_file(prospective)}"
    )

else:

    lines.append(
        "WF2_A_FROZEN_40_MATCHED_PAIRS.jsonl NOT FOUND"
    )


lines.append("")
lines.append("=" * 80)
lines.append("INTERPRETATION RULE")
lines.append("=" * 80)
lines.append("")
lines.append(
    "This report performs authority resolution only."
)
lines.append(
    "It does not promote any detector to WF2 authority."
)
lines.append(
    "It does not generate prospective observations."
)
lines.append(
    "It does not cross Boundary 1."
)


OUT.write_text(
    "\n".join(lines),
    encoding="utf-8"
)

print(
    OUT.read_text(
        encoding="utf-8"
    )
)
