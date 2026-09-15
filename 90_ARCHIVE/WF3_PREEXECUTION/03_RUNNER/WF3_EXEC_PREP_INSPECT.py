import csv
import json
import re
from pathlib import Path

ROOT = Path(r"C:\RSOS")
WF3 = ROOT / "RSOS_EXPERIMENTS_WF3_GENEALOGY_POPULATION_CONFIRMATION"

CONTROL = WF3 / "00_CONTROL"
INPUTS = WF3 / "02_INPUTS"
DIAG = WF3 / "06_DIAGNOSTIC"

MANIFEST = INPUTS / "WF3_BOUNDARY1_INPUT_REFERENCE_MANIFEST.csv"
SAME_REFS = CONTROL / "WF3_SAME_AUTHOR_CONTENT_REFERENCES.csv"
POP_MANIFEST = INPUTS / "WF3_EXTERNAL_POPULATION_FILE_MANIFEST.csv"

for d in [CONTROL, INPUTS, DIAG]:
    d.mkdir(parents=True, exist_ok=True)

try:
    import pandas as pd
except Exception as e:
    raise SystemExit(f"pandas unavailable: {e}")

try:
    import pyarrow.parquet as pq
except Exception:
    pq = None


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def inspect_file(path):
    p = Path(path)

    result = {
        "path": str(p),
        "exists": p.exists(),
        "extension": p.suffix.lower(),
        "size_bytes": p.stat().st_size if p.exists() else None,
        "row_count": None,
        "column_count": None,
        "columns": "",
        "id_columns": "",
        "time_columns": "",
        "control_columns": "",
        "user_session_columns": "",
        "numeric_column_count": None,
        "same_author_value_signal": False,
        "error": "",
    }

    if not p.exists():
        return result

    try:
        cols = []
        sample = None

        if p.suffix.lower() == ".parquet":
            if pq is not None:
                pf = pq.ParquetFile(p)
                cols = list(pf.schema_arrow.names)
                result["row_count"] = pf.metadata.num_rows
                result["column_count"] = len(cols)

                try:
                    sample = pf.read_row_group(
                        0,
                        columns=cols[:min(len(cols), 200)]
                    ).slice(0, 200).to_pandas()
                except Exception:
                    sample = None
            else:
                sample = pd.read_parquet(p).head(200)
                cols = list(sample.columns)

        elif p.suffix.lower() == ".csv":
            sample = pd.read_csv(p, nrows=200, low_memory=False)
            cols = list(sample.columns)

        elif p.suffix.lower() in {".jsonl", ".ndjson"}:
            sample = pd.read_json(p, lines=True, nrows=200)
            cols = list(sample.columns)

        elif p.suffix.lower() == ".json":
            if p.stat().st_size <= 50 * 1024 * 1024:
                data = json.loads(
                    p.read_text(encoding="utf-8-sig", errors="replace")
                )

                if isinstance(data, list) and data:
                    sample = pd.DataFrame(data[:200])

                elif isinstance(data, dict):
                    for value in data.values():
                        if isinstance(value, list) and value:
                            sample = pd.DataFrame(value[:200])
                            break

                if sample is not None:
                    cols = list(sample.columns)

        else:
            return result

        result["column_count"] = len(cols)
        result["columns"] = "|".join(map(str, cols))

        low = {str(c).lower(): str(c) for c in cols}

        id_cols = [
            c for c in cols
            if re.search(
                r"(unit_id|conversation_id|source_id|message_id|record_id|seed_id)",
                str(c),
                re.I,
            )
        ]

        time_cols = [
            c for c in cols
            if re.search(
                r"(timestamp|date|time|chronolog|temporal|stage|era|early|late)",
                str(c),
                re.I,
            )
        ]

        control_cols = [
            c for c in cols
            if re.search(
                r"(same.?author|non.?lineage|control|lineage|class|label|group|subset)",
                str(c),
                re.I,
            )
        ]

        user_cols = [
            c for c in cols
            if re.search(
                r"(user_id|session_id|conversation_id|source_id)",
                str(c),
                re.I,
            )
        ]

        result["id_columns"] = "|".join(map(str, id_cols))
        result["time_columns"] = "|".join(map(str, time_cols))
        result["control_columns"] = "|".join(map(str, control_cols))
        result["user_session_columns"] = "|".join(map(str, user_cols))

        if sample is not None and len(sample.columns):
            result["numeric_column_count"] = int(
                sum(
                    pd.api.types.is_numeric_dtype(sample[c])
                    for c in sample.columns
                )
            )

            text = sample.astype(str).to_string(index=False)

            if re.search(
                r"(same.?author|non.?lineage|nonlineage|author.?control)",
                text,
                re.I,
            ):
                result["same_author_value_signal"] = True

    except Exception as exc:
        result["error"] = str(exc)[:500]

    return result


if not MANIFEST.exists():
    raise SystemExit(f"Missing Boundary-1 manifest: {MANIFEST}")

manifest = read_csv_rows(MANIFEST)

# ------------------------------------------------------------
# 1. Inspect direct upstream table references
# ------------------------------------------------------------

direct_paths = []

for row in manifest:
    path = row.get("PATH", "").strip()

    if (
        row.get("TYPE") == "FILE_REFERENCE"
        and path
        and Path(path).suffix.lower()
        in {".csv", ".json", ".jsonl", ".ndjson", ".parquet"}
    ):
        direct_paths.append((row.get("ROLE", ""), path))

seen = set()
schema_rows = []

for role, path in direct_paths:
    key = str(Path(path)).lower()

    if key in seen:
        continue

    seen.add(key)

    info = inspect_file(path)
    info["manifest_role"] = role
    schema_rows.append(info)

schema_path = DIAG / "WF3_EXEC_PREP_SCHEMA_INVENTORY.csv"

pd.DataFrame(schema_rows).to_csv(
    schema_path,
    index=False,
)

# ------------------------------------------------------------
# 2. Resolve same-author control references
# ------------------------------------------------------------

same_reference_files = []

if SAME_REFS.exists():
    for row in read_csv_rows(SAME_REFS):
        p = row.get("FILE", "").strip()
        if p:
            same_reference_files.append(Path(p))

reference_lines = []
referenced_names = set()

file_pattern = re.compile(
    r"([A-Za-z0-9_.()\- ]+\.(?:csv|json|jsonl|ndjson|parquet|zip))",
    re.I,
)

for p in same_reference_files:
    if not p.exists():
        continue

    try:
        text = p.read_text(
            encoding="utf-8-sig",
            errors="replace",
        )

        lines = text.splitlines()

        for number, line in enumerate(lines, 1):
            if re.search(
                r"(same.?author|non.?lineage|nonlineage|author.?control)",
                line,
                re.I,
            ):
                reference_lines.append(
                    {
                        "source_file": str(p),
                        "line_number": number,
                        "line": line[:4000],
                    }
                )

                for m in file_pattern.findall(line):
                    referenced_names.add(m.strip())

    except Exception:
        pass

pd.DataFrame(reference_lines).to_csv(
    DIAG / "WF3_SAME_AUTHOR_REFERENCE_LINES.csv",
    index=False,
)

# ------------------------------------------------------------
# 3. Build basename index and resolve mentioned artifacts
# ------------------------------------------------------------

basename_index = {}

for p in ROOT.rglob("*"):
    try:
        if not p.is_file():
            continue

        if str(p).lower().startswith(str(WF3).lower()):
            continue

        basename_index.setdefault(
            p.name.lower(),
            []
        ).append(str(p))

    except Exception:
        continue

resolved_refs = []

for name in sorted(referenced_names):
    hits = basename_index.get(
        Path(name).name.lower(),
        [],
    )

    for hit in hits:
        info = inspect_file(hit)

        resolved_refs.append(
            {
                "referenced_name": name,
                **info,
            }
        )

resolved_path = DIAG / "WF3_SAME_AUTHOR_RESOLVED_REFERENCES.csv"

pd.DataFrame(resolved_refs).to_csv(
    resolved_path,
    index=False,
)

# ------------------------------------------------------------
# 4. Identify row-level same-author control candidates
# ------------------------------------------------------------

control_candidates = []

for row in schema_rows + resolved_refs:

    control_signal = bool(
        row.get("control_columns")
        or row.get("same_author_value_signal")
    )

    time_signal = bool(
        row.get("time_columns")
    )

    id_signal = bool(
        row.get("id_columns")
    )

    if control_signal:
        score = 0

        if row.get("same_author_value_signal"):
            score += 6

        if row.get("control_columns"):
            score += 4

        if time_signal:
            score += 3

        if id_signal:
            score += 2

        if (
            row.get("numeric_column_count")
            and row.get("numeric_column_count") >= 5
        ):
            score += 2

        control_candidates.append(
            {
                "score": score,
                **row,
            }
        )

control_candidates = sorted(
    control_candidates,
    key=lambda x: x.get("score", 0),
    reverse=True,
)

pd.DataFrame(control_candidates).to_csv(
    DIAG / "WF3_V15_SAME_AUTHOR_ROWLEVEL_CANDIDATES.csv",
    index=False,
)

# ------------------------------------------------------------
# 5. V15 candidate structural tables
# ------------------------------------------------------------

v15_candidates = []

for row in schema_rows:
    role = row.get("manifest_role", "")

    if role not in {
        "HISTORICAL_LINEAGE",
        "SAME_AUTHOR_NON_LINEAGE",
        "RELATIONAL_TOPOLOGICAL",
        "TEMPORAL_METADATA",
        "FROZEN_FEATURES",
    }:
        continue

    score = 0

    if row.get("id_columns"):
        score += 2

    if row.get("time_columns"):
        score += 3

    if row.get("control_columns"):
        score += 3

    nnum = row.get("numeric_column_count")

    if isinstance(nnum, int) and nnum >= 5:
        score += 3

    v15_candidates.append(
        {
            "score": score,
            **row,
        }
    )

v15_candidates.sort(
    key=lambda x: x.get("score", 0),
    reverse=True,
)

pd.DataFrame(v15_candidates).to_csv(
    DIAG / "WF3_V15_CANDIDATE_TABLES.csv",
    index=False,
)

# ------------------------------------------------------------
# 6. V16 population schema
# ------------------------------------------------------------

population_rows = []

if POP_MANIFEST.exists():
    for row in read_csv_rows(POP_MANIFEST):

        p = row.get("FULL_PATH", "").strip()

        if not p:
            continue

        info = inspect_file(p)

        population_rows.append(
            {
                "file_name": row.get("FILE_NAME", ""),
                **info,
            }
        )

pd.DataFrame(population_rows).to_csv(
    DIAG / "WF3_V16_POPULATION_SCHEMA.csv",
    index=False,
)

known_rows = [
    r.get("row_count")
    for r in population_rows
    if isinstance(r.get("row_count"), int)
]

same_author_usable = any(
    (
        bool(r.get("same_author_value_signal"))
        and bool(r.get("id_columns"))
        and bool(r.get("time_columns"))
    )
    for r in control_candidates
)

summary = {
    "stage": "WF3_EXECUTION_PREPARATION",
    "scientific_outcomes_inspected": False,
    "upstream_files_modified": False,
    "boundary_1_remains_frozen": True,
    "schema_tables_inspected": len(schema_rows),
    "same_author_reference_files": len(same_reference_files),
    "same_author_reference_lines": len(reference_lines),
    "same_author_named_artifacts_resolved": len(resolved_refs),
    "same_author_rowlevel_candidates": len(control_candidates),
    "same_author_execution_ready_signal": same_author_usable,
    "v15_candidate_tables": len(v15_candidates),
    "external_population_files": len(population_rows),
    "external_population_known_rows_sum": (
        int(sum(known_rows))
        if known_rows
        else None
    ),
    "next_status": (
        "READY_TO_BUILD_STANDARDIZED_INPUTS"
        if same_author_usable
        else "SAME_AUTHOR_ROWLEVEL_SOURCE_REQUIRES_RESOLUTION"
    ),
}

summary_path = DIAG / "WF3_EXEC_PREP_SUMMARY.json"

summary_path.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print()
print("============================================================")
print(" WF3 EXECUTION PREPARATION COMPLETE")
print("============================================================")
print("Scientific outcomes inspected : NO")
print("Upstream files modified        : NO")
print("Boundary 1 remains frozen      : YES")
print()
print("Schema tables inspected        :", len(schema_rows))
print("Same-author reference files    :", len(same_reference_files))
print("Same-author reference lines    :", len(reference_lines))
print("Resolved referenced artifacts  :", len(resolved_refs))
print("Same-author row candidates     :", len(control_candidates))
print("V15 candidate tables           :", len(v15_candidates))
print("External population files      :", len(population_rows))
print()
print("NEXT STATUS:")
print(" ", summary["next_status"])
print()
print("SUMMARY:")
print(" ", summary_path)
print("============================================================")
