import csv
import json
import re
import zipfile
from pathlib import Path

import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception:
    pq = None


WF3 = Path(
    r"C:\RSOS\RSOS_EXPERIMENTS_WF3_GENEALOGY_POPULATION_CONFIRMATION"
)

CONTROL = WF3 / "00_CONTROL"
INPUTS = WF3 / "02_INPUTS"
DIAG = WF3 / "06_DIAGNOSTIC"

ALLOWLIST = INPUTS / "WF3_AUTHORITATIVE_REUSE_FILE_ALLOWLIST.csv"
POP_MANIFEST = INPUTS / "WF3_EXTERNAL_POPULATION_FILE_MANIFEST.csv"

for d in [CONTROL, INPUTS, DIAG]:
    d.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    rows = list(rows)

    if fields is None:
        if rows:
            fields = list(rows[0].keys())
        else:
            fields = ["EMPTY"]

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        w.writeheader()

        if rows:
            w.writerows(rows)


if not ALLOWLIST.exists():
    raise SystemExit(
        f"Missing authority allowlist: {ALLOWLIST}"
    )

if not POP_MANIFEST.exists():
    raise SystemExit(
        f"Missing population manifest: {POP_MANIFEST}"
    )


allow = read_csv(ALLOWLIST)
pop = read_csv(POP_MANIFEST)


# ============================================================
# 1. AUTHORITY-CONSTRAINED TABLE SCHEMAS
# ============================================================

schema_rows = []


def classify_columns(columns):
    cols = [str(c) for c in columns]

    id_cols = [
        c for c in cols
        if re.search(
            r"(?i)(conversation|unit|source|record|seed|message).*id|^id$",
            c,
        )
    ]

    time_cols = [
        c for c in cols
        if re.search(
            r"(?i)(timestamp|created|date|time|temporal|chronolog|stage|era|early|late)",
            c,
        )
    ]

    control_cols = [
        c for c in cols
        if re.search(
            r"(?i)(same.?author|non.?lineage|control|lineage|label|class|group|subset)",
            c,
        )
    ]

    feature_cols = [
        c for c in cols
        if re.search(
            r"(?i)(feature|vector|score|similar|compat|operator|relation|topolog|graph|edge|fingerprint|coordinate|metric)",
            c,
        )
    ]

    population_cols = [
        c for c in cols
        if re.search(
            r"(?i)(user_id|session_id|conversation_id|source_id)",
            c,
        )
    ]

    return {
        "ID_COLUMNS": "|".join(id_cols),
        "TIME_COLUMNS": "|".join(time_cols),
        "CONTROL_COLUMNS": "|".join(control_cols),
        "FEATURE_COLUMNS": "|".join(feature_cols),
        "POPULATION_ID_COLUMNS": "|".join(population_cols),
    }


for row in allow:

    path_text = row.get("FULL_PATH", "").strip()

    if not path_text:
        continue

    p = Path(path_text)

    if not p.exists() or not p.is_file():
        continue

    ext = p.suffix.lower()

    if ext not in {
        ".csv",
        ".parquet",
        ".json",
        ".jsonl",
        ".ndjson",
    }:
        continue

    columns = []
    row_count = None
    error = ""

    try:

        if ext == ".parquet" and pq is not None:

            pf = pq.ParquetFile(p)

            columns = list(
                pf.schema_arrow.names
            )

            row_count = int(
                pf.metadata.num_rows
            )

        elif ext == ".csv":

            df = pd.read_csv(
                p,
                nrows=0,
            )

            columns = list(df.columns)

        elif ext in {
            ".jsonl",
            ".ndjson",
        }:

            # Read only one record to discover keys.
            with p.open(
                "r",
                encoding="utf-8",
                errors="replace",
            ) as f:

                line = f.readline()

            if line.strip():
                obj = json.loads(line)

                if isinstance(obj, dict):
                    columns = list(obj.keys())

        elif ext == ".json":

            # Only inspect compact JSON structures.
            if p.stat().st_size <= 10 * 1024 * 1024:

                obj = json.loads(
                    p.read_text(
                        encoding="utf-8-sig",
                        errors="replace",
                    )
                )

                if isinstance(obj, dict):
                    columns = list(obj.keys())

                elif (
                    isinstance(obj, list)
                    and obj
                    and isinstance(obj[0], dict)
                ):
                    columns = list(
                        obj[0].keys()
                    )

    except Exception as e:
        error = str(e)[:500]

    signals = classify_columns(
        columns
    )

    candidate_score = 0

    if signals["ID_COLUMNS"]:
        candidate_score += 2

    if signals["TIME_COLUMNS"]:
        candidate_score += 3

    if signals["CONTROL_COLUMNS"]:
        candidate_score += 4

    if signals["FEATURE_COLUMNS"]:
        candidate_score += 4

    schema_rows.append(
        {
            "SOURCE_FAMILY":
                row.get(
                    "SOURCE_FAMILY",
                    "",
                ),

            "AUTHORITY_ROLE":
                row.get(
                    "AUTHORITY_ROLE",
                    "",
                ),

            "CATEGORY":
                row.get(
                    "CATEGORY",
                    "",
                ),

            "FILE_NAME":
                p.name,

            "FULL_PATH":
                str(p),

            "EXTENSION":
                ext,

            "ROW_COUNT":
                row_count,

            "COLUMN_COUNT":
                len(columns),

            "COLUMNS":
                "|".join(
                    map(str, columns)
                ),

            **signals,

            "V15_CANDIDATE_SCORE":
                candidate_score,

            "ERROR":
                error,
        }
    )


schema_path = (
    DIAG
    / "WF3_AUTHORITY_CONSTRAINED_SCHEMA_INVENTORY.csv"
)

write_csv(
    schema_path,
    schema_rows,
)


# ============================================================
# 2. ZIP ENTRY INVENTORY
# ============================================================

zip_rows = []

for row in allow:

    path_text = row.get(
        "FULL_PATH",
        "",
    ).strip()

    if not path_text:
        continue

    p = Path(path_text)

    if (
        not p.exists()
        or p.suffix.lower() != ".zip"
    ):
        continue

    try:

        with zipfile.ZipFile(
            p,
            "r",
        ) as z:

            for info in z.infolist():

                if info.is_dir():
                    continue

                name = info.filename

                score = 0

                if re.search(
                    r"(?i)(same.?author|non.?lineage|author.?control)",
                    name,
                ):
                    score += 10

                if re.search(
                    r"(?i)(genealog|lineage|chronolog|temporal)",
                    name,
                ):
                    score += 6

                if re.search(
                    r"(?i)(feature|vector|coordinate|fingerprint)",
                    name,
                ):
                    score += 5

                if re.search(
                    r"(?i)(operator|relational|topolog|graph|edge)",
                    name,
                ):
                    score += 5

                zip_rows.append(
                    {
                        "SOURCE_FAMILY":
                            row.get(
                                "SOURCE_FAMILY",
                                "",
                            ),

                        "ARCHIVE":
                            str(p),

                        "ENTRY":
                            name,

                        "SIZE_BYTES":
                            info.file_size,

                        "V15_RELEVANCE_SCORE":
                            score,
                    }
                )

    except Exception:
        pass


zip_path = (
    DIAG
    / "WF3_AUTHORITY_ZIP_ENTRY_INDEX.csv"
)

write_csv(
    zip_path,
    zip_rows,
)


# ============================================================
# 3. SEARCH AUTHORITATIVE TEXT RECORDS FOR PRIOR POPULATION USE
# ============================================================

population_names = set()

for row in pop:

    name = row.get(
        "FILE_NAME",
        "",
    ).strip()

    if name:
        population_names.add(
            name
        )


text_authority_files = []

for row in allow:

    path_text = row.get(
        "FULL_PATH",
        "",
    ).strip()

    if not path_text:
        continue

    p = Path(path_text)

    if (
        p.exists()
        and p.is_file()
        and p.suffix.lower()
        in {
            ".txt",
            ".md",
            ".json",
            ".csv",
            ".yaml",
            ".yml",
        }
        and p.stat().st_size
        <= 20 * 1024 * 1024
    ):
        text_authority_files.append(
            p
        )


overlap_refs = []

for p in text_authority_files:

    try:

        text = p.read_text(
            encoding="utf-8-sig",
            errors="replace",
        )

    except Exception:
        continue

    for population_name in population_names:

        if population_name in text:

            overlap_refs.append(
                {
                    "POPULATION_FILE":
                        population_name,

                    "AUTHORITY_REFERENCE_FILE":
                        str(p),

                    "REFERENCE_TYPE":
                        "EXACT_FILENAME_REFERENCE",
                }
            )

    # Also detect explicit shard references even if filenames differ.
    for m in re.finditer(
        r"(?i)(?:shard[_\- ]*)?(\d{1,3})(?:\s*/\s*86)?",
        text,
    ):

        context_start = max(
            0,
            m.start() - 70,
        )

        context_end = min(
            len(text),
            m.end() + 70,
        )

        context = re.sub(
            r"\s+",
            " ",
            text[
                context_start:
                context_end
            ],
        )

        if re.search(
            r"(?i)(wildchat|shard|external|population|parquet)",
            context,
        ):

            overlap_refs.append(
                {
                    "POPULATION_FILE":
                        f"SHARD_NUMBER_REFERENCE:{m.group(1)}",

                    "AUTHORITY_REFERENCE_FILE":
                        str(p),

                    "REFERENCE_TYPE":
                        context[:500],
                }
            )


overlap_path = (
    DIAG
    / "WF3_UPSTREAM_POPULATION_OVERLAP_REFERENCES.csv"
)

write_csv(
    overlap_path,
    overlap_refs,
)


# ============================================================
# 4. RAW EXTERNAL POPULATION SCHEMA/METADATA
# ============================================================

exact_referenced = {
    r["POPULATION_FILE"]
    for r in overlap_refs
    if not r["POPULATION_FILE"].startswith(
        "SHARD_NUMBER_REFERENCE:"
    )
}


shard_numbers_referenced = set()

for r in overlap_refs:

    value = r[
        "POPULATION_FILE"
    ]

    if value.startswith(
        "SHARD_NUMBER_REFERENCE:"
    ):

        try:
            shard_numbers_referenced.add(
                int(
                    value.split(":")[1]
                )
            )
        except Exception:
            pass


population_rows = []

for row in pop:

    path_text = row.get(
        "FULL_PATH",
        "",
    ).strip()

    if not path_text:
        continue

    p = Path(path_text)

    exists = p.exists()
    columns = []
    row_count = None
    error = ""

    shard_number = None

    m = re.search(
        r"(?i)train-(\d+)-of-\d+\.parquet$",
        p.name,
    )

    if m:
        shard_number = int(
            m.group(1)
        )

    try:

        if (
            exists
            and p.suffix.lower()
            == ".parquet"
            and pq is not None
        ):

            pf = pq.ParquetFile(
                p
            )

            columns = list(
                pf.schema_arrow.names
            )

            row_count = int(
                pf.metadata.num_rows
            )

    except Exception as e:
        error = str(e)[:500]

    signals = classify_columns(
        columns
    )

    exact_overlap = (
        p.name
        in exact_referenced
    )

    shard_overlap = (
        shard_number
        in shard_numbers_referenced
        if shard_number is not None
        else False
    )

    population_rows.append(
        {
            "FILE_NAME":
                p.name,

            "FULL_PATH":
                str(p),

            "EXISTS":
                exists,

            "SIZE_BYTES":
                (
                    p.stat().st_size
                    if exists
                    else None
                ),

            "SHARD_NUMBER":
                shard_number,

            "ROW_COUNT":
                row_count,

            "COLUMN_COUNT":
                len(columns),

            "COLUMNS":
                "|".join(columns),

            "POPULATION_ID_COLUMNS":
                signals[
                    "POPULATION_ID_COLUMNS"
                ],

            "EXACT_UPSTREAM_REFERENCE":
                exact_overlap,

            "SHARD_UPSTREAM_REFERENCE":
                shard_overlap,

            "PRELIMINARY_STATUS":
                (
                    "HISTORICAL_OVERLAP_CANDIDATE"
                    if (
                        exact_overlap
                        or shard_overlap
                    )
                    else "FRESH_POPULATION_CANDIDATE"
                ),

            "ERROR":
                error,
        }
    )


population_path = (
    DIAG
    / "WF3_V16_POPULATION_SOURCE_INVENTORY.csv"
)

write_csv(
    population_path,
    population_rows,
)


# ============================================================
# 5. V15 ROW-LEVEL CANDIDATE SHORTLIST
# ============================================================

v15_rows = []

for row in schema_rows:

    score = int(
        row.get(
            "V15_CANDIDATE_SCORE",
            0,
        )
        or 0
    )

    if score >= 4:

        v15_rows.append(
            row
        )


for row in zip_rows:

    score = int(
        row.get(
            "V15_RELEVANCE_SCORE",
            0,
        )
        or 0
    )

    if score >= 5:

        v15_rows.append(
            {
                "SOURCE_FAMILY":
                    row[
                        "SOURCE_FAMILY"
                    ],

                "AUTHORITY_ROLE":
                    "ZIP_ENTRY",

                "CATEGORY":
                    "ARCHIVE_ENTRY",

                "FILE_NAME":
                    Path(
                        row["ENTRY"]
                    ).name,

                "FULL_PATH":
                    (
                        row["ARCHIVE"]
                        + "::"
                        + row["ENTRY"]
                    ),

                "EXTENSION":
                    Path(
                        row["ENTRY"]
                    ).suffix.lower(),

                "ROW_COUNT":
                    None,

                "COLUMN_COUNT":
                    None,

                "COLUMNS":
                    "",

                "ID_COLUMNS":
                    "",

                "TIME_COLUMNS":
                    "",

                "CONTROL_COLUMNS":
                    "",

                "FEATURE_COLUMNS":
                    "",

                "POPULATION_ID_COLUMNS":
                    "",

                "V15_CANDIDATE_SCORE":
                    score,

                "ERROR":
                    "",
            }
        )


v15_rows.sort(
    key=lambda r: int(
        r.get(
            "V15_CANDIDATE_SCORE",
            0,
        )
        or 0
    ),
    reverse=True,
)


v15_path = (
    DIAG
    / "WF3_V15_ROWLEVEL_CANDIDATES.csv"
)

write_csv(
    v15_path,
    v15_rows,
)


# ============================================================
# 6. SUMMARY
# ============================================================

fresh_population = [
    r
    for r in population_rows
    if r[
        "PRELIMINARY_STATUS"
    ]
    == "FRESH_POPULATION_CANDIDATE"
]


historical_population = [
    r
    for r in population_rows
    if r[
        "PRELIMINARY_STATUS"
    ]
    == "HISTORICAL_OVERLAP_CANDIDATE"
]


fresh_rows = sum(
    int(r["ROW_COUNT"])
    for r in fresh_population
    if isinstance(
        r.get("ROW_COUNT"),
        int,
    )
)


same_author_candidates = [
    r
    for r in v15_rows
    if (
        re.search(
            r"(?i)(same.?author|non.?lineage|author.?control)",
            (
                str(
                    r.get(
                        "FULL_PATH",
                        "",
                    )
                )
                + " "
                + str(
                    r.get(
                        "CONTROL_COLUMNS",
                        "",
                    )
                )
            ),
        )
    )
]


summary = {
    "stage":
        "WF3_AUTHORITY_CONSTRAINED_STANDARDIZATION_PREFLIGHT",

    "scientific_outcomes_inspected":
        False,

    "boundary_1_remains_frozen":
        True,

    "authority_addendum_remains_frozen":
        True,

    "authority_schema_tables":
        len(
            schema_rows
        ),

    "authority_zip_entries":
        len(
            zip_rows
        ),

    "v15_candidate_sources":
        len(
            v15_rows
        ),

    "v15_same_author_candidate_sources":
        len(
            same_author_candidates
        ),

    "population_files":
        len(
            population_rows
        ),

    "fresh_population_candidates":
        len(
            fresh_population
        ),

    "historical_overlap_candidates":
        len(
            historical_population
        ),

    "fresh_population_rows_metadata_sum":
        fresh_rows,

    "next_status":
        (
            "READY_FOR_STANDARDIZED_INPUT_BUILD"
            if (
                same_author_candidates
                and fresh_population
            )
            else
            "INPUT_RESOLUTION_REMAINS_REQUIRED"
        ),
}


summary_path = (
    DIAG
    / "WF3_STANDARDIZATION_PREFLIGHT_SUMMARY.json"
)

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# 7. TERMINAL REPORT
# ============================================================

print()
print("============================================================")
print(" WF3 STANDARDIZATION PREFLIGHT COMPLETE")
print("============================================================")
print("Scientific outcomes inspected      : NO")
print("Boundary 1 remains frozen          : YES")
print("Authority addendum remains frozen  : YES")
print()
print(
    "Authority schema tables           :",
    len(schema_rows),
)
print(
    "Authority ZIP entries             :",
    len(zip_rows),
)
print(
    "V15 candidate sources             :",
    len(v15_rows),
)
print(
    "V15 same-author candidates        :",
    len(same_author_candidates),
)
print()
print(
    "External population files         :",
    len(population_rows),
)
print(
    "Fresh population candidates       :",
    len(fresh_population),
)
print(
    "Historical overlap candidates     :",
    len(historical_population),
)
print(
    "Fresh population row-count sum    :",
    fresh_rows,
)

print()
print("TOP V15 CANDIDATES:")

for r in v15_rows[:20]:

    print(
        " ",
        r.get(
            "V15_CANDIDATE_SCORE",
            0,
        ),
        "|",
        r.get(
            "SOURCE_FAMILY",
            "",
        ),
        "|",
        r.get(
            "FULL_PATH",
            "",
        ),
    )

print()
print("POPULATION FILE STATUS:")

for r in population_rows:

    print(
        " ",
        r[
            "PRELIMINARY_STATUS"
        ],
        "| shard=",
        r[
            "SHARD_NUMBER"
        ],
        "| rows=",
        r[
            "ROW_COUNT"
        ],
        "|",
        r[
            "FULL_PATH"
        ],
    )

print()
print("NEXT STATUS:")
print(
    " ",
    summary[
        "next_status"
    ],
)

print()
print("SUMMARY:")
print(
    " ",
    summary_path,
)

print("============================================================")
