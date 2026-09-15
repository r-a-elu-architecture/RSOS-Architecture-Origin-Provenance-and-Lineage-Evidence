import csv
import json
import re
import zipfile
from pathlib import Path
from collections import defaultdict

import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception:
    pq = None


ROOT = Path(r"C:\RSOS")
WF3 = ROOT / "RSOS_EXPERIMENTS_WF3_GENEALOGY_POPULATION_CONFIRMATION"

INPUTS = WF3 / "02_INPUTS"
DIAG = WF3 / "06_DIAGNOSTIC"

ALLOWLIST = INPUTS / "WF3_AUTHORITATIVE_REUSE_FILE_ALLOWLIST.csv"
POPINV = DIAG / "WF3_V16_POPULATION_SOURCE_INVENTORY.csv"

F2_ZIP = Path(
    r"C:\RSOS\Training data\F2_BR2_DETERMINISTIC_CORPUS_FINGERPRINT_REFREEZE_COMPLETE.zip"
)

for d in [INPUTS, DIAG]:
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
        fields = list(rows[0].keys()) if rows else ["EMPTY"]

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


allow = read_csv(ALLOWLIST)
popinv = read_csv(POPINV)


# ============================================================
# PART A - V15 SAME-AUTHOR / NON-LINEAGE CONTROL RECOVERY
# ============================================================

CONTROL_TERM = re.compile(
    r"(?i)"
    r"(same[\s_-]*author|"
    r"non[\s_-]*lineage|"
    r"author[\s_-]*control|"
    r"same[\s_-]*human|"
    r"authorship[\s_-]*control)"
)

FILE_TOKEN = re.compile(
    r"([A-Za-z0-9_.\-]+"
    r"\.(?:csv|parquet|json|jsonl|ndjson|zip|txt))",
    re.I,
)

reference_hits = []


def record_text_hits(source, text, source_family):
    for m in CONTROL_TERM.finditer(text):

        lo = max(0, m.start() - 500)
        hi = min(len(text), m.end() + 500)

        context = re.sub(
            r"\s+",
            " ",
            text[lo:hi],
        )[:1200]

        mentioned = sorted(
            set(
                x.group(1)
                for x in FILE_TOKEN.finditer(context)
            )
        )

        reference_hits.append(
            {
                "SOURCE_FAMILY": source_family,
                "REFERENCE_SOURCE": source,
                "MATCH": m.group(0),
                "MENTIONED_FILES": "|".join(mentioned),
                "CONTEXT": context,
            }
        )


# Search authoritative loose text records.
for row in allow:

    ptxt = row.get("FULL_PATH", "").strip()

    if not ptxt:
        continue

    p = Path(ptxt)

    if not p.exists() or not p.is_file():
        continue

    if p.suffix.lower() not in {
        ".txt",
        ".md",
        ".json",
        ".csv",
        ".yaml",
        ".yml",
    }:
        continue

    try:
        if p.stat().st_size > 25 * 1024 * 1024:
            continue

        text = p.read_text(
            encoding="utf-8-sig",
            errors="replace",
        )

        record_text_hits(
            str(p),
            text,
            row.get("SOURCE_FAMILY", ""),
        )

    except Exception:
        pass


# Search inside authoritative ZIPs.
zip_sources = []

for row in allow:

    p = Path(
        row.get(
            "FULL_PATH",
            "",
        )
    )

    if (
        p.exists()
        and p.is_file()
        and p.suffix.lower() == ".zip"
    ):
        zip_sources.append(
            (
                row.get(
                    "SOURCE_FAMILY",
                    "",
                ),
                p,
            )
        )

if F2_ZIP.exists():
    zip_sources.append(
        (
            "F2",
            F2_ZIP,
        )
    )


seen_zips = set()

for family, zp in zip_sources:

    zkey = str(zp).lower()

    if zkey in seen_zips:
        continue

    seen_zips.add(zkey)

    try:
        with zipfile.ZipFile(zp, "r") as z:

            for info in z.infolist():

                if info.is_dir():
                    continue

                name = info.filename

                if CONTROL_TERM.search(name):

                    reference_hits.append(
                        {
                            "SOURCE_FAMILY": family,
                            "REFERENCE_SOURCE":
                                str(zp) + "::" + name,
                            "MATCH":
                                "CONTROL_TERM_IN_ARCHIVE_ENTRY_NAME",
                            "MENTIONED_FILES":
                                Path(name).name,
                            "CONTEXT":
                                name,
                        }
                    )

                ext = Path(name).suffix.lower()

                if ext not in {
                    ".txt",
                    ".md",
                    ".json",
                    ".csv",
                    ".yaml",
                    ".yml",
                }:
                    continue

                if info.file_size > 8 * 1024 * 1024:
                    continue

                try:
                    raw = z.read(info)

                    text = raw.decode(
                        "utf-8",
                        errors="replace",
                    )

                    record_text_hits(
                        str(zp) + "::" + name,
                        text,
                        family,
                    )

                except Exception:
                    pass

    except Exception:
        pass


refs_path = (
    DIAG
    / "WF3_V15_SAME_AUTHOR_AUTHORITY_REFERENCES.csv"
)

write_csv(
    refs_path,
    reference_hits,
)


# ============================================================
# Build basename index to resolve files explicitly named by
# authoritative records.
# ============================================================

mentioned_names = set()

for r in reference_hits:

    for x in r.get(
        "MENTIONED_FILES",
        "",
    ).split("|"):

        x = x.strip()

        if x:
            mentioned_names.add(
                x.lower()
            )


basename_index = defaultdict(list)

if mentioned_names:

    for p in ROOT.rglob("*"):

        try:
            if not p.is_file():
                continue

            if str(p).lower().startswith(
                str(WF3).lower()
            ):
                continue

            name = p.name.lower()

            if name in mentioned_names:
                basename_index[name].append(
                    str(p)
                )

        except Exception:
            continue


resolved_control = []

for r in reference_hits:

    names = [
        x.strip()
        for x in r.get(
            "MENTIONED_FILES",
            "",
        ).split("|")
        if x.strip()
    ]

    for name in names:

        for resolved in basename_index.get(
            name.lower(),
            [],
        ):

            resolved_control.append(
                {
                    "SOURCE_FAMILY":
                        r["SOURCE_FAMILY"],

                    "REFERENCE_SOURCE":
                        r["REFERENCE_SOURCE"],

                    "MENTIONED_FILE":
                        name,

                    "RESOLVED_PATH":
                        resolved,

                    "STATUS":
                        "AUTHORITY_REFERENCED_CONTROL_CANDIDATE",
                }
            )


resolved_path = (
    DIAG
    / "WF3_V15_SAME_AUTHOR_RESOLVED_FILES.csv"
)

write_csv(
    resolved_path,
    resolved_control,
)


# ============================================================
# PART B - V16 EXTERNAL POPULATION ID/PROVENANCE RESOLUTION
# ============================================================

ID_PRIORITY = [
    "conversation_hash",
    "conversation_id",
    "conv_id",
    "session_id",
    "dialogue_id",
    "chat_id",
    "id",
]


def choose_id_column(columns):
    lookup = {
        str(c).lower(): str(c)
        for c in columns
    }

    for preferred in ID_PRIORITY:

        if preferred in lookup:
            return lookup[preferred]

    for c in columns:

        s = str(c)

        if re.search(
            r"(?i)(conversation|session|dialog|chat).*id",
            s,
        ):
            return s

    return None


# ------------------------------------------------------------
# Historical raw shard IDs
# Use one physical copy per shard number.
# ------------------------------------------------------------

historical_by_shard = {}

for r in popinv:

    status = r.get(
        "PRELIMINARY_STATUS",
        "",
    )

    if status != "HISTORICAL_OVERLAP_CANDIDATE":
        continue

    p = Path(
        r.get(
            "FULL_PATH",
            "",
        )
    )

    if (
        not p.exists()
        or p.suffix.lower() != ".parquet"
    ):
        continue

    # Exclude derived feature tables.
    if re.search(
        r"(?i)(FEATURE|SCORE|FROZEN_FEATURE)",
        p.name,
    ):
        continue

    shard = r.get(
        "SHARD_NUMBER",
        "",
    )

    if shard in {
        "",
        None,
        "None",
    }:
        continue

    # Keep only one physical copy of each shard.
    if shard not in historical_by_shard:
        historical_by_shard[shard] = p


historical_ids = set()
historical_id_columns = set()
historical_id_files = []


if pq is not None:

    for shard, p in sorted(
        historical_by_shard.items(),
        key=lambda x: int(x[0]),
    ):

        try:

            pf = pq.ParquetFile(p)

            cols = list(
                pf.schema_arrow.names
            )

            id_col = choose_id_column(
                cols
            )

            if not id_col:
                historical_id_files.append(
                    {
                        "SHARD": shard,
                        "PATH": str(p),
                        "ID_COLUMN": "",
                        "IDS_READ": 0,
                        "STATUS": "NO_ID_COLUMN",
                    }
                )
                continue

            historical_id_columns.add(
                id_col
            )

            table = pq.read_table(
                p,
                columns=[id_col],
            )

            values = table[
                id_col
            ].to_pylist()

            count = 0

            for value in values:

                if value is None:
                    continue

                historical_ids.add(
                    str(value)
                )

                count += 1

            historical_id_files.append(
                {
                    "SHARD": shard,
                    "PATH": str(p),
                    "ID_COLUMN": id_col,
                    "IDS_READ": count,
                    "STATUS": "OK",
                }
            )

        except Exception as e:

            historical_id_files.append(
                {
                    "SHARD": shard,
                    "PATH": str(p),
                    "ID_COLUMN": "",
                    "IDS_READ": 0,
                    "STATUS":
                        "ERROR:" + str(e)[:300],
                }
            )


historical_ids_path = (
    DIAG
    / "WF3_V16_HISTORICAL_ID_EXTRACTION.csv"
)

write_csv(
    historical_ids_path,
    historical_id_files,
)


# ------------------------------------------------------------
# Classify current "fresh" candidates conservatively.
# ------------------------------------------------------------

candidate_results = []


def inspect_jsonl(path):
    line_count = 0
    parsed_count = 0
    ids = set()
    id_col = None
    keys = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            line_count += 1

            try:
                obj = json.loads(line)
            except Exception:
                continue

            if not isinstance(
                obj,
                dict,
            ):
                continue

            parsed_count += 1

            if not keys:
                keys = list(
                    obj.keys()
                )

                id_col = choose_id_column(
                    keys
                )

            if (
                id_col
                and id_col in obj
                and obj[id_col] is not None
            ):
                ids.add(
                    str(
                        obj[id_col]
                    )
                )

    return {
        "LINE_COUNT": line_count,
        "PARSED_COUNT": parsed_count,
        "KEYS": keys,
        "ID_COLUMN": id_col,
        "IDS": ids,
    }


for r in popinv:

    if r.get(
        "PRELIMINARY_STATUS",
        "",
    ) != "FRESH_POPULATION_CANDIDATE":
        continue

    p = Path(
        r.get(
            "FULL_PATH",
            "",
        )
    )

    if not p.exists():

        candidate_results.append(
            {
                "FILE_NAME": p.name,
                "FULL_PATH": str(p),
                "TYPE": p.suffix.lower(),
                "RECORD_COUNT": 0,
                "ID_COLUMN": "",
                "UNIQUE_IDS": 0,
                "HISTORICAL_ID_OVERLAP": 0,
                "OVERLAP_RATE": "",
                "FINAL_PROVENANCE_STATUS":
                    "EXCLUDE_MISSING_FILE",
                "REASON":
                    "File does not exist.",
            }
        )
        continue


    # Known historical derived artifact.
    if (
        "F2_EXTERNAL_POPULATION_RARITY"
        in str(p)
        and re.search(
            r"(?i)(FEATURE|SCORE|FROZEN)",
            p.name,
        )
    ):

        candidate_results.append(
            {
                "FILE_NAME": p.name,
                "FULL_PATH": str(p),
                "TYPE": p.suffix.lower(),
                "RECORD_COUNT":
                    r.get(
                        "ROW_COUNT",
                        "",
                    ),
                "ID_COLUMN": "",
                "UNIQUE_IDS": "",
                "HISTORICAL_ID_OVERLAP": "",
                "OVERLAP_RATE": "",
                "FINAL_PROVENANCE_STATUS":
                    "HISTORICAL_DERIVED_EXCLUDE",
                "REASON":
                    "Derived artifact from prior F2 external-population experiment.",
            }
        )
        continue


    if p.suffix.lower() in {
        ".jsonl",
        ".ndjson",
    }:

        try:

            info = inspect_jsonl(
                p
            )

            ids = info[
                "IDS"
            ]

            overlap = (
                len(
                    ids
                    & historical_ids
                )
                if ids
                and historical_ids
                else 0
            )

            rate = (
                overlap
                / len(ids)
                if ids
                else None
            )

            if not info["ID_COLUMN"]:

                final_status = (
                    "UNRESOLVED_NO_STABLE_ID"
                )

                reason = (
                    "No stable conversation/session ID column found; freshness cannot be established."
                )

            elif not historical_ids:

                final_status = (
                    "UNRESOLVED_NO_HISTORICAL_ID_BASE"
                )

                reason = (
                    "Historical shard IDs could not be extracted."
                )

            elif overlap > 0:

                final_status = (
                    "HISTORICAL_OVERLAP_EXCLUDE"
                )

                reason = (
                    "Stable IDs overlap with previously used historical WildChat population."
                )

            else:

                final_status = (
                    "FRESH_ID_DISJOINT_CANDIDATE"
                )

                reason = (
                    "No stable-ID overlap detected against previously used historical shards; still requires final freeze before V16."
                )

            candidate_results.append(
                {
                    "FILE_NAME":
                        p.name,

                    "FULL_PATH":
                        str(p),

                    "TYPE":
                        p.suffix.lower(),

                    "RECORD_COUNT":
                        info[
                            "LINE_COUNT"
                        ],

                    "ID_COLUMN":
                        info[
                            "ID_COLUMN"
                        ]
                        or "",

                    "UNIQUE_IDS":
                        len(ids),

                    "HISTORICAL_ID_OVERLAP":
                        overlap,

                    "OVERLAP_RATE":
                        (
                            rate
                            if rate is not None
                            else ""
                        ),

                    "FINAL_PROVENANCE_STATUS":
                        final_status,

                    "REASON":
                        reason,
                }
            )

        except Exception as e:

            candidate_results.append(
                {
                    "FILE_NAME": p.name,
                    "FULL_PATH": str(p),
                    "TYPE": p.suffix.lower(),
                    "RECORD_COUNT": "",
                    "ID_COLUMN": "",
                    "UNIQUE_IDS": "",
                    "HISTORICAL_ID_OVERLAP": "",
                    "OVERLAP_RATE": "",
                    "FINAL_PROVENANCE_STATUS":
                        "UNRESOLVED_READ_ERROR",
                    "REASON":
                        str(e)[:500],
                }
            )

    else:

        candidate_results.append(
            {
                "FILE_NAME": p.name,
                "FULL_PATH": str(p),
                "TYPE": p.suffix.lower(),
                "RECORD_COUNT":
                    r.get(
                        "ROW_COUNT",
                        "",
                    ),
                "ID_COLUMN": "",
                "UNIQUE_IDS": "",
                "HISTORICAL_ID_OVERLAP": "",
                "OVERLAP_RATE": "",
                "FINAL_PROVENANCE_STATUS":
                    "UNRESOLVED_NON_JSONL_CANDIDATE",
                "REASON":
                    "Candidate requires dedicated provenance resolution.",
            }
        )


population_resolution_path = (
    DIAG
    / "WF3_V16_POPULATION_PROVENANCE_RESOLUTION.csv"
)

write_csv(
    population_resolution_path,
    candidate_results,
)


# ============================================================
# Detect obvious whole-file / split-family relationship
# ============================================================

split_dir = ROOT / "wildchat_split"

split_relation = {
    "WHOLE_FILE_EXISTS":
        (split_dir / "wildchat.jsonl").exists(),

    "PART_FILES":
        sorted(
            str(x)
            for x in split_dir.glob(
                "wildchat_part_*.jsonl"
            )
        ),

    "WHOLE_FILE_SIZE":
        (
            (split_dir / "wildchat.jsonl").stat().st_size
            if (
                split_dir
                / "wildchat.jsonl"
            ).exists()
            else None
        ),
}

part_size_sum = 0

for item in split_relation[
    "PART_FILES"
]:

    p = Path(item)

    if p.exists():
        part_size_sum += (
            p.stat().st_size
        )

split_relation[
    "PART_FILE_SIZE_SUM"
] = part_size_sum

split_relation[
    "NOTE"
] = (
    "Whole and part files are treated as one candidate dataset family, not independent populations."
)

split_relation_path = (
    DIAG
    / "WF3_V16_WILDCHAT_SPLIT_RELATION.json"
)

split_relation_path.write_text(
    json.dumps(
        split_relation,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# SUMMARY
# ============================================================

resolved_control_count = len(
    resolved_control
)

fresh_id_disjoint = [
    r
    for r in candidate_results
    if r[
        "FINAL_PROVENANCE_STATUS"
    ]
    == "FRESH_ID_DISJOINT_CANDIDATE"
]

population_unresolved = [
    r
    for r in candidate_results
    if r[
        "FINAL_PROVENANCE_STATUS"
    ].startswith(
        "UNRESOLVED"
    )
]

summary = {
    "stage":
        "WF3_V15_V16_INPUT_RESOLUTION",

    "scientific_outcomes_inspected":
        False,

    "boundary_1_remains_frozen":
        True,

    "authority_addendum_remains_frozen":
        True,

    "same_author_reference_hits":
        len(
            reference_hits
        ),

    "same_author_resolved_file_candidates":
        resolved_control_count,

    "historical_unique_ids":
        len(
            historical_ids
        ),

    "fresh_id_disjoint_population_candidates":
        len(
            fresh_id_disjoint
        ),

    "population_unresolved":
        len(
            population_unresolved
        ),

    "next_status":
        (
            "READY_FOR_FINAL_INPUT_SELECTION_FREEZE"
            if (
                resolved_control_count > 0
                and len(
                    fresh_id_disjoint
                ) > 0
                and len(
                    population_unresolved
                ) == 0
            )
            else
            "TARGETED_INPUT_RESOLUTION_STILL_REQUIRED"
        ),
}


summary_path = (
    DIAG
    / "WF3_V15_V16_INPUT_RESOLUTION_SUMMARY.json"
)

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("============================================================")
print(" WF3 V15/V16 INPUT RESOLUTION COMPLETE")
print("============================================================")
print("Scientific outcomes inspected         : NO")
print("Boundary 1 remains frozen             : YES")
print("Authority addendum remains frozen     : YES")
print()

print(
    "Same-author authority references      :",
    len(reference_hits),
)

print(
    "Resolved same-author file candidates  :",
    resolved_control_count,
)

print()
print(
    "Historical unique population IDs      :",
    len(historical_ids),
)

print(
    "Historical ID columns                 :",
    "|".join(
        sorted(
            historical_id_columns
        )
    ),
)

print()
print("V16 CANDIDATE RESOLUTION:")

for r in candidate_results:

    print(
        " ",
        r[
            "FINAL_PROVENANCE_STATUS"
        ],
        "| records=",
        r[
            "RECORD_COUNT"
        ],
        "| id=",
        r[
            "ID_COLUMN"
        ],
        "| overlap=",
        r[
            "HISTORICAL_ID_OVERLAP"
        ],
        "|",
        r[
            "FULL_PATH"
        ],
    )

print()
print("TOP SAME-AUTHOR RESOLVED CANDIDATES:")

for r in resolved_control[:30]:

    print(
        " ",
        r[
            "SOURCE_FAMILY"
        ],
        "|",
        r[
            "MENTIONED_FILE"
        ],
        "|",
        r[
            "RESOLVED_PATH"
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
