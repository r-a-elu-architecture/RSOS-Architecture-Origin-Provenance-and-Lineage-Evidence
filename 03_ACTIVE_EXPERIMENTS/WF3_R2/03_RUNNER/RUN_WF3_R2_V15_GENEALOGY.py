import csv
import itertools
import json
import re
from pathlib import Path

import numpy as np

from WF3_R2_COMMON import (
    read_csv_rows,
    write_csv_rows,
    verify_reference_rows,
    numeric_variable_profiles,
    aggregate_profiles,
    robust_standardize,
    euclidean_distance_matrix,
    spearman,
)

ROOT = Path(
    r"C:\RSOS\RSOS_EXPERIMENTS_WF3_GENEALOGY_POPULATION_CONFIRMATION_R2"
)

MANIFEST = (
    ROOT
    / "02_INPUTS"
    / "GENEALOGY_DIRECTIONALITY"
    / "WF3_V15_CANONICAL_REFERENCE_MANIFEST.csv"
)

OUT = (
    ROOT
    / "05_OUTPUTS"
    / "V15_GENEALOGY_TEMPORAL_DIRECTIONALITY"
)

DIAG = (
    ROOT
    / "06_DIAGNOSTIC"
    / "V15_GENEALOGY_TEMPORAL_DIRECTIONALITY"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

DIAG.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 20260915

ALLOWED_BUCKETS = {
    "05_OUTPUTS",
    "07_ADJUDICATION",
    "09_FREEZE",
}

ALLOWED_EXT = {
    ".csv",
    ".json",
    ".parquet",
}

POSITIVE = re.compile(
    r"(graph|edge|operator|topology|relation|feature|vector|"
    r"contrast|condition|specificity|metric|result|summary|"
    r"calibration)",
    re.I,
)

NEGATIVE = re.compile(
    r"(sha256|hash_manifest|environment|readme|package_status)",
    re.I,
)

PRIMARY_EDGES = [
    ("RUN-004", "RUN-005"),
    ("RUN-005", "RUN-006"),
    ("RUN-006", "RUN-007"),
    ("RUN-007", "RUN-008"),
    ("RUN-008", "EXP-010"),
    ("RUN-003", "EXP-008"),
    ("EXP-009", "EXP-014"),
]

SECONDARY_EDGES = [
    ("EXP-002", "RUN-001"),
]

F1_PATH = [
    "RUN-004",
    "RUN-005",
    "RUN-006",
    "RUN-007",
    "RUN-008",
    "EXP-010",
]


rows = verify_reference_rows(
    read_csv_rows(
        MANIFEST
    )
)


selection = []
selected = []

for row in rows:
    p = Path(
        row[
            "CANONICAL_FILE"
        ]
    )

    reason = ""

    if (
        row[
            "DESTINATION_BUCKET"
        ]
        not in ALLOWED_BUCKETS
    ):
        reason = "EXCLUDE_BUCKET"

    elif p.suffix.lower() not in ALLOWED_EXT:
        reason = "EXCLUDE_EXTENSION"

    elif p.stat().st_size > 209715200:
        reason = "EXCLUDE_SIZE"

    elif not POSITIVE.search(
        p.name
    ):
        reason = "EXCLUDE_FILENAME"

    elif NEGATIVE.search(
        p.name
    ):
        reason = "EXCLUDE_NEGATIVE_PATTERN"

    else:
        reason = "SELECT_PRIMARY"


    out = {
        **row,
        "V15_DECISION":
            reason,
    }

    selection.append(out)

    if reason == "SELECT_PRIMARY":
        selected.append(out)


write_csv_rows(
    OUT
    / "V15_FILE_SELECTION.csv",
    selection,
)


entity_profiles = {}
parse_diag = []

for row in selected:
    entity = row[
        "ENTITY_ID"
    ]

    path = Path(
        row[
            "CANONICAL_FILE"
        ]
    )

    try:
        profiles = numeric_variable_profiles(
            path
        )

        parse_diag.append({
            "ENTITY_ID":
                entity,
            "CANONICAL_FILE":
                str(path),
            "STATUS":
                "PARSED",
            "NUMERIC_VARIABLE_PROFILES":
                len(profiles),
            "ERROR":
                "",
        })

        if profiles:
            entity_profiles.setdefault(
                entity,
                []
            ).extend(
                profiles
            )

    except Exception as exc:
        parse_diag.append({
            "ENTITY_ID":
                entity,
            "CANONICAL_FILE":
                str(path),
            "STATUS":
                "PARSE_ERROR",
            "NUMERIC_VARIABLE_PROFILES":
                0,
            "ERROR":
                str(exc),
        })


write_csv_rows(
    DIAG
    / "V15_PARSE_DIAGNOSTICS.csv",
    parse_diag,
)


entity_ids = sorted(
    entity_profiles
)

vectors = []
valid_entities = []

signature_rows = []

for entity in entity_ids:
    vector = aggregate_profiles(
        entity_profiles[
            entity
        ]
    )

    if vector is None:
        continue

    valid_entities.append(
        entity
    )

    vectors.append(
        vector
    )

    signature_rows.append({
        "ENTITY_ID":
            entity,
        **{
            f"D{i:03d}":
                float(value)
            for i, value
            in enumerate(vector)
        },
    })


if len(valid_entities) < 4:
    raise RuntimeError(
        "V15 has fewer than four valid entity signatures."
    )


write_csv_rows(
    OUT
    / "V15_ENTITY_SIGNATURES.csv",
    signature_rows,
)


z = robust_standardize(
    np.vstack(
        vectors
    )
)

dist = euclidean_distance_matrix(
    z
)

index = {
    entity: i
    for i, entity
    in enumerate(
        valid_entities
    )
}


distance_rows = []

for i, a in enumerate(
    valid_entities
):
    for j, b in enumerate(
        valid_entities
    ):
        distance_rows.append({
            "ENTITY_A":
                a,
            "ENTITY_B":
                b,
            "DISTANCE":
                float(
                    dist[i, j]
                ),
        })


write_csv_rows(
    OUT
    / "V15_DISTANCE_MATRIX.csv",
    distance_rows,
)


edge_rows = []
normalized_ranks = []

eligible_primary_edges = []

for parent, child in PRIMARY_EDGES:
    if (
        parent not in index
        or child not in index
    ):
        edge_rows.append({
            "EDGE_TYPE":
                "PRIMARY",
            "PARENT":
                parent,
            "CHILD":
                child,
            "STATUS":
                "MISSING_SIGNATURE",
            "CHILD_DISTANCE":
                "",
            "CHILD_RANK":
                "",
            "NORMALIZED_RANK":
                "",
        })
        continue

    pi = index[parent]

    candidates = [
        e
        for e in valid_entities
        if e != parent
    ]

    ordered = sorted(
        candidates,
        key=lambda e:
            dist[
                pi,
                index[e]
            ],
    )

    rank = (
        ordered.index(
            child
        )
        + 1
    )

    norm_rank = (
        (rank - 1)
        / max(
            1,
            len(ordered) - 1,
        )
    )

    normalized_ranks.append(
        norm_rank
    )

    eligible_primary_edges.append(
        (
            parent,
            child,
        )
    )

    edge_rows.append({
        "EDGE_TYPE":
            "PRIMARY",
        "PARENT":
            parent,
        "CHILD":
            child,
        "STATUS":
            "ELIGIBLE",
        "CHILD_DISTANCE":
            float(
                dist[
                    pi,
                    index[child]
                ]
            ),
        "CHILD_RANK":
            rank,
        "NORMALIZED_RANK":
            float(
                norm_rank
            ),
    })


for parent, child in SECONDARY_EDGES:
    if (
        parent not in index
        or child not in index
    ):
        status = "MISSING_SIGNATURE"
        d = ""
    else:
        status = "ELIGIBLE"
        d = float(
            dist[
                index[parent],
                index[child]
            ]
        )

    edge_rows.append({
        "EDGE_TYPE":
            "SECONDARY_RELATION",
        "PARENT":
            parent,
        "CHILD":
            child,
        "STATUS":
            status,
        "CHILD_DISTANCE":
            d,
        "CHILD_RANK":
            "",
        "NORMALIZED_RANK":
            "",
    })


write_csv_rows(
    OUT
    / "V15_EDGE_TESTS.csv",
    edge_rows,
)


rng = np.random.default_rng(
    SEED
)

observed_mean_rank = (
    float(
        np.mean(
            normalized_ranks
        )
    )
    if normalized_ranks
    else None
)

null_stats = []

if eligible_primary_edges:
    eligible_pool = list(
        valid_entities
    )

    for _ in range(10000):
        trial = []

        for parent, child in eligible_primary_edges:
            candidates = [
                e
                for e in eligible_pool
                if e != parent
            ]

            random_child = candidates[
                int(
                    rng.integers(
                        0,
                        len(candidates),
                    )
                )
            ]

            pi = index[parent]

            ordered = sorted(
                candidates,
                key=lambda e:
                    dist[
                        pi,
                        index[e]
                    ],
            )

            rank = (
                ordered.index(
                    random_child
                )
                + 1
            )

            norm_rank = (
                (rank - 1)
                / max(
                    1,
                    len(ordered) - 1,
                )
            )

            trial.append(
                norm_rank
            )

        null_stats.append(
            float(
                np.mean(
                    trial
                )
            )
        )


edge_p = None

if (
    observed_mean_rank is not None
    and null_stats
):
    edge_p = (
        1
        + sum(
            x <= observed_mean_rank
            for x in null_stats
        )
    ) / (
        1
        + len(
            null_stats
        )
    )


f1_order = {
    "status":
        "NOT_ELIGIBLE",
}


if all(
    entity in index
    for entity in F1_PATH
):
    stages = F1_PATH[:-1]
    final = F1_PATH[-1]

    distances_to_final = [
        float(
            dist[
                index[e],
                index[final]
            ]
        )
        for e in stages
    ]

    rho = spearman(
        list(
            range(
                1,
                len(stages) + 1,
            )
        ),
        distances_to_final,
    )

    actual_cost = 0.0

    for a, b in zip(
        F1_PATH[:-1],
        F1_PATH[1:],
    ):
        actual_cost += float(
            dist[
                index[a],
                index[b]
            ]
        )


    permutation_costs = []

    for perm in itertools.permutations(
        stages
    ):
        path = list(
            perm
        ) + [
            final
        ]

        cost = 0.0

        for a, b in zip(
            path[:-1],
            path[1:],
        ):
            cost += float(
                dist[
                    index[a],
                    index[b]
                ]
            )

        permutation_costs.append(
            cost
        )


    exact_p = (
        sum(
            x <= actual_cost
            for x in permutation_costs
        )
        / len(
            permutation_costs
        )
    )


    f1_order = {
        "status":
            "ELIGIBLE",
        "path":
            F1_PATH,
        "distances_to_final":
            distances_to_final,
        "spearman_stage_vs_distance":
            rho,
        "actual_adjacent_distance_cost":
            actual_cost,
        "exact_permutations":
            len(
                permutation_costs
            ),
        "exact_order_p":
            exact_p,
    }


(
    OUT
    / "V15_F1_ORDER_TEST.json"
).write_text(
    json.dumps(
        f1_order,
        indent=2,
    ),
    encoding="utf-8",
)


summary = {
    "module":
        "V15_GENEALOGY_TEMPORAL_DIRECTIONALITY",
    "random_seed":
        SEED,
    "valid_entity_signatures":
        len(
            valid_entities
        ),
    "primary_edges_total":
        len(
            PRIMARY_EDGES
        ),
    "primary_edges_eligible":
        len(
            eligible_primary_edges
        ),
    "observed_mean_normalized_child_rank":
        observed_mean_rank,
    "edge_permutation_iterations":
        10000,
    "edge_permutation_p":
        edge_p,
    "f1_order":
        f1_order,
    "alpha":
        0.05,
}


(
    OUT
    / "V15_FINAL_SUMMARY.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)

print(
    json.dumps(
        summary,
        indent=2,
    )
)
