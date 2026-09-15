from pathlib import Path
import csv
import json
import math
import hashlib
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

EXP = Path(
    r"C:\RSOS\RSOS_EXPERIMENTS_WF1_A_V12_R_CORRECTIVE_CAUSAL_TOPOLOGY"
)

OUT = EXP / "OUTPUT"

SENS = Path(__file__).resolve().parent

RAW = OUT / "01_RAW_RESPONSES.jsonl"
ERR = OUT / "02_ERRORS.jsonl"
MISSING = OUT / "15_MISSING_CELLS.csv"
COMP = OUT / "00A_VALID_TEXT_COMPLETENESS.json"
NATIVE_RAW = OUT / "26_NATIVE_RAW.jsonl"
NATIVE_MISSING = OUT / "26_NATIVE_MISSING_CELLS.csv"
RUNNER = OUT / "00_FROZEN_RUNNER.py"

EXPECTED_RUNNER_SHA = (
    "17c25511463d842b7b312aa3bf31bb31703cb1d2336bcaa6f8d5915bb016a1f2"
)

EXPECTED_MAIN = 8160
EXPECTED_NATIVE = 720
EXPECTED_MAIN_VALID = 8131
EXPECTED_MAIN_MISSING = 29

SEEDS = list(range(1080, 1160))
REPS = [0, 1, 2]

N_MONTE_CARLO = 100000
RNG_SEED = 91820260913


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def load_jsonl(path):
    rows = []

    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


def holm_adjust(p_values):
    n = len(p_values)

    order = sorted(
        range(n),
        key=lambda i: p_values[i]
    )

    adjusted = [1.0] * n
    running = 0.0

    for rank, idx in enumerate(order):
        value = min(
            1.0,
            (n - rank) * p_values[idx]
        )

        running = max(running, value)
        adjusted[idx] = running

    return adjusted


def logcomb(n, k):
    if k < 0 or k > n:
        return float("-inf")

    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
    )


def hypergeom_upper_tail(N, K, draws, observed):
    upper = min(K, draws)

    logs = []

    denom = logcomb(N, draws)

    for x in range(observed, upper + 1):
        if draws - x > N - K:
            continue

        lp = (
            logcomb(K, x)
            + logcomb(N - K, draws - x)
            - denom
        )

        logs.append(lp)

    if not logs:
        return 0.0

    m = max(logs)

    return min(
        1.0,
        math.exp(m)
        * sum(math.exp(v - m) for v in logs)
    )


def chi_square_stat(counts):
    counts = np.asarray(counts, dtype=float)

    expected = np.full(
        len(counts),
        counts.sum() / len(counts)
    )

    mask = expected > 0

    return float(
        np.sum(
            ((counts[mask] - expected[mask]) ** 2)
            /
            expected[mask]
        )
    )


# ============================================================
# VERIFY INPUT STATE
# ============================================================

required = [
    RAW,
    ERR,
    MISSING,
    COMP,
    NATIVE_RAW,
    NATIVE_MISSING,
    RUNNER,
]

for p in required:
    if not p.exists():
        raise RuntimeError(
            f"Required file missing: {p}"
        )

runner_sha = sha256_file(RUNNER)

if runner_sha.lower() != EXPECTED_RUNNER_SHA.lower():
    raise RuntimeError(
        "Frozen runner SHA mismatch"
    )

completeness = json.loads(
    COMP.read_text(encoding="utf-8")
)

if int(completeness["successful_main_cells"]) != EXPECTED_MAIN_VALID:
    raise RuntimeError(
        "Unexpected final main valid-cell count"
    )

if int(completeness["missing_main_cells"]) != EXPECTED_MAIN_MISSING:
    raise RuntimeError(
        "Unexpected final missing-cell count"
    )

if int(completeness["successful_native_cells"]) != EXPECTED_NATIVE:
    raise RuntimeError(
        "Unexpected native valid-cell count"
    )


# ============================================================
# LOAD MAIN SUCCESS DATA
# ============================================================

raw_rows = load_jsonl(RAW)

success = {}

for row in raw_rows:
    key = str(row["cell_key"])

    if key not in success:
        success[key] = row


# ============================================================
# LOAD FINAL MISSING CELLS
# ============================================================

missing_df = pd.read_csv(MISSING)

missing_keys = set(
    missing_df["CELL_KEY"].astype(str)
)

if len(missing_keys) != EXPECTED_MAIN_MISSING:
    raise RuntimeError(
        f"Expected 29 missing cells, got {len(missing_keys)}"
    )


# ============================================================
# DERIVE CONDITIONS FROM COMPLETE OBSERVED+MISSING UNIVERSE
# ============================================================

all_keys_seen = set(success) | missing_keys

conditions = sorted({
    k.split("|")[1]
    for k in all_keys_seen
})

if len(conditions) != 34:
    raise RuntimeError(
        f"Expected 34 conditions, got {len(conditions)}"
    )

expected_keys = {
    f"{seed}|{condition}|{rep}"
    for seed in SEEDS
    for condition in conditions
    for rep in REPS
}

if len(expected_keys) != EXPECTED_MAIN:
    raise RuntimeError(
        f"Expected universe size != {EXPECTED_MAIN}"
    )

if len(success) != EXPECTED_MAIN_VALID:
    raise RuntimeError(
        f"Unique success count = {len(success)}, expected {EXPECTED_MAIN_VALID}"
    )

overlap = set(success) & missing_keys

if overlap:
    raise RuntimeError(
        f"Success/missing overlap detected: {len(overlap)}"
    )

unaccounted = (
    expected_keys
    -
    set(success)
    -
    missing_keys
)

unexpected = (
    set(success)
    |
    missing_keys
) - expected_keys

if unaccounted:
    raise RuntimeError(
        f"Unaccounted expected cells: {len(unaccounted)}"
    )

if unexpected:
    raise RuntimeError(
        f"Unexpected cell keys: {len(unexpected)}"
    )


# ============================================================
# LOAD ERROR HISTORY
# ============================================================

error_rows = load_jsonl(ERR)

error_records_by_cell = defaultdict(list)

for row in error_rows:
    error_records_by_cell[
        str(row["cell_key"])
    ].append(row)

persistent_rows = []

for key in sorted(missing_keys):
    parts = key.split("|")

    records = error_records_by_cell[key]

    failures = []

    for record in records:
        failures.extend(
            record.get("failures", [])
        )

    cats = sorted({
        str(x.get("category", ""))
        for x in failures
    })

    errors = sorted({
        str(x.get("error", ""))
        for x in failures
    })

    persistent_rows.append({
        "CELL_KEY": key,
        "SEED": int(parts[0]),
        "CONDITION": parts[1],
        "REP": int(parts[2]),
        "FAILED_COLLECTION_PASSES": len(records),
        "TOTAL_FAILED_API_ATTEMPTS": len(failures),
        "FAILURE_CATEGORIES": ";".join(cats),
        "FAILURE_ERRORS": ";".join(errors),
    })

persistent_df = pd.DataFrame(
    persistent_rows
)

persistent_df.to_csv(
    SENS / "01_PERSISTENT_RESIDUE.csv",
    index=False
)


# ============================================================
# CONDITION-LEVEL MISSINGNESS
# ============================================================

condition_missing = Counter(
    key.split("|")[1]
    for key in missing_keys
)

condition_rows = []

raw_ps = []

for condition in conditions:
    m = int(
        condition_missing.get(
            condition,
            0
        )
    )

    expected_n = len(SEEDS) * len(REPS)

    p = hypergeom_upper_tail(
        EXPECTED_MAIN,
        expected_n,
        EXPECTED_MAIN_MISSING,
        m
    )

    raw_ps.append(p)

    condition_rows.append({
        "CONDITION": condition,
        "EXPECTED_CELLS": expected_n,
        "VALID_CELLS": expected_n - m,
        "MISSING_CELLS": m,
        "MISSING_RATE": m / expected_n,
        "ENRICHMENT_P_RAW": p,
    })

condition_adjusted = holm_adjust(
    raw_ps
)

for row, adj in zip(
    condition_rows,
    condition_adjusted
):
    row["ENRICHMENT_P_HOLM"] = adj

condition_df = pd.DataFrame(
    condition_rows
).sort_values(
    ["MISSING_CELLS", "CONDITION"],
    ascending=[False, True]
)

condition_df.to_csv(
    SENS / "02_MISSINGNESS_BY_CONDITION.csv",
    index=False
)


# ============================================================
# SEED-LEVEL MISSINGNESS
# ============================================================

seed_missing = Counter(
    int(key.split("|")[0])
    for key in missing_keys
)

seed_rows = []

for seed in SEEDS:
    m = int(
        seed_missing.get(
            seed,
            0
        )
    )

    expected_n = (
        len(conditions)
        *
        len(REPS)
    )

    seed_rows.append({
        "SEED": seed,
        "EXPECTED_CELLS": expected_n,
        "VALID_CELLS": expected_n - m,
        "MISSING_CELLS": m,
        "MISSING_RATE": m / expected_n,
    })

seed_df = pd.DataFrame(
    seed_rows
).sort_values(
    ["MISSING_CELLS", "SEED"],
    ascending=[False, True]
)

seed_df.to_csv(
    SENS / "03_MISSINGNESS_BY_SEED.csv",
    index=False
)


# ============================================================
# REP-LEVEL MISSINGNESS
# ============================================================

rep_missing = Counter(
    int(key.split("|")[2])
    for key in missing_keys
)

rep_rows = []

for rep in REPS:
    m = int(
        rep_missing.get(
            rep,
            0
        )
    )

    expected_n = (
        len(SEEDS)
        *
        len(conditions)
    )

    rep_rows.append({
        "REP": rep,
        "EXPECTED_CELLS": expected_n,
        "VALID_CELLS": expected_n - m,
        "MISSING_CELLS": m,
        "MISSING_RATE": m / expected_n,
    })

rep_df = pd.DataFrame(
    rep_rows
)

rep_df.to_csv(
    SENS / "04_MISSINGNESS_BY_REP.csv",
    index=False
)


# ============================================================
# SEED x CONDITION REPLICATE CLUSTERING
# ============================================================

group_counts = Counter()

for key in missing_keys:
    seed, condition, rep = key.split("|")

    group_counts[
        (int(seed), condition)
    ] += 1

group_rows = []

for seed in SEEDS:
    for condition in conditions:
        m = int(
            group_counts.get(
                (seed, condition),
                0
            )
        )

        group_rows.append({
            "SEED": seed,
            "CONDITION": condition,
            "EXPECTED_REPS": 3,
            "MISSING_REPS": m,
            "VALID_REPS": 3 - m,
        })

group_df = pd.DataFrame(
    group_rows
).sort_values(
    ["MISSING_REPS", "SEED", "CONDITION"],
    ascending=[False, True, True]
)

group_df.to_csv(
    SENS / "05_SEED_CONDITION_REPLICATE_CLUSTERS.csv",
    index=False
)

observed_triples = int(
    (group_df["MISSING_REPS"] == 3).sum()
)

observed_doubles_or_more = int(
    (group_df["MISSING_REPS"] >= 2).sum()
)

observed_max_group = int(
    group_df["MISSING_REPS"].max()
)


# ============================================================
# GLOBAL CONCENTRATION STATISTICS
# ============================================================

condition_counts = np.array([
    condition_missing.get(c, 0)
    for c in conditions
])

seed_counts = np.array([
    seed_missing.get(s, 0)
    for s in SEEDS
])

rep_counts = np.array([
    rep_missing.get(r, 0)
    for r in REPS
])

obs_condition_chi = chi_square_stat(
    condition_counts
)

obs_seed_chi = chi_square_stat(
    seed_counts
)

obs_rep_chi = chi_square_stat(
    rep_counts
)

obs_max_condition = int(
    condition_counts.max()
)

obs_max_seed = int(
    seed_counts.max()
)


# ============================================================
# MONTE CARLO UNDER UNIFORM CELL MISSINGNESS
#
# Null:
# exactly 29 of the 8160 cells are missing,
# uniformly without replacement.
#
# This does NOT assert MCAR is true.
# It asks how concentrated the observed residue is
# relative to a uniform-missing-cell null.
# ============================================================

rng = np.random.default_rng(
    RNG_SEED
)

condition_index = {
    c: i
    for i, c in enumerate(conditions)
}

flat_condition = np.empty(
    EXPECTED_MAIN,
    dtype=np.int16
)

flat_seed = np.empty(
    EXPECTED_MAIN,
    dtype=np.int16
)

flat_rep = np.empty(
    EXPECTED_MAIN,
    dtype=np.int8
)

flat_group = np.empty(
    EXPECTED_MAIN,
    dtype=np.int32
)

idx = 0
group_id = 0

for si, seed in enumerate(SEEDS):
    for ci, condition in enumerate(conditions):
        for ri, rep in enumerate(REPS):
            flat_condition[idx] = ci
            flat_seed[idx] = si
            flat_rep[idx] = ri
            flat_group[idx] = group_id
            idx += 1

        group_id += 1

if idx != EXPECTED_MAIN:
    raise RuntimeError(
        f"Flattened design generated {idx} cells"
    )

sim_condition_chi_ge = 0
sim_seed_chi_ge = 0
sim_rep_chi_ge = 0

sim_max_condition_ge = 0
sim_max_seed_ge = 0

sim_triples_ge = 0
sim_doubles_ge = 0
sim_max_group_ge = 0

for simulation in range(
    N_MONTE_CARLO
):
    selected = rng.choice(
        EXPECTED_MAIN,
        size=EXPECTED_MAIN_MISSING,
        replace=False
    )

    cc = np.bincount(
        flat_condition[selected],
        minlength=len(conditions)
    )

    sc = np.bincount(
        flat_seed[selected],
        minlength=len(SEEDS)
    )

    rc = np.bincount(
        flat_rep[selected],
        minlength=len(REPS)
    )

    gc = np.bincount(
        flat_group[selected],
        minlength=len(SEEDS) * len(conditions)
    )

    if chi_square_stat(cc) >= obs_condition_chi - 1e-12:
        sim_condition_chi_ge += 1

    if chi_square_stat(sc) >= obs_seed_chi - 1e-12:
        sim_seed_chi_ge += 1

    if chi_square_stat(rc) >= obs_rep_chi - 1e-12:
        sim_rep_chi_ge += 1

    if int(cc.max()) >= obs_max_condition:
        sim_max_condition_ge += 1

    if int(sc.max()) >= obs_max_seed:
        sim_max_seed_ge += 1

    triples = int(
        np.sum(gc == 3)
    )

    doubles = int(
        np.sum(gc >= 2)
    )

    max_group = int(
        gc.max()
    )

    if triples >= observed_triples:
        sim_triples_ge += 1

    if doubles >= observed_doubles_or_more:
        sim_doubles_ge += 1

    if max_group >= observed_max_group:
        sim_max_group_ge += 1


def mc_p(count):
    return (
        count + 1
    ) / (
        N_MONTE_CARLO + 1
    )


# ============================================================
# SUMMARY / INTERPRETATION BOUNDARY
# ============================================================

overall_missing_rate = (
    EXPECTED_MAIN_MISSING
    /
    EXPECTED_MAIN
)

max_condition_missing = int(
    condition_df["MISSING_CELLS"].max()
)

max_condition_rate = float(
    condition_df["MISSING_RATE"].max()
)

max_seed_missing = int(
    seed_df["MISSING_CELLS"].max()
)

max_seed_rate = float(
    seed_df["MISSING_RATE"].max()
)

summary = {
    "experiment":
        "WF1-A V12-R R1",

    "analysis":
        "S1 persistent-missingness sensitivity",

    "api_calls":
        0,

    "runner_sha256":
        runner_sha,

    "design": {
        "seeds":
            len(SEEDS),

        "conditions":
            len(conditions),

        "reps":
            len(REPS),

        "expected_main_cells":
            EXPECTED_MAIN,

        "valid_main_cells":
            EXPECTED_MAIN_VALID,

        "persistent_missing_cells":
            EXPECTED_MAIN_MISSING,

        "overall_missing_rate":
            overall_missing_rate,

        "native_valid":
            720,

        "native_missing":
            0,
    },

    "persistent_failure_history": {
        "all_cells_failed_collection_passes":
            sorted(
                persistent_df[
                    "FAILED_COLLECTION_PASSES"
                ].unique().tolist()
            ),

        "all_cells_failed_api_attempts":
            sorted(
                persistent_df[
                    "TOTAL_FAILED_API_ATTEMPTS"
                ].unique().tolist()
            ),

        "failure_categories":
            sorted(
                persistent_df[
                    "FAILURE_CATEGORIES"
                ].unique().tolist()
            ),
    },

    "descriptive_concentration": {
        "max_missing_in_any_condition":
            max_condition_missing,

        "max_condition_missing_rate":
            max_condition_rate,

        "max_missing_in_any_seed":
            max_seed_missing,

        "max_seed_missing_rate":
            max_seed_rate,

        "seed_condition_groups_with_3_of_3_missing":
            observed_triples,

        "seed_condition_groups_with_at_least_2_missing":
            observed_doubles_or_more,
    },

    "uniform_missing_cell_null": {
        "monte_carlo_draws":
            N_MONTE_CARLO,

        "rng_seed":
            RNG_SEED,

        "condition_distribution_chi_square_stat":
            obs_condition_chi,

        "condition_distribution_mc_p":
            mc_p(sim_condition_chi_ge),

        "seed_distribution_chi_square_stat":
            obs_seed_chi,

        "seed_distribution_mc_p":
            mc_p(sim_seed_chi_ge),

        "rep_distribution_chi_square_stat":
            obs_rep_chi,

        "rep_distribution_mc_p":
            mc_p(sim_rep_chi_ge),

        "max_condition_count_mc_p":
            mc_p(sim_max_condition_ge),

        "max_seed_count_mc_p":
            mc_p(sim_max_seed_ge),

        "triple_rep_cluster_mc_p":
            mc_p(sim_triples_ge),

        "double_or_more_cluster_mc_p":
            mc_p(sim_doubles_ge),

        "max_seed_condition_cluster_mc_p":
            mc_p(sim_max_group_ge),
    },

    "interpretation_boundary": [
        (
            "This analysis tests concentration of persistent "
            "missingness only."
        ),
        (
            "It does not test the RSOS scientific hypothesis."
        ),
        (
            "A small Monte Carlo p-value means the 29 missing "
            "cells are more structurally concentrated than a "
            "uniform 29-of-8160 missing-cell null."
        ),
        (
            "It does not establish why those cells failed."
        ),
        (
            "The 29 cells remain missing and are not imputed."
        ),
        (
            "No additional R1 sampling is permitted."
        ),
        (
            "Outcome-level sensitivity must be evaluated "
            "separately."
        ),
    ],
}

summary_path = (
    SENS
    /
    "06_S1_MISSINGNESS_SUMMARY.json"
)

summary_path.write_text(
    json.dumps(
        summary,
        indent=2
    ),
    encoding="utf-8"
)


# ============================================================
# HUMAN-READABLE REPORT
# ============================================================

top_conditions = (
    condition_df[
        [
            "CONDITION",
            "MISSING_CELLS",
            "VALID_CELLS",
            "MISSING_RATE",
            "ENRICHMENT_P_RAW",
            "ENRICHMENT_P_HOLM",
        ]
    ]
    .head(15)
    .to_string(index=False)
)

clustered = (
    group_df[
        group_df["MISSING_REPS"] >= 2
    ][
        [
            "SEED",
            "CONDITION",
            "MISSING_REPS",
            "VALID_REPS",
        ]
    ]
    .to_string(index=False)
)

report = f"""
WF1-A V12-R R1
SENSITIVITY S1 — PERSISTENT MISSINGNESS

COLLECTION
==========
Expected main: {EXPECTED_MAIN}
Valid main: {EXPECTED_MAIN_VALID}
Persistent missing: {EXPECTED_MAIN_MISSING}
Missing rate: {overall_missing_rate:.8%}

Native:
720 / 720 valid

Persistent cells:
29 / 29 failed all 3 collection passes.
29 / 29 accumulated 24 failed API attempts.

TOP CONDITIONS BY PERSISTENT MISSINGNESS
========================================
{top_conditions}

SEED x CONDITION GROUPS WITH >=2 OF 3 REPS MISSING
===================================================
{clustered}

GLOBAL CONCENTRATION TESTS
==========================
Condition chi-square statistic:
{obs_condition_chi:.8f}

Condition-distribution Monte Carlo p:
{mc_p(sim_condition_chi_ge):.8g}

Seed chi-square statistic:
{obs_seed_chi:.8f}

Seed-distribution Monte Carlo p:
{mc_p(sim_seed_chi_ge):.8g}

Rep chi-square statistic:
{obs_rep_chi:.8f}

Rep-distribution Monte Carlo p:
{mc_p(sim_rep_chi_ge):.8g}

Observed max missing in one condition:
{obs_max_condition}

Max-condition-count Monte Carlo p:
{mc_p(sim_max_condition_ge):.8g}

Observed max missing in one seed:
{obs_max_seed}

Max-seed-count Monte Carlo p:
{mc_p(sim_max_seed_ge):.8g}

Observed seed×condition 3/3 missing clusters:
{observed_triples}

3/3 cluster Monte Carlo p:
{mc_p(sim_triples_ge):.8g}

Observed seed×condition >=2/3 missing clusters:
{observed_doubles_or_more}

>=2/3 cluster Monte Carlo p:
{mc_p(sim_doubles_ge):.8g}

INTERPRETATION BOUNDARY
=======================
This is a missingness-mechanism sensitivity audit.

It DOES NOT test whether the RSOS topology hypothesis is true.

It DOES test whether the final 29-cell residue is compatible with a
simple uniform missing-cell model.

No missing value is imputed.
No API call is made.
No runner, prompt, schema, seed, condition or validator is modified.
No additional R1 collection is permitted.
"""

(
    SENS
    /
    "07_S1_HUMAN_READABLE_REPORT.txt"
).write_text(
    report.strip() + "\n",
    encoding="utf-8"
)


# ============================================================
# HASH ALL S1 OUTPUTS
# ============================================================

manifest_rows = []

for p in sorted(
    SENS.iterdir(),
    key=lambda x: x.name
):
    if (
        p.is_file()
        and
        p.name
        !=
        "08_S1_SHA256_MANIFEST.csv"
    ):
        manifest_rows.append({
            "FILE":
                p.name,

            "LENGTH":
                p.stat().st_size,

            "SHA256":
                sha256_file(p),
        })

manifest_path = (
    SENS
    /
    "08_S1_SHA256_MANIFEST.csv"
)

pd.DataFrame(
    manifest_rows
).to_csv(
    manifest_path,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print(
    "============================================================"
)
print(
    " WF1-A SENSITIVITY S1 COMPLETE"
)
print(
    "============================================================"
)

print(
    f"Valid main               : "
    f"{EXPECTED_MAIN_VALID}/{EXPECTED_MAIN}"
)

print(
    f"Persistent missing       : "
    f"{EXPECTED_MAIN_MISSING}/{EXPECTED_MAIN}"
)

print(
    f"Overall missing rate     : "
    f"{overall_missing_rate:.6%}"
)

print(
    f"Max condition missing    : "
    f"{obs_max_condition}/240"
)

print(
    f"Condition global MC p    : "
    f"{mc_p(sim_condition_chi_ge):.8g}"
)

print(
    f"Max seed missing         : "
    f"{obs_max_seed}/102"
)

print(
    f"Seed global MC p         : "
    f"{mc_p(sim_seed_chi_ge):.8g}"
)

print(
    f"Rep global MC p          : "
    f"{mc_p(sim_rep_chi_ge):.8g}"
)

print(
    f"3/3 seed-condition groups: "
    f"{observed_triples}"
)

print(
    f"3/3 cluster MC p         : "
    f"{mc_p(sim_triples_ge):.8g}"
)

print(
    f">=2/3 groups             : "
    f"{observed_doubles_or_more}"
)

print(
    f">=2/3 cluster MC p       : "
    f"{mc_p(sim_doubles_ge):.8g}"
)

print()
print(
    "TOP CONDITION RESIDUES"
)
print(
    "----------------------"
)

print(
    condition_df[
        [
            "CONDITION",
            "MISSING_CELLS",
            "VALID_CELLS",
            "MISSING_RATE",
            "ENRICHMENT_P_HOLM",
        ]
    ]
    .head(15)
    .to_string(index=False)
)

print()
print(
    f"Sensitivity folder       : {SENS}"
)

print(
    f"Summary                  : {summary_path}"
)

print(
    f"Manifest                 : {manifest_path}"
)

print()
print(
    "NO API CALLS WERE MADE."
)

print(
    "NO R1 DATA WERE MODIFIED."
)

print(
    "NEXT: OUTCOME-LEVEL MISSINGNESS SENSITIVITY."
)

print(
    "============================================================"
)
