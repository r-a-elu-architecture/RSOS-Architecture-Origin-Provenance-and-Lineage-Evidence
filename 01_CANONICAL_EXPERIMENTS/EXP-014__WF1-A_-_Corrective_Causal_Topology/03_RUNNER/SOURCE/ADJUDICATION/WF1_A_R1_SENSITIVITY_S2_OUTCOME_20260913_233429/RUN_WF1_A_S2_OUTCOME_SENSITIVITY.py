from pathlib import Path
import importlib.util
import hashlib
import json
import shutil
from collections import Counter, defaultdict

import pandas as pd


# ============================================================
# PATHS / FROZEN BOUNDARIES
# ============================================================

EXP = Path(
    r"C:\RSOS\RSOS_EXPERIMENTS_WF1_A_V12_R_CORRECTIVE_CAUSAL_TOPOLOGY"
)

OUTPUT = EXP / "OUTPUT"

RUNNER = OUTPUT / "00_FROZEN_RUNNER.py"
RAW = OUTPUT / "01_RAW_RESPONSES.jsonl"
NATIVE_RAW = OUTPUT / "26_NATIVE_RAW.jsonl"
MISSING = OUTPUT / "15_MISSING_CELLS.csv"
COMPLETENESS = OUTPUT / "00A_VALID_TEXT_COMPLETENESS.json"

S2 = Path(__file__).resolve().parent

EXPECTED_RUNNER_SHA = (
    "17c25511463d842b7b312aa3bf31bb31703cb1d2336bcaa6f8d5915bb016a1f2"
)

COLLECTION_FREEZE_SHA = (
    "a28a5d017abb014d276605e405e70ca6b37624fe115263604074909a9fdfddaf"
)

S1_ZIP_SHA = (
    "03e93f81908db7b500374ede252360f41bbc09b57d3339b64a3ec25ecfe8532d"
)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def safe_get(obj, *keys):
    cur = obj

    for key in keys:
        if not isinstance(cur, dict):
            return None

        if key not in cur:
            return None

        cur = cur[key]

    return cur


# ============================================================
# VERIFY IMMUTABLE SOURCE STATE
# ============================================================

for p in [
    RUNNER,
    RAW,
    NATIVE_RAW,
    MISSING,
    COMPLETENESS,
]:
    if not p.exists():
        raise RuntimeError(
            f"Required source missing: {p}"
        )

runner_sha = sha256_file(
    RUNNER
)

if runner_sha.lower() != EXPECTED_RUNNER_SHA.lower():
    raise RuntimeError(
        "Frozen runner SHA mismatch."
    )

comp = json.loads(
    COMPLETENESS.read_text(
        encoding="utf-8"
    )
)

if int(
    comp["successful_main_cells"]
) != 8131:
    raise RuntimeError(
        "Expected final main valid = 8131."
    )

if int(
    comp["missing_main_cells"]
) != 29:
    raise RuntimeError(
        "Expected final main missing = 29."
    )

if int(
    comp["successful_native_cells"]
) != 720:
    raise RuntimeError(
        "Expected native valid = 720."
    )

if int(
    comp["missing_native_cells"]
) != 0:
    raise RuntimeError(
        "Expected native missing = 0."
    )


# ============================================================
# IMPORT EXACT FROZEN R1 AS A MODULE
#
# __name__ != "__main__", so main() is NOT executed.
# Therefore no API collection path is entered.
# ============================================================

spec = importlib.util.spec_from_file_location(
    "wf1a_frozen_r1",
    RUNNER
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        "Could not import frozen runner."
    )

m = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    m
)


# ============================================================
# VERIFY FROZEN DESIGN FROM THE RUNNER ITSELF
# ============================================================

original_seeds = list(
    m.SEED_IDS
)

original_dev = set(
    m.DEV
)

original_conf = set(
    m.CONF
)

original_hold = set(
    m.HOLD
)

conditions = list(
    m.CONDITIONS
)

n_reps = int(
    m.N_REPS
)

if len(original_seeds) != 80:
    raise RuntimeError(
        f"Expected 80 seeds, got {len(original_seeds)}"
    )

if len(conditions) != 34:
    raise RuntimeError(
        f"Expected 34 conditions, got {len(conditions)}"
    )

if n_reps != 3:
    raise RuntimeError(
        f"Expected 3 reps, got {n_reps}"
    )


# ============================================================
# LOAD FINAL OBSERVED CELLS
# ============================================================

successful_all = m.load_unique(
    RAW
)

native_all = m.load_unique(
    NATIVE_RAW
)

if len(successful_all) != 8131:
    raise RuntimeError(
        f"Unique main successes = {len(successful_all)}, expected 8131."
    )

if len(native_all) != 720:
    raise RuntimeError(
        f"Unique native successes = {len(native_all)}, expected 720."
    )


# ============================================================
# LOAD FINAL PERSISTENT RESIDUE
# ============================================================

missing_df = pd.read_csv(
    MISSING
)

missing_keys = set(
    missing_df[
        "CELL_KEY"
    ].astype(str)
)

if len(missing_keys) != 29:
    raise RuntimeError(
        f"Expected 29 final missing keys, got {len(missing_keys)}."
    )

missing_group_count = Counter()
missing_seed_count = Counter()

for key in missing_keys:
    seed_s, condition, rep_s = key.split("|")

    seed = int(seed_s)

    missing_group_count[
        (seed, condition)
    ] += 1

    missing_seed_count[
        seed
    ] += 1


# ============================================================
# DEFINE TWO NO-IMPUTATION SENSITIVITY SETS
# ============================================================

# S2-A:
# Remove only seeds for which at least one condition has
# all 3 replicates missing.
#
# Every retained seed-condition therefore has >=1 genuine
# observed replicate. No imputation.

triple_missing_seeds = {
    seed
    for (seed, condition), count
    in missing_group_count.items()
    if count == 3
}

s2a_seeds = [
    seed
    for seed in original_seeds
    if seed not in triple_missing_seeds
]


# S2-B:
# Remove every seed touched by even one persistent missing cell.
#
# Every retained seed has all:
# 34 conditions x 3 reps = 102 genuine observations.

any_missing_seeds = set(
    missing_seed_count
)

s2b_seeds = [
    seed
    for seed in original_seeds
    if seed not in any_missing_seeds
]


# ============================================================
# AUDIT SCENARIO SEED COUNTS
# ============================================================

scenario_seed_audit = []

for scenario, seeds in [
    (
        "S2A_GROUP_COMPLETE",
        s2a_seeds
    ),
    (
        "S2B_SEED_COMPLETE",
        s2b_seeds
    ),
]:
    ss = set(
        seeds
    )

    scenario_seed_audit.append({
        "SCENARIO":
            scenario,

        "TOTAL_SEEDS":
            len(seeds),

        "DEV_SEEDS":
            len(
                original_dev
                &
                ss
            ),

        "CONFIRMATION_SEEDS":
            len(
                original_conf
                &
                ss
            ),

        "FINAL_HOLDOUT_SEEDS":
            len(
                original_hold
                &
                ss
            ),
    })

pd.DataFrame(
    scenario_seed_audit
).to_csv(
    S2
    /
    "01_SCENARIO_SEED_COUNTS.csv",
    index=False
)


# ============================================================
# PERSISTENT-MISSINGNESS IMPACT MAP
# ============================================================

impact_rows = []

for condition in conditions:
    n = sum(
        1
        for key in missing_keys
        if key.split("|")[1]
        ==
        condition
    )

    edge_primary_status = ""

    if condition == "E7_DELETE":
        edge_primary_status = (
            "EXCLUDED_FROM_PURE9_DELETE"
        )

    elif condition in {
        "E7_REVERSE",
        "E9_REVERSE",
    }:
        edge_primary_status = (
            "EXCLUDED_FROM_PURE8_REVERSE"
        )

    elif (
        condition.startswith("E")
        and
        condition.endswith("_DELETE")
    ):
        edge_primary_status = (
            "IN_PURE9_DELETE"
        )

    elif (
        condition.startswith("E")
        and
        condition.endswith("_REVERSE")
    ):
        edge_primary_status = (
            "IN_PURE8_REVERSE"
        )

    elif condition in {
        "R_DEGREE_MATCHED_REWIRE",
        "X_COMPLEXITY_MATCHED",
        "Y_TOPOLOGY_MATCHED",
    }:
        edge_primary_status = (
            "HARD_CLASSIFIER_CONTROL"
        )

    impact_rows.append({
        "CONDITION":
            condition,

        "PERSISTENT_MISSING":
            n,

        "EDGE_OR_HARD_CONTROL_ROLE":
            edge_primary_status,
    })

impact_df = pd.DataFrame(
    impact_rows
).sort_values(
    [
        "PERSISTENT_MISSING",
        "CONDITION",
    ],
    ascending=[
        False,
        True,
    ]
)

impact_df.to_csv(
    S2
    /
    "02_MISSINGNESS_PRIMARY_IMPACT_MAP.csv",
    index=False
)


# ============================================================
# RUN ONE SENSITIVITY SCENARIO USING EXACT FROZEN ANALYSIS
# ============================================================

def run_scenario(
    scenario_name,
    seeds
):
    scenario_out = (
        S2
        /
        scenario_name
    )

    scenario_out.mkdir(
        parents=True,
        exist_ok=True
    )

    seed_set = set(
        seeds
    )

    dev = (
        original_dev
        &
        seed_set
    )

    conf = (
        original_conf
        &
        seed_set
    )

    hold = (
        original_hold
        &
        seed_set
    )

    if len(dev) < 5:
        raise RuntimeError(
            f"{scenario_name}: too few DEV seeds."
        )

    if len(conf) < 5:
        raise RuntimeError(
            f"{scenario_name}: too few CONF seeds."
        )

    if len(hold) < 5:
        raise RuntimeError(
            f"{scenario_name}: too few HOLD seeds."
        )

    main = {
        key: row
        for key, row
        in successful_all.items()
        if int(
            row["seed"]
        )
        in seed_set
    }

    native = {
        key: row
        for key, row
        in native_all.items()
        if int(
            row["seed"]
        )
        in seed_set
    }

    # --------------------------------------------------------
    # Verify every retained seed-condition has >=1 real cell.
    # --------------------------------------------------------

    group_counts = defaultdict(
        int
    )

    for row in main.values():
        group_counts[
            (
                int(
                    row["seed"]
                ),
                row["condition"],
            )
        ] += 1

    absent_groups = []

    for seed in seeds:
        for condition in conditions:
            if (
                group_counts[
                    (
                        seed,
                        condition
                    )
                ]
                ==
                0
            ):
                absent_groups.append(
                    (
                        seed,
                        condition
                    )
                )

    if absent_groups:
        raise RuntimeError(
            f"{scenario_name}: "
            f"{len(absent_groups)} completely absent "
            f"seed-condition groups remain."
        )

    # --------------------------------------------------------
    # Balanced-seed scenario must be completely observed.
    # --------------------------------------------------------

    if (
        scenario_name
        ==
        "S2B_SEED_COMPLETE"
    ):
        expected_main = (
            len(seeds)
            *
            len(conditions)
            *
            n_reps
        )

        if len(main) != expected_main:
            raise RuntimeError(
                f"{scenario_name}: expected "
                f"{expected_main} complete main cells, "
                f"found {len(main)}."
            )

    # --------------------------------------------------------
    # Patch only ANALYSIS SAMPLE GLOBALS.
    #
    # We are not modifying:
    # feature definitions,
    # contrasts,
    # classifiers,
    # thresholds,
    # multiplicity,
    # interventions,
    # payload definitions,
    # or raw observations.
    # --------------------------------------------------------

    m.SEED_IDS = list(
        seeds
    )

    m.DEV = set(
        dev
    )

    m.CONF = set(
        conf
    )

    m.HOLD = set(
        hold
    )

    m.EXPECTED_MAIN_CELLS = (
        len(seeds)
        *
        len(conditions)
        *
        n_reps
    )

    m.EXPECTED_NATIVE_CELLS = (
        len(seeds)
        *
        n_reps
        *
        len(
            m.NATIVE_FRAMES
        )
    )

    m.OUT = scenario_out

    # Explicitly preserve frozen permutation count.
    m.N_PERMUTATIONS = 20000

    # --------------------------------------------------------
    # Rebuild exact frozen deterministic payload/meta caches.
    # No API call occurs here.
    # --------------------------------------------------------

    _, payload_cache, meta_cache = (
        m.build_caches()
    )

    # --------------------------------------------------------
    # Execute exact frozen downstream analysis functions.
    # --------------------------------------------------------

    summary = m.run_analysis(
        main,
        meta_cache,
        payload_cache,
        native
    )

    metadata = {
        "scenario":
            scenario_name,

        "runner_sha256":
            runner_sha,

        "no_api_calls":
            True,

        "no_imputation":
            True,

        "included_seeds":
            list(
                seeds
            ),

        "seed_counts": {
            "all":
                len(seeds),

            "development":
                len(dev),

            "confirmation":
                len(conf),

            "final_holdout":
                len(hold),
        },

        "main_observed_cells":
            len(main),

        "native_observed_cells":
            len(native),

        "seed_condition_groups":
            len(
                seeds
            )
            *
            len(
                conditions
            ),

        "completely_absent_seed_condition_groups":
            0,

        "analysis_method":
            (
                "Exact frozen R1 run_analysis() "
                "with sensitivity sample deletion only."
            ),
    }

    (
        scenario_out
        /
        "00_SENSITIVITY_SCENARIO_METADATA.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8"
    )

    return {
        "name":
            scenario_name,

        "seeds":
            list(
                seeds
            ),

        "dev":
            dev,

        "conf":
            conf,

        "hold":
            hold,

        "main_n":
            len(main),

        "native_n":
            len(native),

        "summary":
            summary,

        "out":
            scenario_out,
    }


# ============================================================
# EXECUTE S2-A AND S2-B
# ============================================================

results = []

results.append(
    run_scenario(
        "S2A_GROUP_COMPLETE",
        s2a_seeds
    )
)

results.append(
    run_scenario(
        "S2B_SEED_COMPLETE",
        s2b_seeds
    )
)


# ============================================================
# COMPARISON TABLE
# ============================================================

comparison_rows = []

for r in results:
    s = r[
        "summary"
    ]

    comparison_rows.append({
        "SCENARIO":
            r["name"],

        "SEEDS":
            len(
                r["seeds"]
            ),

        "DEV":
            len(
                r["dev"]
            ),

        "CONF":
            len(
                r["conf"]
            ),

        "HOLD":
            len(
                r["hold"]
            ),

        "OBSERVED_MAIN_CELLS":
            r["main_n"],

        "CLASSIFICATION":
            safe_get(
                s,
                "classification"
            ),

        "PRIMARY_STRONG_GATE":
            safe_get(
                s,
                "primary_strong_gate"
            ),

        "HARD_AUC_CONFIRMATION":
            safe_get(
                s,
                "hard_control_auc",
                "confirmation"
            ),

        "HARD_AUC_FINAL":
            safe_get(
                s,
                "hard_control_auc",
                "final_holdout"
            ),

        "PURE9_DELETE_CONF_EXCESS":
            safe_get(
                s,
                "pure9_delete",
                "confirmation",
                "excess"
            ),

        "PURE9_DELETE_CONF_HOLM_P":
            safe_get(
                s,
                "pure9_delete",
                "confirmation",
                "holm_p"
            ),

        "PURE9_DELETE_FINAL_EXCESS":
            safe_get(
                s,
                "pure9_delete",
                "final_holdout",
                "excess"
            ),

        "PURE9_DELETE_FINAL_HOLM_P":
            safe_get(
                s,
                "pure9_delete",
                "final_holdout",
                "holm_p"
            ),

        "PURE8_REVERSE_CONF_EXCESS":
            safe_get(
                s,
                "pure8_reverse",
                "confirmation",
                "excess"
            ),

        "PURE8_REVERSE_CONF_HOLM_P":
            safe_get(
                s,
                "pure8_reverse",
                "confirmation",
                "holm_p"
            ),

        "PURE8_REVERSE_FINAL_EXCESS":
            safe_get(
                s,
                "pure8_reverse",
                "final_holdout",
                "excess"
            ),

        "PURE8_REVERSE_FINAL_HOLM_P":
            safe_get(
                s,
                "pure8_reverse",
                "final_holdout",
                "holm_p"
            ),

        "NATIVE_TRIANGULATED":
            safe_get(
                s,
                "native_triangulated"
            ),
    })

comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df.to_csv(
    S2
    /
    "03_S2_OUTCOME_COMPARISON.csv",
    index=False
)


# ============================================================
# ROBUSTNESS ADJUDICATION
# ============================================================

def truthy(v):
    return (
        v is True
        or
        str(v).lower()
        ==
        "true"
    )


strong_both = all(
    truthy(
        row[
            "PRIMARY_STRONG_GATE"
        ]
    )
    for row in comparison_rows
)

auc_gate_both = all(
    (
        row[
            "HARD_AUC_CONFIRMATION"
        ]
        is not None
        and
        float(
            row[
                "HARD_AUC_CONFIRMATION"
            ]
        )
        >=
        0.85
        and
        row[
            "HARD_AUC_FINAL"
        ]
        is not None
        and
        float(
            row[
                "HARD_AUC_FINAL"
            ]
        )
        >=
        0.85
    )
    for row in comparison_rows
)

same_classification = (
    len({
        str(
            row[
                "CLASSIFICATION"
            ]
        )
        for row
        in comparison_rows
    })
    ==
    1
)

robustness = {
    "analysis":
        "WF1-A S2 outcome-level structured-missingness sensitivity",

    "runner_sha256":
        runner_sha,

    "collection_freeze_zip_sha256":
        COLLECTION_FREEZE_SHA,

    "s1_zip_sha256":
        S1_ZIP_SHA,

    "api_calls":
        0,

    "imputation":
        False,

    "scenarios": {
        "S2A_GROUP_COMPLETE": {
            "description":
                (
                    "Remove seeds with any seed-condition "
                    "having 3/3 persistent missing reps. "
                    "Retained seed-condition groups use "
                    "only genuinely observed reps."
                ),

            "excluded_seeds":
                sorted(
                    triple_missing_seeds
                ),
        },

        "S2B_SEED_COMPLETE": {
            "description":
                (
                    "Remove every seed touched by any "
                    "persistent missing cell. Every retained "
                    "seed is fully balanced across all "
                    "34 conditions x 3 reps."
                ),

            "excluded_seeds":
                sorted(
                    any_missing_seeds
                ),
        },
    },

    "same_classification_across_sensitivities":
        same_classification,

    "strong_primary_gate_passes_both":
        strong_both,

    "hard_auc_085_gate_passes_both":
        auc_gate_both,

    "interpretation": (
        "If classification and strong/hard-control gates "
        "survive both deletion sensitivities, the core "
        "WF1-A conclusion is robust to the observed "
        "structured 29-cell residue under these "
        "no-imputation sensitivity analyses. "
        "If they do not survive, missingness materially "
        "affects scientific adjudication."
    ),

    "limitations": [
        (
            "This does not recover or infer the 29 "
            "unobserved API outputs."
        ),
        (
            "This does not prove missingness mechanism."
        ),
        (
            "This does not establish global uniqueness."
        ),
        (
            "This does not establish literal neural or "
            "weight-level topology."
        ),
    ],
}

(
    S2
    /
    "04_S2_ROBUSTNESS_ADJUDICATION.json"
).write_text(
    json.dumps(
        robustness,
        indent=2
    ),
    encoding="utf-8"
)


# ============================================================
# HUMAN-READABLE REPORT
# ============================================================

report = []

report.append(
    "WF1-A V12-R R1"
)

report.append(
    "S2 OUTCOME-LEVEL STRUCTURED-MISSINGNESS SENSITIVITY"
)

report.append(
    "=" * 72
)

report.append("")

report.append(
    f"Frozen runner SHA256: {runner_sha}"
)

report.append(
    "API calls: 0"
)

report.append(
    "Imputation: NONE"
)

report.append(
    "Additional R1 collection: NONE"
)

report.append("")

report.append(
    f"Original final observed main cells: 8131 / 8160"
)

report.append(
    f"Persistent missing cells: 29 / 8160"
)

report.append("")

report.append(
    f"Seeds with any 3/3 missing condition: "
    f"{sorted(triple_missing_seeds)}"
)

report.append(
    f"Seeds touched by any missing cell: "
    f"{sorted(any_missing_seeds)}"
)

report.append("")

report.append(
    "SCENARIO COMPARISON"
)

report.append(
    "-" * 72
)

report.append(
    comparison_df.to_string(
        index=False
    )
)

report.append("")

report.append(
    "ROBUSTNESS GATES"
)

report.append(
    "-" * 72
)

report.append(
    f"Same classification across S2-A/S2-B: "
    f"{same_classification}"
)

report.append(
    f"Strong primary gate passes both: "
    f"{strong_both}"
)

report.append(
    f"Hard-control AUC >= 0.85 in CONF+HOLD for both: "
    f"{auc_gate_both}"
)

report.append("")

report.append(
    "INTERPRETATION BOUNDARY"
)

report.append(
    "-" * 72
)

report.append(
    "These are deletion sensitivities using exact frozen R1 "
    "analysis functions."
)

report.append(
    "No missing response is fabricated, imputed, replaced, "
    "or resampled."
)

report.append(
    "A robust result here means the core inference survives "
    "the observed structured missingness under both "
    "available-group and fully-balanced-seed restrictions."
)

(
    S2
    /
    "05_S2_HUMAN_READABLE_REPORT.txt"
).write_text(
    "\n".join(
        report
    )
    +
    "\n",
    encoding="utf-8"
)


# ============================================================
# HASH EVERYTHING
# ============================================================

manifest_rows = []

for p in sorted(
    S2.rglob("*"),
    key=lambda x: str(x)
):
    if not p.is_file():
        continue

    if (
        p.name
        ==
        "06_S2_SHA256_MANIFEST.csv"
    ):
        continue

    manifest_rows.append({
        "RELATIVE_PATH":
            str(
                p.relative_to(
                    S2
                )
            ),

        "LENGTH":
            p.stat().st_size,

        "SHA256":
            sha256_file(
                p
            ),
    })

manifest = (
    S2
    /
    "06_S2_SHA256_MANIFEST.csv"
)

pd.DataFrame(
    manifest_rows
).to_csv(
    manifest,
    index=False
)


# ============================================================
# CONSOLE RESULTS
# ============================================================

print()
print(
    "=" * 72
)

print(
    " WF1-A S2 OUTCOME SENSITIVITY COMPLETE"
)

print(
    "=" * 72
)

print()

print(
    f"Original main       : 8131 / 8160"
)

print(
    f"Persistent missing  : 29"
)

print()

print(
    f"3/3-affected seeds  : "
    f"{len(triple_missing_seeds)} "
    f"{sorted(triple_missing_seeds)}"
)

print(
    f"Any-missing seeds   : "
    f"{len(any_missing_seeds)} "
    f"{sorted(any_missing_seeds)}"
)

print()

print(
    comparison_df.to_string(
        index=False
    )
)

print()

print(
    "ROBUSTNESS"
)

print(
    "----------"
)

print(
    f"Same classification     : "
    f"{same_classification}"
)

print(
    f"Strong gate both        : "
    f"{strong_both}"
)

print(
    f"Hard AUC >=.85 both     : "
    f"{auc_gate_both}"
)

print()

print(
    f"S2 folder: {S2}"
)

print(
    f"Manifest : {manifest}"
)

print()

print(
    "NO API CALLS."
)

print(
    "NO IMPUTATION."
)

print(
    "NO R1 DATA MODIFIED."
)

print(
    "=" * 72
)
