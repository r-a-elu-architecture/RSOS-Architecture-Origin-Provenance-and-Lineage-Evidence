from pathlib import Path
import json
import hashlib
import shutil
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    roc_auc_score,
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.ensemble import ExtraTreesClassifier


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(
    r"C:\RSOS\Training data\LMARENA_V12_V18_UPDATED"
)

OUT = Path(
    r"C:\RSOS\Training data\F1_DEV_LMARENA_BLIND_R1"
)

BLIND_PATH = (
    BASE
    / "04_BLIND_STRUCTURAL_MATRIX"
    / "train-00000-of-00007__BLIND_STRUCTURAL_MATRIX.parquet"
)

ARM_PATH = (
    BASE
    / "03_ARM_LEVEL"
    / "train-00000-of-00007__V12_V18_ARMS.parquet"
)

CONDITION_REGISTRY = (
    BASE
    / "08_LOGIC_REGISTRY"
    / "UPDATED_V12_V18_CONDITION_REGISTRY.csv"
)

LOGIC_REGISTRY = (
    BASE
    / "08_LOGIC_REGISTRY"
    / "V12_V18_LOGIC_REGISTRY.csv"
)

DATASET_MANIFEST = (
    BASE
    / "00_MANIFEST"
    / "DATASET_MANIFEST.json"
)

SOURCE_DIR = Path(
    r"C:\RSOS\Training data\training data"
)

MASTER_SEED = 120918

CANDIDATE_K = list(
    range(2, 11)
)

K_SELECTION_SEEDS = [
    MASTER_SEED,
    MASTER_SEED + 1,
    MASTER_SEED + 2,
]

VERSION = "F1_DEV_LMARENA_BLIND_R1"

COMPLETE_MARKER = (
    OUT
    / "F1_DEV_COMPLETE.marker"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

DIRS = {
    "manifest":
        OUT / "00_MANIFEST",

    "calibration":
        OUT / "01_CALIBRATION",

    "blind":
        OUT / "02_BLIND_DISCOVERY",

    "validation":
        OUT / "03_VALIDATION",

    "holdout":
        OUT / "04_DEVELOPMENT_HOLDOUT",

    "unblind":
        OUT / "05_POSTFREEZE_UNBLIND_AUDIT",

    "status":
        OUT / "06_STATUS_LEDGER",
}


if COMPLETE_MARKER.exists():
    raise SystemExit(
        "\nA completed F1-DEV R1 already exists.\n"
        "It will NOT be overwritten.\n"
        f"{OUT}\n"
    )


for p in DIRS.values():
    p.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# HELPERS
# ============================================================

def utcnow():
    return datetime.now(
        timezone.utc
    ).isoformat()


def sha256_file(path):
    h = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        for chunk in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b""
        ):
            h.update(
                chunk
            )

    return h.hexdigest()


def json_sha256(obj):
    b = json.dumps(
        obj,
        sort_keys=True,
        separators=(
            ",",
            ":"
        )
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        b
    ).hexdigest()


def safe_div(a, b):
    a = pd.to_numeric(
        a,
        errors="coerce"
    )

    b = pd.to_numeric(
        b,
        errors="coerce"
    )

    return (
        a
        /
        b.replace(
            0,
            np.nan
        )
    )


def make_shuffled(
    X,
    seed
):
    rng = np.random.default_rng(
        seed
    )

    Y = np.array(
        X,
        copy=True
    )

    for j in range(
        Y.shape[1]
    ):
        Y[:, j] = Y[
            rng.permutation(
                len(Y)
            ),
            j
        ]

    return Y


def calibration_dataset(
    X,
    seed
):
    negative = make_shuffled(
        X,
        seed
    )

    XX = np.vstack(
        [
            X,
            negative
        ]
    )

    yy = np.concatenate(
        [
            np.ones(
                len(X),
                dtype=int
            ),
            np.zeros(
                len(X),
                dtype=int
            )
        ]
    )

    return XX, yy


def cluster_metrics(
    Z,
    labels,
    seed
):
    labels = np.asarray(
        labels
    )

    unique = np.unique(
        labels
    )

    if len(unique) < 2:
        return {
            "SILHOUETTE":
                np.nan,

            "DAVIES_BOULDIN":
                np.nan,

            "CALINSKI_HARABASZ":
                np.nan,

            "MIN_CLUSTER_FRAC":
                1.0
        }

    counts = np.bincount(
        labels
    )

    min_frac = (
        counts[
            counts > 0
        ].min()
        /
        len(labels)
    )

    sample_size = min(
        2500,
        len(Z)
    )

    sil = silhouette_score(
        Z,
        labels,
        sample_size=sample_size,
        random_state=seed
    )

    # DB / CH can also be expensive with very large N.
    # Use deterministic sample for diagnostics.
    if len(Z) > 5000:

        rng = np.random.default_rng(
            seed
        )

        idx = rng.choice(
            len(Z),
            size=5000,
            replace=False
        )

        ZZ = Z[
            idx
        ]

        LL = labels[
            idx
        ]

    else:
        ZZ = Z
        LL = labels

    db = davies_bouldin_score(
        ZZ,
        LL
    )

    ch = calinski_harabasz_score(
        ZZ,
        LL
    )

    return {
        "SILHOUETTE":
            float(
                sil
            ),

        "DAVIES_BOULDIN":
            float(
                db
            ),

        "CALINSKI_HARABASZ":
            float(
                ch
            ),

        "MIN_CLUSTER_FRAC":
            float(
                min_frac
            )
    }


# ============================================================
# PROVENANCE
# ============================================================

print()
print("=" * 68)
print("F1-DEV — BLIND STRUCTURAL REDISCOVERY")
print("=" * 68)
print()

print("Hashing input files...")


input_files = [
    BLIND_PATH,
    ARM_PATH,
    CONDITION_REGISTRY,
    LOGIC_REGISTRY,
    DATASET_MANIFEST,
]


input_hash_rows = []

for p in input_files:

    input_hash_rows.append(
        {
            "FILE":
                p.name,

            "PATH":
                str(p),

            "BYTES":
                p.stat().st_size,

            "SHA256":
                sha256_file(
                    p
                )
        }
    )


pd.DataFrame(
    input_hash_rows
).to_csv(
    DIRS["manifest"]
    / "INPUT_SHA256_MANIFEST.csv",
    index=False
)


reserved = sorted(
    SOURCE_DIR.glob(
        "train-*-of-00086.parquet"
    )
)


(
    DIRS["manifest"]
    / "RESERVED_EXTERNAL_CONTROLS_NOT_READ.txt"
).write_text(
    "\n".join(
        str(x)
        for x
        in reserved
    ),
    encoding="utf-8"
)


# ============================================================
# LOAD BLIND DATA ONLY
# ============================================================

print()
print("BLIND PHASE ACTIVE")
print("Model/provider/winner/raw text will NOT be read yet.")
print()


blind = pd.read_parquet(
    BLIND_PATH
)


required_blind = {
    "BLIND_ID",
    "DEVELOPMENT_SPLIT",
    "ERA_2025_06_08",
    "LANGUAGE",
    "IS_CODE",
}


missing = required_blind - set(
    blind.columns
)


if missing:
    raise RuntimeError(
        f"Blind matrix missing columns: {sorted(missing)}"
    )


if blind[
    "BLIND_ID"
].duplicated().any():

    raise RuntimeError(
        "Duplicate BLIND_ID values detected."
    )


# ============================================================
# FORBIDDEN-COLUMN AUDIT
# ============================================================

forbidden_tokens = [
    "MODEL",
    "PROVIDER",
    "WINNER",
    "CONVERSATION",
    "RAW_TEXT",
    "RSOS",
    "RSSO",
    "RSIA",
    "RSX",
]


forbidden_found = []

for col in blind.columns:

    uc = col.upper()

    for token in forbidden_tokens:

        if token in uc:
            forbidden_found.append(
                col
            )


forbidden_found = sorted(
    set(
        forbidden_found
    )
)


if forbidden_found:

    raise RuntimeError(
        "Blind matrix contains forbidden columns: "
        + ", ".join(
            forbidden_found
        )
    )


# ============================================================
# SPLIT AUDIT
# ============================================================

expected_splits = {
    "DEVELOPMENT_TRAIN",
    "DEVELOPMENT_VALIDATION",
    "DEVELOPMENT_HOLDOUT",
}


actual_splits = set(
    blind[
        "DEVELOPMENT_SPLIT"
    ].dropna().unique()
)


if actual_splits != expected_splits:

    raise RuntimeError(
        f"Unexpected split labels: {actual_splits}"
    )


split_counts = (
    blind[
        "DEVELOPMENT_SPLIT"
    ]
    .value_counts()
    .rename_axis(
        "SPLIT"
    )
    .reset_index(
        name="ROWS"
    )
)


split_counts.to_csv(
    DIRS["manifest"]
    / "BLIND_SPLIT_COUNTS.csv",
    index=False
)


# ============================================================
# DERIVE NORMALIZED STRUCTURAL FEATURES
#
# Key improvement:
# avoid allowing absolute length alone to dominate blind
# discovery.
# ============================================================

F = pd.DataFrame(
    {
        "BLIND_ID":
            blind[
                "BLIND_ID"
            ],

        "DEVELOPMENT_SPLIT":
            blind[
                "DEVELOPMENT_SPLIT"
            ],

        "TYPE_TOKEN_RATIO":
            blind[
                "TYPE_TOKEN_RATIO"
            ],

        "ROLE_ALTERNATION_RATIO":
            blind[
                "ROLE_ALTERNATION_RATIO"
            ],

        "DUPLICATE_TURN_RATIO":
            blind[
                "DUPLICATE_TURN_RATIO"
            ],

        "UPPERCASE_RATIO":
            blind[
                "UPPERCASE_RATIO"
            ],

        "DIGIT_RATIO":
            blind[
                "DIGIT_RATIO"
            ],

        "COMPRESSION_RATIO":
            blind[
                "COMPRESSION_RATIO"
            ],

        "USER_CHAR_SHARE":
            safe_div(
                blind[
                    "USER_CHARS"
                ],
                blind[
                    "TOTAL_CHARS"
                ]
            ),

        "USER_TURN_SHARE":
            safe_div(
                blind[
                    "N_USER_TURNS"
                ],
                blind[
                    "N_TURNS"
                ]
            ),

        "MEAN_MAX_TURN_RATIO":
            safe_div(
                blind[
                    "MEAN_TURN_CHARS"
                ],
                blind[
                    "MAX_TURN_CHARS"
                ]
            ),

        "WORDS_PER_1K_CHAR":
            1000
            *
            safe_div(
                blind[
                    "WORD_COUNT"
                ],
                blind[
                    "TOTAL_CHARS"
                ]
            ),

        "TURNS_PER_1K_CHAR":
            1000
            *
            safe_div(
                blind[
                    "N_TURNS"
                ],
                blind[
                    "TOTAL_CHARS"
                ]
            ),

        "PARAGRAPHS_PER_TURN":
            safe_div(
                blind[
                    "PARAGRAPH_COUNT"
                ],
                blind[
                    "N_TURNS"
                ]
            ),

        "CODE_FENCES_PER_1K_CHAR":
            1000
            *
            safe_div(
                blind[
                    "CODE_FENCE_COUNT"
                ],
                blind[
                    "TOTAL_CHARS"
                ]
            ),

        "HEADERS_PER_1K_CHAR":
            1000
            *
            safe_div(
                blind[
                    "MARKDOWN_HEADER_COUNT"
                ],
                blind[
                    "TOTAL_CHARS"
                ]
            ),

        "LIST_ITEMS_PER_1K_CHAR":
            1000
            *
            safe_div(
                blind[
                    "LIST_ITEM_COUNT"
                ],
                blind[
                    "TOTAL_CHARS"
                ]
            ),

        "COLONS_PER_1K_CHAR":
            1000
            *
            safe_div(
                blind[
                    "COLON_COUNT"
                ],
                blind[
                    "TOTAL_CHARS"
                ]
            ),

        "BRACKETS_PER_1K_CHAR":
            1000
            *
            safe_div(
                blind[
                    "BRACKET_COUNT"
                ],
                blind[
                    "TOTAL_CHARS"
                ]
            ),
    }
)


feature_names = [
    x
    for x
    in F.columns
    if x
    not in {
        "BLIND_ID",
        "DEVELOPMENT_SPLIT"
    }
]


F.to_parquet(
    DIRS["blind"]
    / "F1_PRIMARY_DERIVED_BLIND_FEATURE_MATRIX.parquet",
    index=False,
    compression="zstd"
)


# ============================================================
# TRAIN / VALIDATION ONLY
#
# HOLDOUT IS NOT USED UNTIL THE SPEC IS FROZEN BELOW.
# ============================================================

train_mask = (
    F[
        "DEVELOPMENT_SPLIT"
    ]
    ==
    "DEVELOPMENT_TRAIN"
)


val_mask = (
    F[
        "DEVELOPMENT_SPLIT"
    ]
    ==
    "DEVELOPMENT_VALIDATION"
)


train = F.loc[
    train_mask,
    feature_names
].copy()


val = F.loc[
    val_mask,
    feature_names
].copy()


# ============================================================
# TRAIN-DERIVED IMPUTATION + CLIPPING
# ============================================================

medians = train.median(
    numeric_only=True
)


train = train.fillna(
    medians
)


val = val.fillna(
    medians
)


lower = train.quantile(
    0.005
)


upper = train.quantile(
    0.995
)


train = train.clip(
    lower=lower,
    upper=upper,
    axis=1
)


val = val.clip(
    lower=lower,
    upper=upper,
    axis=1
)


# Remove train-constant features only.

variances = train.var()


kept_features = [
    c
    for c
    in feature_names
    if (
        np.isfinite(
            variances[
                c
            ]
        )
        and
        variances[
            c
        ]
        >
        1e-12
    )
]


dropped_features = sorted(
    set(
        feature_names
    )
    -
    set(
        kept_features
    )
)


train = train[
    kept_features
]


val = val[
    kept_features
]


scaler = StandardScaler()


X_train = scaler.fit_transform(
    train
)


X_val = scaler.transform(
    val
)


# ============================================================
# PCA — TRAIN FIT ONLY
# ============================================================

pca = PCA(
    n_components=0.90,
    svd_solver="full"
)


Z_train = pca.fit_transform(
    X_train
)


Z_val = pca.transform(
    X_val
)


# ============================================================
# F1 CALIBRATION
#
# Positive = empirical joint structural relation.
# Negative = exactly same univariate marginals but every
# feature independently permuted.
#
# Therefore classifier cannot win from single-feature
# distributions alone. It must exploit inter-feature
# relationships.
# ============================================================

print(
    "Running F1 relational-dependence calibration..."
)


cal_X_train, cal_y_train = (
    calibration_dataset(
        X_train,
        MASTER_SEED + 100
    )
)


cal_X_val, cal_y_val = (
    calibration_dataset(
        X_val,
        MASTER_SEED + 101
    )
)


calibrator = ExtraTreesClassifier(
    n_estimators=300,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    random_state=MASTER_SEED,
    n_jobs=-1
)


calibrator.fit(
    cal_X_train,
    cal_y_train
)


val_prob = calibrator.predict_proba(
    cal_X_val
)[:, 1]


calibration_validation_auc = (
    roc_auc_score(
        cal_y_val,
        val_prob
    )
)


pd.DataFrame(
    {
        "FEATURE":
            kept_features,

        "IMPORTANCE":
            calibrator.feature_importances_
    }
).sort_values(
    "IMPORTANCE",
    ascending=False
).to_csv(
    DIRS["calibration"]
    / "F1_CALIBRATION_FEATURE_IMPORTANCE.csv",
    index=False
)


# ============================================================
# BLIND FEATURE-RELATION GRAPH
#
# Top train correlations selected BEFORE HOLDOUT.
# ============================================================

train_corr = pd.DataFrame(
    X_train,
    columns=kept_features
).corr(
    method="spearman"
)


val_corr = pd.DataFrame(
    X_val,
    columns=kept_features
).corr(
    method="spearman"
)


edge_candidates = []


for i in range(
    len(
        kept_features
    )
):

    for j in range(
        i + 1,
        len(
            kept_features
        )
    ):

        f1 = kept_features[
            i
        ]

        f2 = kept_features[
            j
        ]

        rho = float(
            train_corr.loc[
                f1,
                f2
            ]
        )

        if np.isfinite(
            rho
        ):

            edge_candidates.append(
                (
                    f1,
                    f2,
                    rho,
                    abs(
                        rho
                    )
                )
            )


edge_candidates.sort(
    key=lambda x:
        x[3],
    reverse=True
)


top_edges = edge_candidates[
    :
    min(
        40,
        len(
            edge_candidates
        )
    )
]


graph_rows = []


for rank, (
    f1,
    f2,
    train_rho,
    _
) in enumerate(
    top_edges,
    start=1
):

    val_rho = float(
        val_corr.loc[
            f1,
            f2
        ]
    )

    graph_rows.append(
        {
            "EDGE_RANK":
                rank,

            "FEATURE_1":
                f1,

            "FEATURE_2":
                f2,

            "TRAIN_SPEARMAN":
                train_rho,

            "VALIDATION_SPEARMAN":
                val_rho,

            "VALIDATION_SIGN_MATCH":
                bool(
                    np.sign(
                        train_rho
                    )
                    ==
                    np.sign(
                        val_rho
                    )
                )
        }
    )


graph_df = pd.DataFrame(
    graph_rows
)


graph_df.to_csv(
    DIRS["blind"]
    / "F1_BLIND_FEATURE_RELATION_GRAPH_PREHOLDOUT.csv",
    index=False
)


# ============================================================
# BLIND CLUSTER MODEL SELECTION
#
# No model/provider/winner/raw text available.
# Selection uses TRAIN + VALIDATION only.
# ============================================================

print(
    "Selecting blind structural K using validation only..."
)


selection_rows = []


for k in CANDIDATE_K:

    predictions = []
    silhouettes = []
    min_cluster_fracs = []


    for seed in K_SELECTION_SEEDS:

        km = KMeans(
            n_clusters=k,
            random_state=seed,
            n_init=8,
            max_iter=400
        )


        km.fit(
            Z_train
        )


        pred = km.predict(
            Z_val
        )


        predictions.append(
            pred
        )


        counts = np.bincount(
            pred,
            minlength=k
        )


        nonzero = counts[
            counts > 0
        ]


        min_frac = (
            nonzero.min()
            /
            len(
                pred
            )
        )


        min_cluster_fracs.append(
            min_frac
        )


        sil = silhouette_score(
            Z_val,
            pred,
            sample_size=min(
                1500,
                len(
                    Z_val
                )
            ),
            random_state=seed
        )


        silhouettes.append(
            sil
        )


    aris = []


    for a, b in combinations(
        predictions,
        2
    ):

        aris.append(
            adjusted_rand_score(
                a,
                b
            )
        )


    mean_sil = float(
        np.mean(
            silhouettes
        )
    )


    mean_ari = float(
        np.mean(
            aris
        )
    )


    min_frac = float(
        np.min(
            min_cluster_fracs
        )
    )


    small_cluster_penalty = max(
        0.0,
        0.03
        -
        min_frac
    )


    composite = (
        mean_sil
        +
        0.25
        *
        mean_ari
        -
        small_cluster_penalty
    )


    selection_rows.append(
        {
            "K":
                k,

            "MEAN_VALIDATION_SILHOUETTE":
                mean_sil,

            "SEED_STABILITY_ARI":
                mean_ari,

            "MIN_CLUSTER_FRAC":
                min_frac,

            "SMALL_CLUSTER_PENALTY":
                small_cluster_penalty,

            "COMPOSITE_SELECTION_SCORE":
                composite
        }
    )


selection_df = pd.DataFrame(
    selection_rows
)


selection_df.to_csv(
    DIRS["blind"]
    / "F1_CLUSTER_MODEL_SELECTION.csv",
    index=False
)


selected_row = selection_df.sort_values(
    [
        "COMPOSITE_SELECTION_SCORE",
        "K"
    ],
    ascending=[
        False,
        True
    ]
).iloc[
    0
]


selected_k = int(
    selected_row[
        "K"
    ]
)


print(
    f"Selected K = {selected_k}"
)


# ============================================================
# FINAL BLIND CLUSTER FIT
# ============================================================

final_cluster = KMeans(
    n_clusters=selected_k,
    random_state=MASTER_SEED,
    n_init=30,
    max_iter=600
)


final_cluster.fit(
    Z_train
)


train_labels_raw = final_cluster.predict(
    Z_train
)


val_labels_raw = final_cluster.predict(
    Z_val
)


# Deterministic U-ID ordering using centroid coordinates.

centroids = final_cluster.cluster_centers_


cluster_order = sorted(
    range(
        selected_k
    ),
    key=lambda c:
        tuple(
            centroids[
                c,
                :
                min(
                    3,
                    centroids.shape[
                        1
                    ]
                )
            ]
        )
)


cluster_to_u = {
    cluster_id:
        f"U{rank:03d}"
    for rank, cluster_id
    in enumerate(
        cluster_order,
        start=1
    )
}


# ============================================================
# PRE-HOLDOUT METRICS
# ============================================================

train_metrics = cluster_metrics(
    Z_train,
    train_labels_raw,
    MASTER_SEED
)


validation_metrics = cluster_metrics(
    Z_val,
    val_labels_raw,
    MASTER_SEED + 1
)


# ============================================================
# FREEZE SPECIFICATION BEFORE DEVELOPMENT HOLDOUT
# ============================================================

freeze_spec = {
    "version":
        VERSION,

    "frozen_utc":
        utcnow(),

    "dataset_role":
        "DEVELOPMENT_ONLY",

    "confirmatory_claim_allowed":
        False,

    "master_seed":
        MASTER_SEED,

    "blind_source":
        str(
            BLIND_PATH
        ),

    "forbidden_columns":
        forbidden_tokens,

    "primary_features":
        kept_features,

    "dropped_constant_features":
        dropped_features,

    "clip_quantiles":
        [
            0.005,
            0.995
        ],

    "median_imputation":
        {
            k:
                float(
                    medians[
                        k
                    ]
                )
            for k
            in kept_features
        },

    "clip_lower":
        {
            k:
                float(
                    lower[
                        k
                    ]
                )
            for k
            in kept_features
        },

    "clip_upper":
        {
            k:
                float(
                    upper[
                        k
                    ]
                )
            for k
            in kept_features
        },

    "scaler_mean":
        {
            k:
                float(v)
            for k, v
            in zip(
                kept_features,
                scaler.mean_
            )
        },

    "scaler_scale":
        {
            k:
                float(v)
            for k, v
            in zip(
                kept_features,
                scaler.scale_
            )
        },

    "pca_components":
        int(
            Z_train.shape[
                1
            ]
        ),

    "pca_explained_variance_ratio":
        [
            float(x)
            for x
            in pca.explained_variance_ratio_
        ],

    "candidate_k":
        CANDIDATE_K,

    "selected_k":
        selected_k,

    "k_selection_seeds":
        K_SELECTION_SEEDS,

    "cluster_to_u":
        {
            str(k):
                v
            for k, v
            in cluster_to_u.items()
        },

    "calibration":
        {
            "positive":
                "empirical joint feature structure",

            "negative":
                "independent within-feature permutations preserving univariate marginals",

            "classifier":
                "ExtraTreesClassifier",

            "n_estimators":
                300,

            "min_samples_leaf":
                5
        },

    "holdout_rule":
        (
            "DEVELOPMENT_HOLDOUT is evaluated only after "
            "this specification is written and hashed."
        ),

    "external_control_rule":
        (
            "train-*-of-00086.parquet files remain unread "
            "and reserved for later F2/V16 population rarity."
        ),

    "f3_rule":
        (
            "Historical LM Arena data cannot satisfy F3 prospective confirmation."
        )
}


freeze_path = (
    DIRS["manifest"]
    /
    "F1_FROZEN_SPEC_BEFORE_DEVELOPMENT_HOLDOUT.json"
)


freeze_path.write_text(
    json.dumps(
        freeze_spec,
        indent=2
    ),
    encoding="utf-8"
)


freeze_hash = sha256_file(
    freeze_path
)


(
    DIRS["manifest"]
    /
    "F1_FROZEN_SPEC_SHA256.txt"
).write_text(
    freeze_hash
    +
    "\n",
    encoding="utf-8"
)


print()
print(
    "BLIND SPECIFICATION FROZEN"
)

print(
    "SHA256:",
    freeze_hash
)

print()


# ============================================================
# DEVELOPMENT HOLDOUT BEGINS ONLY NOW
# ============================================================

print(
    "Opening DEVELOPMENT_HOLDOUT after freeze..."
)


hold_mask = (
    F[
        "DEVELOPMENT_SPLIT"
    ]
    ==
    "DEVELOPMENT_HOLDOUT"
)


hold = F.loc[
    hold_mask,
    kept_features
].copy()


hold = hold.fillna(
    medians
)


hold = hold.clip(
    lower=lower[
        kept_features
    ],
    upper=upper[
        kept_features
    ],
    axis=1
)


X_hold = scaler.transform(
    hold
)


Z_hold = pca.transform(
    X_hold
)


hold_labels_raw = final_cluster.predict(
    Z_hold
)


# ============================================================
# CALIBRATION HOLDOUT
# ============================================================

cal_X_hold, cal_y_hold = (
    calibration_dataset(
        X_hold,
        MASTER_SEED + 102
    )
)


hold_prob = calibrator.predict_proba(
    cal_X_hold
)[:, 1]


calibration_holdout_auc = (
    roc_auc_score(
        cal_y_hold,
        hold_prob
    )
)


calibration_results = pd.DataFrame(
    [
        {
            "PARTITION":
                "VALIDATION",

            "ROC_AUC":
                calibration_validation_auc
        },

        {
            "PARTITION":
                "DEVELOPMENT_HOLDOUT",

            "ROC_AUC":
                calibration_holdout_auc
        }
    ]
)


calibration_results[
    "INTERPRETATION"
] = calibration_results[
    "ROC_AUC"
].apply(
    lambda x:
        (
            "STRONG_RELATIONAL_RECOVERABILITY"
            if x >= 0.80
            else
            (
                "MODERATE_RELATIONAL_RECOVERABILITY"
                if x >= 0.65
                else
                "WEAK_RELATIONAL_RECOVERABILITY"
            )
        )
)


calibration_results.to_csv(
    DIRS["calibration"]
    / "F1_RELATIONAL_DEPENDENCE_CALIBRATION.csv",
    index=False
)


# ============================================================
# HOLDOUT FEATURE-GRAPH STABILITY
# ============================================================

hold_corr = pd.DataFrame(
    X_hold,
    columns=kept_features
).corr(
    method="spearman"
)


graph_hold_rows = []


for row in graph_rows:

    f1 = row[
        "FEATURE_1"
    ]

    f2 = row[
        "FEATURE_2"
    ]

    hold_rho = float(
        hold_corr.loc[
            f1,
            f2
        ]
    )

    r = dict(
        row
    )

    r[
        "HOLDOUT_SPEARMAN"
    ] = hold_rho

    r[
        "HOLDOUT_SIGN_MATCH"
    ] = bool(
        np.sign(
            r[
                "TRAIN_SPEARMAN"
            ]
        )
        ==
        np.sign(
            hold_rho
        )
    )

    graph_hold_rows.append(
        r
    )


graph_final = pd.DataFrame(
    graph_hold_rows
)


graph_final.to_csv(
    DIRS["holdout"]
    / "F1_FEATURE_RELATION_GRAPH_HOLDOUT.csv",
    index=False
)


train_edge_weights = graph_final[
    "TRAIN_SPEARMAN"
].to_numpy()


val_edge_weights = graph_final[
    "VALIDATION_SPEARMAN"
].to_numpy()


hold_edge_weights = graph_final[
    "HOLDOUT_SPEARMAN"
].to_numpy()


graph_validation_weight_corr = float(
    np.corrcoef(
        train_edge_weights,
        val_edge_weights
    )[0, 1]
)


graph_holdout_weight_corr = float(
    np.corrcoef(
        train_edge_weights,
        hold_edge_weights
    )[0, 1]
)


graph_validation_sign = float(
    graph_final[
        "VALIDATION_SIGN_MATCH"
    ].mean()
)


graph_holdout_sign = float(
    graph_final[
        "HOLDOUT_SIGN_MATCH"
    ].mean()
)


pd.DataFrame(
    [
        {
            "PARTITION":
                "VALIDATION",

            "EDGE_WEIGHT_CORRELATION_WITH_TRAIN":
                graph_validation_weight_corr,

            "EDGE_SIGN_AGREEMENT":
                graph_validation_sign
        },

        {
            "PARTITION":
                "DEVELOPMENT_HOLDOUT",

            "EDGE_WEIGHT_CORRELATION_WITH_TRAIN":
                graph_holdout_weight_corr,

            "EDGE_SIGN_AGREEMENT":
                graph_holdout_sign
        }
    ]
).to_csv(
    DIRS["holdout"]
    / "F1_BLIND_GRAPH_STABILITY.csv",
    index=False
)


# ============================================================
# CLUSTER HOLDOUT METRICS
# ============================================================

holdout_metrics = cluster_metrics(
    Z_hold,
    hold_labels_raw,
    MASTER_SEED + 2
)


metric_rows = []


for partition, metrics in [
    (
        "TRAIN",
        train_metrics
    ),
    (
        "VALIDATION",
        validation_metrics
    ),
    (
        "DEVELOPMENT_HOLDOUT",
        holdout_metrics
    )
]:

    row = {
        "PARTITION":
            partition,

        "SELECTED_K":
            selected_k
    }

    row.update(
        metrics
    )

    metric_rows.append(
        row
    )


metrics_df = pd.DataFrame(
    metric_rows
)


metrics_df.to_csv(
    DIRS["holdout"]
    / "F1_BLIND_CLUSTER_METRICS.csv",
    index=False
)


# ============================================================
# BLIND U ASSIGNMENTS
# ============================================================

all_features = F[
    kept_features
].copy()


all_features = all_features.fillna(
    medians
)


all_features = all_features.clip(
    lower=lower[
        kept_features
    ],
    upper=upper[
        kept_features
    ],
    axis=1
)


X_all = scaler.transform(
    all_features
)


Z_all = pca.transform(
    X_all
)


all_labels_raw = final_cluster.predict(
    Z_all
)


all_distances = final_cluster.transform(
    Z_all
).min(
    axis=1
)


assignments = pd.DataFrame(
    {
        "BLIND_ID":
            F[
                "BLIND_ID"
            ].values,

        "DEVELOPMENT_SPLIT":
            F[
                "DEVELOPMENT_SPLIT"
            ].values,

        "RAW_CLUSTER_ID":
            all_labels_raw,

        "U_ID":
            [
                cluster_to_u[
                    int(x)
                ]
                for x
                in all_labels_raw
            ],

        "DISTANCE_TO_CENTROID":
            all_distances
    }
)


for i in range(
    min(
        8,
        Z_all.shape[
            1
        ]
    )
):

    assignments[
        f"PC{i+1}"
    ] = Z_all[
        :,
        i
    ]


assignments.to_parquet(
    DIRS["blind"]
    / "F1_BLIND_U_ASSIGNMENTS.parquet",
    index=False,
    compression="zstd"
)


assignments.to_csv(
    DIRS["blind"]
    / "F1_BLIND_U_ASSIGNMENTS.csv",
    index=False
)


# ============================================================
# CENTROIDS
# ============================================================

centroid_rows = []


for raw_id in range(
    selected_k
):

    row = {
        "RAW_CLUSTER_ID":
            raw_id,

        "U_ID":
            cluster_to_u[
                raw_id
            ]
    }

    for i in range(
        centroids.shape[
            1
        ]
    ):

        row[
            f"PC{i+1}_CENTROID"
        ] = float(
            centroids[
                raw_id,
                i
            ]
        )

    centroid_rows.append(
        row
    )


pd.DataFrame(
    centroid_rows
).sort_values(
    "U_ID"
).to_csv(
    DIRS["blind"]
    / "F1_U_CENTROIDS.csv",
    index=False
)


# ============================================================
# ONLY NOW: POST-FREEZE UNBLIND AUDIT
#
# This is deliberately after:
# - blind feature definition
# - calibration definition
# - K selection
# - cluster fit
# - development-holdout analysis
# ============================================================

print()
print("=" * 68)
print("POST-FREEZE UNBLIND AUDIT")
print("=" * 68)
print()


meta_columns = [
    "BLIND_ID",
    "RECORD_ID",
    "PAIR_ID",
    "EVALUATION_SESSION_ID",
    "DEVELOPMENT_SPLIT",
    "ERA_2025_06_08",
    "LANGUAGE",
    "IS_CODE",
    "MODEL",
    "PROVIDER",
    "WINNER_RAW",
    "PERSPECTIVE_LABEL",
]


meta = pd.read_parquet(
    ARM_PATH,
    columns=meta_columns
)


if meta[
    "BLIND_ID"
].duplicated().any():

    raise RuntimeError(
        "ARM metadata has duplicate BLIND_ID."
    )


# ============================================================
# SPLIT LEAKAGE AUDIT BY SESSION
# ============================================================

session_split_counts = (
    meta.groupby(
        "EVALUATION_SESSION_ID"
    )[
        "DEVELOPMENT_SPLIT"
    ]
    .nunique()
)


leaking_sessions = int(
    (
        session_split_counts
        >
        1
    ).sum()
)


if leaking_sessions != 0:

    raise RuntimeError(
        f"Session leakage detected: {leaking_sessions} sessions"
    )


merged = assignments.merge(
    meta,
    on="BLIND_ID",
    how="left",
    validate="one_to_one",
    suffixes=(
        "_BLIND",
        "_META"
    )
)


unmatched = int(
    merged[
        "MODEL"
    ].isna().sum()
)


if unmatched != 0:

    raise RuntimeError(
        f"{unmatched} blind rows failed post-freeze metadata merge."
    )


# ============================================================
# POST-FREEZE CONFOUND AUDIT
# ============================================================

confound_variables = [
    "PROVIDER",
    "MODEL",
    "WINNER_RAW",
    "PERSPECTIVE_LABEL",
    "LANGUAGE",
    "IS_CODE",
    "ERA_2025_06_08",
]


confound_rows = []


for partition in [
    "DEVELOPMENT_VALIDATION",
    "DEVELOPMENT_HOLDOUT"
]:

    part = merged[
        merged[
            "DEVELOPMENT_SPLIT_META"
        ]
        ==
        partition
    ].copy()


    y_cluster = (
        part[
            "U_ID"
        ]
        .fillna(
            "MISSING"
        )
        .astype(
            str
        )
    )


    for variable in confound_variables:

        y_meta = (
            part[
                variable
            ]
            .fillna(
                "MISSING"
            )
            .astype(
                str
            )
        )


        nmi = normalized_mutual_info_score(
            y_cluster,
            y_meta
        )


        ami = adjusted_mutual_info_score(
            y_cluster,
            y_meta
        )


        confound_rows.append(
            {
                "PARTITION":
                    partition,

                "VARIABLE":
                    variable,

                "N_CATEGORIES":
                    int(
                        y_meta.nunique()
                    ),

                "NMI_WITH_U_ID":
                    float(
                        nmi
                    ),

                "AMI_WITH_U_ID":
                    float(
                        ami
                    )
            }
        )


confound_df = pd.DataFrame(
    confound_rows
)


confound_df.to_csv(
    DIRS["unblind"]
    / "F1_POSTFREEZE_CONFOUND_AUDIT.csv",
    index=False
)


# Provider x U table in development holdout

hold_unblind = merged[
    merged[
        "DEVELOPMENT_SPLIT_META"
    ]
    ==
    "DEVELOPMENT_HOLDOUT"
]


provider_table = pd.crosstab(
    hold_unblind[
        "U_ID"
    ],
    hold_unblind[
        "PROVIDER"
    ]
)


provider_table.to_csv(
    DIRS["unblind"]
    / "F1_U_BY_PROVIDER_DEVELOPMENT_HOLDOUT.csv"
)


# ============================================================
# PRESERVE-FIRST RETENTION AUDIT
# ============================================================

feature_missing_rows = []


for feature in kept_features:

    n_missing = int(
        F[
            feature
        ].isna().sum()
    )

    feature_missing_rows.append(
        {
            "FEATURE":
                feature,

            "MISSING_ROWS_BEFORE_IMPUTATION":
                n_missing,

            "MISSING_RATE":
                (
                    n_missing
                    /
                    len(
                        F
                    )
                )
        }
    )


pd.DataFrame(
    feature_missing_rows
).to_csv(
    DIRS["unblind"]
    / "F1_PRESERVE_FIRST_FEATURE_MISSINGNESS.csv",
    index=False
)


retention = {
    "ARM_ROWS":
        int(
            len(
                meta
            )
        ),

    "BLIND_ROWS":
        int(
            len(
                blind
            )
        ),

    "UNIQUE_ARM_BLIND_IDS":
        int(
            meta[
                "BLIND_ID"
            ].nunique()
        ),

    "UNIQUE_BLIND_MATRIX_IDS":
        int(
            blind[
                "BLIND_ID"
            ].nunique()
        ),

    "POSTFREEZE_MATCHED_ROWS":
        int(
            len(
                merged
            )
        ),

    "UNMATCHED_ROWS":
        unmatched,

    "SESSION_SPLIT_LEAKAGE_COUNT":
        leaking_sessions,

    "ROW_RETENTION_RATE":
        (
            len(
                blind
            )
            /
            len(
                meta
            )
        )
        if len(
            meta
        )
        else np.nan,

    "FORBIDDEN_COLUMN_COUNT":
        len(
            forbidden_found
        )
}


pd.DataFrame(
    [
        retention
    ]
).to_csv(
    DIRS["unblind"]
    / "F1_PRESERVE_FIRST_RETENTION.csv",
    index=False
)


# ============================================================
# STATUS LEDGER
# ============================================================

if (
    calibration_validation_auc >= 0.80
    and
    calibration_holdout_auc >= 0.80
):

    calibration_status = (
        "STRONG_RELATIONAL_RECOVERABILITY"
    )

elif (
    calibration_validation_auc >= 0.65
    and
    calibration_holdout_auc >= 0.65
):

    calibration_status = (
        "MODERATE_RELATIONAL_RECOVERABILITY"
    )

else:

    calibration_status = (
        "WEAK_RELATIONAL_RECOVERABILITY"
    )


status_rows = [
    {
        "TEST":
            "BLIND_MATRIX_FORBIDDEN_COLUMN_AUDIT",

        "STATUS":
            "PASS"
            if len(
                forbidden_found
            )
            ==
            0
            else
            "FAIL",

        "VALUE":
            len(
                forbidden_found
            ),

        "CLAIM_SCOPE":
            "DEVELOPMENT"
    },

    {
        "TEST":
            "SESSION_SPLIT_LEAKAGE",

        "STATUS":
            "PASS"
            if leaking_sessions
            ==
            0
            else
            "FAIL",

        "VALUE":
            leaking_sessions,

        "CLAIM_SCOPE":
            "DEVELOPMENT"
    },

    {
        "TEST":
            "PRESERVE_FIRST_ROW_RETENTION",

        "STATUS":
            "PASS"
            if (
                retention[
                    "ROW_RETENTION_RATE"
                ]
                ==
                1.0
            )
            else
            "PARTIAL",

        "VALUE":
            retention[
                "ROW_RETENTION_RATE"
            ],

        "CLAIM_SCOPE":
            "DEVELOPMENT"
    },

    {
        "TEST":
            "RELATIONAL_DEPENDENCE_CALIBRATION",

        "STATUS":
            calibration_status,

        "VALUE":
            calibration_holdout_auc,

        "CLAIM_SCOPE":
            "DEVELOPMENT"
    },

    {
        "TEST":
            "BLIND_STRUCTURAL_DISCOVERY",

        "STATUS":
            "MEASURED",

        "VALUE":
            selected_k,

        "CLAIM_SCOPE":
            "DEVELOPMENT_ONLY"
    },

    {
        "TEST":
            "LLM_GATED_COMPARATOR",

        "STATUS":
            "UNTESTED",

        "VALUE":
            np.nan,

        "CLAIM_SCOPE":
            "NONE"
    },

    {
        "TEST":
            "F2_EXTERNAL_POPULATION_RARITY",

        "STATUS":
            "NOT_RUN_RESERVED",

        "VALUE":
            np.nan,

        "CLAIM_SCOPE":
            "NONE"
    },

    {
        "TEST":
            "F3_PROSPECTIVE_REPLICATION",

        "STATUS":
            "NOT_ELIGIBLE_HISTORICAL_DATA",

        "VALUE":
            np.nan,

        "CLAIM_SCOPE":
            "NONE"
    }
]


status_df = pd.DataFrame(
    status_rows
)


status_df.to_csv(
    DIRS["status"]
    / "F1_DEV_STATUS_LEDGER.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

max_confound_row = (
    confound_df[
        confound_df[
            "PARTITION"
        ]
        ==
        "DEVELOPMENT_HOLDOUT"
    ]
    .sort_values(
        "AMI_WITH_U_ID",
        ascending=False
    )
    .iloc[
        0
    ]
)


summary = {
    "version":
        VERSION,

    "completed_utc":
        utcnow(),

    "dataset_role":
        "DEVELOPMENT_ONLY",

    "rows":
        int(
            len(
                blind
            )
        ),

    "train_rows":
        int(
            train_mask.sum()
        ),

    "validation_rows":
        int(
            val_mask.sum()
        ),

    "development_holdout_rows":
        int(
            hold_mask.sum()
        ),

    "primary_feature_count":
        int(
            len(
                kept_features
            )
        ),

    "pca_components":
        int(
            Z_train.shape[
                1
            ]
        ),

    "selected_k":
        selected_k,

    "calibration_validation_auc":
        float(
            calibration_validation_auc
        ),

    "calibration_holdout_auc":
        float(
            calibration_holdout_auc
        ),

    "calibration_status":
        calibration_status,

    "validation_silhouette":
        float(
            validation_metrics[
                "SILHOUETTE"
            ]
        ),

    "holdout_silhouette":
        float(
            holdout_metrics[
                "SILHOUETTE"
            ]
        ),

    "graph_validation_weight_correlation":
        graph_validation_weight_corr,

    "graph_holdout_weight_correlation":
        graph_holdout_weight_corr,

    "graph_validation_sign_agreement":
        graph_validation_sign,

    "graph_holdout_sign_agreement":
        graph_holdout_sign,

    "largest_postfreeze_holdout_confound":
        str(
            max_confound_row[
                "VARIABLE"
            ]
        ),

    "largest_postfreeze_holdout_confound_AMI":
        float(
            max_confound_row[
                "AMI_WITH_U_ID"
            ]
        ),

    "row_retention_rate":
        float(
            retention[
                "ROW_RETENTION_RATE"
            ]
        ),

    "session_split_leakage_count":
        leaking_sessions,

    "frozen_spec_sha256":
        freeze_hash,

    "reserved_external_controls_read":
        False,

    "confirmatory_claim_allowed":
        False
}


summary_path = (
    DIRS["manifest"]
    / "F1_DEV_SUMMARY.json"
)


summary_path.write_text(
    json.dumps(
        summary,
        indent=2
    ),
    encoding="utf-8"
)


summary_txt = f"""
F1-DEV LM ARENA BLIND STRUCTURAL REDISCOVERY R1

ROLE
DEVELOPMENT ONLY

ROWS
Total: {len(blind):,}
Train: {int(train_mask.sum()):,}
Validation: {int(val_mask.sum()):,}
Development holdout: {int(hold_mask.sum()):,}

PRIMARY STRUCTURAL FEATURES
{len(kept_features)}

PCA COMPONENTS
{Z_train.shape[1]}

SELECTED BLIND U-CLUSTERS
{selected_k}

RELATIONAL DEPENDENCE CALIBRATION
Validation AUC: {calibration_validation_auc:.6f}
Development holdout AUC: {calibration_holdout_auc:.6f}
Status: {calibration_status}

BLIND CLUSTER QUALITY
Validation silhouette: {validation_metrics["SILHOUETTE"]:.6f}
Development holdout silhouette: {holdout_metrics["SILHOUETTE"]:.6f}

FEATURE-RELATION GRAPH
Validation edge-weight correlation: {graph_validation_weight_corr:.6f}
Holdout edge-weight correlation: {graph_holdout_weight_corr:.6f}
Validation sign agreement: {graph_validation_sign:.6f}
Holdout sign agreement: {graph_holdout_sign:.6f}

POST-FREEZE CONFOUND AUDIT
Largest holdout AMI variable: {max_confound_row["VARIABLE"]}
Largest holdout AMI: {float(max_confound_row["AMI_WITH_U_ID"]):.6f}

PRESERVE-FIRST RETENTION
Row retention: {retention["ROW_RETENTION_RATE"]:.6f}
Session leakage: {leaking_sessions}

FROZEN SPEC SHA256
{freeze_hash}

RESERVED train-*-of-00086.parquet CONTROLS READ?
NO

F3 PROSPECTIVE CLAIM?
NOT ELIGIBLE — historical development dataset.

THIS RUN DOES NOT ESTABLISH AN RSOS CLAIM.
It calibrates and audits the blind structural measurement system.
"""


(
    DIRS["manifest"]
    / "F1_DEV_SUMMARY.txt"
).write_text(
    summary_txt.strip()
    +
    "\n",
    encoding="utf-8"
)


# ============================================================
# FINAL OUTPUT INDEX
# ============================================================

output_rows = []


for p in sorted(
    x
    for x
    in OUT.rglob(
        "*"
    )
    if x.is_file()
):

    output_rows.append(
        {
            "FILE":
                p.name,

            "RELATIVE_PATH":
                str(
                    p.relative_to(
                        OUT
                    )
                ),

            "BYTES":
                p.stat().st_size
        }
    )


pd.DataFrame(
    output_rows
).to_csv(
    DIRS["manifest"]
    / "OUTPUT_FILE_INDEX.csv",
    index=False
)


COMPLETE_MARKER.write_text(
    (
        "F1_DEV_COMPLETE\n"
        +
        utcnow()
        +
        "\n"
        +
        freeze_hash
        +
        "\n"
    ),
    encoding="utf-8"
)


print()
print("=" * 68)
print("F1-DEV R1 COMPLETE")
print("=" * 68)
print()

print(
    f"Rows: {len(blind):,}"
)

print(
    f"Primary features: {len(kept_features)}"
)

print(
    f"PCA components: {Z_train.shape[1]}"
)

print(
    f"Selected K: {selected_k}"
)

print()

print(
    "Relational calibration:"
)

print(
    f"  validation AUC = "
    f"{calibration_validation_auc:.6f}"
)

print(
    f"  holdout AUC    = "
    f"{calibration_holdout_auc:.6f}"
)

print(
    f"  status         = "
    f"{calibration_status}"
)

print()

print(
    "Blind clustering:"
)

print(
    f"  validation silhouette = "
    f"{validation_metrics['SILHOUETTE']:.6f}"
)

print(
    f"  holdout silhouette    = "
    f"{holdout_metrics['SILHOUETTE']:.6f}"
)

print()

print(
    "Relational graph stability:"
)

print(
    f"  validation weight r = "
    f"{graph_validation_weight_corr:.6f}"
)

print(
    f"  holdout weight r    = "
    f"{graph_holdout_weight_corr:.6f}"
)

print(
    f"  validation sign     = "
    f"{graph_validation_sign:.6f}"
)

print(
    f"  holdout sign        = "
    f"{graph_holdout_sign:.6f}"
)

print()

print(
    "Largest post-freeze holdout confound:"
)

print(
    f"  {max_confound_row['VARIABLE']} "
    f"AMI="
    f"{float(max_confound_row['AMI_WITH_U_ID']):.6f}"
)

print()

print(
    f"Frozen spec SHA256:"
)

print(
    freeze_hash
)

print()

print(
    "Reserved 00086 controls read: NO"
)

print(
    "Confirmatory claim allowed: NO"
)

print()

print(
    f"Results: {OUT}"
)