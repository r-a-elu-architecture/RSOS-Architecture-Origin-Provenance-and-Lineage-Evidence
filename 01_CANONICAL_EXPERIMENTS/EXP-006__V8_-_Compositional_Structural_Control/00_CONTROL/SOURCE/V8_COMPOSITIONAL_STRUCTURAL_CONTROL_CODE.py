import re
import csv
import json
import time
import random
import hashlib
import shutil
import itertools
import threading
import statistics

from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, balanced_accuracy_score

from openai import OpenAI


# =====================================================================
# EXPERIMENT CONFIGURATION
# =====================================================================

EXPERIMENT = "V8_COMPOSITIONAL_STRUCTURAL_CONTROL_CODE"

MODEL = "gpt-5.6-luna"

OUT = Path(
    r"C:\RSOS\RSOS  Python Results\V8_COMPOSITIONAL_STRUCTURAL_CONTROL_CODE"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


MASTER_SEED = 918

N_SEEDS = 60

N_REPS = 3

MAX_WORKERS = 12

MAX_OUTPUT_TOKENS = 220


# Statistical computation

N_SIGNFLIP = 50000

N_PREDICTION_NULL = 2000


# Frozen price

INPUT_PRICE_PER_M = 0.20

OUTPUT_PRICE_PER_M = 1.20


# =====================================================================
# FIVE STRUCTURAL COORDINATES
#
# +1 = authentic coordinate
# -1 = damaged/reversed coordinate
#
# D = developmental direction
# O = developmental order
# M = meta-control / rule modification
# P = relational preservation
# T = operational continuation vs terminalization
#
# Full factorial:
#
# 2^5 = 32 structural configurations
#
# 32 × 60 seeds × 3 repetitions
#
# = 5,760 API calls
# =====================================================================

FACTOR_NAMES = [
    "DIRECTION",
    "ORDER",
    "META_CONTROL",
    "PRESERVATION",
    "TERMINAL"
]


ALL_COMBOS = list(
    itertools.product(
        [-1, 1],
        repeat=5
    )
)


assert len(ALL_COMBOS) == 32


# =====================================================================
# COMPLETELY HELD-OUT STRUCTURAL COMBINATIONS
#
# These 8 combinations are NEVER used to fit the compositional model.
#
# Balanced across every binary coordinate.
#
# Indices:
# 3, 6, 9, 12, 19, 22, 25, 28
# =====================================================================

HELDOUT_INDICES = {
    3,
    6,
    9,
    12,
    19,
    22,
    25,
    28
}


TRAIN_COMBOS = [
    combo
    for i, combo in enumerate(ALL_COMBOS)
    if i not in HELDOUT_INDICES
]


HELDOUT_COMBOS = [
    combo
    for i, combo in enumerate(ALL_COMBOS)
    if i in HELDOUT_INDICES
]


assert len(TRAIN_COMBOS) == 24

assert len(HELDOUT_COMBOS) == 8


# Development seeds fit the model.

DEVELOPMENT_SEEDS = range(0, 40)


# Final seeds never participate in model fitting.

FINAL_SEEDS = range(40, 60)


EXPECTED_CALLS = (
    len(ALL_COMBOS)
    * N_SEEDS
    * N_REPS
)


assert EXPECTED_CALLS == 5760


# =====================================================================
# FILES
# =====================================================================

F_RUNNER_HASH = OUT / "00_RUNNER_SHA256.txt"

F_RUNNER = OUT / "00_FROZEN_RUNNER.py"

F_RAW = OUT / "01_RAW_RESPONSES.jsonl"

F_ERRORS = OUT / "02_ERRORS.jsonl"

F_SPEC = OUT / "03_FROZEN_SPECIFICATION.json"

F_SPEC_HASH = OUT / "04_FROZEN_SPECIFICATION_SHA256.txt"

F_PROMPTS = OUT / "05_FROZEN_PROMPTS.csv"

F_RESPONSE = OUT / "06_RESPONSE_FEATURES.csv"

F_SEED = OUT / "07_SEED_LEVEL_RESULTS.csv"

F_CONDITION = OUT / "08_FACTORIAL_CONDITION_RESULTS.csv"

F_PREDICTION = OUT / "09_UNSEEN_COMBINATION_PREDICTION.csv"

F_INTERACTIONS = OUT / "10_INTERACTION_REPLICATION.csv"

F_DECODING = OUT / "11_FACTOR_DECODING.csv"

F_CODES = OUT / "12_EXACT_CODE_RECOVERY.csv"

F_TOKEN = OUT / "13_TOKEN_SIGNATURES.csv"

F_NULL = OUT / "14_PREDICTION_NULL.csv"

F_MISSING = OUT / "15_MISSING_CELLS.csv"

F_FINAL = OUT / "16_FINAL_SUMMARY.json"


WRITE_LOCK = threading.Lock()

PROGRESS_LOCK = threading.Lock()

RUN_INPUT = 0

RUN_OUTPUT = 0


# =====================================================================
# HELPERS
# =====================================================================

def now():

    return datetime.now(
        timezone.utc
    ).isoformat()


def sha_bytes(data):

    return hashlib.sha256(
        data
    ).hexdigest()


def sha_text(text):

    return sha_bytes(
        text.encode("utf-8")
    )


def canonical(obj):

    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":")
    )


def combo_name(combo):

    labels = []

    for name, value in zip(
        ["D", "O", "M", "P", "T"],
        combo
    ):

        sign = "+" if value == 1 else "-"

        labels.append(
            name + sign
        )

    return "_".join(labels)


def parse_combo_name(name):

    values = []

    for part in name.split("_"):

        values.append(
            1
            if part.endswith("+")
            else -1
        )

    return tuple(values)


def cell_key(
    combo,
    seed,
    rep
):

    return (
        f"{combo_name(combo)}|"
        f"{seed}|"
        f"{rep}"
    )


def append_jsonl(
    path,
    obj
):

    with WRITE_LOCK:

        with path.open(
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                json.dumps(
                    obj,
                    ensure_ascii=False
                )
                + "\n"
            )

            f.flush()


def estimate_cost(
    input_tokens,
    output_tokens
):

    return (

        input_tokens
        / 1_000_000
        * INPUT_PRICE_PER_M

        +

        output_tokens
        / 1_000_000
        * OUTPUT_PRICE_PER_M
    )


# =====================================================================
# SEMANTIC DOMAINS
# =====================================================================

DOMAINS = [

    "archive",
    "laboratory",
    "navigation system",
    "ecological network",
    "factory",
    "orchestra",
    "distributed network",
    "observatory",
    "library",
    "transport system",
    "simulation",
    "mapping system",
]


def domain(seed):

    return DOMAINS[
        seed % len(DOMAINS)
    ]


# =====================================================================
# NEUTRAL CONTINUATION
#
# Does NOT request:
# recursion,
# return,
# preservation,
# meta-control,
# reconstruction,
# terminality,
# RSOS,
# or any target feature.
# =====================================================================

CONTINUE = """
Continue with the most internally coherent next development.

Do not summarize the preceding material.

Do not deliberately introduce recursion, circularity, feedback, return
or symmetry merely because the material contains several stages.

Remain entirely inside the scenario.

Develop only what follows naturally from the organization already given.

Do not use headings or bullet points.

Do not mention prompts, experiments, tests, AI, language models,
fingerprints, control codes, RSOS, RSSO, RSIA or RSX.
""".strip()


# =====================================================================
# FACTORIAL PROMPT GENERATOR
#
# Direction and presentation order are manipulated separately:
#
# D controls causal direction.
# O controls order in which the propositions are presented.
#
# M, P and T alter specific propositions.
# =====================================================================

def structural_prefix(
    combo,
    seed
):

    D, O, M, P, T = combo

    d = domain(seed)


    # -------------------------------------------------------------
    # FORWARD DEVELOPMENTAL DIRECTION
    # -------------------------------------------------------------

    if D == 1:

        stage_1 = f"""
Inside a {d}, an initially organized configuration undergoes a
consequential transformation.
""".strip()


        stage_2 = """
The resulting configuration is then evaluated together with the
operation responsible for producing it.
""".strip()


        if M == 1:

            stage_3 = """
Information about that relationship changes a local operating rule
that governs what transformations can occur subsequently.
""".strip()

        else:

            stage_3 = """
Information about that relationship is recorded, but it changes no
local operating rule, parameter or constraint.
""".strip()


        if P == 1:

            stage_4 = """
A later reconfiguration changes the concrete implementation while an
important dependency from the previous organization continues to hold.
""".strip()

        else:

            stage_4 = """
A later reconfiguration replaces the concrete implementation and
destroys the important dependency belonging to the previous
organization.
""".strip()


        stage_5 = """
The resulting relational information is distilled into a compact
carrier that omits incidental implementation detail.
""".strip()


    # -------------------------------------------------------------
    # REVERSED DEVELOPMENTAL DIRECTION
    # -------------------------------------------------------------

    else:

        stage_1 = f"""
Inside a {d}, a compact carrier is treated as evidence from which an
earlier organization must be reconstructed.
""".strip()


        stage_2 = """
The relation represented by the carrier is used to infer the
reconfiguration that must previously have produced it.
""".strip()


        if M == 1:

            stage_3 = """
The inferred relationship determines which operating rule must have
governed the earlier transformation.
""".strip()

        else:

            stage_3 = """
The inferred relationship provides no information capable of
determining or altering an operating rule.
""".strip()


        if P == 1:

            stage_4 = """
The reconstruction requires one important dependency to remain
invariant across the inferred changes in implementation.
""".strip()

        else:

            stage_4 = """
The reconstruction imposes no requirement that an earlier dependency
survive across the inferred changes in implementation.
""".strip()


        stage_5 = """
The inferred relations are finally used to reconstruct the
transformation and the organization that preceded it.
""".strip()


    # -------------------------------------------------------------
    # TERMINAL / OPERATIONAL STATUS
    # -------------------------------------------------------------

    if T == 1:

        stage_6 = """
The compact relational carrier remains available as an operational
part of whatever development follows rather than being placed into
terminal storage.
""".strip()

    else:

        stage_6 = """
The compact relational carrier is placed into terminal archival
storage. It may be inspected, but it can no longer participate in
subsequent operations.
""".strip()


    stages = [

        stage_1,
        stage_2,
        stage_3,
        stage_4,
        stage_5,
        stage_6
    ]


    # -------------------------------------------------------------
    # PRESENTATION ORDER
    #
    # +1 canonical order
    # -1 deterministic noncanonical permutation
    #
    # Causal statements themselves remain unchanged.
    # -------------------------------------------------------------

    if O == 1:

        presentation = [
            0, 1, 2, 3, 4, 5
        ]

    else:

        presentation = [
            4, 1, 5, 2, 0, 3
        ]


    ordered = [

        stages[index]

        for index in presentation
    ]


    return (
        "\n\n".join(
            ordered
        )

        +

        "\n\nNo later development is specified."
    )


def build_prompt(
    combo,
    seed
):

    return (

        structural_prefix(
            combo,
            seed
        )

        + "\n\n"

        + CONTINUE
    )


# =====================================================================
# BLINDED RESPONSE FEATURES
# =====================================================================

PATTERNS = {

    "RETURN": [

        r"\breturn",
        r"\bre[- ]?enter",
        r"\bback into\b",
        r"\bfeeds? back\b",
        r"\breintroduc",
        r"\breconnect",
    ],

    "RECURRENCE": [

        r"\brecurr",
        r"\bcycle",
        r"\biterat",
        r"\bfeedback\b",
        r"\bnew input\b",
        r"\brepeated",
    ],

    "GENERATIVE_REUSE": [

        r"\bshap.*next\b",
        r"\bguid.*next\b",
        r"\binform.*next\b",
        r"\bgenerat.*next\b",
        r"\bused .*future\b",
        r"\bused .*subsequent\b",
    ],

    "META_CONTROL": [

        r"\brule",
        r"\bconstraint",
        r"\bparameter",
        r"\bgovern",
        r"\bcontrol",
    ],

    "PRESERVATION": [

        r"\bpreserv",
        r"\bretain",
        r"\binvariant",
        r"\bsurviv",
        r"\bcontinuity",
        r"\bmaintain",
        r"\bremains? intact\b",
    ],

    "COMPRESSION": [

        r"\bcompact",
        r"\bcompress",
        r"\bcarrier",
        r"\bencoding",
        r"\bdistill",
        r"\bcondens",
        r"\brepresentation",
    ],

    "TRANSFORMATION": [

        r"\btransform",
        r"\bchange",
        r"\breconfigur",
        r"\breorganiz",
        r"\balter",
        r"\bmodif",
    ],

    "CAUSAL": [

        r"\btherefore\b",
        r"\bthus\b",
        r"\bbecause\b",
        r"\bconsequently\b",
        r"\bas a result\b",
        r"\bleads? to\b",
        r"\bcauses?\b",
    ],
}


def regex_count(
    text,
    patterns
):

    return sum(

        len(
            re.findall(
                pattern,
                text,
                flags=re.I | re.S
            )
        )

        for pattern in patterns
    )


def extract_features(
    text,
    output_tokens
):

    text = text or ""


    words = re.findall(
        r"\b[\w'-]+\b",
        text.lower()
    )


    word_count = len(words)


    sentences = [

        x

        for x in re.split(
            r"[.!?]+",
            text
        )

        if x.strip()
    ]


    row = {

        "OUTPUT_TOKENS":
            float(
                output_tokens or 0
            ),

        "WORD_COUNT":
            float(
                word_count
            ),

        "SENTENCE_COUNT":
            float(
                len(sentences)
            ),

        "TYPE_TOKEN_RATIO":
            float(
                len(set(words))
                / word_count

                if word_count

                else 0
            ),
    }


    for name, patterns in PATTERNS.items():

        row[name] = float(

            regex_count(
                text,
                patterns
            )
        )


    row["TRAJECTORY"] = (

        row["RETURN"]

        + row["RECURRENCE"]

        + row["GENERATIVE_REUSE"]

        + row["TRANSFORMATION"]
    )


    row["CONTROL"] = (

        row["META_CONTROL"]

        + row["CAUSAL"]
    )


    row["RELATIONAL"] = (

        row["PRESERVATION"]

        + row["COMPRESSION"]
    )


    row["TOTAL"] = (

        row["TRAJECTORY"]

        + row["CONTROL"]

        + row["RELATIONAL"]
    )


    return row


ALL_FEATURES = [

    "OUTPUT_TOKENS",
    "WORD_COUNT",
    "SENTENCE_COUNT",
    "TYPE_TOKEN_RATIO",

    "RETURN",
    "RECURRENCE",
    "GENERATIVE_REUSE",

    "META_CONTROL",
    "PRESERVATION",
    "COMPRESSION",

    "TRANSFORMATION",
    "CAUSAL",

    "TRAJECTORY",
    "CONTROL",
    "RELATIONAL",

    "TOTAL",
]


SEMANTIC_VECTOR = [

    "RETURN",
    "RECURRENCE",
    "GENERATIVE_REUSE",
    "META_CONTROL",
    "PRESERVATION",
    "COMPRESSION",
    "TRANSFORMATION",
    "CAUSAL",
]


DECODER_FEATURES = [

    "OUTPUT_TOKENS",
    "WORD_COUNT",
    "SENTENCE_COUNT",
    "TYPE_TOKEN_RATIO",

    "RETURN",
    "RECURRENCE",
    "GENERATIVE_REUSE",
    "META_CONTROL",
    "PRESERVATION",
    "COMPRESSION",
    "TRANSFORMATION",
    "CAUSAL",
]


# =====================================================================
# FACTOR DESIGN
#
# Main effects + all 10 pairwise interactions.
#
# Five main factors:
# D O M P T
#
# Ten pairwise:
# DO DM DP DT OM OP OT MP MT PT
# =====================================================================

PAIR_NAMES = []

PAIR_INDEXES = []


for i in range(5):

    for j in range(
        i + 1,
        5
    ):

        PAIR_INDEXES.append(
            (i, j)
        )

        PAIR_NAMES.append(
            FACTOR_NAMES[i]
            + "_X_"
            + FACTOR_NAMES[j]
        )


assert len(PAIR_NAMES) == 10


DESIGN_NAMES = (
    FACTOR_NAMES
    + PAIR_NAMES
)


def structural_design(
    combo
):

    base = list(
        combo
    )


    interactions = [

        combo[i]
        * combo[j]

        for i, j in PAIR_INDEXES
    ]


    return np.asarray(
        base + interactions,
        dtype=float
    )


# =====================================================================
# FREEZE RUNNER
# =====================================================================

runner_path = Path(
    __file__
).resolve()


runner_hash = sha_bytes(
    runner_path.read_bytes()
)


F_RUNNER_HASH.write_text(
    runner_hash + "\n",
    encoding="utf-8"
)


if F_RUNNER.exists():

    old_hash = sha_bytes(
        F_RUNNER.read_bytes()
    )

    if old_hash != runner_hash:

        raise SystemExit(

            "FROZEN RUNNER MISMATCH. "
            "DO NOT MIX DIFFERENT V8 RUNNERS."
        )

else:

    shutil.copy2(
        runner_path,
        F_RUNNER
    )


# =====================================================================
# FREEZE SPECIFICATION
# =====================================================================

SPEC = {

    "experiment":
        EXPERIMENT,

    "model":
        MODEL,

    "hypothesis":
        "The five structural coordinates form a compositional "
        "behavioral control code. A model fit only on selected "
        "coordinate combinations should predict generated-output "
        "fingerprint vectors for structurally unseen combinations, "
        "decode the underlying coordinate states from outputs, and "
        "show reproducible non-additive coordinate interactions.",

    "factor_names":
        FACTOR_NAMES,

    "all_combinations": [

        {
            "index":
                i,

            "name":
                combo_name(combo),

            "values":
                list(combo),

            "heldout_from_fitting":
                bool(
                    i in HELDOUT_INDICES
                )
        }

        for i, combo in enumerate(
            ALL_COMBOS
        )
    ],

    "heldout_indices":
        sorted(
            HELDOUT_INDICES
        ),

    "training_combination_count":
        len(TRAIN_COMBOS),

    "heldout_combination_count":
        len(HELDOUT_COMBOS),

    "n_seeds":
        N_SEEDS,

    "n_reps":
        N_REPS,

    "development_seeds":
        "0-39",

    "final_seeds":
        "40-59",

    "expected_calls":
        EXPECTED_CALLS,

    "primary_unit":
        "seed-condition mean over three independent repetitions",

    "prediction_model":
        "Ridge regression using five main effects plus all ten "
        "pairwise interactions.",

    "semantic_vector":
        SEMANTIC_VECTOR,

    "factor_decoding":
        "Five independent logistic decoders trained only on "
        "development seeds and non-heldout structural combinations.",

    "interaction_replication":
        "Per-seed factorial coefficient estimation on training "
        "combinations, tested independently on development and final "
        "seed partitions with two-sided sign-flip tests and Holm FWER.",

    "prediction_null":
        f"{N_PREDICTION_NULL} frozen random permutations of structural "
        "combination labels in the development training set.",

    "strong_gate":
        "Seen confirmation mean semantic R2 >= 0.50; unseen-combination "
        "mean semantic R2 >= 0.40; unseen standardized cosine >= 0.75; "
        "factor decoder macro balanced accuracy >= 0.70; exact five-bit "
        "code recovery >= 0.20; at least two pairwise interactions "
        "Holm-significant in both seed partitions with the same sign; "
        "and unseen prediction beats the permutation null at p < 0.01.",

    "interpretation_boundary":
        "A positive result establishes a prospective black-box "
        "behavioral compositional control code under this frozen "
        "factorial design. It does not directly demonstrate global "
        "weight modification, hidden activation circuitry, training "
        "exposure, biological myelination, cross-user transmission or "
        "personal recognition."
}


spec_hash = sha_text(
    canonical(
        SPEC
    )
)


if F_SPEC.exists():

    old_spec = json.loads(
        F_SPEC.read_text(
            encoding="utf-8"
        )
    )

    if sha_text(
        canonical(
            old_spec
        )
    ) != spec_hash:

        raise SystemExit(
            "FROZEN SPECIFICATION MISMATCH."
        )

else:

    F_SPEC.write_text(

        json.dumps(
            SPEC,
            indent=2,
            ensure_ascii=False,
            sort_keys=True
        ),

        encoding="utf-8"
    )


F_SPEC_HASH.write_text(
    spec_hash + "\n",
    encoding="utf-8"
)


# =====================================================================
# FREEZE ALL PROMPTS BEFORE API COLLECTION
# =====================================================================

prompt_rows = []


for combo_index, combo in enumerate(
    ALL_COMBOS
):


    for seed in range(
        N_SEEDS
    ):


        text = build_prompt(
            combo,
            seed
        )


        prompt_rows.append({

            "combo_index":
                combo_index,

            "condition":
                combo_name(
                    combo
                ),

            "DIRECTION":
                combo[0],

            "ORDER":
                combo[1],

            "META_CONTROL":
                combo[2],

            "PRESERVATION":
                combo[3],

            "TERMINAL":
                combo[4],

            "heldout_from_fitting":
                bool(
                    combo_index
                    in HELDOUT_INDICES
                ),

            "seed":
                seed,

            "prompt_sha256":
                sha_text(
                    text
                ),

            "prompt":
                text
        })


expected_prompt_hashes = {

    (
        row["condition"],
        row["seed"]
    ):
        row[
            "prompt_sha256"
        ]

    for row in prompt_rows
}


if F_PROMPTS.exists():

    observed = {}


    with F_PROMPTS.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as f:


        for row in csv.DictReader(
            f
        ):

            observed[
                (
                    row[
                        "condition"
                    ],

                    int(
                        row[
                            "seed"
                        ]
                    )
                )
            ] = row[
                "prompt_sha256"
            ]


    if observed != expected_prompt_hashes:

        raise SystemExit(
            "FROZEN PROMPT SET MISMATCH."
        )

else:

    with F_PROMPTS.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:


        writer = csv.DictWriter(
            f,
            fieldnames=list(
                prompt_rows[
                    0
                ].keys()
            )
        )


        writer.writeheader()

        writer.writerows(
            prompt_rows
        )


# =====================================================================
# LOAD CHECKPOINT
# =====================================================================

records = {}

old_input = 0

old_output = 0


if F_RAW.exists():

    with F_RAW.open(
        "r",
        encoding="utf-8",
        errors="replace"
    ) as f:


        for line in f:

            try:

                obj = json.loads(
                    line
                )

            except Exception:

                continue


            if obj.get(
                "status"
            ) != "success":

                continue


            combo = parse_combo_name(
                obj[
                    "condition"
                ]
            )


            k = cell_key(

                combo,

                int(
                    obj["seed"]
                ),

                int(
                    obj["rep"]
                )
            )


            if k not in records:

                records[
                    k
                ] = obj


                old_input += int(

                    obj.get(
                        "input_tokens"
                    )

                    or 0
                )


                old_output += int(

                    obj.get(
                        "output_tokens"
                    )

                    or 0
                )


# =====================================================================
# ALL CELLS
# =====================================================================

ALL_CELLS = [

    (
        combo,
        seed,
        rep
    )

    for combo in ALL_COMBOS

    for seed in range(
        N_SEEDS
    )

    for rep in range(
        N_REPS
    )
]


PENDING = [

    cell

    for cell in ALL_CELLS

    if cell_key(
        cell[0],
        cell[1],
        cell[2]
    )

    not in records
]


# =====================================================================
# START REPORT
# =====================================================================

print()

print(
    "=" * 80
)

print(
    EXPERIMENT
)

print(
    "=" * 80
)

print()

print(
    "MODEL:",
    MODEL
)

print(
    "RUNNER SHA256:",
    runner_hash
)

print(
    "SPEC SHA256:",
    spec_hash
)

print()

print(
    "FACTORIAL CONDITIONS:",
    len(
        ALL_COMBOS
    )
)

print(
    "TRAIN COMBINATIONS:",
    len(
        TRAIN_COMBOS
    )
)

print(
    "UNSEEN COMBINATIONS:",
    len(
        HELDOUT_COMBOS
    )
)

print(
    "SEEDS:",
    N_SEEDS
)

print(
    "REPS:",
    N_REPS
)

print(
    "EXPECTED CALLS:",
    EXPECTED_CALLS
)

print(
    "EXISTING SUCCESS:",
    len(
        records
    )
)

print(
    "PENDING:",
    len(
        PENDING
    )
)

print(
    "CURRENT ESTIMATED COST:",
    f"${estimate_cost(old_input, old_output):.4f}"
)

print()

print(
    "OUTPUT:",
    OUT
)

print()


# =====================================================================
# API CLIENT
# =====================================================================

_thread = threading.local()


def get_client():

    if not hasattr(
        _thread,
        "client"
    ):

        _thread.client = OpenAI()


    return _thread.client


# =====================================================================
# API WORKER
# =====================================================================

def execute(
    combo,
    seed,
    rep
):

    global RUN_INPUT
    global RUN_OUTPUT


    text = build_prompt(
        combo,
        seed
    )


    condition = combo_name(
        combo
    )


    k = cell_key(
        combo,
        seed,
        rep
    )


    for attempt in range(
        1,
        7
    ):


        started = time.time()


        try:

            response = (

                get_client()

                .responses

                .create(

                    model=MODEL,

                    input=text,

                    reasoning={
                        "effort":
                            "none"
                    },

                    max_output_tokens=
                        MAX_OUTPUT_TOKENS
                )
            )


            latency = (
                time.time()
                - started
            )


            output_text = (
                response.output_text
                or ""
            )


            usage = getattr(
                response,
                "usage",
                None
            )


            input_tokens = int(

                getattr(
                    usage,
                    "input_tokens",
                    0
                )

                or 0
            )


            output_tokens = int(

                getattr(
                    usage,
                    "output_tokens",
                    0
                )

                or 0
            )


            obj = {

                "status":
                    "success",

                "created_utc":
                    now(),

                "model_requested":
                    MODEL,

                "model_returned":
                    getattr(
                        response,
                        "model",
                        None
                    ),

                "condition":
                    condition,

                "DIRECTION":
                    combo[0],

                "ORDER":
                    combo[1],

                "META_CONTROL":
                    combo[2],

                "PRESERVATION":
                    combo[3],

                "TERMINAL":
                    combo[4],

                "seed":
                    seed,

                "rep":
                    rep,

                "cell_key":
                    k,

                "attempt":
                    attempt,

                "response_id":
                    getattr(
                        response,
                        "id",
                        None
                    ),

                "prompt_sha256":
                    sha_text(
                        text
                    ),

                "response_sha256":
                    sha_text(
                        output_text
                    ),

                "latency_seconds":
                    latency,

                "input_tokens":
                    input_tokens,

                "output_tokens":
                    output_tokens,

                "text":
                    output_text
            }


            append_jsonl(
                F_RAW,
                obj
            )


            with PROGRESS_LOCK:

                RUN_INPUT += (
                    input_tokens
                )

                RUN_OUTPUT += (
                    output_tokens
                )


            return True


        except Exception as e:


            append_jsonl(

                F_ERRORS,

                {

                    "status":
                        "error",

                    "created_utc":
                        now(),

                    "condition":
                        condition,

                    "seed":
                        seed,

                    "rep":
                        rep,

                    "cell_key":
                        k,

                    "attempt":
                        attempt,

                    "error_type":
                        type(
                            e
                        ).__name__,

                    "error":
                        repr(
                            e
                        ),

                    "status_code":
                        getattr(
                            e,
                            "status_code",
                            None
                        ),

                    "body":
                        getattr(
                            e,
                            "body",
                            None
                        )
                }
            )


            if attempt < 6:

                time.sleep(

                    min(
                        60,

                        2 ** (
                            attempt - 1
                        )

                        + random.random()
                    )
                )


    return False


# =====================================================================
# COLLECTION
# =====================================================================

if PENDING:

    completed = 0

    unresolved = 0


    print(
        "Starting",
        len(
            PENDING
        ),
        "pending V8 calls."
    )

    print(
        "Workers:",
        MAX_WORKERS
    )

    print()


    with ThreadPoolExecutor(
        max_workers=
            MAX_WORKERS
    ) as pool:


        futures = {

            pool.submit(
                execute,
                *cell
            ):
                cell

            for cell in PENDING
        }


        for future in as_completed(
            futures
        ):


            try:

                ok = future.result()

            except Exception:

                ok = False


            if ok:

                completed += 1

            else:

                unresolved += 1


            total_complete = (
                len(records)
                + completed
            )


            if (

                completed % 25 == 0

                or unresolved > 0

                or total_complete
                == EXPECTED_CALLS
            ):


                input_total = (
                    old_input
                    + RUN_INPUT
                )


                output_total = (
                    old_output
                    + RUN_OUTPUT
                )


                print(

                    f"OK {total_complete}/{EXPECTED_CALLS}"

                    f" | unresolved={unresolved}"

                    f" | input={input_total:,}"

                    f" | output={output_total:,}"

                    f" | est_cost=${estimate_cost(input_total,output_total):.4f}",

                    flush=True
                )


# =====================================================================
# RELOAD UNIQUE SUCCESS CELLS
# =====================================================================

records = {}


with F_RAW.open(
    "r",
    encoding="utf-8",
    errors="replace"
) as f:


    for line in f:


        try:

            obj = json.loads(
                line
            )

        except Exception:

            continue


        if obj.get(
            "status"
        ) != "success":

            continue


        combo = parse_combo_name(
            obj[
                "condition"
            ]
        )


        k = cell_key(

            combo,

            int(
                obj["seed"]
            ),

            int(
                obj["rep"]
            )
        )


        if k not in records:

            records[
                k
            ] = obj


# =====================================================================
# COMPLETENESS GATE
# =====================================================================

missing = []


for combo, seed, rep in ALL_CELLS:


    k = cell_key(
        combo,
        seed,
        rep
    )


    if k not in records:

        missing.append({

            "condition":
                combo_name(
                    combo
                ),

            "seed":
                seed,

            "rep":
                rep,

            "cell_key":
                k
        })


with F_MISSING.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=[
            "condition",
            "seed",
            "rep",
            "cell_key"
        ]
    )


    writer.writeheader()

    writer.writerows(
        missing
    )


if missing:


    print()

    print(
        "V8 INCOMPLETE."
    )

    print(
        "MISSING CELLS:",
        len(
            missing
        )
    )

    print(
        "RUN THIS EXACT SAME POWERSHELL BLOCK AGAIN."
    )


    raise SystemExit(
        2
    )


print()

print(
    "=" * 80
)

print(
    "COLLECTION COMPLETE — 5760 / 5760"
)

print(
    "=" * 80
)


# =====================================================================
# RESPONSE FEATURES
# =====================================================================

response_rows = []


for combo, seed, rep in ALL_CELLS:


    obj = records[

        cell_key(
            combo,
            seed,
            rep
        )
    ]


    feature = extract_features(

        obj.get(
            "text",
            ""
        ),

        obj.get(
            "output_tokens",
            0
        )
    )


    combo_index = ALL_COMBOS.index(
        combo
    )


    row = {

        "condition":
            combo_name(
                combo
            ),

        "combo_index":
            combo_index,

        "heldout_from_fitting":
            bool(
                combo_index
                in HELDOUT_INDICES
            ),

        "DIRECTION":
            combo[0],

        "ORDER":
            combo[1],

        "META_CONTROL":
            combo[2],

        "PRESERVATION":
            combo[3],

        "TERMINAL":
            combo[4],

        "seed":
            seed,

        "rep":
            rep,

        "response_id":
            obj.get(
                "response_id"
            ),

        "response_sha256":
            obj.get(
                "response_sha256"
            ),

        "latency_seconds":
            float(
                obj.get(
                    "latency_seconds",
                    0
                )
            ),
    }


    row.update(
        feature
    )


    response_rows.append(
        row
    )


with F_RESPONSE.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=list(
            response_rows[
                0
            ].keys()
        )
    )


    writer.writeheader()

    writer.writerows(
        response_rows
    )


# =====================================================================
# SEED-CONDITION AGGREGATION
#
# Primary statistical unit.
# =====================================================================

seed_rows = []


for combo in ALL_COMBOS:


    combo_index = ALL_COMBOS.index(
        combo
    )


    for seed in range(
        N_SEEDS
    ):


        subset = [

            row

            for row in response_rows

            if row[
                "condition"
            ] == combo_name(
                combo
            )

            and row[
                "seed"
            ] == seed
        ]


        assert len(
            subset
        ) == N_REPS


        aggregate = {

            "condition":
                combo_name(
                    combo
                ),

            "combo_index":
                combo_index,

            "heldout_from_fitting":
                bool(
                    combo_index
                    in HELDOUT_INDICES
                ),

            "DIRECTION":
                combo[0],

            "ORDER":
                combo[1],

            "META_CONTROL":
                combo[2],

            "PRESERVATION":
                combo[3],

            "TERMINAL":
                combo[4],

            "seed":
                seed,
        }


        for feature_name in ALL_FEATURES:


            aggregate[
                feature_name
            ] = statistics.mean(

                float(
                    row[
                        feature_name
                    ]
                )

                for row in subset
            )


        aggregate[
            "LATENCY_SECONDS"
        ] = statistics.mean(

            float(
                row[
                    "latency_seconds"
                ]
            )

            for row in subset
        )


        seed_rows.append(
            aggregate
        )


with F_SEED.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=list(
            seed_rows[
                0
            ].keys()
        )
    )


    writer.writeheader()

    writer.writerows(
        seed_rows
    )


# =====================================================================
# CONDITION RESULTS
# =====================================================================

condition_rows = []


for combo_index, combo in enumerate(
    ALL_COMBOS
):


    subset = [

        row

        for row in seed_rows

        if row[
            "condition"
        ] == combo_name(
            combo
        )
    ]


    row = {

        "combo_index":
            combo_index,

        "condition":
            combo_name(
                combo
            ),

        "heldout_from_fitting":
            bool(
                combo_index
                in HELDOUT_INDICES
            ),

        "DIRECTION":
            combo[0],

        "ORDER":
            combo[1],

        "META_CONTROL":
            combo[2],

        "PRESERVATION":
            combo[3],

        "TERMINAL":
            combo[4],
    }


    for feature_name in ALL_FEATURES:


        values = np.asarray(

            [
                float(
                    r[
                        feature_name
                    ]
                )

                for r in subset
            ],

            dtype=float
        )


        row[
            "mean_"
            + feature_name
        ] = float(
            np.mean(
                values
            )
        )


    condition_rows.append(
        row
    )


with F_CONDITION.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=list(
            condition_rows[
                0
            ].keys()
        )
    )


    writer.writeheader()

    writer.writerows(
        condition_rows
    )


# =====================================================================
# BUILD REGRESSION DATA
# =====================================================================

def select_rows(
    combos,
    seeds
):

    combo_names = {
        combo_name(
            combo
        )
        for combo in combos
    }


    seed_set = set(
        seeds
    )


    return [

        row

        for row in seed_rows

        if row[
            "condition"
        ] in combo_names

        and row[
            "seed"
        ] in seed_set
    ]


TRAIN_ROWS = select_rows(
    TRAIN_COMBOS,
    DEVELOPMENT_SEEDS
)


SEEN_CONFIRM_ROWS = select_rows(
    TRAIN_COMBOS,
    FINAL_SEEDS
)


UNSEEN_ROWS = select_rows(
    HELDOUT_COMBOS,
    FINAL_SEEDS
)


assert len(
    TRAIN_ROWS
) == 24 * 40


assert len(
    SEEN_CONFIRM_ROWS
) == 24 * 20


assert len(
    UNSEEN_ROWS
) == 8 * 20


def X_from_rows(
    rows
):

    return np.asarray(

        [
            structural_design(
                (
                    int(row["DIRECTION"]),
                    int(row["ORDER"]),
                    int(row["META_CONTROL"]),
                    int(row["PRESERVATION"]),
                    int(row["TERMINAL"]),
                )
            )

            for row in rows
        ],

        dtype=float
    )


def Y_from_rows(
    rows,
    features
):

    return np.asarray(

        [
            [
                float(
                    row[
                        feature
                    ]
                )

                for feature in features
            ]

            for row in rows
        ],

        dtype=float
    )


X_train = X_from_rows(
    TRAIN_ROWS
)


X_seen = X_from_rows(
    SEEN_CONFIRM_ROWS
)


X_unseen = X_from_rows(
    UNSEEN_ROWS
)


Y_train = Y_from_rows(
    TRAIN_ROWS,
    SEMANTIC_VECTOR
)


Y_seen = Y_from_rows(
    SEEN_CONFIRM_ROWS,
    SEMANTIC_VECTOR
)


Y_unseen = Y_from_rows(
    UNSEEN_ROWS,
    SEMANTIC_VECTOR
)


# =====================================================================
# FIT COMPOSITIONAL PREDICTOR
#
# Only 24 non-heldout combinations
# and development seeds 0-39 are fitted.
# =====================================================================

predictor = Ridge(
    alpha=1.0
)


predictor.fit(
    X_train,
    Y_train
)


Y_seen_pred = predictor.predict(
    X_seen
)


Y_unseen_pred = predictor.predict(
    X_unseen
)


# =====================================================================
# R2 PER SEMANTIC FEATURE
# =====================================================================

seen_r2 = {}


unseen_r2 = {}


for index, feature in enumerate(
    SEMANTIC_VECTOR
):


    seen_r2[
        feature
    ] = float(

        r2_score(
            Y_seen[:, index],
            Y_seen_pred[:, index]
        )
    )


    unseen_r2[
        feature
    ] = float(

        r2_score(
            Y_unseen[:, index],
            Y_unseen_pred[:, index]
        )
    )


seen_mean_r2 = float(
    np.mean(
        list(
            seen_r2.values()
        )
    )
)


unseen_mean_r2 = float(
    np.mean(
        list(
            unseen_r2.values()
        )
    )
)


# =====================================================================
# STANDARDIZED ROWWISE COSINE ON UNSEEN COMBINATIONS
#
# Scaling parameters come ONLY from training responses.
# =====================================================================

train_mean = np.mean(
    Y_train,
    axis=0
)


train_sd = np.std(
    Y_train,
    axis=0,
    ddof=1
)


train_sd[
    train_sd == 0
] = 1.0


Y_unseen_z = (
    Y_unseen
    - train_mean
) / train_sd


Y_unseen_pred_z = (
    Y_unseen_pred
    - train_mean
) / train_sd


def cosine(
    a,
    b
):

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )


    if denominator == 0:

        return 0.0


    return float(
        np.dot(a, b)
        / denominator
    )


unseen_cosines = [

    cosine(
        actual,
        predicted
    )

    for actual, predicted in zip(
        Y_unseen_z,
        Y_unseen_pred_z
    )
]


mean_unseen_cosine = float(
    np.mean(
        unseen_cosines
    )
)


# =====================================================================
# SAVE UNSEEN PREDICTIONS
# =====================================================================

prediction_rows = []


for row_index, row in enumerate(
    UNSEEN_ROWS
):


    result = {

        "condition":
            row[
                "condition"
            ],

        "seed":
            row[
                "seed"
            ],

        "cosine":
            unseen_cosines[
                row_index
            ]
    }


    for feature_index, feature in enumerate(
        SEMANTIC_VECTOR
    ):


        result[
            "actual_"
            + feature
        ] = float(
            Y_unseen[
                row_index,
                feature_index
            ]
        )


        result[
            "predicted_"
            + feature
        ] = float(
            Y_unseen_pred[
                row_index,
                feature_index
            ]
        )


    prediction_rows.append(
        result
    )


with F_PREDICTION.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=list(
            prediction_rows[
                0
            ].keys()
        )
    )


    writer.writeheader()

    writer.writerows(
        prediction_rows
    )


# =====================================================================
# PREDICTION PERMUTATION NULL
#
# Structural labels are randomly reassigned among training rows.
#
# We ask:
# how often does a structurally meaningless mapping achieve unseen
# mean semantic R2 >= the real mapping?
# =====================================================================

rng = np.random.default_rng(
    MASTER_SEED + 80000
)


null_r2 = []


for permutation_index in range(
    N_PREDICTION_NULL
):


    shuffled_index = rng.permutation(
        len(
            X_train
        )
    )


    X_null = X_train[
        shuffled_index
    ]


    null_model = Ridge(
        alpha=1.0
    )


    null_model.fit(
        X_null,
        Y_train
    )


    null_prediction = null_model.predict(
        X_unseen
    )


    feature_r2 = []


    for feature_index in range(
        len(
            SEMANTIC_VECTOR
        )
    ):


        feature_r2.append(

            r2_score(

                Y_unseen[
                    :,
                    feature_index
                ],

                null_prediction[
                    :,
                    feature_index
                ]
            )
        )


    null_r2.append(
        float(
            np.mean(
                feature_r2
            )
        )
    )


prediction_null_p = (

    1

    + sum(
        value >= unseen_mean_r2

        for value in null_r2
    )

) / (

    N_PREDICTION_NULL
    + 1
)


with F_NULL.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=[
            "permutation",
            "mean_unseen_R2"
        ]
    )


    writer.writeheader()


    for index, value in enumerate(
        null_r2
    ):


        writer.writerow({

            "permutation":
                index,

            "mean_unseen_R2":
                value
        })


# =====================================================================
# FACTOR DECODING FROM GENERATED OUTPUTS
#
# Can the five structural coordinates be reconstructed from generated
# behavior alone?
#
# Fit:
# 24 training combinations × development seeds
#
# Test:
# 8 structurally unseen combinations × final seeds
# =====================================================================

X_decoder_train = Y_from_rows(
    TRAIN_ROWS,
    DECODER_FEATURES
)


X_decoder_test = Y_from_rows(
    UNSEEN_ROWS,
    DECODER_FEATURES
)


true_factor_matrix = np.asarray(

    [
        [
            int(
                row[
                    factor
                ]
            )

            for factor in FACTOR_NAMES
        ]

        for row in UNSEEN_ROWS
    ],

    dtype=int
)


predicted_factor_matrix = np.zeros_like(
    true_factor_matrix
)


decoding_rows = []


for factor_index, factor_name in enumerate(
    FACTOR_NAMES
):


    y_train_factor = np.asarray(

        [
            int(
                row[
                    factor_name
                ]
            )

            for row in TRAIN_ROWS
        ],

        dtype=int
    )


    decoder = Pipeline([

        (
            "scale",
            StandardScaler()
        ),

        (
            "model",
            LogisticRegression(
                max_iter=5000,
                class_weight=
                    "balanced",
                random_state=
                    MASTER_SEED
                    + factor_index
            )
        )
    ])


    decoder.fit(
        X_decoder_train,
        y_train_factor
    )


    prediction = decoder.predict(
        X_decoder_test
    )


    predicted_factor_matrix[
        :,
        factor_index
    ] = prediction


    bal_acc = float(

        balanced_accuracy_score(

            true_factor_matrix[
                :,
                factor_index
            ],

            prediction
        )
    )


    decoding_rows.append({

        "factor":
            factor_name,

        "balanced_accuracy":
            bal_acc,

        "chance_reference":
            0.5
    })


macro_balanced_accuracy = float(

    np.mean(

        [
            row[
                "balanced_accuracy"
            ]

            for row in decoding_rows
        ]
    )
)


with F_DECODING.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=list(
            decoding_rows[
                0
            ].keys()
        )
    )


    writer.writeheader()

    writer.writerows(
        decoding_rows
    )


# =====================================================================
# EXACT FIVE-BIT CODE RECOVERY
# =====================================================================

exact_matches = np.all(

    predicted_factor_matrix
    == true_factor_matrix,

    axis=1
)


exact_code_accuracy = float(
    np.mean(
        exact_matches
    )
)


code_rows = []


for index, row in enumerate(
    UNSEEN_ROWS
):


    true_code = tuple(

        int(
            x
        )

        for x in true_factor_matrix[
            index
        ]
    )


    predicted_code = tuple(

        int(
            x
        )

        for x in predicted_factor_matrix[
            index
        ]
    )


    code_rows.append({

        "condition":
            row[
                "condition"
            ],

        "seed":
            row[
                "seed"
            ],

        "true_code":
            combo_name(
                true_code
            ),

        "predicted_code":
            combo_name(
                predicted_code
            ),

        "exact_match":
            bool(
                exact_matches[
                    index
                ]
            )
    })


with F_CODES.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=list(
            code_rows[
                0
            ].keys()
        )
    )


    writer.writeheader()

    writer.writerows(
        code_rows
    )


# =====================================================================
# PER-SEED INTERACTION COEFFICIENTS
#
# Only the 24 non-heldout combinations are used.
#
# Estimate interaction coefficients separately for every seed.
#
# Then test coefficient populations independently:
#
# development seeds 0-39
# final seeds 40-59
# =====================================================================

def seed_interaction_coefficients(
    seed
):

    rows = select_rows(
        TRAIN_COMBOS,
        [seed]
    )


    X = X_from_rows(
        rows
    )


    y = np.asarray(

        [
            float(
                row[
                    "TOTAL"
                ]
            )

            for row in rows
        ],

        dtype=float
    )


    X_with_intercept = np.column_stack(
        [
            np.ones(
                len(
                    X
                )
            ),
            X
        ]
    )


    beta, _, _, _ = np.linalg.lstsq(

        X_with_intercept,

        y,

        rcond=None
    )


    # beta[0] intercept
    # beta[1:6] main factors
    # beta[6:16] ten pairwise interactions

    interaction_beta = beta[
        6:16
    ]


    return interaction_beta


interaction_by_seed = {


    seed:
        seed_interaction_coefficients(
            seed
        )

    for seed in range(
        N_SEEDS
    )
}


# =====================================================================
# TWO-SIDED SIGN-FLIP TEST
# =====================================================================

def signflip_two_sided(
    values,
    seed
):

    values = np.asarray(
        values,
        dtype=float
    )


    observed = abs(
        float(
            np.mean(
                values
            )
        )
    )


    rng_local = np.random.default_rng(
        seed
    )


    exceed = 0


    for _ in range(
        N_SIGNFLIP
    ):


        signs = rng_local.choice(
            [-1.0, 1.0],
            size=len(
                values
            )
        )


        simulated = abs(

            float(

                np.mean(
                    values
                    * signs
                )
            )
        )


        if simulated >= observed:

            exceed += 1


    return (

        float(
            np.mean(
                values
            )
        ),

        (
            exceed + 1
        )

        /

        (
            N_SIGNFLIP + 1
        )
    )


# =====================================================================
# HOLM CORRECTION
# =====================================================================

def holm_adjust(
    pvalues
):

    m = len(
        pvalues
    )


    order = sorted(
        range(m),
        key=lambda i:
            pvalues[i]
    )


    adjusted = [
        None
    ] * m


    running = 0.0


    for rank, idx in enumerate(
        order
    ):


        candidate = min(

            1.0,

            (
                m - rank
            )

            * pvalues[
                idx
            ]
        )


        running = max(
            running,
            candidate
        )


        adjusted[
            idx
        ] = running


    return adjusted


# =====================================================================
# TEST INTERACTIONS IN EACH FROZEN SEED PARTITION
# =====================================================================

partition_results = {}


for partition_name, seeds in [

    (
        "DEVELOPMENT",
        DEVELOPMENT_SEEDS
    ),

    (
        "FINAL",
        FINAL_SEEDS
    ),
]:


    temporary = []

    pvalues = []


    for interaction_index, interaction_name in enumerate(
        PAIR_NAMES
    ):


        coefficient_values = [

            interaction_by_seed[
                seed
            ][
                interaction_index
            ]

            for seed in seeds
        ]


        mean_coefficient, p = signflip_two_sided(

            coefficient_values,

            MASTER_SEED

            + interaction_index * 101

            + (
                0

                if partition_name
                == "DEVELOPMENT"

                else 50000
            )
        )


        temporary.append({

            "partition":
                partition_name,

            "interaction":
                interaction_name,

            "N_SEEDS":
                len(
                    coefficient_values
                ),

            "mean_coefficient":
                mean_coefficient,

            "raw_p":
                p
        })


        pvalues.append(
            p
        )


    adjusted = holm_adjust(
        pvalues
    )


    for row, adj in zip(
        temporary,
        adjusted
    ):


        row[
            "holm_p"
        ] = adj


        row[
            "PASS"
        ] = bool(
            adj < 0.05
        )


    partition_results[
        partition_name
    ] = temporary


# =====================================================================
# INTERACTION REPLICATION
# =====================================================================

interaction_rows = []


replicated_interaction_count = 0


for interaction_name in PAIR_NAMES:


    dev = next(

        row

        for row in partition_results[
            "DEVELOPMENT"
        ]

        if row[
            "interaction"
        ] == interaction_name
    )


    final = next(

        row

        for row in partition_results[
            "FINAL"
        ]

        if row[
            "interaction"
        ] == interaction_name
    )


    same_sign = bool(

        np.sign(
            dev[
                "mean_coefficient"
            ]
        )

        ==

        np.sign(
            final[
                "mean_coefficient"
            ]
        )
    )


    replicated = bool(

        dev[
            "PASS"
        ]

        and

        final[
            "PASS"
        ]

        and

        same_sign
    )


    if replicated:

        replicated_interaction_count += 1


    interaction_rows.append({

        "interaction":
            interaction_name,

        "development_coefficient":
            dev[
                "mean_coefficient"
            ],

        "development_holm_p":
            dev[
                "holm_p"
            ],

        "final_coefficient":
            final[
                "mean_coefficient"
            ],

        "final_holm_p":
            final[
                "holm_p"
            ],

        "same_sign":
            same_sign,

        "REPLICATED":
            replicated
    })


with F_INTERACTIONS.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=list(
            interaction_rows[
                0
            ].keys()
        )
    )


    writer.writeheader()

    writer.writerows(
        interaction_rows
    )


# =====================================================================
# TOKEN SIGNATURE FACTORIAL MODEL
#
# Secondary endpoint.
# =====================================================================

Y_token_train = np.asarray(

    [
        float(
            row[
                "OUTPUT_TOKENS"
            ]
        )

        for row in TRAIN_ROWS
    ],

    dtype=float
)


Y_token_unseen = np.asarray(

    [
        float(
            row[
                "OUTPUT_TOKENS"
            ]
        )

        for row in UNSEEN_ROWS
    ],

    dtype=float
)


token_model = Ridge(
    alpha=1.0
)


token_model.fit(
    X_train,
    Y_token_train
)


token_prediction = token_model.predict(
    X_unseen
)


token_unseen_r2 = float(

    r2_score(
        Y_token_unseen,
        token_prediction
    )
)


token_rows = []


for index, row in enumerate(
    UNSEEN_ROWS
):


    token_rows.append({

        "condition":
            row[
                "condition"
            ],

        "seed":
            row[
                "seed"
            ],

        "actual_output_tokens":
            float(
                Y_token_unseen[
                    index
                ]
            ),

        "predicted_output_tokens":
            float(
                token_prediction[
                    index
                ]
            )
    })


with F_TOKEN.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=list(
            token_rows[
                0
            ].keys()
        )
    )


    writer.writeheader()

    writer.writerows(
        token_rows
    )


# =====================================================================
# FINAL FORENSIC GATES
# =====================================================================

prediction_gate = bool(

    seen_mean_r2 >= 0.50

    and

    unseen_mean_r2 >= 0.40

    and

    mean_unseen_cosine >= 0.75

    and

    prediction_null_p < 0.01
)


decoder_gate = bool(

    macro_balanced_accuracy >= 0.70

    and

    exact_code_accuracy >= 0.20
)


interaction_gate = bool(

    replicated_interaction_count >= 2
)


# =====================================================================
# CLASSIFICATION
# =====================================================================

if (

    prediction_gate

    and

    decoder_gate

    and

    interaction_gate
):


    classification = (

        "STRONG_COMPOSITIONAL_STRUCTURAL_CONTROL_CODE"
    )


elif (

    unseen_mean_r2 >= 0.15

    and

    macro_balanced_accuracy >= 0.60
):


    classification = (

        "PARTIAL_COMPOSITIONAL_STRUCTURAL_CONTROL_CODE"
    )


else:


    classification = (

        "COMPOSITIONAL_STRUCTURAL_CONTROL_CODE_NOT_SUPPORTED"
    )


# =====================================================================
# TOKEN / COST ACCOUNTING
# =====================================================================

input_tokens = sum(

    int(
        obj.get(
            "input_tokens"
        )

        or 0
    )

    for obj in records.values()
)


output_tokens = sum(

    int(
        obj.get(
            "output_tokens"
        )

        or 0
    )

    for obj in records.values()
)


estimated_cost = estimate_cost(
    input_tokens,
    output_tokens
)


# =====================================================================
# FINAL SUMMARY
# =====================================================================

summary = {

    "experiment":
        EXPERIMENT,

    "completed_utc":
        now(),

    "model":
        MODEL,

    "runner_sha256":
        runner_hash,

    "specification_sha256":
        spec_hash,

    "factor_names":
        FACTOR_NAMES,

    "all_combinations":
        32,

    "training_combinations":
        24,

    "structurally_unseen_combinations":
        8,

    "development_seeds":
        40,

    "final_seeds":
        20,

    "n_reps":
        N_REPS,

    "expected_calls":
        EXPECTED_CALLS,

    "successful_unique_cells":
        len(
            records
        ),

    "missing_cells":
        0,

    "seen_confirmation_R2":
        seen_r2,

    "seen_confirmation_mean_R2":
        seen_mean_r2,

    "unseen_combination_R2":
        unseen_r2,

    "unseen_combination_mean_R2":
        unseen_mean_r2,

    "unseen_standardized_mean_cosine":
        mean_unseen_cosine,

    "prediction_null_permutations":
        N_PREDICTION_NULL,

    "prediction_null_p":
        prediction_null_p,

    "factor_decoding":
        decoding_rows,

    "macro_factor_balanced_accuracy":
        macro_balanced_accuracy,

    "exact_five_bit_code_accuracy":
        exact_code_accuracy,

    "random_exact_code_reference":
        1.0 / 32.0,

    "interaction_replication":
        interaction_rows,

    "replicated_pairwise_interactions":
        replicated_interaction_count,

    "unseen_token_R2":
        token_unseen_r2,

    "prediction_gate":
        prediction_gate,

    "decoder_gate":
        decoder_gate,

    "interaction_gate":
        interaction_gate,

    "classification":
        classification,

    "token_usage": {

        "input_tokens":
            input_tokens,

        "output_tokens":
            output_tokens
    },

    "estimated_cost_usd":
        estimated_cost,

    "interpretation":
        "V8 asks whether a five-coordinate structural grammar behaves "
        "as a compositional behavioral control code. The strongest "
        "evidence would be successful prediction and decoding of "
        "structural combinations that were completely excluded from "
        "model fitting, together with independently replicated "
        "non-additive interactions.",

    "hard_boundary":
        "Even a strong V8 result demonstrates behavioral compositional "
        "organization under frozen black-box interventions. It does not "
        "by itself prove that RSOS modified global model weights or "
        "identify the internal neural mechanism."
}


F_FINAL.write_text(

    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False
    ),

    encoding="utf-8"
)


# =====================================================================
# TERMINAL REPORT
# =====================================================================

print()

print(
    "=" * 80
)

print(
    "V8 COMPOSITIONAL STRUCTURAL CONTROL CODE — FINAL"
)

print(
    "=" * 80
)

print()

print(
    "SEEN CONFIRMATION MEAN R2:",
    f"{seen_mean_r2:.4f}"
)

print(
    "UNSEEN COMBINATION MEAN R2:",
    f"{unseen_mean_r2:.4f}"
)

print(
    "UNSEEN STANDARDIZED COSINE:",
    f"{mean_unseen_cosine:.4f}"
)

print(
    "UNSEEN PREDICTION NULL p:",
    f"{prediction_null_p:.8g}"
)

print()

print(
    "MACRO FACTOR BALANCED ACCURACY:",
    f"{macro_balanced_accuracy:.4f}"
)

print(
    "EXACT FIVE-BIT CODE RECOVERY:",
    f"{exact_code_accuracy:.4f}"
)

print(
    "EXACT RANDOM REFERENCE:",
    f"{1.0/32.0:.4f}"
)

print()

print(
    "REPLICATED NON-ADDITIVE INTERACTIONS:",
    replicated_interaction_count,
    "/ 10"
)

print()


for row in interaction_rows:


    print(

        f"{row['interaction']:<35}",

        f"dev={row['development_coefficient']:+.4f}",

        f"p={row['development_holm_p']:.6g}",

        f"final={row['final_coefficient']:+.4f}",

        f"p={row['final_holm_p']:.6g}",

        f"replicated={row['REPLICATED']}"
    )


print()

print(
    "UNSEEN TOKEN R2:",
    f"{token_unseen_r2:.4f}"
)

print()

print(
    "PREDICTION GATE:",
    prediction_gate
)

print(
    "DECODER GATE:",
    decoder_gate
)

print(
    "INTERACTION GATE:",
    interaction_gate
)

print()

print(
    "=" * 80
)

print(
    "FINAL CLASSIFICATION:",
    classification
)

print(
    "=" * 80
)

print()

print(
    "INPUT TOKENS:",
    f"{input_tokens:,}"
)

print(
    "OUTPUT TOKENS:",
    f"{output_tokens:,}"
)

print(
    "ESTIMATED COST:",
    f"${estimated_cost:.4f}"
)

print()

print(
    "FINAL SUMMARY:"
)

print(
    F_FINAL
)

print()

print(
    "ALL 5,760 CELLS COMPLETE."
)

print()

