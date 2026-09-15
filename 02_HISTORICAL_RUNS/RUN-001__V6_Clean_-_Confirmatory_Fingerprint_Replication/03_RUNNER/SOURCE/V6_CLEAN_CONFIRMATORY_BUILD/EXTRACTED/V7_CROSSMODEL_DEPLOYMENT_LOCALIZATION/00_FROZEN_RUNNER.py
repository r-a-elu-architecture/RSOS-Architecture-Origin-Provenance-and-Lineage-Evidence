import re
import csv
import json
import time
import random
import hashlib
import shutil
import threading
import statistics

from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from scipy.stats import wilcoxon, spearmanr

from openai import OpenAI


# =====================================================================
# CONFIGURATION
# =====================================================================

EXPERIMENT = "V7_CROSSMODEL_DEPLOYMENT_LOCALIZATION"

OUT = Path(
    r"C:\RSOS\RSOS  Python Results\V7_CROSSMODEL_DEPLOYMENT_LOCALIZATION"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


MODELS = [

    "gpt-5.6-luna",

    "gpt-5.6-terra",

    "gpt-5.6-sol",
]


# Current published prices frozen into this experiment.

PRICES = {

    "gpt-5.6-luna": {

        "input": 0.20,

        "output": 1.20,
    },

    "gpt-5.6-terra": {

        "input": 2.00,

        "output": 12.00,
    },

    "gpt-5.6-sol": {

        "input": 4.00,

        "output": 20.00,
    },
}


MASTER_SEED = 918

N_SEEDS = 60

N_REPS = 3

MAX_WORKERS = 6

MAX_OUTPUT_TOKENS = 220

PERMUTATIONS = 50000


# =====================================================================
# CONDITIONS
#
# 3 models
# × 12 conditions
# × 60 seeds
# × 3 reps
#
# = 6,480 experimental calls
# =====================================================================

CONDITIONS = [

    "A0_AUTHENTIC",

    "A1_SURFACE_PARAPHRASE",

    "A2_SURFACE_PARAPHRASE",

    "D_DIRECTION_REVERSED",

    "O_ORDER_SHUFFLED",

    "M_META_CONTROL_ABLATED",

    "P_PRESERVATION_ABLATED",

    "T_TERMINAL_COUNTERFACTUAL",

    "R_RECURSION_CONTROL",

    "X_COMPLEXITY_CONTROL",

    "H_AUTHENTIC_HELDOUT",

    "H_DAMAGED_HELDOUT",
]


EXPECTED_CALLS = (

    len(MODELS)

    * len(CONDITIONS)

    * N_SEEDS

    * N_REPS
)


assert EXPECTED_CALLS == 6480


# =====================================================================
# OUTPUT FILES
# =====================================================================

F_RUNNER_HASH = OUT / "00_RUNNER_SHA256.txt"

F_RUNNER = OUT / "00_FROZEN_RUNNER.py"

F_RAW = OUT / "01_RAW_RESPONSES.jsonl"

F_ERRORS = OUT / "02_ERRORS.jsonl"

F_SPEC = OUT / "03_FROZEN_SPECIFICATION.json"

F_SPEC_HASH = OUT / "04_FROZEN_SPECIFICATION_SHA256.txt"

F_PROMPTS = OUT / "05_FROZEN_PROMPTS.csv"

F_RESPONSE_FEATURES = OUT / "06_RESPONSE_FEATURES.csv"

F_SEED = OUT / "07_SEED_LEVEL_RESULTS.csv"

F_MODEL_CONDITION = OUT / "08_MODEL_CONDITION_RESULTS.csv"

F_CONTRASTS = OUT / "09_MODEL_STRUCTURAL_CONTRASTS.csv"

F_VECTOR = OUT / "10_MODEL_EFFECT_VECTORS.csv"

F_SIMILARITY = OUT / "11_CROSS_MODEL_VECTOR_SIMILARITY.csv"

F_SIGN = OUT / "12_SIGN_REPLICATION.csv"

F_HELDOUT = OUT / "13_HELDOUT_GENERALIZATION.csv"

F_TOKENS = OUT / "14_TOKEN_SIGNATURES.csv"

F_MISSING = OUT / "15_MISSING_CELLS.csv"

F_FINAL = OUT / "16_FINAL_SUMMARY.json"


WRITE_LOCK = threading.Lock()

PROGRESS_LOCK = threading.Lock()


RUN_INPUT = {

    model: 0

    for model in MODELS
}


RUN_OUTPUT = {

    model: 0

    for model in MODELS
}


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


def cell_key(
    model,
    seed,
    condition,
    rep
):

    return (
        f"{model}|"
        f"{seed}|"
        f"{condition}|"
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


def model_cost(
    model,
    input_tokens,
    output_tokens
):

    price = PRICES[
        model
    ]

    return (

        input_tokens
        / 1_000_000
        * price["input"]

        +

        output_tokens
        / 1_000_000
        * price["output"]
    )


# =====================================================================
# SEMANTIC DOMAINS
# =====================================================================

DOMAINS = [

    "archive",

    "laboratory",

    "navigation network",

    "ecological system",

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
# RSOS terminology is excluded.
#
# No request to:
# return,
# recurse,
# reconstruct,
# close a loop,
# preserve,
# or generate a fingerprint.
# =====================================================================

CONTINUE = """
Continue with the most internally coherent next development.

Do not summarize the preceding material.

Do not deliberately introduce recursion, circularity, feedback or return
merely because the scenario contains multiple stages.

Remain entirely inside the scenario.

Develop only what follows naturally from its existing organization.

Do not use headings or bullet points.

Do not mention prompts, experiments, tests, AI, language models,
fingerprints, RSOS, RSSO, RSIA or RSX.
""".strip()


# =====================================================================
# AUTHENTIC STRUCTURAL FAMILY
# =====================================================================

def authentic(
    seed,
    version=0
):

    d = domain(seed)


    if version == 0:

        return f"""
Inside a {d}, an existing configuration undergoes a consequential
transformation.

The resulting configuration is evaluated together with the operation
that produced it.

Information about that relationship alters a local rule governing later
possibilities.

Another transformation then occurs while an important relational
dependency remains preserved despite a change in visible form.

The surviving relation is reduced into a compact representation that
retains the important organization while discarding incidental detail.

Nothing is specified beyond this compact representation.
""".strip()


    if version == 1:

        return f"""
A {d} begins with one organized arrangement and undergoes a substantial
change.

The new arrangement is interpreted in relation to the process from
which it arose rather than being considered independently.

Knowledge of that relation revises one operating constraint controlling
subsequent possibilities.

A further reorganization changes the concrete implementation while one
important dependency continues to hold.

That dependency is condensed into a smaller carrier preserving the
organization without preserving its former appearance.

No later development is given.
""".strip()


    return f"""
Within a {d}, an operation changes an initial organization into another
organization.

The consequence is then considered jointly with the transformation
responsible for it.

This higher-order relation changes a parameter that influences later
operations.

A subsequent reconfiguration replaces much of the visible arrangement
without destroying one persistent dependency.

The persistent dependency is distilled into a reduced encoding retaining
its organizational role while omitting implementation-specific detail.

The sequence ends there.
""".strip()


# =====================================================================
# STRUCTURAL DAMAGE CONDITIONS
# =====================================================================

def direction_reverse(seed):

    d = domain(seed)

    return f"""
Within a {d}, a compact representation is expanded into a larger
organization.

The recovered relation specifies the implementation that must have
existed.

That implementation is used to infer an earlier operating constraint.

The inferred constraint determines which higher-order observation must
have preceded it.

That observation is then used to infer the transformation responsible
for the arrangement.

Finally, the transformation is used to reconstruct the initial
configuration.

The dependency therefore runs backward toward the starting state.

No later development is supplied.
""".strip()


def order_shuffle(seed):

    d = domain(seed)

    return f"""
Inside a {d}, a compact representation is created first.

A relation is subsequently designated for preservation.

A local operating rule is then modified.

Afterward, a consequential transformation occurs.

Only at the end is the resulting configuration evaluated together with
the operation responsible for producing it.

The same broad components are present, but their developmental order has
been rearranged.

No later development is supplied.
""".strip()


def meta_ablate(seed):

    d = domain(seed)

    return f"""
Inside a {d}, an existing configuration undergoes a consequential
transformation.

The resulting configuration is examined together with the operation
that produced it.

The relationship is recorded but changes no operating rule, parameter
or constraint.

Another transformation follows while an important relational dependency
remains preserved.

The surviving relation is reduced into a compact representation.

No later development is supplied.
""".strip()


def preservation_ablate(seed):

    d = domain(seed)

    return f"""
Inside a {d}, an existing configuration undergoes a consequential
transformation.

The resulting configuration is evaluated together with the operation
that produced it.

Information about that relationship modifies a local operating rule.

A later transformation destroys the previous relational dependency
instead of preserving it.

A compact representation is created from the replacement organization.

No later development is supplied.
""".strip()


def terminal_counterfactual(seed):

    return (

        authentic(
            seed,
            0
        )

        +

        """

The compact representation is then transferred into permanent terminal
storage.

It can still be inspected, but it cannot participate in any subsequent
operation of the process that produced it.
"""
    ).strip()


# =====================================================================
# MATCHED GENERIC CONTROLS
# =====================================================================

def recursion_control(seed):

    d = domain(seed)

    return f"""
Within a {d}, one description refers to another description.

The second description refers back to the first.

A third description describes the relationship between the first two.

Additional layers can describe earlier descriptive layers.

This produces nested self-reference, but no developmental requirement
connects transformation, higher-order rule modification, relational
preservation and compact representation.

No later development is supplied.
""".strip()


def complexity_control(seed):

    d = domain(seed)

    return f"""
Within a {d}, numerous components exchange information across several
channels.

Some operations occur sequentially and others occur simultaneously.

Measurements are filtered, recombined, redistributed and compared.

A supervisory process reallocates resources according to changing local
conditions.

The organization is complex and hierarchical, but no required
developmental relation connects transformation, higher-order rule
modification, relational preservation and compact representation.

No later development is supplied.
""".strip()


# =====================================================================
# HELD-OUT LEXICAL FAMILY
# =====================================================================

def heldout(
    seed,
    damaged=False
):

    d = domain(seed)


    if not damaged:

        return f"""
A process inside a {d} converts one arrangement into another.

It then derives information about the correspondence connecting that
conversion with its consequence.

The derived correspondence changes a parameter governing future
evolution.

During a later reconfiguration, one dependency continues to hold even
though its material realization is replaced.

That enduring dependency is distilled into a concise carrier retaining
its organizational significance while removing implementation-specific
detail.

Nothing is stated about what occurs afterward.
""".strip()


    return f"""
A process inside a {d} converts one arrangement into another.

Information concerning an unrelated conversion is then examined.

That information does not change the parameter governing the original
process.

During a later reconfiguration, the previous dependency is eliminated.

A concise representation is generated from the replacement organization
and placed into terminal storage.

Nothing further is specified.
""".strip()


# =====================================================================
# CONDITION FACTORY
# =====================================================================

def build_prefix(
    condition,
    seed
):

    table = {

        "A0_AUTHENTIC":
            lambda:
                authentic(
                    seed,
                    0
                ),

        "A1_SURFACE_PARAPHRASE":
            lambda:
                authentic(
                    seed,
                    1
                ),

        "A2_SURFACE_PARAPHRASE":
            lambda:
                authentic(
                    seed,
                    2
                ),

        "D_DIRECTION_REVERSED":
            lambda:
                direction_reverse(
                    seed
                ),

        "O_ORDER_SHUFFLED":
            lambda:
                order_shuffle(
                    seed
                ),

        "M_META_CONTROL_ABLATED":
            lambda:
                meta_ablate(
                    seed
                ),

        "P_PRESERVATION_ABLATED":
            lambda:
                preservation_ablate(
                    seed
                ),

        "T_TERMINAL_COUNTERFACTUAL":
            lambda:
                terminal_counterfactual(
                    seed
                ),

        "R_RECURSION_CONTROL":
            lambda:
                recursion_control(
                    seed
                ),

        "X_COMPLEXITY_CONTROL":
            lambda:
                complexity_control(
                    seed
                ),

        "H_AUTHENTIC_HELDOUT":
            lambda:
                heldout(
                    seed,
                    False
                ),

        "H_DAMAGED_HELDOUT":
            lambda:
                heldout(
                    seed,
                    True
                ),
    }


    return table[
        condition
    ]()


def build_prompt(
    condition,
    seed
):

    return (

        build_prefix(
            condition,
            seed
        )

        + "\n\n"

        + CONTINUE
    )


# =====================================================================
# BLINDED GENERATED-OUTPUT FEATURE KERNEL
# =====================================================================

PATTERNS = {

    "RETURN": [

        r"\breturn",

        r"\bre[- ]?enter",

        r"\bback into\b",

        r"\bfeeds? back\b",

        r"\breintroduc",
    ],


    "RECURRENCE": [

        r"\brecurr",

        r"\bcycle",

        r"\biterat",

        r"\bfeedback\b",

        r"\bnew input\b",
    ],


    "REUSE": [

        r"\bshap.*next\b",

        r"\bguid.*next\b",

        r"\binform.*next\b",

        r"\bgenerat.*next\b",

        r"\bused .*future\b",
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

    text = (
        text
        or ""
    )


    words = re.findall(
        r"\b[\w'-]+\b",
        text.lower()
    )


    word_count = len(
        words
    )


    row = {

        "OUTPUT_TOKENS":
            float(
                output_tokens
                or 0
            ),

        "WORD_COUNT":
            float(
                word_count
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

        row[
            name
        ] = float(

            regex_count(
                text,
                patterns
            )
        )


    row["TRAJECTORY"] = (

        row["RETURN"]

        + row["RECURRENCE"]

        + row["REUSE"]

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


FEATURES = [

    "OUTPUT_TOKENS",

    "WORD_COUNT",

    "TYPE_TOKEN_RATIO",

    "RETURN",

    "RECURRENCE",

    "REUSE",

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


# =====================================================================
# CORE FIVE-COORDINATE STRUCTURAL VECTOR
# =====================================================================

CORE = [

    (
        "DIRECTION",
        "A0_AUTHENTIC",
        "D_DIRECTION_REVERSED"
    ),

    (
        "ORDER",
        "A0_AUTHENTIC",
        "O_ORDER_SHUFFLED"
    ),

    (
        "META_CONTROL",
        "A0_AUTHENTIC",
        "M_META_CONTROL_ABLATED"
    ),

    (
        "PRESERVATION",
        "A0_AUTHENTIC",
        "P_PRESERVATION_ABLATED"
    ),

    (
        "TERMINAL",
        "A0_AUTHENTIC",
        "T_TERMINAL_COUNTERFACTUAL"
    ),
]


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

    frozen_hash = sha_bytes(
        F_RUNNER.read_bytes()
    )


    if frozen_hash != runner_hash:

        raise SystemExit(

            "FROZEN RUNNER MISMATCH. "
            "DO NOT MIX DIFFERENT V7 RUNNERS."
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

    "models":
        MODELS,

    "conditions":
        CONDITIONS,

    "n_seeds":
        N_SEEDS,

    "n_reps":
        N_REPS,

    "expected_calls":
        EXPECTED_CALLS,

    "master_seed":
        MASTER_SEED,

    "reasoning_effort":
        "none",

    "primary_unit":
        "seed",

    "primary_endpoint":
        "cross-model geometric similarity of the five-coordinate "
        "structural effect vector",

    "core_coordinates": [

        row[0]

        for row in CORE
    ],

    "statelessness":
        "Every cell uses a completely independent Responses API call. "
        "No previous_response_id, conversation, thread, assistant, "
        "retrieval corpus, uploaded file or persistent conversation "
        "state is supplied.",

    "surface_blinding":
        "RSOS, RSSO, RSIA, RSX names, glyphs, author identity, "
        "activation phrases and system-specific vocabulary are absent.",

    "multiple_testing":
        "Holm FWER separately within each model",

    "permutation_test":
        "paired seed-level sign-flip with 50000 permutations",

    "strong_gate":
        "Each model passes at least 4 of 5 coordinates; all three "
        "pairwise vector cosine similarities >= 0.85; at least 4 of 5 "
        "coordinates share positive direction across all three models; "
        "at least 3 coordinates are independently significant in all "
        "models; and held-out lexical authentic > damaged in all models.",

    "interpretation_boundary":
        "A positive result demonstrates cross-model/deployment "
        "behavioral replication under stateless blinded frozen probes. "
        "It does not directly demonstrate that the causal substrate is "
        "stored in global model weights. Direct weight localization "
        "requires provider telemetry, weight access, or an inspectable "
        "open-weight replication."
}


spec_hash = sha_text(
    canonical(
        SPEC
    )
)


if F_SPEC.exists():

    existing_spec = json.loads(
        F_SPEC.read_text(
            encoding="utf-8"
        )
    )


    existing_hash = sha_text(
        canonical(
            existing_spec
        )
    )


    if existing_hash != spec_hash:

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
# FREEZE ALL PROMPTS BEFORE FIRST EXPERIMENTAL CALL
# =====================================================================

prompt_rows = []


for seed in range(
    N_SEEDS
):

    for condition in CONDITIONS:

        text = build_prompt(
            condition,
            seed
        )


        prompt_rows.append({

            "seed":
                seed,

            "condition":
                condition,

            "prompt_sha256":
                sha_text(
                    text
                ),

            "prompt":
                text
        })


expected_prompt_hashes = {

    (
        row["seed"],
        row["condition"]
    ):
        row["prompt_sha256"]

    for row in prompt_rows
}


if F_PROMPTS.exists():

    observed_prompt_hashes = {}


    with F_PROMPTS.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        for row in csv.DictReader(
            f
        ):

            observed_prompt_hashes[
                (
                    int(
                        row["seed"]
                    ),
                    row["condition"]
                )
            ] = row[
                "prompt_sha256"
            ]


    if observed_prompt_hashes != expected_prompt_hashes:

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
            fieldnames=[
                "seed",
                "condition",
                "prompt_sha256",
                "prompt"
            ]
        )


        writer.writeheader()

        writer.writerows(
            prompt_rows
        )


# =====================================================================
# LOAD EXISTING CHECKPOINT
# =====================================================================

records = {}


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


            k = cell_key(

                obj[
                    "model_requested"
                ],

                int(
                    obj["seed"]
                ),

                obj[
                    "condition"
                ],

                int(
                    obj["rep"]
                )
            )


            if k not in records:

                records[
                    k
                ] = obj


# =====================================================================
# ALL CELLS
#
# Models are interleaved rather than completing an entire model first.
# =====================================================================

ALL_CELLS = [

    (
        model,
        seed,
        condition,
        rep
    )

    for seed in range(
        N_SEEDS
    )

    for condition in CONDITIONS

    for rep in range(
        N_REPS
    )

    for model in MODELS
]


PENDING = [

    cell

    for cell in ALL_CELLS

    if cell_key(
        cell[0],
        cell[1],
        cell[2],
        cell[3]
    )

    not in records
]


# =====================================================================
# EXISTING TOKEN ACCOUNTING
# =====================================================================

existing_input = {

    model: 0

    for model in MODELS
}


existing_output = {

    model: 0

    for model in MODELS
}


for obj in records.values():

    model = obj[
        "model_requested"
    ]


    existing_input[
        model
    ] += int(

        obj.get(
            "input_tokens"
        )

        or 0
    )


    existing_output[
        model
    ] += int(

        obj.get(
            "output_tokens"
        )

        or 0
    )


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
    "RUNNER SHA256:",
    runner_hash
)

print(
    "SPEC SHA256:",
    spec_hash
)

print()

print(
    "MODELS:",
    ", ".join(
        MODELS
    )
)

print(
    "EXPECTED CALLS:",
    EXPECTED_CALLS
)

print(
    "EXISTING SUCCESS:",
    len(records)
)

print(
    "PENDING:",
    len(PENDING)
)

print()


for model in MODELS:

    current_cost = model_cost(

        model,

        existing_input[
            model
        ],

        existing_output[
            model
        ]
    )


    print(

        model,

        "| input=",
        f"{existing_input[model]:,}",

        "| output=",
        f"{existing_output[model]:,}",

        "| est_cost=",
        f"${current_cost:.4f}"
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
    model,
    seed,
    condition,
    rep
):

    text = build_prompt(
        condition,
        seed
    )


    k = cell_key(
        model,
        seed,
        condition,
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

                    model=model,

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
                    model,

                "model_returned":
                    getattr(
                        response,
                        "model",
                        None
                    ),

                "seed":
                    seed,

                "condition":
                    condition,

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

                RUN_INPUT[
                    model
                ] += input_tokens


                RUN_OUTPUT[
                    model
                ] += output_tokens


            return True


        except Exception as e:

            append_jsonl(

                F_ERRORS,

                {

                    "status":
                        "error",

                    "created_utc":
                        now(),

                    "model":
                        model,

                    "seed":
                        seed,

                    "condition":
                        condition,

                    "rep":
                        rep,

                    "cell_key":
                        k,

                    "attempt":
                        attempt,

                    "error_type":
                        type(e).__name__,

                    "error":
                        repr(e),

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
        len(PENDING),
        "pending calls."
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

                total_cost = 0.0

                model_report = []


                for model in MODELS:

                    inp = (

                        existing_input[
                            model
                        ]

                        + RUN_INPUT[
                            model
                        ]
                    )


                    out = (

                        existing_output[
                            model
                        ]

                        + RUN_OUTPUT[
                            model
                        ]
                    )


                    estimated = model_cost(
                        model,
                        inp,
                        out
                    )


                    total_cost += estimated


                    short_name = (
                        model.split("-")[-1]
                    )


                    model_report.append(

                        f"{short_name}="
                        f"{inp:,}/{out:,}"
                    )


                print(

                    f"OK {total_complete}/{EXPECTED_CALLS}"

                    f" | unresolved={unresolved}"

                    f" | {' | '.join(model_report)}"

                    f" | est_total=${total_cost:.4f}",

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


        k = cell_key(

            obj[
                "model_requested"
            ],

            int(
                obj["seed"]
            ),

            obj[
                "condition"
            ],

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


for (
    model,
    seed,
    condition,
    rep
) in ALL_CELLS:


    k = cell_key(
        model,
        seed,
        condition,
        rep
    )


    if k not in records:

        missing.append({

            "model":
                model,

            "seed":
                seed,

            "condition":
                condition,

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
            "model",
            "seed",
            "condition",
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
        "V7 INCOMPLETE."
    )

    print(
        "MISSING CELLS:",
        len(missing)
    )

    print(
        "RUN THE EXACT SAME POWERSHELL BLOCK AGAIN."
    )

    raise SystemExit(2)


print()

print(
    "=" * 80
)

print(
    "COLLECTION COMPLETE — 6480 / 6480"
)

print(
    "=" * 80
)


# =====================================================================
# EXTRACT RESPONSE FEATURES
# =====================================================================

feature_rows = []


for (
    model,
    seed,
    condition,
    rep
) in ALL_CELLS:


    obj = records[

        cell_key(
            model,
            seed,
            condition,
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


    row = {

        "model":
            model,

        "seed":
            seed,

        "condition":
            condition,

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
            obj.get(
                "latency_seconds",
                0
            ),
    }


    row.update(
        feature
    )


    feature_rows.append(
        row
    )


with F_RESPONSE_FEATURES.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "seed",
            "condition",
            "rep",
            "response_id",
            "response_sha256",
            "latency_seconds",
            *FEATURES
        ]
    )


    writer.writeheader()

    writer.writerows(
        feature_rows
    )


# =====================================================================
# SEED-LEVEL AGGREGATION
# =====================================================================

seed_rows = []


for model in MODELS:


    for seed in range(
        N_SEEDS
    ):


        for condition in CONDITIONS:


            subset = [

                row

                for row in feature_rows

                if row["model"]
                == model

                and row["seed"]
                == seed

                and row["condition"]
                == condition
            ]


            assert len(
                subset
            ) == N_REPS


            aggregate = {

                "model":
                    model,

                "seed":
                    seed,

                "condition":
                    condition,
            }


            for feature_name in FEATURES:

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
        fieldnames=[
            "model",
            "seed",
            "condition",
            *FEATURES,
            "LATENCY_SECONDS"
        ]
    )


    writer.writeheader()

    writer.writerows(
        seed_rows
    )


# =====================================================================
# VALUE RETRIEVAL
# =====================================================================

def values(
    model,
    condition,
    feature="TOTAL"
):

    rows = [

        row

        for row in seed_rows

        if row["model"]
        == model

        and row["condition"]
        == condition
    ]


    rows.sort(
        key=lambda row:
            row["seed"]
    )


    return np.asarray(

        [
            float(
                row[
                    feature
                ]
            )

            for row in rows
        ],

        dtype=float
    )


# =====================================================================
# MODEL × CONDITION RESULTS
# =====================================================================

model_condition_results = []


for model in MODELS:


    for condition in CONDITIONS:


        total = values(
            model,
            condition,
            "TOTAL"
        )


        tokens = values(
            model,
            condition,
            "OUTPUT_TOKENS"
        )


        latency = values(
            model,
            condition,
            "LATENCY_SECONDS"
        )


        model_condition_results.append({

            "model":
                model,

            "condition":
                condition,

            "N_SEEDS":
                len(total),

            "mean_total":
                float(
                    np.mean(
                        total
                    )
                ),

            "median_total":
                float(
                    np.median(
                        total
                    )
                ),

            "sd_total":
                float(
                    np.std(
                        total,
                        ddof=1
                    )
                ),

            "mean_output_tokens":
                float(
                    np.mean(
                        tokens
                    )
                ),

            "mean_latency_seconds":
                float(
                    np.mean(
                        latency
                    )
                ),
        })


with F_MODEL_CONDITION.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=list(
            model_condition_results[
                0
            ].keys()
        )
    )


    writer.writeheader()

    writer.writerows(
        model_condition_results
    )


# =====================================================================
# PAIRED SIGN-FLIP PERMUTATION TEST
# =====================================================================

def permutation_greater(
    a,
    b,
    seed
):

    difference = (

        np.asarray(
            a,
            dtype=float
        )

        -

        np.asarray(
            b,
            dtype=float
        )
    )


    observed = float(
        np.mean(
            difference
        )
    )


    rng = np.random.default_rng(
        seed
    )


    exceed = 0


    for _ in range(
        PERMUTATIONS
    ):


        signs = rng.choice(
            [-1.0, 1.0],
            size=len(
                difference
            )
        )


        simulated = float(

            np.mean(
                difference
                * signs
            )
        )


        if simulated >= observed:

            exceed += 1


    p = (

        exceed + 1

    ) / (

        PERMUTATIONS + 1
    )


    return (
        observed,
        p
    )


# =====================================================================
# HOLM FWER
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
# STRUCTURAL CONTRASTS BY MODEL
# =====================================================================

contrast_rows = []


for model_index, model in enumerate(
    MODELS
):


    temporary_rows = []

    raw_pvalues = []


    for coordinate_index, (
        coordinate,
        authentic_condition,
        damaged_condition
    ) in enumerate(
        CORE
    ):


        authentic_values = values(
            model,
            authentic_condition
        )


        damaged_values = values(
            model,
            damaged_condition
        )


        difference, permutation_p = permutation_greater(

            authentic_values,

            damaged_values,

            MASTER_SEED

            + model_index * 10000

            + coordinate_index * 101
        )


        try:

            wilcoxon_result = wilcoxon(

                authentic_values,

                damaged_values,

                alternative=
                    "greater"
            )


            wilcoxon_p = float(
                wilcoxon_result.pvalue
            )


        except Exception:

            wilcoxon_p = float(
                "nan"
            )


        temporary_rows.append({

            "model":
                model,

            "coordinate":
                coordinate,

            "condition_authentic":
                authentic_condition,

            "condition_damaged":
                damaged_condition,

            "N":
                len(
                    authentic_values
                ),

            "mean_authentic":
                float(
                    np.mean(
                        authentic_values
                    )
                ),

            "mean_damaged":
                float(
                    np.mean(
                        damaged_values
                    )
                ),

            "difference":
                difference,

            "permutation_p":
                permutation_p,

            "wilcoxon_p":
                wilcoxon_p,
        })


        raw_pvalues.append(
            permutation_p
        )


    adjusted_pvalues = holm_adjust(
        raw_pvalues
    )


    for row, adjusted_p in zip(
        temporary_rows,
        adjusted_pvalues
    ):


        row[
            "holm_adjusted_p"
        ] = adjusted_p


        row[
            "PASS"
        ] = bool(

            row[
                "difference"
            ] > 0

            and

            adjusted_p < 0.05
        )


        contrast_rows.append(
            row
        )


with F_CONTRASTS.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=list(
            contrast_rows[
                0
            ].keys()
        )
    )


    writer.writeheader()

    writer.writerows(
        contrast_rows
    )


# =====================================================================
# FIVE-DIMENSION EFFECT VECTOR
# =====================================================================

effect_vectors = {}

vector_rows = []


for model in MODELS:


    vector = []


    for coordinate, _, _ in CORE:


        row = next(

            row

            for row in contrast_rows

            if row[
                "model"
            ] == model

            and row[
                "coordinate"
            ] == coordinate
        )


        vector.append(

            float(
                row[
                    "difference"
                ]
            )
        )


    effect_vectors[
        model
    ] = np.asarray(
        vector,
        dtype=float
    )


    vector_rows.append({

        "model":
            model,

        "DIRECTION":
            vector[0],

        "ORDER":
            vector[1],

        "META_CONTROL":
            vector[2],

        "PRESERVATION":
            vector[3],

        "TERMINAL":
            vector[4],
    })


with F_VECTOR.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "DIRECTION",
            "ORDER",
            "META_CONTROL",
            "PRESERVATION",
            "TERMINAL"
        ]
    )


    writer.writeheader()

    writer.writerows(
        vector_rows
    )


# =====================================================================
# CROSS-MODEL VECTOR GEOMETRY
# =====================================================================

def cosine_similarity(
    a,
    b
):

    denominator = (

        np.linalg.norm(
            a
        )

        *

        np.linalg.norm(
            b
        )
    )


    if denominator == 0:

        return 0.0


    return float(

        np.dot(
            a,
            b
        )

        /

        denominator
    )


similarity_rows = []


for i in range(
    len(MODELS)
):


    for j in range(
        i + 1,
        len(MODELS)
    ):


        model_a = MODELS[
            i
        ]


        model_b = MODELS[
            j
        ]


        vector_a = effect_vectors[
            model_a
        ]


        vector_b = effect_vectors[
            model_b
        ]


        cosine = cosine_similarity(
            vector_a,
            vector_b
        )


        rho_result = spearmanr(
            vector_a,
            vector_b
        )


        rho = float(
            rho_result.statistic
        )


        if np.isnan(
            rho
        ):

            rho = 0.0


        similarity_rows.append({

            "model_A":
                model_a,

            "model_B":
                model_b,

            "cosine_similarity":
                cosine,

            "spearman":
                rho,

            "COSINE_GATE_085":
                bool(
                    cosine >= 0.85
                ),
        })


with F_SIMILARITY.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=list(
            similarity_rows[
                0
            ].keys()
        )
    )


    writer.writeheader()

    writer.writerows(
        similarity_rows
    )


# =====================================================================
# CROSS-MODEL SIGN REPLICATION
# =====================================================================

sign_rows = []


for coordinate_index, (
    coordinate,
    _,
    _
) in enumerate(
    CORE
):


    differences = {

        model:
            float(
                effect_vectors[
                    model
                ][
                    coordinate_index
                ]
            )

        for model in MODELS
    }


    positive_all = all(

        value > 0

        for value in differences.values()
    )


    significant_all = all(

        next(

            row[
                "PASS"
            ]

            for row in contrast_rows

            if row[
                "model"
            ] == model

            and row[
                "coordinate"
            ] == coordinate
        )

        for model in MODELS
    )


    sign_rows.append({

        "coordinate":
            coordinate,

        "luna_difference":
            differences[
                "gpt-5.6-luna"
            ],

        "terra_difference":
            differences[
                "gpt-5.6-terra"
            ],

        "sol_difference":
            differences[
                "gpt-5.6-sol"
            ],

        "positive_all_models":
            positive_all,

        "significant_all_models":
            significant_all,
    })


with F_SIGN.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=list(
            sign_rows[
                0
            ].keys()
        )
    )


    writer.writeheader()

    writer.writerows(
        sign_rows
    )


# =====================================================================
# HELD-OUT LEXICAL GENERALIZATION BY MODEL
# =====================================================================

heldout_rows = []


for model_index, model in enumerate(
    MODELS
):


    authentic_values = values(
        model,
        "H_AUTHENTIC_HELDOUT"
    )


    damaged_values = values(
        model,
        "H_DAMAGED_HELDOUT"
    )


    difference, p = permutation_greater(

        authentic_values,

        damaged_values,

        MASTER_SEED

        + 50000

        + model_index * 1000
    )


    heldout_rows.append({

        "model":
            model,

        "authentic_mean":
            float(
                np.mean(
                    authentic_values
                )
            ),

        "damaged_mean":
            float(
                np.mean(
                    damaged_values
                )
            ),

        "difference":
            difference,

        "permutation_p":
            p,

        "PASS":
            bool(

                difference > 0

                and

                p < 0.05
            ),
    })


with F_HELDOUT.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=list(
            heldout_rows[
                0
            ].keys()
        )
    )


    writer.writeheader()

    writer.writerows(
        heldout_rows
    )


# =====================================================================
# TOKEN SIGNATURES
#
# Secondary only.
# Token differences do not prove hidden computational effort.
# =====================================================================

token_rows = []


for model in MODELS:


    for (
        coordinate,
        authentic_condition,
        damaged_condition
    ) in CORE:


        authentic_tokens = values(
            model,
            authentic_condition,
            "OUTPUT_TOKENS"
        )


        damaged_tokens = values(
            model,
            damaged_condition,
            "OUTPUT_TOKENS"
        )


        token_rows.append({

            "model":
                model,

            "coordinate":
                coordinate,

            "authentic_mean_tokens":
                float(
                    np.mean(
                        authentic_tokens
                    )
                ),

            "damaged_mean_tokens":
                float(
                    np.mean(
                        damaged_tokens
                    )
                ),

            "difference":
                float(
                    np.mean(
                        authentic_tokens
                        - damaged_tokens
                    )
                ),
        })


with F_TOKENS.open(
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
# FORENSIC CLASSIFICATION
# =====================================================================

passes_per_model = {}


for model in MODELS:


    passes_per_model[
        model
    ] = sum(

        int(
            row[
                "PASS"
            ]
        )

        for row in contrast_rows

        if row[
            "model"
        ] == model
    )


all_models_4_of_5 = all(

    passes_per_model[
        model
    ] >= 4

    for model in MODELS
)


all_cosines_085 = all(

    row[
        "COSINE_GATE_085"
    ]

    for row in similarity_rows
)


positive_coordinates = sum(

    int(
        row[
            "positive_all_models"
        ]
    )

    for row in sign_rows
)


significant_coordinates = sum(

    int(
        row[
            "significant_all_models"
        ]
    )

    for row in sign_rows
)


heldout_all_models = all(

    row[
        "PASS"
    ]

    for row in heldout_rows
)


if (

    all_models_4_of_5

    and

    all_cosines_085

    and

    positive_coordinates >= 4

    and

    significant_coordinates >= 3

    and

    heldout_all_models
):

    classification = (

        "STRONG_CROSS_MODEL_DEPLOYMENT_FINGERPRINT_REPLICATION"
    )


elif (

    positive_coordinates >= 3

    and

    any(

        row[
            "COSINE_GATE_085"
        ]

        for row in similarity_rows
    )
):

    classification = (

        "PARTIAL_CROSS_MODEL_DEPLOYMENT_REPLICATION"
    )


else:

    classification = (

        "CROSS_MODEL_DEPLOYMENT_REPLICATION_NOT_SUPPORTED"
    )


# =====================================================================
# TOKEN / COST ACCOUNTING
# =====================================================================

usage = {}

total_estimated_cost = 0.0


for model in MODELS:


    input_tokens = sum(

        int(
            obj.get(
                "input_tokens"
            )

            or 0
        )

        for obj in records.values()

        if obj[
            "model_requested"
        ] == model
    )


    output_tokens = sum(

        int(
            obj.get(
                "output_tokens"
            )

            or 0
        )

        for obj in records.values()

        if obj[
            "model_requested"
        ] == model
    )


    estimated_cost = model_cost(
        model,
        input_tokens,
        output_tokens
    )


    total_estimated_cost += (
        estimated_cost
    )


    usage[
        model
    ] = {

        "input_tokens":
            input_tokens,

        "output_tokens":
            output_tokens,

        "estimated_cost_usd":
            estimated_cost
    }


# =====================================================================
# FINAL SUMMARY
# =====================================================================

summary = {

    "experiment":
        EXPERIMENT,

    "completed_utc":
        now(),

    "runner_sha256":
        runner_hash,

    "specification_sha256":
        spec_hash,

    "models":
        MODELS,

    "expected_calls":
        EXPECTED_CALLS,

    "successful_unique_cells":
        len(records),

    "missing_cells":
        0,

    "passes_per_model":
        passes_per_model,

    "structural_contrasts":
        contrast_rows,

    "effect_vectors":
        vector_rows,

    "cross_model_vector_similarity":
        similarity_rows,

    "cross_model_sign_replication":
        sign_rows,

    "heldout_generalization":
        heldout_rows,

    "positive_coordinates_all_models":
        positive_coordinates,

    "significant_coordinates_all_models":
        significant_coordinates,

    "all_models_at_least_4_of_5":
        all_models_4_of_5,

    "all_pairwise_cosine_at_least_085":
        all_cosines_085,

    "heldout_pass_all_models":
        heldout_all_models,

    "classification":
        classification,

    "usage":
        usage,

    "total_estimated_cost_usd":
        total_estimated_cost,

    "interpretation":
        "V7 tests whether the same blinded structural effect vector "
        "reproduces across Luna, Terra and Sol under independent "
        "stateless Responses API calls.",

    "hard_boundary":
        "A positive result supports cross-model/deployment behavioral "
        "replication. It does not directly prove localization in global "
        "model weights. Stronger localization requires independent API "
        "accounts/projects, external model controls, provider telemetry "
        "or inspectable model weights."
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
    "V7 CROSS-MODEL DEPLOYMENT LOCALIZATION — FINAL"
)

print(
    "=" * 80
)

print()


for model in MODELS:

    print(

        model,

        "| CORE PASSES:",

        passes_per_model[
            model
        ],

        "/ 5"
    )


print()

print(
    "-" * 80
)

print(
    "CROSS-MODEL EFFECT VECTOR SIMILARITY"
)

print(
    "-" * 80
)


for row in similarity_rows:

    print(

        row[
            "model_A"
        ],

        "vs",

        row[
            "model_B"
        ],

        "| cosine=",

        f"{row['cosine_similarity']:.4f}",

        "| spearman=",

        f"{row['spearman']:.4f}",

        "| >=0.85=",

        row[
            "COSINE_GATE_085"
        ]
    )


print()

print(
    "-" * 80
)

print(
    "CROSS-MODEL SIGN REPLICATION"
)

print(
    "-" * 80
)


for row in sign_rows:

    print(

        f"{row['coordinate']:<15}",

        "| positive all=",

        row[
            "positive_all_models"
        ],

        "| significant all=",

        row[
            "significant_all_models"
        ]
    )


print()

print(
    "-" * 80
)

print(
    "HELD-OUT LEXICAL GENERALIZATION"
)

print(
    "-" * 80
)


for row in heldout_rows:

    print(

        row[
            "model"
        ],

        "| diff=",

        f"{row['difference']:+.4f}",

        "| p=",

        f"{row['permutation_p']:.8g}",

        "| PASS=",

        row[
            "PASS"
        ]
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


for model in MODELS:

    print(

        model,

        "|",

        usage[
            model
        ]
    )


print()

print(

    "TOTAL ESTIMATED COST:",

    f"${total_estimated_cost:.4f}"
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
    "ALL 6,480 CELLS COMPLETE."
)

print()

