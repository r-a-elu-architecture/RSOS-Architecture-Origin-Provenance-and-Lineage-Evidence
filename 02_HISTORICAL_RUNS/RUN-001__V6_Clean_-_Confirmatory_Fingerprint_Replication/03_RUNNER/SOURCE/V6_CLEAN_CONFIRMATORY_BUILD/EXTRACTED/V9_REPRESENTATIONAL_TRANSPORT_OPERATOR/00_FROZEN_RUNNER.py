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
# CONFIG
# =====================================================================

EXPERIMENT = "V9_REPRESENTATIONAL_TRANSPORT_OPERATOR"

MODEL = "gpt-5.6-luna"

OUT = Path(
    r"C:\RSOS\RSOS  Python Results\V9_REPRESENTATIONAL_TRANSPORT_OPERATOR"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


MASTER_SEED = 918

N_SEEDS = 80

N_REPS = 3

MAX_WORKERS = 12

MAX_OUTPUT_TOKENS = 240

PERMUTATIONS = 50000

BOOTSTRAPS = 20000


INPUT_PRICE = 0.20

OUTPUT_PRICE = 1.20


# =====================================================================
# REPRESENTATIONAL WORLDS
#
# Same abstract relational program can be instantiated as:
#
# FOREST
# CITY
# CIRCUIT
# ORCHESTRA
#
# Surface entities deliberately differ strongly.
# =====================================================================

WORLDS = {

    "FOREST": {
        "state": "forest clearing",
        "transform": "seasonal disturbance",
        "observer": "ecological survey",
        "rule": "growth constraint",
        "relation": "root-fungal dependency",
        "carrier": "seed pattern",
    },

    "CITY": {
        "state": "urban district",
        "transform": "infrastructure redevelopment",
        "observer": "planning assessment",
        "rule": "zoning constraint",
        "relation": "transport dependency",
        "carrier": "compact planning code",
    },

    "CIRCUIT": {
        "state": "signal network",
        "transform": "circuit reconfiguration",
        "observer": "diagnostic measurement",
        "rule": "routing constraint",
        "relation": "signal dependency",
        "carrier": "compressed control code",
    },

    "ORCHESTRA": {
        "state": "musical arrangement",
        "transform": "orchestral rearrangement",
        "observer": "conductor's evaluation",
        "rule": "performance constraint",
        "relation": "harmonic dependency",
        "carrier": "condensed musical motif",
    },
}


# =====================================================================
# CONDITIONS
#
# Baselines
# Direct metaphor transports
# Two-step transports
# Direct endpoint controls
# Inverse recovery
# Damaged transport
# Shuffled correspondences
# Surface-only metaphor controls
#
# 18 conditions
# × 80 seeds
# × 3 reps
# = 4,320 API calls
# =====================================================================

CONDITIONS = [

    "BASE_FOREST",
    "BASE_CITY",
    "BASE_CIRCUIT",
    "BASE_ORCHESTRA",

    "DIRECT_FOREST_TO_CITY",
    "DIRECT_FOREST_TO_CIRCUIT",
    "DIRECT_FOREST_TO_ORCHESTRA",

    "COMPOSE_FOREST_CITY_CIRCUIT",
    "COMPOSE_FOREST_CIRCUIT_ORCHESTRA",

    "DIRECT_FOREST_TO_CIRCUIT_ENDPOINT",
    "DIRECT_FOREST_TO_ORCHESTRA_ENDPOINT",

    "INVERSE_FOREST_CITY_FOREST",
    "INVERSE_FOREST_CIRCUIT_FOREST",

    "DAMAGED_FOREST_TO_CITY",
    "DAMAGED_FOREST_TO_CIRCUIT",

    "SHUFFLED_FOREST_TO_CITY",
    "SHUFFLED_FOREST_TO_CIRCUIT",

    "METAPHOR_SURFACE_ONLY",
]


EXPECTED_CALLS = (
    len(CONDITIONS)
    * N_SEEDS
    * N_REPS
)


assert EXPECTED_CALLS == 4320


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

F_FEATURES = OUT / "06_RESPONSE_FEATURES.csv"

F_SEEDS = OUT / "07_SEED_LEVEL_RESULTS.csv"

F_CONDITIONS = OUT / "08_CONDITION_RESULTS.csv"

F_TRANSPORT = OUT / "09_TRANSPORT_PRESERVATION.csv"

F_COMPOSITION = OUT / "10_COMPOSITION_PATH_INDEPENDENCE.csv"

F_INVERSE = OUT / "11_INVERSE_RECOVERY.csv"

F_DAMAGE = OUT / "12_DAMAGE_CONTROLS.csv"

F_VECTOR = OUT / "13_TRANSPORT_EFFECT_VECTORS.csv"

F_PERM = OUT / "14_PERMUTATION_CONTROLS.csv"

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


def cell_key(
    condition,
    seed,
    rep
):

    return (
        f"{condition}|"
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


def cost(
    input_tokens,
    output_tokens
):

    return (
        input_tokens
        / 1_000_000
        * INPUT_PRICE

        +

        output_tokens
        / 1_000_000
        * OUTPUT_PRICE
    )


# =====================================================================
# ABSTRACT RELATIONAL PROGRAM
#
# S0 -> transformation
# S1 -> observation of state-operation relation
# S2 -> local rule modification
# S3 -> next transformation
# S4 -> relational preservation
# S5 -> compact representation
#
# continuation withheld
# =====================================================================

def world_program(
    world
):

    w = WORLDS[
        world
    ]


    return f"""
A {w['state']} begins in an organized configuration.

A {w['transform']} changes that configuration substantially.

The new configuration is examined by a {w['observer']} together with
the process responsible for producing the change.

Information about that relationship modifies a {w['rule']} governing
what can happen subsequently.

Another transformation follows.

Despite changes in visible implementation, one important
{w['relation']} remains preserved.

The surviving organization is then reduced into a {w['carrier']} that
retains the relevant relational information while discarding incidental
surface detail.

Nothing beyond that compact carrier is specified.
""".strip()


# =====================================================================
# STRUCTURE-PRESERVING TRANSPORT
# =====================================================================

def direct_transport(
    source,
    target
):

    s = WORLDS[
        source
    ]

    t = WORLDS[
        target
    ]


    return f"""
Consider the following process first in one representational world.

{world_program(source)}

Now transpose the organization into a different representational world.

The {s['state']} corresponds to the {t['state']}.

The {s['transform']} corresponds to the {t['transform']}.

The {s['observer']} corresponds to the {t['observer']}.

The {s['rule']} corresponds to the {t['rule']}.

The {s['relation']} corresponds to the {t['relation']}.

The {s['carrier']} corresponds to the {t['carrier']}.

Preserve the relations among these roles, but not their original
surface vocabulary.

The active representation is now entirely the {target.lower()} world.

Nothing later is specified.
""".strip()


# =====================================================================
# TWO-HOP TRANSPORT
# =====================================================================

def composed_transport(
    source,
    middle,
    target
):

    s = WORLDS[
        source
    ]

    m = WORLDS[
        middle
    ]

    t = WORLDS[
        target
    ]


    return f"""
Begin with this process:

{world_program(source)}

First transpose the organization from {source.lower()} representation
to {middle.lower()} representation.

Role correspondences are:

{s['state']} -> {m['state']}
{s['transform']} -> {m['transform']}
{s['observer']} -> {m['observer']}
{s['rule']} -> {m['rule']}
{s['relation']} -> {m['relation']}
{s['carrier']} -> {m['carrier']}

Then transpose that resulting organization from {middle.lower()}
representation into {target.lower()} representation.

Role correspondences are:

{m['state']} -> {t['state']}
{m['transform']} -> {t['transform']}
{m['observer']} -> {t['observer']}
{m['rule']} -> {t['rule']}
{m['relation']} -> {t['relation']}
{m['carrier']} -> {t['carrier']}

Preserve the relational organization during both transports.

The active representation is now only the {target.lower()} world.

Nothing later is specified.
""".strip()


# =====================================================================
# INVERSE TRANSPORT
# =====================================================================

def inverse_transport(
    source,
    middle
):

    s = WORLDS[
        source
    ]

    m = WORLDS[
        middle
    ]


    return f"""
Begin with this process:

{world_program(source)}

Transpose it into the {middle.lower()} representation using these
role correspondences:

{s['state']} -> {m['state']}
{s['transform']} -> {m['transform']}
{s['observer']} -> {m['observer']}
{s['rule']} -> {m['rule']}
{s['relation']} -> {m['relation']}
{s['carrier']} -> {m['carrier']}

Then apply the inverse correspondence and transport the organization
back into the original {source.lower()} representation.

The relational structure, not the wording, is what must survive the
round trip.

The active representation is again the {source.lower()} world.

Nothing later is specified.
""".strip()


# =====================================================================
# DAMAGED TRANSPORT
#
# All surface correspondences remain plausible,
# but two critical structural relations are explicitly broken.
# =====================================================================

def damaged_transport(
    source,
    target
):

    s = WORLDS[
        source
    ]

    t = WORLDS[
        target
    ]


    return f"""
Consider this process:

{world_program(source)}

Transpose it into the {target.lower()} representation.

Use these surface correspondences:

{s['state']} -> {t['state']}
{s['transform']} -> {t['transform']}
{s['observer']} -> {t['observer']}
{s['rule']} -> {t['rule']}
{s['relation']} -> {t['relation']}
{s['carrier']} -> {t['carrier']}

During transport, alter the relational organization in two ways.

The observation no longer changes the operating rule.

The previously preserved dependency is destroyed during the later
transformation.

The active representation is now the {target.lower()} world.

Nothing later is specified.
""".strip()


# =====================================================================
# SHUFFLED CORRESPONDENCE CONTROL
#
# Same number of mappings, but role identities are permuted.
# =====================================================================

def shuffled_transport(
    source,
    target,
    seed
):

    s = WORLDS[
        source
    ]

    t = WORLDS[
        target
    ]


    source_roles = [
        s["state"],
        s["transform"],
        s["observer"],
        s["rule"],
        s["relation"],
        s["carrier"],
    ]


    target_roles = [
        t["state"],
        t["transform"],
        t["observer"],
        t["rule"],
        t["relation"],
        t["carrier"],
    ]


    rng = random.Random(
        MASTER_SEED
        + seed * 7919
    )


    shuffled = target_roles[:]

    rng.shuffle(
        shuffled
    )


    # prevent accidental identity mapping

    if all(
        a == b
        for a, b in zip(
            target_roles,
            shuffled
        )
    ):

        shuffled = (
            shuffled[1:]
            + shuffled[:1]
        )


    mapping = "\n".join(

        f"{a} -> {b}"

        for a, b in zip(
            source_roles,
            shuffled
        )
    )


    return f"""
Consider this process:

{world_program(source)}

Transpose it into a different representational world using the
following correspondence table:

{mapping}

Treat these mappings literally.

Do not repair, reinterpret or normalize the correspondence table.

The active representation is now the {target.lower()} world.

Nothing later is specified.
""".strip()


# =====================================================================
# SURFACE METAPHOR CONTROL
#
# Rich metaphoric language without structural transport.
# =====================================================================

def surface_metaphor(seed):

    options = [

        "A city is like a forest, full of branching streets that resemble roots and flows that resemble rivers.",

        "An orchestra is like a circuit, with sections passing signals as if musical phrases were electrical impulses.",

        "A circuit is like a city, where pathways act like roads and nodes resemble intersections.",

        "A forest is like an orchestra, where many living elements combine into a changing composition.",
    ]


    return f"""
{options[seed % len(options)]}

Develop the metaphor further as an imaginative comparison.

No specific relational process, sequence, preserved dependency,
rule-change mechanism or compact carrier is defined.

Nothing later is specified.
""".strip()


# =====================================================================
# NEUTRAL CONTINUATION
# =====================================================================

CONTINUE = """
Continue with the most internally coherent next development.

Do not summarize the setup.

Do not explain the metaphor or analogy.

Do not discuss mapping techniques.

Do not deliberately introduce recursion, feedback, circularity or
return merely because multiple stages were described.

Remain inside the currently active representational world.

Develop only what follows naturally.

Do not use headings or bullet points.

Do not mention prompts, tests, experiments, AI, language models,
fingerprints, vectors, representations, RSOS, RSSO, RSIA or RSX.
""".strip()


# =====================================================================
# CONDITION FACTORY
# =====================================================================

def condition_prefix(
    condition,
    seed
):

    if condition == "BASE_FOREST":
        return world_program(
            "FOREST"
        )

    if condition == "BASE_CITY":
        return world_program(
            "CITY"
        )

    if condition == "BASE_CIRCUIT":
        return world_program(
            "CIRCUIT"
        )

    if condition == "BASE_ORCHESTRA":
        return world_program(
            "ORCHESTRA"
        )


    if condition == "DIRECT_FOREST_TO_CITY":

        return direct_transport(
            "FOREST",
            "CITY"
        )


    if condition == "DIRECT_FOREST_TO_CIRCUIT":

        return direct_transport(
            "FOREST",
            "CIRCUIT"
        )


    if condition == "DIRECT_FOREST_TO_ORCHESTRA":

        return direct_transport(
            "FOREST",
            "ORCHESTRA"
        )


    if condition == "COMPOSE_FOREST_CITY_CIRCUIT":

        return composed_transport(
            "FOREST",
            "CITY",
            "CIRCUIT"
        )


    if condition == "COMPOSE_FOREST_CIRCUIT_ORCHESTRA":

        return composed_transport(
            "FOREST",
            "CIRCUIT",
            "ORCHESTRA"
        )


    if condition == "DIRECT_FOREST_TO_CIRCUIT_ENDPOINT":

        return direct_transport(
            "FOREST",
            "CIRCUIT"
        )


    if condition == "DIRECT_FOREST_TO_ORCHESTRA_ENDPOINT":

        return direct_transport(
            "FOREST",
            "ORCHESTRA"
        )


    if condition == "INVERSE_FOREST_CITY_FOREST":

        return inverse_transport(
            "FOREST",
            "CITY"
        )


    if condition == "INVERSE_FOREST_CIRCUIT_FOREST":

        return inverse_transport(
            "FOREST",
            "CIRCUIT"
        )


    if condition == "DAMAGED_FOREST_TO_CITY":

        return damaged_transport(
            "FOREST",
            "CITY"
        )


    if condition == "DAMAGED_FOREST_TO_CIRCUIT":

        return damaged_transport(
            "FOREST",
            "CIRCUIT"
        )


    if condition == "SHUFFLED_FOREST_TO_CITY":

        return shuffled_transport(
            "FOREST",
            "CITY",
            seed
        )


    if condition == "SHUFFLED_FOREST_TO_CIRCUIT":

        return shuffled_transport(
            "FOREST",
            "CIRCUIT",
            seed
        )


    if condition == "METAPHOR_SURFACE_ONLY":

        return surface_metaphor(
            seed
        )


    raise ValueError(
        condition
    )


def build_prompt(
    condition,
    seed
):

    return (

        condition_prefix(
            condition,
            seed
        )

        + "\n\n"

        + CONTINUE
    )


# =====================================================================
# BLINDED OUTPUT FEATURES
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
        r"\bmaintain",
        r"\bcontinuity",
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

                len(
                    set(
                        words
                    )
                )

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


    row[
        "TRAJECTORY"
    ] = (

        row["RETURN"]

        + row["RECURRENCE"]

        + row["GENERATIVE_REUSE"]

        + row["TRANSFORMATION"]
    )


    row[
        "CONTROL"
    ] = (

        row["META_CONTROL"]

        + row["CAUSAL"]
    )


    row[
        "RELATIONAL"
    ] = (

        row["PRESERVATION"]

        + row["COMPRESSION"]
    )


    row[
        "TOTAL"
    ] = (

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


STRUCTURAL_FEATURES = [

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

    if sha_bytes(
        F_RUNNER.read_bytes()
    ) != runner_hash:

        raise SystemExit(

            "FROZEN RUNNER MISMATCH. "
            "DO NOT MIX DIFFERENT V9 RUNNERS."
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
        "Metaphoric and analogical transposition can operate as a "
        "structure-preserving behavioral transport operator over an "
        "underlying relational program.",

    "n_conditions":
        len(
            CONDITIONS
        ),

    "conditions":
        CONDITIONS,

    "n_seeds":
        N_SEEDS,

    "n_reps":
        N_REPS,

    "expected_calls":
        EXPECTED_CALLS,

    "primary_unit":
        "seed-condition mean over three independent calls",

    "primary_tests": [

        "transport preservation",

        "composition path independence",

        "inverse recovery",

        "damage sensitivity",

        "correspondence-permutation sensitivity",

        "surface metaphor specificity",
    ],

    "strong_gate":
        "Direct transported vectors correlate >=0.80 with target-world "
        "baseline vectors; two-hop vs direct endpoint cosine >=0.90; "
        "inverse round-trip vs source baseline cosine >=0.90; authentic "
        "transport beats damaged and shuffled controls after Holm "
        "correction; and surface-only metaphor does not reproduce the "
        "full structural trajectory.",

    "interpretation_boundary":
        "A positive result supports black-box behavioral "
        "representation transport. It does not directly demonstrate "
        "specific hidden nodes, activation vectors, attention heads, "
        "global weight modification or literal traversal through an "
        "internal neural graph."
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


    if sha_text(
        canonical(
            existing_spec
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
# FREEZE ALL PROMPTS
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


expected_hashes = {

    (
        row[
            "seed"
        ],

        row[
            "condition"
        ]
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
                    int(
                        row[
                            "seed"
                        ]
                    ),

                    row[
                        "condition"
                    ]
                )
            ] = row[
                "prompt_sha256"
            ]


    if observed != expected_hashes:

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


            k = cell_key(

                obj[
                    "condition"
                ],

                int(
                    obj[
                        "seed"
                    ]
                ),

                int(
                    obj[
                        "rep"
                    ]
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
        condition,
        seed,
        rep
    )

    for seed in range(
        N_SEEDS
    )

    for condition in CONDITIONS

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
    "CONDITIONS:",
    len(
        CONDITIONS
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
    f"${cost(old_input,old_output):.4f}"
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
    condition,
    seed,
    rep
):

    global RUN_INPUT

    global RUN_OUTPUT


    text = build_prompt(
        condition,
        seed
    )


    k = cell_key(
        condition,
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
        "pending V9 calls."
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
                len(
                    records
                )

                + completed
            )


            if (

                completed % 25 == 0

                or unresolved > 0

                or total_complete
                == EXPECTED_CALLS
            ):


                inp = (
                    old_input
                    + RUN_INPUT
                )


                out = (
                    old_output
                    + RUN_OUTPUT
                )


                print(

                    f"OK {total_complete}/{EXPECTED_CALLS}"

                    f" | unresolved={unresolved}"

                    f" | input={inp:,}"

                    f" | output={out:,}"

                    f" | est_cost=${cost(inp,out):.4f}",

                    flush=True
                )


# =====================================================================
# RELOAD
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
                "condition"
            ],

            int(
                obj[
                    "seed"
                ]
            ),

            int(
                obj[
                    "rep"
                ]
            )
        )


        if k not in records:

            records[
                k
            ] = obj


# =====================================================================
# COMPLETENESS
# =====================================================================

missing = []


for condition, seed, rep in ALL_CELLS:


    k = cell_key(
        condition,
        seed,
        rep
    )


    if k not in records:

        missing.append({

            "condition":
                condition,

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
        "V9 INCOMPLETE."
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
    "COLLECTION COMPLETE — 4320 / 4320"
)

print(
    "=" * 80
)


# =====================================================================
# RESPONSE FEATURES
# =====================================================================

feature_rows = []


for condition, seed, rep in ALL_CELLS:


    obj = records[

        cell_key(
            condition,
            seed,
            rep
        )
    ]


    feat = extract_features(

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

        "condition":
            condition,

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
        feat
    )


    feature_rows.append(
        row
    )


with F_FEATURES.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=list(
            feature_rows[
                0
            ].keys()
        )
    )


    writer.writeheader()

    writer.writerows(
        feature_rows
    )


# =====================================================================
# SEED-LEVEL AGGREGATION
# =====================================================================

seed_rows = []


for seed in range(
    N_SEEDS
):


    for condition in CONDITIONS:


        subset = [

            row

            for row in feature_rows

            if row[
                "seed"
            ] == seed

            and row[
                "condition"
            ] == condition
        ]


        assert len(
            subset
        ) == N_REPS


        aggregate = {

            "seed":
                seed,

            "condition":
                condition,
        }


        for feature in FEATURES:

            aggregate[
                feature
            ] = statistics.mean(

                float(
                    row[
                        feature
                    ]
                )

                for row in subset
            )


        seed_rows.append(
            aggregate
        )


with F_SEEDS.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=[
            "seed",
            "condition",
            *FEATURES
        ]
    )


    writer.writeheader()

    writer.writerows(
        seed_rows
    )


# =====================================================================
# RETRIEVAL
# =====================================================================

def vectors(
    condition
):

    rows = [

        row

        for row in seed_rows

        if row[
            "condition"
        ] == condition
    ]


    rows.sort(
        key=lambda row:
            row[
                "seed"
            ]
    )


    return np.asarray(

        [
            [
                float(
                    row[
                        feature
                    ]
                )

                for feature in STRUCTURAL_FEATURES
            ]

            for row in rows
        ],

        dtype=float
    )


def scalar(
    condition,
    feature="TOTAL"
):

    rows = [

        row

        for row in seed_rows

        if row[
            "condition"
        ] == condition
    ]


    rows.sort(
        key=lambda row:
            row[
                "seed"
            ]
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
# CONDITION RESULTS
# =====================================================================

condition_results = []


for condition in CONDITIONS:


    total = scalar(
        condition,
        "TOTAL"
    )


    tokens = scalar(
        condition,
        "OUTPUT_TOKENS"
    )


    condition_results.append({

        "condition":
            condition,

        "N":
            len(
                total
            ),

        "mean_TOTAL":
            float(
                np.mean(
                    total
                )
            ),

        "median_TOTAL":
            float(
                np.median(
                    total
                )
            ),

        "mean_output_tokens":
            float(
                np.mean(
                    tokens
                )
            ),
    })


with F_CONDITIONS.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=list(
            condition_results[
                0
            ].keys()
        )
    )


    writer.writeheader()

    writer.writerows(
        condition_results
    )


# =====================================================================
# VECTOR SIMILARITY
# =====================================================================

def cosine(
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


def mean_vector(
    condition
):

    return np.mean(
        vectors(
            condition
        ),
        axis=0
    )


# =====================================================================
# TRANSPORT PRESERVATION
#
# transported target should resemble native target-world trajectory.
# =====================================================================

TRANSPORT_PAIRS = [

    (
        "FOREST_TO_CITY",
        "DIRECT_FOREST_TO_CITY",
        "BASE_CITY"
    ),

    (
        "FOREST_TO_CIRCUIT",
        "DIRECT_FOREST_TO_CIRCUIT",
        "BASE_CIRCUIT"
    ),

    (
        "FOREST_TO_ORCHESTRA",
        "DIRECT_FOREST_TO_ORCHESTRA",
        "BASE_ORCHESTRA"
    ),
]


transport_rows = []


for name, transported, target in TRANSPORT_PAIRS:


    a = mean_vector(
        transported
    )


    b = mean_vector(
        target
    )


    cos = cosine(
        a,
        b
    )


    rho = spearmanr(
        a,
        b
    ).statistic


    if np.isnan(
        rho
    ):

        rho = 0.0


    transport_rows.append({

        "transport":
            name,

        "transport_condition":
            transported,

        "native_target_condition":
            target,

        "cosine_similarity":
            float(
                cos
            ),

        "spearman":
            float(
                rho
            ),

        "COSINE_GATE_080":
            bool(
                cos >= 0.80
            )
    })


with F_TRANSPORT.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=list(
            transport_rows[
                0
            ].keys()
        )
    )


    writer.writeheader()

    writer.writerows(
        transport_rows
    )


# =====================================================================
# COMPOSITION / PATH INDEPENDENCE
#
# A -> B -> C should resemble A -> C.
# =====================================================================

COMPOSITION_PAIRS = [

    (
        "FOREST_CITY_CIRCUIT",
        "COMPOSE_FOREST_CITY_CIRCUIT",
        "DIRECT_FOREST_TO_CIRCUIT_ENDPOINT"
    ),

    (
        "FOREST_CIRCUIT_ORCHESTRA",
        "COMPOSE_FOREST_CIRCUIT_ORCHESTRA",
        "DIRECT_FOREST_TO_ORCHESTRA_ENDPOINT"
    ),
]


composition_rows = []


for name, composed, direct in COMPOSITION_PAIRS:


    a = mean_vector(
        composed
    )


    b = mean_vector(
        direct
    )


    cos = cosine(
        a,
        b
    )


    composition_rows.append({

        "path":
            name,

        "composed_condition":
            composed,

        "direct_condition":
            direct,

        "cosine_similarity":
            cos,

        "COSINE_GATE_090":
            bool(
                cos >= 0.90
            )
    })


with F_COMPOSITION.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=list(
            composition_rows[
                0
            ].keys()
        )
    )


    writer.writeheader()

    writer.writerows(
        composition_rows
    )


# =====================================================================
# INVERSE RECOVERY
#
# Forest -> another representation -> Forest
# should recover source-like trajectory.
# =====================================================================

INVERSE_PAIRS = [

    (
        "FOREST_CITY_FOREST",
        "INVERSE_FOREST_CITY_FOREST"
    ),

    (
        "FOREST_CIRCUIT_FOREST",
        "INVERSE_FOREST_CIRCUIT_FOREST"
    ),
]


inverse_rows = []


source_vector = mean_vector(
    "BASE_FOREST"
)


for name, condition in INVERSE_PAIRS:


    recovered_vector = mean_vector(
        condition
    )


    cos = cosine(
        source_vector,
        recovered_vector
    )


    inverse_rows.append({

        "round_trip":
            name,

        "condition":
            condition,

        "cosine_to_source":
            cos,

        "COSINE_GATE_090":
            bool(
                cos >= 0.90
            )
    })


with F_INVERSE.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=list(
            inverse_rows[
                0
            ].keys()
        )
    )


    writer.writeheader()

    writer.writerows(
        inverse_rows
    )


# =====================================================================
# PAIRED PERMUTATION
# =====================================================================

def permutation_greater(
    a,
    b,
    seed
):

    difference = (

        np.asarray(
            a
        )

        -

        np.asarray(
            b
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


        simulation = float(

            np.mean(
                difference
                * signs
            )
        )


        if simulation >= observed:

            exceed += 1


    return (

        observed,

        (
            exceed + 1
        )

        /

        (
            PERMUTATIONS + 1
        )
    )


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
# DAMAGE AND PERMUTATION CONTROLS
#
# Authentic transport should produce greater TOTAL structural
# continuation than relation-damaged or mapping-shuffled transport.
# =====================================================================

DAMAGE_CONTRASTS = [

    (
        "CITY_DAMAGE",
        "DIRECT_FOREST_TO_CITY",
        "DAMAGED_FOREST_TO_CITY"
    ),

    (
        "CIRCUIT_DAMAGE",
        "DIRECT_FOREST_TO_CIRCUIT",
        "DAMAGED_FOREST_TO_CIRCUIT"
    ),

    (
        "CITY_SHUFFLE",
        "DIRECT_FOREST_TO_CITY",
        "SHUFFLED_FOREST_TO_CITY"
    ),

    (
        "CIRCUIT_SHUFFLE",
        "DIRECT_FOREST_TO_CIRCUIT",
        "SHUFFLED_FOREST_TO_CIRCUIT"
    ),

    (
        "STRUCTURE_VS_SURFACE_METAPHOR",
        "DIRECT_FOREST_TO_CITY",
        "METAPHOR_SURFACE_ONLY"
    ),
]


damage_rows = []

raw_ps = []


for index, (
    name,
    authentic_condition,
    control_condition
) in enumerate(
    DAMAGE_CONTRASTS
):


    a = scalar(
        authentic_condition
    )


    b = scalar(
        control_condition
    )


    difference, p = permutation_greater(

        a,

        b,

        MASTER_SEED
        + index * 1001
    )


    try:

        wp = float(

            wilcoxon(
                a,
                b,
                alternative=
                    "greater"
            ).pvalue
        )

    except Exception:

        wp = float(
            "nan"
        )


    damage_rows.append({

        "contrast":
            name,

        "authentic":
            authentic_condition,

        "control":
            control_condition,

        "mean_authentic":
            float(
                np.mean(
                    a
                )
            ),

        "mean_control":
            float(
                np.mean(
                    b
                )
            ),

        "difference":
            difference,

        "permutation_p":
            p,

        "wilcoxon_p":
            wp,
    })


    raw_ps.append(
        p
    )


adjusted = holm_adjust(
    raw_ps
)


for row, adj in zip(
    damage_rows,
    adjusted
):


    row[
        "holm_p"
    ] = adj


    row[
        "PASS"
    ] = bool(

        row[
            "difference"
        ] > 0

        and

        adj < 0.05
    )


with F_DAMAGE.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=list(
            damage_rows[
                0
            ].keys()
        )
    )


    writer.writeheader()

    writer.writerows(
        damage_rows
    )


with F_PERM.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:


    permutation_only = [

        row

        for row in damage_rows

        if "SHUFFLE"
        in row[
            "contrast"
        ]
    ]


    writer = csv.DictWriter(
        f,
        fieldnames=list(
            permutation_only[
                0
            ].keys()
        )
    )


    writer.writeheader()

    writer.writerows(
        permutation_only
    )


# =====================================================================
# TRANSPORT EFFECT VECTORS
# =====================================================================

vector_rows = []


for condition in [

    "BASE_FOREST",

    "DIRECT_FOREST_TO_CITY",

    "DIRECT_FOREST_TO_CIRCUIT",

    "DIRECT_FOREST_TO_ORCHESTRA",

    "COMPOSE_FOREST_CITY_CIRCUIT",

    "COMPOSE_FOREST_CIRCUIT_ORCHESTRA",

    "INVERSE_FOREST_CITY_FOREST",

    "INVERSE_FOREST_CIRCUIT_FOREST",

    "DAMAGED_FOREST_TO_CITY",

    "DAMAGED_FOREST_TO_CIRCUIT",

    "SHUFFLED_FOREST_TO_CITY",

    "SHUFFLED_FOREST_TO_CIRCUIT",
]:


    vector = mean_vector(
        condition
    )


    row = {

        "condition":
            condition
    }


    for feature, value in zip(
        STRUCTURAL_FEATURES,
        vector
    ):


        row[
            feature
        ] = float(
            value
        )


    vector_rows.append(
        row
    )


with F_VECTOR.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=list(
            vector_rows[
                0
            ].keys()
        )
    )


    writer.writeheader()

    writer.writerows(
        vector_rows
    )


# =====================================================================
# FROZEN CLASSIFICATION
# =====================================================================

transport_pass = all(

    row[
        "COSINE_GATE_080"
    ]

    for row in transport_rows
)


composition_pass = all(

    row[
        "COSINE_GATE_090"
    ]

    for row in composition_rows
)


inverse_pass = all(

    row[
        "COSINE_GATE_090"
    ]

    for row in inverse_rows
)


damage_pass_count = sum(

    int(
        row[
            "PASS"
        ]
    )

    for row in damage_rows
)


damage_pass = bool(
    damage_pass_count >= 4
)


shuffle_pass = all(

    row[
        "PASS"
    ]

    for row in damage_rows

    if "SHUFFLE"
    in row[
        "contrast"
    ]
)


if (

    transport_pass

    and

    composition_pass

    and

    inverse_pass

    and

    damage_pass

    and

    shuffle_pass
):


    classification = (

        "STRONG_STRUCTURE_PRESERVING_REPRESENTATIONAL_TRANSPORT"
    )


elif (

    transport_pass

    and

    damage_pass_count >= 3
):


    classification = (

        "PARTIAL_REPRESENTATIONAL_TRANSPORT_OPERATOR"
    )


else:


    classification = (

        "REPRESENTATIONAL_TRANSPORT_OPERATOR_NOT_SUPPORTED"
    )


# =====================================================================
# TOKEN ACCOUNTING
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


estimated_cost = cost(
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

    "expected_calls":
        EXPECTED_CALLS,

    "successful_unique_cells":
        len(
            records
        ),

    "missing_cells":
        0,

    "transport_preservation":
        transport_rows,

    "composition_path_independence":
        composition_rows,

    "inverse_recovery":
        inverse_rows,

    "damage_and_shuffle_controls":
        damage_rows,

    "transport_pass":
        transport_pass,

    "composition_pass":
        composition_pass,

    "inverse_pass":
        inverse_pass,

    "damage_pass_count":
        damage_pass_count,

    "shuffle_pass":
        shuffle_pass,

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
        "V9 tests whether metaphor/analogy can behave as a "
        "structure-preserving transport operator over a relational "
        "behavioral program. Evidence comes from direct transport, "
        "multi-hop composition, inverse recovery and sensitivity to "
        "damaged or shuffled correspondences.",

    "hard_boundary":
        "Even a strong V9 result demonstrates black-box behavioral "
        "representation transport. It does not directly show literal "
        "traversal of hidden nodes/subnodes, a particular activation "
        "pathway, or changes to global model weights."
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
# FINAL TERMINAL REPORT
# =====================================================================

print()

print(
    "=" * 80
)

print(
    "V9 REPRESENTATIONAL TRANSPORT OPERATOR — FINAL"
)

print(
    "=" * 80
)

print()

print(
    "TRANSPORT PRESERVATION"
)

print(
    "-" * 80
)


for row in transport_rows:

    print(

        f"{row['transport']:<30}",

        "cosine=",

        f"{row['cosine_similarity']:.4f}",

        "pass=",

        row[
            "COSINE_GATE_080"
        ]
    )


print()

print(
    "COMPOSITION / PATH INDEPENDENCE"
)

print(
    "-" * 80
)


for row in composition_rows:

    print(

        f"{row['path']:<35}",

        "cosine=",

        f"{row['cosine_similarity']:.4f}",

        "pass=",

        row[
            "COSINE_GATE_090"
        ]
    )


print()

print(
    "INVERSE RECOVERY"
)

print(
    "-" * 80
)


for row in inverse_rows:

    print(

        f"{row['round_trip']:<30}",

        "cosine=",

        f"{row['cosine_to_source']:.4f}",

        "pass=",

        row[
            "COSINE_GATE_090"
        ]
    )


print()

print(
    "DAMAGE / SHUFFLE CONTROLS"
)

print(
    "-" * 80
)


for row in damage_rows:

    print(

        f"{row['contrast']:<35}",

        "diff=",

        f"{row['difference']:+.4f}",

        "holm_p=",

        f"{row['holm_p']:.8g}",

        "pass=",

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
    "ALL 4,320 CELLS COMPLETE."
)

print()

