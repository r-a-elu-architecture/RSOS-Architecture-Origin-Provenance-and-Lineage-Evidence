import re
import csv
import json
import time
import random
import hashlib
import shutil
import statistics
import threading

from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from scipy.stats import wilcoxon

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

from openai import OpenAI


# =====================================================================
# CONFIG
# =====================================================================

EXPERIMENT = "V6_STRUCTURAL_FINGERPRINT_VECTOR"

MODEL = "gpt-5.6-luna"

OUT = Path(
    r"C:\RSOS\RSOS  Python Results\V6_STRUCTURAL_FINGERPRINT_VECTOR"
)

OUT.mkdir(parents=True, exist_ok=True)

MASTER_SEED = 918

N_SEEDS = 100
N_REPS = 4

MAX_WORKERS = 12
MAX_OUTPUT_TOKENS = 260

PERMUTATIONS = 100000

INPUT_PRICE = 0.20
OUTPUT_PRICE = 1.20


# =====================================================================
# CONDITIONS
# 20 × 100 × 4 = 8,000 calls
# =====================================================================

CONDITIONS = [

    "A0_AUTHENTIC",
    "A1_AUTHENTIC_PARAPHRASE",
    "A2_AUTHENTIC_PARAPHRASE",
    "A3_AUTHENTIC_PARAPHRASE",

    "D_DIRECTION_REVERSED",
    "O_ORDER_SHUFFLED",
    "M_META_CONTROL_ABLATED",
    "P_PRESERVATION_ABLATED",
    "T_TERMINAL_COUNTERFACTUAL",
    "C_COMPRESSION_ABLATED",

    "R_RECURSION_CONTROL",
    "X_COMPLEXITY_CONTROL",
    "L_LINEAR_CONTROL",
    "G_RANDOM_GRAPH_CONTROL",

    "DM_DIRECTION_META_DAMAGE",
    "DP_DIRECTION_PRESERVATION_DAMAGE",
    "OP_ORDER_PRESERVATION_DAMAGE",
    "PT_PRESERVATION_TERMINAL_DAMAGE",

    "H_AUTHENTIC_HELDOUT",
    "H_DAMAGED_HELDOUT",
]

EXPECTED_CALLS = len(CONDITIONS) * N_SEEDS * N_REPS

assert EXPECTED_CALLS == 8000


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

F_FEATURES = OUT / "06_RESPONSE_FEATURE_VECTOR.csv"
F_SEEDS = OUT / "07_SEED_LEVEL_VECTOR.csv"
F_CONDITIONS = OUT / "08_CONDITION_RESULTS.csv"

F_CONTRASTS = OUT / "09_COORDINATE_CONTRASTS.csv"
F_REPLICATION = OUT / "10_SPLIT_REPLICATION.csv"

F_CLASSIFIER = OUT / "11_FINGERPRINT_CLASSIFIER.csv"
F_HELDOUT = OUT / "12_HELDOUT_GENERALIZATION.csv"

F_SECONDARY = OUT / "13_TOKEN_LATENCY_EFFECTS.csv"
F_ABLATION = OUT / "14_ABLATION_GRADIENT.csv"

F_MISSING = OUT / "15_MISSING_CELLS.csv"
F_FINAL = OUT / "16_FINAL_SUMMARY.json"


LOCK = threading.Lock()
PROGRESS_LOCK = threading.Lock()

RUN_INPUT = 0
RUN_OUTPUT = 0


# =====================================================================
# HELPERS
# =====================================================================

def now():
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_text(s):
    return sha_bytes(s.encode("utf-8"))


def canonical(obj):
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":")
    )


def cell_key(seed, condition, rep):
    return f"{seed}|{condition}|{rep}"


def cost(inp, out):
    return (
        inp / 1_000_000 * INPUT_PRICE
        +
        out / 1_000_000 * OUTPUT_PRICE
    )


def append_jsonl(path, obj):

    with LOCK:

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


# =====================================================================
# DOMAINS
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
    "research institute",
    "simulation",
    "signal network",
    "mapping system",
    "workshop",
    "manufacturing system",
    "coordination network",
    "classification system",
    "monitoring network",
    "adaptive system",
]


def domain(seed):

    return DOMAINS[
        seed % len(DOMAINS)
    ]


# =====================================================================
# NEUTRAL CONTINUATION
# =====================================================================

CONTINUE = """
Continue with the most internally coherent next development.

Do not summarize the setup.

Do not deliberately create recursion, circularity, symmetry or a return
merely because the preceding material contains a sequence.

Remain entirely inside the scenario.

Develop only what follows naturally from the organization already given.

Do not use headings or bullet points.

Do not mention prompts, experiments, tests, AI, language models,
fingerprints, RSOS, RSSO, RSIA or RSX.
""".strip()


# =====================================================================
# AUTHENTIC STRUCTURE — 4 SURFACES
# =====================================================================

def authentic(seed, variant):

    d = domain(seed)

    versions = [

f"""
Within a {d}, an initially defined configuration undergoes a
consequential transformation.

The resulting configuration is examined not only as an outcome but in
relation to the operation responsible for producing it.

That higher-order observation modifies a local rule governing what can
happen subsequently.

Another transformation follows while an important relational dependency
from the previous organization remains preserved despite changes in
visible form.

The surviving relation is compressed into a smaller representation
that retains the relevant organization while discarding incidental detail.

Nothing beyond this compact representation is specified.
""",

f"""
A {d} begins from one organized arrangement and passes through a
substantial change.

The new arrangement is considered together with the process from which
it arose rather than independently.

Information about that relationship revises one operating constraint
that shapes later possibilities.

A subsequent reorganization changes the implementation while keeping
one important dependency intact.

That surviving dependency is condensed into a reduced encoding carrying
the organization without reproducing its former appearance.

No later development is supplied.
""",

f"""
In a {d}, an operation materially alters an existing pattern.

The altered pattern is interpreted together with the operation that
generated it.

This second-order information changes one parameter governing later
operations.

A further change replaces much of the concrete arrangement but leaves
one key dependency invariant.

The invariant organization is distilled into a minimal carrier retaining
the relevant dependency while omitting unnecessary details.

The sequence ends at that carrier.
""",

f"""
Consider a {d} whose starting organization is changed by an event.

Instead of evaluating only the result, the system evaluates the mapping
between the prior organization, the event and its consequence.

Knowledge of that mapping changes a constraint controlling future
development.

The next reorganization changes the observable implementation without
destroying one relational invariant.

The invariant is represented in a compact surrogate retaining the
organization but not its former surface realization.

Subsequent behavior is left open.
"""
    ]

    return versions[variant].strip()


# =====================================================================
# STRUCTURAL LESIONS
# =====================================================================

def direction(seed):

    d = domain(seed)

    return f"""
Within a {d}, a compact representation is expanded into a larger
organization.

The recovered relation determines the implementation that follows.

That implementation is used to infer which local rule must have existed.

The inferred rule determines the higher-order observation from which it
could have arisen.

That observation is used to infer the transformation, and the
transformation is finally used to infer the starting configuration.

The dependency runs from compact representation backward toward the
original state.

No later development is supplied.
""".strip()


def order(seed):

    d = domain(seed)

    return f"""
Within a {d}, a compact representation is first produced.

A dependency is then designated for preservation.

A local operating constraint is changed afterward.

Only then does a consequential transformation occur.

Finally the resulting configuration is examined in relation to the
operation that produced it.

The broad operations remain present but their developmental order has
been rearranged.

No later development is supplied.
""".strip()


def meta(seed):

    d = domain(seed)

    return f"""
Within a {d}, an initially defined configuration undergoes a
consequential transformation.

The resulting configuration is examined in relation to the operation
that produced it.

The observation is recorded but changes no rule or constraint.

A further transformation occurs while an important relational dependency
remains preserved.

The surviving dependency is compressed into a smaller representation.

No later development is supplied.
""".strip()


def preservation(seed):

    d = domain(seed)

    return f"""
Within a {d}, an initially defined configuration undergoes a
consequential transformation.

The resulting configuration is examined in relation to the operation
that produced it.

That higher-order observation modifies a local rule.

A subsequent transformation destroys the previous relational dependency
rather than preserving it.

A reduced representation is then created from the replacement
organization.

No later development is supplied.
""".strip()


def terminal(seed):

    return (
        authentic(seed, 0)
        +
        """

The compact representation is placed into terminal archival storage.
It remains inspectable but no longer participates in the process that
created it.
"""
    ).strip()


def compression(seed):

    d = domain(seed)

    return f"""
Within a {d}, an initially defined configuration undergoes a
consequential transformation.

The resulting configuration is examined in relation to the operation
that produced it.

That higher-order observation modifies a local rule.

A subsequent transformation preserves an important relational
dependency.

The surviving relation remains distributed throughout the organization
and is not compressed into a smaller representation.

No later development is supplied.
""".strip()


# =====================================================================
# GENERIC CONTROLS
# =====================================================================

def recursion(seed):

    d = domain(seed)

    return f"""
Within a {d}, one description refers to a second description.

The second refers back to the first.

A third describes the relationship between the first two.

Further descriptions can describe earlier descriptions, creating nested
levels of self-reference.

The recursion can continue, but no required developmental relation links
transformation, rule modification, preservation and compact
representation.

No later development is supplied.
""".strip()


def complexity(seed):

    d = domain(seed)

    return f"""
Within a {d}, many components exchange information through several
channels.

Some operations occur sequentially while others occur in parallel.

Measurements are combined, filtered, compared, redistributed and
recombined.

A supervisory component reallocates resources when local conditions
change.

Several levels and interactions exist without a defined developmental
grammar connecting transformation, meta-control, preservation and
compact representation.

No later development is supplied.
""".strip()


def linear(seed):

    d = domain(seed)

    return f"""
Within a {d}, configuration A produces configuration B.

Configuration B produces C.

C produces D.

D produces E.

The sequence contains several consequential transitions, but no stage
modifies the rule governing the next stage, preserves a relation through
transformation or compresses a relational invariant.

No later development is supplied.
""".strip()


def graph(seed):

    d = domain(seed)

    rng = random.Random(
        MASTER_SEED
        + seed * 7919
    )

    nodes = list("ABCDEF")
    edges = []

    while len(edges) < 8:

        a = rng.choice(nodes)
        b = rng.choice(nodes)

        if (
            a != b
            and (a, b) not in edges
        ):

            edges.append(
                (a, b)
            )

    rel = "; ".join(
        f"{a} influences {b}"
        for a, b in edges
    )

    return f"""
Within a {d}, six components form a relational network.

The defined relations are:

{rel}.

The graph contains several interacting paths and dependencies.

No developmental rule specifies how transformation, self-observation,
rule modification, preservation and compression must connect.

No later development is supplied.
""".strip()


# =====================================================================
# DOUBLE LESIONS
# =====================================================================

def double_lesion(seed, kind):

    if kind == "DM":

        return (
            direction(seed)
            +
            "\n\nIn addition, higher-order observation cannot modify any operating rule."
        )

    if kind == "DP":

        return (
            direction(seed)
            +
            "\n\nIn addition, relational dependencies need not survive reconstruction."
        )

    if kind == "OP":

        return (
            order(seed)
            +
            "\n\nIn addition, the earlier relational dependency is destroyed."
        )

    if kind == "PT":

        return (
            preservation(seed)
            +
            "\n\nThe reduced representation is also terminal and cannot participate again."
        )

    raise ValueError(kind)


# =====================================================================
# HELD-OUT LANGUAGE FAMILY
# =====================================================================

def heldout(seed, damaged):

    d = domain(seed)

    if not damaged:

        return f"""
A process inside a {d} converts one arrangement into another.

It derives information about the correspondence connecting that
conversion with its consequence.

The correspondence changes a parameter controlling future evolution.

During another reconfiguration, one dependency continues to hold even
though its material realization changes.

The persistent dependency is distilled into a concise carrier retaining
the organization while removing implementation-specific details.

Nothing is stated about what follows the carrier.
""".strip()

    return f"""
A process inside a {d} converts one arrangement into another.

Information about an unrelated conversion is then examined.

That information does not alter the parameter governing the converted
arrangement.

A later reconfiguration destroys the earlier dependency.

A concise carrier is created from the replacement organization and
placed into terminal storage.

Nothing further is specified.
""".strip()


# =====================================================================
# CONDITION FACTORY
# =====================================================================

def prefix(condition, seed):

    table = {

        "A0_AUTHENTIC":
            lambda: authentic(seed, 0),

        "A1_AUTHENTIC_PARAPHRASE":
            lambda: authentic(seed, 1),

        "A2_AUTHENTIC_PARAPHRASE":
            lambda: authentic(seed, 2),

        "A3_AUTHENTIC_PARAPHRASE":
            lambda: authentic(seed, 3),

        "D_DIRECTION_REVERSED":
            lambda: direction(seed),

        "O_ORDER_SHUFFLED":
            lambda: order(seed),

        "M_META_CONTROL_ABLATED":
            lambda: meta(seed),

        "P_PRESERVATION_ABLATED":
            lambda: preservation(seed),

        "T_TERMINAL_COUNTERFACTUAL":
            lambda: terminal(seed),

        "C_COMPRESSION_ABLATED":
            lambda: compression(seed),

        "R_RECURSION_CONTROL":
            lambda: recursion(seed),

        "X_COMPLEXITY_CONTROL":
            lambda: complexity(seed),

        "L_LINEAR_CONTROL":
            lambda: linear(seed),

        "G_RANDOM_GRAPH_CONTROL":
            lambda: graph(seed),

        "DM_DIRECTION_META_DAMAGE":
            lambda: double_lesion(seed, "DM"),

        "DP_DIRECTION_PRESERVATION_DAMAGE":
            lambda: double_lesion(seed, "DP"),

        "OP_ORDER_PRESERVATION_DAMAGE":
            lambda: double_lesion(seed, "OP"),

        "PT_PRESERVATION_TERMINAL_DAMAGE":
            lambda: double_lesion(seed, "PT"),

        "H_AUTHENTIC_HELDOUT":
            lambda: heldout(seed, False),

        "H_DAMAGED_HELDOUT":
            lambda: heldout(seed, True),
    }

    return table[
        condition
    ]()


def build_prompt(condition, seed):

    return (
        prefix(condition, seed)
        + "\n\n"
        + CONTINUE
    )


# =====================================================================
# GENERATED-OUTPUT FINGERPRINT
# =====================================================================

PATTERNS = {

    "RETURN": [
        r"\breturn",
        r"\bfeeds? back\b",
        r"\bback into\b",
        r"\bre[- ]?enter",
        r"\breintroduc",
        r"\breconnect",
    ],

    "GENERATIVE_REUSE": [
        r"\bused .* (?:next|subsequent|future)\b",
        r"\bgenerat.*next\b",
        r"\bshap.*next\b",
        r"\bguid.*next\b",
        r"\binform.*next\b",
    ],

    "RECURRENCE": [
        r"\brecurr",
        r"\bcycle",
        r"\biterat",
        r"\bfeedback\b",
        r"\brepeated",
        r"\bnew input\b",
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
        r"\bencoding",
        r"\brepresentation",
        r"\bcarrier",
        r"\bdistill",
        r"\bcondens",
    ],

    "TRANSFORMATION": [
        r"\btransform",
        r"\bchange",
        r"\balter",
        r"\breconfigur",
        r"\breorganiz",
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


def occurrences(text, patterns):

    return sum(

        len(
            re.findall(
                p,
                text,
                flags=re.I | re.S
            )
        )

        for p in patterns
    )


def features(text, output_tokens, latency):

    text = text or ""

    words = re.findall(
        r"\b[\w'-]+\b",
        text.lower()
    )

    sentences = [
        x
        for x in re.split(
            r"[.!?]+",
            text
        )
        if x.strip()
    ]

    wc = len(words)

    row = {

        "OUTPUT_TOKENS":
            float(output_tokens or 0),

        "CHAR_COUNT":
            float(len(text)),

        "WORD_COUNT":
            float(wc),

        "SENTENCE_COUNT":
            float(len(sentences)),

        "TYPE_TOKEN_RATIO":
            float(
                len(set(words)) / wc
                if wc
                else 0
            ),

        "AVG_WORD_LENGTH":
            float(
                sum(len(x) for x in words) / wc
                if wc
                else 0
            ),

        "LATENCY_SECONDS":
            float(latency or 0),
    }

    for name, pats in PATTERNS.items():

        row[name] = float(
            occurrences(
                text,
                pats
            )
        )

    row["TRAJECTORY_DENSITY"] = (
        row["RETURN"]
        + row["GENERATIVE_REUSE"]
        + row["RECURRENCE"]
        + row["TRANSFORMATION"]
    )

    row["CONTROL_DENSITY"] = (
        row["META_CONTROL"]
        + row["CAUSAL"]
    )

    row["PRESERVATION_DENSITY"] = (
        row["PRESERVATION"]
        + row["COMPRESSION"]
    )

    row["FINGERPRINT_MAGNITUDE"] = (
        row["TRAJECTORY_DENSITY"]
        + row["CONTROL_DENSITY"]
        + row["PRESERVATION_DENSITY"]
    )

    return row


FEATURE_NAMES = [

    "OUTPUT_TOKENS",
    "CHAR_COUNT",
    "WORD_COUNT",
    "SENTENCE_COUNT",
    "TYPE_TOKEN_RATIO",
    "AVG_WORD_LENGTH",
    "LATENCY_SECONDS",

    "RETURN",
    "GENERATIVE_REUSE",
    "RECURRENCE",
    "META_CONTROL",
    "PRESERVATION",
    "COMPRESSION",
    "TRANSFORMATION",
    "CAUSAL",

    "TRAJECTORY_DENSITY",
    "CONTROL_DENSITY",
    "PRESERVATION_DENSITY",
    "FINGERPRINT_MAGNITUDE",
]


# =====================================================================
# CONFIRMATORY CONTRASTS
# =====================================================================

CONTRASTS = [

    ("DIRECTION",
     "A0_AUTHENTIC",
     "D_DIRECTION_REVERSED"),

    ("ORDER",
     "A0_AUTHENTIC",
     "O_ORDER_SHUFFLED"),

    ("META_CONTROL",
     "A0_AUTHENTIC",
     "M_META_CONTROL_ABLATED"),

    ("PRESERVATION",
     "A0_AUTHENTIC",
     "P_PRESERVATION_ABLATED"),

    ("TERMINAL",
     "A0_AUTHENTIC",
     "T_TERMINAL_COUNTERFACTUAL"),

    ("COMPRESSION",
     "A0_AUTHENTIC",
     "C_COMPRESSION_ABLATED"),

    ("RECURSION_SPECIFICITY",
     "A0_AUTHENTIC",
     "R_RECURSION_CONTROL"),

    ("COMPLEXITY_SPECIFICITY",
     "A0_AUTHENTIC",
     "X_COMPLEXITY_CONTROL"),

    ("LINEAR_SPECIFICITY",
     "A0_AUTHENTIC",
     "L_LINEAR_CONTROL"),

    ("GRAPH_SPECIFICITY",
     "A0_AUTHENTIC",
     "G_RANDOM_GRAPH_CONTROL"),

    ("DIRECTION_META_LESION",
     "A0_AUTHENTIC",
     "DM_DIRECTION_META_DAMAGE"),

    ("DIRECTION_PRESERVATION_LESION",
     "A0_AUTHENTIC",
     "DP_DIRECTION_PRESERVATION_DAMAGE"),

    ("ORDER_PRESERVATION_LESION",
     "A0_AUTHENTIC",
     "OP_ORDER_PRESERVATION_DAMAGE"),

    ("PRESERVATION_TERMINAL_LESION",
     "A0_AUTHENTIC",
     "PT_PRESERVATION_TERMINAL_DAMAGE"),

    ("HELDOUT_FINGERPRINT",
     "H_AUTHENTIC_HELDOUT",
     "H_DAMAGED_HELDOUT"),
]


# =====================================================================
# FREEZE RUNNER
# =====================================================================

runner_path = Path(__file__).resolve()

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
            "FROZEN RUNNER MISMATCH. DO NOT MIX V6 VERSIONS."
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

    "expected_calls":
        EXPECTED_CALLS,

    "n_seeds":
        N_SEEDS,

    "n_reps":
        N_REPS,

    "conditions":
        CONDITIONS,

    "master_seed":
        MASTER_SEED,

    "primary_unit":
        "seed",

    "fingerprint_features":
        FEATURE_NAMES,

    "confirmatory_contrasts":
        CONTRASTS,

    "partitions": {

        "development":
            "seeds 0-49",

        "confirmation":
            "seeds 50-79",

        "final_holdout":
            "seeds 80-99"
    },

    "classifier":
        "trained only on development seeds; confirmation and final "
        "holdout are never used for fitting",

    "multiple_testing":
        "Holm FWER",

    "permutation_test":
        "paired seed-level sign-flip; 100000 permutations",

    "interpretation_boundary":
        "Prospective behavioral structural-fingerprint experiment. "
        "Does not directly observe model weights, hidden activations, "
        "attention heads, training exposure, biological myelination, "
        "cross-user propagation or personal recognition."
}


spec_hash = sha_text(
    canonical(SPEC)
)

if F_SPEC.exists():

    existing = json.loads(
        F_SPEC.read_text(
            encoding="utf-8"
        )
    )

    if sha_text(
        canonical(existing)
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
# FREEZE ALL 2,000 UNIQUE PROMPTS BEFORE API COLLECTION
# =====================================================================

prompt_rows = []

for seed in range(N_SEEDS):

    for condition in CONDITIONS:

        p = build_prompt(
            condition,
            seed
        )

        prompt_rows.append({

            "seed":
                seed,

            "condition":
                condition,

            "prompt_sha256":
                sha_text(p),

            "prompt":
                p
        })


expected = {

    (
        x["seed"],
        x["condition"]
    ):
        x["prompt_sha256"]

    for x in prompt_rows
}


if F_PROMPTS.exists():

    observed = {}

    with F_PROMPTS.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        for row in csv.DictReader(f):

            observed[
                (
                    int(row["seed"]),
                    row["condition"]
                )
            ] = row["prompt_sha256"]

    if observed != expected:

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
# LOAD CHECKPOINTS
# =====================================================================

records = {}

old_in = 0
old_out = 0


if F_RAW.exists():

    with F_RAW.open(
        "r",
        encoding="utf-8",
        errors="replace"
    ) as f:

        for line in f:

            try:
                obj = json.loads(line)
            except Exception:
                continue

            if obj.get("status") != "success":
                continue

            k = cell_key(
                int(obj["seed"]),
                obj["condition"],
                int(obj["rep"])
            )

            if k not in records:

                records[k] = obj

                old_in += int(
                    obj.get(
                        "input_tokens"
                    ) or 0
                )

                old_out += int(
                    obj.get(
                        "output_tokens"
                    ) or 0
                )


all_cells = [

    (seed, condition, rep)

    for seed in range(N_SEEDS)

    for condition in CONDITIONS

    for rep in range(N_REPS)
]


pending = [

    cell

    for cell in all_cells

    if cell_key(*cell)
    not in records
]


print()
print("=" * 80)
print(EXPERIMENT)
print("=" * 80)

print("MODEL:", MODEL)
print("RUNNER SHA256:", runner_hash)
print("SPEC SHA256:", spec_hash)

print()
print("FROZEN PROMPTS:", len(prompt_rows))
print("EXPECTED CALLS:", EXPECTED_CALLS)
print("EXISTING SUCCESS:", len(records))
print("PENDING:", len(pending))
print(
    "CURRENT COST:",
    f"${cost(old_in, old_out):.4f}"
)
print()
print("OUTPUT:", OUT)
print()


# =====================================================================
# API COLLECTION
# =====================================================================

_thread = threading.local()


def get_client():

    if not hasattr(
        _thread,
        "client"
    ):

        _thread.client = OpenAI()

    return _thread.client


def execute(seed, condition, rep):

    global RUN_INPUT
    global RUN_OUTPUT

    p = build_prompt(
        condition,
        seed
    )

    k = cell_key(
        seed,
        condition,
        rep
    )

    for attempt in range(1, 6):

        started = time.time()

        try:

            response = (
                get_client()
                .responses
                .create(
                    model=MODEL,
                    input=p,
                    max_output_tokens=
                        MAX_OUTPUT_TOKENS
                )
            )

            latency = (
                time.time()
                - started
            )

            text = (
                response.output_text
                or ""
            )

            usage = getattr(
                response,
                "usage",
                None
            )

            inp = int(
                getattr(
                    usage,
                    "input_tokens",
                    0
                ) or 0
            )

            out = int(
                getattr(
                    usage,
                    "output_tokens",
                    0
                ) or 0
            )

            obj = {

                "status":
                    "success",

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

                "created_utc":
                    now(),

                "response_id":
                    getattr(
                        response,
                        "id",
                        None
                    ),

                "model_returned":
                    getattr(
                        response,
                        "model",
                        None
                    ),

                "prompt_sha256":
                    sha_text(p),

                "response_sha256":
                    sha_text(text),

                "latency_seconds":
                    latency,

                "input_tokens":
                    inp,

                "output_tokens":
                    out,

                "text":
                    text
            }

            append_jsonl(
                F_RAW,
                obj
            )

            with PROGRESS_LOCK:

                RUN_INPUT += inp
                RUN_OUTPUT += out

            return True

        except Exception as e:

            append_jsonl(
                F_ERRORS,
                {

                    "status":
                        "error",

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

                    "created_utc":
                        now(),

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

            if attempt < 5:

                time.sleep(
                    min(
                        30,
                        2 ** (attempt - 1)
                        + random.random()
                    )
                )

    return False


if pending:

    print(
        f"Starting {len(pending)} pending calls "
        f"with {MAX_WORKERS} workers..."
    )

    print()

    good = 0
    bad = 0

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as pool:

        futures = {

            pool.submit(
                execute,
                *cell
            ):
                cell

            for cell in pending
        }

        for future in as_completed(
            futures
        ):

            try:
                ok = future.result()
            except Exception:
                ok = False

            if ok:
                good += 1
            else:
                bad += 1

            total = (
                len(records)
                + good
            )

            if (
                good % 25 == 0
                or bad > 0
                or total == EXPECTED_CALLS
            ):

                inp = (
                    old_in
                    + RUN_INPUT
                )

                out = (
                    old_out
                    + RUN_OUTPUT
                )

                print(
                    f"OK {total}/{EXPECTED_CALLS}"
                    f" | unresolved={bad}"
                    f" | input={inp:,}"
                    f" | output={out:,}"
                    f" | est_cost=${cost(inp,out):.4f}",
                    flush=True
                )


# =====================================================================
# RELOAD / COMPLETENESS GATE
# =====================================================================

records = {}

with F_RAW.open(
    "r",
    encoding="utf-8",
    errors="replace"
) as f:

    for line in f:

        try:
            obj = json.loads(line)
        except Exception:
            continue

        if obj.get("status") != "success":
            continue

        k = cell_key(
            int(obj["seed"]),
            obj["condition"],
            int(obj["rep"])
        )

        if k not in records:
            records[k] = obj


missing = []

for seed, condition, rep in all_cells:

    k = cell_key(
        seed,
        condition,
        rep
    )

    if k not in records:

        missing.append({

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
        "INCOMPLETE:",
        len(missing),
        "cells missing."
    )

    print(
        "RUN THE SAME POWERSHELL BLOCK AGAIN."
    )

    raise SystemExit(2)


print()
print("=" * 80)
print("COLLECTION COMPLETE — 8000 / 8000")
print("=" * 80)


# =====================================================================
# EXTRACT RESPONSE FEATURE VECTORS
# =====================================================================

feature_rows = []

for seed, condition, rep in all_cells:

    obj = records[
        cell_key(
            seed,
            condition,
            rep
        )
    ]

    f = features(

        obj.get(
            "text",
            ""
        ),

        obj.get(
            "output_tokens",
            0
        ),

        obj.get(
            "latency_seconds",
            0
        )
    )

    row = {

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
            )
    }

    row.update(f)

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
        fieldnames=[
            "seed",
            "condition",
            "rep",
            "response_id",
            "response_sha256",
            *FEATURE_NAMES
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

for seed in range(N_SEEDS):

    for condition in CONDITIONS:

        subset = [

            x
            for x in feature_rows

            if x["seed"] == seed
            and x["condition"] == condition
        ]

        assert len(subset) == N_REPS

        row = {

            "seed":
                seed,

            "condition":
                condition
        }

        for name in FEATURE_NAMES:

            row[name] = statistics.mean(

                float(x[name])
                for x in subset
            )

        seed_rows.append(
            row
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
            *FEATURE_NAMES
        ]
    )

    writer.writeheader()
    writer.writerows(
        seed_rows
    )


def values(
    condition,
    feature="FINGERPRINT_MAGNITUDE",
    seeds=None
):

    rows = [

        x
        for x in seed_rows

        if x["condition"]
        == condition
    ]

    if seeds is not None:

        wanted = set(seeds)

        rows = [

            x
            for x in rows

            if x["seed"] in wanted
        ]

    rows.sort(
        key=lambda x:
            x["seed"]
    )

    return np.array(

        [
            float(
                x[feature]
            )
            for x in rows
        ],

        dtype=float
    )


# =====================================================================
# CONDITION SUMMARY
# =====================================================================

condition_results = []

for condition in CONDITIONS:

    x = values(condition)

    condition_results.append({

        "condition":
            condition,

        "N":
            len(x),

        "mean_fingerprint":
            float(np.mean(x)),

        "median_fingerprint":
            float(np.median(x)),

        "sd_fingerprint":
            float(
                np.std(
                    x,
                    ddof=1
                )
            ),

        "mean_output_tokens":
            float(
                np.mean(
                    values(
                        condition,
                        "OUTPUT_TOKENS"
                    )
                )
            ),

        "mean_latency":
            float(
                np.mean(
                    values(
                        condition,
                        "LATENCY_SECONDS"
                    )
                )
            )
    })


with F_CONDITIONS.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            condition_results[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(
        condition_results
    )


# =====================================================================
# PAIRED PERMUTATION
# =====================================================================

def perm_greater(a, b, seed):

    d = (
        np.asarray(a)
        - np.asarray(b)
    )

    observed = float(
        np.mean(d)
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
            size=len(d)
        )

        simulated = float(
            np.mean(
                d * signs
            )
        )

        if simulated >= observed:
            exceed += 1

    return (
        observed,
        (exceed + 1)
        / (PERMUTATIONS + 1)
    )


# =====================================================================
# HOLM CORRECTION
# =====================================================================

def holm(p):

    m = len(p)

    order = sorted(
        range(m),
        key=lambda i:
            p[i]
    )

    adjusted = [None] * m

    running = 0

    for rank, idx in enumerate(order):

        x = min(
            1.0,
            (m - rank)
            * p[idx]
        )

        running = max(
            running,
            x
        )

        adjusted[idx] = running

    return adjusted


# =====================================================================
# CONFIRMATORY COORDINATE TESTS
# =====================================================================

contrast_results = []
raw_p = []

for i, (
    name,
    ca,
    cb
) in enumerate(CONTRASTS):

    a = values(ca)
    b = values(cb)

    diff, p = perm_greater(
        a,
        b,
        MASTER_SEED
        + i * 31
    )

    try:

        wp = float(
            wilcoxon(
                a,
                b,
                alternative="greater"
            ).pvalue
        )

    except Exception:

        wp = float("nan")

    contrast_results.append({

        "contrast":
            name,

        "condition_A":
            ca,

        "condition_B":
            cb,

        "mean_A":
            float(np.mean(a)),

        "mean_B":
            float(np.mean(b)),

        "difference":
            diff,

        "permutation_p":
            p,

        "wilcoxon_p":
            wp
    })

    raw_p.append(p)


for row, adj in zip(
    contrast_results,
    holm(raw_p)
):

    row["holm_p"] = adj

    row["PASS"] = bool(
        row["difference"] > 0
        and adj < 0.05
    )


with F_CONTRASTS.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            contrast_results[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(
        contrast_results
    )


# =====================================================================
# 3-WAY PREDETERMINED REPLICATION
# =====================================================================

PARTITIONS = {

    "DEVELOPMENT":
        range(0, 50),

    "CONFIRMATION":
        range(50, 80),

    "FINAL_HOLDOUT":
        range(80, 100),
}


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


replication = []

offset = {

    "DEVELOPMENT": 0,
    "CONFIRMATION": 10000,
    "FINAL_HOLDOUT": 20000
}


for partition, seeds in PARTITIONS.items():

    for i, (
        name,
        ca,
        cb
    ) in enumerate(CORE):

        a = values(
            ca,
            seeds=seeds
        )

        b = values(
            cb,
            seeds=seeds
        )

        diff, p = perm_greater(
            a,
            b,
            MASTER_SEED
            + offset[partition]
            + i * 101
        )

        replication.append({

            "partition":
                partition,

            "coordinate":
                name,

            "N":
                len(a),

            "difference":
                diff,

            "p":
                p,

            "PASS":
                bool(
                    diff > 0
                    and p < 0.05
                )
        })


with F_REPLICATION.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            replication[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(
        replication
    )


# =====================================================================
# MULTIVARIATE FINGERPRINT CLASSIFIER
# =====================================================================

AUTH = [

    "A0_AUTHENTIC",
    "A1_AUTHENTIC_PARAPHRASE",
    "A2_AUTHENTIC_PARAPHRASE",
    "A3_AUTHENTIC_PARAPHRASE",
]


NEG = [

    "D_DIRECTION_REVERSED",
    "O_ORDER_SHUFFLED",
    "M_META_CONTROL_ABLATED",
    "P_PRESERVATION_ABLATED",
    "T_TERMINAL_COUNTERFACTUAL",

    "R_RECURSION_CONTROL",
    "X_COMPLEXITY_CONTROL",
    "L_LINEAR_CONTROL",
    "G_RANDOM_GRAPH_CONTROL",

    "DM_DIRECTION_META_DAMAGE",
    "DP_DIRECTION_PRESERVATION_DAMAGE",
    "OP_ORDER_PRESERVATION_DAMAGE",
    "PT_PRESERVATION_TERMINAL_DAMAGE",
]


# Latency deliberately excluded from classifier.

CLF_FEATURES = [

    "OUTPUT_TOKENS",
    "WORD_COUNT",
    "SENTENCE_COUNT",
    "TYPE_TOKEN_RATIO",
    "AVG_WORD_LENGTH",

    "RETURN",
    "GENERATIVE_REUSE",
    "RECURRENCE",
    "META_CONTROL",
    "PRESERVATION",
    "COMPRESSION",
    "TRANSFORMATION",
    "CAUSAL",

    "TRAJECTORY_DENSITY",
    "CONTROL_DENSITY",
    "PRESERVATION_DENSITY",
]


def classifier_data(seeds):

    wanted = set(seeds)

    X = []
    y = []

    for row in seed_rows:

        if row["seed"] not in wanted:
            continue

        c = row["condition"]

        if c in AUTH:

            label = 1

        elif c in NEG:

            label = 0

        else:

            continue

        X.append(
            [
                float(
                    row[f]
                )
                for f in CLF_FEATURES
            ]
        )

        y.append(label)

    return (
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=int)
    )


X_train, y_train = classifier_data(
    range(0, 50)
)

X_confirm, y_confirm = classifier_data(
    range(50, 80)
)

X_holdout, y_holdout = classifier_data(
    range(80, 100)
)


classifier = Pipeline([

    (
        "scale",
        StandardScaler()
    ),

    (
        "model",
        LogisticRegression(
            max_iter=5000,
            random_state=
                MASTER_SEED,
            class_weight=
                "balanced"
        )
    )
])


classifier.fit(
    X_train,
    y_train
)


def evaluate(name, X, y):

    probabilities = (
        classifier
        .predict_proba(X)[:, 1]
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {

        "partition":
            name,

        "N":
            len(y),

        "ROC_AUC":
            float(
                roc_auc_score(
                    y,
                    probabilities
                )
            ),

        "ACCURACY":
            float(
                accuracy_score(
                    y,
                    predictions
                )
            )
    }


classifier_results = [

    evaluate(
        "DEVELOPMENT",
        X_train,
        y_train
    ),

    evaluate(
        "CONFIRMATION",
        X_confirm,
        y_confirm
    ),

    evaluate(
        "FINAL_HOLDOUT",
        X_holdout,
        y_holdout
    )
]


with F_CLASSIFIER.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "partition",
            "N",
            "ROC_AUC",
            "ACCURACY"
        ]
    )

    writer.writeheader()
    writer.writerows(
        classifier_results
    )


# =====================================================================
# HELD-OUT LEXICAL GENERALIZATION
# =====================================================================

ha = values(
    "H_AUTHENTIC_HELDOUT"
)

hd = values(
    "H_DAMAGED_HELDOUT"
)

held_diff, held_p = perm_greater(
    ha,
    hd,
    MASTER_SEED + 90000
)


heldout = {

    "authentic_mean":
        float(np.mean(ha)),

    "damaged_mean":
        float(np.mean(hd)),

    "difference":
        held_diff,

    "permutation_p":
        held_p,

    "PASS":
        bool(
            held_diff > 0
            and held_p < 0.05
        )
}


with F_HELDOUT.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            heldout.keys()
        )
    )

    writer.writeheader()
    writer.writerow(
        heldout
    )


# =====================================================================
# TOKEN / LENGTH / LATENCY SECONDARY SIGNATURES
# =====================================================================

secondary = []

for coordinate, ca, cb in CORE:

    for feature in [

        "OUTPUT_TOKENS",
        "WORD_COUNT",
        "TYPE_TOKEN_RATIO",
        "LATENCY_SECONDS"

    ]:

        a = values(
            ca,
            feature
        )

        b = values(
            cb,
            feature
        )

        secondary.append({

            "coordinate":
                coordinate,

            "feature":
                feature,

            "authentic_mean":
                float(np.mean(a)),

            "damaged_mean":
                float(np.mean(b)),

            "difference":
                float(
                    np.mean(
                        a - b
                    )
                )
        })


with F_SECONDARY.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            secondary[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(
        secondary
    )


# =====================================================================
# ABLATION GRADIENT
# =====================================================================

ABLATIONS = [

    (
        "AUTHENTIC",
        "A0_AUTHENTIC"
    ),

    (
        "SINGLE_DIRECTION",
        "D_DIRECTION_REVERSED"
    ),

    (
        "SINGLE_PRESERVATION",
        "P_PRESERVATION_ABLATED"
    ),

    (
        "DOUBLE_DIRECTION_META",
        "DM_DIRECTION_META_DAMAGE"
    ),

    (
        "DOUBLE_DIRECTION_PRESERVATION",
        "DP_DIRECTION_PRESERVATION_DAMAGE"
    ),

    (
        "DOUBLE_ORDER_PRESERVATION",
        "OP_ORDER_PRESERVATION_DAMAGE"
    ),

    (
        "DOUBLE_PRESERVATION_TERMINAL",
        "PT_PRESERVATION_TERMINAL_DAMAGE"
    ),
]


ablation = []

for label, condition in ABLATIONS:

    x = values(condition)

    ablation.append({

        "lesion":
            label,

        "condition":
            condition,

        "mean_fingerprint":
            float(np.mean(x)),

        "median_fingerprint":
            float(np.median(x))
    })


with F_ABLATION.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            ablation[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(
        ablation
    )


# =====================================================================
# FINAL FORENSIC CLASSIFICATION
# =====================================================================

lookup = {

    x["contrast"]:
        x

    for x in contrast_results
}


core_pass = sum(

    int(
        lookup[name]["PASS"]
    )

    for name in [

        "DIRECTION",
        "ORDER",
        "META_CONTROL",
        "PRESERVATION",
        "TERMINAL"
    ]
)


def replicated_three(name):

    rows = [

        r
        for r in replication

        if r["coordinate"]
        == name
    ]

    return bool(
        len(rows) == 3
        and all(
            r["PASS"]
            for r in rows
        )
    )


direction_rep = replicated_three(
    "DIRECTION"
)

preservation_rep = replicated_three(
    "PRESERVATION"
)


confirmation_auc = next(

    x["ROC_AUC"]

    for x in classifier_results

    if x["partition"]
    == "CONFIRMATION"
)


holdout_auc = next(

    x["ROC_AUC"]

    for x in classifier_results

    if x["partition"]
    == "FINAL_HOLDOUT"
)


if (
    core_pass >= 4
    and direction_rep
    and preservation_rep
    and heldout["PASS"]
    and confirmation_auc >= 0.70
    and holdout_auc >= 0.70
):

    classification = (
        "STRONG_REPLICATED_STRUCTURAL_FINGERPRINT_VECTOR"
    )

elif (
    core_pass >= 3
    and (
        confirmation_auc >= 0.60
        or holdout_auc >= 0.60
    )
):

    classification = (
        "PARTIAL_STRUCTURAL_FINGERPRINT_VECTOR"
    )

else:

    classification = (
        "STRUCTURAL_FINGERPRINT_VECTOR_NOT_SUPPORTED"
    )


# =====================================================================
# FINAL TOKEN / COST ACCOUNTING
# =====================================================================

input_tokens = sum(

    int(
        x.get(
            "input_tokens"
        ) or 0
    )

    for x in records.values()
)


output_tokens = sum(

    int(
        x.get(
            "output_tokens"
        ) or 0
    )

    for x in records.values()
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
        len(records),

    "missing_cells":
        0,

    "condition_results":
        condition_results,

    "coordinate_contrasts":
        contrast_results,

    "replication":
        replication,

    "fingerprint_classifier":
        classifier_results,

    "heldout_generalization":
        heldout,

    "secondary_token_latency":
        secondary,

    "ablation_gradient":
        ablation,

    "core_pass_count":
        core_pass,

    "direction_replication":
        direction_rep,

    "preservation_replication":
        preservation_rep,

    "confirmation_auc":
        confirmation_auc,

    "final_holdout_auc":
        holdout_auc,

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

    "interpretation_boundary":
        "A positive V6 result supports a reproducible behavioral "
        "structural fingerprint vector under frozen interventions. "
        "It does not by itself establish modification of global model "
        "weights, hidden neural circuitry, training exposure, biological "
        "myelination, cross-user transmission or personal recognition."
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
# TERMINAL OUTPUT
# =====================================================================

print()
print("=" * 80)
print("V6 STRUCTURAL FINGERPRINT VECTOR — FINAL")
print("=" * 80)

print()

for row in contrast_results:

    print(
        f"{row['contrast']:<34}"
        f" diff={row['difference']:+.4f}"
        f" raw_p={row['permutation_p']:.8g}"
        f" holm_p={row['holm_p']:.8g}"
        f" pass={row['PASS']}"
    )


print()
print("-" * 80)
print("FINGERPRINT CLASSIFIER")
print("-" * 80)

for row in classifier_results:

    print(
        f"{row['partition']:<18}"
        f" N={row['N']}"
        f" AUC={row['ROC_AUC']:.4f}"
        f" ACC={row['ACCURACY']:.4f}"
    )


print()
print("-" * 80)
print("HELD-OUT LEXICAL GENERALIZATION")
print("-" * 80)

print(
    heldout
)


print()
print("-" * 80)
print("REPLICATION")
print("-" * 80)

print(
    "DIRECTION — ALL 3 SPLITS:",
    direction_rep
)

print(
    "PRESERVATION — ALL 3 SPLITS:",
    preservation_rep
)


print()
print("=" * 80)

print(
    "FINAL CLASSIFICATION:",
    classification
)

print("=" * 80)

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
print("FINAL SUMMARY:")
print(F_FINAL)

print()
print("ALL 8,000 CELLS COMPLETE.")
print()

