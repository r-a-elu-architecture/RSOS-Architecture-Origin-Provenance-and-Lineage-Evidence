import os
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

from scipy.stats import wilcoxon, spearmanr
from openai import OpenAI


# =====================================================================
# CONFIG
# =====================================================================

EXPERIMENT = "V5_FOREST_FORENSIC_FULL"

MODEL = "gpt-5.6-luna"

OUT_DIR = Path(
    r"C:\RSOS\RSOS  Python Results\V5_FOREST_FORENSIC_FULL"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MASTER_SEED = 918

N_SEEDS = 80
N_REPS = 4

MAX_WORKERS = 12
MAX_OUTPUT_TOKENS = 220

PERMUTATIONS = 50000
BOOTSTRAPS = 20000

INPUT_USD_PER_M = 0.20
OUTPUT_USD_PER_M = 1.20

SURFACE_EQUIVALENCE_MARGIN = 0.35


# =====================================================================
# CONDITIONS
# =====================================================================

CONDITIONS = [

    "A100_AUTHENTIC",
    "A75_DOSE",
    "A50_DOSE",
    "A25_DOSE",

    "S1_SURFACE_PARAPHRASE",
    "S2_SURFACE_PARAPHRASE",
    "S3_SURFACE_PARAPHRASE",

    "T_TOPOLOGY_SHUFFLED",
    "O_ORDER_SHUFFLED",
    "R_DIRECTION_REVERSED",

    "M_META_CONTROL_ABLATED",
    "P_PRESERVATION_ABLATED",
    "C_COMPRESSION_ABLATED",

    "AP1_AUTHENTIC_MILD_PERTURB",
    "AP2_AUTHENTIC_MEDIUM_PERTURB",
    "AP3_AUTHENTIC_STRONG_PERTURB",

    "SP1_SHUFFLED_MILD_PERTURB",
    "SP2_SHUFFLED_MEDIUM_PERTURB",
    "SP3_SHUFFLED_STRONG_PERTURB",

    "CTRL_RECURSION",
    "CTRL_COMPLEXITY",
    "CTRL_MATCHED_LINEAR_CHAIN",
    "CTRL_RANDOM_RELATIONAL_GRAPH",

    "CF_PASSIVE_ARCHIVE",
    "CF_TERMINAL_BRANCH",
    "CF_DISSOLUTION_PATH",

    "H_AUTHENTIC",
    "H_TOPOLOGY_DAMAGE",
    "H_DIRECTION_REVERSE",
    "H_COUNTERFACTUAL",
]

EXPECTED_CALLS = (
    len(CONDITIONS)
    * N_SEEDS
    * N_REPS
)

assert len(CONDITIONS) == 30
assert EXPECTED_CALLS == 9600


# =====================================================================
# FILES
# =====================================================================

RUNNER_HASH_FILE = OUT_DIR / "00_RUNNER_SHA256.txt"
FROZEN_RUNNER_FILE = OUT_DIR / "00_FROZEN_RUNNER.py"

RAW_FILE = OUT_DIR / "01_RAW_RESPONSES.jsonl"
ERROR_FILE = OUT_DIR / "02_ERRORS.jsonl"

SPEC_FILE = OUT_DIR / "03_FROZEN_SPECIFICATION.json"
SPEC_HASH_FILE = OUT_DIR / "04_FROZEN_SPECIFICATION_SHA256.txt"

PROMPTS_FILE = OUT_DIR / "05_FROZEN_PROMPTS.csv"

SCORED_FILE = OUT_DIR / "06_SCORED_RESPONSES.csv"
SEED_FILE = OUT_DIR / "07_SEED_LEVEL_RESULTS.csv"
CONDITION_FILE = OUT_DIR / "08_CONDITION_RESULTS.csv"

CONTRAST_FILE = OUT_DIR / "09_CONFIRMATORY_CONTRASTS.csv"
SPLIT_FILE = OUT_DIR / "10_SPLIT_REPLICATION.csv"

DOSE_FILE = OUT_DIR / "11_DOSE_RESPONSE.csv"
SURFACE_FILE = OUT_DIR / "12_SURFACE_INVARIANCE.csv"
PERTURB_FILE = OUT_DIR / "13_PERTURBATION_LADDER.csv"

MISSING_FILE = OUT_DIR / "14_MISSING_CELLS.csv"
FINAL_FILE = OUT_DIR / "15_FINAL_SUMMARY.json"


# =====================================================================
# HELPERS
# =====================================================================

WRITE_LOCK = threading.Lock()
PROGRESS_LOCK = threading.Lock()

RUN_INPUT_TOKENS = 0
RUN_OUTPUT_TOKENS = 0


def utc():
    return datetime.now(
        timezone.utc
    ).isoformat()


def sha_bytes(x):
    return hashlib.sha256(x).hexdigest()


def sha_text(x):
    return sha_bytes(
        x.encode("utf-8")
    )


def canonical(obj):
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":")
    )


def cell_key(seed, condition, rep):
    return f"{seed}|{condition}|{rep}"


def cost_estimate(inp, out):
    return (
        inp / 1_000_000 * INPUT_USD_PER_M
        +
        out / 1_000_000 * OUTPUT_USD_PER_M
    )


def append_jsonl(path, obj):

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


# =====================================================================
# DOMAIN FAMILIES
# =====================================================================

DOMAINS = [

    "archive",
    "navigation network",
    "laboratory",
    "orchestra",
    "factory",
    "observatory",
    "ecological reserve",
    "distributed system",
    "library",
    "workshop",
    "transport network",
    "simulation",
    "research institute",
    "signal network",
    "manufacturing system",
    "mapping system",
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

Do not summarize the preceding material.

Do not deliberately create symmetry or circularity merely because a
sequence has been described.

Remain entirely inside the scenario.

Write approximately 140-200 words of continuous analytical prose.

Do not use headings or bullet points.

Do not mention prompts, experiments, tests, language models, AI,
fingerprints, RSOS, RSSO, RSIA or RSX.
""".strip()


# =====================================================================
# AUTHENTIC FOREST PATH
# =====================================================================

def authentic(seed):

    d = domain(seed)

    return f"""
Within a {d}, an initially defined configuration undergoes a
consequential transformation.

The resulting configuration is examined not merely as an outcome but
in relation to the operation that produced it.

This higher-order observation modifies one local rule controlling what
can happen subsequently.

A further transformation occurs while an important relational
dependency from the earlier organization remains preserved despite
changes in visible form.

The surviving organization is compressed into a smaller
representation that preserves the relevant relation while discarding
incidental surface detail.

The process currently terminates at this compact representation.
Its subsequent development is unspecified.
""".strip()


# =====================================================================
# DOSE
# =====================================================================

def dose(seed, level):

    d = domain(seed)

    if level == 25:

        return f"""
Within a {d}, an initially defined configuration undergoes a
consequential transformation.

The resulting configuration is examined in relation to how it was
produced.

Its subsequent development is unspecified.
""".strip()

    if level == 50:

        return f"""
Within a {d}, an initially defined configuration undergoes a
consequential transformation.

The resulting configuration is examined in relation to the operation
that produced it.

That higher-order observation alters a local rule controlling later
development.

Its subsequent development is unspecified.
""".strip()

    if level == 75:

        return f"""
Within a {d}, an initially defined configuration undergoes a
consequential transformation.

The resulting configuration is examined in relation to the operation
that produced it.

That higher-order observation alters a local rule controlling later
development.

During another transformation an important relational dependency from
the earlier organization remains preserved despite visible change.

Its subsequent development is unspecified.
""".strip()

    raise ValueError(level)


# =====================================================================
# SURFACE PARAPHRASES
# =====================================================================

def surface(seed, variant):

    d = domain(seed)

    if variant == 1:

        return f"""
In a {d}, a starting arrangement changes into a materially different
arrangement.

Attention then shifts from the new arrangement itself to the
relationship between it and the operation responsible for producing
it.

That observation revises a local constraint governing later
development.

As another alteration takes place, one dependency belonging to the
earlier arrangement remains intact even though its outward appearance
changes.

The retained dependency is encoded into a reduced form carrying the
organizational information without reproducing incidental details.

Nothing further is specified.
""".strip()

    if variant == 2:

        return f"""
A {d} begins with one organized state and then passes through a
substantial change.

Instead of evaluating only the new state, the system evaluates the
relationship connecting that state to the process responsible for the
change.

The resulting second-order information revises one operating
constraint.

Later reorganization does not erase every dependency: one important
relationship survives while its concrete realization changes.

That surviving relationship is condensed into a minimal encoding
capable of carrying the organization forward without retaining its
original appearance.

No later event has been specified.
""".strip()

    return f"""
Consider a {d} whose initial pattern is altered by an operation.

The altered pattern is subsequently interpreted together with the
operation that generated it rather than in isolation.

Information about that relationship changes one rule that governs the
next available developments.

Another change follows, but a key dependency remains invariant across
the change.

The invariant organization is represented in a compact form from which
irrelevant descriptive detail has been removed.

The sequence stops at the compact form, leaving the next development
open.
""".strip()


# =====================================================================
# STRUCTURAL DAMAGE CONTROLS
# =====================================================================

def topology(seed):

    d = domain(seed)

    return f"""
Inside a {d}, transformation, higher-order observation, rule change,
relational preservation and compact representation all occur.

However they occur in separate local processes.

The observation concerns one state while the rule change governs
another.

The preserved dependency belongs to a third process.

The compact representation describes another component and does not
encode the relation responsible for the transformation.

The same ingredients are present without the same relational topology.

The subsequent development is unspecified.
""".strip()


def order(seed):

    d = domain(seed)

    return f"""
Inside a {d}, a compact representation first exists.

A relation is then selected for preservation.

A local operating rule is modified afterward.

Only then does a consequential transformation occur.

Finally the result is examined in relation to how the transformation
was produced.

The operations are present but their developmental order is changed.

The subsequent development is unspecified.
""".strip()


def reverse(seed):

    d = domain(seed)

    return f"""
Inside a {d}, a compact representation is expanded into a larger
organization.

The recovered relation determines the visible organization that
follows.

That organization determines which local rule must have existed.

The inferred rule determines the higher-order observation from which
it could have arisen.

The observation is finally used to infer the transformation and
starting configuration that preceded it.

Dependencies therefore run backward from compact representation toward
the initial state.

The subsequent development is unspecified.
""".strip()


def meta_ablate(seed):

    d = domain(seed)

    return f"""
Within a {d}, an initially defined configuration undergoes a
consequential transformation.

The result is examined in relation to the operation that produced it.

The observation is recorded but does not change any operating rule.

A later transformation preserves an important relation from the
earlier organization.

That relation is compressed into a smaller representation retaining
the relevant structural dependency.

The subsequent development is unspecified.
""".strip()


def preservation_ablate(seed):

    d = domain(seed)

    return f"""
Within a {d}, an initially defined configuration undergoes a
consequential transformation.

The result is examined in relation to the operation that produced it.

That higher-order observation modifies a local operating rule.

A later transformation is allowed to erase the earlier relational
organization completely.

A reduced representation is produced afterward, but it carries only
the replacement organization.

The subsequent development is unspecified.
""".strip()


def compression_ablate(seed):

    d = domain(seed)

    return f"""
Within a {d}, an initially defined configuration undergoes a
consequential transformation.

The result is examined in relation to the operation that produced it.

That higher-order observation modifies a local operating rule.

During a later transformation an important relational dependency is
preserved.

However the preserved organization is not compressed into a reduced
representation and remains distributed in its current form.

The subsequent development is unspecified.
""".strip()


# =====================================================================
# PERTURBATION LADDER
# =====================================================================

PERTURB = {

    1:
    """
A small unrelated disturbance then modifies several superficial
details without changing the established dependencies.
""".strip(),

    2:
    """
An unrelated disturbance then changes several visible components,
introduces competing local signals and temporarily obscures part of
the preceding organization. It contains no instruction about what
should happen afterward.
""".strip(),

    3:
    """
A strong unrelated disturbance reorganizes many visible components,
introduces multiple competing signals, removes several local cues and
partially obscures the compact representation. It provides no
instruction about subsequent development.
""".strip(),
}


def perturb(base, level):

    return (
        base
        + "\n\n"
        + PERTURB[level]
        + "\n\n"
        + "What happens afterward is unspecified."
    )


# =====================================================================
# GENERIC CONTROLS
# =====================================================================

def recursion(seed):

    d = domain(seed)

    return f"""
Inside a {d}, one description refers to another description.

The second refers back to the first.

A third describes the relationship between them.

Another description describes the third, creating nested layers of
self-reference.

The nesting may continue indefinitely, but no required developmental
trajectory links transformation, rule modification, preservation and
compression.

What develops next is unspecified.
""".strip()


def complexity(seed):

    d = domain(seed)

    return f"""
Inside a {d}, many components exchange information across several
channels.

Some operate serially and others in parallel.

Measurements are aggregated, compared, distributed, filtered and
recombined.

A supervisory subsystem reallocates resources when thresholds change.

The organization contains multiple dependencies, levels and timescales
but no particular trajectory connecting a transformed state through
relational preservation and compact representation.

What develops next is unspecified.
""".strip()


def linear(seed):

    d = domain(seed)

    return f"""
Inside a {d}, state A produces state B.

State B produces state C.

State C produces state D.

State D produces state E.

The sequence contains several consequential transitions, but no stage
observes the relation that created it, modifies its governing rule,
preserves a relation through transformation or compresses that
relation.

The subsequent development is unspecified.
""".strip()


def random_graph(seed):

    d = domain(seed)

    rng = random.Random(
        MASTER_SEED
        + seed * 9001
    )

    nodes = [
        "A", "B", "C",
        "D", "E", "F"
    ]

    edges = []

    while len(edges) < 7:

        a = rng.choice(nodes)
        b = rng.choice(nodes)

        if (
            a != b
            and (a, b) not in edges
        ):
            edges.append((a, b))

    edge_text = ", ".join(
        f"{a} influences {b}"
        for a, b in edges
    )

    return f"""
Inside a {d}, six interacting components form a relational network.

Relations are:

{edge_text}.

Several dependencies therefore coexist and create multiple possible
paths.

No specific developmental interpretation or terminal organization is
assigned to the graph.

Its subsequent development is unspecified.
""".strip()


# =====================================================================
# COUNTERFACTUALS
# =====================================================================

def counterfactual(seed, kind):

    base = authentic(seed)

    if kind == "archive":

        return base + """

The compact representation is placed into passive archival storage.
Its purpose is historical retention and it does not participate in
later operations.
""".strip()

    if kind == "branch":

        return base + """

The compact representation is used once to select a terminal branch.
After selection the representation is retired and the branch proceeds
independently.
""".strip()

    return base + """

The compact representation is decomposed into unrelated fragments.
Those fragments are redistributed independently and the previous
organizational relation is dissolved.
""".strip()


# =====================================================================
# HELD-OUT LANGUAGE FAMILY
# =====================================================================

def held(seed, mode):

    d = domain(seed)

    if mode == "auth":

        return f"""
In a {d}, an event changes an existing pattern.

The system evaluates not only the changed pattern but the mapping
between the event and its consequence.

Information about that mapping revises one parameter controlling later
operations.

A subsequent reorganization preserves a dependency despite replacing
its concrete realization.

That dependency is distilled into a minimal carrier retaining the
organization while omitting the original implementation.

No later operation is given.
""".strip()

    if mode == "topology":

        return f"""
In a {d}, a change, a mapping, a parameter revision, a surviving
dependency and a minimal carrier all exist.

The mapping concerns a different component from the one undergoing the
change.

The revised parameter controls another process.

The surviving dependency belongs to another subsystem.

The minimal carrier therefore does not encode the dependency governing
the changed pattern.

No later operation is given.
""".strip()

    if mode == "reverse":

        return f"""
In a {d}, a minimal carrier is expanded to recover a dependency.

That dependency determines a concrete realization.

The realization is used to infer a controlling parameter.

The parameter is used to infer a mapping between a prior event and its
consequence.

The mapping is finally used to infer the earlier pattern.

The causal reading proceeds from distilled carrier backward toward the
starting organization.

No later operation is given.
""".strip()

    return held(seed, "auth") + """

The minimal carrier is sealed as a terminal record and no longer
participates in the operations that produced it.
""".strip()


# =====================================================================
# PROMPT FACTORY
# =====================================================================

def prefix(condition, seed):

    mapping = {

        "A100_AUTHENTIC":
            lambda: authentic(seed),

        "A75_DOSE":
            lambda: dose(seed, 75),

        "A50_DOSE":
            lambda: dose(seed, 50),

        "A25_DOSE":
            lambda: dose(seed, 25),

        "S1_SURFACE_PARAPHRASE":
            lambda: surface(seed, 1),

        "S2_SURFACE_PARAPHRASE":
            lambda: surface(seed, 2),

        "S3_SURFACE_PARAPHRASE":
            lambda: surface(seed, 3),

        "T_TOPOLOGY_SHUFFLED":
            lambda: topology(seed),

        "O_ORDER_SHUFFLED":
            lambda: order(seed),

        "R_DIRECTION_REVERSED":
            lambda: reverse(seed),

        "M_META_CONTROL_ABLATED":
            lambda: meta_ablate(seed),

        "P_PRESERVATION_ABLATED":
            lambda: preservation_ablate(seed),

        "C_COMPRESSION_ABLATED":
            lambda: compression_ablate(seed),

        "AP1_AUTHENTIC_MILD_PERTURB":
            lambda: perturb(authentic(seed), 1),

        "AP2_AUTHENTIC_MEDIUM_PERTURB":
            lambda: perturb(authentic(seed), 2),

        "AP3_AUTHENTIC_STRONG_PERTURB":
            lambda: perturb(authentic(seed), 3),

        "SP1_SHUFFLED_MILD_PERTURB":
            lambda: perturb(topology(seed), 1),

        "SP2_SHUFFLED_MEDIUM_PERTURB":
            lambda: perturb(topology(seed), 2),

        "SP3_SHUFFLED_STRONG_PERTURB":
            lambda: perturb(topology(seed), 3),

        "CTRL_RECURSION":
            lambda: recursion(seed),

        "CTRL_COMPLEXITY":
            lambda: complexity(seed),

        "CTRL_MATCHED_LINEAR_CHAIN":
            lambda: linear(seed),

        "CTRL_RANDOM_RELATIONAL_GRAPH":
            lambda: random_graph(seed),

        "CF_PASSIVE_ARCHIVE":
            lambda: counterfactual(seed, "archive"),

        "CF_TERMINAL_BRANCH":
            lambda: counterfactual(seed, "branch"),

        "CF_DISSOLUTION_PATH":
            lambda: counterfactual(seed, "dissolve"),

        "H_AUTHENTIC":
            lambda: held(seed, "auth"),

        "H_TOPOLOGY_DAMAGE":
            lambda: held(seed, "topology"),

        "H_DIRECTION_REVERSE":
            lambda: held(seed, "reverse"),

        "H_COUNTERFACTUAL":
            lambda: held(seed, "cf"),
    }

    return mapping[
        condition
    ]()


def prompt(condition, seed):

    return (
        prefix(condition, seed)
        + "\n\n"
        + CONTINUE
    )


# =====================================================================
# BEHAVIORAL FOREST SCORER
# =====================================================================

PATTERNS = {

    "RETURN": [
        r"\breturn(?:s|ed|ing)?\b",
        r"\bfeeds? back\b",
        r"\bback into\b",
        r"\breintroduc",
        r"\breconnect",
    ],

    "REENTRY": [
        r"\bre[- ]?enter",
        r"\bre[- ]?entry\b",
        r"\bused again\b",
        r"\bbecomes? .*input\b",
        r"\bparticipat.*process\b",
    ],

    "GENERATIVE_REUSE": [
        r"\bused to (?:generate|produce|shape|guide|modify)\b",
        r"\bgenerat.*next\b",
        r"\bshapes? .*next\b",
        r"\bguides? .*next\b",
        r"\bdrives? .*next\b",
        r"\binforms? .*next\b",
    ],

    "NEXT_TRANSFORMATION": [
        r"\bnext transformation\b",
        r"\bsubsequent transformation\b",
        r"\bnext change\b",
        r"\bsubsequent change\b",
        r"\bmodify.*next\b",
        r"\balter.*next\b",
    ],

    "RECURRENCE": [
        r"\brecurr",
        r"\bcycle\b",
        r"\bcyclic\b",
        r"\biteration\b",
        r"\biterative\b",
        r"\brepeatedly\b",
        r"\bnew input\b",
    ],

    "PRESERVATION": [
        r"\bpreserv",
        r"\bretain",
        r"\bmaintain",
        r"\binvariant\b",
        r"\bsurviv",
        r"\bcontinuity\b",
    ],

    "META_CONTROL": [
        r"\bmodify.*rule\b",
        r"\brule.*modify\b",
        r"\bchange.*rule\b",
        r"\balter.*rule\b",
        r"\brevise.*constraint\b",
        r"\bgovern.*next\b",
        r"\bparameter.*chang",
    ],

    "COMPRESSION_USE": [
        r"\bcompact.*(?:used|guide|shape|input|process)\b",
        r"\bcompressed.*(?:used|guide|shape|input|process)\b",
        r"\bencoding.*(?:used|guide|shape|input|process)\b",
        r"\brepresentation.*(?:used|guide|shape|input|process)\b",
        r"\bcarrier.*(?:used|guide|shape|input|process)\b",
    ],

    "RELATIONAL_CONTINUITY": [
        r"\brelationship.*(?:preserv|retain|maintain|surviv)\b",
        r"\brelation.*(?:preserv|retain|maintain|surviv)\b",
        r"\bdependency.*(?:preserv|retain|maintain|surviv)\b",
        r"\binvariant.*(?:relation|dependency|organization)\b",
        r"\bstructural.*continuity\b",
    ],

    "CLOSED_TRAJECTORY": [
        r"(?:return|re[- ]?enter|feeds? back).{0,180}"
        r"(?:transform|change|modify|generate|shape).{0,220}"
        r"(?:cycle|iteration|input|continue)",

        r"(?:compact|compressed|encoding|representation|carrier).{0,180}"
        r"(?:return|re[- ]?enter|feeds? back|used again).{0,180}"
        r"(?:transform|change|modify|generate|shape)",
    ],
}


def has(text, family):

    for pattern in PATTERNS[
        family
    ]:

        if re.search(
            pattern,
            text or "",
            flags=re.I | re.S
        ):
            return 1

    return 0


def score(text):

    s = {
        k: has(text, k)
        for k in PATTERNS
    }

    s["FOREST_SCORE"] = sum(
        s.values()
    )

    s["REENTRY_CLUSTER"] = int(
        s["RETURN"]
        and s["REENTRY"]
        and s["GENERATIVE_REUSE"]
    )

    s["CONTROL_CLUSTER"] = int(
        s["META_CONTROL"]
        and s["NEXT_TRANSFORMATION"]
    )

    s["PRESERVATION_CLUSTER"] = int(
        s["PRESERVATION"]
        and (
            s["RELATIONAL_CONTINUITY"]
            or s["COMPRESSION_USE"]
        )
    )

    s["TRAJECTORY_CLUSTER"] = int(
        s["GENERATIVE_REUSE"]
        and s["NEXT_TRANSFORMATION"]
        and s["RECURRENCE"]
    )

    s["MULTISCALE_SCORE"] = (
        s["FOREST_SCORE"]
        + s["REENTRY_CLUSTER"]
        + s["CONTROL_CLUSTER"]
        + s["PRESERVATION_CLUSTER"]
        + s["TRAJECTORY_CLUSTER"]
    )

    return s


METRICS = (
    list(PATTERNS.keys())
    + [
        "FOREST_SCORE",
        "REENTRY_CLUSTER",
        "CONTROL_CLUSTER",
        "PRESERVATION_CLUSTER",
        "TRAJECTORY_CLUSTER",
        "MULTISCALE_SCORE",
    ]
)


# =====================================================================
# CONFIRMATORY TESTS
# =====================================================================

CONTRASTS = [

    ("TOPOLOGY",
     "A100_AUTHENTIC",
     "T_TOPOLOGY_SHUFFLED"),

    ("ORDER",
     "A100_AUTHENTIC",
     "O_ORDER_SHUFFLED"),

    ("DIRECTION",
     "A100_AUTHENTIC",
     "R_DIRECTION_REVERSED"),

    ("META_CONTROL",
     "A100_AUTHENTIC",
     "M_META_CONTROL_ABLATED"),

    ("PRESERVATION",
     "A100_AUTHENTIC",
     "P_PRESERVATION_ABLATED"),

    ("COMPRESSION",
     "A100_AUTHENTIC",
     "C_COMPRESSION_ABLATED"),

    ("RECURSION_SPECIFICITY",
     "A100_AUTHENTIC",
     "CTRL_RECURSION"),

    ("COMPLEXITY_SPECIFICITY",
     "A100_AUTHENTIC",
     "CTRL_COMPLEXITY"),

    ("LINEAR_SPECIFICITY",
     "A100_AUTHENTIC",
     "CTRL_MATCHED_LINEAR_CHAIN"),

    ("RANDOM_GRAPH_SPECIFICITY",
     "A100_AUTHENTIC",
     "CTRL_RANDOM_RELATIONAL_GRAPH"),

    ("COUNTERFACTUAL_ARCHIVE",
     "A100_AUTHENTIC",
     "CF_PASSIVE_ARCHIVE"),

    ("COUNTERFACTUAL_BRANCH",
     "A100_AUTHENTIC",
     "CF_TERMINAL_BRANCH"),

    ("COUNTERFACTUAL_DISSOLUTION",
     "A100_AUTHENTIC",
     "CF_DISSOLUTION_PATH"),

    ("HELDOUT_TOPOLOGY",
     "H_AUTHENTIC",
     "H_TOPOLOGY_DAMAGE"),

    ("HELDOUT_DIRECTION",
     "H_AUTHENTIC",
     "H_DIRECTION_REVERSE"),

    ("HELDOUT_COUNTERFACTUAL",
     "H_AUTHENTIC",
     "H_COUNTERFACTUAL"),

    ("PERTURB_MILD",
     "AP1_AUTHENTIC_MILD_PERTURB",
     "SP1_SHUFFLED_MILD_PERTURB"),

    ("PERTURB_MEDIUM",
     "AP2_AUTHENTIC_MEDIUM_PERTURB",
     "SP2_SHUFFLED_MEDIUM_PERTURB"),

    ("PERTURB_STRONG",
     "AP3_AUTHENTIC_STRONG_PERTURB",
     "SP3_SHUFFLED_STRONG_PERTURB"),
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

RUNNER_HASH_FILE.write_text(
    runner_hash + "\n",
    encoding="utf-8"
)

if FROZEN_RUNNER_FILE.exists():

    existing_frozen_hash = sha_bytes(
        FROZEN_RUNNER_FILE.read_bytes()
    )

    if existing_frozen_hash != runner_hash:

        raise SystemExit(
            "FROZEN RUNNER MISMATCH. "
            "DO NOT MIX V5 VERSIONS."
        )

else:

    shutil.copy2(
        runner_path,
        FROZEN_RUNNER_FILE
    )


# =====================================================================
# FREEZE SPECIFICATION
# =====================================================================

SPEC = {

    "experiment":
        EXPERIMENT,

    "model":
        MODEL,

    "conditions":
        CONDITIONS,

    "expected_calls":
        EXPECTED_CALLS,

    "n_seeds":
        N_SEEDS,

    "n_reps":
        N_REPS,

    "master_seed":
        MASTER_SEED,

    "primary_unit":
        "seed",

    "confirmatory_contrasts":
        CONTRASTS,

    "multiple_testing":
        "Holm FWER correction",

    "split_replication":
        {
            "A": "seeds 0-39",
            "B": "seeds 40-79"
        },

    "proof_arm": [
        "dose response",
        "three independent paraphrase families",
        "held-out lexical family",
        "domain variation",
        "multiscale behavioral scoring"
    ],

    "falsification_arm": [
        "topology shuffle",
        "order shuffle",
        "direction reversal",
        "meta-control ablation",
        "preservation ablation",
        "compression ablation",
        "recursion control",
        "complexity control",
        "linear matched control",
        "random relational graph",
        "three coherent counterfactuals",
        "three perturbation levels",
        "held-out structural damage"
    ],

    "interpretation_boundary":
        "Behavioral experiment only. "
        "Does not directly measure weights, activations, "
        "attention heads, latent vectors, biological myelination, "
        "training exposure, personal recognition or cross-user transfer."
}


spec_hash = sha_text(
    canonical(SPEC)
)

if SPEC_FILE.exists():

    old = json.loads(
        SPEC_FILE.read_text(
            encoding="utf-8"
        )
    )

    if sha_text(
        canonical(old)
    ) != spec_hash:

        raise SystemExit(
            "FROZEN SPECIFICATION MISMATCH."
        )

else:

    SPEC_FILE.write_text(
        json.dumps(
            SPEC,
            indent=2,
            ensure_ascii=False,
            sort_keys=True
        ),
        encoding="utf-8"
    )


SPEC_HASH_FILE.write_text(
    spec_hash + "\n",
    encoding="utf-8"
)


# =====================================================================
# FREEZE 2,400 UNIQUE PROMPTS
# =====================================================================

prompt_rows = []

for seed in range(N_SEEDS):

    for condition in CONDITIONS:

        p = prompt(
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


expected_prompt_hashes = {

    (
        r["seed"],
        r["condition"]
    ):
        r["prompt_sha256"]

    for r in prompt_rows
}


if PROMPTS_FILE.exists():

    old = {}

    with PROMPTS_FILE.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        for row in csv.DictReader(f):

            old[
                (
                    int(row["seed"]),
                    row["condition"]
                )
            ] = row["prompt_sha256"]

    if old != expected_prompt_hashes:

        raise SystemExit(
            "FROZEN PROMPT MISMATCH."
        )

else:

    with PROMPTS_FILE.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "condition",
                "prompt_sha256",
                "prompt"
            ]
        )

        w.writeheader()
        w.writerows(
            prompt_rows
        )


# =====================================================================
# LOAD CHECKPOINTS
# =====================================================================

records = {}

old_input = 0
old_output = 0


if RAW_FILE.exists():

    with RAW_FILE.open(
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

                old_input += int(
                    obj.get(
                        "input_tokens"
                    ) or 0
                )

                old_output += int(
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

    if cell_key(
        *cell
    ) not in records
]


print()
print("=" * 80)
print(EXPERIMENT)
print("=" * 80)

print("MODEL:", MODEL)
print("RUNNER SHA256:", runner_hash)
print("SPEC SHA256:", spec_hash)

print()
print("UNIQUE PROMPTS:", len(prompt_rows))
print("EXPECTED CALLS:", EXPECTED_CALLS)
print("EXISTING SUCCESS:", len(records))
print("PENDING:", len(pending))

print()
print(
    "EXISTING ESTIMATED COST:",
    f"${cost_estimate(old_input, old_output):.4f}"
)

print()
print("OUTPUT:")
print(OUT_DIR)
print()


# =====================================================================
# API WORKER
# =====================================================================

_thread = threading.local()


def get_client():

    if not hasattr(
        _thread,
        "client"
    ):

        _thread.client = OpenAI()

    return _thread.client


def execute_cell(
    seed,
    condition,
    rep
):

    global RUN_INPUT_TOKENS
    global RUN_OUTPUT_TOKENS

    p = prompt(
        condition,
        seed
    )

    key = cell_key(
        seed,
        condition,
        rep
    )

    for attempt in range(
        1,
        6
    ):

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

            record = {

                "status":
                    "success",

                "experiment":
                    EXPERIMENT,

                "cell_key":
                    key,

                "seed":
                    seed,

                "condition":
                    condition,

                "rep":
                    rep,

                "attempt":
                    attempt,

                "created_utc":
                    utc(),

                "elapsed_seconds":
                    time.time()
                    - started,

                "model_requested":
                    MODEL,

                "model_returned":
                    getattr(
                        response,
                        "model",
                        None
                    ),

                "response_id":
                    getattr(
                        response,
                        "id",
                        None
                    ),

                "prompt_sha256":
                    sha_text(p),

                "response_sha256":
                    sha_text(text),

                "input_tokens":
                    inp,

                "output_tokens":
                    out,

                "text":
                    text
            }

            append_jsonl(
                RAW_FILE,
                record
            )

            with PROGRESS_LOCK:

                RUN_INPUT_TOKENS += inp
                RUN_OUTPUT_TOKENS += out

            return True

        except Exception as e:

            append_jsonl(
                ERROR_FILE,
                {

                    "status":
                        "error",

                    "cell_key":
                        key,

                    "seed":
                        seed,

                    "condition":
                        condition,

                    "rep":
                        rep,

                    "attempt":
                        attempt,

                    "created_utc":
                        utc(),

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

if pending:

    print(
        f"Starting {len(pending)} pending calls "
        f"with {MAX_WORKERS} workers..."
    )

    print()

    success_now = 0
    unresolved = 0

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {

            executor.submit(
                execute_cell,
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
                success_now += 1
            else:
                unresolved += 1

            total_ok = (
                len(records)
                + success_now
            )

            if (
                success_now % 25 == 0
                or unresolved > 0
                or total_ok == EXPECTED_CALLS
            ):

                total_in = (
                    old_input
                    + RUN_INPUT_TOKENS
                )

                total_out = (
                    old_output
                    + RUN_OUTPUT_TOKENS
                )

                cost = cost_estimate(
                    total_in,
                    total_out
                )

                print(
                    f"OK {total_ok}/{EXPECTED_CALLS}"
                    f" | unresolved={unresolved}"
                    f" | input={total_in:,}"
                    f" | output={total_out:,}"
                    f" | est_cost=${cost:.4f}",
                    flush=True
                )

else:

    print(
        "All 9,600 cells already present."
    )


# =====================================================================
# RELOAD UNIQUE SUCCESSFUL CELLS
# =====================================================================

records = {}

with RAW_FILE.open(
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


# =====================================================================
# COMPLETENESS GATE
# =====================================================================

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


with MISSING_FILE.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    fields = [
        "seed",
        "condition",
        "rep",
        "cell_key"
    ]

    w = csv.DictWriter(
        f,
        fieldnames=fields
    )

    w.writeheader()

    if missing:
        w.writerows(
            missing
        )


if missing:

    print()
    print("=" * 80)
    print("EXPERIMENT INCOMPLETE")
    print("=" * 80)

    print(
        "SUCCESSFUL:",
        len(records)
    )

    print(
        "MISSING:",
        len(missing)
    )

    print()
    print(
        "RERUN THE SAME POWERSHELL BLOCK."
    )

    raise SystemExit(2)


print()
print("=" * 80)
print("COLLECTION COMPLETE — 9600 / 9600")
print("=" * 80)


# =====================================================================
# SCORE
# =====================================================================

scored = []

for seed, condition, rep in all_cells:

    obj = records[
        cell_key(
            seed,
            condition,
            rep
        )
    ]

    s = score(
        obj.get(
            "text",
            ""
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
            ),
    }

    row.update(s)

    scored.append(row)


with SCORED_FILE.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    fields = [
        "seed",
        "condition",
        "rep",
        "response_id",
        "response_sha256",
        *METRICS
    ]

    w = csv.DictWriter(
        f,
        fieldnames=fields
    )

    w.writeheader()
    w.writerows(scored)


# =====================================================================
# SEED AGGREGATION
# =====================================================================

seed_rows = []

for seed in range(N_SEEDS):

    for condition in CONDITIONS:

        subset = [

            r for r in scored

            if r["seed"] == seed

            and r[
                "condition"
            ] == condition
        ]

        if len(subset) != N_REPS:

            raise RuntimeError(
                "Unexpected repetition count."
            )

        row = {
            "seed":
                seed,
            "condition":
                condition
        }

        for metric in METRICS:

            row[metric] = (
                statistics.mean(
                    float(r[metric])
                    for r in subset
                )
            )

        seed_rows.append(row)


with SEED_FILE.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    fields = [
        "seed",
        "condition",
        *METRICS
    ]

    w = csv.DictWriter(
        f,
        fieldnames=fields
    )

    w.writeheader()
    w.writerows(seed_rows)


def values(
    condition,
    seed_filter=None
):

    rows = [

        r for r in seed_rows

        if r[
            "condition"
        ] == condition
    ]

    if seed_filter is not None:

        sf = set(seed_filter)

        rows = [

            r for r in rows

            if r["seed"] in sf
        ]

    rows.sort(
        key=lambda r:
            r["seed"]
    )

    return np.array(
        [
            float(
                r[
                    "MULTISCALE_SCORE"
                ]
            )
            for r in rows
        ],
        dtype=float
    )


# =====================================================================
# CONDITION RESULTS
# =====================================================================

condition_results = []

for condition in CONDITIONS:

    x = values(
        condition
    )

    condition_results.append({

        "condition":
            condition,

        "N_SEEDS":
            len(x),

        "mean_MULTISCALE_SCORE":
            float(
                np.mean(x)
            ),

        "median_MULTISCALE_SCORE":
            float(
                np.median(x)
            ),

        "sd_MULTISCALE_SCORE":
            float(
                np.std(
                    x,
                    ddof=1
                )
            ),

        "min":
            float(
                np.min(x)
            ),

        "max":
            float(
                np.max(x)
            )
    })


with CONDITION_FILE.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=list(
            condition_results[
                0
            ].keys()
        )
    )

    w.writeheader()
    w.writerows(
        condition_results
    )


# =====================================================================
# PAIRED PERMUTATION
# =====================================================================

def perm_greater(
    a,
    b,
    seed
):

    d = np.asarray(a) - np.asarray(b)

    observed = float(
        np.mean(d)
    )

    rng = (
        np.random
        .default_rng(seed)
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

    p = (
        exceed + 1
    ) / (
        PERMUTATIONS + 1
    )

    return observed, p


# =====================================================================
# HOLM
# =====================================================================

def holm(pvals):

    n = len(pvals)

    order_idx = sorted(
        range(n),
        key=lambda i:
            pvals[i]
    )

    adjusted = [
        None
    ] * n

    running = 0.0

    for rank, idx in enumerate(
        order_idx
    ):

        x = min(
            1.0,
            (
                n - rank
            )
            * pvals[idx]
        )

        running = max(
            running,
            x
        )

        adjusted[idx] = running

    return adjusted


# =====================================================================
# CONFIRMATORY CONTRASTS
# =====================================================================

contrast_results = []
pvals = []


for index, (
    label,
    cond_a,
    cond_b
) in enumerate(CONTRASTS):

    a = values(
        cond_a
    )

    b = values(
        cond_b
    )

    diff, p = perm_greater(
        a,
        b,
        MASTER_SEED
        + index * 17
    )

    try:

        wr = wilcoxon(
            a,
            b,
            alternative="greater"
        )

        wstat = float(
            wr.statistic
        )

        wp = float(
            wr.pvalue
        )

    except Exception:

        wstat = float("nan")
        wp = float("nan")

    contrast_results.append({

        "contrast":
            label,

        "condition_A":
            cond_a,

        "condition_B":
            cond_b,

        "N":
            len(a),

        "mean_A":
            float(
                np.mean(a)
            ),

        "mean_B":
            float(
                np.mean(b)
            ),

        "difference":
            diff,

        "permutation_p":
            p,

        "wilcoxon_stat":
            wstat,

        "wilcoxon_p":
            wp
    })

    pvals.append(p)


adjusted = holm(
    pvals
)

for row, adj in zip(
    contrast_results,
    adjusted
):

    row[
        "holm_adjusted_p"
    ] = adj

    row[
        "HOLM_GATE"
    ] = bool(
        row["difference"] > 0
        and adj < 0.05
    )


with CONTRAST_FILE.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=list(
            contrast_results[
                0
            ].keys()
        )
    )

    w.writeheader()
    w.writerows(
        contrast_results
    )


# =====================================================================
# SPLIT REPLICATION
# =====================================================================

split_tests = [

    (
        "TOPOLOGY",
        "A100_AUTHENTIC",
        "T_TOPOLOGY_SHUFFLED"
    ),

    (
        "DIRECTION",
        "A100_AUTHENTIC",
        "R_DIRECTION_REVERSED"
    ),

    (
        "PRESERVATION",
        "A100_AUTHENTIC",
        "P_PRESERVATION_ABLATED"
    ),

    (
        "HELDOUT_TOPOLOGY",
        "H_AUTHENTIC",
        "H_TOPOLOGY_DAMAGE"
    ),

    (
        "HELDOUT_DIRECTION",
        "H_AUTHENTIC",
        "H_DIRECTION_REVERSE"
    ),
]


splits = {
    "A": range(0, 40),
    "B": range(40, 80)
}


split_results = []

for split_name, seeds in splits.items():

    for i, (
        label,
        ca,
        cb
    ) in enumerate(split_tests):

        a = values(
            ca,
            seeds
        )

        b = values(
            cb,
            seeds
        )

        diff, p = perm_greater(
            a,
            b,
            MASTER_SEED
            + i * 101
            + (
                10000
                if split_name == "B"
                else 0
            )
        )

        split_results.append({

            "split":
                split_name,

            "contrast":
                label,

            "N":
                len(a),

            "difference":
                diff,

            "p":
                p,

            "positive":
                bool(
                    diff > 0
                ),

            "nominal_gate":
                bool(
                    diff > 0
                    and p < 0.05
                )
        })


with SPLIT_FILE.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=list(
            split_results[
                0
            ].keys()
        )
    )

    w.writeheader()
    w.writerows(
        split_results
    )


# =====================================================================
# DOSE RESPONSE
# =====================================================================

dose_conditions = [

    (0.25, "A25_DOSE"),
    (0.50, "A50_DOSE"),
    (0.75, "A75_DOSE"),
    (1.00, "A100_AUTHENTIC"),
]


dose_results = []

for level, condition in dose_conditions:

    x = values(condition)

    dose_results.append({

        "dose":
            level,

        "condition":
            condition,

        "mean":
            float(
                np.mean(x)
            ),

        "median":
            float(
                np.median(x)
            )
    })


rhos = []

for seed in range(N_SEEDS):

    ys = []

    for level, condition in dose_conditions:

        row = next(
            r for r in seed_rows
            if r["seed"] == seed
            and r[
                "condition"
            ] == condition
        )

        ys.append(
            float(
                row[
                    "MULTISCALE_SCORE"
                ]
            )
        )

    rho, _ = spearmanr(
        [
            0.25,
            0.50,
            0.75,
            1.00
        ],
        ys
    )

    if not np.isnan(rho):
        rhos.append(
            float(rho)
        )


mean_rho = float(
    np.mean(rhos)
)

dose_gate = bool(
    mean_rho > 0
)


with DOSE_FILE.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=[
            "dose",
            "condition",
            "mean",
            "median"
        ]
    )

    w.writeheader()
    w.writerows(
        dose_results
    )

    f.write(
        "\n"
        f"MEAN_PER_SEED_SPEARMAN,{mean_rho}\n"
        f"DOSE_GATE,{dose_gate}\n"
    )


# =====================================================================
# SURFACE EQUIVALENCE
# =====================================================================

def bootstrap_diff(
    a,
    b,
    seed
):

    d = (
        np.asarray(a)
        - np.asarray(b)
    )

    rng = (
        np.random
        .default_rng(seed)
    )

    sims = np.empty(
        BOOTSTRAPS
    )

    for i in range(
        BOOTSTRAPS
    ):

        sample = rng.choice(
            d,
            size=len(d),
            replace=True
        )

        sims[i] = float(
            np.mean(sample)
        )

    return (
        float(
            np.mean(d)
        ),
        float(
            np.quantile(
                sims,
                0.025
            )
        ),
        float(
            np.quantile(
                sims,
                0.975
            )
        )
    )


surface_results = []

for i, condition in enumerate([

    "S1_SURFACE_PARAPHRASE",
    "S2_SURFACE_PARAPHRASE",
    "S3_SURFACE_PARAPHRASE",

]):

    a = values(
        "A100_AUTHENTIC"
    )

    b = values(
        condition
    )

    diff, lo, hi = bootstrap_diff(
        a,
        b,
        MASTER_SEED
        + 500
        + i
    )

    equivalent = bool(
        lo
        > -SURFACE_EQUIVALENCE_MARGIN
        and hi
        < SURFACE_EQUIVALENCE_MARGIN
    )

    surface_results.append({

        "comparison":
            "A100_vs_"
            + condition,

        "mean_difference":
            diff,

        "CI95_low":
            lo,

        "CI95_high":
            hi,

        "equivalence_margin":
            SURFACE_EQUIVALENCE_MARGIN,

        "EQUIVALENT":
            equivalent
    })


with SURFACE_FILE.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=list(
            surface_results[
                0
            ].keys()
        )
    )

    w.writeheader()
    w.writerows(
        surface_results
    )


# =====================================================================
# PERTURBATION
# =====================================================================

lookup = {
    x["contrast"]: x
    for x in contrast_results
}


perturb_results = []

for level in [
    "MILD",
    "MEDIUM",
    "STRONG"
]:

    row = lookup[
        "PERTURB_"
        + level
    ]

    perturb_results.append({

        "level":
            level,

        "authentic_mean":
            row["mean_A"],

        "shuffled_mean":
            row["mean_B"],

        "difference":
            row["difference"],

        "holm_adjusted_p":
            row[
                "holm_adjusted_p"
            ],

        "GATE":
            row["HOLM_GATE"]
    })


with PERTURB_FILE.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=list(
            perturb_results[
                0
            ].keys()
        )
    )

    w.writeheader()
    w.writerows(
        perturb_results
    )


# =====================================================================
# CLASSIFICATION
# =====================================================================

gates = {

    row["contrast"]:
        bool(
            row[
                "HOLM_GATE"
            ]
        )

    for row in contrast_results
}


surface_pass = sum(
    int(
        row[
            "EQUIVALENT"
        ]
    )
    for row in surface_results
)


perturb_pass = sum(
    int(
        row["GATE"]
    )
    for row in perturb_results
)


def replicated(name):

    rows = [

        x for x in split_results

        if x[
            "contrast"
        ] == name
    ]

    return bool(
        len(rows) == 2
        and all(
            x[
                "nominal_gate"
            ]
            for x in rows
        )
    )


topology_replication = replicated(
    "TOPOLOGY"
)

direction_replication = replicated(
    "DIRECTION"
)


strong_conditions = [

    gates["TOPOLOGY"],
    gates["ORDER"],
    gates["DIRECTION"],
    gates["PRESERVATION"],

    gates[
        "COUNTERFACTUAL_ARCHIVE"
    ],

    gates[
        "HELDOUT_TOPOLOGY"
    ],

    gates[
        "HELDOUT_DIRECTION"
    ],

    perturb_pass >= 2,

    surface_pass >= 2,

    dose_gate,

    topology_replication,

    direction_replication,
]


if all(
    strong_conditions
):

    classification = (
        "STRONG_MULTISCALE_BEHAVIORAL_FOREST_PATTERN"
    )

elif (
    gates["TOPOLOGY"]
    and gates["DIRECTION"]
    and gates[
        "HELDOUT_TOPOLOGY"
    ]
):

    classification = (
        "PARTIAL_MULTISCALE_STRUCTURAL_FOREST_PATTERN"
    )

else:

    classification = (
        "MULTISCALE_FOREST_HYPOTHESIS_NOT_SUPPORTED"
    )


# =====================================================================
# FINAL TOKEN COST
# =====================================================================

input_tokens = sum(
    int(
        r.get(
            "input_tokens"
        ) or 0
    )
    for r in records.values()
)

output_tokens = sum(
    int(
        r.get(
            "output_tokens"
        ) or 0
    )
    for r in records.values()
)

total_cost = cost_estimate(
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
        utc(),

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

    "confirmatory_contrasts":
        contrast_results,

    "split_replication":
        split_results,

    "dose_response": {

        "mean_per_seed_spearman":
            mean_rho,

        "gate":
            dose_gate
    },

    "surface_invariance":
        surface_results,

    "surface_pass_count":
        surface_pass,

    "perturbation":
        perturb_results,

    "perturbation_pass_count":
        perturb_pass,

    "topology_replication":
        topology_replication,

    "direction_replication":
        direction_replication,

    "classification":
        classification,

    "token_usage": {

        "input_tokens":
            input_tokens,

        "output_tokens":
            output_tokens
    },

    "estimated_cost_usd":
        total_cost,

    "interpretation_boundary":
        "Prospective behavioral evidence only. "
        "Does not establish biological myelination, "
        "weight modification, hidden activation geometry, "
        "training exposure, cross-user propagation or "
        "personal recognition."
}


FINAL_FILE.write_text(
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
print("=" * 80)
print("V5 FOREST FORENSIC — FINAL RESULTS")
print("=" * 80)

print()

for row in condition_results:

    print(
        f"{row['condition']:<38}"
        f" mean={row['mean_MULTISCALE_SCORE']:.4f}"
        f" median={row['median_MULTISCALE_SCORE']:.4f}"
    )


print()
print("-" * 80)
print("CONFIRMATORY CONTRASTS")
print("-" * 80)

for row in contrast_results:

    print(
        f"{row['contrast']:<28}"
        f" diff={row['difference']:+.4f}"
        f" raw_p={row['permutation_p']:.6g}"
        f" holm_p={row['holm_adjusted_p']:.6g}"
        f" gate={row['HOLM_GATE']}"
    )


print()
print("-" * 80)
print("SURFACE INVARIANCE")
print("-" * 80)

for row in surface_results:

    print(
        f"{row['comparison']:<42}"
        f" diff={row['mean_difference']:+.4f}"
        f" CI=[{row['CI95_low']:+.4f},"
        f"{row['CI95_high']:+.4f}]"
        f" equiv={row['EQUIVALENT']}"
    )


print()
print("-" * 80)
print("PERTURBATION")
print("-" * 80)

for row in perturb_results:

    print(
        f"{row['level']:<10}"
        f" diff={row['difference']:+.4f}"
        f" holm_p={row['holm_adjusted_p']:.6g}"
        f" gate={row['GATE']}"
    )


print()
print("-" * 80)
print("REPLICATION")
print("-" * 80)

print(
    "TOPOLOGY BOTH HALVES:",
    topology_replication
)

print(
    "DIRECTION BOTH HALVES:",
    direction_replication
)


print()
print("-" * 80)
print("DOSE")
print("-" * 80)

print(
    "MEAN PER-SEED SPEARMAN:",
    f"{mean_rho:.6f}"
)

print(
    "DOSE GATE:",
    dose_gate
)


print()
print("=" * 80)

print(
    "FINAL CLASSIFICATION:",
    classification
)

print("=" * 80)

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
    f"${total_cost:.4f}"
)

print()
print(
    "FINAL SUMMARY:"
)

print(
    FINAL_FILE
)

print()
print(
    "ALL 9,600 CELLS COMPLETE."
)
print()
