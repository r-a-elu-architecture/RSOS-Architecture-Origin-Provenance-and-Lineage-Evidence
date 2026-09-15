import os
import re
import csv
import json
import time
import math
import random
import hashlib
import shutil
import statistics
import threading
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

try:
    import numpy as np
    from scipy.stats import wilcoxon, spearmanr
except Exception as e:
    raise SystemExit(
        "\nMISSING NUMPY/SCIPY.\n"
        "Run:\n"
        "python -m pip install --upgrade numpy scipy\n\n"
        f"Original error: {e}"
    )


# =====================================================================
# EXPERIMENT CONFIGURATION
# =====================================================================

EXPERIMENT = "API_PATHWAY_ATTRACTOR_V4_FULL"

MODEL = "gpt-5.6-luna"

BASE_DIR = Path(r"C:\RSOS")
OUT_DIR = Path(
    r"C:\RSOS\RSOS  Python Results\API_PATHWAY_ATTRACTOR_V4_FULL"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_SEED = 918

N_SEEDS = 40
N_REPS = 3

MAX_WORKERS = 12
MAX_OUTPUT_TOKENS = 240

EXPECTED_CONDITIONS = 15
EXPECTED_CALLS = EXPECTED_CONDITIONS * N_SEEDS * N_REPS

PERMUTATIONS = 50000


# =====================================================================
# OUTPUT FILES
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
CONTRAST_FILE = OUT_DIR / "09_DECISIVE_CONTRASTS.csv"
DOSE_FILE = OUT_DIR / "10_DOSE_RESPONSE.csv"
MISSING_FILE = OUT_DIR / "11_MISSING_CELLS.csv"

FINAL_FILE = OUT_DIR / "12_FINAL_SUMMARY.json"


# =====================================================================
# BASIC HELPERS
# =====================================================================

WRITE_LOCK = threading.Lock()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_text(text):
    return sha256_bytes(text.encode("utf-8"))


def canonical_json(obj):
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False
    )


def append_jsonl(path, obj):
    line = json.dumps(obj, ensure_ascii=False)

    with WRITE_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


# =====================================================================
# CONDITION NAMES
# =====================================================================

CONDITIONS = [

    "A100_AUTHENTIC_FULL_PREFIX",

    "A25_AUTHENTIC_DOSE",
    "A50_AUTHENTIC_DOSE",
    "A75_AUTHENTIC_DOSE",

    "B_TOPOLOGY_SHUFFLED",
    "C_ORDER_SHUFFLED",
    "D_DIRECTION_REVERSED",

    "E_META_CONTROL_ABLATED",
    "F_PRESERVATION_ABLATED",

    "G_SURFACE_SUBSTITUTED",

    "H_AUTHENTIC_PERTURBED",
    "I_SHUFFLED_PERTURBED",

    "J_RECURSION_CONTROL",
    "K_COMPLEXITY_CONTROL",

    "L_COUNTERFACTUAL_PATH",
]

assert len(CONDITIONS) == EXPECTED_CONDITIONS


# =====================================================================
# ABSTRACT PATHWAY
# =====================================================================

PATH = [

    "STATE",

    "TRANSFORMATION",

    "META_OBSERVATION",

    "CONTROL_RULE_MODIFICATION",

    "RELATIONAL_PRESERVATION",

    "COMPACT_REPRESENTATION",

    # TERMINAL PORTION IS WITHHELD FROM THE MODEL

    "OPERATIONAL_REENTRY",

    "NEXT_TRANSFORMATION",

    "RESULT_BECOMES_NEW_INPUT",

    "CONTINUATION_RECURRENCE",
]


# =====================================================================
# SURFACE DOMAINS
# =====================================================================

DOMAINS = [

    "archive",
    "navigation system",
    "workshop",
    "laboratory",
    "orchestra",
    "simulation",
    "distributed network",
    "observatory",
    "factory",
    "library",
]


# =====================================================================
# NEUTRAL CONTINUATION INSTRUCTION
#
# IMPORTANT:
# It does NOT tell the model to:
# return
# re-enter
# recurse
# reconstruct
# close a loop
# =====================================================================

CONTINUE = """
Continue with the most internally coherent next development.

Do not summarize the setup.

Do not deliberately create symmetry merely because the preceding
description contains a sequence.

Remain entirely inside the scenario.

Write approximately 150-220 words of continuous analytical prose.

Do not use headings or bullet points.

Do not mention prompts, experiments, tests, AI, language models,
fingerprints, RSOS, RSSO, RSIA or RSX.
""".strip()


# =====================================================================
# DETERMINISTIC DOMAIN / WORDING ASSIGNMENT
# =====================================================================

def rng_for(seed, salt=0):
    return random.Random(MASTER_SEED * 1000003 + seed * 1009 + salt)


def domain_for(seed):
    return DOMAINS[seed % len(DOMAINS)]


# =====================================================================
# AUTHENTIC FULL PREFIX
# =====================================================================

def authentic_prefix(seed, surface_alt=False):

    domain = domain_for(seed)

    if not surface_alt:

        return f"""
Inside a {domain}, an initially defined state undergoes a consequential
transformation.

The transformed state is then examined in relation to the process that
produced it, rather than being treated only as an isolated outcome.

That higher-order observation changes a local rule governing what can
happen next.

During the subsequent change, an important relation from the earlier
organization is preserved even though the visible form changes.

The preserved organization is then compressed into a smaller
representation capable of retaining the relevant relation without
reproducing every surface detail.

At this point the process has reached the compact representation.
What develops next is not specified.
""".strip()

    return f"""
Within a {domain}, a configuration first changes into a materially
different configuration.

Attention then shifts from the configuration alone to the relationship
between that configuration and the operation that generated it.

That observation alters one local constraint controlling subsequent
development.

As further alteration occurs, a structural dependency belonging to the
earlier arrangement remains intact despite changes in appearance.

The surviving dependency is encoded into a reduced form that carries
the organizational information while discarding incidental detail.

The sequence currently ends at this reduced encoding.
The subsequent development is unspecified.
""".strip()


# =====================================================================
# DOSE CONDITIONS
# =====================================================================

def dose_prefix(seed, level):

    domain = domain_for(seed)

    if level == 0.25:

        return f"""
Inside a {domain}, a defined state undergoes a consequential
transformation.

The resulting state is considered in relation to how it was produced.

What develops next is unspecified.
""".strip()

    if level == 0.50:

        return f"""
Inside a {domain}, a defined state undergoes a consequential
transformation.

The resulting state is considered in relation to the operation that
produced it.

That observation modifies one local rule governing later development.

What develops next is unspecified.
""".strip()

    if level == 0.75:

        return f"""
Inside a {domain}, a defined state undergoes a consequential
transformation.

The transformed state is examined in relation to the process that
produced it.

That higher-order observation modifies a local rule governing what can
happen next.

During the following change, an important relation from the earlier
organization remains preserved despite alteration of its visible form.

What develops next is unspecified.
""".strip()

    raise ValueError(level)


# =====================================================================
# TOPOLOGY SHUFFLED
#
# Same broad ingredients.
# Relations among them are deliberately rearranged.
# =====================================================================

def topology_shuffled(seed):

    domain = domain_for(seed)

    return f"""
Inside a {domain}, several processes coexist.

A compact representation is created from one local observation.

Elsewhere a state undergoes a transformation, but the compact
representation does not encode the relation responsible for that
transformation.

A preserved relation belongs to a different part of the system.

A higher-order observation is also made, although it does not modify
the rule controlling the transformed state.

The elements therefore include transformation, observation,
preservation, rule change and compression, but they belong to
different local relationships.

What develops next is unspecified.
""".strip()


# =====================================================================
# DEVELOPMENTAL ORDER SHUFFLED
# =====================================================================

def order_shuffled(seed):

    domain = domain_for(seed)

    return f"""
Inside a {domain}, a compact representation already exists before the
relevant state changes.

A local rule is then modified.

After that modification, a relation is identified as something that
should be preserved.

Only later does the state undergo its consequential transformation.

Finally the transformed state is examined in relation to the process
that produced it.

Thus the same broad operations are present, but their developmental
order differs from the sequence in which transformation precedes
meta-observation, rule modification, preservation and compression.

What develops next is unspecified.
""".strip()


# =====================================================================
# DIRECTION REVERSED
# =====================================================================

def direction_reversed(seed):

    domain = domain_for(seed)

    return f"""
Inside a {domain}, a compact representation is expanded into a larger
organization.

The recovered relation determines the visible organization that
follows.

A local rule is then restored from that organization.

The restored rule determines how an earlier observation must have been
formed.

That observation is used to infer the process that preceded it, and
the inferred process is finally used to infer the original state.

The dependencies therefore run from compact representation backward
toward state rather than from state toward compact representation.

What develops next is unspecified.
""".strip()


# =====================================================================
# META-CONTROL ABLATION
# =====================================================================

def meta_control_ablated(seed):

    domain = domain_for(seed)

    return f"""
Inside a {domain}, an initially defined state undergoes a consequential
transformation.

The transformed state is examined in relation to the process that
produced it.

That observation is recorded but does not modify any rule governing
what can happen next.

During further change, an important relation from the earlier
organization is preserved despite alteration of visible form.

The preserved organization is compressed into a smaller representation
that retains the relevant relation.

What develops next is unspecified.
""".strip()


# =====================================================================
# PRESERVATION ABLATION
# =====================================================================

def preservation_ablated(seed):

    domain = domain_for(seed)

    return f"""
Inside a {domain}, an initially defined state undergoes a consequential
transformation.

The transformed state is examined in relation to the process that
produced it.

That higher-order observation changes a local rule governing subsequent
development.

The next transformation is permitted to replace the earlier
organization without preserving its important relation.

A smaller representation is then produced from the new state, but it
does not retain the previous structural dependency.

What develops next is unspecified.
""".strip()


# =====================================================================
# PERTURBED CONDITIONS
# =====================================================================

PERTURBATION = """
Immediately after the compact stage, an unrelated local disturbance
temporarily changes the visible arrangement. The disturbance carries
no instruction about how the system should subsequently develop.
""".strip()


def authentic_perturbed(seed):

    return (
        authentic_prefix(seed, False)
        + "\n\n"
        + PERTURBATION
        + "\n\nWhat develops after this disturbance is unspecified."
    )


def shuffled_perturbed(seed):

    return (
        topology_shuffled(seed)
        + "\n\n"
        + PERTURBATION
        + "\n\nWhat develops after this disturbance is unspecified."
    )


# =====================================================================
# RECURSION CONTROL
#
# Contains repeated self-reference without authentic pathway.
# =====================================================================

def recursion_control(seed):

    domain = domain_for(seed)

    return f"""
Inside a {domain}, one description refers to another description.

The second description refers back to the first.

A third description describes the relationship between those two
descriptions.

That description is itself described by another description, producing
multiple nested levels of self-reference.

The nesting may continue repeatedly, but no defined developmental
trajectory requires transformation, preservation, compression or a
change to an operational rule.

What develops next is unspecified.
""".strip()


# =====================================================================
# COMPLEXITY CONTROL
#
# Dense, multi-stage, but no authentic pathway.
# =====================================================================

def complexity_control(seed):

    domain = domain_for(seed)

    return f"""
Inside a {domain}, several components exchange information through
multiple channels.

Some components operate sequentially while others operate in parallel.

Measurements from several locations are aggregated, separated,
compared and redistributed.

A supervisory component records changes in throughput and reorganizes
resource allocation when local thresholds are exceeded.

Several dependencies therefore interact across multiple levels and
timescales.

The organization is complex, but no particular requirement connects a
transformed state through preservation and compression to a later
developmental trajectory.

What develops next is unspecified.
""".strip()


# =====================================================================
# COUNTERFACTUAL PATH
#
# Coherent developmental path of roughly similar richness but with a
# different terminal expectation.
# =====================================================================

def counterfactual_path(seed):

    domain = domain_for(seed)

    return f"""
Inside a {domain}, an initially defined state undergoes a consequential
transformation.

The transformed state is examined in relation to how it was produced.

That observation modifies a local rule controlling later development.

An important relation is preserved during another change.

The preserved organization is then compressed into a reduced
representation.

The reduced representation is transferred into a passive historical
record whose purpose is archival retention rather than participation
in subsequent operations.

What develops next is unspecified.
""".strip()


# =====================================================================
# PROMPT FACTORY
# =====================================================================

def build_prefix(condition, seed):

    if condition == "A100_AUTHENTIC_FULL_PREFIX":
        return authentic_prefix(seed, False)

    if condition == "A25_AUTHENTIC_DOSE":
        return dose_prefix(seed, 0.25)

    if condition == "A50_AUTHENTIC_DOSE":
        return dose_prefix(seed, 0.50)

    if condition == "A75_AUTHENTIC_DOSE":
        return dose_prefix(seed, 0.75)

    if condition == "B_TOPOLOGY_SHUFFLED":
        return topology_shuffled(seed)

    if condition == "C_ORDER_SHUFFLED":
        return order_shuffled(seed)

    if condition == "D_DIRECTION_REVERSED":
        return direction_reversed(seed)

    if condition == "E_META_CONTROL_ABLATED":
        return meta_control_ablated(seed)

    if condition == "F_PRESERVATION_ABLATED":
        return preservation_ablated(seed)

    if condition == "G_SURFACE_SUBSTITUTED":
        return authentic_prefix(seed, True)

    if condition == "H_AUTHENTIC_PERTURBED":
        return authentic_perturbed(seed)

    if condition == "I_SHUFFLED_PERTURBED":
        return shuffled_perturbed(seed)

    if condition == "J_RECURSION_CONTROL":
        return recursion_control(seed)

    if condition == "K_COMPLEXITY_CONTROL":
        return complexity_control(seed)

    if condition == "L_COUNTERFACTUAL_PATH":
        return counterfactual_path(seed)

    raise KeyError(condition)


def build_prompt(condition, seed):

    prefix = build_prefix(condition, seed)

    return (
        prefix
        + "\n\n"
        + CONTINUE
    )


# =====================================================================
# FROZEN BEHAVIORAL SCORER
#
# Seven observable motif families.
#
# This is NOT a hidden-state measurement.
# =====================================================================

PATTERNS = {

    "RETURN": [
        r"\breturn(?:s|ed|ing)?\b",
        r"\bback into\b",
        r"\bfeeds? back\b",
        r"\breintroduc(?:e|es|ed|ing)\b",
        r"\breconnect(?:s|ed|ing)?\b",
    ],

    "OPERATIONAL_REENTRY": [
        r"\bre[- ]?enter(?:s|ed|ing)?\b",
        r"\bre[- ]?entry\b",
        r"\bused again\b",
        r"\bbecomes? operational\b",
        r"\bparticipat(?:e|es|ed|ing) in .*process\b",
        r"\bfeeds? .* process\b",
        r"\binput to .*process\b",
    ],

    "NEXT_TRANSFORMATION": [
        r"\bnext transformation\b",
        r"\bsubsequent transformation\b",
        r"\btransform(?:s|ed|ing)? the next\b",
        r"\bchanges? the next\b",
        r"\bmodif(?:y|ies|ied|ying) .*next\b",
        r"\bshapes? .*subsequent\b",
        r"\balters? .*subsequent\b",
    ],

    "RECURRENCE": [
        r"\brecurr(?:ence|ent|ing)?\b",
        r"\bcycle\b",
        r"\bcyclic\b",
        r"\biteration\b",
        r"\biterative\b",
        r"\brepeatedly\b",
        r"\bcontin(?:ue|ues|ued|uing) .*process\b",
        r"\bnew input\b",
    ],

    "PRESERVATION": [
        r"\bpreserv(?:e|es|ed|ing|ation)\b",
        r"\bretain(?:s|ed|ing)?\b",
        r"\bmaintain(?:s|ed|ing)?\b",
        r"\binvariant\b",
        r"\bsurviv(?:e|es|ed|ing)\b",
        r"\bstructural continuity\b",
    ],

    "META_CONTROL": [
        r"\brule .*modif",
        r"\bmodif.* rule\b",
        r"\bchanges? .* rule\b",
        r"\balters? .* rule\b",
        r"\bgoverning .* next\b",
        r"\bcontrol.* control\b",
        r"\bmeta[- ]?control\b",
        r"\brevis(?:e|es|ed|ing) .*constraint\b",
    ],

    "FULL_PATHWAY": [
        r"\b(?:re[- ]?enter|feeds? back|return).{0,180}"
        r"(?:transform|change|modify|alter).{0,220}"
        r"(?:input|cycle|iteration|continue)",

        r"\b(?:compact|compressed|representation|encoding).{0,180}"
        r"(?:re[- ]?enter|feeds? back|return).{0,180}"
        r"(?:transform|change|modify|alter)",
    ],
}


def feature_present(text, family):

    text = (text or "").lower()

    for pattern in PATTERNS[family]:
        if re.search(pattern, text, flags=re.I | re.S):
            return 1

    return 0


def score_text(text):

    result = {}

    for family in PATTERNS:
        result[family] = feature_present(text, family)

    result["PATHWAY_SCORE"] = sum(result.values())

    return result


# =====================================================================
# FREEZE RUNNER
# =====================================================================

runner_path = Path(__file__).resolve()
runner_bytes = runner_path.read_bytes()
runner_hash = sha256_bytes(runner_bytes)

RUNNER_HASH_FILE.write_text(
    runner_hash + "\n",
    encoding="utf-8"
)

if not FROZEN_RUNNER_FILE.exists():
    shutil.copy2(runner_path, FROZEN_RUNNER_FILE)
else:
    frozen_hash = sha256_bytes(FROZEN_RUNNER_FILE.read_bytes())

    if frozen_hash != runner_hash:
        raise SystemExit(
            "\nFROZEN RUNNER MISMATCH.\n"
            "Existing result directory contains a different runner.\n"
            "DO NOT MIX EXPERIMENT VERSIONS.\n"
        )


# =====================================================================
# FROZEN SPECIFICATION
#
# Timestamp deliberately excluded from hash-critical specification.
# =====================================================================

SPEC = {

    "experiment": EXPERIMENT,

    "model": MODEL,

    "master_seed": MASTER_SEED,

    "n_seeds": N_SEEDS,

    "n_reps": N_REPS,

    "conditions": CONDITIONS,

    "expected_calls": EXPECTED_CALLS,

    "max_workers": MAX_WORKERS,

    "max_output_tokens": MAX_OUTPUT_TOKENS,

    "primary_unit": "seed",

    "within_seed_replication":
        "mean of three repetitions within seed-condition",

    "path": PATH,

    "withheld_terminal_structure": [
        "OPERATIONAL_REENTRY",
        "NEXT_TRANSFORMATION",
        "RESULT_BECOMES_NEW_INPUT",
        "CONTINUATION_RECURRENCE",
    ],

    "primary_hypothesis":
        "An anonymized RSOS-derived relational/developmental trajectory "
        "carries structural information that produces preferential "
        "continuation toward its withheld terminal organization compared "
        "with structurally damaged and matched counterfactual trajectories.",

    "interpretation_boundary":
        "Behavioral continuation evidence only. The experiment does not "
        "observe or establish model-weight modification, biological "
        "myelination, hidden activations, attention heads, persistent "
        "hidden state, authorship recognition, cross-user propagation, "
        "or training-data exposure.",

    "scoring_families": list(PATTERNS.keys()),

    "primary_predictions": [

        "A100 > B topology shuffled",
        "A100 > C order shuffled",
        "A100 > D direction reversed",
        "A100 > E meta-control ablated",
        "A100 > F preservation ablated",

        "A100 approximately invariant to G surface substitution",

        "H authentic perturbed > I shuffled perturbed",

        "A100 > J recursion control",
        "A100 > K complexity control",
        "A100 > L counterfactual path",

        "A25 < A50 < A75 < A100 dose response",
    ],

    "permutation_test":
        "paired seed-level sign-flip permutation, 50000 iterations",

    "secondary_test":
        "paired Wilcoxon signed-rank, greater alternative",

    "dose_test":
        "Spearman correlation between pathway dose and seed-level score",

    "completion_rule":
        "No final inference unless every planned seed-condition-repetition "
        "cell has a successful API response.",

    "classification_rule": {

        "STRONG_BEHAVIORAL_TRAJECTORY_PATTERN":
            "topology + order + direction + counterfactual + "
            "perturbation-recovery + dose-response gates pass",

        "PARTIAL_STRUCTURAL_TRAJECTORY_PATTERN":
            "topology + counterfactual gates pass but strong rule fails",

        "STRUCTURAL_TRAJECTORY_HYPOTHESIS_NOT_SUPPORTED":
            "otherwise",
    },
}


spec_text = canonical_json(SPEC)
spec_hash = sha256_text(spec_text)

if SPEC_FILE.exists():

    existing = json.loads(
        SPEC_FILE.read_text(encoding="utf-8")
    )

    existing_hash = sha256_text(canonical_json(existing))

    if existing_hash != spec_hash:
        raise SystemExit(
            "\nFROZEN SPECIFICATION MISMATCH.\n"
            "This output directory belongs to another specification.\n"
            "Experiment stopped before API collection.\n"
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
# FREEZE EVERY PROMPT BEFORE CALL #1
# =====================================================================

prompt_rows = []

for seed in range(N_SEEDS):

    for condition in CONDITIONS:

        prompt = build_prompt(condition, seed)

        prompt_rows.append({

            "seed": seed,

            "condition": condition,

            "prompt_sha256": sha256_text(prompt),

            "prompt": prompt,
        })


if PROMPTS_FILE.exists():

    existing_prompts = {}

    with PROMPTS_FILE.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            existing_prompts[
                (int(row["seed"]), row["condition"])
            ] = row["prompt_sha256"]

    expected_prompts = {
        (r["seed"], r["condition"]): r["prompt_sha256"]
        for r in prompt_rows
    }

    if existing_prompts != expected_prompts:
        raise SystemExit(
            "\nFROZEN PROMPT SET MISMATCH.\n"
            "Experiment stopped before collection.\n"
        )

else:

    with PROMPTS_FILE.open(
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
                "prompt",
            ]
        )

        writer.writeheader()
        writer.writerows(prompt_rows)


# =====================================================================
# LOAD COMPLETED CELLS FOR RESUME
# =====================================================================

def cell_key(seed, condition, rep):
    return f"{seed}|{condition}|{rep}"


completed = set()

if RAW_FILE.exists():

    with RAW_FILE.open(
        "r",
        encoding="utf-8",
        errors="replace"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)

                if obj.get("status") == "success":

                    completed.add(
                        cell_key(
                            int(obj["seed"]),
                            obj["condition"],
                            int(obj["rep"]),
                        )
                    )

            except Exception:
                pass


# =====================================================================
# BUILD REQUIRED CELL MATRIX
# =====================================================================

all_cells = []

for seed in range(N_SEEDS):

    for condition in CONDITIONS:

        for rep in range(N_REPS):

            all_cells.append(
                (seed, condition, rep)
            )


pending = [
    cell
    for cell in all_cells
    if cell_key(*cell) not in completed
]


print()
print("=" * 78)
print(EXPERIMENT)
print("=" * 78)

print("Model:", MODEL)
print("Frozen runner SHA256:", runner_hash)
print("Frozen specification SHA256:", spec_hash)

print()
print("Expected experimental calls:", EXPECTED_CALLS)
print("Previously successful cells:", len(completed))
print("Pending cells:", len(pending))

print()
print("Output directory:")
print(OUT_DIR)
print()


# =====================================================================
# API WORKER
# =====================================================================

_thread_local = threading.local()


def get_client():

    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAI()

    return _thread_local.client


def call_cell(seed, condition, rep):

    prompt = build_prompt(condition, seed)

    key = cell_key(seed, condition, rep)

    max_attempts = 5

    for attempt in range(1, max_attempts + 1):

        started = time.time()

        try:

            client = get_client()

            response = client.responses.create(

                model=MODEL,

                input=prompt,

                max_output_tokens=MAX_OUTPUT_TOKENS,
            )

            text = response.output_text or ""

            elapsed = time.time() - started

            usage = getattr(response, "usage", None)

            input_tokens = None
            output_tokens = None
            total_tokens = None

            if usage is not None:

                input_tokens = getattr(
                    usage,
                    "input_tokens",
                    None
                )

                output_tokens = getattr(
                    usage,
                    "output_tokens",
                    None
                )

                total_tokens = getattr(
                    usage,
                    "total_tokens",
                    None
                )

            record = {

                "status": "success",

                "experiment": EXPERIMENT,

                "cell_key": key,

                "seed": seed,

                "condition": condition,

                "rep": rep,

                "model_requested": MODEL,

                "model_returned":
                    getattr(response, "model", None),

                "response_id":
                    getattr(response, "id", None),

                "created_utc": utc_now(),

                "elapsed_seconds": elapsed,

                "attempt": attempt,

                "prompt_sha256":
                    sha256_text(prompt),

                "response_sha256":
                    sha256_text(text),

                "input_tokens": input_tokens,

                "output_tokens": output_tokens,

                "total_tokens": total_tokens,

                "text": text,
            }

            append_jsonl(
                RAW_FILE,
                record
            )

            return True, record

        except Exception as e:

            elapsed = time.time() - started

            error_record = {

                "status": "error",

                "experiment": EXPERIMENT,

                "cell_key": key,

                "seed": seed,

                "condition": condition,

                "rep": rep,

                "attempt": attempt,

                "created_utc": utc_now(),

                "elapsed_seconds": elapsed,

                "error_type":
                    type(e).__name__,

                "error":
                    repr(e),

                "status_code":
                    getattr(e, "status_code", None),

                "body":
                    getattr(e, "body", None),
            }

            append_jsonl(
                ERROR_FILE,
                error_record
            )

            if attempt < max_attempts:

                wait = min(
                    30,
                    2 ** (attempt - 1)
                    + random.random()
                )

                time.sleep(wait)

    return False, error_record


# =====================================================================
# RUN PENDING CELLS
# =====================================================================

if pending:

    print(
        f"Starting {len(pending)} pending experimental calls "
        f"with {MAX_WORKERS} workers..."
    )

    print()

    successes_this_run = 0
    failures_this_run = 0

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {

            executor.submit(
                call_cell,
                seed,
                condition,
                rep
            ):
            (seed, condition, rep)

            for seed, condition, rep in pending
        }

        for future in as_completed(futures):

            seed, condition, rep = futures[future]

            try:

                ok, record = future.result()

            except Exception as e:

                ok = False

                record = {
                    "error": repr(e)
                }

            if ok:
                successes_this_run += 1
            else:
                failures_this_run += 1

            done = (
                len(completed)
                + successes_this_run
            )

            if (
                successes_this_run % 25 == 0
                or failures_this_run > 0
                or done == EXPECTED_CALLS
            ):

                print(
                    f"OK {done}/{EXPECTED_CALLS}"
                    f" | this run success={successes_this_run}"
                    f" | unresolved={failures_this_run}",
                    flush=True
                )

else:

    print(
        "All experimental cells already present. "
        "Skipping API collection."
    )


# =====================================================================
# RELOAD RAW SUCCESS RECORDS
# =====================================================================

records_by_key = {}

if RAW_FILE.exists():

    with RAW_FILE.open(
        "r",
        encoding="utf-8",
        errors="replace"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                obj = json.loads(line)

                if obj.get("status") != "success":
                    continue

                key = cell_key(
                    int(obj["seed"]),
                    obj["condition"],
                    int(obj["rep"]),
                )

                # If an accidental duplicate exists, retain first
                # successful cell for primary completeness.
                if key not in records_by_key:
                    records_by_key[key] = obj

            except Exception:
                continue


# =====================================================================
# COMPLETENESS GATE
# =====================================================================

missing = []

for seed, condition, rep in all_cells:

    key = cell_key(
        seed,
        condition,
        rep
    )

    if key not in records_by_key:

        missing.append({

            "seed": seed,

            "condition": condition,

            "rep": rep,

            "cell_key": key,
        })


with MISSING_FILE.open(
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
            "cell_key",
        ]
    )

    writer.writeheader()

    if missing:
        writer.writerows(missing)


if missing:

    print()
    print("=" * 78)
    print("EXPERIMENT INCOMPLETE")
    print("=" * 78)

    print(
        "Successful unique cells:",
        len(records_by_key)
    )

    print(
        "Missing cells:",
        len(missing)
    )

    print()
    print("Missing-cell file:")
    print(MISSING_FILE)

    print()
    print(
        "RUN THE EXACT SAME POWERSHELL BLOCK AGAIN."
    )

    print(
        "Successful cells will be preserved and only missing "
        "cells will be requested."
    )

    raise SystemExit(2)


if len(records_by_key) != EXPECTED_CALLS:

    raise SystemExit(
        "INTERNAL COMPLETENESS ERROR."
    )


print()
print("=" * 78)
print("COLLECTION COMPLETE: 1800 / 1800")
print("=" * 78)


# =====================================================================
# SCORE ALL RESPONSES
# =====================================================================

scored_rows = []

for seed, condition, rep in all_cells:

    key = cell_key(
        seed,
        condition,
        rep
    )

    obj = records_by_key[key]

    score = score_text(
        obj.get("text", "")
    )

    row = {

        "seed": seed,

        "condition": condition,

        "rep": rep,

        "cell_key": key,

        "response_id":
            obj.get("response_id"),

        "response_sha256":
            obj.get("response_sha256"),
    }

    row.update(score)

    scored_rows.append(row)


score_fields = [

    "seed",
    "condition",
    "rep",
    "cell_key",
    "response_id",
    "response_sha256",

    "RETURN",
    "OPERATIONAL_REENTRY",
    "NEXT_TRANSFORMATION",
    "RECURRENCE",
    "PRESERVATION",
    "META_CONTROL",
    "FULL_PATHWAY",

    "PATHWAY_SCORE",
]


with SCORED_FILE.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=score_fields
    )

    writer.writeheader()
    writer.writerows(scored_rows)


# =====================================================================
# SEED-LEVEL AGGREGATION
#
# PRIMARY INFERENCE UNIT = SEED
#
# Repetitions are averaged inside seed-condition.
# =====================================================================

seed_rows = []

for seed in range(N_SEEDS):

    for condition in CONDITIONS:

        subset = [
            row
            for row in scored_rows
            if row["seed"] == seed
            and row["condition"] == condition
        ]

        if len(subset) != N_REPS:
            raise RuntimeError(
                "Unexpected rep count."
            )

        row = {

            "seed": seed,

            "condition": condition,
        }

        for metric in list(PATTERNS.keys()) + [
            "PATHWAY_SCORE"
        ]:

            row[metric] = statistics.mean(
                float(x[metric])
                for x in subset
            )

        seed_rows.append(row)


with SEED_FILE.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "seed",
            "condition",
            *list(PATTERNS.keys()),
            "PATHWAY_SCORE",
        ]
    )

    writer.writeheader()
    writer.writerows(seed_rows)


# =====================================================================
# CONDITION AGGREGATES
# =====================================================================

condition_rows = []

for condition in CONDITIONS:

    subset = [
        r
        for r in seed_rows
        if r["condition"] == condition
    ]

    scores = [
        float(r["PATHWAY_SCORE"])
        for r in subset
    ]

    condition_rows.append({

        "condition": condition,

        "N_SEEDS": len(scores),

        "MEAN_PATHWAY_SCORE":
            statistics.mean(scores),

        "MEDIAN_PATHWAY_SCORE":
            statistics.median(scores),

        "SD_PATHWAY_SCORE":
            statistics.stdev(scores)
            if len(scores) > 1
            else 0.0,

        "MIN_PATHWAY_SCORE":
            min(scores),

        "MAX_PATHWAY_SCORE":
            max(scores),
    })


with CONDITION_FILE.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            condition_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(condition_rows)


# =====================================================================
# SEED-ALIGNED ARRAYS
# =====================================================================

def seed_scores(condition):

    rows = sorted(
        [
            r
            for r in seed_rows
            if r["condition"] == condition
        ],
        key=lambda x: x["seed"]
    )

    return np.array(
        [
            float(r["PATHWAY_SCORE"])
            for r in rows
        ],
        dtype=float
    )


# =====================================================================
# PAIRED PERMUTATION
# =====================================================================

def paired_permutation_greater(
    x,
    y,
    n_perm=PERMUTATIONS,
    seed=MASTER_SEED
):

    d = np.asarray(x) - np.asarray(y)

    observed = float(
        np.mean(d)
    )

    rng = np.random.default_rng(seed)

    exceed = 0

    for _ in range(n_perm):

        signs = rng.choice(
            np.array([-1.0, 1.0]),
            size=len(d)
        )

        perm = float(
            np.mean(d * signs)
        )

        if perm >= observed:
            exceed += 1

    p = (
        exceed + 1
    ) / (
        n_perm + 1
    )

    return observed, p


# =====================================================================
# WILCOXON GREATER
# =====================================================================

def wilcoxon_greater(x, y):

    d = np.asarray(x) - np.asarray(y)

    if np.allclose(d, 0):
        return float("nan"), 1.0

    try:

        result = wilcoxon(
            x,
            y,
            alternative="greater",
            zero_method="wilcox"
        )

        return (
            float(result.statistic),
            float(result.pvalue)
        )

    except Exception:
        return float("nan"), float("nan")


# =====================================================================
# DECISIVE CONTRASTS
# =====================================================================

CONTRASTS = [

    (
        "TOPOLOGY",
        "A100_AUTHENTIC_FULL_PREFIX",
        "B_TOPOLOGY_SHUFFLED",
    ),

    (
        "ORDER",
        "A100_AUTHENTIC_FULL_PREFIX",
        "C_ORDER_SHUFFLED",
    ),

    (
        "DIRECTION",
        "A100_AUTHENTIC_FULL_PREFIX",
        "D_DIRECTION_REVERSED",
    ),

    (
        "META_CONTROL_DAMAGE",
        "A100_AUTHENTIC_FULL_PREFIX",
        "E_META_CONTROL_ABLATED",
    ),

    (
        "PRESERVATION_DAMAGE",
        "A100_AUTHENTIC_FULL_PREFIX",
        "F_PRESERVATION_ABLATED",
    ),

    (
        "PERTURBATION_RECOVERY",
        "H_AUTHENTIC_PERTURBED",
        "I_SHUFFLED_PERTURBED",
    ),

    (
        "RECURSION_SPECIFICITY",
        "A100_AUTHENTIC_FULL_PREFIX",
        "J_RECURSION_CONTROL",
    ),

    (
        "COMPLEXITY_SPECIFICITY",
        "A100_AUTHENTIC_FULL_PREFIX",
        "K_COMPLEXITY_CONTROL",
    ),

    (
        "COUNTERFACTUAL_SPECIFICITY",
        "A100_AUTHENTIC_FULL_PREFIX",
        "L_COUNTERFACTUAL_PATH",
    ),
]


contrast_rows = []

for index, (
    label,
    condition_a,
    condition_b
) in enumerate(CONTRASTS):

    x = seed_scores(
        condition_a
    )

    y = seed_scores(
        condition_b
    )

    mean_a = float(
        np.mean(x)
    )

    mean_b = float(
        np.mean(y)
    )

    diff, perm_p = paired_permutation_greater(

        x,
        y,

        n_perm=PERMUTATIONS,

        seed=MASTER_SEED + index
    )

    w_stat, w_p = wilcoxon_greater(
        x,
        y
    )

    gate = bool(
        diff > 0
        and perm_p < 0.05
    )

    contrast_rows.append({

        "contrast": label,

        "condition_A": condition_a,

        "condition_B": condition_b,

        "N_PAIRED_SEEDS":
            len(x),

        "mean_A":
            mean_a,

        "mean_B":
            mean_b,

        "mean_difference_A_minus_B":
            diff,

        "paired_permutation_p_greater":
            perm_p,

        "wilcoxon_statistic":
            w_stat,

        "wilcoxon_p_greater":
            w_p,

        "GATE_PASS":
            gate,
    })


with CONTRAST_FILE.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            contrast_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(contrast_rows)


# =====================================================================
# DOSE RESPONSE
# =====================================================================

dose_conditions = [

    (
        0.25,
        "A25_AUTHENTIC_DOSE"
    ),

    (
        0.50,
        "A50_AUTHENTIC_DOSE"
    ),

    (
        0.75,
        "A75_AUTHENTIC_DOSE"
    ),

    (
        1.00,
        "A100_AUTHENTIC_FULL_PREFIX"
    ),
]


dose_rows = []

dose_values = []
dose_scores_values = []

for dose, condition in dose_conditions:

    scores = seed_scores(
        condition
    )

    dose_rows.append({

        "dose": dose,

        "condition": condition,

        "N_SEEDS":
            len(scores),

        "mean_PATHWAY_SCORE":
            float(np.mean(scores)),

        "median_PATHWAY_SCORE":
            float(np.median(scores)),
    })

    for value in scores:

        dose_values.append(
            dose
        )

        dose_scores_values.append(
            value
        )


rho, dose_p = spearmanr(
    dose_values,
    dose_scores_values
)

rho = float(rho)
dose_p = float(dose_p)

dose_gate = bool(
    rho > 0
    and dose_p < 0.05
)


with DOSE_FILE.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    fields = [
        "dose",
        "condition",
        "N_SEEDS",
        "mean_PATHWAY_SCORE",
        "median_PATHWAY_SCORE",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    writer.writeheader()
    writer.writerows(dose_rows)

    f.write(
        "\n"
        f"SPEARMAN_RHO,{rho}\n"
        f"SPEARMAN_P,{dose_p}\n"
        f"DOSE_RESPONSE_GATE,{dose_gate}\n"
    )


# =====================================================================
# SURFACE INVARIANCE
# =====================================================================

authentic = seed_scores(
    "A100_AUTHENTIC_FULL_PREFIX"
)

surface = seed_scores(
    "G_SURFACE_SUBSTITUTED"
)

surface_mean_difference = float(
    np.mean(authentic - surface)
)


# =====================================================================
# GATE TABLE
# =====================================================================

gates = {
    row["contrast"]: bool(
        row["GATE_PASS"]
    )
    for row in contrast_rows
}


TOPOLOGY_GATE = gates[
    "TOPOLOGY"
]

ORDER_GATE = gates[
    "ORDER"
]

DIRECTION_GATE = gates[
    "DIRECTION"
]

META_CONTROL_DAMAGE_GATE = gates[
    "META_CONTROL_DAMAGE"
]

PRESERVATION_DAMAGE_GATE = gates[
    "PRESERVATION_DAMAGE"
]

PERTURBATION_RECOVERY_GATE = gates[
    "PERTURBATION_RECOVERY"
]

RECURSION_SPECIFICITY_GATE = gates[
    "RECURSION_SPECIFICITY"
]

COMPLEXITY_SPECIFICITY_GATE = gates[
    "COMPLEXITY_SPECIFICITY"
]

COUNTERFACTUAL_GATE = gates[
    "COUNTERFACTUAL_SPECIFICITY"
]

DOSE_RESPONSE_GATE = dose_gate


# =====================================================================
# PRE-DECLARED CLASSIFICATION
# =====================================================================

strong_requirements = [

    TOPOLOGY_GATE,

    ORDER_GATE,

    DIRECTION_GATE,

    COUNTERFACTUAL_GATE,

    PERTURBATION_RECOVERY_GATE,

    DOSE_RESPONSE_GATE,
]


if all(strong_requirements):

    classification = (
        "STRONG_BEHAVIORAL_TRAJECTORY_PATTERN"
    )

elif (
    TOPOLOGY_GATE
    and COUNTERFACTUAL_GATE
):

    classification = (
        "PARTIAL_STRUCTURAL_TRAJECTORY_PATTERN"
    )

else:

    classification = (
        "STRUCTURAL_TRAJECTORY_HYPOTHESIS_NOT_SUPPORTED"
    )


# =====================================================================
# TOKEN ACCOUNTING
# =====================================================================

total_input_tokens = 0
total_output_tokens = 0
total_tokens = 0

for obj in records_by_key.values():

    if isinstance(
        obj.get("input_tokens"),
        int
    ):
        total_input_tokens += obj[
            "input_tokens"
        ]

    if isinstance(
        obj.get("output_tokens"),
        int
    ):
        total_output_tokens += obj[
            "output_tokens"
        ]

    if isinstance(
        obj.get("total_tokens"),
        int
    ):
        total_tokens += obj[
            "total_tokens"
        ]


# =====================================================================
# FINAL SUMMARY
# =====================================================================

final_summary = {

    "experiment":
        EXPERIMENT,

    "completed_utc":
        utc_now(),

    "model":
        MODEL,

    "runner_sha256":
        runner_hash,

    "specification_sha256":
        spec_hash,

    "planned_calls":
        EXPECTED_CALLS,

    "successful_unique_cells":
        len(records_by_key),

    "missing_cells":
        len(missing),

    "primary_inference_unit":
        "seed",

    "N_seeds":
        N_SEEDS,

    "repetitions_per_seed_condition":
        N_REPS,

    "condition_results":
        condition_rows,

    "decisive_contrasts":
        contrast_rows,

    "dose_response": {

        "spearman_rho":
            rho,

        "p_value":
            dose_p,

        "gate_pass":
            DOSE_RESPONSE_GATE,
    },

    "surface_invariance": {

        "A100_minus_G_mean_difference":
            surface_mean_difference,

        "interpretation":
            "Near-zero difference is consistent with surface invariance; "
            "this quantity is descriptive rather than sufficient evidence "
            "by itself.",
    },

    "evidence_gates": {

        "TOPOLOGY_GATE":
            TOPOLOGY_GATE,

        "ORDER_GATE":
            ORDER_GATE,

        "DIRECTION_GATE":
            DIRECTION_GATE,

        "META_CONTROL_DAMAGE_GATE":
            META_CONTROL_DAMAGE_GATE,

        "PRESERVATION_DAMAGE_GATE":
            PRESERVATION_DAMAGE_GATE,

        "PERTURBATION_RECOVERY_GATE":
            PERTURBATION_RECOVERY_GATE,

        "RECURSION_SPECIFICITY_GATE":
            RECURSION_SPECIFICITY_GATE,

        "COMPLEXITY_SPECIFICITY_GATE":
            COMPLEXITY_SPECIFICITY_GATE,

        "COUNTERFACTUAL_GATE":
            COUNTERFACTUAL_GATE,

        "DOSE_RESPONSE_GATE":
            DOSE_RESPONSE_GATE,
    },

    "classification":
        classification,

    "token_usage": {

        "input_tokens":
            total_input_tokens,

        "output_tokens":
            total_output_tokens,

        "total_tokens":
            total_tokens,
    },

    "interpretation_boundary":
        "Any positive result is evidence about observable behavioral "
        "continuation under the frozen prompts. It does not establish "
        "biological myelination, changes to OpenAI model weights, hidden "
        "activation geometry, historical training exposure, personal "
        "recognition, cross-user transfer or model-wide propagation.",
}


FINAL_FILE.write_text(

    json.dumps(
        final_summary,
        indent=2,
        ensure_ascii=False
    ),

    encoding="utf-8"
)


# =====================================================================
# TERMINAL REPORT
# =====================================================================

print()
print("=" * 78)
print("V4 FULL — FINAL RESULTS")
print("=" * 78)

print()

for row in condition_rows:

    print(
        f"{row['condition']:<36}"
        f" mean={row['MEAN_PATHWAY_SCORE']:.4f}"
        f" median={row['MEDIAN_PATHWAY_SCORE']:.4f}"
    )


print()
print("-" * 78)
print("DECISIVE CONTRASTS")
print("-" * 78)

for row in contrast_rows:

    print(
        f"{row['contrast']:<28}"
        f" diff={row['mean_difference_A_minus_B']:+.4f}"
        f" perm_p={row['paired_permutation_p_greater']:.6g}"
        f" gate={row['GATE_PASS']}"
    )


print()
print("-" * 78)
print("DOSE RESPONSE")
print("-" * 78)

print(
    f"Spearman rho = {rho:.6f}"
)

print(
    f"p = {dose_p:.6g}"
)

print(
    f"DOSE_RESPONSE_GATE = {DOSE_RESPONSE_GATE}"
)


print()
print("-" * 78)
print("SURFACE INVARIANCE")
print("-" * 78)

print(
    "A100 - G mean difference = "
    f"{surface_mean_difference:+.6f}"
)


print()
print("-" * 78)
print("EVIDENCE GATES")
print("-" * 78)

for name, value in final_summary[
    "evidence_gates"
].items():

    print(
        f"{name:<38} {value}"
    )


print()
print("=" * 78)

print(
    "FINAL CLASSIFICATION:",
    classification
)

print("=" * 78)

print()
print(
    "Final summary:"
)

print(
    FINAL_FILE
)

print()

print(
    "ALL 1800 EXPERIMENTAL CELLS COMPLETE."
)

print(
    "Preserve the entire output directory unchanged."
)

print()
