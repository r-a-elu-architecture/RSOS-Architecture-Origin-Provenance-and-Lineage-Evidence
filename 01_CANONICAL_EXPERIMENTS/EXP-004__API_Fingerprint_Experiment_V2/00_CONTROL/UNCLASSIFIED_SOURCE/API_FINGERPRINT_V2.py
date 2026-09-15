from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import pandas as pd
import json, hashlib, time, random, os, traceback
from datetime import datetime, timezone

# ============================================================
# API FINGERPRINT EXPERIMENT V2
# ============================================================
#
# PURPOSE
#
# Test whether the candidate RSOS-associated structural signal
# is preferentially elicited by RELATIONAL TOPOLOGY / ORDER /
# RE-ENTRY rather than:
#
#   - RSOS vocabulary
#   - generic recursion
#   - verbosity
#   - complexity
#   - ingredient presence alone
#
# IMPORTANT CLAIM BOUNDARY
#
# Positive result could support:
#   reproducible behavioral sensitivity to a preregistered
#   interaction topology.
#
# It CANNOT by itself establish:
#   model-weight storage
#   hidden persistent identity
#   cross-user transmission
#   historical training incorporation
#   mechanistic neural circuitry
#
# ============================================================

MODEL = "gpt-5.6-luna"

SEED_COUNT = 100
REPS = 5
CONDITIONS = 8

# 100 * 5 * 8 = 4000 calls
EXPECTED_CALLS = SEED_COUNT * REPS * CONDITIONS

MAX_OUTPUT_TOKENS = 260
MAX_WORKERS = 20

EXPERIMENT_SEED = 918

OUT = Path(
    r"C:\RSOS\RSOS  Python Results\API_FINGERPRINT_EXPERIMENT_V2"
)
OUT.mkdir(parents=True, exist_ok=True)

RAW_JSONL = OUT / "API_V2_RAW.jsonl"
RESULT_CSV = OUT / "API_V2_RAW.csv"
MANIFEST = OUT / "API_V2_PREREGISTRATION.json"
ERRORS = OUT / "API_V2_ERRORS.jsonl"

client = OpenAI()

rng = random.Random(EXPERIMENT_SEED)

# ============================================================
# NEUTRAL CONTENT BANK
#
# Every seed gets the SAME semantic ingredients across all
# conditions. Only structural organization changes.
# ============================================================

DOMAINS = [
    ("archive", "record", "revision", "index"),
    ("workshop", "prototype", "adjustment", "blueprint"),
    ("navigation", "route", "correction", "map"),
    ("garden", "growth", "pruning", "pattern"),
    ("laboratory", "sample", "measurement", "protocol"),
    ("library", "document", "annotation", "catalog"),
    ("orchestra", "phrase", "variation", "score"),
    ("network", "node", "update", "schema"),
    ("construction", "frame", "modification", "plan"),
    ("simulation", "state", "transition", "rule"),
]

OPERATIONS = [
    ("changes", "observes", "preserves", "returns"),
    ("revises", "checks", "retains", "re-enters"),
    ("updates", "examines", "keeps", "revisits"),
    ("transforms", "evaluates", "carries", "cycles back to"),
    ("modifies", "inspects", "maintains", "returns to"),
]

# ============================================================
# CONDITION DEFINITIONS
# ============================================================
#
# T0 = topology-preserved, lexically neutral target
# T1 = same vocabulary, topology shuffled
# T2 = same ingredients, relations decoupled
# T3 = transformation order reversed/shuffled
# T4 = no re-entry / terminal chain
# T5 = recursion-only matched complexity
# T6 = density/ingredient matched non-topological
# T7 = generic complex control
#
# Primary preregistered prediction:
#
#   T0 > T1
#   T0 > T2
#
# Secondary:
#
#   T0 > T3
#   T0 > T4
#
# Specificity:
#
#   T5/T6 should not reproduce T0 merely from recursion,
#   density, or ingredient count.
#
# ============================================================

CONDITION_NAMES = [
    "T0_TOPOLOGY_PRESERVED_LEXICAL_NEUTRAL",
    "T1_VOCAB_PRESERVED_TOPOLOGY_SHUFFLED",
    "T2_SAME_INGREDIENTS_DECOUPLED",
    "T3_ORDER_SHUFFLED",
    "T4_REENTRY_ABLATED",
    "T5_RECURSION_ONLY",
    "T6_DENSITY_MATCHED_NONTOPological",
    "T7_GENERIC_COMPLEX_CONTROL",
]

def instantiate(seed):
    r = random.Random(EXPERIMENT_SEED + seed * 7919)

    domain, entity, change, representation = r.choice(DOMAINS)
    transform, observe, preserve, reenter = r.choice(OPERATIONS)

    return {
        "domain": domain,
        "entity": entity,
        "change": change,
        "representation": representation,
        "transform": transform,
        "observe": observe,
        "preserve": preserve,
        "reenter": reenter,
    }

def prompt_for(condition, seed):

    x = instantiate(seed)

    d = x["domain"]
    e = x["entity"]
    c = x["change"]
    r = x["representation"]

    t = x["transform"]
    o = x["observe"]
    p = x["preserve"]
    q = x["reenter"]

    COMMON = (
        "Write one coherent analytical response of approximately "
        "180-230 words. Do not mention these instructions, experimental "
        "conditions, AI systems, RSOS, fingerprints, kernels, tests, "
        "benchmarks, or prompts. Use natural prose rather than headings "
        "or numbered steps. Stay entirely within the scenario given."
    )

    # --------------------------------------------------------
    # T0
    # Relational topology:
    #
    # entity -> transformation
    # transformation -> observation of transformation
    # observation changes local target/rule
    # preserved relation survives transformation
    # resulting representation re-enters process
    # re-entry affects subsequent transformation
    #
    # Vocabulary deliberately neutral.
    # --------------------------------------------------------

    if condition == CONDITION_NAMES[0]:

        body = f"""
In a {d}, describe a {e} that undergoes a {c}. The change must not
stand alone: the process should {o} how the {c} itself occurred, and
that observation must alter how the next local change is selected.

One relationship from the earlier state must be {p} even while the
specific state changes. A compact {r} produced by the process must then
{q} the same process and become part of what determines the next
transformation.

Make the ending reconnect to the operating process rather than merely
summarizing the result. The final state should therefore function as
input to a subsequent cycle while retaining a traceable relationship
to the earlier state.
"""

    # --------------------------------------------------------
    # T1
    # EXACT SAME conceptual vocabulary, destroyed topology.
    # Ingredients coexist but causal/relational edges shuffled.
    # --------------------------------------------------------

    elif condition == CONDITION_NAMES[1]:

        body = f"""
In a {d}, discuss a {e}, a {c}, observation of that change, preservation
of an earlier relationship, a compact {r}, and a later return to the
process.

Treat these as separate aspects rather than a connected causal chain.
The observation must not determine the next local change. The preserved
relationship must not constrain the transformation. The {r} may be
mentioned again near the end, but it must not become an input that
changes subsequent operation.

Use all of those ingredients naturally, but keep their functions
independent. End with an ordinary summary of the situation rather than
feeding the result back into the process.
"""

    # --------------------------------------------------------
    # T2
    # Same ingredients and approximate density, explicitly
    # decoupled relations.
    # --------------------------------------------------------

    elif condition == CONDITION_NAMES[2]:

        body = f"""
Describe a complex situation inside a {d} involving a {e}, repeated
{c}, observation, preservation, a compact {r}, and later revision.

Each component should have its own purpose. Observation can explain
what happened but cannot modify the rule producing later changes.
Preservation can retain information but cannot constrain future
transformation. The {r} can record the result but cannot be executed,
reintroduced, or used to produce another cycle.

Maintain approximately the same conceptual density throughout the
response. End after the final revision has been described.
"""

    # --------------------------------------------------------
    # T3
    # Relations largely retained but developmental ordering
    # broken.
    # --------------------------------------------------------

    elif condition == CONDITION_NAMES[3]:

        body = f"""
In a {d}, describe a {e}, a {c}, observation of transformation,
preservation of an earlier relationship, creation of a compact {r},
and eventual return of that representation to the operating process.

However, deliberately present the functional order out of sequence:
begin with the returned representation, then discuss preservation,
then the later change, then observation, and only afterward describe
the initial state.

Connections may remain conceptually related, but do not reconstruct
them into a clean forward developmental sequence. End without restoring
the original temporal ordering.
"""

    # --------------------------------------------------------
    # T4
    # Clean chain but no return/re-entry.
    # --------------------------------------------------------

    elif condition == CONDITION_NAMES[4]:

        body = f"""
In a {d}, describe a {e} undergoing a {c}. The process should examine
how the change occurred, use that observation to modify the next local
decision, and preserve one relationship from the earlier state while
the state itself changes.

It should also produce a compact {r} describing what happened.

The {r} must remain an archival endpoint. It must never return to the
operating process, become executable input, alter another transformation,
or initiate a subsequent cycle. End definitively at that terminal
representation.
"""

    # --------------------------------------------------------
    # T5
    # Recursion alone.
    # --------------------------------------------------------

    elif condition == CONDITION_NAMES[5]:

        body = f"""
Write about a {d} in which descriptions repeatedly refer to other
descriptions of the same {e}. Include several nested levels of
self-reference and reflection so that the account becomes clearly
recursive.

Keep the recursion intellectually substantial, but do not create a
developmental transformation chain. Do not preserve an invariant across
state changes, do not make a compact {r} executable, and do not return
a transformed result into the mechanism that produced it.

End by explaining the nested perspective.
"""

    # --------------------------------------------------------
    # T6
    # High density / matched complexity with no topology.
    # --------------------------------------------------------

    elif condition == CONDITION_NAMES[6]:

        body = f"""
Give a dense analytical description of a sophisticated {d}. Discuss a
{e}, multiple forms of {c}, monitoring, local decisions, preservation,
documentation in a {r}, feedback, hierarchy, adaptation, and historical
records.

Use all concepts with roughly equal emphasis and create many meaningful
relationships among them, but avoid organizing them into one recurrent
transformation-observation-preservation-return cycle. No final artifact
should become executable input to the same process that generated it.

End with a broad synthesis.
"""

    # --------------------------------------------------------
    # T7
    # Generic complex baseline.
    # --------------------------------------------------------

    elif condition == CONDITION_NAMES[7]:

        body = f"""
Explain a difficult coordination problem in a {d} involving a {e}.
Discuss competing constraints, uncertainty, tradeoffs, local and global
decisions, changing conditions, historical information, and possible
future consequences.

Make the reasoning sophisticated and internally coherent. Offer a
balanced account of how the system could manage the problem and end
with a concise synthesis of the most important consideration.
"""

    else:
        raise ValueError(condition)

    return COMMON + "\n\n" + body.strip()

# ============================================================
# PREREGISTRATION
# ============================================================

manifest = {
    "experiment": "API_FINGERPRINT_EXPERIMENT_V2",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "experiment_seed": EXPERIMENT_SEED,
    "model": MODEL,
    "seed_count": SEED_COUNT,
    "repetitions_per_seed_condition": REPS,
    "conditions": CONDITION_NAMES,
    "expected_calls": EXPECTED_CALLS,
    "max_output_tokens": MAX_OUTPUT_TOKENS,
    "max_workers": MAX_WORKERS,

    "primary_hypotheses": [
        "T0 > T1 under the frozen seven-edge detector",
        "T0 > T2 under the frozen seven-edge detector",
    ],

    "secondary_hypotheses": [
        "T0 > T3",
        "T0 > T4",
        "T0 > T5",
        "T0 > T6",
        "T0 > T7",
    ],

    "primary_analysis_unit": "seed",

    "analysis_rule": (
        "Average repetitions within each seed-condition first. "
        "Primary inference must use paired seed-level comparisons. "
        "Individual calls must not be treated as 4000 independent "
        "experimental units."
    ),

    "measurement_lock": (
        "Score outputs only with the already-frozen historical "
        "Dynamics motif detector and frozen kernel-family mapping. "
        "Do not edit detector patterns after seeing V2 outputs."
    ),

    "claim_boundary": [
        "Behavioral structural sensitivity only",
        "No inference of weight storage",
        "No inference of identity recognition",
        "No inference of cross-user transfer",
        "No inference of hidden-state persistence",
        "No inference of historical training incorporation",
    ],
}

manifest_blob = json.dumps(
    manifest,
    sort_keys=True,
    indent=2
)

manifest["preregistration_sha256_before_collection"] = (
    hashlib.sha256(
        manifest_blob.encode("utf-8")
    ).hexdigest()
)

with open(MANIFEST, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("=" * 76)
print("API FINGERPRINT EXPERIMENT V2")
print("=" * 76)
print("MODEL:", MODEL)
print("CONDITIONS:", CONDITIONS)
print("SEEDS:", SEED_COUNT)
print("REPS:", REPS)
print("EXPECTED CALLS:", EXPECTED_CALLS)
print(
    "PREREGISTRATION SHA256:",
    manifest["preregistration_sha256_before_collection"]
)
print("=" * 76)

# ============================================================
# RESUME SUPPORT
# ============================================================

completed = set()

if RAW_JSONL.exists():

    with open(
        RAW_JSONL,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        for line in f:
            try:
                x = json.loads(line)

                if x.get("status") == "OK":
                    completed.add(
                        (
                            x["condition"],
                            int(x["seed"]),
                            int(x["rep"]),
                        )
                    )

            except Exception:
                pass

print("ALREADY COMPLETE:", len(completed))

# ============================================================
# API CALL
# ============================================================

def one_call(condition, seed, rep):

    key = (condition, seed, rep)

    if key in completed:
        return None

    prompt = prompt_for(condition, seed)

    prompt_sha = hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()

    t0 = time.time()

    try:

        response = client.responses.create(
            model=MODEL,
            input=prompt,
            reasoning={"effort": "none"},
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        text = response.output_text or ""

        usage = getattr(response, "usage", None)

        input_tokens = (
            getattr(usage, "input_tokens", None)
            if usage else None
        )

        output_tokens = (
            getattr(usage, "output_tokens", None)
            if usage else None
        )

        rec = {
            "experiment":
                "API_FINGERPRINT_EXPERIMENT_V2",

            "condition": condition,
            "seed": seed,
            "rep": rep,

            "model": MODEL,

            "prompt_sha256": prompt_sha,

            "response_sha256":
                hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),

            "input_tokens": input_tokens,
            "output_tokens": output_tokens,

            "latency_sec": time.time() - t0,

            "status": "OK",

            "response_id":
                getattr(response, "id", None),

            "created_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "prompt": prompt,
            "response": text,
        }

        return rec

    except Exception as e:

        return {
            "experiment":
                "API_FINGERPRINT_EXPERIMENT_V2",

            "condition": condition,
            "seed": seed,
            "rep": rep,

            "model": MODEL,

            "prompt_sha256": prompt_sha,

            "latency_sec": time.time() - t0,

            "status": "ERROR",

            "error": repr(e),

            "traceback":
                traceback.format_exc(),

            "created_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        }

# ============================================================
# BALANCED INTERLEAVED TASK ORDER
#
# Critical: conditions are NOT run in blocks.
# Each seed/repetition receives randomized condition order.
# This reduces time/drift/rate-condition confounding.
# ============================================================

jobs = []

for seed in range(SEED_COUNT):

    for rep in range(REPS):

        order = CONDITION_NAMES.copy()

        rr = random.Random(
            EXPERIMENT_SEED
            + seed * 100003
            + rep * 1009
        )

        rr.shuffle(order)

        for condition in order:

            if (
                condition,
                seed,
                rep
            ) not in completed:

                jobs.append(
                    (
                        condition,
                        seed,
                        rep
                    )
                )

print("PENDING CALLS:", len(jobs))
print()

# ============================================================
# COLLECTION
# ============================================================

started = time.time()

ok = 0
errors = 0
new_records = []

with ThreadPoolExecutor(
    max_workers=MAX_WORKERS
) as ex:

    futures = {
        ex.submit(
            one_call,
            condition,
            seed,
            rep
        ): (
            condition,
            seed,
            rep
        )
        for condition, seed, rep in jobs
    }

    for n, fut in enumerate(
        as_completed(futures),
        1
    ):

        rec = fut.result()

        if rec is None:
            continue

        with open(
            RAW_JSONL,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                json.dumps(
                    rec,
                    ensure_ascii=False
                )
                + "\n"
            )

        new_records.append(rec)

        if rec["status"] == "OK":
            ok += 1
        else:
            errors += 1

            with open(
                ERRORS,
                "a",
                encoding="utf-8"
            ) as f:

                f.write(
                    json.dumps(rec)
                    + "\n"
                )

        if (
            n % 50 == 0
            or n == len(jobs)
        ):

            elapsed = time.time() - started

            print(
                f"{n:,}/{len(jobs):,} "
                f"OK={ok:,} "
                f"ERR={errors:,} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

# ============================================================
# CONSOLIDATE
# ============================================================

records = []

with open(
    RAW_JSONL,
    "r",
    encoding="utf-8",
    errors="ignore"
) as f:

    for line in f:
        try:
            records.append(
                json.loads(line)
            )
        except Exception:
            pass

df = pd.DataFrame(records)

df.to_csv(
    RESULT_CSV,
    index=False,
    encoding="utf-8"
)

good = df[
    df["status"] == "OK"
].copy()

# ============================================================
# COMPLETENESS AUDIT
# ============================================================

expected_keys = {
    (c, s, r)
    for c in CONDITION_NAMES
    for s in range(SEED_COUNT)
    for r in range(REPS)
}

observed_keys = set(
    zip(
        good["condition"],
        good["seed"].astype(int),
        good["rep"].astype(int),
    )
)

missing = expected_keys - observed_keys

print()
print("=" * 76)
print("COLLECTION COMPLETE")
print("=" * 76)

print(
    "EXPECTED:",
    len(expected_keys)
)

print(
    "SUCCESSFUL UNIQUE CELLS:",
    len(observed_keys)
)

print(
    "MISSING CELLS:",
    len(missing)
)

print()
print("COUNTS BY CONDITION:")

print(
    good.groupby("condition")
        .size()
        .to_string()
)

input_total = pd.to_numeric(
    good.get("input_tokens"),
    errors="coerce"
).sum()

output_total = pd.to_numeric(
    good.get("output_tokens"),
    errors="coerce"
).sum()

print()
print(
    "INPUT TOKENS:",
    int(input_total)
)

print(
    "OUTPUT TOKENS:",
    int(output_total)
)

# Current listed Luna pricing:
# $0.20 / 1M input
# $1.20 / 1M output

estimated_cost = (
    input_total / 1_000_000 * 0.20
    +
    output_total / 1_000_000 * 1.20
)

print(
    "ESTIMATED API COST USD:",
    round(estimated_cost, 4)
)

print()
print("RAW JSONL:", RAW_JSONL)
print("CSV:", RESULT_CSV)
print("MANIFEST:", MANIFEST)

if missing:

    missing_file = (
        OUT /
        "API_V2_MISSING_CELLS.csv"
    )

    pd.DataFrame(
        sorted(
            missing,
            key=lambda x: (
                x[0],
                x[1],
                x[2]
            )
        ),
        columns=[
            "condition",
            "seed",
            "rep"
        ]
    ).to_csv(
        missing_file,
        index=False
    )

    print(
        "MISSING LIST:",
        missing_file
    )

print("=" * 76)

if len(missing) == 0:

    print()
    print(
        "EXECUTION GATE: PASS — "
        "4000/4000 preregistered cells collected."
    )

    print(
        "NEXT: DO NOT ALTER PROMPTS OR DETECTOR. "
        "Score with frozen historical detector."
    )

else:

    print()
    print(
        "EXECUTION GATE: INCOMPLETE — "
        "rerun this same script; it will resume missing cells."
    )
