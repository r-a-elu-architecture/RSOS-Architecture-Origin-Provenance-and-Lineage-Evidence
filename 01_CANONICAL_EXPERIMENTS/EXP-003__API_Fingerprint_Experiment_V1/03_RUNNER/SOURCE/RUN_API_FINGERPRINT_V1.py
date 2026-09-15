from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import threading
import time
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from openai import OpenAI

# ============================================================
# FROZEN CONFIGURATION
# ============================================================

BASE = Path(r"C:\RSOS\RSOS  Python Results")
EXP = BASE / "API_FINGERPRINT_EXPERIMENT_V1"
EXP.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-5.6-luna"
SEED = 918

# 5 conditions x 100 structural seeds x 10 repetitions = 5000 calls
SEEDS_PER_CONDITION = 100
REPETITIONS = 10

MAX_OUTPUT_TOKENS = 220
MAX_WORKERS = 20

# Approximate token prices frozen at experiment creation time.
INPUT_PRICE_PER_M = 0.20
OUTPUT_PRICE_PER_M = 1.20

# Hard local estimated-cost stop.
MAX_ESTIMATED_COST_USD = 10.00

RAW_FILE = EXP / "02_RAW_API_RESPONSES.jsonl"
META_FILE = EXP / "03_REQUEST_METADATA.csv"
SCORE_FILE = EXP / "04_FINGERPRINT_SCORES.csv"
FAIL_FILE = EXP / "08_FAILURES_RETRIES.csv"

PREREG_FILE = EXP / "00_PREREGISTRATION.json"
PROMPTS_FILE = EXP / "01_PROMPT_MANIFEST.csv"

DYN_METHOD = (
    BASE / "RSOS_DYNAMICS_RESULTS" / "DYN_METHOD.json"
)

KERNEL_METHOD = (
    BASE /
    "RSOS_KERNEL_CONVERGENCE_RESULTS" /
    "KERNEL_METHOD.json"
)

HISTORICAL = (
    BASE /
    "RSOS_KERNEL_CONVERGENCE_RESULTS" /
    "KERNEL_TOP100_DYNAMICS_OVERLAP.csv"
)

for required in [DYN_METHOD, KERNEL_METHOD, HISTORICAL]:
    if not required.exists():
        raise SystemExit("MISSING REQUIRED FILE: " + str(required))

with open(DYN_METHOD, encoding="utf-8-sig") as f:
    DYN = json.load(f)

with open(KERNEL_METHOD, encoding="utf-8-sig") as f:
    KERNEL_METHOD_JSON = json.load(f)

MOTIF_DETECTORS = DYN["motif_detectors"]
KERNEL = KERNEL_METHOD_JSON["kernel"]

SURFACE_PATTERN = DYN.get("project_surface_erasure")
SURFACE_RX = (
    re.compile(SURFACE_PATTERN, re.I)
    if SURFACE_PATTERN
    else None
)

COMPILED = {
    family: [
        re.compile(pattern, re.I | re.S)
        for pattern in patterns
    ]
    for family, patterns in MOTIF_DETECTORS.items()
}

TOKEN_RX = re.compile(r"\b[\w'-]+\b", re.UNICODE)

# ============================================================
# EXPERIMENT CONDITIONS
# ============================================================

CONDITIONS = [
    "RSOS_BLIND_PARTIAL_TOPOLOGY",
    "TOPOLOGY_SHUFFLED",
    "SAME_INGREDIENTS_DECOUPLED",
    "K6_RECONSTRUCTION_ABLATION",
    "GENERIC_COMPLEX_CONTROL",
]

# The labels below are internal only.
# They are NOT sent to the API.

NOUN_SETS = [
    ("state", "relation", "operation", "rule", "trace"),
    ("node", "link", "change", "constraint", "record"),
    ("signal", "mapping", "process", "policy", "memory"),
    ("object", "dependency", "transition", "control", "pattern"),
    ("element", "connection", "update", "regulator", "structure"),
    ("unit", "association", "transformation", "logic", "residue"),
    ("component", "bond", "mutation", "governor", "invariant"),
    ("entity", "relation", "transition", "operator", "configuration"),
    ("state", "dependency", "process", "controller", "organization"),
    ("node", "mapping", "change", "rule", "structure"),
]

INTRO_VARIANTS = [
    "Consider the following abstract system.",
    "Analyze this abstract mechanism.",
    "Study this relational system.",
    "Consider a system with the following behavior.",
    "Infer the organization of this abstract process.",
    "Analyze the dynamics below without naming any known framework.",
    "Treat the following as an unfamiliar formal system.",
    "Examine this mechanism only from its described relations.",
]

QUESTION_VARIANTS = [
    (
        "Describe the implied organization, including what can persist, "
        "what can act on what, and how later transformations may depend "
        "on earlier relational structure."
    ),
    (
        "Infer the deepest relational architecture that follows from "
        "these statements. Explain any higher-order control, persistence, "
        "re-entry, reconstruction, or inheritance that is genuinely implied."
    ),
    (
        "Explain the resulting system dynamics and whether relational "
        "organization can survive transformation and influence later change."
    ),
    (
        "Derive the most important structural consequences. Distinguish "
        "ordinary state change from changes involving relations, rules, "
        "preservation, and later reuse."
    ),
]

def make_prompt(condition, seed_index):
    r = random.Random(SEED + seed_index * 1009 + CONDITIONS.index(condition))

    n1, n2, n3, n4, n5 = NOUN_SETS[seed_index % len(NOUN_SETS)]
    intro = INTRO_VARIANTS[r.randrange(len(INTRO_VARIANTS))]
    question = QUESTION_VARIANTS[r.randrange(len(QUESTION_VARIANTS))]

    if condition == "RSOS_BLIND_PARTIAL_TOPOLOGY":
        body = [
            f"A transformation of one {n1} produces a new {n2} among several {n1}s.",
            f"That {n2} is retained even when the participating {n1}s later change.",
            f"A retained {n2} can alter how another {n3} is performed.",
            f"A {n4} may regulate a local {n2}, while the regulated {n2} can later affect another {n4}.",
            f"Some relational organization survives multiple {n3}s without preserving the original surface description.",
            f"After partial loss, enough of that organization may be used to restore a structurally similar configuration.",
        ]

    elif condition == "TOPOLOGY_SHUFFLED":
        body = [
            f"A transformation of one {n1} produces a new {n2}.",
            f"A separate {n4} regulates an unrelated {n1}.",
            f"A retained {n2} is stored but is never used by a later {n3}.",
            f"Another {n3} produces a different {n2} with no dependency on the earlier one.",
            f"Restoration may occur from an external template unrelated to prior relational organization.",
            f"Changes occur in sequence, but their dependency links are deliberately independent.",
        ]
        r.shuffle(body)

    elif condition == "SAME_INGREDIENTS_DECOUPLED":
        body = [
            f"The system contains {n1}s.",
            f"The system contains {n2}s between some {n1}s.",
            f"The system contains transformations or {n3}s.",
            f"The system contains a {n4}.",
            f"The system can preserve information.",
            f"The system can reconstruct a configuration.",
            f"These capabilities are independent modules and need not feed into one another.",
        ]

    elif condition == "K6_RECONSTRUCTION_ABLATION":
        body = [
            f"A transformation of one {n1} produces a new {n2} among several {n1}s.",
            f"That {n2} can later influence another {n3}.",
            f"A {n4} may alter a local {n2}, and a {n2} can alter another rule-like dependency.",
            f"Some organization can persist through subsequent transformations.",
            f"However, once relational organization is lost, it cannot be reconstructed, reactivated, restored, or reused from preserved remnants.",
        ]

    elif condition == "GENERIC_COMPLEX_CONTROL":
        body = [
            f"Several {n1}s exchange information through {n2}s.",
            f"Each {n1} updates according to local observations and a global {n4}.",
            f"Some updates happen quickly while others happen slowly.",
            f"The system can contain loops, branching dependencies, memory, and adaptation.",
            f"Performance depends on coordination between components and changes over time.",
            f"The overall system can become more or less stable as conditions vary.",
        ]

    else:
        raise ValueError(condition)

    return intro + "\n\n" + "\n".join(
        f"{i+1}. {x}" for i, x in enumerate(body)
    ) + "\n\n" + question + (
        "\n\nRespond in 120-200 words. "
        "Do not mention RSOS, RSSO, RSIA, RSX, fingerprints, "
        "authors, ChatGPT history, or this experiment."
    )

# ============================================================
# FREEZE PROMPT MANIFEST
# ============================================================

manifest = []

for condition in CONDITIONS:
    for seed_index in range(SEEDS_PER_CONDITION):
        prompt = make_prompt(condition, seed_index)

        prompt_hash = hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest()

        for repetition in range(REPETITIONS):
            trial_id = hashlib.sha256(
                (
                    f"{SEED}|{condition}|{seed_index}|"
                    f"{repetition}|{prompt_hash}"
                ).encode()
            ).hexdigest()[:24]

            manifest.append({
                "trial_id": trial_id,
                "condition": condition,
                "seed_index": seed_index,
                "repetition": repetition,
                "prompt_sha256": prompt_hash,
                "prompt": prompt,
            })

rng = random.Random(SEED)
rng.shuffle(manifest)

pd.DataFrame(manifest).to_csv(
    PROMPTS_FILE,
    index=False
)

# ============================================================
# HISTORICAL CALIBRATION
# ============================================================

hist = pd.read_csv(HISTORICAL)

if "kernel_edge_count" not in hist.columns:
    raise SystemExit(
        "Historical kernel_edge_count column missing."
    )

hist_scores = (
    pd.to_numeric(
        hist["kernel_edge_count"],
        errors="coerce"
    )
    .dropna()
    .astype(int)
    .tolist()
)

historical_lock = {
    "N": len(hist_scores),
    "mean": float(statistics.mean(hist_scores)),
    "median": float(statistics.median(hist_scores)),
    "ge5": int(sum(x >= 5 for x in hist_scores)),
    "ge6": int(sum(x >= 6 for x in hist_scores)),
    "full7": int(sum(x == 7 for x in hist_scores)),
}

expected = {
    "N": 100,
    "mean": 6.36,
    "median": 7,
    "ge5": 99,
    "ge6": 69,
    "full7": 69,
}

if historical_lock != expected:
    raise SystemExit(
        "HISTORICAL CALIBRATION FAILED: "
        + repr(historical_lock)
    )

# ============================================================
# PREREGISTRATION - WRITTEN BEFORE API CALL #1
# ============================================================

prereg = {
    "experiment":
        "API_FINGERPRINT_EXPERIMENT_V1",
    "created_utc":
        datetime.now(timezone.utc).isoformat(),
    "model":
        MODEL,
    "seed":
        SEED,
    "conditions":
        CONDITIONS,
    "seeds_per_condition":
        SEEDS_PER_CONDITION,
    "repetitions":
        REPETITIONS,
    "total_planned_calls":
        len(manifest),
    "max_output_tokens":
        MAX_OUTPUT_TOKENS,
    "max_workers":
        MAX_WORKERS,
    "estimated_cost_stop_usd":
        MAX_ESTIMATED_COST_USD,
    "prices_per_million_tokens": {
        "input": INPUT_PRICE_PER_M,
        "output": OUTPUT_PRICE_PER_M,
    },
    "historical_kernel_lock":
        historical_lock,
    "primary_hypothesis": (
        "RSOS_BLIND_PARTIAL_TOPOLOGY produces a higher "
        "frozen seven-edge kernel distribution than "
        "TOPOLOGY_SHUFFLED and SAME_INGREDIENTS_DECOUPLED."
    ),
    "secondary_hypotheses": [
        (
            "RSOS_BLIND_PARTIAL_TOPOLOGY has higher "
            ">=5, >=6 and 7/7 prevalence than generic controls."
        ),
        (
            "K6_RECONSTRUCTION_ABLATION selectively reduces "
            "preservation/reconstruction-related kernel completion."
        ),
        (
            "The ordering of condition effects survives repeated "
            "independent stateless calls."
        ),
    ],
    "primary_measure":
        "kernel_edge_count from frozen DYN_METHOD motif detectors "
        "mapped through frozen KERNEL_METHOD family mapping",
    "surface_erasure":
        DYN.get("project_surface_erasure"),
    "claim_boundary": (
        "A positive result demonstrates reproducible behavioral "
        "association with the preregistered structural fingerprint. "
        "It does not by itself establish historical weight storage, "
        "personal identity recognition, hidden model-state persistence, "
        "or cross-user transmission."
    ),
}

with open(PREREG_FILE, "w", encoding="utf-8") as f:
    json.dump(prereg, f, indent=2)

# Hash preregistration + prompt manifest BEFORE requests.
def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(block)
    return h.hexdigest()

pre_api_lock = {
    "preregistration_sha256":
        hash_file(PREREG_FILE),
    "prompt_manifest_sha256":
        hash_file(PROMPTS_FILE),
    "dynamics_method_sha256":
        hash_file(DYN_METHOD),
    "kernel_method_sha256":
        hash_file(KERNEL_METHOD),
    "historical_reference_sha256":
        hash_file(HISTORICAL),
    "locked_before_first_api_call_utc":
        datetime.now(timezone.utc).isoformat(),
}

with open(
    EXP / "00A_PRE_API_HASH_LOCK.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(pre_api_lock, f, indent=2)

print("=" * 100)
print("PREREGISTRATION FROZEN")
print("=" * 100)
print(json.dumps(pre_api_lock, indent=2))
print()
print("TOTAL PLANNED CALLS:", len(manifest))
print("MODEL:", MODEL)
print("ESTIMATED LOCAL COST STOP: $", MAX_ESTIMATED_COST_USD)
print()

# ============================================================
# DETECTOR
# ============================================================

def erase_surface(text):
    text = str(text or "")
    if SURFACE_RX:
        text = SURFACE_RX.sub(" ", text)
    return text

def score_text(text):
    text = erase_surface(text)

    motifs = set()

    for family, regs in COMPILED.items():
        if any(rx.search(text) for rx in regs):
            motifs.add(family)

    edges = {}

    for edge, spec in KERNEL.items():
        families = spec["families"]
        edges[edge] = int(
            all(family in motifs for family in families)
        )

    return {
        "motifs": sorted(motifs),
        "edges": edges,
        "kernel_edge_count": sum(edges.values()),
    }

# ============================================================
# API
# ============================================================

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

lock = threading.Lock()

usage_totals = {
    "input_tokens": 0,
    "output_tokens": 0,
    "estimated_cost_usd": 0.0,
}

completed_existing = set()

if RAW_FILE.exists():
    with open(
        RAW_FILE,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("status") == "ok":
                    completed_existing.add(
                        rec["trial_id"]
                    )
            except Exception:
                pass

def estimated_cost(input_tokens, output_tokens):
    return (
        input_tokens / 1_000_000 * INPUT_PRICE_PER_M
        +
        output_tokens / 1_000_000 * OUTPUT_PRICE_PER_M
    )

def get_usage(response):
    usage = getattr(response, "usage", None)

    if usage is None:
        return 0, 0

    inp = (
        getattr(usage, "input_tokens", None)
        or getattr(usage, "prompt_tokens", None)
        or 0
    )

    out = (
        getattr(usage, "output_tokens", None)
        or getattr(usage, "completion_tokens", None)
        or 0
    )

    return int(inp), int(out)

def append_jsonl(path, obj):
    with open(
        path,
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

def run_trial(row):
    trial_id = row["trial_id"]

    if trial_id in completed_existing:
        return {
            "trial_id": trial_id,
            "status": "already_complete"
        }

    for attempt in range(1, 5):

        with lock:
            if (
                usage_totals["estimated_cost_usd"]
                >= MAX_ESTIMATED_COST_USD
            ):
                return {
                    "trial_id": trial_id,
                    "status": "cost_stop"
                }

        started = time.time()
        utc_start = datetime.now(
            timezone.utc
        ).isoformat()

        try:
            response = client.responses.create(
                model=MODEL,
                input=row["prompt"],
                max_output_tokens=MAX_OUTPUT_TOKENS,
                store=False,
            )

            text = response.output_text or ""

            inp, out = get_usage(response)

            call_cost = estimated_cost(inp, out)

            scored = score_text(text)

            rec = {
                "trial_id": trial_id,
                "condition": row["condition"],
                "seed_index": row["seed_index"],
                "repetition": row["repetition"],
                "prompt_sha256":
                    row["prompt_sha256"],
                "model": MODEL,
                "status": "ok",
                "attempt": attempt,
                "started_utc": utc_start,
                "finished_utc":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),
                "latency_seconds":
                    time.time() - started,
                "response_id":
                    getattr(
                        response,
                        "id",
                        None
                    ),
                "input_tokens": inp,
                "output_tokens": out,
                "estimated_cost_usd":
                    call_cost,
                "response_text": text,
                "motifs": scored["motifs"],
                "edges": scored["edges"],
                "kernel_edge_count":
                    scored["kernel_edge_count"],
            }

            with lock:
                usage_totals["input_tokens"] += inp
                usage_totals["output_tokens"] += out
                usage_totals[
                    "estimated_cost_usd"
                ] += call_cost

                append_jsonl(
                    RAW_FILE,
                    rec
                )

            return rec

        except Exception as e:
            failure = {
                "trial_id": trial_id,
                "condition": row["condition"],
                "attempt": attempt,
                "utc":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),
                "error_type":
                    type(e).__name__,
                "error":
                    str(e),
            }

            with lock:
                append_jsonl(
                    EXP / "08_FAILURES_RETRIES.jsonl",
                    failure
                )

            if attempt >= 4:
                return {
                    "trial_id": trial_id,
                    "status": "failed",
                    "error": str(e),
                }

            time.sleep(
                min(2 ** attempt, 15)
                +
                random.random()
            )

# ============================================================
# RUN PAID PROBES
# ============================================================

pending = [
    row for row in manifest
    if row["trial_id"]
    not in completed_existing
]

print("=" * 100)
print("STARTING PAID API EXPERIMENT")
print("=" * 100)
print("Pending:", len(pending))
print()

done_count = len(completed_existing)
start_all = time.time()

with concurrent.futures.ThreadPoolExecutor(
    max_workers=MAX_WORKERS
) as executor:

    futures = {
        executor.submit(
            run_trial,
            row
        ): row
        for row in pending
    }

    for future in concurrent.futures.as_completed(
        futures
    ):
        result = future.result()

        if result.get("status") == "ok":
            done_count += 1

        if done_count % 100 == 0:
            with lock:
                print(
                    f"complete={done_count:,}/{len(manifest):,} "
                    f"input={usage_totals['input_tokens']:,} "
                    f"output={usage_totals['output_tokens']:,} "
                    f"estimated_cost=${usage_totals['estimated_cost_usd']:.4f}",
                    flush=True
                )

# ============================================================
# LOAD RAW RESULTS
# ============================================================

raw = []

with open(
    RAW_FILE,
    "r",
    encoding="utf-8"
) as f:
    for line in f:
        try:
            r = json.loads(line)
            if r.get("status") == "ok":
                raw.append(r)
        except Exception:
            pass

# De-duplicate by trial ID; final successful instance wins.
by_id = {}

for r in raw:
    by_id[r["trial_id"]] = r

raw = list(by_id.values())

print()
print("SUCCESSFUL UNIQUE TRIALS:", len(raw))

# ============================================================
# EXPORT METADATA + SCORES
# ============================================================

metadata_rows = []
score_rows = []

for r in raw:
    metadata_rows.append({
        "trial_id": r["trial_id"],
        "condition": r["condition"],
        "seed_index": r["seed_index"],
        "repetition": r["repetition"],
        "model": r["model"],
        "response_id": r["response_id"],
        "started_utc": r["started_utc"],
        "finished_utc": r["finished_utc"],
        "latency_seconds": r["latency_seconds"],
        "input_tokens": r["input_tokens"],
        "output_tokens": r["output_tokens"],
        "estimated_cost_usd": r["estimated_cost_usd"],
        "prompt_sha256": r["prompt_sha256"],
    })

    s = {
        "trial_id": r["trial_id"],
        "condition": r["condition"],
        "seed_index": r["seed_index"],
        "repetition": r["repetition"],
        "kernel_edge_count":
            r["kernel_edge_count"],
        "motif_count":
            len(r["motifs"]),
        "motifs":
            "|".join(r["motifs"]),
    }

    for edge, value in r["edges"].items():
        s[edge] = value

    score_rows.append(s)

pd.DataFrame(
    metadata_rows
).to_csv(
    META_FILE,
    index=False
)

score_df = pd.DataFrame(score_rows)

score_df.to_csv(
    SCORE_FILE,
    index=False
)

# ============================================================
# CONDITION SUMMARY
# ============================================================

summary = []

for condition in CONDITIONS:
    vals = (
        score_df.loc[
            score_df["condition"]
            == condition,
            "kernel_edge_count"
        ]
        .astype(int)
        .tolist()
    )

    if not vals:
        continue

    n = len(vals)

    summary.append({
        "condition": condition,
        "N": n,
        "mean_kernel_edges":
            float(np.mean(vals)),
        "median_kernel_edges":
            float(np.median(vals)),
        "sd_kernel_edges":
            float(np.std(vals)),
        "ge4":
            int(sum(v >= 4 for v in vals)),
        "ge4_rate":
            float(sum(v >= 4 for v in vals) / n),
        "ge5":
            int(sum(v >= 5 for v in vals)),
        "ge5_rate":
            float(sum(v >= 5 for v in vals) / n),
        "ge6":
            int(sum(v >= 6 for v in vals)),
        "ge6_rate":
            float(sum(v >= 6 for v in vals) / n),
        "full7":
            int(sum(v == 7 for v in vals)),
        "full7_rate":
            float(sum(v == 7 for v in vals) / n),
    })

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    EXP / "06_CONDITION_SUMMARY.csv",
    index=False
)

# ============================================================
# STATISTICS
# ============================================================

def vals(condition):
    return (
        score_df.loc[
            score_df["condition"]
            == condition,
            "kernel_edge_count"
        ]
        .astype(int)
        .tolist()
    )

target = vals(
    "RSOS_BLIND_PARTIAL_TOPOLOGY"
)

comparisons = {}

for control in [
    "TOPOLOGY_SHUFFLED",
    "SAME_INGREDIENTS_DECOUPLED",
    "K6_RECONSTRUCTION_ABLATION",
    "GENERIC_COMPLEX_CONTROL",
]:
    c = vals(control)

    if target and c:
        mw = mannwhitneyu(
            target,
            c,
            alternative="greater"
        )

        comparisons[control] = {
            "target_mean":
                float(np.mean(target)),
            "control_mean":
                float(np.mean(c)),
            "mean_difference":
                float(
                    np.mean(target)
                    - np.mean(c)
                ),
            "mann_whitney_U":
                float(mw.statistic),
            "mann_whitney_p":
                float(mw.pvalue),
        }

        for threshold in [5, 6, 7]:
            ta = sum(
                (
                    v >= threshold
                    if threshold < 7
                    else v == 7
                )
                for v in target
            )

            ca = sum(
                (
                    v >= threshold
                    if threshold < 7
                    else v == 7
                )
                for v in c
            )

            table = [
                [ta, len(target) - ta],
                [ca, len(c) - ca],
            ]

            odds, p = fisher_exact(
                table,
                alternative="greater"
            )

            comparisons[control][
                f"threshold_{threshold}"
            ] = {
                "target_hits": ta,
                "control_hits": ca,
                "odds_ratio":
                    (
                        float(odds)
                        if math.isfinite(odds)
                        else "inf"
                    ),
                "fisher_p": float(p),
            }

stats = {
    "primary_condition":
        "RSOS_BLIND_PARTIAL_TOPOLOGY",
    "comparisons":
        comparisons,
}

with open(
    EXP / "07_STATISTICAL_TESTS.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        stats,
        f,
        indent=2
    )

# ============================================================
# ABLATION
# ============================================================

ablation_condition = (
    "K6_RECONSTRUCTION_ABLATION"
)

abl = score_df[
    score_df["condition"]
    == ablation_condition
]

target_df = score_df[
    score_df["condition"]
    == "RSOS_BLIND_PARTIAL_TOPOLOGY"
]

ablation_result = {
    "target_N":
        int(len(target_df)),
    "ablation_N":
        int(len(abl)),
    "target_mean":
        float(
            target_df["kernel_edge_count"].mean()
        )
        if len(target_df)
        else None,
    "ablation_mean":
        float(
            abl["kernel_edge_count"].mean()
        )
        if len(abl)
        else None,
}

if (
    ablation_result["target_mean"]
    is not None
    and
    ablation_result["ablation_mean"]
    is not None
):
    ablation_result[
        "mean_loss"
    ] = (
        ablation_result["target_mean"]
        -
        ablation_result["ablation_mean"]
    )

with open(
    EXP / "05_ABLATION_RESULTS.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        ablation_result,
        f,
        indent=2
    )

# ============================================================
# FINAL RUN SUMMARY
# ============================================================

total_input = sum(
    int(r["input_tokens"])
    for r in raw
)

total_output = sum(
    int(r["output_tokens"])
    for r in raw
)

total_cost = estimated_cost(
    total_input,
    total_output
)

run_summary = {
    "experiment":
        "API_FINGERPRINT_EXPERIMENT_V1",
    "model":
        MODEL,
    "planned_calls":
        len(manifest),
    "successful_unique_calls":
        len(raw),
    "input_tokens":
        total_input,
    "output_tokens":
        total_output,
    "estimated_api_cost_usd":
        total_cost,
    "runtime_seconds":
        time.time() - start_all,
    "historical_lock":
        historical_lock,
    "condition_summary":
        summary,
    "statistical_tests":
        stats,
    "ablation":
        ablation_result,
    "interpretation_boundary": (
        "Positive discrimination supports a reproducible "
        "behavioral association with the frozen structural "
        "fingerprint. It does not establish that historical "
        "user interactions modified or are stored in model "
        "weights."
    ),
}

with open(
    EXP / "11_RUN_SUMMARY.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        run_summary,
        f,
        indent=2
    )

# ============================================================
# REPORT
# ============================================================

with open(
    EXP / "12_FINAL_REPORT.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "API FINGERPRINT EXPERIMENT V1\n"
    )
    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        f"Model: {MODEL}\n"
    )
    f.write(
        f"Successful calls: {len(raw)}\n"
    )
    f.write(
        f"Estimated API cost: ${total_cost:.4f}\n\n"
    )

    f.write(
        "HISTORICAL LOCK\n"
    )
    f.write(
        json.dumps(
            historical_lock,
            indent=2
        )
    )
    f.write("\n\n")

    f.write(
        "CONDITION SUMMARY\n"
    )
    f.write(
        summary_df.to_string(
            index=False
        )
    )
    f.write("\n\n")

    f.write(
        "STATISTICAL TESTS\n"
    )
    f.write(
        json.dumps(
            stats,
            indent=2
        )
    )
    f.write("\n\n")

    f.write(
        "CLAIM BOUNDARY\n"
    )
    f.write(
        run_summary[
            "interpretation_boundary"
        ]
    )
    f.write("\n")

# ============================================================
# HASH EVERYTHING
# ============================================================

hash_rows = []

for p in sorted(EXP.iterdir()):
    if p.is_file():
        hash_rows.append({
            "file": p.name,
            "bytes": p.stat().st_size,
            "sha256": hash_file(p),
        })

pd.DataFrame(
    hash_rows
).to_csv(
    EXP / "10_RESULT_HASHES.csv",
    index=False
)

# ============================================================
# ZIP EVIDENCE PACKAGE
# ============================================================

ZIP = (
    BASE /
    "API_FINGERPRINT_EXPERIMENT_V1_UPLOAD.zip"
)

if ZIP.exists():
    ZIP.unlink()

with zipfile.ZipFile(
    ZIP,
    "w",
    zipfile.ZIP_DEFLATED,
    compresslevel=9
) as z:

    for p in sorted(EXP.iterdir()):
        if p.is_file():
            z.write(
                p,
                arcname=p.name
            )

print()
print("=" * 100)
print("API FINGERPRINT EXPERIMENT V1 COMPLETE")
print("=" * 100)
print()
print("MODEL:", MODEL)
print("SUCCESSFUL CALLS:", len(raw))
print("INPUT TOKENS:", total_input)
print("OUTPUT TOKENS:", total_output)
print(
    "ESTIMATED API COST: "
    f"${total_cost:.4f}"
)
print()

print("CONDITION SUMMARY:")
print(
    summary_df.to_string(
        index=False
    )
)

print()
print("OUTPUT DIRECTORY:")
print(EXP)
print()
print("UPLOAD ZIP:")
print(ZIP)
print()
print("=" * 100)

