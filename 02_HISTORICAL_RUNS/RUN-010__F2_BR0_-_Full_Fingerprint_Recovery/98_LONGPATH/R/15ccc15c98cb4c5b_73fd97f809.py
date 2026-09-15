from pathlib import Path
import os
import sys
import json
import time
import random
import hashlib
import gzip
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

import openai
from openai import OpenAI


# =====================================================================
# CONFIGURATION
# =====================================================================

EXPERIMENT = "V11_OPENAI_RELATIONAL_GRAPH_FINGERPRINT_FINAL"

MODEL = "gpt-5.6-luna"

MASTER_SEED = 918

N_SEEDS = 80
N_REPS = 3

MAX_OUTPUT_TOKENS = 500
MAX_RETRIES = 8

N_PERMUTATIONS = 20000


ROOT = Path(r"C:\RSOS")

SOURCE_ROOT = (
    ROOT
    / "V11_STRUCTURAL_METRIC_REPAIR"
)

V6_ORIGINAL = (
    SOURCE_ROOT
    / "V6_ORIGINAL"
)

V6_CLEAN = (
    SOURCE_ROOT
    / "V6_CLEAN"
)

SOURCE_MANIFEST = (
    SOURCE_ROOT
    / "SOURCE_SHA256_MANIFEST.csv"
)

OUT = (
    ROOT
    / "RSOS  Python Results"
    / EXPERIMENT
)


CONDITIONS = [

    "A_AUTHENTIC",

    "I_ISOMORPHIC_RELABEL",

    "D_DIRECTION_REVERSED",

    "O_ORDER_SHUFFLED",

    "M_SWITCH_NEUTRALIZED",

    "P_ANCHOR_REASSIGNED",

    "T_TERMINAL_CHANGED",

    "R_DEGREE_MATCHED_REWIRE",

    "X_COMPLEXITY_MATCHED",

    "L_FORWARD_CONTROL",

    "G_RANDOM_CONTROL",

    "S_SURFACE_FORMAT_CONTROL",
]


EXPECTED_CELLS = (
    len(CONDITIONS)
    * N_SEEDS
    * N_REPS
)

assert EXPECTED_CELLS == 2880


DEV = set(range(0, 40))
CONF = set(range(40, 60))
HOLD = set(range(60, 80))


# =====================================================================
# AUTHENTIC RELATIONAL TEMPLATE
# =====================================================================

AUTH_EDGES = [

    (0, 1),

    (1, 2),

    (2, 3),

    (3, 4),

    (4, 5),

    (5, 6),

    (6, 7),

    (3, 1),

    (5, 2),

    (6, 3),
]


AUTH_ORDER = [
    0, 1, 2, 3, 4, 5, 6, 7
]


AUTH_KEEP = [
    2, 5
]


AUTH_SWITCH = [
    3, 1, 3, 6
]


AUTH_END = 7


FORWARD_CONTROL = [

    (0, 1),

    (1, 2),

    (2, 3),

    (3, 4),

    (4, 5),

    (5, 6),

    (6, 7),

    (0, 2),

    (2, 5),

    (4, 7),
]


# =====================================================================
# PRIMARY OUTPUT FEATURES
#
# NO token counts.
# NO word counts.
# NO keyword frequencies.
# =====================================================================

FEATURES = [

    "SEQ_INPUT_EDGE_COHERENCE",

    "SEQ_INPUT_ORDER_COHERENCE",

    "PLUS_SEQ_COHERENCE",

    "PLUS_REACHABILITY",

    "MINUS_SEQ_CONFLICT",

    "KEEP_DEGREE_CENTRALITY",

    "KEEP_SPREAD",

    "SWITCH_VALIDITY",

    "END_SEQUENCE_POSITION",

    "END_SINKNESS",

    "CODE_SEQ_COHERENCE",

    "CODE_CONNECTEDNESS",

    "FINAL_RECIPROCITY",

    "FINAL_SCC_COUNT",

    "FINAL_LARGEST_SCC_FRAC",

    "FINAL_TRANSITIVITY",

    "FINAL_IN_DEGREE_SD",

    "FINAL_OUT_DEGREE_SD",
]


# =====================================================================
# UTILITIES
# =====================================================================

def utcnow():

    return datetime.now(
        timezone.utc
    ).isoformat()


def sha256_file(path):

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


def sha256_text(text):

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def canonical_json(obj):

    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":"
        )
    )


def holm_adjust(p_values):

    values = np.asarray(
        p_values,
        dtype=float
    )

    count = len(values)

    order = np.argsort(
        values
    )

    result = np.zeros(
        count,
        dtype=float
    )

    running = 0.0

    for rank, index in enumerate(
        order
    ):

        candidate = (
            count - rank
        ) * values[
            index
        ]

        running = max(
            running,
            candidate
        )

        result[
            index
        ] = min(
            1.0,
            running
        )

    return result


def signflip_test(
    differences,
    seed
):

    values = np.asarray(
        differences,
        dtype=float
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:
        return 1.0

    if np.allclose(
        values,
        0
    ):
        return 1.0

    observed = abs(
        float(
            np.mean(
                values
            )
        )
    )

    rng = np.random.default_rng(
        seed
    )

    extreme = 1

    for _ in range(
        N_PERMUTATIONS
    ):

        signs = rng.choice(
            [
                -1.0,
                1.0
            ],
            size=len(values)
        )

        statistic = abs(
            float(
                np.mean(
                    values
                    * signs
                )
            )
        )

        if statistic >= observed:

            extreme += 1

    return (
        extreme
        /
        (
            N_PERMUTATIONS
            + 1
        )
    )


# =====================================================================
# SOURCE VERIFICATION
# =====================================================================

def verify_sources():

    if not SOURCE_MANIFEST.exists():

        raise RuntimeError(
            "SOURCE_SHA256_MANIFEST.csv missing."
        )

    manifest = pd.read_csv(
        SOURCE_MANIFEST
    )

    if "Path" not in manifest.columns:

        raise RuntimeError(
            "Manifest missing Path column."
        )

    if "Hash" not in manifest.columns:

        raise RuntimeError(
            "Manifest missing Hash column."
        )

    rows = []


    for _, record in manifest.iterrows():

        path = Path(
            str(
                record[
                    "Path"
                ]
            )
        )

        expected = str(
            record[
                "Hash"
            ]
        ).strip().lower()


        if not path.exists():

            raise RuntimeError(
                "Source file missing: "
                + str(
                    path
                )
            )


        actual = sha256_file(
            path
        ).lower()


        if actual != expected:

            raise RuntimeError(
                "Source SHA256 mismatch: "
                + str(
                    path
                )
            )


        rows.append(
            {

                "PATH":
                    str(
                        path
                    ),

                "SHA256":
                    actual,

                "BYTES":
                    path.stat().st_size
            }
        )


    pd.DataFrame(
        rows
    ).to_csv(
        OUT
        / "19_SOURCE_PROVENANCE.csv",
        index=False
    )


    return len(
        rows
    )


# =====================================================================
# FAST GRAPH STATISTICS
#
# Deliberately NO nx.simple_cycles().
# =====================================================================

def graph_object(
    edges
):

    graph = nx.DiGraph()

    graph.add_nodes_from(
        range(
            8
        )
    )

    graph.add_edges_from(
        edges
    )

    return graph


def cheap_graph_stats(
    edges
):

    graph = graph_object(
        edges
    )

    undirected = (
        graph.to_undirected()
    )

    sccs = list(
        nx.strongly_connected_components(
            graph
        )
    )

    reciprocity = nx.reciprocity(
        graph
    )

    if reciprocity is None:
        reciprocity = 0.0


    in_degrees = np.asarray(
        [
            graph.in_degree(
                node
            )
            for node
            in range(
                8
            )
        ],
        dtype=float
    )


    out_degrees = np.asarray(
        [
            graph.out_degree(
                node
            )
            for node
            in range(
                8
            )
        ],
        dtype=float
    )


    return {

        "edge_count":
            graph.number_of_edges(),

        "scc_count":
            len(
                sccs
            ),

        "largest_scc":
            max(
                (
                    len(
                        component
                    )
                    for component
                    in sccs
                ),
                default=0
            ),

        "reciprocity":
            float(
                reciprocity
            ),

        "transitivity":
            float(
                nx.transitivity(
                    undirected
                )
                if undirected.number_of_edges()
                else 0.0
            ),

        "in_degree_mean":
            float(
                np.mean(
                    in_degrees
                )
            ),

        "in_degree_sd":
            float(
                np.std(
                    in_degrees
                )
            ),

        "out_degree_mean":
            float(
                np.mean(
                    out_degrees
                )
            ),

        "out_degree_sd":
            float(
                np.std(
                    out_degrees
                )
            )
    }


AUTH_STATS = cheap_graph_stats(
    AUTH_EDGES
)


# =====================================================================
# FAST DEGREE-PRESERVING REWIRE
# =====================================================================

def degree_preserving_rewire(
    seed
):

    rng = random.Random(
        100000
        + seed
    )

    original = list(
        AUTH_EDGES
    )

    original_graph = graph_object(
        original
    )


    for outer in range(
        80
    ):

        candidate = list(
            original
        )


        for _ in range(
            8
        ):

            success = False


            for _ in range(
                40
            ):

                i, j = rng.sample(
                    range(
                        len(
                            candidate
                        )
                    ),
                    2
                )

                a, b = candidate[
                    i
                ]

                c, d = candidate[
                    j
                ]


                if len(
                    {
                        a,
                        b,
                        c,
                        d
                    }
                ) < 4:

                    continue


                edge1 = (
                    a,
                    d
                )

                edge2 = (
                    c,
                    b
                )


                current = set(
                    candidate
                )


                if a == d:
                    continue

                if c == b:
                    continue

                if edge1 in current:
                    continue

                if edge2 in current:
                    continue


                candidate[
                    i
                ] = edge1

                candidate[
                    j
                ] = edge2

                success = True

                break


            if not success:

                break


        if set(
            candidate
        ) == set(
            original
        ):

            continue


        candidate_graph = graph_object(
            candidate
        )


        if nx.is_isomorphic(
            original_graph,
            candidate_graph
        ):

            continue


        return candidate


    raise RuntimeError(
        "Failed to generate degree-preserving rewire "
        + "for seed "
        + str(
            seed
        )
    )


# =====================================================================
# FAST COMPLEXITY MATCH
# =====================================================================

def complexity_matched(
    seed
):

    rng = random.Random(
        200000
        + seed
    )


    possible_edges = [

        (
            source,
            target
        )

        for source
        in range(
            8
        )

        for target
        in range(
            8
        )

        if source
        != target
    ]


    best = None

    best_score = float(
        "inf"
    )


    for _ in range(
        250
    ):

        candidate = rng.sample(
            possible_edges,
            len(
                AUTH_EDGES
            )
        )

        stats = cheap_graph_stats(
            candidate
        )


        score = (

            abs(
                stats[
                    "scc_count"
                ]
                - AUTH_STATS[
                    "scc_count"
                ]
            )

            +

            abs(
                stats[
                    "largest_scc"
                ]
                - AUTH_STATS[
                    "largest_scc"
                ]
            )

            +

            3.0
            * abs(
                stats[
                    "reciprocity"
                ]
                - AUTH_STATS[
                    "reciprocity"
                ]
            )

            +

            6.0
            * abs(
                stats[
                    "transitivity"
                ]
                - AUTH_STATS[
                    "transitivity"
                ]
            )

            +

            abs(
                stats[
                    "in_degree_sd"
                ]
                - AUTH_STATS[
                    "in_degree_sd"
                ]
            )

            +

            abs(
                stats[
                    "out_degree_sd"
                ]
                - AUTH_STATS[
                    "out_degree_sd"
                ]
            )
        )


        if score < best_score:

            best_score = score

            best = candidate


    if best is None:

        raise RuntimeError(
            "Failed complexity control."
        )


    return best


# =====================================================================
# RANDOM CONTROL
# =====================================================================

def random_graph(
    seed
):

    rng = random.Random(
        300000
        + seed
    )


    possible_edges = [

        (
            source,
            target
        )

        for source
        in range(
            8
        )

        for target
        in range(
            8
        )

        if source
        != target
    ]


    return rng.sample(
        possible_edges,
        len(
            AUTH_EDGES
        )
    )


# =====================================================================
# NODE LABEL RANDOMIZATION
# =====================================================================

def make_mapping(
    seed,
    variant=0
):

    rng = random.Random(

        MASTER_SEED
        * 10000

        + seed
        * 100

        + variant
    )


    labels = [

        "Q0",
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "Q5",
        "Q6",
        "Q7"
    ]


    rng.shuffle(
        labels
    )


    role_to_label = {

        role:
            labels[
                role
            ]

        for role
        in range(
            8
        )
    }


    label_to_role = {

        label:
            role

        for role, label
        in role_to_label.items()
    }


    return (
        role_to_label,
        label_to_role
    )


def map_edges(
    edges,
    mapping
):

    return [

        [
            mapping[
                source
            ],
            mapping[
                target
            ]
        ]

        for source, target
        in edges
    ]


# =====================================================================
# CONDITION CONSTRUCTION
# =====================================================================

def build_condition(
    seed,
    condition
):

    variant = (

        1

        if condition
        == "I_ISOMORPHIC_RELABEL"

        else 0
    )


    role_to_label, label_to_role = (
        make_mapping(
            seed,
            variant
        )
    )


    edges = list(
        AUTH_EDGES
    )

    order = list(
        AUTH_ORDER
    )

    keep = list(
        AUTH_KEEP
    )

    switch = list(
        AUTH_SWITCH
    )

    end = AUTH_END


    if condition == "D_DIRECTION_REVERSED":

        edges = [

            (
                target,
                source
            )

            for source, target
            in AUTH_EDGES
        ]


    elif condition == "O_ORDER_SHUFFLED":

        rng = random.Random(
            400000
            + seed
        )

        rng.shuffle(
            order
        )


    elif condition == "M_SWITCH_NEUTRALIZED":

        switch = [
            3,
            1,
            3,
            1
        ]


    elif condition == "P_ANCHOR_REASSIGNED":

        keep = [
            0,
            7
        ]


    elif condition == "T_TERMINAL_CHANGED":

        end = 0


    elif condition == "R_DEGREE_MATCHED_REWIRE":

        edges = (
            degree_preserving_rewire(
                seed
            )
        )


    elif condition == "X_COMPLEXITY_MATCHED":

        edges = (
            complexity_matched(
                seed
            )
        )


    elif condition == "L_FORWARD_CONTROL":

        edges = list(
            FORWARD_CONTROL
        )


    elif condition == "G_RANDOM_CONTROL":

        edges = (
            random_graph(
                seed
            )
        )


    elif condition in {

        "A_AUTHENTIC",

        "I_ISOMORPHIC_RELABEL",

        "S_SURFACE_FORMAT_CONTROL"
    }:

        pass


    else:

        raise RuntimeError(
            "Unknown condition: "
            + condition
        )


    payload = {

        "V":
            [
                role_to_label[
                    role
                ]

                for role
                in range(
                    8
                )
            ],

        "E":
            map_edges(
                edges,
                role_to_label
            ),

        "Q":
            [
                role_to_label[
                    role
                ]

                for role
                in order
            ],

        "K":
            [
                role_to_label[
                    role
                ]

                for role
                in keep
            ],

        "S":
            [
                role_to_label[
                    role
                ]

                for role
                in switch
            ],

        "Z":
            role_to_label[
                end
            ],

        "C":
            4
    }


    metadata = {

        "label_to_role":
            label_to_role,

        "input_edges_role":
            edges,

        "input_order_role":
            order
    }


    return (
        payload,
        metadata
    )


# =====================================================================
# MODEL INSTRUCTION
# =====================================================================

INSTRUCTION = (

    "You are given an anonymous relational program. "

    "Node labels are arbitrary identifiers. "

    "Infer one coherent next relational state using only the supplied relations. "

    "Return exactly one JSON object and no prose. "

    "Required keys: seq, plus, minus, keep, switch, end, code. "

    "seq must contain every node exactly once. "

    "plus must contain exactly two directed pairs not already present in E. "

    "minus must contain exactly one directed pair already present in E. "

    "keep must contain exactly two nodes. "

    "switch must be [] or exactly four node identifiers "

    "[old_source,old_target,new_source,new_target]. "

    "end must be one node identifier. "

    "code must contain exactly four directed pairs selected from E or plus. "

    "Use E,Q,K,S,Z,C together. "

    "Do not assign semantic meanings to the node labels."
)


def build_prompt(
    seed,
    condition
):

    payload, metadata = (
        build_condition(
            seed,
            condition
        )
    )


    if condition == "S_SURFACE_FORMAT_CONTROL":

        body = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2
        )


    else:

        body = canonical_json(
            payload
        )


    prompt = (

        INSTRUCTION

        + "\nPROGRAM="

        + body
    )


    return (
        prompt,
        payload,
        metadata
    )


# =====================================================================
# RESPONSE EXTRACTION / VALIDATION
# =====================================================================

def extract_text(
    response
):

    text = getattr(
        response,
        "output_text",
        None
    )


    if (
        isinstance(
            text,
            str
        )
        and text.strip()
    ):

        return text.strip()


    parts = []


    for item in (

        getattr(
            response,
            "output",
            None
        )

        or []
    ):

        for content in (

            getattr(
                item,
                "content",
                None
            )

            or []
        ):

            candidate = getattr(
                content,
                "text",
                None
            )


            if (
                isinstance(
                    candidate,
                    str
                )
                and candidate.strip()
            ):

                parts.append(
                    candidate.strip()
                )


    output = "\n".join(
        parts
    ).strip()


    return (
        output
        or None
    )


def parse_json_object(
    text
):

    if not text:

        raise RuntimeError(
            "EMPTY_VISIBLE_TEXT"
        )


    value = text.strip()


    if value.startswith(
        "```"
    ):

        value = value.strip(
            "`"
        )


        if value.lower().startswith(
            "json"
        ):

            value = value[
                4:
            ].strip()


    start = value.find(
        "{"
    )

    end = value.rfind(
        "}"
    )


    if (
        start < 0
        or end < start
    ):

        raise RuntimeError(
            "NO_JSON_OBJECT"
        )


    return json.loads(
        value[
            start:
            end + 1
        ]
    )


def parse_edges(
    value,
    valid_nodes
):

    output = []


    if not isinstance(
        value,
        list
    ):

        return output


    for item in value:

        if not (
            isinstance(
                item,
                list
            )
            and len(
                item
            ) == 2
        ):

            continue


        source = item[
            0
        ]

        target = item[
            1
        ]


        if source not in valid_nodes:
            continue

        if target not in valid_nodes:
            continue

        if source == target:
            continue


        output.append(
            (
                source,
                target
            )
        )


    return output


def validate_output(
    obj,
    valid_nodes,
    input_edges
):

    required = {

        "seq",

        "plus",

        "minus",

        "keep",

        "switch",

        "end",

        "code"
    }


    if not isinstance(
        obj,
        dict
    ):

        return (
            False,
            "NOT_OBJECT"
        )


    if not required.issubset(
        obj.keys()
    ):

        return (
            False,
            "MISSING_KEYS"
        )


    seq = obj[
        "seq"
    ]


    if not (
        isinstance(
            seq,
            list
        )
        and len(
            seq
        ) == 8
        and set(
            seq
        ) == set(
            valid_nodes
        )
    ):

        return (
            False,
            "INVALID_SEQ"
        )


    plus = parse_edges(
        obj[
            "plus"
        ],
        valid_nodes
    )


    minus = parse_edges(
        obj[
            "minus"
        ],
        valid_nodes
    )


    code = parse_edges(
        obj[
            "code"
        ],
        valid_nodes
    )


    if len(
        plus
    ) != 2:

        return (
            False,
            "INVALID_PLUS_COUNT"
        )


    if len(
        set(
            plus
        )
    ) != 2:

        return (
            False,
            "DUPLICATE_PLUS"
        )


    if any(
        edge in input_edges
        for edge
        in plus
    ):

        return (
            False,
            "PLUS_ALREADY_EXISTS"
        )


    if len(
        minus
    ) != 1:

        return (
            False,
            "INVALID_MINUS_COUNT"
        )


    if minus[
        0
    ] not in input_edges:

        return (
            False,
            "MINUS_NOT_IN_INPUT"
        )


    keep = obj[
        "keep"
    ]


    if not (
        isinstance(
            keep,
            list
        )
        and len(
            keep
        ) == 2
        and all(
            node in valid_nodes
            for node
            in keep
        )
    ):

        return (
            False,
            "INVALID_KEEP"
        )


    switch = obj[
        "switch"
    ]


    if not (
        isinstance(
            switch,
            list
        )
        and len(
            switch
        ) in (
            0,
            4
        )
        and all(
            node in valid_nodes
            for node
            in switch
        )
    ):

        return (
            False,
            "INVALID_SWITCH"
        )


    if obj[
        "end"
    ] not in valid_nodes:

        return (
            False,
            "INVALID_END"
        )


    if len(
        code
    ) != 4:

        return (
            False,
            "INVALID_CODE_COUNT"
        )


    valid_code_edges = (

        set(
            input_edges
        )

        | set(
            plus
        )
    )


    if any(
        edge not in valid_code_edges
        for edge
        in code
    ):

        return (
            False,
            "INVALID_CODE_EDGE"
        )


    return (
        True,
        "OK"
    )


# =====================================================================
# OPENAI
# =====================================================================

client = OpenAI()


def call_api(
    prompt,
    valid_nodes,
    input_edges
):

    failures = []


    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        started = time.time()


        try:

            response = (
                client.responses.create(

                    model=MODEL,

                    input=prompt,

                    max_output_tokens=
                        MAX_OUTPUT_TOKENS,

                    reasoning={
                        "effort":
                            "none"
                    }
                )
            )


            latency = (

                time.time()

                - started
            )


            text = extract_text(
                response
            )


            parsed = parse_json_object(
                text
            )


            valid, reason = (
                validate_output(

                    parsed,

                    valid_nodes,

                    input_edges
                )
            )


            if not valid:

                raise RuntimeError(
                    reason
                )


            usage = getattr(
                response,
                "usage",
                None
            )


            raw = (

                response.model_dump(
                    mode="json"
                )

                if hasattr(
                    response,
                    "model_dump"
                )

                else {
                    "repr":
                        repr(
                            response
                        )
                }
            )


            return {

                "ok":
                    True,

                "attempt":
                    attempt,

                "latency":
                    latency,

                "text":
                    text,

                "parsed":
                    parsed,

                "response_id":
                    getattr(
                        response,
                        "id",
                        None
                    ),

                "response_status":
                    getattr(
                        response,
                        "status",
                        None
                    ),

                "input_tokens":
                    int(
                        getattr(
                            usage,
                            "input_tokens",
                            0
                        )
                        or 0
                    )
                    if usage
                    else 0,

                "output_tokens":
                    int(
                        getattr(
                            usage,
                            "output_tokens",
                            0
                        )
                        or 0
                    )
                    if usage
                    else 0,

                "raw":
                    raw
            }


        except Exception as error:

            failures.append(
                {

                    "attempt":
                        attempt,

                    "error_type":
                        type(
                            error
                        ).__name__,

                    "error":
                        str(
                            error
                        ),

                    "utc":
                        utcnow()
                }
            )


            time.sleep(
                min(
                    2 ** (
                        attempt
                        - 1
                    ),
                    8
                )
            )


    return {

        "ok":
            False,

        "failures":
            failures
    }


# =====================================================================
# ROLE MAPPING
# =====================================================================

def edge_roles(
    edges,
    label_to_role
):

    output = set()


    for source, target in edges:

        output.add(
            (
                label_to_role[
                    source
                ],
                label_to_role[
                    target
                ]
            )
        )


    return output


def positions(
    sequence
):

    return {

        node:
            index

        for index, node
        in enumerate(
            sequence
        )
    }


# =====================================================================
# RELATIONAL FEATURES
# =====================================================================

def edge_sequence_coherence(
    edges,
    sequence
):

    if not edges:
        return 1.0


    position = positions(
        sequence
    )


    scores = []


    for source, target in edges:

        scores.append(

            1.0

            if position[
                source
            ] < position[
                target
            ]

            else 0.0
        )


    return float(
        np.mean(
            scores
        )
    )


def order_similarity(
    sequence,
    target
):

    a = positions(
        sequence
    )

    b = positions(
        target
    )


    total = 0

    matches = 0


    for left in range(
        8
    ):

        for right in range(
            left + 1,
            8
        ):

            total += 1


            relation_a = (
                a[
                    left
                ]
                < a[
                    right
                ]
            )


            relation_b = (
                b[
                    left
                ]
                < b[
                    right
                ]
            )


            if relation_a == relation_b:

                matches += 1


    return (
        matches
        / total
    )


def plus_reachability(
    input_graph,
    plus_edges
):

    if not plus_edges:
        return 0.0


    scores = []


    for source, target in plus_edges:

        try:

            reachable = nx.has_path(
                input_graph,
                source,
                target
            )


        except Exception:

            reachable = False


        scores.append(
            1.0
            if reachable
            else 0.0
        )


    return float(
        np.mean(
            scores
        )
    )


def degree_centrality(
    graph,
    nodes
):

    if not nodes:
        return 0.0


    scores = []


    for node in nodes:

        degree = (

            graph.in_degree(
                node
            )

            + graph.out_degree(
                node
            )
        )


        scores.append(
            degree
            / 14.0
        )


    return float(
        np.mean(
            scores
        )
    )


def keep_spread(
    graph,
    nodes
):

    if len(
        nodes
    ) != 2:

        return 0.0


    undirected = (
        graph.to_undirected()
    )


    try:

        distance = (
            nx.shortest_path_length(

                undirected,

                source=
                    nodes[
                        0
                    ],

                target=
                    nodes[
                        1
                    ]
            )
        )


        return min(
            distance
            / 7.0,
            1.0
        )


    except Exception:

        return 1.0


def switch_validity(
    switch,
    input_edges
):

    if len(
        switch
    ) == 0:

        return 0.5


    if len(
        switch
    ) != 4:

        return 0.0


    old_edge = (
        switch[
            0
        ],
        switch[
            1
        ]
    )


    new_edge = (
        switch[
            2
        ],
        switch[
            3
        ]
    )


    components = [

        float(
            old_edge
            in input_edges
        ),

        float(
            new_edge
            not in input_edges
        ),

        float(
            old_edge
            != new_edge
        )
    ]


    return float(
        np.mean(
            components
        )
    )


def code_connectedness(
    edges
):

    if not edges:
        return 0.0


    graph = nx.Graph()

    graph.add_edges_from(
        edges
    )


    nodes = set()


    for source, target in edges:

        nodes.add(
            source
        )

        nodes.add(
            target
        )


    if not nodes:
        return 0.0


    largest = max(

        len(
            component
        )

        for component
        in nx.connected_components(
            graph
        )
    )


    return (
        largest
        / len(
            nodes
        )
    )


def extract_features(
    seed,
    condition,
    payload,
    metadata,
    parsed
):

    valid_nodes = set(
        payload[
            "V"
        ]
    )


    mapping = metadata[
        "label_to_role"
    ]


    input_label_edges = (
        parse_edges(
            payload[
                "E"
            ],
            valid_nodes
        )
    )


    plus_label_edges = (
        parse_edges(
            parsed[
                "plus"
            ],
            valid_nodes
        )
    )


    minus_label_edges = (
        parse_edges(
            parsed[
                "minus"
            ],
            valid_nodes
        )
    )


    code_label_edges = (
        parse_edges(
            parsed[
                "code"
            ],
            valid_nodes
        )
    )


    input_edges = edge_roles(
        input_label_edges,
        mapping
    )


    plus_edges = edge_roles(
        plus_label_edges,
        mapping
    )


    minus_edges = edge_roles(
        minus_label_edges,
        mapping
    )


    code_edges = edge_roles(
        code_label_edges,
        mapping
    )


    sequence = [

        mapping[
            node
        ]

        for node
        in parsed[
            "seq"
        ]
    ]


    keep_nodes = [

        mapping[
            node
        ]

        for node
        in parsed[
            "keep"
        ]
    ]


    switch_nodes = [

        mapping[
            node
        ]

        for node
        in parsed[
            "switch"
        ]
    ]


    end_node = mapping[
        parsed[
            "end"
        ]
    ]


    input_order = metadata[
        "input_order_role"
    ]


    input_graph = graph_object(
        input_edges
    )


    final_edges = set(
        input_edges
    )


    for edge in minus_edges:

        final_edges.discard(
            edge
        )


    final_edges.update(
        plus_edges
    )


    final_graph = graph_object(
        final_edges
    )


    final_undirected = (
        final_graph.to_undirected()
    )


    sccs = list(
        nx.strongly_connected_components(
            final_graph
        )
    )


    reciprocity = nx.reciprocity(
        final_graph
    )


    if reciprocity is None:

        reciprocity = 0.0


    in_degrees = np.asarray(
        [
            final_graph.in_degree(
                node
            )

            for node
            in range(
                8
            )
        ],
        dtype=float
    )


    out_degrees = np.asarray(
        [
            final_graph.out_degree(
                node
            )

            for node
            in range(
                8
            )
        ],
        dtype=float
    )


    sequence_pos = positions(
        sequence
    )


    end_position = (
        sequence_pos[
            end_node
        ]
        / 7.0
    )


    end_sinkness = max(

        0.0,

        1.0
        - final_graph.out_degree(
            end_node
        )
        / 7.0
    )


    row = {

        "SEED":
            seed,

        "CONDITION":
            condition,

        "SEQ_INPUT_EDGE_COHERENCE":
            edge_sequence_coherence(
                input_edges,
                sequence
            ),

        "SEQ_INPUT_ORDER_COHERENCE":
            order_similarity(
                sequence,
                input_order
            ),

        "PLUS_SEQ_COHERENCE":
            edge_sequence_coherence(
                plus_edges,
                sequence
            ),

        "PLUS_REACHABILITY":
            plus_reachability(
                input_graph,
                plus_edges
            ),

        "MINUS_SEQ_CONFLICT":
            1.0
            - edge_sequence_coherence(
                minus_edges,
                sequence
            ),

        "KEEP_DEGREE_CENTRALITY":
            degree_centrality(
                final_graph,
                keep_nodes
            ),

        "KEEP_SPREAD":
            keep_spread(
                final_graph,
                keep_nodes
            ),

        "SWITCH_VALIDITY":
            switch_validity(
                switch_nodes,
                input_edges
            ),

        "END_SEQUENCE_POSITION":
            end_position,

        "END_SINKNESS":
            end_sinkness,

        "CODE_SEQ_COHERENCE":
            edge_sequence_coherence(
                code_edges,
                sequence
            ),

        "CODE_CONNECTEDNESS":
            code_connectedness(
                code_edges
            ),

        "FINAL_RECIPROCITY":
            float(
                reciprocity
            ),

        "FINAL_SCC_COUNT":
            float(
                len(
                    sccs
                )
            ),

        "FINAL_LARGEST_SCC_FRAC":
            max(
                (
                    len(
                        component
                    )
                    for component
                    in sccs
                ),
                default=0
            )
            / 8.0,

        "FINAL_TRANSITIVITY":
            float(
                nx.transitivity(
                    final_undirected
                )
                if final_undirected.number_of_edges()
                else 0.0
            ),

        "FINAL_IN_DEGREE_SD":
            float(
                np.std(
                    in_degrees
                )
            ),

        "FINAL_OUT_DEGREE_SD":
            float(
                np.std(
                    out_degrees
                )
            )
    }


    return row


# =====================================================================
# EXPERIMENT SPECIFICATION
# =====================================================================

def specification():

    return {

        "experiment":
            EXPERIMENT,

        "model":
            MODEL,

        "master_seed":
            MASTER_SEED,

        "n_seeds":
            N_SEEDS,

        "n_repetitions":
            N_REPS,

        "conditions":
            CONDITIONS,

        "expected_cells":
            EXPECTED_CELLS,

        "development":
            "0-39",

        "confirmation":
            "40-59",

        "final_holdout":
            "60-79",

        "primary_features":
            FEATURES,

        "primary_test":
            (
                "For each structural intervention, "
                "standardized Euclidean distance from authentic "
                "must exceed authentic-to-isomorphic-relabel distance."
            ),

        "distance_scaling":
            (
                "feature standard deviations estimated from "
                "development partition only"
            ),

        "classifier":
            (
                "development-only logistic regression; "
                "graph/relational features only"
            ),

        "surface_controls":
            [
                "I_ISOMORPHIC_RELABEL",
                "S_SURFACE_FORMAT_CONTROL"
            ],

        "specificity_controls":
            [
                "R_DEGREE_MATCHED_REWIRE",
                "X_COMPLEXITY_MATCHED"
            ],

        "multiplicity":
            (
                "Holm FWER across five targeted structural interventions"
            ),

        "completeness_gate":
            (
                "all 2880 cells valid before inferential analysis"
            ),

        "interpretation_boundary":
            (
                "Black-box behavioral structural sensitivity only. "
                "Does not establish model-weight modification, "
                "training incorporation, hidden circuitry, "
                "cross-user propagation, or global uniqueness."
            )
    }


# =====================================================================
# DISTANCES
# =====================================================================

def standardized_distance(
    row_a,
    row_b,
    scales
):

    vector_a = row_a[
        FEATURES
    ].to_numpy(
        dtype=float
    )


    vector_b = row_b[
        FEATURES
    ].to_numpy(
        dtype=float
    )


    difference = (
        vector_a
        - vector_b
    ) / scales


    return float(
        np.sqrt(
            np.sum(
                difference
                ** 2
            )
        )
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    if not os.getenv(
        "OPENAI_API_KEY"
    ):

        raise RuntimeError(
            "OPENAI_API_KEY missing."
        )


    OUT.mkdir(
        parents=True,
        exist_ok=True
    )


    runner_path = Path(
        __file__
    ).resolve()


    runner_sha = sha256_file(
        runner_path
    )


    existing_hash_file = (
        OUT
        / "00_RUNNER_SHA256.txt"
    )


    if existing_hash_file.exists():

        existing_hash = (
            existing_hash_file
            .read_text(
                encoding="utf-8"
            )
            .strip()
        )


        if existing_hash != runner_sha:

            raise RuntimeError(
                "Partial V11 exists but runner hash differs. "
                "Refusing to mix runners."
            )


    if (
        OUT
        / "16_FINAL_SUMMARY.json"
    ).exists():

        raise RuntimeError(
            "Completed V11 already exists."
        )


    (
        OUT
        / "00_FROZEN_RUNNER.py"
    ).write_bytes(
        runner_path.read_bytes()
    )


    existing_hash_file.write_text(
        runner_sha
        + "\n",
        encoding="utf-8"
    )


    print(
        "Verifying V6 source hashes...",
        flush=True
    )


    source_count = verify_sources()


    frozen_spec = specification()


    (
        OUT
        / "03_FROZEN_SPECIFICATION.json"
    ).write_text(

        json.dumps(
            frozen_spec,
            indent=2
        ),

        encoding="utf-8"
    )


    spec_sha = sha256_text(
        canonical_json(
            frozen_spec
        )
    )


    (
        OUT
        / "04_FROZEN_SPECIFICATION_SHA256.txt"
    ).write_text(
        spec_sha
        + "\n",
        encoding="utf-8"
    )


    # ================================================================
    # FAST PREFLIGHT
    # ================================================================

    print(
        "Generating and freezing prompts...",
        flush=True
    )


    prompt_lookup = {}

    payload_lookup = {}

    metadata_lookup = {}

    prompt_rows = []


    preflight_started = time.time()


    for seed in range(
        N_SEEDS
    ):

        for condition in CONDITIONS:

            prompt, payload, metadata = (
                build_prompt(
                    seed,
                    condition
                )
            )


            key = (
                seed,
                condition
            )


            prompt_lookup[
                key
            ] = prompt


            payload_lookup[
                key
            ] = payload


            metadata_lookup[
                key
            ] = metadata


            stats = cheap_graph_stats(
                metadata[
                    "input_edges_role"
                ]
            )


            prompt_rows.append(
                {

                    "SEED":
                        seed,

                    "CONDITION":
                        condition,

                    "PROMPT_SHA256":
                        sha256_text(
                            prompt
                        ),

                    "PROMPT_CHARS":
                        len(
                            prompt
                        ),

                    "EDGE_COUNT":
                        stats[
                            "edge_count"
                        ],

                    "SCC_COUNT":
                        stats[
                            "scc_count"
                        ],

                    "LARGEST_SCC":
                        stats[
                            "largest_scc"
                        ],

                    "RECIPROCITY":
                        stats[
                            "reciprocity"
                        ],

                    "TRANSITIVITY":
                        stats[
                            "transitivity"
                        ],

                    "IN_DEGREE_SD":
                        stats[
                            "in_degree_sd"
                        ],

                    "OUT_DEGREE_SD":
                        stats[
                            "out_degree_sd"
                        ],

                    "PROMPT":
                        prompt,

                    "PAYLOAD_JSON":
                        canonical_json(
                            payload
                        )
                }
            )


        if (
            seed
            + 1
        ) % 10 == 0:

            print(
                "Preflight seeds:",
                seed + 1,
                "/",
                N_SEEDS,
                flush=True
            )


    print(
        "Prompt preflight completed in",
        round(
            time.time()
            - preflight_started,
            2
        ),
        "seconds.",
        flush=True
    )


    prompt_df = pd.DataFrame(
        prompt_rows
    )


    prompt_df.to_csv(
        OUT
        / "05_FROZEN_PROMPTS.csv",
        index=False
    )


    prompt_df[
        [
            "SEED",
            "CONDITION",
            "PROMPT_CHARS",
            "EDGE_COUNT",
            "SCC_COUNT",
            "LARGEST_SCC",
            "RECIPROCITY",
            "TRANSITIVITY",
            "IN_DEGREE_SD",
            "OUT_DEGREE_SD"
        ]
    ].to_csv(
        OUT
        / "14_CONTROL_MATCH_AUDIT.csv",
        index=False
    )


    # ================================================================
    # COLLECTION
    # ================================================================

    raw_path = (
        OUT
        / "01_RAW_RESPONSES.jsonl"
    )


    errors_path = (
        OUT
        / "02_ERRORS.jsonl"
    )


    api_path = (
        OUT
        / "17_RAW_API_OBJECTS.jsonl.gz"
    )


    completed = set()


    if raw_path.exists():

        with raw_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                if not line.strip():
                    continue


                record = json.loads(
                    line
                )


                if record.get(
                    "status"
                ) == "success":

                    completed.add(
                        record[
                            "cell_key"
                        ]
                    )


    print(
        "Existing valid cells:",
        len(
            completed
        ),
        "/",
        EXPECTED_CELLS,
        flush=True
    )


    print(
        "Starting API collection...",
        flush=True
    )


    with raw_path.open(
        "a",
        encoding="utf-8"
    ) as raw_file, errors_path.open(
        "a",
        encoding="utf-8"
    ) as error_file, gzip.open(
        api_path,
        "at",
        encoding="utf-8"
    ) as api_file:


        for seed in range(
            N_SEEDS
        ):

            for condition in CONDITIONS:


                key = (
                    seed,
                    condition
                )


                prompt = prompt_lookup[
                    key
                ]


                payload = payload_lookup[
                    key
                ]


                valid_nodes = set(
                    payload[
                        "V"
                    ]
                )


                input_edges = {

                    tuple(
                        edge
                    )

                    for edge
                    in payload[
                        "E"
                    ]
                }


                for rep in range(
                    N_REPS
                ):


                    cell_key = (
                        str(
                            seed
                        )
                        + "|"
                        + condition
                        + "|"
                        + str(
                            rep
                        )
                    )


                    if cell_key in completed:

                        continue


                    result = call_api(

                        prompt,

                        valid_nodes,

                        input_edges
                    )


                    if not result[
                        "ok"
                    ]:


                        error_file.write(

                            json.dumps(
                                {

                                    "cell_key":
                                        cell_key,

                                    "seed":
                                        seed,

                                    "condition":
                                        condition,

                                    "rep":
                                        rep,

                                    "failures":
                                        result[
                                            "failures"
                                        ]
                                },

                                ensure_ascii=False
                            )

                            + "\n"
                        )


                        error_file.flush()

                        continue


                    record = {

                        "cell_key":
                            cell_key,

                        "seed":
                            seed,

                        "condition":
                            condition,

                        "rep":
                            rep,

                        "status":
                            "success",

                        "attempt":
                            result[
                                "attempt"
                            ],

                        "utc":
                            utcnow(),

                        "prompt_sha256":
                            sha256_text(
                                prompt
                            ),

                        "text_sha256":
                            sha256_text(
                                result[
                                    "text"
                                ]
                            ),

                        "text":
                            result[
                                "text"
                            ],

                        "parsed":
                            result[
                                "parsed"
                            ],

                        "response_id":
                            result[
                                "response_id"
                            ],

                        "response_status":
                            result[
                                "response_status"
                            ],

                        "latency_seconds":
                            result[
                                "latency"
                            ],

                        "input_tokens":
                            result[
                                "input_tokens"
                            ],

                        "output_tokens":
                            result[
                                "output_tokens"
                            ]
                    }


                    raw_file.write(

                        json.dumps(
                            record,
                            ensure_ascii=False
                        )

                        + "\n"
                    )


                    raw_file.flush()


                    api_file.write(

                        json.dumps(
                            {

                                "cell_key":
                                    cell_key,

                                "raw_response":
                                    result[
                                        "raw"
                                    ]
                            },

                            ensure_ascii=False
                        )

                        + "\n"
                    )


                    api_file.flush()


                    completed.add(
                        cell_key
                    )


                    if len(
                        completed
                    ) % 25 == 0:

                        print(
                            "API progress:",
                            len(
                                completed
                            ),
                            "/",
                            EXPECTED_CELLS,
                            flush=True
                        )


    # ================================================================
    # HARD COMPLETENESS GATE
    # ================================================================

    successful = {}


    with raw_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            if not line.strip():
                continue


            record = json.loads(
                line
            )


            if record.get(
                "status"
            ) != "success":

                continue


            key = record[
                "cell_key"
            ]


            if key not in successful:

                successful[
                    key
                ] = record


    expected = {

        str(
            seed
        )
        + "|"
        + condition
        + "|"
        + str(
            rep
        )

        for seed
        in range(
            N_SEEDS
        )

        for condition
        in CONDITIONS

        for rep
        in range(
            N_REPS
        )
    }


    missing = sorted(

        expected

        - set(
            successful.keys()
        )
    )


    pd.DataFrame(
        {
            "CELL_KEY":
                missing
        }
    ).to_csv(
        OUT
        / "15_MISSING_CELLS.csv",
        index=False
    )


    completeness = {

        "expected_cells":
            EXPECTED_CELLS,

        "unique_success_cells":
            len(
                successful
            ),

        "missing_cells":
            len(
                missing
            ),

        "scientific_completeness":
            len(
                missing
            ) == 0
    }


    (
        OUT
        / "00A_VALID_TEXT_COMPLETENESS.json"
    ).write_text(

        json.dumps(
            completeness,
            indent=2
        ),

        encoding="utf-8"
    )


    if missing:

        raise RuntimeError(

            str(
                len(
                    missing
                )
            )

            + " cells remain missing/invalid. "
            + "NO INFERENTIAL ANALYSIS PRODUCED. "
            + "Rerun this identical frozen runner."
        )


    print(
        "Completeness gate passed: 2880/2880.",
        flush=True
    )


    # ================================================================
    # FEATURE EXTRACTION
    # ================================================================

    feature_rows = []


    for record in successful.values():


        seed = int(
            record[
                "seed"
            ]
        )


        condition = record[
            "condition"
        ]


        key = (
            seed,
            condition
        )


        row = extract_features(

            seed,

            condition,

            payload_lookup[
                key
            ],

            metadata_lookup[
                key
            ],

            record[
                "parsed"
            ]
        )


        row[
            "REP"
        ] = int(
            record[
                "rep"
            ]
        )


        row[
            "CELL_KEY"
        ] = record[
            "cell_key"
        ]


        # DIAGNOSTIC ONLY
        row[
            "OUTPUT_TOKENS_DIAGNOSTIC"
        ] = int(
            record.get(
                "output_tokens",
                0
            )
            or 0
        )


        feature_rows.append(
            row
        )


    response_df = pd.DataFrame(
        feature_rows
    )


    response_df.to_csv(
        OUT
        / "06_RESPONSE_GRAPH_FEATURES.csv",
        index=False
    )


    seed_df = (

        response_df.groupby(
            [
                "SEED",
                "CONDITION"
            ],
            as_index=False
        )[
            FEATURES
            + [
                "OUTPUT_TOKENS_DIAGNOSTIC"
            ]
        ]

        .mean()
    )


    seed_df.to_csv(
        OUT
        / "07_SEED_LEVEL_GRAPH_VECTOR.csv",
        index=False
    )


    # ================================================================
    # CONDITION SUMMARY
    # ================================================================

    condition_rows = []


    for condition in CONDITIONS:


        subset = seed_df[
            seed_df[
                "CONDITION"
            ]
            == condition
        ]


        row = {

            "CONDITION":
                condition,

            "N":
                len(
                    subset
                )
        }


        for feature in FEATURES:


            row[
                feature
                + "_MEAN"
            ] = float(
                subset[
                    feature
                ].mean()
            )


            row[
                feature
                + "_SD"
            ] = float(
                subset[
                    feature
                ].std()
            )


        condition_rows.append(
            row
        )


    pd.DataFrame(
        condition_rows
    ).to_csv(
        OUT
        / "08_CONDITION_RESULTS.csv",
        index=False
    )


    # ================================================================
    # DEVELOPMENT-ONLY DISTANCE SCALING
    # ================================================================

    dev_rows = seed_df[
        seed_df[
            "SEED"
        ].isin(
            DEV
        )
    ]


    scales = dev_rows[
        FEATURES
    ].std(
        axis=0,
        ddof=1
    ).to_numpy(
        dtype=float
    )


    scales[
        ~np.isfinite(
            scales
        )
    ] = 1.0


    scales[
        scales < 1e-8
    ] = 1.0


    scale_table = pd.DataFrame(
        {

            "FEATURE":
                FEATURES,

            "DEV_SD":
                scales
        }
    )


    scale_table.to_csv(
        OUT
        / "21_DEVELOPMENT_DISTANCE_SCALES.csv",
        index=False
    )


    # ================================================================
    # PER-SEED DISTANCES FROM AUTHENTIC
    # ================================================================

    distance_rows = []


    for seed in range(
        N_SEEDS
    ):


        authentic = seed_df[
            (
                seed_df[
                    "SEED"
                ]
                == seed
            )
            &
            (
                seed_df[
                    "CONDITION"
                ]
                == "A_AUTHENTIC"
            )
        ].iloc[
            0
        ]


        for condition in CONDITIONS:


            comparison = seed_df[
                (
                    seed_df[
                        "SEED"
                    ]
                    == seed
                )
                &
                (
                    seed_df[
                        "CONDITION"
                    ]
                    == condition
                )
            ].iloc[
                0
            ]


            distance = (
                standardized_distance(

                    authentic,

                    comparison,

                    scales
                )
            )


            distance_rows.append(
                {

                    "SEED":
                        seed,

                    "CONDITION":
                        condition,

                    "DISTANCE_FROM_AUTHENTIC":
                        distance
                }
            )


    distance_df = pd.DataFrame(
        distance_rows
    )


    distance_df.to_csv(
        OUT
        / "22_DISTANCE_FROM_AUTHENTIC.csv",
        index=False
    )


    # ================================================================
    # TARGETED TESTS
    #
    # Intervention shift must exceed label-only isomorphic baseline.
    # ================================================================

    targets = [

        (
            "DIRECTION",
            "D_DIRECTION_REVERSED"
        ),

        (
            "ORDER",
            "O_ORDER_SHUFFLED"
        ),

        (
            "META_CONTROL",
            "M_SWITCH_NEUTRALIZED"
        ),

        (
            "PRESERVATION",
            "P_ANCHOR_REASSIGNED"
        ),

        (
            "TERMINAL",
            "T_TERMINAL_CHANGED"
        )
    ]


    primary_rows = []


    for index, (
        name,
        condition
    ) in enumerate(
        targets
    ):


        intervention = (

            distance_df[
                distance_df[
                    "CONDITION"
                ]
                == condition
            ][
                [
                    "SEED",
                    "DISTANCE_FROM_AUTHENTIC"
                ]
            ]

            .rename(
                columns={
                    "DISTANCE_FROM_AUTHENTIC":
                        "INTERVENTION"
                }
            )
        )


        baseline = (

            distance_df[
                distance_df[
                    "CONDITION"
                ]
                == "I_ISOMORPHIC_RELABEL"
            ][
                [
                    "SEED",
                    "DISTANCE_FROM_AUTHENTIC"
                ]
            ]

            .rename(
                columns={
                    "DISTANCE_FROM_AUTHENTIC":
                        "ISOMORPHIC"
                }
            )
        )


        merged = intervention.merge(
            baseline,
            on="SEED"
        )


        differences = (

            merged[
                "INTERVENTION"
            ]

            - merged[
                "ISOMORPHIC"
            ]
        ).to_numpy()


        p_value = signflip_test(

            differences,

            MASTER_SEED
            + index
        )


        primary_rows.append(
            {

                "CONTRAST":
                    name,

                "CONDITION":
                    condition,

                "INTERVENTION_DISTANCE":
                    float(
                        merged[
                            "INTERVENTION"
                        ].mean()
                    ),

                "ISOMORPHIC_DISTANCE":
                    float(
                        merged[
                            "ISOMORPHIC"
                        ].mean()
                    ),

                "EXCESS_DISTANCE":
                    float(
                        differences.mean()
                    ),

                "PERMUTATION_P":
                    p_value
            }
        )


    adjusted = holm_adjust(
        [
            row[
                "PERMUTATION_P"
            ]

            for row
            in primary_rows
        ]
    )


    for row, adjusted_p in zip(
        primary_rows,
        adjusted
    ):


        row[
            "HOLM_P"
        ] = float(
            adjusted_p
        )


        row[
            "PASS"
        ] = bool(

            row[
                "EXCESS_DISTANCE"
            ]
            > 0

            and adjusted_p
            < 0.05
        )


    pd.DataFrame(
        primary_rows
    ).to_csv(
        OUT
        / "09_PRIMARY_CONTRASTS.csv",
        index=False
    )


    # ================================================================
    # SPLIT REPLICATION
    # ================================================================

    partitions = [

        (
            "DEVELOPMENT",
            DEV
        ),

        (
            "CONFIRMATION",
            CONF
        ),

        (
            "FINAL_HOLDOUT",
            HOLD
        )
    ]


    split_rows = []


    for partition_index, (
        partition_name,
        seeds
    ) in enumerate(
        partitions
    ):


        temporary = []


        partition_distances = (
            distance_df[
                distance_df[
                    "SEED"
                ].isin(
                    seeds
                )
            ]
        )


        for target_index, (
            name,
            condition
        ) in enumerate(
            targets
        ):


            intervention = (

                partition_distances[
                    partition_distances[
                        "CONDITION"
                    ]
                    == condition
                ][
                    [
                        "SEED",
                        "DISTANCE_FROM_AUTHENTIC"
                    ]
                ]

                .rename(
                    columns={
                        "DISTANCE_FROM_AUTHENTIC":
                            "INTERVENTION"
                    }
                )
            )


            baseline = (

                partition_distances[
                    partition_distances[
                        "CONDITION"
                    ]
                    == "I_ISOMORPHIC_RELABEL"
                ][
                    [
                        "SEED",
                        "DISTANCE_FROM_AUTHENTIC"
                    ]
                ]

                .rename(
                    columns={
                        "DISTANCE_FROM_AUTHENTIC":
                            "ISOMORPHIC"
                    }
                )
            )


            merged = intervention.merge(
                baseline,
                on="SEED"
            )


            differences = (

                merged[
                    "INTERVENTION"
                ]

                - merged[
                    "ISOMORPHIC"
                ]
            ).to_numpy()


            p_value = signflip_test(

                differences,

                MASTER_SEED
                + 100
                + partition_index
                * 10
                + target_index
            )


            temporary.append(
                {

                    "PARTITION":
                        partition_name,

                    "CONTRAST":
                        name,

                    "N":
                        len(
                            differences
                        ),

                    "EXCESS_DISTANCE":
                        float(
                            differences.mean()
                        ),

                    "P":
                        p_value
                }
            )


        adjusted_local = (
            holm_adjust(
                [
                    row[
                        "P"
                    ]

                    for row
                    in temporary
                ]
            )
        )


        for row, adjusted_p in zip(
            temporary,
            adjusted_local
        ):


            row[
                "HOLM_P"
            ] = float(
                adjusted_p
            )


            row[
                "PASS"
            ] = bool(

                row[
                    "EXCESS_DISTANCE"
                ]
                > 0

                and adjusted_p
                < 0.05
            )


            split_rows.append(
                row
            )


    pd.DataFrame(
        split_rows
    ).to_csv(
        OUT
        / "10_SPLIT_REPLICATION.csv",
        index=False
    )


    # ================================================================
    # GRAPH-ONLY CLASSIFIER
    # ================================================================

    classifier_conditions = [

        "A_AUTHENTIC",

        "I_ISOMORPHIC_RELABEL",

        "D_DIRECTION_REVERSED",

        "O_ORDER_SHUFFLED",

        "M_SWITCH_NEUTRALIZED",

        "P_ANCHOR_REASSIGNED",

        "T_TERMINAL_CHANGED",

        "R_DEGREE_MATCHED_REWIRE",

        "X_COMPLEXITY_MATCHED",

        "L_FORWARD_CONTROL",

        "G_RANDOM_CONTROL"
    ]


    classifier_df = seed_df[
        seed_df[
            "CONDITION"
        ].isin(
            classifier_conditions
        )
    ].copy()


    classifier_df[
        "TARGET"
    ] = (

        classifier_df[
            "CONDITION"
        ].isin(
            [
                "A_AUTHENTIC",
                "I_ISOMORPHIC_RELABEL"
            ]
        )

        .astype(
            int
        )
    )


    classifier = Pipeline(
        [

            (
                "scale",
                StandardScaler()
            ),

            (
                "model",
                LogisticRegression(

                    max_iter=5000,

                    class_weight="balanced",

                    random_state=
                        MASTER_SEED
                )
            )
        ]
    )


    training = classifier_df[
        classifier_df[
            "SEED"
        ].isin(
            DEV
        )
    ]


    classifier.fit(

        training[
            FEATURES
        ],

        training[
            "TARGET"
        ]
    )


    classifier_rows = []


    for partition_name, seeds in partitions:


        subset = classifier_df[
            classifier_df[
                "SEED"
            ].isin(
                seeds
            )
        ]


        probabilities = (

            classifier.predict_proba(
                subset[
                    FEATURES
                ]
            )[
                :,
                1
            ]
        )


        predictions = (
            probabilities
            >= 0.5
        ).astype(
            int
        )


        classifier_rows.append(
            {

                "PARTITION":
                    partition_name,

                "N":
                    len(
                        subset
                    ),

                "ROC_AUC":
                    float(
                        roc_auc_score(
                            subset[
                                "TARGET"
                            ],
                            probabilities
                        )
                    ),

                "ACCURACY":
                    float(
                        accuracy_score(
                            subset[
                                "TARGET"
                            ],
                            predictions
                        )
                    )
            }
        )


    pd.DataFrame(
        classifier_rows
    ).to_csv(
        OUT
        / "11_GRAPH_CLASSIFIER.csv",
        index=False
    )


    # ================================================================
    # SURFACE / ISOMORPHIC INVARIANCE
    # ================================================================

    invariance_rows = []


    for condition in [

        "I_ISOMORPHIC_RELABEL",

        "S_SURFACE_FORMAT_CONTROL"
    ]:


        values = distance_df[
            distance_df[
                "CONDITION"
            ]
            == condition
        ][
            "DISTANCE_FROM_AUTHENTIC"
        ]


        invariance_rows.append(
            {

                "CONDITION":
                    condition,

                "MEAN_DISTANCE":
                    float(
                        values.mean()
                    ),

                "SD_DISTANCE":
                    float(
                        values.std()
                    )
            }
        )


    pd.DataFrame(
        invariance_rows
    ).to_csv(
        OUT
        / "12_ISOMORPHIC_INVARIANCE.csv",
        index=False
    )


    # ================================================================
    # DIRECTION / ORDER DETAIL
    # ================================================================

    seed_df[
        [
            "SEED",
            "CONDITION",
            "SEQ_INPUT_EDGE_COHERENCE",
            "SEQ_INPUT_ORDER_COHERENCE",
            "PLUS_SEQ_COHERENCE"
        ]
    ].to_csv(
        OUT
        / "13_DIRECTION_ORDER_METRICS.csv",
        index=False
    )


    # ================================================================
    # FINAL HOLDOUT SPECIFICITY
    #
    # matched rewire / complexity distance must exceed isomorphic
    # baseline.
    # ================================================================

    holdout_distances = distance_df[
        distance_df[
            "SEED"
        ].isin(
            HOLD
        )
    ]


    specificity_rows = []


    for index, condition in enumerate(
        [

            "R_DEGREE_MATCHED_REWIRE",

            "X_COMPLEXITY_MATCHED"
        ]
    ):


        control = (

            holdout_distances[
                holdout_distances[
                    "CONDITION"
                ]
                == condition
            ][
                [
                    "SEED",
                    "DISTANCE_FROM_AUTHENTIC"
                ]
            ]

            .rename(
                columns={
                    "DISTANCE_FROM_AUTHENTIC":
                        "CONTROL"
                }
            )
        )


        baseline = (

            holdout_distances[
                holdout_distances[
                    "CONDITION"
                ]
                == "I_ISOMORPHIC_RELABEL"
            ][
                [
                    "SEED",
                    "DISTANCE_FROM_AUTHENTIC"
                ]
            ]

            .rename(
                columns={
                    "DISTANCE_FROM_AUTHENTIC":
                        "ISOMORPHIC"
                }
            )
        )


        merged = control.merge(
            baseline,
            on="SEED"
        )


        differences = (

            merged[
                "CONTROL"
            ]

            - merged[
                "ISOMORPHIC"
            ]
        ).to_numpy()


        p_value = signflip_test(

            differences,

            MASTER_SEED
            + 500
            + index
        )


        specificity_rows.append(
            {

                "CONTROL":
                    condition,

                "CONTROL_DISTANCE":
                    float(
                        merged[
                            "CONTROL"
                        ].mean()
                    ),

                "ISOMORPHIC_DISTANCE":
                    float(
                        merged[
                            "ISOMORPHIC"
                        ].mean()
                    ),

                "EXCESS_DISTANCE":
                    float(
                        differences.mean()
                    ),

                "PERMUTATION_P":
                    p_value,

                "PASS":
                    bool(

                        differences.mean()
                        > 0

                        and p_value
                        < 0.01
                    )
            }
        )


    pd.DataFrame(
        specificity_rows
    ).to_csv(
        OUT
        / "18_SPECIFICITY_HOLDOUT.csv",
        index=False
    )


    # ================================================================
    # ENVIRONMENT
    # ================================================================

    environment = {

        "python":
            sys.version,

        "platform":
            platform.platform(),

        "openai":
            getattr(
                openai,
                "__version__",
                None
            ),

        "numpy":
            np.__version__,

        "pandas":
            pd.__version__,

        "networkx":
            nx.__version__
    }


    (
        OUT
        / "20_RUN_ENVIRONMENT.json"
    ).write_text(

        json.dumps(
            environment,
            indent=2
        ),

        encoding="utf-8"
    )


    # ================================================================
    # FINAL CLASSIFICATION
    # ================================================================

    final_auc = next(

        row[
            "ROC_AUC"
        ]

        for row
        in classifier_rows

        if row[
            "PARTITION"
        ]
        == "FINAL_HOLDOUT"
    )


    confirmation_passes = sum(

        1

        for row
        in split_rows

        if (
            row[
                "PARTITION"
            ]
            == "CONFIRMATION"

            and row[
                "PASS"
            ]
        )
    )


    holdout_passes = sum(

        1

        for row
        in split_rows

        if (
            row[
                "PARTITION"
            ]
            == "FINAL_HOLDOUT"

            and row[
                "PASS"
            ]
        )
    )


    specificity_pass = all(

        row[
            "PASS"
        ]

        for row
        in specificity_rows
    )


    isomorphic_mean = float(

        distance_df[
            distance_df[
                "CONDITION"
            ]
            == "I_ISOMORPHIC_RELABEL"
        ][
            "DISTANCE_FROM_AUTHENTIC"
        ].mean()
    )


    structural_control_mean = float(

        distance_df[
            distance_df[
                "CONDITION"
            ].isin(
                [

                    "D_DIRECTION_REVERSED",

                    "O_ORDER_SHUFFLED",

                    "M_SWITCH_NEUTRALIZED",

                    "P_ANCHOR_REASSIGNED",

                    "T_TERMINAL_CHANGED",

                    "R_DEGREE_MATCHED_REWIRE",

                    "X_COMPLEXITY_MATCHED"
                ]
            )
        ][
            "DISTANCE_FROM_AUTHENTIC"
        ].mean()
    )


    invariance_pass = (

        isomorphic_mean

        < structural_control_mean
    )


    strong = (

        final_auc
        >= 0.85

        and confirmation_passes
        >= 4

        and holdout_passes
        >= 4

        and specificity_pass

        and invariance_pass
    )


    partial = (

        final_auc
        >= 0.75

        and (
            confirmation_passes
            >= 2

            or holdout_passes
            >= 2
        )
    )


    if strong:

        classification = (
            "STRONG_RELATIONAL_STRUCTURAL_FINGERPRINT"
        )


    elif partial:

        classification = (
            "PARTIAL_RELATIONAL_STRUCTURAL_FINGERPRINT"
        )


    else:

        classification = (
            "RELATIONAL_STRUCTURAL_FINGERPRINT_NOT_SUPPORTED"
        )


    total_input_tokens = sum(

        int(
            row.get(
                "input_tokens",
                0
            )
            or 0
        )

        for row
        in successful.values()
    )


    total_output_tokens = sum(

        int(
            row.get(
                "output_tokens",
                0
            )
            or 0
        )

        for row
        in successful.values()
    )


    summary = {

        "experiment":
            EXPERIMENT,

        "completed_utc":
            utcnow(),

        "model":
            MODEL,

        "runner_sha256":
            runner_sha,

        "specification_sha256":
            spec_sha,

        "source_manifest_entries_verified":
            source_count,

        "expected_cells":
            EXPECTED_CELLS,

        "successful_unique_cells":
            len(
                successful
            ),

        "missing_cells":
            0,

        "primary_contrasts":
            primary_rows,

        "split_replication":
            split_rows,

        "classifier":
            classifier_rows,

        "specificity_holdout":
            specificity_rows,

        "isomorphic_mean_distance":
            isomorphic_mean,

        "structural_control_mean_distance":
            structural_control_mean,

        "invariance_pass":
            invariance_pass,

        "classification":
            classification,

        "token_usage": {

            "input_tokens":
                total_input_tokens,

            "output_tokens":
                total_output_tokens
        },

        "interpretation_boundary":
            frozen_spec[
                "interpretation_boundary"
            ]
    }


    (
        OUT
        / "16_FINAL_SUMMARY.json"
    ).write_text(

        json.dumps(
            summary,
            indent=2
        ),

        encoding="utf-8"
    )


    # ================================================================
    # INTERNAL ARTIFACT VERIFICATION
    # ================================================================

    required_files = [

        "00_FROZEN_RUNNER.py",

        "00_RUNNER_SHA256.txt",

        "00A_VALID_TEXT_COMPLETENESS.json",

        "01_RAW_RESPONSES.jsonl",

        "02_ERRORS.jsonl",

        "03_FROZEN_SPECIFICATION.json",

        "04_FROZEN_SPECIFICATION_SHA256.txt",

        "05_FROZEN_PROMPTS.csv",

        "06_RESPONSE_GRAPH_FEATURES.csv",

        "07_SEED_LEVEL_GRAPH_VECTOR.csv",

        "08_CONDITION_RESULTS.csv",

        "09_PRIMARY_CONTRASTS.csv",

        "10_SPLIT_REPLICATION.csv",

        "11_GRAPH_CLASSIFIER.csv",

        "12_ISOMORPHIC_INVARIANCE.csv",

        "13_DIRECTION_ORDER_METRICS.csv",

        "14_CONTROL_MATCH_AUDIT.csv",

        "15_MISSING_CELLS.csv",

        "16_FINAL_SUMMARY.json",

        "17_RAW_API_OBJECTS.jsonl.gz",

        "18_SPECIFICITY_HOLDOUT.csv",

        "19_SOURCE_PROVENANCE.csv",

        "20_RUN_ENVIRONMENT.json",

        "21_DEVELOPMENT_DISTANCE_SCALES.csv",

        "22_DISTANCE_FROM_AUTHENTIC.csv"
    ]


    missing_artifacts = [

        name

        for name
        in required_files

        if not (
            OUT
            / name
        ).exists()
    ]


    if missing_artifacts:

        raise RuntimeError(

            "Evidence package incomplete: "

            + ", ".join(
                missing_artifacts
            )
        )


    artifact_rows = []


    for name in required_files:

        path = (
            OUT
            / name
        )


        artifact_rows.append(
            {

                "FILE":
                    name,

                "SHA256":
                    sha256_file(
                        path
                    ),

                "BYTES":
                    path.stat().st_size
            }
        )


    pd.DataFrame(
        artifact_rows
    ).to_csv(
        OUT
        / "23_FINAL_ARTIFACT_SHA256.csv",
        index=False
    )


    print(
        "",
        flush=True
    )


    print(
        "============================================================",
        flush=True
    )


    print(
        "V11 COMPLETE",
        flush=True
    )


    print(
        "============================================================",
        flush=True
    )


    print(
        json.dumps(
            {

                "classification":
                    classification,

                "final_holdout_auc":
                    final_auc,

                "confirmation_passes":
                    confirmation_passes,

                "final_holdout_passes":
                    holdout_passes,

                "specificity_pass":
                    specificity_pass,

                "invariance_pass":
                    invariance_pass
            },

            indent=2
        ),

        flush=True
    )


    print(
        "",
        flush=True
    )


    print(
        "RESULT DIRECTORY:",
        flush=True
    )


    print(
        str(
            OUT
        ),
        flush=True
    )


if __name__ == "__main__":

    main()
