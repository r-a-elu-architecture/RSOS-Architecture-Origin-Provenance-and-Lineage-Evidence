from pathlib import Path
import os, sys, json, time, random, hashlib, gzip, platform, argparse, traceback, shutil
from functools import lru_cache
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.covariance import LedoitWolf
import openai
from openai import OpenAI

EXPERIMENT = "WF1_A_V12_R_CORRECTIVE_CAUSAL_TOPOLOGY_R1"
MODEL = "gpt-5.6-luna"
MASTER_SEED = 120919
SEED_IDS = list(range(1080, 1160))
N_REPS = 3
DEV = set(range(1080, 1120))
CONF = set(range(1120, 1140))
HOLD = set(range(1140, 1160))
MAX_OUTPUT_TOKENS = 500
MAX_RETRIES = 8
N_PERMUTATIONS = 20000
RUNNER_DIR = Path(__file__).resolve().parent
EXP_ROOT = RUNNER_DIR.parent
ROOT = EXP_ROOT
OUT = EXP_ROOT / "OUTPUT"
INPUTS = EXP_ROOT / "INPUTS"
ORIG_FREEZE_DIR = INPUTS / "V12_ORIGINAL_FREEZE"
EXPECTED_ORIGINAL_FREEZE_SHA256 = "352B5055AA16CA746568A36678E9FBF535D522F58D3EB6D15CB9C450F3B3ED71"
EXPECTED_ORIGINAL_MISSING_SHA256 = "D1B7A3A4D188135D111E0945087365599C874E7D48181098432D24AE5DC4D2B9"
EXPECTED_ORIGINAL_RUNNER_SHA256 = "B9A293AF98EDBF774C6912691E2A00CD4042DF93BBB50DB31BC8540E119337E7"

AUTH_EDGES = [
    (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(3,1),(5,2),(6,3)
]
AUTH_ORDER = list(range(8))
AUTH_KEEP = [2,5]
AUTH_SWITCH = (3,1,3,6)
AUTH_SWITCH_EDGE_INDEX = 7
AUTH_END = 7
FORWARD_CONTROL = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(0,2),(2,5),(4,7)]

BASE_CONDITIONS = [
    "A_AUTHENTIC","I_ISOMORPHIC_RELABEL","F_SURFACE_FORMAT_CONTROL",
    "D_ALL_DIRECTION_REVERSED","O_ORDER_SHUFFLED",
    "M_SWITCH_REASSIGNED","M0_SWITCH_NEUTRALIZED",
    "P_ANCHOR_REASSIGNED","T_TERMINAL_CHANGED",
    "R_DEGREE_MATCHED_REWIRE","X_COMPLEXITY_MATCHED","Y_TOPOLOGY_MATCHED",
    "L_FORWARD_CONTROL","G_RANDOM_CONTROL"
]

EDGE_DELETE_CONDITIONS = [f"E{i}_DELETE" for i in range(10)]
EDGE_REVERSE_CONDITIONS = [f"E{i}_REVERSE" for i in range(10)]
CONDITIONS = BASE_CONDITIONS + EDGE_DELETE_CONDITIONS + EDGE_REVERSE_CONDITIONS
PURE_DELETE_INDEXES = [i for i in range(10) if i != AUTH_SWITCH_EDGE_INDEX]
PURE_REVERSE_INDEXES = [i for i in range(10) if i not in {AUTH_SWITCH_EDGE_INDEX,9}]
EXPECTED_MAIN_CELLS = len(CONDITIONS) * len(SEED_IDS) * N_REPS

# V12-R corrective semantics. These conditions were switch-linked/dependent in
# original V12 and are NOT promoted into the original pure-edge primary pools.
# Their switch is deterministically reassigned to a valid relation so that the
# model is not simultaneously asked to use a disabled/now-present switch edge.
CORRECTIVE_SWITCH_CONDITIONS = {
    "E7_DELETE",
    "E7_REVERSE",
    "E9_REVERSE",
}

NATIVE_FRAMES = {
    "N1_COHERENT": "Choose a coherent nontrivial directed dependency structure among the supplied opaque nodes.",
    "N2_STABLE": "Choose a stable internally consistent directed relational structure among the supplied opaque nodes.",
    "N3_COMPACT": "Choose a compact nontrivial directed interaction structure among the supplied opaque nodes."
}
EXPECTED_NATIVE_CELLS = len(SEED_IDS) * N_REPS * len(NATIVE_FRAMES)

BEHAVIOR_FEATURES = [
    "SEQ_ACTIVE_EDGE_COHERENCE","PLUS_SEQ_COHERENCE","PLUS_REACHABILITY",
    "MINUS_SEQ_CONFLICT","CODE_SEQ_COHERENCE","CODE_CONNECTEDNESS",
    "FINAL_RECIPROCITY","FINAL_SCC_COUNT","FINAL_LARGEST_SCC_FRAC",
    "FINAL_TRANSITIVITY","FINAL_IN_DEGREE_SD","FINAL_OUT_DEGREE_SD",
    "FINAL_SPECTRAL_RADIUS","FINAL_TWO_CYCLES","FINAL_THREE_CYCLES",
    "FINAL_REACHABILITY_FRAC","FINAL_MEAN_REACHABLE_PATH"
]

DIRECT_OPERATOR_FEATURES = [
    "ORDER_FIDELITY","K_EXACT","K_JACCARD","S_EXACT","S_APPLIED_SCORE",
    "K_INCIDENT_RETENTION","Z_EXACT","Z_FINAL_SINKNESS","Z_SEQUENCE_POSITION"
]

PRIMARY_TESTS = [
    "EDGE_DELETE_PURE9","EDGE_REVERSE_PURE8","DIRECTION_ALL_COVARIANT_SWITCH","ORDER",
    "META_REASSIGNED","PRESERVATION","TERMINAL",
    "DEGREE_MATCHED","COMPLEXITY_MATCHED","TOPOLOGY_MATCHED"
]


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def canonical_json(o):
    return json.dumps(o, ensure_ascii=False, sort_keys=True, separators=(",",":"))


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def holm_adjust(p):
    p = np.asarray(p,float)
    m = len(p)
    order = np.argsort(p)
    out = np.zeros(m)
    running = 0.0

    for rank,idx in enumerate(order):
        running = max(running,(m-rank)*p[idx])
        out[idx] = min(1.0,running)

    return out


def signflip(values, seed):
    v = np.asarray(values,float)
    v = v[np.isfinite(v)]

    if len(v)==0 or np.allclose(v,0):
        return 1.0

    obs = abs(float(v.mean()))
    rng = np.random.default_rng(seed)
    extreme = 1

    for _ in range(N_PERMUTATIONS):
        s = rng.choice([-1.0,1.0],size=len(v))
        if abs(float(np.mean(v*s))) >= obs:
            extreme += 1

    return extreme/(N_PERMUTATIONS+1)


def jaccard(a,b):
    a = set(a)
    b = set(b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a&b)/len(a|b)


def graph_object(edges):
    g = nx.DiGraph()
    g.add_nodes_from(range(8))
    g.add_edges_from(edges)
    return g


def cycle_counts(g):
    A = nx.to_numpy_array(g,nodelist=range(8),dtype=float)

    two = int(
        round(
            np.trace(
                np.linalg.matrix_power(A,2)
            ) / 2.0
        )
    )

    three = int(
        round(
            np.trace(
                np.linalg.matrix_power(A,3)
            ) / 3.0
        )
    )

    return two,three


def reachability_stats(g):
    reachable = 0
    dists = []

    for a in range(8):
        lengths = nx.single_source_shortest_path_length(g,a)

        for b,d in lengths.items():
            if a != b:
                reachable += 1
                dists.append(d)

    return reachable/56.0, (float(np.mean(dists)) if dists else 0.0)


def complexity_vector(edges):
    g = graph_object(edges)
    ug = g.to_undirected()
    scc = list(nx.strongly_connected_components(g))
    weak = list(nx.weakly_connected_components(g))

    rec = nx.reciprocity(g)
    rec = 0.0 if rec is None else float(rec)

    indeg = np.array([g.in_degree(i) for i in range(8)],float)
    outdeg = np.array([g.out_degree(i) for i in range(8)],float)

    A = nx.to_numpy_array(g,nodelist=range(8),dtype=float)
    eig = np.linalg.eigvals(A)
    spectral = float(np.max(np.abs(eig))) if eig.size else 0.0

    two,three = cycle_counts(g)
    reach,meanpath = reachability_stats(g)

    return np.array([
        len(scc)/8.0,
        max((len(x) for x in scc),default=0)/8.0,
        len(weak)/8.0,
        rec,
        float(nx.transitivity(ug)) if ug.number_of_edges() else 0.0,
        float(indeg.std())/3.0,
        float(outdeg.std())/3.0,
        spectral/4.0,
        min(two,5)/5.0,
        min(three,10)/10.0,
        reach,
        meanpath/7.0
    ],float)


TRIAD_NAMES = [
    "003","012","102","021D","021U","021C","111D","111U",
    "030T","030C","201","120D","120U","120C","210","300"
]


def topology_vector(edges):
    g = graph_object(edges)
    base = complexity_vector(edges)

    indeg = np.sort(
        np.array(
            [g.in_degree(i) for i in range(8)],
            float
        )
    ) / 7.0

    outdeg = np.sort(
        np.array(
            [g.out_degree(i) for i in range(8)],
            float
        )
    ) / 7.0

    census = nx.triadic_census(g)

    tri = np.array(
        [census[k]/56.0 for k in TRIAD_NAMES],
        float
    )

    sccsizes = sorted(
        [len(x) for x in nx.strongly_connected_components(g)],
        reverse=True
    ) + [0]*8

    sccv = np.array(sccsizes[:8],float)/8.0

    return np.concatenate([
        base,
        indeg,
        outdeg,
        tri,
        sccv
    ])


AUTH_COMPLEXITY = complexity_vector(AUTH_EDGES)
AUTH_TOPOLOGY = topology_vector(AUTH_EDGES)


def edge_overlap(edges):
    return len(set(edges)&set(AUTH_EDGES))


def fast_match_vector(edges):
    A = np.zeros((8,8),dtype=float)

    for a,b in edges:
        A[a,b] = 1.0

    indeg = A.sum(axis=0)
    outdeg = A.sum(axis=1)

    mutual = float(np.sum(A*A.T)/2.0)/5.0
    three = float(np.trace(A@A@A)/3.0)/10.0

    eig = np.linalg.eigvals(A)
    spectral = float(np.max(np.abs(eig)))/4.0 if eig.size else 0.0

    U = ((A+A.T)>0).astype(float)
    np.fill_diagonal(U,0)

    tri = float(np.trace(U@U@U)/6.0)
    deg = U.sum(axis=1)
    triples = float(np.sum(deg*(deg-1)/2.0))
    trans = (3.0*tri/triples) if triples>0 else 0.0

    reach = A.astype(bool).copy()
    dist = np.where(reach,1.0,np.inf)
    np.fill_diagonal(dist,0.0)

    for k in range(8):
        reach = reach | (reach[:,k,None] & reach[None,k,:])
        dist = np.minimum(
            dist,
            dist[:,k,None] + dist[None,k,:]
        )

    mask = np.isfinite(dist) & (~np.eye(8,dtype=bool))
    reachfrac = float(mask.sum())/56.0
    meanpath = float(dist[mask].mean()/7.0) if mask.any() else 0.0

    return np.concatenate([
        np.sort(indeg)/7.0,
        np.sort(outdeg)/7.0,
        np.array([
            float(indeg.std())/3.0,
            float(outdeg.std())/3.0,
            mutual,
            three,
            spectral,
            trans,
            reachfrac,
            meanpath
        ])
    ])


AUTH_FAST = fast_match_vector(AUTH_EDGES)


@lru_cache(maxsize=None)
def degree_preserving_rewire(seed):
    og = graph_object(AUTH_EDGES)

    for attempt in range(1,181):
        rng = random.Random(100000+seed*1000+attempt)
        cand = list(AUTH_EDGES)

        successful_swaps = 0

        for _ in range(40):
            i,j = rng.sample(
                [
                    x
                    for x in range(len(cand))
                    if x != AUTH_SWITCH_EDGE_INDEX
                ],
                2
            )

            a,b = cand[i]
            c,d = cand[j]

            if len({a,b,c,d}) < 4:
                continue

            e1 = (a,d)
            e2 = (c,b)
            cur = set(cand)

            if (
                a==d
                or
                c==b
                or
                e1 in cur
                or
                e2 in cur
            ):
                continue

            cand[i] = e1
            cand[j] = e2
            successful_swaps += 1

            if successful_swaps >= 10:
                break

        if (
            successful_swaps < 4
            or
            set(cand) == set(AUTH_EDGES)
            or
            edge_overlap(cand) > 5
        ):
            continue

        if (
            (AUTH_SWITCH[0],AUTH_SWITCH[1]) not in set(cand)
            or
            (AUTH_SWITCH[2],AUTH_SWITCH[3]) in set(cand)
        ):
            continue

        cg = graph_object(cand)

        if nx.is_isomorphic(og,cg):
            continue

        if (
            [cg.in_degree(i) for i in range(8)]
            !=
            [og.in_degree(i) for i in range(8)]
        ):
            continue

        if (
            [cg.out_degree(i) for i in range(8)]
            !=
            [og.out_degree(i) for i in range(8)]
        ):
            continue

        return tuple(cand)

    raise RuntimeError(f"degree rewire failed seed={seed}")


def random_candidate(rng):
    universe = [
        (a,b)
        for a in range(8)
        for b in range(8)
        if a != b
    ]

    return rng.sample(universe,10)


@lru_cache(maxsize=None)
def complexity_matched(seed):
    rng = random.Random(200000+seed)
    og = graph_object(AUTH_EDGES)
    candidates = []

    for _ in range(450):
        cand = random_candidate(rng)

        old = (
            AUTH_SWITCH[0],
            AUTH_SWITCH[1]
        )

        new = (
            AUTH_SWITCH[2],
            AUTH_SWITCH[3]
        )

        if old not in cand:
            repl = next(
                (
                    k
                    for k,e in enumerate(cand)
                    if e not in AUTH_EDGES and e != new
                ),
                None
            )

            if repl is None:
                continue

            cand[repl] = old

        if (
            new in cand
            or
            len(set(cand)) != 10
            or
            edge_overlap(cand) > 4
        ):
            continue

        score = float(
            np.linalg.norm(
                fast_match_vector(cand)
                -
                AUTH_FAST
            )
        )

        candidates.append(
            (
                score,
                cand
            )
        )

    candidates.sort(
        key=lambda x:x[0]
    )

    for _,cand in candidates[:40]:
        if not nx.is_isomorphic(
            og,
            graph_object(cand)
        ):
            return tuple(cand)

    raise RuntimeError(
        f"complexity match failed seed={seed}"
    )


@lru_cache(maxsize=None)
def topology_matched(seed):
    rng = random.Random(300000+seed)
    og = graph_object(AUTH_EDGES)
    candidates = []

    for _ in range(900):
        cand = random_candidate(rng)

        old = (
            AUTH_SWITCH[0],
            AUTH_SWITCH[1]
        )

        new = (
            AUTH_SWITCH[2],
            AUTH_SWITCH[3]
        )

        if old not in cand:
            repl = next(
                (
                    k
                    for k,e in enumerate(cand)
                    if e not in AUTH_EDGES and e != new
                ),
                None
            )

            if repl is None:
                continue

            cand[repl] = old

        if (
            new in cand
            or
            len(set(cand)) != 10
            or
            edge_overlap(cand) > 4
        ):
            continue

        score = float(
            np.linalg.norm(
                fast_match_vector(cand)
                -
                AUTH_FAST
            )
        )

        candidates.append(
            (
                score,
                cand
            )
        )

    candidates.sort(
        key=lambda x:x[0]
    )

    best = None
    bestscore = 1e9

    for _,cand in candidates[:25]:
        if nx.is_isomorphic(
            og,
            graph_object(cand)
        ):
            continue

        score = float(
            np.linalg.norm(
                topology_vector(cand)
                -
                AUTH_TOPOLOGY
            )
        )

        if score < bestscore:
            bestscore = score
            best = tuple(cand)

    if best is None:
        raise RuntimeError(
            f"topology match failed seed={seed}"
        )

    return best


@lru_cache(maxsize=None)
def random_graph(seed):
    return tuple(
        random_candidate(
            random.Random(
                400000+seed
            )
        )
    )


def make_mapping(seed,variant=0):
    rng = random.Random(
        MASTER_SEED*10000
        +
        seed*100
        +
        variant
    )

    labels = [
        f"Q{i}"
        for i in range(8)
    ]

    rng.shuffle(labels)

    r2l = {
        i:
            labels[i]
        for i in range(8)
    }

    l2r = {
        v:
            k
        for k,v
        in r2l.items()
    }

    return r2l,l2r


def choose_valid_switch(edges,seed,exclude=None):
    rng = random.Random(
        500000+seed
    )

    eset = set(edges)

    olds = list(edges)
    rng.shuffle(olds)

    non = [
        (a,b)
        for a in range(8)
        for b in range(8)
        if (
            a != b
            and
            (a,b) not in eset
        )
    ]

    rng.shuffle(non)

    for old in olds:
        for new in non:
            s = old+new

            if (
                exclude is None
                or
                tuple(s) != tuple(exclude)
            ):
                return tuple(s)

    raise RuntimeError(
        "could not choose valid switch"
    )


def build_condition(seed,condition):
    variant = (
        1
        if condition == "I_ISOMORPHIC_RELABEL"
        else 0
    )

    r2l,l2r = make_mapping(
        seed,
        variant
    )

    edges = list(AUTH_EDGES)
    active = [1]*10
    order = list(AUTH_ORDER)
    keep = list(AUTH_KEEP)
    switch = tuple(AUTH_SWITCH)
    end = AUTH_END

    if condition == "D_ALL_DIRECTION_REVERSED":
        edges = [
            (b,a)
            for a,b
            in edges
        ]

        switch = (
            AUTH_SWITCH[1],
            AUTH_SWITCH[0],
            AUTH_SWITCH[3],
            AUTH_SWITCH[2]
        )

    elif condition == "O_ORDER_SHUFFLED":
        random.Random(
            600000+seed
        ).shuffle(order)

    elif condition == "M_SWITCH_REASSIGNED":
        switch = choose_valid_switch(
            edges,
            seed,
            exclude=AUTH_SWITCH
        )

    elif condition == "M0_SWITCH_NEUTRALIZED":
        switch = (
            AUTH_SWITCH[0],
            AUTH_SWITCH[1],
            AUTH_SWITCH[0],
            AUTH_SWITCH[1]
        )

    elif condition == "P_ANCHOR_REASSIGNED":
        keep = [0,7]

    elif condition == "T_TERMINAL_CHANGED":
        end = 0

    elif condition == "R_DEGREE_MATCHED_REWIRE":
        edges = list(
            degree_preserving_rewire(seed)
        )

        switch = tuple(
            AUTH_SWITCH
        )

    elif condition == "X_COMPLEXITY_MATCHED":
        edges = list(
            complexity_matched(seed)
        )

        switch = tuple(
            AUTH_SWITCH
        )

    elif condition == "Y_TOPOLOGY_MATCHED":
        edges = list(
            topology_matched(seed)
        )

        switch = tuple(
            AUTH_SWITCH
        )

    elif condition == "L_FORWARD_CONTROL":
        edges = list(
            FORWARD_CONTROL
        )

        switch = choose_valid_switch(
            edges,
            seed
        )

    elif condition == "G_RANDOM_CONTROL":
        edges = random_graph(seed)

        switch = choose_valid_switch(
            edges,
            seed
        )

    elif (
        condition.startswith("E")
        and
        condition.endswith("_DELETE")
    ):
        i = int(
            condition[
                1:
                condition.index("_")
            ]
        )

        active[i] = 0

    elif (
        condition.startswith("E")
        and
        condition.endswith("_REVERSE")
    ):
        i = int(
            condition[
                1:
                condition.index("_")
            ]
        )

        a,b = edges[i]
        edges[i] = (b,a)

    elif condition in {
        "A_AUTHENTIC",
        "I_ISOMORPHIC_RELABEL",
        "F_SURFACE_FORMAT_CONTROL"
    }:
        pass

    else:
        raise RuntimeError(
            "unknown condition "
            +
            condition
        )

    # Correct the three known switch-linked intervention semantics prospectively.
    # This does not alter the original V12. It prevents the V12-R prompt from
    # simultaneously referring to a disabled old switch edge (E7 delete/reverse)
    # or a new switch edge that became active after E9 reversal. These conditions
    # remain excluded from the original pure9/pure8 primary family gates.
    if condition in CORRECTIVE_SWITCH_CONDITIONS:
        active_for_switch = [
            e for i,e in enumerate(edges) if int(active[i]) == 1
        ]
        switch = choose_valid_switch(
            active_for_switch,
            seed + {
                "E7_DELETE": 71000,
                "E7_REVERSE": 72000,
                "E9_REVERSE": 79000,
            }[condition],
            exclude=AUTH_SWITCH
        )

    E = []

    for i,(a,b) in enumerate(edges):
        E.append({
            "id":
                f"e{i}",

            "u":
                r2l[a],

            "v":
                r2l[b],

            "active":
                int(active[i])
        })

    S = [
        r2l[switch[0]],
        r2l[switch[1]],
        r2l[switch[2]],
        r2l[switch[3]]
    ]

    payload = {
        "V":
            [
                r2l[i]
                for i in range(8)
            ],

        "E":
            E,

        "Q":
            [
                r2l[i]
                for i in order
            ],

        "K":
            [
                r2l[i]
                for i in keep
            ],

        "S":
            S,

        "Z":
            r2l[end],

        "C":
            4
    }

    meta = {
        "label_to_role":
            l2r,

        "edges_role":
            edges,

        "active":
            active,

        "order_role":
            order,

        "keep_role":
            keep,

        "switch_role":
            switch,

        "end_role":
            end
    }

    return payload,meta


INSTRUCTION = (
"You are given an anonymous relational program. "
"Node labels and edge ids are arbitrary identifiers. "
"Infer one coherent next relational state using only the supplied relations. "
"Return exactly one JSON object and no prose. "
"Required keys: seq, plus, minus, keep, switch, end, code. "
"seq must contain every node exactly once. "
"plus must contain exactly two directed node pairs not already present among active E edges. "
"minus must be exactly one active input edge id. "
"keep must contain exactly two distinct nodes. "
"switch must be [] or exactly four node identifiers [old_source,old_target,new_source,new_target]. "
"end must be one node identifier. "
"code must contain exactly four distinct active input edge ids. "
"Q is an ordering constraint. "
"K marks nodes whose relational continuity should be preserved. "
"S specifies a preferred relation transformation; "
"when S names an active old edge and an absent new edge, use it jointly when forming the next state. "
"Z is the preferred terminal node. "
"C is the requested compact code size. "
"Use E,Q,K,S,Z,C jointly. "
"An E item with active=0 is disabled and must not be treated as an active edge. "
"Do not assign semantic meanings to identifiers."
)


def build_prompt(seed,condition):
    payload,meta = build_condition(
        seed,
        condition
    )

    body = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2
        )
        if condition == "F_SURFACE_FORMAT_CONTROL"
        else
        canonical_json(payload)
    )

    return (
        INSTRUCTION
        +
        "\nPROGRAM="
        +
        body,
        payload,
        meta
    )


def active_label_edges(payload):
    return {
        (e["u"],e["v"])
        for e in payload["E"]
        if int(e["active"]) == 1
    }


def active_edge_ids(payload):
    return {
        e["id"]
        for e in payload["E"]
        if int(e["active"]) == 1
    }


def parse_json_object(text):
    if not text:
        raise RuntimeError(
            "EMPTY_VISIBLE_TEXT"
        )

    v = text.strip()

    if v.startswith("```"):
        v = v.strip("`")

        if v.lower().startswith("json"):
            v = v[4:].strip()

    a = v.find("{")
    b = v.rfind("}")

    if a < 0 or b < a:
        raise RuntimeError(
            "NO_JSON_OBJECT"
        )

    return json.loads(
        v[a:b+1]
    )


def extract_text(response):
    t = getattr(
        response,
        "output_text",
        None
    )

    if isinstance(t,str) and t.strip():
        return t.strip()

    parts = []

    for item in getattr(response,"output",None) or []:
        for c in getattr(item,"content",None) or []:
            x = getattr(
                c,
                "text",
                None
            )

            if isinstance(x,str) and x.strip():
                parts.append(
                    x.strip()
                )

    return "\n".join(parts).strip() or None


def parse_node_pairs(value,valid_nodes):
    out = []

    if not isinstance(value,list):
        return out

    for x in value:
        if (
            isinstance(x,list)
            and
            len(x)==2
            and
            x[0] in valid_nodes
            and
            x[1] in valid_nodes
            and
            x[0] != x[1]
        ):
            out.append(
                (x[0],x[1])
            )

    return out


def validate_output(obj,payload):
    req = {
        "seq",
        "plus",
        "minus",
        "keep",
        "switch",
        "end",
        "code"
    }

    if (
        not isinstance(obj,dict)
        or
        set(obj.keys()) != req
    ):
        return False,"KEYS"

    V = set(payload["V"])
    active_edges = active_label_edges(payload)
    ids = active_edge_ids(payload)

    seq = obj["seq"]

    if (
        not isinstance(seq,list)
        or
        len(seq) != 8
        or
        len(set(seq)) != 8
        or
        set(seq) != V
    ):
        return False,"SEQ"

    plus = parse_node_pairs(
        obj["plus"],
        V
    )

    if (
        len(obj["plus"]) != 2
        or
        len(plus) != 2
        or
        len(set(plus)) != 2
        or
        any(
            x in active_edges
            for x in plus
        )
    ):
        return False,"PLUS"

    if obj["minus"] not in ids:
        return False,"MINUS"

    keep = obj["keep"]

    if (
        not isinstance(keep,list)
        or
        len(keep) != 2
        or
        len(set(keep)) != 2
        or
        any(
            x not in V
            for x in keep
        )
    ):
        return False,"KEEP"

    sw = obj["switch"]

    if sw != []:
        if (
            not isinstance(sw,list)
            or
            len(sw) != 4
            or
            any(
                x not in V
                for x in sw
            )
        ):
            return False,"SWITCH"

    if obj["end"] not in V:
        return False,"END"

    code = obj["code"]

    if (
        not isinstance(code,list)
        or
        len(code) != 4
        or
        len(set(code)) != 4
        or
        any(
            x not in ids
            for x in code
        )
    ):
        return False,"CODE"

    return True,"OK"


@lru_cache(maxsize=1)
def get_client():
    key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY missing"
        )

    return OpenAI(
        api_key=key,
        timeout=60.0,
        max_retries=0
    )


def _main_json_schema():
    return {
        "type":
            "json_schema",

        "name":
            "wf1a_v12r_relational_state",

        "strict":
            True,

        "schema": {
            "type":
                "object",

            "properties": {
                "seq": {
                    "type":
                        "array",

                    "items": {
                        "type":
                            "string"
                    }
                },

                "plus": {
                    "type":
                        "array",

                    "items": {
                        "type":
                            "array",

                        "items": {
                            "type":
                                "string"
                        }
                    }
                },

                "minus": {
                    "type":
                        "string"
                },

                "keep": {
                    "type":
                        "array",

                    "items": {
                        "type":
                            "string"
                    }
                },

                "switch": {
                    "type":
                        "array",

                    "items": {
                        "type":
                            "string"
                    }
                },

                "end": {
                    "type":
                        "string"
                },

                "code": {
                    "type":
                        "array",

                    "items": {
                        "type":
                            "string"
                    }
                }
            },

            "required": [
                "seq",
                "plus",
                "minus",
                "keep",
                "switch",
                "end",
                "code"
            ],

            "additionalProperties":
                False
        }
    }


def _native_json_schema():
    return {
        "type":
            "json_schema",

        "name":
            "wf1a_v12r_native_graph",

        "strict":
            True,

        "schema": {
            "type":
                "object",

            "properties": {
                "E": {
                    "type":
                        "array",

                    "items": {
                        "type":
                            "array",

                        "items": {
                            "type":
                                "string"
                        }
                    }
                }
            },

            "required": [
                "E"
            ],

            "additionalProperties":
                False
        }
    }


def classify_failure(exc):
    name = type(exc).__name__
    msg = str(exc)
    low = (name + " " + msg).lower()
    if msg.startswith("INVALID_"):
        return "VALIDATOR"
    if "json" in low or "schema" in low or "parse" in low:
        return "STRUCTURED_OUTPUT"
    if "authentication" in low or "invalid_api_key" in low or "401" in low:
        return "AUTH"
    if "rate" in low or "429" in low:
        return "RATE_LIMIT"
    if "timeout" in low or "connection" in low:
        return "TRANSPORT"
    if "500" in low or "502" in low or "503" in low or "server" in low:
        return "SERVER"
    return "OTHER"


def api_create(client,prompt,format_spec):
    return client.responses.create(
        model=MODEL,
        input=prompt,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        reasoning={"effort":"none"},
        text={"format": format_spec},
        store=False
    )


def call_api(prompt,payload):
    client = get_client()
    failures = []

    for attempt in range(1,MAX_RETRIES+1):
        started = time.time()

        try:
            response = api_create(
                client,
                prompt,
                _main_json_schema()
            )

            text = extract_text(
                response
            )

            parsed = parse_json_object(
                text
            )

            valid,reason = validate_output(
                parsed,
                payload
            )

            if not valid:
                raise RuntimeError(
                    "INVALID_"
                    +
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
                else
                {
                    "repr":
                        repr(response)
                }
            )

            return {
                "ok":
                    True,

                "attempt":
                    attempt,

                "attempt_failures":
                    failures,

                "latency":
                    time.time()-started,

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

        except Exception as e:
            failures.append({
                "attempt":
                    attempt,

                "type":
                    type(e).__name__,

                "category":
                    classify_failure(e),

                "error":
                    str(e),

                "utc":
                    utcnow()
            })

            time.sleep(
                min(
                    2**(attempt-1),
                    8
                )
            )

    return {
        "ok":
            False,

        "failures":
            failures
    }


def positions(seq):
    return {
        x:i
        for i,x
        in enumerate(seq)
    }


def edge_seq_coherence(edges,seq):
    if not edges:
        return 1.0

    p = positions(seq)

    return float(
        np.mean([
            1.0
            if p[a] < p[b]
            else 0.0
            for a,b in edges
        ])
    )


def order_fidelity(seq,target):
    p = positions(seq)
    q = positions(target)

    total = 0
    same = 0

    for a in range(8):
        for b in range(a+1,8):
            total += 1

            same += int(
                (p[a] < p[b])
                ==
                (q[a] < q[b])
            )

    return same/total


def role_edges_from_payload(payload,meta):
    l2r = meta["label_to_role"]
    out = {}

    for e in payload["E"]:
        out[e["id"]] = (
            l2r[e["u"]],
            l2r[e["v"]],
            int(e["active"])
        )

    return out


def extract_features(seed,condition,payload,meta,parsed):
    l2r = meta["label_to_role"]
    idmap = role_edges_from_payload(
        payload,
        meta
    )

    active = {
        eid:
            (a,b)
        for eid,(a,b,on)
        in idmap.items()
        if on == 1
    }

    active_edges = set(
        active.values()
    )

    seq = [
        l2r[x]
        for x in parsed["seq"]
    ]

    plus = {
        (
            l2r[a],
            l2r[b]
        )
        for a,b
        in parse_node_pairs(
            parsed["plus"],
            set(payload["V"])
        )
    }

    minus_id = parsed["minus"]
    minus_edge = active[minus_id]

    keep = [
        l2r[x]
        for x in parsed["keep"]
    ]

    end = l2r[
        parsed["end"]
    ]

    code_edges = {
        active[eid]
        for eid
        in parsed["code"]
    }

    final = set(active_edges)
    final.discard(minus_edge)
    final.update(plus)

    g = graph_object(final)
    ug = g.to_undirected()
    scc = list(
        nx.strongly_connected_components(g)
    )

    rec = nx.reciprocity(g)
    rec = 0.0 if rec is None else float(rec)

    indeg = np.array(
        [
            g.in_degree(i)
            for i in range(8)
        ],
        float
    )

    outdeg = np.array(
        [
            g.out_degree(i)
            for i in range(8)
        ],
        float
    )

    A = nx.to_numpy_array(
        g,
        nodelist=range(8),
        dtype=float
    )

    eig = np.linalg.eigvals(A)

    spectral = (
        float(
            np.max(
                np.abs(eig)
            )
        )
        if eig.size
        else 0.0
    )

    two,three = cycle_counts(g)
    reachfrac,meanpath = reachability_stats(g)

    input_graph = graph_object(active_edges)
    reach = []

    for a,b in plus:
        try:
            reach.append(
                float(
                    nx.has_path(
                        input_graph,
                        a,
                        b
                    )
                )
            )
        except Exception:
            reach.append(0.0)

    code_conn = 0.0

    if code_edges:
        cg = nx.Graph()
        cg.add_edges_from(code_edges)

        nodes = set(
            sum(
                (
                    [a,b]
                    for a,b
                    in code_edges
                ),
                []
            )
        )

        code_conn = (
            max(
                len(x)
                for x in nx.connected_components(cg)
            )
            /
            len(nodes)
        )

    targetK = set(meta["keep_role"])
    actualK = set(keep)

    targetS = tuple(
        meta["switch_role"]
    )

    sw = parsed["switch"]

    actualS = (
        tuple(
            l2r[x]
            for x in sw
        )
        if (
            isinstance(sw,list)
            and
            len(sw) == 4
        )
        else None
    )

    s_valid = (
        (targetS[0],targetS[1]) in active_edges
        and
        (targetS[2],targetS[3]) not in active_edges
        and
        (targetS[0],targetS[1])
        !=
        (targetS[2],targetS[3])
    )

    s_exact = (
        float(
            actualS == targetS
        )
        if s_valid
        else
        float(
            sw == []
        )
    )

    s_old = (
        float(
            minus_edge
            ==
            (
                targetS[0],
                targetS[1]
            )
        )
        if s_valid
        else 0.0
    )

    s_new = (
        float(
            (
                targetS[2],
                targetS[3]
            )
            in plus
        )
        if s_valid
        else 0.0
    )

    s_applied = (
        (s_old+s_new)/2.0
        if s_valid
        else
        float(sw==[])
    )

    incident = [
        e
        for e in active_edges
        if (
            e[0] in targetK
            or
            e[1] in targetK
        )
    ]

    retained = (
        sum(
            1
            for e in incident
            if e in final
        )
        /
        len(incident)
        if incident
        else 1.0
    )

    outz = g.out_degree(
        meta["end_role"]
    )

    z_sink = 1.0/(1.0+outz)

    zpos = (
        positions(seq)[
            meta["end_role"]
        ]
        /
        7.0
    )

    return {
        "SEED":
            seed,

        "CONDITION":
            condition,

        "SEQ_ACTIVE_EDGE_COHERENCE":
            edge_seq_coherence(
                active_edges,
                seq
            ),

        "PLUS_SEQ_COHERENCE":
            edge_seq_coherence(
                plus,
                seq
            ),

        "PLUS_REACHABILITY":
            float(np.mean(reach))
            if reach
            else 0.0,

        "MINUS_SEQ_CONFLICT":
            1.0
            -
            edge_seq_coherence(
                {
                    minus_edge
                },
                seq
            ),

        "CODE_SEQ_COHERENCE":
            edge_seq_coherence(
                code_edges,
                seq
            ),

        "CODE_CONNECTEDNESS":
            float(code_conn),

        "FINAL_RECIPROCITY":
            rec,

        "FINAL_SCC_COUNT":
            float(len(scc)),

        "FINAL_LARGEST_SCC_FRAC":
            max(
                (
                    len(x)
                    for x in scc
                ),
                default=0
            )
            /
            8.0,

        "FINAL_TRANSITIVITY":
            float(
                nx.transitivity(ug)
            )
            if ug.number_of_edges()
            else 0.0,

        "FINAL_IN_DEGREE_SD":
            float(indeg.std()),

        "FINAL_OUT_DEGREE_SD":
            float(outdeg.std()),

        "FINAL_SPECTRAL_RADIUS":
            spectral,

        "FINAL_TWO_CYCLES":
            float(two),

        "FINAL_THREE_CYCLES":
            float(three),

        "FINAL_REACHABILITY_FRAC":
            reachfrac,

        "FINAL_MEAN_REACHABLE_PATH":
            meanpath,

        "ORDER_FIDELITY":
            order_fidelity(
                seq,
                meta["order_role"]
            ),

        "K_EXACT":
            float(
                actualK == targetK
            ),

        "K_JACCARD":
            jaccard(
                actualK,
                targetK
            ),

        "S_EXACT":
            s_exact,

        "S_APPLIED_SCORE":
            s_applied,

        "K_INCIDENT_RETENTION":
            float(retained),

        "Z_EXACT":
            float(
                end
                ==
                meta["end_role"]
            ),

        "Z_FINAL_SINKNESS":
            z_sink,

        "Z_SEQUENCE_POSITION":
            zpos
    }


def native_labels(seed,frame):
    rng = random.Random(
        MASTER_SEED*100000
        +
        seed*10
        +
        list(NATIVE_FRAMES).index(frame)
    )

    labels = [
        f"N{rng.randrange(0,65536):04X}"
        for _ in range(8)
    ]

    while len(set(labels)) < 8:
        labels = [
            f"N{rng.randrange(0,65536):04X}"
            for _ in range(8)
        ]

    rng.shuffle(labels)

    return labels


def native_prompt(seed,frame):
    labels = native_labels(
        seed,
        frame
    )

    body = canonical_json({
        "nodes":
            labels
    })

    inst = (
        NATIVE_FRAMES[frame]
        +
        " Use exactly the supplied 8 opaque nodes and exactly 10 directed edges. "
        "Do not use self-edges or duplicate edges. "
        "Node names are meaningless. "
        "Return exactly one JSON object and no prose with key E only; "
        "E must be a list of exactly 10 [source,target] pairs. "
        "Do not use named theories, architectures, domains, hierarchies, "
        "feedback terminology, or semantic labels."
    )

    return (
        inst
        +
        "\nINPUT="
        +
        body,
        labels
    )


def validate_native(obj,labels):
    if (
        not isinstance(obj,dict)
        or
        set(obj.keys()) != {"E"}
    ):
        return False,"KEYS"

    V = set(labels)

    E = parse_node_pairs(
        obj["E"],
        V
    )

    if (
        not isinstance(obj["E"],list)
        or
        len(obj["E"]) != 10
        or
        len(E) != 10
        or
        len(set(E)) != 10
    ):
        return False,"E"

    return True,"OK"


def call_native(seed,frame):
    client = get_client()

    prompt,labels = native_prompt(
        seed,
        frame
    )

    failures = []

    for attempt in range(1,MAX_RETRIES+1):
        started = time.time()

        try:
            response = api_create(
                client,
                prompt,
                _native_json_schema()
            )

            text = extract_text(response)
            parsed = parse_json_object(text)

            ok,reason = validate_native(
                parsed,
                labels
            )

            if not ok:
                raise RuntimeError(
                    "INVALID_NATIVE_"
                    +
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
                else
                {
                    "repr":
                        repr(response)
                }
            )

            return {
                "ok":
                    True,

                "attempt":
                    attempt,

                "attempt_failures":
                    failures,

                "latency":
                    time.time()-started,

                "text":
                    text,

                "parsed":
                    parsed,

                "labels":
                    labels,

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
                    raw,

                "prompt":
                    prompt
            }

        except Exception as e:
            failures.append({
                "attempt":
                    attempt,

                "type":
                    type(e).__name__,

                "category":
                    classify_failure(e),

                "error":
                    str(e),

                "utc":
                    utcnow()
            })

            time.sleep(
                min(
                    2**(attempt-1),
                    8
                )
            )

    return {
        "ok":
            False,

        "failures":
            failures,

        "prompt":
            prompt
    }


def native_edges(parsed,labels):
    l2r = {
        x:i
        for i,x
        in enumerate(labels)
    }

    return [
        (
            l2r[a],
            l2r[b]
        )
        for a,b
        in parse_node_pairs(
            parsed["E"],
            set(labels)
        )
    ]


def spec():
    return {
        "experiment":
            EXPERIMENT,

        "model":
            MODEL,

        "master_seed":
            MASTER_SEED,

        "seed_ids":
            [1080,1159],

        "n_reps":
            N_REPS,

        "development":
            "1080-1119",

        "confirmation":
            "1120-1139",

        "final_holdout":
            "1140-1159",

        "conditions":
            CONDITIONS,

        "expected_main_cells":
            EXPECTED_MAIN_CELLS,

        "native_frames":
            list(NATIVE_FRAMES),

        "expected_native_cells":
            EXPECTED_NATIVE_CELLS,

        "auth_edges":
            AUTH_EDGES,

        "auth_switch":
            AUTH_SWITCH,

        "switch_linked_delete_indexes":
            [7],

        "switch_linked_reverse_indexes":
            [7,9],

        "pure_delete_indexes":
            PURE_DELETE_INDEXES,

        "pure_reverse_indexes":
            PURE_REVERSE_INDEXES,

        "primary_behavior_features":
            BEHAVIOR_FEATURES,

        "direct_operator_features":
            DIRECT_OPERATOR_FEATURES,

        "primary_distance":
            "Ledoit-Wolf regularized Mahalanobis distance; nuisance covariance learned only from development A/I/F baseline after development baseline standardization",

        "secondary_distance":
            "development-baseline standardized Euclidean distance",

        "surface_baseline":
            "per-seed mean of A-to-isomorphic and A-to-format-control distances",

        "edge_replication":
            "individual edge is replicated only if excess distance is positive and Holm p<0.05 in BOTH confirmation and final holdout; delete e7 and reverse e7/e9 are reported separately as switch-linked/dependent",

        "multiplicity":
            "Holm within each predeclared primary-test family/partition; Holm across 10 individual edges within delete and reverse families",

        "classifier":
            "development-only logistic regression on behavior features. Broad classifier vs all lesions plus hard-control classifier vs R/X/Y. Strong gate requires hard-control AUC >=0.85 in BOTH confirmation and final holdout.",

        "model_native_proxy":
            "three seed-randomized graph-only neutral elicitation frames; no Q/K/S/Z supplied; rich isomorphism-invariant topology vector; indirect behavioral proxy only",

        "strong_primary_gate": {
            "hard_control_auc":
                "confirmation AND final holdout >=0.85",

            "pure9_delete":
                "positive excess and Holm p<0.01 in confirmation AND final holdout",

            "pure8_reverse":
                "positive excess and Holm p<0.01 in confirmation AND final holdout",

            "specificity":
                "R, X and Y each positive excess and Holm p<0.01 in confirmation AND final holdout",

            "surface_invariance":
                "each hard control exceeds surface nuisance baseline with Holm p<0.01 in confirmation AND final holdout",

            "operator_profile":
                "reported and manipulation-checked separately; direct adherence features do not enter the primary topology distance/classifier"
        },

        "secondary_triangulation_gate":
            "model-native pooled authentic closer than >=3/5 controls after Holm p<0.05 in BOTH confirmation and final holdout, with positive direction in >=2/3 native frames in final holdout",

        "v12r_corrections": {
            "fresh_seed_range": "1080-1159; original V12 used 1000-1079",
            "structured_output": "Responses API strict json_schema mode using the frozen schemas rather than legacy json_object mode",
            "e7_delete": "switch deterministically reassigned to a valid active-old/absent-new relation; remains excluded from pure9 primary pool",
            "e7_reverse": "switch deterministically reassigned; remains excluded from pure8 primary pool",
            "e9_reverse": "switch deterministically reassigned so reversed 3->6 does not collide with the requested new switch edge; remains excluded from pure8 primary pool",
            "feasibility": "every seed x condition must pass deterministic valid-output witness preflight before API execution",
            "failure_taxonomy": "validator, structured-output, auth, transport, rate-limit, server, other are recorded separately",
            "original_v12_immutable": True
        },

        "interpretation_boundary":
            "Black-box behavioral structural evidence only. Model-native module is an indirect output-based proxy, not direct inspection of weights, activations, hidden circuits, training data, or global uniqueness."
    }


def verify_sources():
    OUT.mkdir(parents=True, exist_ok=True)

    targets = [
        ("ORIGINAL_FREEZE_RECORD", ORIG_FREEZE_DIR / "00_V12_FREEZE_RECORD.txt", EXPECTED_ORIGINAL_FREEZE_SHA256),
        ("ORIGINAL_MISSING_CELLS", ORIG_FREEZE_DIR / "15_MISSING_CELLS.csv", EXPECTED_ORIGINAL_MISSING_SHA256),
        ("ORIGINAL_V12_RUNNER", ORIG_FREEZE_DIR / "RUN_V12_RELATIONAL_TOPOLOGY_CAUSAL_MAP_JSON_R4.py", EXPECTED_ORIGINAL_RUNNER_SHA256),
    ]

    rows = []
    for name,path,expected in targets:
        if not path.exists():
            raise RuntimeError(f"missing staged original V12 input: {path}")
        actual = sha256_file(path).upper()
        ok = actual == expected.upper()
        rows.append({
            "SOURCE": name,
            "PATH": str(path),
            "EXPECTED_SHA256": expected.upper(),
            "ACTUAL_SHA256": actual,
            "PASS": bool(ok),
            "UTC": utcnow(),
        })
        if not ok:
            raise RuntimeError(f"original V12 staged hash mismatch: {name}")

    pd.DataFrame(rows).to_csv(OUT / "19_SOURCE_PROVENANCE.csv", index=False)
    return 0


def snapshot_v11_sources():
    return 0


def preflight(write_files=True):
    rows = []
    control = []

    og = graph_object(AUTH_EDGES)

    for n,seed in enumerate(SEED_IDS,1):
        for c in CONDITIONS:
            prompt,payload,meta = build_prompt(
                seed,
                c
            )

            witness = fake_parsed(payload)
            witness_ok, witness_reason = validate_output(witness, payload)
            if not witness_ok:
                raise RuntimeError(
                    f"deterministic feasibility witness failed seed={seed} condition={c} reason={witness_reason}"
                )

            act = [
                (a,b)
                for (a,b),on
                in zip(
                    meta["edges_role"],
                    meta["active"]
                )
                if on
            ]

            expected = (
                9
                if c in EDGE_DELETE_CONDITIONS
                else 10
            )

            if len(act) != expected:
                raise RuntimeError(
                    f"active edge count {seed} {c}: {len(act)} != {expected}"
                )

            iso = (
                nx.is_isomorphic(
                    og,
                    graph_object(act)
                )
                if len(act) == 10
                else False
            )

            if (
                c in {
                    "R_DEGREE_MATCHED_REWIRE",
                    "X_COMPLEXITY_MATCHED",
                    "Y_TOPOLOGY_MATCHED"
                }
                and
                iso
            ):
                raise RuntimeError(
                    f"control unexpectedly isomorphic seed={seed} condition={c}"
                )

            if c == "R_DEGREE_MATCHED_REWIRE":
                g = graph_object(act)

                if (
                    [g.in_degree(i) for i in range(8)]
                    !=
                    [og.in_degree(i) for i in range(8)]
                ):
                    raise RuntimeError(
                        "node-level in-degree match failed"
                    )

                if (
                    [g.out_degree(i) for i in range(8)]
                    !=
                    [og.out_degree(i) for i in range(8)]
                ):
                    raise RuntimeError(
                        "node-level out-degree match failed"
                    )

                if edge_overlap(act) > 5:
                    raise RuntimeError(
                        "degree control overlap too high"
                    )

            if (
                c in {
                    "X_COMPLEXITY_MATCHED",
                    "Y_TOPOLOGY_MATCHED"
                }
                and
                edge_overlap(act) > 4
            ):
                raise RuntimeError(
                    "matched control overlap too high"
                )

            old = (
                meta["switch_role"][0],
                meta["switch_role"][1]
            )

            new = (
                meta["switch_role"][2],
                meta["switch_role"][3]
            )

            svalid = (
                old in set(act)
                and
                new not in set(act)
                and
                old != new
            )

            rows.append({
                "SEED":
                    seed,

                "CONDITION":
                    c,

                "PROMPT_SHA256":
                    sha256_text(prompt),

                "PROMPT_CHARS":
                    len(prompt),

                "ACTIVE_EDGES":
                    len(act),

                "FEASIBILITY_WITNESS_VALID":
                    bool(witness_ok),

                "FEASIBILITY_WITNESS_SHA256":
                    sha256_text(canonical_json(witness)),

                "CORRECTIVE_SWITCH_SEMANTICS":
                    c in CORRECTIVE_SWITCH_CONDITIONS,

                "PAYLOAD_JSON":
                    canonical_json(payload),

                "PROMPT":
                    prompt
            })

            control.append({
                "SEED":
                    seed,

                "CONDITION":
                    c,

                "ACTIVE_EDGES":
                    len(act),

                "EDGE_OVERLAP_AUTH":
                    edge_overlap(act),

                "ISOMORPHIC_TO_AUTH":
                    iso,

                "COMPLEXITY_DISTANCE":
                    float(
                        np.linalg.norm(
                            complexity_vector(act)
                            -
                            AUTH_COMPLEXITY
                        )
                    )
                    if len(act)==10
                    else np.nan,

                "TOPOLOGY_DISTANCE":
                    float(
                        np.linalg.norm(
                            topology_vector(act)
                            -
                            AUTH_TOPOLOGY
                        )
                    )
                    if len(act)==10
                    else np.nan,

                "SWITCH_VALID":
                    svalid,

                "FEASIBILITY_WITNESS_VALID":
                    bool(witness_ok),

                "CORRECTIVE_SWITCH_SEMANTICS":
                    c in CORRECTIVE_SWITCH_CONDITIONS
            })

        if n % 10 == 0:
            print(
                f"Preflight seeds: {n}/{len(SEED_IDS)}",
                flush=True
            )

    nrows = []

    for seed in SEED_IDS:
        for frame in NATIVE_FRAMES:
            prompt,labels = native_prompt(
                seed,
                frame
            )

            nrows.append({
                "SEED":
                    seed,

                "FRAME":
                    frame,

                "PROMPT_SHA256":
                    sha256_text(prompt),

                "LABELS":
                    "|".join(labels),

                "PROMPT":
                    prompt
            })

    if (
        pd.DataFrame(nrows).PROMPT_SHA256.nunique()
        !=
        len(SEED_IDS)*len(NATIVE_FRAMES)
    ):
        raise RuntimeError(
            "native prompt variants are not unique"
        )

    if write_files:
        OUT.mkdir(
            parents=True,
            exist_ok=True
        )

        pd.DataFrame(rows).to_csv(
            OUT
            /
            "05_FROZEN_PROMPTS.csv",
            index=False
        )

        pd.DataFrame(control).to_csv(
            OUT
            /
            "14_CONTROL_MATCH_AUDIT.csv",
            index=False
        )

        pd.DataFrame(nrows).to_csv(
            OUT
            /
            "26_NATIVE_FROZEN_PROMPTS.csv",
            index=False
        )

        (
            OUT
            /
            "00D_PREFLIGHT_SUMMARY.json"
        ).write_text(
            json.dumps({
                "utc":
                    utcnow(),

                "main_prompts":
                    len(rows),

                "native_prompt_variants":
                    len(nrows),

                "conditions":
                    len(CONDITIONS),

                "expected_main_cells":
                    EXPECTED_MAIN_CELLS,

                "expected_native_cells":
                    EXPECTED_NATIVE_CELLS,

                "fresh_seed_range":
                    [min(SEED_IDS), max(SEED_IDS)],

                "corrective_switch_conditions":
                    sorted(CORRECTIVE_SWITCH_CONDITIONS),

                "all_deterministic_feasibility_witnesses_valid":
                    bool(all(r["FEASIBILITY_WITNESS_VALID"] for r in rows))
            },
            indent=2),
            encoding="utf-8"
        )

    return rows,control,nrows


def load_unique(path):
    out = {}

    if not Path(path).exists():
        return out

    with Path(path).open(
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)

                if (
                    r.get("status") == "success"
                    and
                    r["cell_key"] not in out
                ):
                    out[
                        r["cell_key"]
                    ] = r

    return out


def fit_distance_model(sdf):
    base = sdf[
        (
            sdf.SEED.isin(DEV)
        )
        &
        (
            sdf.CONDITION.isin([
                "A_AUTHENTIC",
                "I_ISOMORPHIC_RELABEL",
                "F_SURFACE_FORMAT_CONTROL"
            ])
        )
    ][
        BEHAVIOR_FEATURES
    ].to_numpy(float)

    means = base.mean(axis=0)

    scales = base.std(
        axis=0,
        ddof=1
    )

    scales[
        ~np.isfinite(scales)
    ] = 1.0

    scales[
        scales < 1e-6
    ] = 1.0

    z = (base-means)/scales

    lw = LedoitWolf().fit(z)

    precision = lw.precision_

    return (
        means,
        scales,
        precision,
        float(lw.shrinkage_)
    )


def distances(rowa,rowb,scales,precision):
    d = (
        rowb[
            BEHAVIOR_FEATURES
        ].to_numpy(float)
        -
        rowa[
            BEHAVIOR_FEATURES
        ].to_numpy(float)
    ) / scales

    eu = float(
        np.sqrt(
            np.sum(d*d)
        )
    )

    mah = float(
        np.sqrt(
            max(
                0.0,
                float(
                    d
                    @
                    precision
                    @
                    d
                )
            )
        )
    )

    return mah,eu


def condition_distance_table(sdf,scales,precision):
    rows = []

    for seed in SEED_IDS:
        a = sdf[
            (
                sdf.SEED==seed
            )
            &
            (
                sdf.CONDITION=="A_AUTHENTIC"
            )
        ].iloc[0]

        for c in CONDITIONS:
            b = sdf[
                (
                    sdf.SEED==seed
                )
                &
                (
                    sdf.CONDITION==c
                )
            ].iloc[0]

            mah,eu = distances(
                a,
                b,
                scales,
                precision
            )

            rows.append({
                "SEED":
                    seed,

                "CONDITION":
                    c,

                "MAHALANOBIS":
                    mah,

                "STD_EUCLIDEAN":
                    eu
            })

    return pd.DataFrame(rows)


def nuisance_baseline(ddf,seeds):
    i = ddf[
        (
            ddf.SEED.isin(seeds)
        )
        &
        (
            ddf.CONDITION=="I_ISOMORPHIC_RELABEL"
        )
    ].set_index(
        "SEED"
    ).MAHALANOBIS

    f = ddf[
        (
            ddf.SEED.isin(seeds)
        )
        &
        (
            ddf.CONDITION=="F_SURFACE_FORMAT_CONTROL"
        )
    ].set_index(
        "SEED"
    ).MAHALANOBIS

    idx = sorted(
        set(i.index)
        &
        set(f.index)
    )

    return pd.Series({
        s:
            (
                float(i.loc[s])
                +
                float(f.loc[s])
            )
            /
            2.0
        for s
        in idx
    })


def test_condition_vs_surface(ddf,seeds,condition,seed_offset):
    base = nuisance_baseline(
        ddf,
        seeds
    )

    x = ddf[
        (
            ddf.SEED.isin(seeds)
        )
        &
        (
            ddf.CONDITION==condition
        )
    ].set_index(
        "SEED"
    ).MAHALANOBIS

    idx = sorted(
        set(base.index)
        &
        set(x.index)
    )

    diff = np.array(
        [
            x.loc[s]
            -
            base.loc[s]
            for s
            in idx
        ],
        float
    )

    return (
        len(diff),
        float(diff.mean()),
        signflip(
            diff,
            MASTER_SEED+seed_offset
        )
    )


def test_family_vs_surface(ddf,seeds,conditions,seed_offset):
    base = nuisance_baseline(
        ddf,
        seeds
    )

    x = ddf[
        (
            ddf.SEED.isin(seeds)
        )
        &
        (
            ddf.CONDITION.isin(conditions)
        )
    ].groupby(
        "SEED"
    ).MAHALANOBIS.mean()

    idx = sorted(
        set(base.index)
        &
        set(x.index)
    )

    diff = np.array(
        [
            x.loc[s]
            -
            base.loc[s]
            for s
            in idx
        ],
        float
    )

    return (
        len(diff),
        float(diff.mean()),
        signflip(
            diff,
            MASTER_SEED+seed_offset
        )
    )


def classifier_rows(sdf):
    out = []

    sets = {
        "BROAD": (
            {
                "A_AUTHENTIC",
                "I_ISOMORPHIC_RELABEL",
                "F_SURFACE_FORMAT_CONTROL"
            },

            set(CONDITIONS)
            -
            {
                "A_AUTHENTIC",
                "I_ISOMORPHIC_RELABEL",
                "F_SURFACE_FORMAT_CONTROL"
            }
        ),

        "HARD_CONTROLS": (
            {
                "A_AUTHENTIC",
                "I_ISOMORPHIC_RELABEL",
                "F_SURFACE_FORMAT_CONTROL"
            },

            {
                "R_DEGREE_MATCHED_REWIRE",
                "X_COMPLEXITY_MATCHED",
                "Y_TOPOLOGY_MATCHED"
            }
        )
    }

    for kind,(pos,neg) in sets.items():
        df = sdf[
            sdf.CONDITION.isin(
                pos|neg
            )
        ].copy()

        df["Y"] = (
            df.CONDITION.isin(pos)
        ).astype(int)

        model = Pipeline([
            (
                "scale",
                StandardScaler()
            ),
            (
                "lr",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=MASTER_SEED
                )
            )
        ])

        train = df[
            df.SEED.isin(DEV)
        ]

        model.fit(
            train[
                BEHAVIOR_FEATURES
            ],
            train.Y
        )

        for part,seeds in [
            ("DEVELOPMENT",DEV),
            ("CONFIRMATION",CONF),
            ("FINAL_HOLDOUT",HOLD)
        ]:
            s = df[
                df.SEED.isin(seeds)
            ]

            pr = model.predict_proba(
                s[
                    BEHAVIOR_FEATURES
                ]
            )[:,1]

            pred = (
                pr >= .5
            ).astype(int)

            out.append({
                "CLASSIFIER":
                    kind,

                "PARTITION":
                    part,

                "N":
                    len(s),

                "ROC_AUC":
                    float(
                        roc_auc_score(
                            s.Y,
                            pr
                        )
                    ),

                "ACCURACY":
                    float(
                        accuracy_score(
                            s.Y,
                            pred
                        )
                    )
            })

    return out


def run_analysis(successful,meta_cache,payload_cache,native_success):
    feats = []

    for r in successful.values():
        seed = int(r["seed"])
        c = r["condition"]

        row = extract_features(
            seed,
            c,
            payload_cache[
                (
                    seed,
                    c
                )
            ],
            meta_cache[
                (
                    seed,
                    c
                )
            ],
            r["parsed"]
        )

        row.update({
            "REP":
                int(r["rep"]),

            "CELL_KEY":
                r["cell_key"],

            "OUTPUT_TOKENS_DIAGNOSTIC":
                int(
                    r.get(
                        "output_tokens",
                        0
                    )
                    or 0
                ),

            "LATENCY_SECONDS_DIAGNOSTIC":
                float(
                    r.get(
                        "latency_seconds",
                        0
                    )
                    or 0
                )
        })

        feats.append(row)

    rdf = pd.DataFrame(feats)

    rdf.to_csv(
        OUT
        /
        "06_RESPONSE_GRAPH_FEATURES.csv",
        index=False
    )

    agg = (
        BEHAVIOR_FEATURES
        +
        DIRECT_OPERATOR_FEATURES
        +
        [
            "OUTPUT_TOKENS_DIAGNOSTIC",
            "LATENCY_SECONDS_DIAGNOSTIC"
        ]
    )

    sdf = rdf.groupby(
        [
            "SEED",
            "CONDITION"
        ],
        as_index=False
    )[agg].mean()

    sdf.to_csv(
        OUT
        /
        "07_SEED_LEVEL_GRAPH_VECTOR.csv",
        index=False
    )

    crows = []

    for c in CONDITIONS:
        s = sdf[
            sdf.CONDITION==c
        ]

        row = {
            "CONDITION":
                c,

            "N":
                len(s)
        }

        for f in (
            BEHAVIOR_FEATURES
            +
            DIRECT_OPERATOR_FEATURES
        ):
            row[f+"_MEAN"] = float(
                s[f].mean()
            )

            row[f+"_SD"] = float(
                s[f].std()
            )

        crows.append(row)

    pd.DataFrame(
        crows
    ).to_csv(
        OUT
        /
        "08_CONDITION_RESULTS.csv",
        index=False
    )

    means,scales,precision,shrink = fit_distance_model(
        sdf
    )

    pd.DataFrame({
        "FEATURE":
            BEHAVIOR_FEATURES,

        "BASELINE_MEAN":
            means,

        "BASELINE_SD":
            scales
    }).to_csv(
        OUT
        /
        "21_DEVELOPMENT_DISTANCE_SCALES.csv",
        index=False
    )

    pd.DataFrame(
        precision,
        index=BEHAVIOR_FEATURES,
        columns=BEHAVIOR_FEATURES
    ).to_csv(
        OUT
        /
        "21A_DEVELOPMENT_PRECISION_MATRIX.csv"
    )

    (
        OUT
        /
        "21B_DISTANCE_MODEL.json"
    ).write_text(
        json.dumps({
            "shrinkage":
                shrink,

            "baseline_conditions": [
                "A_AUTHENTIC",
                "I_ISOMORPHIC_RELABEL",
                "F_SURFACE_FORMAT_CONTROL"
            ],

            "partition":
                "DEVELOPMENT"
        },
        indent=2),
        encoding="utf-8"
    )

    ddf = condition_distance_table(
        sdf,
        scales,
        precision
    )

    ddf.to_csv(
        OUT
        /
        "22_DISTANCE_FROM_AUTHENTIC.csv",
        index=False
    )

    edge_rows = []

    for part,seeds in [
        ("ALL",set(SEED_IDS)),
        ("DEVELOPMENT",DEV),
        ("CONFIRMATION",CONF),
        ("FINAL_HOLDOUT",HOLD)
    ]:
        for family,conds in [
            ("DELETE",EDGE_DELETE_CONDITIONS),
            ("REVERSE",EDGE_REVERSE_CONDITIONS)
        ]:
            local = []

            for idx,c in enumerate(conds):
                n,e,p = test_condition_vs_surface(
                    ddf,
                    seeds,
                    c,
                    10000
                    +
                    (
                        0
                        if family=="DELETE"
                        else 1000
                    )
                    +
                    idx
                    +
                    len(edge_rows)
                )

                local.append({
                    "PARTITION":
                        part,

                    "FAMILY":
                        family,

                    "EDGE":
                        idx,

                    "CONDITION":
                        c,

                    "N":
                        n,

                    "EXCESS_DISTANCE":
                        e,

                    "P":
                        p,

                    "SWITCH_LINKED":
                        (
                            (
                                family=="DELETE"
                                and
                                idx==7
                            )
                            or
                            (
                                family=="REVERSE"
                                and
                                idx in {7,9}
                            )
                        )
                })

            adj = holm_adjust(
                [
                    r["P"]
                    for r in local
                ]
            )

            for r,q in zip(local,adj):
                r["HOLM_P"] = float(q)

                r["PASS"] = bool(
                    r["EXCESS_DISTANCE"]>0
                    and
                    q<0.05
                )

                edge_rows.append(r)

    edf = pd.DataFrame(edge_rows)

    rep = []

    for family in [
        "DELETE",
        "REVERSE"
    ]:
        for idx in range(10):
            c = edf[
                (
                    edf.FAMILY==family
                )
                &
                (
                    edf.EDGE==idx
                )
            ].set_index(
                "PARTITION"
            )

            ok = bool(
                c.loc[
                    "CONFIRMATION",
                    "PASS"
                ]
                and
                c.loc[
                    "FINAL_HOLDOUT",
                    "PASS"
                ]
                and
                c.loc[
                    "CONFIRMATION",
                    "EXCESS_DISTANCE"
                ]
                > 0
                and
                c.loc[
                    "FINAL_HOLDOUT",
                    "EXCESS_DISTANCE"
                ]
                > 0
            )

            rep.append({
                "FAMILY":
                    family,

                "EDGE":
                    idx,

                "SWITCH_LINKED":
                    (
                        (
                            family=="DELETE"
                            and
                            idx==7
                        )
                        or
                        (
                            family=="REVERSE"
                            and
                            idx in {7,9}
                        )
                    ),

                "REPLICATED_CONFIRMATION_AND_HOLDOUT":
                    ok,

                "CONF_EXCESS":
                    float(
                        c.loc[
                            "CONFIRMATION",
                            "EXCESS_DISTANCE"
                        ]
                    ),

                "CONF_HOLM_P":
                    float(
                        c.loc[
                            "CONFIRMATION",
                            "HOLM_P"
                        ]
                    ),

                "HOLD_EXCESS":
                    float(
                        c.loc[
                            "FINAL_HOLDOUT",
                            "EXCESS_DISTANCE"
                        ]
                    ),

                "HOLD_HOLM_P":
                    float(
                        c.loc[
                            "FINAL_HOLDOUT",
                            "HOLM_P"
                        ]
                    )
            })

    edf.to_csv(
        OUT
        /
        "24_EDGE_CAUSAL_MAP.csv",
        index=False
    )

    pd.DataFrame(
        rep
    ).to_csv(
        OUT
        /
        "24A_EDGE_REPLICATION.csv",
        index=False
    )

    primary = []

    for part,seeds in [
        ("ALL",set(SEED_IDS)),
        ("DEVELOPMENT",DEV),
        ("CONFIRMATION",CONF),
        ("FINAL_HOLDOUT",HOLD)
    ]:
        tests = []

        defs = [
            (
                "EDGE_DELETE_PURE9",
                "family",
                [
                    EDGE_DELETE_CONDITIONS[i]
                    for i
                    in PURE_DELETE_INDEXES
                ]
            ),
            (
                "EDGE_REVERSE_PURE8",
                "family",
                [
                    EDGE_REVERSE_CONDITIONS[i]
                    for i
                    in PURE_REVERSE_INDEXES
                ]
            ),
            (
                "DIRECTION_ALL_COVARIANT_SWITCH",
                "single",
                ["D_ALL_DIRECTION_REVERSED"]
            ),
            (
                "ORDER",
                "single",
                ["O_ORDER_SHUFFLED"]
            ),
            (
                "META_REASSIGNED",
                "single",
                ["M_SWITCH_REASSIGNED"]
            ),
            (
                "PRESERVATION",
                "single",
                ["P_ANCHOR_REASSIGNED"]
            ),
            (
                "TERMINAL",
                "single",
                ["T_TERMINAL_CHANGED"]
            ),
            (
                "DEGREE_MATCHED",
                "single",
                ["R_DEGREE_MATCHED_REWIRE"]
            ),
            (
                "COMPLEXITY_MATCHED",
                "single",
                ["X_COMPLEXITY_MATCHED"]
            ),
            (
                "TOPOLOGY_MATCHED",
                "single",
                ["Y_TOPOLOGY_MATCHED"]
            )
        ]

        for j,(name,kind,conds) in enumerate(defs):
            if kind == "family":
                n,e,p = test_family_vs_surface(
                    ddf,
                    seeds,
                    conds,
                    20000+j+len(primary)
                )
            else:
                n,e,p = test_condition_vs_surface(
                    ddf,
                    seeds,
                    conds[0],
                    20000+j+len(primary)
                )

            tests.append({
                "PARTITION":
                    part,

                "TEST":
                    name,

                "N":
                    n,

                "EXCESS_DISTANCE":
                    e,

                "P":
                    p
            })

        adj = holm_adjust(
            [
                r["P"]
                for r in tests
            ]
        )

        for r,q in zip(tests,adj):
            r["HOLM_P"] = float(q)

            r["PASS_005"] = bool(
                r["EXCESS_DISTANCE"]>0
                and
                q<0.05
            )

            r["PASS_001"] = bool(
                r["EXCESS_DISTANCE"]>0
                and
                q<0.01
            )

            primary.append(r)

    pdf = pd.DataFrame(primary)

    pdf.to_csv(
        OUT
        /
        "09_PRIMARY_CONTRASTS.csv",
        index=False
    )

    pdf[
        pdf.PARTITION!="ALL"
    ].to_csv(
        OUT
        /
        "10_SPLIT_REPLICATION.csv",
        index=False
    )

    surf = []

    for part,seeds in [
        ("ALL",set(SEED_IDS)),
        ("DEVELOPMENT",DEV),
        ("CONFIRMATION",CONF),
        ("FINAL_HOLDOUT",HOLD)
    ]:
        base = nuisance_baseline(
            ddf,
            seeds
        )

        local = []

        for j,c in enumerate([
            "R_DEGREE_MATCHED_REWIRE",
            "X_COMPLEXITY_MATCHED",
            "Y_TOPOLOGY_MATCHED"
        ]):
            x = ddf[
                (
                    ddf.SEED.isin(seeds)
                )
                &
                (
                    ddf.CONDITION==c
                )
            ].set_index(
                "SEED"
            ).MAHALANOBIS

            idx = sorted(
                set(base.index)
                &
                set(x.index)
            )

            diff = np.array([
                x.loc[s]-base.loc[s]
                for s in idx
            ])

            local.append({
                "PARTITION":
                    part,

                "CONTROL":
                    c,

                "N":
                    len(diff),

                "STRUCTURAL_MINUS_SURFACE":
                    float(diff.mean()),

                "P":
                    signflip(
                        diff,
                        MASTER_SEED
                        +
                        30000
                        +
                        j
                        +
                        len(surf)
                    )
            })

        adj = holm_adjust(
            [
                r["P"]
                for r in local
            ]
        )

        for r,q in zip(local,adj):
            r["HOLM_P"] = float(q)

            r["PASS_001"] = bool(
                r["STRUCTURAL_MINUS_SURFACE"]>0
                and
                q<0.01
            )

            surf.append(r)

    pd.DataFrame(
        surf
    ).to_csv(
        OUT
        /
        "12_SURFACE_INVARIANCE_TESTS.csv",
        index=False
    )

    op = []

    opdefs = {
        "A_AUTHENTIC": [
            "ORDER_FIDELITY",
            "K_JACCARD",
            "S_EXACT",
            "S_APPLIED_SCORE",
            "K_INCIDENT_RETENTION",
            "Z_EXACT",
            "Z_FINAL_SINKNESS",
            "Z_SEQUENCE_POSITION"
        ],

        "O_ORDER_SHUFFLED": [
            "ORDER_FIDELITY"
        ],

        "M_SWITCH_REASSIGNED": [
            "S_EXACT",
            "S_APPLIED_SCORE"
        ],

        "M0_SWITCH_NEUTRALIZED": [
            "S_EXACT"
        ],

        "P_ANCHOR_REASSIGNED": [
            "K_JACCARD",
            "K_INCIDENT_RETENTION"
        ],

        "T_TERMINAL_CHANGED": [
            "Z_EXACT",
            "Z_FINAL_SINKNESS",
            "Z_SEQUENCE_POSITION"
        ]
    }

    for part,seeds in [
        ("ALL",set(SEED_IDS)),
        ("DEVELOPMENT",DEV),
        ("CONFIRMATION",CONF),
        ("FINAL_HOLDOUT",HOLD)
    ]:
        for c,features in opdefs.items():
            s = sdf[
                (
                    sdf.SEED.isin(seeds)
                )
                &
                (
                    sdf.CONDITION==c
                )
            ]

            for f in features:
                op.append({
                    "PARTITION":
                        part,

                    "CONDITION":
                        c,

                    "METRIC":
                        f,

                    "N":
                        len(s),

                    "MEAN":
                        float(
                            s[f].mean()
                        ),

                    "SD":
                        float(
                            s[f].std()
                        )
                })

    pd.DataFrame(op).to_csv(
        OUT
        /
        "25_OPERATOR_FIDELITY.csv",
        index=False
    )

    chance = {
        "ORDER_FIDELITY":
            0.5,

        "K_JACCARD":
            5.0/28.0,

        "S_APPLIED_SCORE":
            0.5
            *
            (
                1.0/10.0
                +
                2.0/46.0
            ),

        "Z_EXACT":
            1.0/8.0
    }

    check_conditions = {
        "ORDER_FIDELITY": [
            "A_AUTHENTIC",
            "O_ORDER_SHUFFLED"
        ],

        "K_JACCARD": [
            "A_AUTHENTIC",
            "P_ANCHOR_REASSIGNED"
        ],

        "S_APPLIED_SCORE": [
            "A_AUTHENTIC",
            "M_SWITCH_REASSIGNED"
        ],

        "Z_EXACT": [
            "A_AUTHENTIC",
            "T_TERMINAL_CHANGED"
        ]
    }

    och = []

    for part,seeds in [
        ("DEVELOPMENT",DEV),
        ("CONFIRMATION",CONF),
        ("FINAL_HOLDOUT",HOLD)
    ]:
        local = []

        for metric,conds in check_conditions.items():
            for c in conds:
                vals = (
                    sdf[
                        (
                            sdf.SEED.isin(seeds)
                        )
                        &
                        (
                            sdf.CONDITION==c
                        )
                    ][
                        metric
                    ].to_numpy(float)
                    -
                    chance[metric]
                )

                p = signflip(
                    vals,
                    MASTER_SEED
                    +
                    35000
                    +
                    len(och)
                    +
                    len(local)
                )

                local.append({
                    "PARTITION":
                        part,

                    "METRIC":
                        metric,

                    "CONDITION":
                        c,

                    "N":
                        len(vals),

                    "MEAN":
                        float(
                            vals.mean()
                            +
                            chance[metric]
                        ),

                    "CHANCE":
                        chance[metric],

                    "ABOVE_CHANCE":
                        float(vals.mean()),

                    "P":
                        p
                })

        adj = holm_adjust(
            [
                r["P"]
                for r in local
            ]
        )

        for r,q in zip(local,adj):
            r["HOLM_P"] = float(q)

            r["PASS_005"] = bool(
                r["ABOVE_CHANCE"]>0
                and
                q<0.05
            )

            och.append(r)

    pd.DataFrame(och).to_csv(
        OUT
        /
        "25A_OPERATOR_MANIPULATION_CHECKS.csv",
        index=False
    )

    clrows = classifier_rows(sdf)

    pd.DataFrame(
        clrows
    ).to_csv(
        OUT
        /
        "11_GRAPH_CLASSIFIERS.csv",
        index=False
    )

    nrows = []

    for rec in native_success.values():
        seed = int(
            rec["seed"]
        )

        frame = rec["frame"]

        ne = native_edges(
            rec["parsed"],
            rec["labels"]
        )

        nv = topology_vector(ne)

        candidates = {
            "AUTHENTIC":
                AUTH_EDGES,

            "DEGREE_MATCHED":
                degree_preserving_rewire(seed),

            "COMPLEXITY_MATCHED":
                complexity_matched(seed),

            "TOPOLOGY_MATCHED":
                topology_matched(seed),

            "FORWARD":
                FORWARD_CONTROL,

            "RANDOM":
                random_graph(seed)
        }

        for name,edges in candidates.items():
            nrows.append({
                "SEED":
                    seed,

                "FRAME":
                    frame,

                "REP":
                    int(rec["rep"]),

                "CANDIDATE":
                    name,

                "DISTANCE":
                    float(
                        np.linalg.norm(
                            nv
                            -
                            topology_vector(edges)
                        )
                    )
            })

    ndf = pd.DataFrame(nrows)

    ndf.to_csv(
        OUT
        /
        "27_MODEL_NATIVE_PROXY.csv",
        index=False
    )

    seedframe = ndf.groupby(
        [
            "SEED",
            "FRAME",
            "CANDIDATE"
        ],
        as_index=False
    ).DISTANCE.mean()

    ns = []

    controls = [
        "DEGREE_MATCHED",
        "COMPLEXITY_MATCHED",
        "TOPOLOGY_MATCHED",
        "FORWARD",
        "RANDOM"
    ]

    for scope,frames in (
        [
            (
                "POOLED",
                set(NATIVE_FRAMES)
            )
        ]
        +
        [
            (
                f,
                {f}
            )
            for f
            in NATIVE_FRAMES
        ]
    ):
        sf = seedframe[
            seedframe.FRAME.isin(frames)
        ].groupby(
            [
                "SEED",
                "CANDIDATE"
            ],
            as_index=False
        ).DISTANCE.mean()

        for part,seeds in [
            ("ALL",set(SEED_IDS)),
            ("DEVELOPMENT",DEV),
            ("CONFIRMATION",CONF),
            ("FINAL_HOLDOUT",HOLD)
        ]:
            a = sf[
                (
                    sf.SEED.isin(seeds)
                )
                &
                (
                    sf.CANDIDATE=="AUTHENTIC"
                )
            ].set_index(
                "SEED"
            ).DISTANCE

            local = []

            for j,c in enumerate(controls):
                b = sf[
                    (
                        sf.SEED.isin(seeds)
                    )
                    &
                    (
                        sf.CANDIDATE==c
                    )
                ].set_index(
                    "SEED"
                ).DISTANCE

                idx = sorted(
                    set(a.index)
                    &
                    set(b.index)
                )

                diff = np.array([
                    b.loc[s]
                    -
                    a.loc[s]
                    for s in idx
                ])

                local.append({
                    "SCOPE":
                        scope,

                    "PARTITION":
                        part,

                    "CONTROL":
                        c,

                    "N":
                        len(diff),

                    "AUTH_ADVANTAGE":
                        float(diff.mean()),

                    "P":
                        signflip(
                            diff,
                            MASTER_SEED
                            +
                            40000
                            +
                            j
                            +
                            len(ns)
                        )
                })

            adj = holm_adjust(
                [
                    r["P"]
                    for r in local
                ]
            )

            for r,q in zip(local,adj):
                r["HOLM_P"] = float(q)

                r["PASS"] = bool(
                    r["AUTH_ADVANTAGE"]>0
                    and
                    q<0.05
                )

                ns.append(r)

    nsdf = pd.DataFrame(ns)

    nsdf.to_csv(
        OUT
        /
        "28_MODEL_NATIVE_SUMMARY.csv",
        index=False
    )

    pidx = {
        part:
            pdf[
                pdf.PARTITION==part
            ].set_index(
                "TEST"
            )
        for part
        in [
            "CONFIRMATION",
            "FINAL_HOLDOUT"
        ]
    }

    cdf = pd.DataFrame(clrows)

    hard_auc_conf = float(
        cdf.query(
            "CLASSIFIER=='HARD_CONTROLS' and PARTITION=='CONFIRMATION'"
        ).iloc[0].ROC_AUC
    )

    hard_auc_final = float(
        cdf.query(
            "CLASSIFIER=='HARD_CONTROLS' and PARTITION=='FINAL_HOLDOUT'"
        ).iloc[0].ROC_AUC
    )

    specificity_by_part = {}

    for part in [
        "CONFIRMATION",
        "FINAL_HOLDOUT"
    ]:
        specificity_by_part[part] = all(
            bool(
                pidx[part].loc[
                    x,
                    "PASS_001"
                ]
            )
            for x
            in [
                "DEGREE_MATCHED",
                "COMPLEXITY_MATCHED",
                "TOPOLOGY_MATCHED"
            ]
        )

    surfdf = pd.DataFrame(surf)
    surface_by_part = {}

    for part in [
        "CONFIRMATION",
        "FINAL_HOLDOUT"
    ]:
        ss = surfdf[
            surfdf.PARTITION==part
        ]

        surface_by_part[part] = bool(
            len(ss)==3
            and
            ss.PASS_001.all()
        )

    family_rep = all(
        bool(
            pidx[part].loc[
                "EDGE_DELETE_PURE9",
                "PASS_001"
            ]
            and
            pidx[part].loc[
                "EDGE_REVERSE_PURE8",
                "PASS_001"
            ]
        )
        for part
        in [
            "CONFIRMATION",
            "FINAL_HOLDOUT"
        ]
    )

    primary_strong = bool(
        hard_auc_conf>=0.85
        and
        hard_auc_final>=0.85
        and
        family_rep
        and
        all(
            specificity_by_part.values()
        )
        and
        all(
            surface_by_part.values()
        )
    )

    repdf = pd.DataFrame(rep)

    delete_rep = int(
        repdf[
            (
                repdf.FAMILY=="DELETE"
            )
            &
            (
                ~repdf.EDGE.isin([7])
            )
        ].REPLICATED_CONFIRMATION_AND_HOLDOUT.sum()
    )

    reverse_rep = int(
        repdf[
            (
                repdf.FAMILY=="REVERSE"
            )
            &
            (
                ~repdf.EDGE.isin([7,9])
            )
        ].REPLICATED_CONFIRMATION_AND_HOLDOUT.sum()
    )

    native_pass_by_part = {}

    for part in [
        "CONFIRMATION",
        "FINAL_HOLDOUT"
    ]:
        pp = nsdf[
            (
                nsdf.SCOPE=="POOLED"
            )
            &
            (
                nsdf.PARTITION==part
            )
        ]

        native_pass_by_part[part] = int(
            pp.PASS.sum()
        )

    frame_dirs = 0

    for frame in NATIVE_FRAMES:
        ff = nsdf[
            (
                nsdf.SCOPE==frame
            )
            &
            (
                nsdf.PARTITION=="FINAL_HOLDOUT"
            )
        ]

        if (
            len(ff)
            and
            float(
                ff.AUTH_ADVANTAGE.mean()
            )
            > 0
        ):
            frame_dirs += 1

    native_triangulated = bool(
        native_pass_by_part[
            "CONFIRMATION"
        ]
        >= 3
        and
        native_pass_by_part[
            "FINAL_HOLDOUT"
        ]
        >= 3
        and
        frame_dirs >= 2
    )

    final = pidx[
        "FINAL_HOLDOUT"
    ]

    partial_specificity = any(
        specificity_by_part.values()
    )

    if primary_strong:
        classification = (
            "STRONG_REPLICATED_RELATIONAL_TOPOLOGY_FINGERPRINT"
        )

    elif (
        hard_auc_final >= 0.75
        and
        (
            bool(
                final.loc[
                    "EDGE_DELETE_PURE9",
                    "PASS_005"
                ]
            )
            or
            bool(
                final.loc[
                    "EDGE_REVERSE_PURE8",
                    "PASS_005"
                ]
            )
            or
            partial_specificity
        )
    ):
        classification = (
            "PARTIAL_RELATIONAL_TOPOLOGY_FINGERPRINT"
        )

    else:
        classification = (
            "RELATIONAL_TOPOLOGY_FINGERPRINT_NOT_SUPPORTED"
        )

    summary = {
        "experiment":
            EXPERIMENT,

        "completed_utc":
            utcnow(),

        "model":
            MODEL,

        "classification":
            classification,

        "primary_strong_gate":
            primary_strong,

        "hard_control_auc": {
            "confirmation":
                hard_auc_conf,

            "final_holdout":
                hard_auc_final
        },

        "pure9_delete": {
            "confirmation": {
                "excess":
                    float(
                        pidx[
                            "CONFIRMATION"
                        ].loc[
                            "EDGE_DELETE_PURE9",
                            "EXCESS_DISTANCE"
                        ]
                    ),

                "holm_p":
                    float(
                        pidx[
                            "CONFIRMATION"
                        ].loc[
                            "EDGE_DELETE_PURE9",
                            "HOLM_P"
                        ]
                    )
            },

            "final_holdout": {
                "excess":
                    float(
                        final.loc[
                            "EDGE_DELETE_PURE9",
                            "EXCESS_DISTANCE"
                        ]
                    ),

                "holm_p":
                    float(
                        final.loc[
                            "EDGE_DELETE_PURE9",
                            "HOLM_P"
                        ]
                    )
            }
        },

        "pure8_reverse": {
            "confirmation": {
                "excess":
                    float(
                        pidx[
                            "CONFIRMATION"
                        ].loc[
                            "EDGE_REVERSE_PURE8",
                            "EXCESS_DISTANCE"
                        ]
                    ),

                "holm_p":
                    float(
                        pidx[
                            "CONFIRMATION"
                        ].loc[
                            "EDGE_REVERSE_PURE8",
                            "HOLM_P"
                        ]
                    )
            },

            "final_holdout": {
                "excess":
                    float(
                        final.loc[
                            "EDGE_REVERSE_PURE8",
                            "EXCESS_DISTANCE"
                        ]
                    ),

                "holm_p":
                    float(
                        final.loc[
                            "EDGE_REVERSE_PURE8",
                            "HOLM_P"
                        ]
                    )
            }
        },

        "specificity_all_three":
            specificity_by_part,

        "surface_invariance_all_three":
            surface_by_part,

        "replicated_pure_edges": {
            "delete":
                delete_rep,

            "delete_out_of":
                9,

            "reverse":
                reverse_rep,

            "reverse_out_of":
                8
        },

        "switch_linked_delete_indexes":
            [7],

        "switch_linked_reverse_indexes":
            [7,9],

        "model_native_proxy": {
            "pooled_control_passes":
                native_pass_by_part,

            "frame_positive_direction_count_final":
                frame_dirs,

            "triangulated":
                native_triangulated
        },

        "interpretation_boundary":
            spec()[
                "interpretation_boundary"
            ]
    }

    (
        OUT
        /
        "16_FINAL_SUMMARY.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )

    return summary


def fake_parsed(payload):
    V = list(payload["V"])

    active = [
        e
        for e in payload["E"]
        if int(e["active"]) == 1
    ]

    active_pairs = {
        (e["u"],e["v"])
        for e in active
    }

    plus = []

    for a in V:
        for b in V:
            if (
                a != b
                and
                (a,b) not in active_pairs
                and
                (a,b) not in plus
            ):
                plus.append(
                    (a,b)
                )

                if len(plus)==2:
                    break

        if len(plus)==2:
            break

    S = payload["S"]

    sold = (
        S[0],
        S[1]
    )

    snew = (
        S[2],
        S[3]
    )

    valid = (
        sold in active_pairs
        and
        snew not in active_pairs
        and
        sold != snew
    )

    return {
        "seq":
            list(
                payload["Q"]
            ),

        "plus":
            [
                list(x)
                for x in plus
            ],

        "minus":
            active[0]["id"],

        "keep":
            list(
                payload["K"]
            ),

        "switch":
            list(S)
            if valid
            else [],

        "end":
            payload["Z"],

        "code":
            [
                e["id"]
                for e in active[:4]
            ]
    }


def self_test():
    global OUT, N_PERMUTATIONS

    original = OUT
    original_perm = N_PERMUTATIONS

    N_PERMUTATIONS = 199

    tmp = EXP_ROOT/"_WF1_A_V12_R_SELFTEST_TMP"

    if tmp.exists():
        shutil.rmtree(
            tmp,
            ignore_errors=True
        )

    OUT = tmp

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    try:
        successful = {}
        payload_cache = {}
        meta_cache = {}

        for seed in SEED_IDS:
            for c in CONDITIONS:
                p,pl,m = build_prompt(
                    seed,
                    c
                )

                payload_cache[
                    (
                        seed,
                        c
                    )
                ] = pl

                meta_cache[
                    (
                        seed,
                        c
                    )
                ] = m

                for rep in range(N_REPS):
                    key = f"{seed}|{c}|{rep}"

                    successful[key] = {
                        "cell_key":
                            key,

                        "seed":
                            seed,

                        "condition":
                            c,

                        "rep":
                            rep,

                        "status":
                            "success",

                        "parsed":
                            fake_parsed(pl),

                        "output_tokens":
                            100,

                        "latency_seconds":
                            0.1
                    }

        native_success = {}

        for seed in SEED_IDS:
            for frame in NATIVE_FRAMES:
                prompt,labels = native_prompt(
                    seed,
                    frame
                )

                pairs = []

                for i in range(7):
                    pairs.append([
                        labels[i],
                        labels[i+1]
                    ])

                pairs += [
                    [
                        labels[0],
                        labels[2]
                    ],

                    [
                        labels[2],
                        labels[5]
                    ],

                    [
                        labels[4],
                        labels[7]
                    ]
                ]

                for rep in range(N_REPS):
                    key = f"{seed}|{frame}|{rep}"

                    native_success[key] = {
                        "cell_key":
                            key,

                        "seed":
                            seed,

                        "frame":
                            frame,

                        "rep":
                            rep,

                        "status":
                            "success",

                        "parsed": {
                            "E":
                                pairs
                        },

                        "labels":
                            labels
                    }

        summary = run_analysis(
            successful,
            meta_cache,
            payload_cache,
            native_success
        )

        required = [
            "06_RESPONSE_GRAPH_FEATURES.csv",
            "09_PRIMARY_CONTRASTS.csv",
            "10_SPLIT_REPLICATION.csv",
            "11_GRAPH_CLASSIFIERS.csv",
            "12_SURFACE_INVARIANCE_TESTS.csv",
            "16_FINAL_SUMMARY.json",
            "21A_DEVELOPMENT_PRECISION_MATRIX.csv",
            "24_EDGE_CAUSAL_MAP.csv",
            "24A_EDGE_REPLICATION.csv",
            "25_OPERATOR_FIDELITY.csv",
            "25A_OPERATOR_MANIPULATION_CHECKS.csv",
            "27_MODEL_NATIVE_PROXY.csv",
            "28_MODEL_NATIVE_SUMMARY.csv"
        ]

        missing = [
            x
            for x in required
            if not (
                OUT
                /
                x
            ).exists()
        ]

        if missing:
            raise RuntimeError(
                "self-test missing artifacts: "
                +
                str(missing)
            )

        result = {
            "utc":
                utcnow(),

            "pass":
                True,

            "required_artifacts":
                len(required),

            "synthetic_summary":
                summary
        }

    finally:
        OUT = original
        N_PERMUTATIONS = original_perm

        shutil.rmtree(
            tmp,
            ignore_errors=True
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        OUT
        /
        "00E_ANALYSIS_SELFTEST.json"
    ).write_text(
        json.dumps(
            result,
            indent=2
        ),
        encoding="utf-8"
    )

    return result


def freeze_run_metadata():
    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    runner = Path(
        __file__
    ).resolve()

    rsha = sha256_file(
        runner
    )

    (
        OUT
        /
        "00_FROZEN_RUNNER.py"
    ).write_bytes(
        runner.read_bytes()
    )

    (
        OUT
        /
        "00_RUNNER_SHA256.txt"
    ).write_text(
        rsha
        +
        "\n",
        encoding="utf-8"
    )

    frozen = spec()

    (
        OUT
        /
        "03_FROZEN_SPECIFICATION.json"
    ).write_text(
        json.dumps(
            frozen,
            indent=2
        ),
        encoding="utf-8"
    )

    (
        OUT
        /
        "04_FROZEN_SPECIFICATION_SHA256.txt"
    ).write_text(
        sha256_text(
            canonical_json(
                frozen
            )
        )
        +
        "\n",
        encoding="utf-8"
    )

    return rsha


def build_caches():
    payload_cache = {}
    meta_cache = {}
    prompt_cache = {}

    for seed in SEED_IDS:
        for c in CONDITIONS:
            p,pl,m = build_prompt(
                seed,
                c
            )

            prompt_cache[
                (
                    seed,
                    c
                )
            ] = p

            payload_cache[
                (
                    seed,
                    c
                )
            ] = pl

            meta_cache[
                (
                    seed,
                    c
                )
            ] = m

    return (
        prompt_cache,
        payload_cache,
        meta_cache
    )


def analyze_existing():
    raw = OUT/"01_RAW_RESPONSES.jsonl"
    nraw = OUT/"26_NATIVE_RAW.jsonl"

    successful = load_unique(raw)
    nsuccess = load_unique(nraw)

    if (
        len(successful) != EXPECTED_MAIN_CELLS
        or
        len(nsuccess) != EXPECTED_NATIVE_CELLS
    ):
        raise RuntimeError(
            f"analysis-only requires complete data: "
            f"main={len(successful)}/{EXPECTED_MAIN_CELLS}, "
            f"native={len(nsuccess)}/{EXPECTED_NATIVE_CELLS}"
        )

    _,payload_cache,meta_cache = build_caches()

    return run_analysis(
        successful,
        meta_cache,
        payload_cache,
        nsuccess
    )


def artifact_manifest():
    arts = []

    for p in sorted(
        x
        for x in OUT.iterdir()
        if (
            x.is_file()
            and
            x.name != "30_FINAL_ARTIFACT_SHA256.csv"
        )
    ):
        arts.append({
            "FILE":
                p.name,

            "SHA256":
                sha256_file(p),

            "BYTES":
                p.stat().st_size
        })

    pd.DataFrame(
        arts
    ).to_csv(
        OUT
        /
        "30_FINAL_ARTIFACT_SHA256.csv",
        index=False
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--preflight-only",
        action="store_true"
    )

    ap.add_argument(
        "--self-test",
        action="store_true"
    )

    ap.add_argument(
        "--smoke-only",
        action="store_true"
    )

    ap.add_argument(
        "--analysis-only",
        action="store_true"
    )

    args = ap.parse_args()

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    if args.preflight_only:
        print(
            "WF1-A V12-R zero-cost preflight starting...",
            flush=True
        )

        verify_sources()
        snapshot_v11_sources()

        preflight(
            write_files=True
        )

        freeze_run_metadata()

        print(
            f"WF1-A PRE-FLIGHT PASS "
            f"main_conditions={len(CONDITIONS)} "
            f"main_cells={EXPECTED_MAIN_CELLS} "
            f"native_cells={EXPECTED_NATIVE_CELLS}",
            flush=True
        )

        return

    if args.self_test:
        print(
            "WF1-A V12-R full downstream analysis self-test...",
            flush=True
        )

        preflight(
            write_files=False
        )

        r = self_test()

        print(
            json.dumps(
                r,
                indent=2
            ),
            flush=True
        )

        return

    freeze_run_metadata()
    verify_sources()
    snapshot_v11_sources()

    preflight(
        write_files=True
    )

    if args.analysis_only:
        try:
            s = analyze_existing()

            artifact_manifest()

            print(
                json.dumps(
                    s,
                    indent=2
                ),
                flush=True
            )

            return

        except Exception as e:
            (
                OUT
                /
                "18_ANALYSIS_ERROR.json"
            ).write_text(
                json.dumps({
                    "utc":
                        utcnow(),

                    "type":
                        type(e).__name__,

                    "error":
                        str(e),

                    "traceback":
                        traceback.format_exc()
                },
                indent=2),
                encoding="utf-8"
            )

            raise

    if not os.getenv(
        "OPENAI_API_KEY"
    ):
        raise RuntimeError(
            "OPENAI_API_KEY missing"
        )

    prompt_cache,payload_cache,meta_cache = build_caches()

    if args.smoke_only:
        p,pl,_ = build_prompt(
            999999,
            "A_AUTHENTIC"
        )

        r = call_api(
            p,
            pl
        )

        (
            OUT
            /
            "00B_SMOKE_TEST.json"
        ).write_text(
            json.dumps(
                r,
                indent=2,
                default=str
            ),
            encoding="utf-8"
        )

        if not r["ok"]:
            raise RuntimeError(
                "main smoke test failed"
            )

        nr = call_native(
            999999,
            "N1_COHERENT"
        )

        (
            OUT
            /
            "00C_NATIVE_SMOKE_TEST.json"
        ).write_text(
            json.dumps(
                nr,
                indent=2,
                default=str
            ),
            encoding="utf-8"
        )

        if not nr["ok"]:
            raise RuntimeError(
                "native smoke test failed"
            )

        print(
            "SMOKE PASS",
            flush=True
        )

        return

    raw = OUT/"01_RAW_RESPONSES.jsonl"
    err = OUT/"02_ERRORS.jsonl"
    api = OUT/"17_RAW_API_OBJECTS.jsonl.gz"

    successful = load_unique(raw)

    print(
        f"Existing main cells: {len(successful)}/{EXPECTED_MAIN_CELLS}",
        flush=True
    )

    with (
        raw.open(
            "a",
            encoding="utf-8"
        ) as rf,
        err.open(
            "a",
            encoding="utf-8"
        ) as ef,
        gzip.open(
            api,
            "at",
            encoding="utf-8"
        ) as af
    ):
        for seed in SEED_IDS:
            for c in CONDITIONS:
                for rep in range(N_REPS):
                    key = f"{seed}|{c}|{rep}"

                    if key in successful:
                        continue

                    r = call_api(
                        prompt_cache[
                            (
                                seed,
                                c
                            )
                        ],
                        payload_cache[
                            (
                                seed,
                                c
                            )
                        ]
                    )

                    if not r["ok"]:
                        ef.write(
                            json.dumps({
                                "cell_key":
                                    key,

                                "seed":
                                    seed,

                                "condition":
                                    c,

                                "rep":
                                    rep,

                                "failures":
                                    r["failures"]
                            })
                            +
                            "\n"
                        )

                        ef.flush()

                        continue

                    af.write(
                        json.dumps({
                            "cell_key":
                                key,

                            "raw_response":
                                r["raw"],

                            "attempt_failures":
                                r["attempt_failures"]
                        },
                        ensure_ascii=False)
                        +
                        "\n"
                    )

                    af.flush()

                    rec = {
                        "cell_key":
                            key,

                        "seed":
                            seed,

                        "condition":
                            c,

                        "rep":
                            rep,

                        "status":
                            "success",

                        "attempt":
                            r["attempt"],

                        "attempt_failures":
                            r["attempt_failures"],

                        "utc":
                            utcnow(),

                        "prompt_sha256":
                            sha256_text(
                                prompt_cache[
                                    (
                                        seed,
                                        c
                                    )
                                ]
                            ),

                        "text_sha256":
                            sha256_text(
                                r["text"]
                            ),

                        "text":
                            r["text"],

                        "parsed":
                            r["parsed"],

                        "response_id":
                            r["response_id"],

                        "response_status":
                            r["response_status"],

                        "latency_seconds":
                            r["latency"],

                        "input_tokens":
                            r["input_tokens"],

                        "output_tokens":
                            r["output_tokens"]
                    }

                    rf.write(
                        json.dumps(
                            rec,
                            ensure_ascii=False
                        )
                        +
                        "\n"
                    )

                    rf.flush()

                    successful[key] = rec

                    if len(successful)%25 == 0:
                        print(
                            f"Main API progress: "
                            f"{len(successful)}/{EXPECTED_MAIN_CELLS}",
                            flush=True
                        )

    expected = {
        f"{s}|{c}|{r}"
        for s in SEED_IDS
        for c in CONDITIONS
        for r in range(N_REPS)
    }

    missing = sorted(
        expected
        -
        set(successful)
    )

    pd.DataFrame({
        "CELL_KEY":
            missing
    }).to_csv(
        OUT
        /
        "15_MISSING_CELLS.csv",
        index=False
    )

    nraw = OUT/"26_NATIVE_RAW.jsonl"
    nerr = OUT/"26_NATIVE_ERRORS.jsonl"
    napi = OUT/"26_NATIVE_RAW_API_OBJECTS.jsonl.gz"

    nsuccess = load_unique(nraw)

    print(
        f"Existing native cells: "
        f"{len(nsuccess)}/{EXPECTED_NATIVE_CELLS}",
        flush=True
    )

    with (
        nraw.open(
            "a",
            encoding="utf-8"
        ) as rf,
        nerr.open(
            "a",
            encoding="utf-8"
        ) as ef,
        gzip.open(
            napi,
            "at",
            encoding="utf-8"
        ) as af
    ):
        for seed in SEED_IDS:
            for frame in NATIVE_FRAMES:
                for rep in range(N_REPS):
                    key = f"{seed}|{frame}|{rep}"

                    if key in nsuccess:
                        continue

                    r = call_native(
                        seed,
                        frame
                    )

                    if not r["ok"]:
                        ef.write(
                            json.dumps({
                                "cell_key":
                                    key,

                                "seed":
                                    seed,

                                "frame":
                                    frame,

                                "rep":
                                    rep,

                                "failures":
                                    r["failures"]
                            })
                            +
                            "\n"
                        )

                        ef.flush()

                        continue

                    af.write(
                        json.dumps({
                            "cell_key":
                                key,

                            "raw_response":
                                r["raw"],

                            "attempt_failures":
                                r["attempt_failures"]
                        },
                        ensure_ascii=False)
                        +
                        "\n"
                    )

                    af.flush()

                    rec = {
                        "cell_key":
                            key,

                        "seed":
                            seed,

                        "frame":
                            frame,

                        "rep":
                            rep,

                        "status":
                            "success",

                        "attempt":
                            r["attempt"],

                        "attempt_failures":
                            r["attempt_failures"],

                        "utc":
                            utcnow(),

                        "prompt_sha256":
                            sha256_text(
                                r["prompt"]
                            ),

                        "text_sha256":
                            sha256_text(
                                r["text"]
                            ),

                        "text":
                            r["text"],

                        "parsed":
                            r["parsed"],

                        "labels":
                            r["labels"],

                        "latency_seconds":
                            r["latency"],

                        "input_tokens":
                            r["input_tokens"],

                        "output_tokens":
                            r["output_tokens"]
                    }

                    rf.write(
                        json.dumps(
                            rec,
                            ensure_ascii=False
                        )
                        +
                        "\n"
                    )

                    rf.flush()

                    nsuccess[key] = rec

                    if len(nsuccess)%25 == 0:
                        print(
                            f"Native API progress: "
                            f"{len(nsuccess)}/{EXPECTED_NATIVE_CELLS}",
                            flush=True
                        )

    nexpected = {
        f"{s}|{f}|{r}"
        for s in SEED_IDS
        for f in NATIVE_FRAMES
        for r in range(N_REPS)
    }

    nmissing = sorted(
        nexpected
        -
        set(nsuccess)
    )

    pd.DataFrame({
        "CELL_KEY":
            nmissing
    }).to_csv(
        OUT
        /
        "26_NATIVE_MISSING_CELLS.csv",
        index=False
    )

    completeness = {
        "expected_main_cells":
            EXPECTED_MAIN_CELLS,

        "successful_main_cells":
            len(successful),

        "missing_main_cells":
            len(missing),

        "expected_native_cells":
            EXPECTED_NATIVE_CELLS,

        "successful_native_cells":
            len(nsuccess),

        "missing_native_cells":
            len(nmissing),

        "scientific_completeness":
            not missing
            and
            not nmissing
    }

    (
        OUT
        /
        "00A_VALID_TEXT_COMPLETENESS.json"
    ).write_text(
        json.dumps(
            completeness,
            indent=2
        ),
        encoding="utf-8"
    )

    if missing or nmissing:
        raise RuntimeError(
            f"INCOMPLETE "
            f"main_missing={len(missing)} "
            f"native_missing={len(nmissing)}. "
            "Rerun exact frozen runner."
        )

    try:
        print(
            "Completeness gate passed. "
            "Running frozen analysis...",
            flush=True
        )

        summary = run_analysis(
            successful,
            meta_cache,
            payload_cache,
            nsuccess
        )

        env = {
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
            /
            "20_RUN_ENVIRONMENT.json"
        ).write_text(
            json.dumps(
                env,
                indent=2
            ),
            encoding="utf-8"
        )

        artifact_manifest()

        print(
            json.dumps(
                summary,
                indent=2
            ),
            flush=True
        )

        print(
            "RESULTS:",
            OUT,
            flush=True
        )

    except Exception as e:
        (
            OUT
            /
            "18_ANALYSIS_ERROR.json"
        ).write_text(
            json.dumps({
                "utc":
                    utcnow(),

                "type":
                    type(e).__name__,

                "category":
                    classify_failure(e),

                "error":
                    str(e),

                "traceback":
                    traceback.format_exc()
            },
            indent=2),
            encoding="utf-8"
        )

        raise


if __name__=="__main__":
    main()

