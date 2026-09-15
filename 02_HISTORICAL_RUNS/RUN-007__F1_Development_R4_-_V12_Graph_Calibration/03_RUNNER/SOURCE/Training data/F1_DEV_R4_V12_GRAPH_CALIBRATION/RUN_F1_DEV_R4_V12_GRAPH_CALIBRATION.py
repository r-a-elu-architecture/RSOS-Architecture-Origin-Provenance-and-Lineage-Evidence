from pathlib import Path
import json, hashlib, math, sys, platform
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx
import sklearn
import scipy
import joblib

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(r"C:\RSOS\Training data\F1_DEV_R4_V12_GRAPH_CALIBRATION")
ROOT.mkdir(parents=True, exist_ok=True)

DIRS = {
    "manifest": ROOT / "00_MANIFEST",
    "unit": ROOT / "01_EXTRACTOR_UNIT_TESTS",
    "auth": ROOT / "02_AUTHENTIC_EDGE_PERTURBATIONS",
    "generic": ROOT / "03_CROSS_TOPOLOGY_GENERALIZATION",
    "family": ROOT / "04_LEAVE_FAMILY_OUT",
    "ledger": ROOT / "05_NO_SILENT_OMISSION",
    "models": ROOT / "06_FROZEN_MODELS",
}
for p in DIRS.values():
    p.mkdir(parents=True, exist_ok=True)

COMPLETE = ROOT / "F1_DEV_R4_COMPLETE.marker"
if COMPLETE.exists():
    raise SystemExit(f"R4 already completed and will not be overwritten:\n{ROOT}")

SEED = 120918
rng = np.random.default_rng(SEED)
VERSION = "F1_DEV_R4_V12_GRAPH_CALIBRATION"

AUTH_EDGES = [
    (0,1),(1,2),(2,3),(3,4),(4,5),
    (5,6),(6,7),(3,1),(5,2),(6,3)
]

FEATURE_NAMES = [
    "RECIPROCITY",
    "SCC_COUNT",
    "LARGEST_SCC_FRAC",
    "TRANSITIVITY",
    "IN_DEGREE_SD",
    "OUT_DEGREE_SD",
    "SPECTRAL_RADIUS",
    "TWO_CYCLES",
    "THREE_CYCLES",
    "REACHABILITY_FRAC",
    "MEAN_REACHABLE_PATH",
]

def utcnow():
    return datetime.now(timezone.utc).isoformat()

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def canonical_edges(edges):
    return tuple(sorted((int(a),int(b)) for a,b in edges))

def valid_graph(edges):
    e = set(canonical_edges(edges))
    return len(e) == len(edges) and all(a != b for a,b in e) and all(0 <= a < 8 and 0 <= b < 8 for a,b in e)

def graph_from(edges):
    G = nx.DiGraph()
    G.add_nodes_from(range(8))
    G.add_edges_from(edges)
    return G

def count_two_cycles(G):
    return sum(1 for a,b in G.edges() if a < b and G.has_edge(b,a))

def count_three_cycles(G):
    # networkx.simple_cycles already emits each directed simple cycle once
    return sum(1 for cyc in nx.simple_cycles(G, length_bound=3) if len(cyc) == 3)

def graph_features(edges, rounding=12):
    G = graph_from(edges)
    A = nx.to_numpy_array(G, nodelist=range(8), dtype=float)

    eig = np.linalg.eigvals(A)
    scc = list(nx.strongly_connected_components(G))

    reciprocal_edges = sum(1 for a,b in G.edges() if G.has_edge(b,a))
    reciprocity = reciprocal_edges / max(1, G.number_of_edges())

    reach = []
    paths = []
    for a in G.nodes():
        for b in G.nodes():
            if a == b:
                continue
            if nx.has_path(G,a,b):
                reach.append(1)
                paths.append(nx.shortest_path_length(G,a,b))
            else:
                reach.append(0)

    vals = [
        reciprocity,
        len(scc),
        max(map(len,scc))/8.0,
        nx.transitivity(G.to_undirected()),
        float(np.std([d for _,d in G.in_degree()])),
        float(np.std([d for _,d in G.out_degree()])),
        float(np.max(np.abs(eig))),
        count_two_cycles(G),
        count_three_cycles(G),
        float(np.mean(reach)),
        float(np.mean(paths)) if paths else 0.0,
    ]
    return np.round(np.asarray(vals,dtype=float), rounding)

def relabel_edges(edges, perm):
    mp = {i:int(perm[i]) for i in range(8)}
    return [(mp[a],mp[b]) for a,b in edges]

def random_nonedge(edges, rng, forbidden=None):
    e = set(canonical_edges(edges))
    cand = [(a,b) for a in range(8) for b in range(8)
            if a != b and (a,b) not in e]
    if forbidden:
        cand = [x for x in cand if x not in forbidden]
    return cand[int(rng.integers(0,len(cand)))]

def reverse_edge(edges, idx):
    e = list(canonical_edges(edges))
    a,b = e[idx]
    rev = (b,a)
    if rev in e:
        return None
    e[idx] = rev
    return canonical_edges(e)

def delete_add(edges, idx, rng):
    e = list(canonical_edges(edges))
    removed = e.pop(idx)
    added = random_nonedge(e, rng, forbidden={removed})
    e.append(added)
    return canonical_edges(e), removed, added

def degree_preserving_swap(edges, rng, max_tries=1000):
    e = set(canonical_edges(edges))
    el = list(e)
    for _ in range(max_tries):
        i,j = rng.choice(len(el),2,replace=False)
        a,b = el[i]
        c,d = el[j]
        if len({a,b,c,d}) < 4:
            continue
        x = (a,d)
        y = (c,b)
        if a == d or c == b or x in e or y in e:
            continue
        new = set(e)
        new.remove((a,b))
        new.remove((c,d))
        new.add(x)
        new.add(y)
        if len(new) == len(e):
            return canonical_edges(new)
    return None

def random_base_graph(rng):
    all_edges = [(a,b) for a in range(8) for b in range(8) if a != b]
    # Prefer nontrivial, reasonably connected 10-edge graphs.
    for _ in range(10000):
        idx = rng.choice(len(all_edges),10,replace=False)
        e = canonical_edges([all_edges[i] for i in idx])
        G = graph_from(e)
        if nx.number_weakly_connected_components(G) <= 2 and len(list(nx.strongly_connected_components(G))) <= 6:
            return e
    raise RuntimeError("Could not generate base graph.")

def feature_delta(base_edges, transformed_edges):
    return np.abs(graph_features(base_edges) - graph_features(transformed_edges))

def topology_id(edges):
    # Weisfeiler-Lehman hash for grouping only; canonical edge tuple also preserved.
    G = graph_from(edges)
    return nx.weisfeiler_lehman_graph_hash(G, iterations=5)

print("="*78)
print("F1-DEV R4 — CORRECTED V12 GRAPH CALIBRATION")
print("="*78)

# ----------------------------------------------------------------
# 1. EXTRACTOR UNIT TESTS
# ----------------------------------------------------------------

unit_rows = []

auth_feat = graph_features(AUTH_EDGES)

# Isomorphic invariance: exact after rounding.
max_iso_delta = 0.0
iso_failures = 0
for i in range(1000):
    perm = rng.permutation(8)
    rel = relabel_edges(AUTH_EDGES,perm)
    d = float(np.max(np.abs(auth_feat - graph_features(rel))))
    max_iso_delta = max(max_iso_delta,d)
    if d != 0.0:
        iso_failures += 1

unit_rows.append({
    "TEST":"ISOMORPHIC_RELABEL_INVARIANCE",
    "N":1000,
    "VALUE":max_iso_delta,
    "FAILURES":iso_failures,
    "STATUS":"PASS" if iso_failures == 0 else "FAIL",
    "NOTE":"Rounded invariant feature vector must be exactly identical under node relabel."
})

# Explicit directed triangle unit test.
tri = [(0,1),(1,2),(2,0)]
tri_G = nx.DiGraph()
tri_G.add_nodes_from(range(3))
tri_G.add_edges_from(tri)
tri_count = sum(1 for cyc in nx.simple_cycles(tri_G, length_bound=3) if len(cyc)==3)

unit_rows.append({
    "TEST":"THREE_CYCLE_COUNT_EXACT",
    "N":1,
    "VALUE":tri_count,
    "FAILURES":0 if tri_count == 1 else 1,
    "STATUS":"PASS" if tri_count == 1 else "FAIL",
    "NOTE":"No divide-by-three correction is applied."
})

# Edge count is intentionally excluded from FEATURE_NAMES.
unit_rows.append({
    "TEST":"EDGE_COUNT_EXCLUDED_FROM_CLASSIFIER_FEATURES",
    "N":1,
    "VALUE":0,
    "FAILURES":0 if "EDGE_COUNT" not in FEATURE_NAMES else 1,
    "STATUS":"PASS" if "EDGE_COUNT" not in FEATURE_NAMES else "FAIL",
    "NOTE":"Prevents deletion-like conditions from being classified by trivial raw edge count."
})

unit_df = pd.DataFrame(unit_rows)
unit_df.to_csv(DIRS["unit"]/ "EXTRACTOR_UNIT_TESTS.csv",index=False)

if not (unit_df["STATUS"] == "PASS").all():
    raise RuntimeError("Extractor unit test failure.")

# ----------------------------------------------------------------
# 2. AUTHENTIC V12 EDGE PERTURBATIONS
# ----------------------------------------------------------------

auth_rows = []
auth = canonical_edges(AUTH_EDGES)
auth_vec = graph_features(auth)

for edge_idx in range(len(auth)):
    edge = auth[edge_idx]

    # Reverse if valid.
    rev = reverse_edge(auth,edge_idx)
    if rev is not None:
        delta = feature_delta(auth,rev)
        auth_rows.append({
            "EDGE_INDEX":edge_idx,
            "EDGE":str(edge),
            "FAMILY":"REVERSE",
            "EDGE_COUNT_MATCHED":True,
            "L1_DELTA":float(delta.sum()),
            "L2_DELTA":float(np.linalg.norm(delta)),
            "N_CHANGED_FEATURES":int((delta > 0).sum()),
            "DETECTED":bool(np.any(delta > 0)),
        })

    # 50 compensated delete-add variants per edge; same edge count.
    for rep in range(50):
        da,removed,added = delete_add(auth,edge_idx,rng)
        delta = feature_delta(auth,da)
        auth_rows.append({
            "EDGE_INDEX":edge_idx,
            "EDGE":str(edge),
            "FAMILY":"DELETE_ADD_COMPENSATED",
            "EDGE_COUNT_MATCHED":True,
            "L1_DELTA":float(delta.sum()),
            "L2_DELTA":float(np.linalg.norm(delta)),
            "N_CHANGED_FEATURES":int((delta > 0).sum()),
            "DETECTED":bool(np.any(delta > 0)),
        })

# Degree-preserving rewires.
for rep in range(500):
    rw = degree_preserving_swap(auth,rng)
    if rw is None:
        continue
    delta = feature_delta(auth,rw)
    auth_rows.append({
        "EDGE_INDEX":-1,
        "EDGE":"NA",
        "FAMILY":"DEGREE_PRESERVING_REWIRE",
        "EDGE_COUNT_MATCHED":True,
        "L1_DELTA":float(delta.sum()),
        "L2_DELTA":float(np.linalg.norm(delta)),
        "N_CHANGED_FEATURES":int((delta > 0).sum()),
        "DETECTED":bool(np.any(delta > 0)),
    })

auth_df = pd.DataFrame(auth_rows)
auth_df.to_csv(DIRS["auth"]/ "AUTHENTIC_V12_EDGE_PERTURBATIONS.csv",index=False)

auth_summary = (
    auth_df.groupby("FAMILY",dropna=False)
    .agg(
        N=("DETECTED","size"),
        DETECTION_RATE=("DETECTED","mean"),
        MEDIAN_L1=("L1_DELTA","median"),
        MEDIAN_L2=("L2_DELTA","median"),
        MEDIAN_CHANGED_FEATURES=("N_CHANGED_FEATURES","median"),
    )
    .reset_index()
)
auth_summary.to_csv(DIRS["auth"]/ "AUTHENTIC_V12_PERTURBATION_SUMMARY.csv",index=False)

# Per-edge summary prevents aggregate averages from hiding undetected edges.
per_edge = (
    auth_df[auth_df["EDGE_INDEX"] >= 0]
    .groupby(["FAMILY","EDGE_INDEX","EDGE"])
    .agg(
        N=("DETECTED","size"),
        DETECTION_RATE=("DETECTED","mean"),
        MIN_L1=("L1_DELTA","min"),
        MEDIAN_L1=("L1_DELTA","median"),
    )
    .reset_index()
)
per_edge.to_csv(DIRS["auth"]/ "AUTHENTIC_V12_PER_EDGE_DETECTION.csv",index=False)

# ----------------------------------------------------------------
# 3. GENERIC CROSS-TOPOLOGY GENERALIZATION
#
# Split by BASE_GRAPH_ID. Test base topologies never occur in training.
# Positive = isomorphic relabel (unchanged topology)
# Negative = same-edge-count perturbation.
# Input to classifier = feature-delta only.
# ----------------------------------------------------------------

rows = []
N_BASES = 1200

for base_id in range(N_BASES):
    base = random_base_graph(rng)
    base_hash = hashlib.sha256(repr(base).encode()).hexdigest()[:20]

    # Positive: isomorphic relabel. Feature delta should be exactly zero.
    for rep in range(2):
        rel = relabel_edges(base,rng.permutation(8))
        d = feature_delta(base,rel)
        rows.append({
            "BASE_GRAPH_ID":base_hash,
            "FAMILY":"ISOMORPHIC",
            "LABEL":1,
            **{f"D_{name}":float(v) for name,v in zip(FEATURE_NAMES,d)}
        })

    # Reverse one valid edge.
    valid_rev = []
    for idx in range(len(base)):
        rr = reverse_edge(base,idx)
        if rr is not None:
            valid_rev.append(rr)
    if valid_rev:
        rr = valid_rev[int(rng.integers(0,len(valid_rev)))]
        d = feature_delta(base,rr)
        rows.append({
            "BASE_GRAPH_ID":base_hash,
            "FAMILY":"REVERSE",
            "LABEL":0,
            **{f"D_{name}":float(v) for name,v in zip(FEATURE_NAMES,d)}
        })

    # Compensated delete-add.
    idx = int(rng.integers(0,len(base)))
    da,_,_ = delete_add(base,idx,rng)
    d = feature_delta(base,da)
    rows.append({
        "BASE_GRAPH_ID":base_hash,
        "FAMILY":"DELETE_ADD_COMPENSATED",
        "LABEL":0,
        **{f"D_{name}":float(v) for name,v in zip(FEATURE_NAMES,d)}
    })

    # Degree-preserving rewire.
    rw = degree_preserving_swap(base,rng)
    if rw is not None:
        d = feature_delta(base,rw)
        rows.append({
            "BASE_GRAPH_ID":base_hash,
            "FAMILY":"DEGREE_PRESERVING_REWIRE",
            "LABEL":0,
            **{f"D_{name}":float(v) for name,v in zip(FEATURE_NAMES,d)}
        })

gen = pd.DataFrame(rows)
delta_cols = [c for c in gen.columns if c.startswith("D_")]

# Round all classifier inputs again defensively.
gen[delta_cols] = gen[delta_cols].round(12)
gen.to_parquet(DIRS["generic"]/ "CROSS_TOPOLOGY_DATA.parquet",index=False,compression="zstd")

gss = GroupShuffleSplit(n_splits=1,test_size=.30,random_state=SEED)
tr_idx,te_idx = next(gss.split(gen[delta_cols],gen["LABEL"],groups=gen["BASE_GRAPH_ID"]))

clf = ExtraTreesClassifier(
    n_estimators=400,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1,
)
clf.fit(gen.iloc[tr_idx][delta_cols],gen.iloc[tr_idx]["LABEL"])
prob = clf.predict_proba(gen.iloc[te_idx][delta_cols])[:,1]
auc = float(roc_auc_score(gen.iloc[te_idx]["LABEL"],prob))

# Ensure base topology separation is exact.
train_bases = set(gen.iloc[tr_idx]["BASE_GRAPH_ID"])
test_bases  = set(gen.iloc[te_idx]["BASE_GRAPH_ID"])
base_overlap = len(train_bases & test_bases)

pd.DataFrame([{
    "TEST":"UNSEEN_BASE_TOPOLOGY_GENERALIZATION",
    "N_BASE_GRAPHS":N_BASES,
    "TRAIN_BASES":len(train_bases),
    "TEST_BASES":len(test_bases),
    "BASE_GRAPH_OVERLAP":base_overlap,
    "AUC":auc,
    "EDGE_COUNT_FEATURE_USED":False,
    "INTERPRETATION":"EXTRACTOR CALIBRATION ONLY; NOT RSOS EVIDENCE",
}]).to_csv(DIRS["generic"]/ "CROSS_TOPOLOGY_GENERALIZATION_RESULT.csv",index=False)

joblib.dump(clf,DIRS["models"]/ "CROSS_TOPOLOGY_CLASSIFIER.joblib",compress=3)

# ----------------------------------------------------------------
# 4. LEAVE-PERTURBATION-FAMILY-OUT
#
# Held-out negative family is never shown in training.
# Base graph split is also enforced.
# ----------------------------------------------------------------

family_rows = []
negative_families = ["REVERSE","DELETE_ADD_COMPENSATED","DEGREE_PRESERVING_REWIRE"]

for held_family in negative_families:
    # Use disjoint base IDs: deterministic hash partition.
    base_ids = sorted(gen["BASE_GRAPH_ID"].unique())
    test_base = {
        b for b in base_ids
        if int(hashlib.sha256(b.encode()).hexdigest()[:8],16) % 5 == 0
    }
    train_base = set(base_ids) - test_base

    train_mask = (
        gen["BASE_GRAPH_ID"].isin(train_base)
        & gen["FAMILY"].ne(held_family)
    )
    # Need positives + other negative families in training.
    test_mask = (
        gen["BASE_GRAPH_ID"].isin(test_base)
        & gen["FAMILY"].isin(["ISOMORPHIC",held_family])
    )

    tr = gen.loc[train_mask]
    te = gen.loc[test_mask]

    if tr["LABEL"].nunique() < 2 or te["LABEL"].nunique() < 2:
        family_rows.append({
            "HELD_OUT_FAMILY":held_family,
            "AUC":np.nan,
            "TRAIN_BASES":len(train_base),
            "TEST_BASES":len(test_base),
            "STATUS":"UNTESTABLE",
        })
        continue

    c = ExtraTreesClassifier(
        n_estimators=400,min_samples_leaf=5,max_features="sqrt",
        class_weight="balanced",random_state=SEED+len(family_rows)+1,n_jobs=-1
    )
    c.fit(tr[delta_cols],tr["LABEL"])
    p = c.predict_proba(te[delta_cols])[:,1]
    fam_auc = float(roc_auc_score(te["LABEL"],p))

    family_rows.append({
        "HELD_OUT_FAMILY":held_family,
        "AUC":fam_auc,
        "TRAIN_BASES":len(train_base),
        "TEST_BASES":len(test_base),
        "BASE_OVERLAP":0,
        "STATUS":"MEASURED",
    })

pd.DataFrame(family_rows).to_csv(
    DIRS["family"]/ "LEAVE_PERTURBATION_FAMILY_OUT.csv",index=False
)

# ----------------------------------------------------------------
# 5. NO-SILENT-OMISSION LEDGER
# ----------------------------------------------------------------

ledger = pd.DataFrame([
    ["ISOMORPHIC_RELABEL_INVARIANCE","RUN","Exact rounded equality across 1000 relabelings"],
    ["THREE_CYCLE_COUNT_BUG","FIXED_AND_TESTED","NetworkX simple_cycles counted once; no divide-by-three"],
    ["TRIVIAL_EDGE_COUNT_DISCRIMINATION","REMOVED","EDGE_COUNT excluded; delete perturbation is compensated by added nonedge"],
    ["FLOATING_POINT_RELABEL_ARTIFACT","CONTROLLED","All graph invariants rounded to 12 decimals before comparisons/classification"],
    ["BASE_TOPOLOGY_LEAKAGE","CONTROLLED","Group split by base graph ID; test bases absent from train"],
    ["LESION_FAMILY_MEMORIZATION","TESTED","Leave-perturbation-family-out evaluation"],
    ["AUTHENTIC_PER_EDGE_SENSITIVITY","RUN","Each authentic edge examined separately"],
    ["V12_SEQUENCE_ORDER_COMPONENT","UNTESTED_IN_R4","Graph calibration does not substitute for sequence/order operator test"],
    ["V12_META_SWITCH_COMPONENT","UNTESTED_IN_R4","Requires dedicated Q/K/S/Z operator calibration"],
    ["V12_PRESERVATION_COMPONENT","UNTESTED_IN_R4","Requires dedicated Q/K/S/Z operator calibration"],
    ["V12_TERMINAL_COMPONENT","UNTESTED_IN_R4","Requires dedicated Q/K/S/Z operator calibration"],
    ["RSOS_REAL_CORPUS_MAPPING","UNTESTED","No positive or negative inference permitted"],
    ["LMARENA_RSOS_MAPPING","UNTESTED","No positive or negative inference permitted"],
    ["F2_EXTERNAL_00086","RESERVED_NOT_READ","No population-rarity inference permitted"],
],columns=["MEASUREMENT","STATUS","DETAIL"])
ledger.to_csv(DIRS["ledger"]/ "NO_SILENT_OMISSION_LEDGER.csv",index=False)

# ----------------------------------------------------------------
# 6. SUMMARY + PROVENANCE
# ----------------------------------------------------------------

runner = Path(__file__).resolve()
runner_hash = sha256_file(runner)

env = {
    "python":sys.version,
    "platform":platform.platform(),
    "numpy":np.__version__,
    "pandas":pd.__version__,
    "networkx":nx.__version__,
    "sklearn":sklearn.__version__,
    "scipy":scipy.__version__,
    "joblib":joblib.__version__,
}
(DIRS["manifest"]/ "ENVIRONMENT.json").write_text(json.dumps(env,indent=2),encoding="utf-8")

model_path = DIRS["models"]/ "CROSS_TOPOLOGY_CLASSIFIER.joblib"
summary = {
    "version":VERSION,
    "completed_utc":utcnow(),
    "runner_sha256":runner_hash,
    "authentic_graph_edges":AUTH_EDGES,
    "feature_names":FEATURE_NAMES,
    "unit_tests":unit_rows,
    "authentic_family_summary":auth_summary.to_dict(orient="records"),
    "cross_topology_generalization_auc":auc,
    "base_graph_overlap":base_overlap,
    "leave_family_out":family_rows,
    "model_sha256":sha256_file(model_path),
    "rsos_specific_status":"UNTESTED",
    "v12_graph_extractor_calibration_status":"CALIBRATED_DEVELOPMENT_ONLY",
    "external_00086_read":False,
}
summary_path = DIRS["manifest"]/ "R4_SUMMARY.json"
summary_path.write_text(json.dumps(summary,indent=2),encoding="utf-8")

txt = [
    "F1-DEV R4 — CORRECTED V12 GRAPH CALIBRATION",
    "",
    f"Runner SHA256: {runner_hash}",
    f"Model SHA256: {summary['model_sha256']}",
    "",
    "UNIT TESTS:",
    unit_df.to_string(index=False),
    "",
    "AUTHENTIC PERTURBATION SUMMARY:",
    auth_summary.to_string(index=False),
    "",
    f"Unseen-base-topology AUC: {auc:.6f}",
    f"Base graph overlap train/test: {base_overlap}",
    "",
    "LEAVE-PERTURBATION-FAMILY-OUT:",
    pd.DataFrame(family_rows).to_string(index=False),
    "",
    "RSOS-SPECIFIC STATUS: UNTESTED",
    "V12 GRAPH EXTRACTOR: DEVELOPMENT CALIBRATION ONLY",
    "00086 EXTERNAL CONTROLS READ: NO",
]
(DIRS["manifest"]/ "R4_SUMMARY.txt").write_text("\n".join(txt)+"\n",encoding="utf-8")

# Copy executed runner into same experiment folder.
runner_copy = ROOT / runner.name
if runner_copy.resolve() != runner.resolve():
    runner_copy.write_bytes(runner.read_bytes())

# Hash all outputs except the final package ZIP (created by PowerShell after Python exits).
manifest_rows = []
for p in sorted(x for x in ROOT.rglob("*") if x.is_file()):
    manifest_rows.append({
        "RELATIVE_PATH":str(p.relative_to(ROOT)),
        "BYTES":p.stat().st_size,
        "SHA256":sha256_file(p),
    })
pd.DataFrame(manifest_rows).to_csv(DIRS["manifest"]/ "OUTPUT_SHA256_MANIFEST.csv",index=False)

COMPLETE.write_text(f"{VERSION}\n{utcnow()}\n{runner_hash}\n",encoding="ascii")

print()
print("="*78)
print("F1-DEV R4 COMPLETE")
print("="*78)
print(f"Unseen-base-topology AUC: {auc:.6f}")
print(f"Base graph overlap: {base_overlap}")
print("RSOS-specific status: UNTESTED")
print("00086 external controls read: NO")
print("Results:",ROOT)
