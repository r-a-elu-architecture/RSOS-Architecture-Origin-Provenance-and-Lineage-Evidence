from pathlib import Path
import json, hashlib, math, platform, sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import networkx as nx
import sklearn
import scipy
import joblib

ROOT = Path(r"C:\RSOS\Training data\F1_DEV_R5_V12_SPECIFICITY_OPERATOR")
ROOT.mkdir(parents=True, exist_ok=True)

DIRS = {
    "manifest": ROOT / "00_MANIFEST",
    "graph": ROOT / "01_GRAPH_SPECIFICITY",
    "operator": ROOT / "02_OPERATOR_SPECIFICITY",
    "combined": ROOT / "03_COMBINED_COLLISION",
    "adversarial": ROOT / "04_ADVERSARIAL_NEAREST_CONTROLS",
    "ledger": ROOT / "05_NO_SILENT_OMISSION",
}
for p in DIRS.values():
    p.mkdir(parents=True, exist_ok=True)

COMPLETE = ROOT / "F1_DEV_R5_COMPLETE.marker"
if COMPLETE.exists():
    raise SystemExit(f"R5 already completed and will not be overwritten:\n{ROOT}")

VERSION = "F1_DEV_R5_V12_SPECIFICITY_OPERATOR"
MASTER_SEED = 120918
rng_dev = np.random.default_rng(MASTER_SEED)
rng_hold = np.random.default_rng(MASTER_SEED + 100000)

AUTH_EDGES = tuple(sorted([
    (0,1),(1,2),(2,3),(3,4),(4,5),
    (5,6),(6,7),(3,1),(5,2),(6,3)
]))
AUTH_Q = (0,1,2,3,4,5,6,7)
AUTH_K = (2,5)
AUTH_S = (3,1,3,6)  # source edge 3->1, switch target 3->6
AUTH_Z = 7

GRAPH_FEATURE_NAMES = [
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

OP_FEATURE_NAMES = [
    "ORDER_DIRECT_EDGE_COHERENCE",
    "ORDER_ADJ_REACHABILITY",
    "ORDER_ALL_FORWARD_REACHABILITY",
    "K_INCIDENT_EDGE_SHARE",
    "K_MUTUAL_REACHABILITY",
    "S_SOURCE_EXISTS",
    "S_TARGET_EXISTS",
    "S_TARGET_REACHABLE",
    "S_APPLICABLE",
    "Z_SINKNESS",
    "Z_REACHABLE_FROM_Q_FRAC",
    "Z_Q_POSITION",
]

def utcnow():
    return datetime.now(timezone.utc).isoformat()

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def graph_from(edges):
    G = nx.DiGraph()
    G.add_nodes_from(range(8))
    G.add_edges_from(edges)
    return G

def canonical_edges(edges):
    return tuple(sorted((int(a),int(b)) for a,b in edges))

def count_two_cycles(G):
    return sum(1 for a,b in G.edges() if a < b and G.has_edge(b,a))

def count_three_cycles(G):
    return sum(1 for cyc in nx.simple_cycles(G, length_bound=3) if len(cyc) == 3)

def graph_features(edges):
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
    return np.round(np.asarray(vals, dtype=float), 12)

def operator_features(edges, Q, K, S, Z):
    G = graph_from(edges)

    direct = []
    adj_reach = []
    for a,b in zip(Q[:-1],Q[1:]):
        direct.append(1.0 if G.has_edge(a,b) else 0.0)
        adj_reach.append(1.0 if nx.has_path(G,a,b) else 0.0)

    all_forward = []
    for i in range(len(Q)):
        for j in range(i+1,len(Q)):
            all_forward.append(1.0 if nx.has_path(G,Q[i],Q[j]) else 0.0)

    Kset = set(K)
    incident = sum(1 for a,b in G.edges() if a in Kset or b in Kset)
    k_mutual = 1.0 if nx.has_path(G,K[0],K[1]) and nx.has_path(G,K[1],K[0]) else 0.0

    sa,sb,ta,tb = S
    source_exists = 1.0 if G.has_edge(sa,sb) else 0.0
    target_exists = 1.0 if G.has_edge(ta,tb) else 0.0
    target_reach = 1.0 if nx.has_path(G,ta,tb) else 0.0
    applicable = 1.0 if source_exists == 1.0 and target_exists == 0.0 else 0.0

    indeg = G.in_degree(Z)
    outdeg = G.out_degree(Z)
    sinkness = indeg / max(1, indeg + outdeg)

    q_without_z = [q for q in Q if q != Z]
    reachable_from_q = np.mean([1.0 if nx.has_path(G,q,Z) else 0.0 for q in q_without_z]) if q_without_z else 0.0
    zpos = Q.index(Z) / max(1, len(Q)-1)

    vals = [
        float(np.mean(direct)),
        float(np.mean(adj_reach)),
        float(np.mean(all_forward)),
        incident / max(1, G.number_of_edges()),
        k_mutual,
        source_exists,
        target_exists,
        target_reach,
        applicable,
        sinkness,
        float(reachable_from_q),
        float(zpos),
    ]
    return np.round(np.asarray(vals,dtype=float),12)

def combined_features(edges,Q,K,S,Z):
    return np.r_[graph_features(edges), operator_features(edges,Q,K,S,Z)]

def is_iso_to_auth(edges):
    return nx.is_isomorphic(graph_from(edges), graph_from(AUTH_EDGES))

def degree_signature(edges):
    G = graph_from(edges)
    return (
        tuple(sorted(d for _,d in G.in_degree())),
        tuple(sorted(d for _,d in G.out_degree())),
    )

AUTH_DEGREE_SIG = degree_signature(AUTH_EDGES)

def degree_preserving_swap(edges, rng, attempts=200):
    e = set(canonical_edges(edges))
    el = list(e)
    for _ in range(attempts):
        i,j = rng.choice(len(el),2,replace=False)
        a,b = el[i]
        c,d = el[j]
        if len({a,b,c,d}) < 4:
            continue
        x=(a,d); y=(c,b)
        if a == d or c == b or x in e or y in e:
            continue
        new=set(e)
        new.remove((a,b)); new.remove((c,d))
        new.add(x); new.add(y)
        if len(new) == len(e):
            return canonical_edges(new)
    return None

def walk_degree_space(start, rng, n_steps):
    e = canonical_edges(start)
    for _ in range(n_steps):
        nxt = degree_preserving_swap(e,rng)
        if nxt is not None:
            e=nxt
    return e

def generate_noniso_graph_controls(rng, target, max_attempts=500000):
    seen=set()
    controls=[]
    attempts=0
    while len(controls) < target and attempts < max_attempts:
        attempts += 1
        steps = int(rng.integers(1,25))
        e = walk_degree_space(AUTH_EDGES,rng,steps)
        if e == AUTH_EDGES or e in seen:
            continue
        if degree_signature(e) != AUTH_DEGREE_SIG:
            continue
        if is_iso_to_auth(e):
            continue
        seen.add(e)
        controls.append(e)
    return controls, attempts

def random_operator_variant(rng):
    # At least one of Q/K/S/Z differs from authentic.
    while True:
        Q = tuple(int(x) for x in rng.permutation(8))
        K = tuple(sorted(int(x) for x in rng.choice(8,size=2,replace=False)))
        # Source and target are two directed node pairs with same source allowed or not.
        sa,sb = rng.choice(8,size=2,replace=False)
        ta,tb = rng.choice(8,size=2,replace=False)
        S = (int(sa),int(sb),int(ta),int(tb))
        Z = int(rng.integers(0,8))
        if (Q,K,S,Z) != (AUTH_Q,AUTH_K,AUTH_S,AUTH_Z):
            return Q,K,S,Z

def robust_scale(matrix):
    matrix=np.asarray(matrix,float)
    med=np.median(matrix,axis=0)
    mad=np.median(np.abs(matrix-med),axis=0)*1.4826
    std=np.std(matrix,axis=0)
    scale=np.where(mad>1e-9,mad,np.where(std>1e-9,std,1.0))
    return med,scale

def distances_to_auth(M, auth, scale):
    return np.sqrt(np.sum(((M-auth)/scale)**2,axis=1))

def collision_stats(df, feature_cols, auth_vec, prefix):
    M=df[feature_cols].to_numpy(float)
    rows=[]
    for decimals in [12,9,6,4,3]:
        a=np.round(auth_vec,decimals)
        c=np.all(np.round(M,decimals)==a,axis=1)
        rows.append({
            "TEST":prefix,
            "ROUND_DECIMALS":decimals,
            "N_CONTROLS":len(df),
            "N_COLLISIONS":int(c.sum()),
            "COLLISION_RATE":float(c.mean()) if len(c) else np.nan,
        })
    return rows

print("="*80)
print("F1-DEV R5 - V12 SPECIFICITY / COLLISION / OPERATOR CALIBRATION")
print("="*80)
print("This is DEVELOPMENT calibration. It does not test the real RSOS corpus.")
print("Reserved 00086 external controls are not accessed.")

# ---------------------------------------------------------------------
# 1. Generate independent DEV and HOLDOUT matched graph-control banks
# ---------------------------------------------------------------------
DEV_TARGET = 3500
HOLD_TARGET = 2500

dev_graphs, dev_attempts = generate_noniso_graph_controls(rng_dev,DEV_TARGET)
hold_graphs, hold_attempts = generate_noniso_graph_controls(rng_hold,HOLD_TARGET)

if len(dev_graphs) < 500 or len(hold_graphs) < 500:
    raise RuntimeError(
        f"Insufficient non-isomorphic matched graph controls: dev={len(dev_graphs)} hold={len(hold_graphs)}"
    )

auth_g = graph_features(AUTH_EDGES)
auth_o = operator_features(AUTH_EDGES,AUTH_Q,AUTH_K,AUTH_S,AUTH_Z)
auth_c = np.r_[auth_g,auth_o]

def graph_bank_df(graphs, partition):
    rows=[]
    for i,e in enumerate(graphs):
        gf=graph_features(e)
        of=operator_features(e,AUTH_Q,AUTH_K,AUTH_S,AUTH_Z)
        row={
            "PARTITION":partition,
            "CONTROL_ID":f"{partition}_G{i:05d}",
            "ISOMORPHIC_TO_AUTH":False,
            "EDGE_COUNT":len(e),
            "EDGES":json.dumps(list(e)),
        }
        for n,v in zip(GRAPH_FEATURE_NAMES,gf):
            row["G_"+n]=float(v)
        for n,v in zip(OP_FEATURE_NAMES,of):
            row["O_"+n]=float(v)
        rows.append(row)
    return pd.DataFrame(rows)

dev_gdf=graph_bank_df(dev_graphs,"DEV")
hold_gdf=graph_bank_df(hold_graphs,"HOLDOUT")
all_gdf=pd.concat([dev_gdf,hold_gdf],ignore_index=True)
all_gdf.to_parquet(DIRS["graph"]/ "MATCHED_NONISOMORPHIC_GRAPH_CONTROLS.parquet",index=False,compression="zstd")

graph_cols=["G_"+x for x in GRAPH_FEATURE_NAMES]
op_cols=["O_"+x for x in OP_FEATURE_NAMES]
combined_cols=graph_cols+op_cols

# ---------------------------------------------------------------------
# 2. Graph-only fingerprint collision / rank
# ---------------------------------------------------------------------
devG=dev_gdf[graph_cols].to_numpy(float)
holdG=hold_gdf[graph_cols].to_numpy(float)
_,gscale=robust_scale(devG)
gdist_dev=distances_to_auth(devG,auth_g,gscale)
gdist_hold=distances_to_auth(holdG,auth_g,gscale)

dev_gdf["GRAPH_DISTANCE"]=gdist_dev
hold_gdf["GRAPH_DISTANCE"]=gdist_hold

graph_collision_rows = collision_stats(dev_gdf,graph_cols,auth_g,"GRAPH_DEV")
graph_collision_rows += collision_stats(hold_gdf,graph_cols,auth_g,"GRAPH_HOLDOUT")
pd.DataFrame(graph_collision_rows).to_csv(DIRS["graph"]/ "GRAPH_FINGERPRINT_COLLISIONS.csv",index=False)

graph_summary=pd.DataFrame([
    {
        "PARTITION":"DEV",
        "N_CONTROLS":len(dev_gdf),
        "MIN_DISTANCE":float(gdist_dev.min()),
        "P01_DISTANCE":float(np.quantile(gdist_dev,.01)),
        "MEDIAN_DISTANCE":float(np.median(gdist_dev)),
        "EXACT_6DP_COLLISIONS":int(np.all(np.round(devG,6)==np.round(auth_g,6),axis=1).sum()),
    },
    {
        "PARTITION":"HOLDOUT",
        "N_CONTROLS":len(hold_gdf),
        "MIN_DISTANCE":float(gdist_hold.min()),
        "P01_DISTANCE":float(np.quantile(gdist_hold,.01)),
        "MEDIAN_DISTANCE":float(np.median(gdist_hold)),
        "EXACT_6DP_COLLISIONS":int(np.all(np.round(holdG,6)==np.round(auth_g,6),axis=1).sum()),
    }
])
graph_summary.to_csv(DIRS["graph"]/ "GRAPH_SPECIFICITY_SUMMARY.csv",index=False)

# ---------------------------------------------------------------------
# 3. Operator-only alternative bank on authentic graph
# ---------------------------------------------------------------------
def operator_bank(rng,n,partition):
    rows=[]
    seen=set()
    while len(rows)<n:
        Q,K,S,Z=random_operator_variant(rng)
        key=(Q,K,S,Z)
        if key in seen:
            continue
        seen.add(key)
        of=operator_features(AUTH_EDGES,Q,K,S,Z)
        row={
            "PARTITION":partition,
            "CONTROL_ID":f"{partition}_O{len(rows):05d}",
            "Q":json.dumps(Q),
            "K":json.dumps(K),
            "S":json.dumps(S),
            "Z":Z,
        }
        for name,val in zip(OP_FEATURE_NAMES,of):
            row["O_"+name]=float(val)
        rows.append(row)
    return pd.DataFrame(rows)

dev_odf=operator_bank(rng_dev,20000,"DEV")
hold_odf=operator_bank(rng_hold,10000,"HOLDOUT")
all_odf=pd.concat([dev_odf,hold_odf],ignore_index=True)
all_odf.to_parquet(DIRS["operator"]/ "ALTERNATIVE_OPERATOR_CONTROLS.parquet",index=False,compression="zstd")

devO=dev_odf[op_cols].to_numpy(float)
holdO=hold_odf[op_cols].to_numpy(float)
_,oscale=robust_scale(devO)
odist_dev=distances_to_auth(devO,auth_o,oscale)
odist_hold=distances_to_auth(holdO,auth_o,oscale)
dev_odf["OPERATOR_DISTANCE"]=odist_dev
hold_odf["OPERATOR_DISTANCE"]=odist_hold

operator_collision_rows=collision_stats(dev_odf,op_cols,auth_o,"OPERATOR_DEV")
operator_collision_rows+=collision_stats(hold_odf,op_cols,auth_o,"OPERATOR_HOLDOUT")
pd.DataFrame(operator_collision_rows).to_csv(DIRS["operator"]/ "OPERATOR_SIGNATURE_COLLISIONS.csv",index=False)

operator_summary=pd.DataFrame([
    {
        "PARTITION":"DEV",
        "N_CONTROLS":len(dev_odf),
        "MIN_DISTANCE":float(odist_dev.min()),
        "P01_DISTANCE":float(np.quantile(odist_dev,.01)),
        "MEDIAN_DISTANCE":float(np.median(odist_dev)),
        "EXACT_6DP_COLLISIONS":int(np.all(np.round(devO,6)==np.round(auth_o,6),axis=1).sum()),
    },
    {
        "PARTITION":"HOLDOUT",
        "N_CONTROLS":len(hold_odf),
        "MIN_DISTANCE":float(odist_hold.min()),
        "P01_DISTANCE":float(np.quantile(odist_hold,.01)),
        "MEDIAN_DISTANCE":float(np.median(odist_hold)),
        "EXACT_6DP_COLLISIONS":int(np.all(np.round(holdO,6)==np.round(auth_o,6),axis=1).sum()),
    }
])
operator_summary.to_csv(DIRS["operator"]/ "OPERATOR_SPECIFICITY_SUMMARY.csv",index=False)

# ---------------------------------------------------------------------
# 4. Combined adversarial controls: non-isomorphic matched graph +
#    alternative operator tuple.
# ---------------------------------------------------------------------
def combined_bank(graphs,rng,n,partition):
    rows=[]
    for i in range(n):
        e=graphs[i % len(graphs)]
        Q,K,S,Z=random_operator_variant(rng)
        cf=combined_features(e,Q,K,S,Z)
        row={
            "PARTITION":partition,
            "CONTROL_ID":f"{partition}_C{i:05d}",
            "EDGES":json.dumps(list(e)),
            "Q":json.dumps(Q),
            "K":json.dumps(K),
            "S":json.dumps(S),
            "Z":Z,
        }
        for name,val in zip(GRAPH_FEATURE_NAMES,cf[:len(GRAPH_FEATURE_NAMES)]):
            row["G_"+name]=float(val)
        for name,val in zip(OP_FEATURE_NAMES,cf[len(GRAPH_FEATURE_NAMES):]):
            row["O_"+name]=float(val)
        rows.append(row)
    return pd.DataFrame(rows)

dev_cdf=combined_bank(dev_graphs,rng_dev,20000,"DEV")
hold_cdf=combined_bank(hold_graphs,rng_hold,10000,"HOLDOUT")
all_cdf=pd.concat([dev_cdf,hold_cdf],ignore_index=True)
all_cdf.to_parquet(DIRS["combined"]/ "COMBINED_ADVERSARIAL_CONTROLS.parquet",index=False,compression="zstd")

devC=dev_cdf[combined_cols].to_numpy(float)
holdC=hold_cdf[combined_cols].to_numpy(float)
_,cscale=robust_scale(devC)
cdist_dev=distances_to_auth(devC,auth_c,cscale)
cdist_hold=distances_to_auth(holdC,auth_c,cscale)
dev_cdf["COMBINED_DISTANCE"]=cdist_dev
hold_cdf["COMBINED_DISTANCE"]=cdist_hold

combined_collision_rows=collision_stats(dev_cdf,combined_cols,auth_c,"COMBINED_DEV")
combined_collision_rows+=collision_stats(hold_cdf,combined_cols,auth_c,"COMBINED_HOLDOUT")
pd.DataFrame(combined_collision_rows).to_csv(DIRS["combined"]/ "COMBINED_SIGNATURE_COLLISIONS.csv",index=False)

combined_summary=pd.DataFrame([
    {
        "PARTITION":"DEV",
        "N_CONTROLS":len(dev_cdf),
        "MIN_DISTANCE":float(cdist_dev.min()),
        "P01_DISTANCE":float(np.quantile(cdist_dev,.01)),
        "MEDIAN_DISTANCE":float(np.median(cdist_dev)),
        "EXACT_6DP_COLLISIONS":int(np.all(np.round(devC,6)==np.round(auth_c,6),axis=1).sum()),
    },
    {
        "PARTITION":"HOLDOUT",
        "N_CONTROLS":len(hold_cdf),
        "MIN_DISTANCE":float(cdist_hold.min()),
        "P01_DISTANCE":float(np.quantile(cdist_hold,.01)),
        "MEDIAN_DISTANCE":float(np.median(cdist_hold)),
        "EXACT_6DP_COLLISIONS":int(np.all(np.round(holdC,6)==np.round(auth_c,6),axis=1).sum()),
    }
])
combined_summary.to_csv(DIRS["combined"]/ "COMBINED_SPECIFICITY_SUMMARY.csv",index=False)

# ---------------------------------------------------------------------
# 5. Adversarial nearest controls preserved for inspection
# ---------------------------------------------------------------------
nearest_graph=hold_gdf.nsmallest(100,"GRAPH_DISTANCE")
nearest_op=hold_odf.nsmallest(100,"OPERATOR_DISTANCE")
nearest_combined=hold_cdf.nsmallest(100,"COMBINED_DISTANCE")
nearest_graph.to_csv(DIRS["adversarial"]/ "TOP100_NEAREST_GRAPH_CONTROLS.csv",index=False)
nearest_op.to_csv(DIRS["adversarial"]/ "TOP100_NEAREST_OPERATOR_CONTROLS.csv",index=False)
nearest_combined.to_csv(DIRS["adversarial"]/ "TOP100_NEAREST_COMBINED_CONTROLS.csv",index=False)

# ---------------------------------------------------------------------
# 6. Exact authentic signature
# ---------------------------------------------------------------------
auth_row={
    "EDGES":json.dumps(list(AUTH_EDGES)),
    "Q":json.dumps(AUTH_Q),
    "K":json.dumps(AUTH_K),
    "S":json.dumps(AUTH_S),
    "Z":AUTH_Z,
}
for n,v in zip(GRAPH_FEATURE_NAMES,auth_g):
    auth_row["G_"+n]=float(v)
for n,v in zip(OP_FEATURE_NAMES,auth_o):
    auth_row["O_"+n]=float(v)
pd.DataFrame([auth_row]).to_csv(DIRS["manifest"]/ "AUTHENTIC_V12_FROZEN_SIGNATURE.csv",index=False)

# ---------------------------------------------------------------------
# 7. No-silent-omission ledger
# ---------------------------------------------------------------------
ledger=pd.DataFrame([
    ["NONISOMORPHIC_DEGREE_MATCHED_GRAPH_CONTROLS","RUN","Exact sorted in/out degree signature + edge count preserved"],
    ["ISOMORPHISM_EXCLUSION","RUN","Every graph control explicitly excluded if isomorphic to authentic"],
    ["GRAPH_FINGERPRINT_COLLISION_12_9_6_4_3_DP","RUN","Exact rounded collision rates reported"],
    ["OPERATOR_ALTERNATIVES_Q_K_S_Z","RUN","Independent alternative operator bank"],
    ["COMBINED_GRAPH_OPERATOR_COLLISION","RUN","Non-isomorphic graph + alternative operator jointly tested"],
    ["ADVERSARIAL_NEAREST_CONTROLS","RUN","100 closest holdout controls preserved"],
    ["V12_DIRECTION_INFORMATION","PARTIAL","Graph direction enters frozen graph features; behavioral direction response not tested"],
    ["V12_ORDER_OPERATOR","CALIBRATED_STRUCTURALLY","Q-derived structural metrics only; no LLM behavioral response"],
    ["V12_META_SWITCH_OPERATOR","CALIBRATED_STRUCTURALLY","S-derived structural metrics only; no LLM behavioral response"],
    ["V12_PRESERVATION_OPERATOR","CALIBRATED_STRUCTURALLY","K-derived structural metrics only; no LLM behavioral response"],
    ["V12_TERMINAL_OPERATOR","CALIBRATED_STRUCTURALLY","Z-derived structural metrics only; no LLM behavioral response"],
    ["REAL_RSOS_CORPUS_MAPPING","UNTESTED","No positive or negative inference"],
    ["LMARENA_RSOS_MAPPING","UNTESTED","No positive or negative inference"],
    ["F2_EXTERNAL_00086_RARITY","RESERVED_NOT_READ","No population rarity inference"],
    ["F3_PROSPECTIVE_REPLICATION","UNTESTED","Historical development calibration only"],
],columns=["MEASUREMENT","STATUS","DETAIL"])
ledger.to_csv(DIRS["ledger"]/ "NO_SILENT_OMISSION_LEDGER.csv",index=False)

# ---------------------------------------------------------------------
# 8. Summary and provenance
# ---------------------------------------------------------------------
graph_hold_coll6=int(np.all(np.round(holdG,6)==np.round(auth_g,6),axis=1).sum())
op_hold_coll6=int(np.all(np.round(holdO,6)==np.round(auth_o,6),axis=1).sum())
comb_hold_coll6=int(np.all(np.round(holdC,6)==np.round(auth_c,6),axis=1).sum())

summary={
    "version":VERSION,
    "completed_utc":utcnow(),
    "authentic_edges":list(AUTH_EDGES),
    "authentic_Q":list(AUTH_Q),
    "authentic_K":list(AUTH_K),
    "authentic_S":list(AUTH_S),
    "authentic_Z":AUTH_Z,
    "dev_graph_controls":len(dev_graphs),
    "holdout_graph_controls":len(hold_graphs),
    "dev_graph_generation_attempts":dev_attempts,
    "holdout_graph_generation_attempts":hold_attempts,
    "graph_holdout_min_distance":float(gdist_hold.min()),
    "graph_holdout_exact_6dp_collisions":graph_hold_coll6,
    "operator_holdout_min_distance":float(odist_hold.min()),
    "operator_holdout_exact_6dp_collisions":op_hold_coll6,
    "combined_holdout_min_distance":float(cdist_hold.min()),
    "combined_holdout_exact_6dp_collisions":comb_hold_coll6,
    "interpretation":{
        "graph_specificity":"Development collision audit only",
        "operator_specificity":"Structural operator calibration only",
        "combined_specificity":"Development collision audit only",
        "rsos_specific_real_corpus":"UNTESTED",
        "external_rarity":"UNTESTED",
    },
    "external_00086_read":False,
}
summary_path=DIRS["manifest"]/ "R5_SUMMARY.json"
summary_path.write_text(json.dumps(summary,indent=2),encoding="utf-8")

txt=[
    "F1-DEV R5 - V12 SPECIFICITY / COLLISION / OPERATOR CALIBRATION",
    "",
    f"Matched non-isomorphic graph controls: DEV={len(dev_graphs)} HOLDOUT={len(hold_graphs)}",
    f"Graph holdout minimum standardized distance: {float(gdist_hold.min()):.6f}",
    f"Graph holdout exact 6dp collisions: {graph_hold_coll6}",
    "",
    f"Operator holdout minimum standardized distance: {float(odist_hold.min()):.6f}",
    f"Operator holdout exact 6dp collisions: {op_hold_coll6}",
    "",
    f"Combined holdout minimum standardized distance: {float(cdist_hold.min()):.6f}",
    f"Combined holdout exact 6dp collisions: {comb_hold_coll6}",
    "",
    "RSOS REAL-CORPUS MAPPING: UNTESTED",
    "LMARENA RSOS MAPPING: UNTESTED",
    "F2 00086 EXTERNAL CONTROLS READ: NO",
]
(DIRS["manifest"]/ "R5_SUMMARY.txt").write_text("\n".join(txt)+"\n",encoding="utf-8")

env={
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

runner=Path(__file__).resolve()
runner_hash=sha256_file(runner)
(DIRS["manifest"]/ "PYTHON_RUNNER_SHA256.txt").write_text(runner_hash+"\n",encoding="ascii")

# Copy executed Python runner into the same folder.
runner_copy=ROOT/runner.name
if runner_copy.resolve()!=runner.resolve():
    runner_copy.write_bytes(runner.read_bytes())

# Hash everything existing before final ZIP.
rows=[]
for p in sorted(x for x in ROOT.rglob("*") if x.is_file()):
    rows.append({
        "RELATIVE_PATH":str(p.relative_to(ROOT)),
        "BYTES":p.stat().st_size,
        "SHA256":sha256_file(p),
    })
pd.DataFrame(rows).to_csv(DIRS["manifest"]/ "OUTPUT_SHA256_MANIFEST.csv",index=False)

COMPLETE.write_text(f"{VERSION}\n{utcnow()}\n{runner_hash}\n",encoding="ascii")

print()
print("="*80)
print("F1-DEV R5 COMPLETE")
print("="*80)
print(f"Matched graph controls: DEV={len(dev_graphs)} HOLDOUT={len(hold_graphs)}")
print(f"Graph holdout exact 6dp collisions: {graph_hold_coll6}")
print(f"Operator holdout exact 6dp collisions: {op_hold_coll6}")
print(f"Combined holdout exact 6dp collisions: {comb_hold_coll6}")
print("RSOS real-corpus mapping: UNTESTED")
print("00086 external controls read: NO")
print("Results:",ROOT)
