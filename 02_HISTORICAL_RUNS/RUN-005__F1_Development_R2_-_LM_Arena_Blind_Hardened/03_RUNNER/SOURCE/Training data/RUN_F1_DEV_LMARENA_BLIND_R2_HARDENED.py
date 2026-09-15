from pathlib import Path
import json, hashlib, itertools, math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib

from scipy.stats import chi2_contingency
from sklearn.linear_model import Ridge
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    roc_auc_score,
    silhouette_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    balanced_accuracy_score,
    f1_score,
)

BASE = Path(r"C:\RSOS\Training data\LMARENA_V12_V18_UPDATED")
OUT  = Path(r"C:\RSOS\Training data\F1_DEV_LMARENA_BLIND_R2_HARDENED")

TRAIN_PATH = BASE / r"06_DEVELOPMENT_SPLITS\DEVELOPMENT_TRAIN\train-00000-of-00007__DEVELOPMENT_TRAIN.parquet"
VAL_PATH   = BASE / r"06_DEVELOPMENT_SPLITS\DEVELOPMENT_VALIDATION\train-00000-of-00007__DEVELOPMENT_VALIDATION.parquet"
HOLD_PATH  = BASE / r"06_DEVELOPMENT_SPLITS\DEVELOPMENT_HOLDOUT\train-00000-of-00007__DEVELOPMENT_HOLDOUT.parquet"

MASTER_SEED = 120918
VERSION = "F1_DEV_LMARENA_BLIND_R2_HARDENED"

DIRS = {
    "manifest": OUT / "00_MANIFEST",
    "cal": OUT / "01_HARDENED_CALIBRATION",
    "cluster": OUT / "02_CLUSTER_NULL_BENCHMARK",
    "strata": OUT / "03_CODE_STRATIFIED",
    "hold": OUT / "04_PHYSICALLY_SEALED_HOLDOUT",
    "audit": OUT / "05_POSTFREEZE_AUDIT",
    "ledger": OUT / "06_OMISSION_LEDGER",
    "models": OUT / "07_FROZEN_MODELS",
}
for p in DIRS.values():
    p.mkdir(parents=True, exist_ok=True)

COMPLETE = OUT / "F1_DEV_R2_COMPLETE.marker"
if COMPLETE.exists():
    raise SystemExit(f"Completed R2 already exists and will not be overwritten:\n{OUT}")

PRE_COLS = [
    "BLIND_ID","RECORD_ID","PAIR_ID","EVALUATION_SESSION_ID","LANGUAGE","IS_CODE",
    "ERA_2025_06_08",
    "N_TURNS","N_USER_TURNS","N_ASSISTANT_TURNS",
    "TOTAL_CHARS","USER_CHARS","ASSISTANT_CHARS","WORD_COUNT",
    "TYPE_TOKEN_RATIO","MEAN_TURN_CHARS","MAX_TURN_CHARS",
    "PARAGRAPH_COUNT","CODE_FENCE_COUNT","MARKDOWN_HEADER_COUNT",
    "LIST_ITEM_COUNT","COLON_COUNT","BRACKET_COUNT",
    "UPPERCASE_RATIO","DIGIT_RATIO","COMPRESSION_RATIO",
]

META_COLS = [
    "BLIND_ID","MODEL","PROVIDER","WINNER_RAW","PERSPECTIVE_LABEL",
    "LANGUAGE","IS_CODE","ERA_2025_06_08"
]

def utcnow():
    return datetime.now(timezone.utc).isoformat()

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def safe_ratio(a, b):
    a = pd.to_numeric(a, errors="coerce").astype(float)
    b = pd.to_numeric(b, errors="coerce").astype(float)
    return a / b.replace(0, np.nan)

def engineered(df):
    x = pd.DataFrame(index=df.index)
    x["LOG_TOTAL_CHARS"] = np.log1p(pd.to_numeric(df["TOTAL_CHARS"], errors="coerce").clip(lower=0))
    x["LOG_N_TURNS"] = np.log1p(pd.to_numeric(df["N_TURNS"], errors="coerce").clip(lower=0))
    x["IS_CODE_NUM"] = df["IS_CODE"].astype(int)
    x["TYPE_TOKEN_RATIO"] = pd.to_numeric(df["TYPE_TOKEN_RATIO"], errors="coerce")
    x["UPPERCASE_RATIO"] = pd.to_numeric(df["UPPERCASE_RATIO"], errors="coerce")
    x["DIGIT_RATIO"] = pd.to_numeric(df["DIGIT_RATIO"], errors="coerce")
    x["COMPRESSION_RATIO"] = pd.to_numeric(df["COMPRESSION_RATIO"], errors="coerce")
    x["USER_CHAR_SHARE"] = safe_ratio(df["USER_CHARS"], df["TOTAL_CHARS"])
    x["MEAN_MAX_TURN_RATIO"] = safe_ratio(df["MEAN_TURN_CHARS"], df["MAX_TURN_CHARS"])
    x["LOG_WORD_COUNT"] = np.log1p(pd.to_numeric(df["WORD_COUNT"], errors="coerce").clip(lower=0))
    x["LOG_PARAGRAPH_COUNT"] = np.log1p(pd.to_numeric(df["PARAGRAPH_COUNT"], errors="coerce").clip(lower=0))
    x["LOG_CODE_FENCE_COUNT"] = np.log1p(pd.to_numeric(df["CODE_FENCE_COUNT"], errors="coerce").clip(lower=0))
    x["LOG_HEADER_COUNT"] = np.log1p(pd.to_numeric(df["MARKDOWN_HEADER_COUNT"], errors="coerce").clip(lower=0))
    x["LOG_LIST_ITEM_COUNT"] = np.log1p(pd.to_numeric(df["LIST_ITEM_COUNT"], errors="coerce").clip(lower=0))
    x["LOG_COLON_COUNT"] = np.log1p(pd.to_numeric(df["COLON_COUNT"], errors="coerce").clip(lower=0))
    x["LOG_BRACKET_COUNT"] = np.log1p(pd.to_numeric(df["BRACKET_COUNT"], errors="coerce").clip(lower=0))
    return x.replace([np.inf,-np.inf], np.nan)

TARGETS = [
    "TYPE_TOKEN_RATIO","UPPERCASE_RATIO","DIGIT_RATIO","COMPRESSION_RATIO",
    "USER_CHAR_SHARE","MEAN_MAX_TURN_RATIO","LOG_WORD_COUNT",
    "LOG_PARAGRAPH_COUNT","LOG_CODE_FENCE_COUNT","LOG_HEADER_COUNT",
    "LOG_LIST_ITEM_COUNT","LOG_COLON_COUNT","LOG_BRACKET_COUNT",
]

def qedges(a, q):
    a = np.asarray(a, dtype=float)
    e = np.unique(np.quantile(a[np.isfinite(a)], np.linspace(0,1,q+1)))
    if len(e) < 2:
        return np.array([-np.inf, np.inf])
    e[0], e[-1] = -np.inf, np.inf
    return e

def bin_with_edges(a, e):
    return np.digitize(np.asarray(a, dtype=float), e[1:-1], right=False)

def make_strata(E, char_edges, turn_edges, include_code):
    a = bin_with_edges(E["LOG_TOTAL_CHARS"], char_edges)
    b = bin_with_edges(E["LOG_N_TURNS"], turn_edges)
    if include_code:
        c = E["IS_CODE_NUM"].to_numpy(dtype=int)
        return np.array([f"{x}|{y}|{z}" for x,y,z in zip(a,b,c)], dtype=object)
    return np.array([f"{x}|{y}" for x,y in zip(a,b)], dtype=object)

def stratified_column_shuffle(X, strata, seed):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    Y = X.copy()
    strata = np.asarray(strata)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        if len(idx) < 3:
            continue
        for j in range(X.shape[1]):
            Y[idx, j] = X[rng.permutation(idx), j]
    return Y

def gaussian_sample(mu, cov, n, seed):
    rng = np.random.default_rng(seed)
    cov = np.asarray(cov, dtype=float)
    cov = (cov + cov.T) / 2
    cov = cov + np.eye(cov.shape[0]) * 1e-6
    return rng.multivariate_normal(mu, cov, size=n)

def cramer_v(x, y):
    tab = pd.crosstab(pd.Series(x, dtype="string"), pd.Series(y, dtype="string"))
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan, np.nan, np.nan, tab.shape
    chi2, p, dof, exp = chi2_contingency(tab.values)
    n = tab.values.sum()
    v = math.sqrt(chi2 / (n * min(tab.shape[0]-1, tab.shape[1]-1)))
    return float(v), float(p), float(chi2), tab.shape

def fit_residualizer(trainE, scope):
    nuisance = ["LOG_TOTAL_CHARS","LOG_N_TURNS"]
    if scope == "ALL":
        nuisance.append("IS_CODE_NUM")
    Xn = trainE[nuisance].copy()
    med = Xn.median()
    Xn = Xn.fillna(med)
    models, target_medians, kept = {}, {}, []
    for t in TARGETS:
        y = trainE[t].astype(float)
        ym = float(y.median()) if y.notna().any() else 0.0
        y = y.fillna(ym)
        if float(y.var()) <= 1e-12:
            continue
        model = Ridge(alpha=1.0)
        model.fit(Xn, y)
        models[t] = model
        target_medians[t] = ym
        kept.append(t)
    return {
        "nuisance": nuisance, "nuisance_medians": med.to_dict(),
        "models": models, "target_medians": target_medians, "targets": kept
    }

def apply_residualizer(E, R):
    Xn = E[R["nuisance"]].copy()
    for c,v in R["nuisance_medians"].items():
        Xn[c] = Xn[c].fillna(v)
    out = pd.DataFrame(index=E.index)
    for t in R["targets"]:
        y = E[t].astype(float).fillna(R["target_medians"][t])
        out[t+"_RESID"] = y.to_numpy() - R["models"][t].predict(Xn)
    return out.replace([np.inf,-np.inf], np.nan).fillna(0.0)

def auc_real_vs_null(Xtr, Xev, Ntr, Nev, seed):
    clf = ExtraTreesClassifier(
        n_estimators=250, min_samples_leaf=5, max_features="sqrt",
        class_weight="balanced", n_jobs=-1, random_state=seed
    )
    XX = np.vstack([Xtr, Ntr])
    yy = np.r_[np.ones(len(Xtr),dtype=int), np.zeros(len(Ntr),dtype=int)]
    clf.fit(XX, yy)
    EE = np.vstack([Xev, Nev])
    ey = np.r_[np.ones(len(Xev),dtype=int), np.zeros(len(Nev),dtype=int)]
    return clf, float(roc_auc_score(ey, clf.predict_proba(EE)[:,1]))

def sampled_silhouette(Z, labels, seed):
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(silhouette_score(
        Z, labels, sample_size=min(2500,len(Z)), random_state=seed
    ))

def fit_scope(train_df, val_df, scope):
    if scope == "ALL":
        tr = train_df.copy()
        va = val_df.copy()
    elif scope == "NONCODE":
        tr = train_df.loc[~train_df["IS_CODE"].astype(bool)].copy()
        va = val_df.loc[~val_df["IS_CODE"].astype(bool)].copy()
    elif scope == "CODE":
        tr = train_df.loc[train_df["IS_CODE"].astype(bool)].copy()
        va = val_df.loc[val_df["IS_CODE"].astype(bool)].copy()
    else:
        raise ValueError(scope)

    Et = engineered(tr)
    Ev = engineered(va)

    R = fit_residualizer(Et, scope)
    Rt = apply_residualizer(Et, R)
    Rv = apply_residualizer(Ev, R)

    q = QuantileTransformer(
        n_quantiles=min(1000, len(Rt)),
        output_distribution="normal",
        subsample=None,
        random_state=MASTER_SEED
    )
    Xt = q.fit_transform(Rt)
    Xv = q.transform(Rv)

    pca = PCA(n_components=0.90, svd_solver="full")
    Zt = pca.fit_transform(Xt)
    Zv = pca.transform(Xv)

    mu = Xt.mean(axis=0)
    cov = np.cov(Xt, rowvar=False)

    ce = qedges(Et.loc[Rt.index, "LOG_TOTAL_CHARS"], 5)
    te = qedges(Et.loc[Rt.index, "LOG_N_TURNS"], 3)
    st = make_strata(Et.loc[Rt.index], ce, te, scope=="ALL")
    sv = make_strata(Ev.loc[Rv.index], ce, te, scope=="ALL")

    sh_tr = stratified_column_shuffle(Xt, st, MASTER_SEED+10)
    sh_va = stratified_column_shuffle(Xv, sv, MASTER_SEED+11)
    g_tr = gaussian_sample(mu, cov, len(Xt), MASTER_SEED+20)
    g_va = gaussian_sample(mu, cov, len(Xv), MASTER_SEED+21)

    shuffle_clf, shuffle_auc_val = auc_real_vs_null(Xt, Xv, sh_tr, sh_va, MASTER_SEED+30)
    gaussian_clf, gaussian_auc_val = auc_real_vs_null(Xt, Xv, g_tr, g_va, MASTER_SEED+40)

    sel = []
    for k in range(2,9):
        km = KMeans(n_clusters=k, n_init=20, max_iter=500, random_state=MASTER_SEED)
        km.fit(Zt)
        pred = km.predict(Zv)
        obs = sampled_silhouette(Zv, pred, MASTER_SEED+k)

        nulls = []
        for r in range(12):
            ngt = gaussian_sample(np.zeros(Zt.shape[1]), np.cov(Zt,rowvar=False), len(Zt), MASTER_SEED+1000+k*100+r)
            ngv = gaussian_sample(np.zeros(Zt.shape[1]), np.cov(Zt,rowvar=False), len(Zv), MASTER_SEED+2000+k*100+r)
            nk = KMeans(n_clusters=k, n_init=8, max_iter=400, random_state=MASTER_SEED+r)
            nk.fit(ngt)
            npred = nk.predict(ngv)
            nulls.append(sampled_silhouette(ngv, npred, MASTER_SEED+r))
        nulls = np.array(nulls, dtype=float)
        pnull = float((1 + np.sum(nulls >= obs)) / (len(nulls)+1))

        preds = []
        rng = np.random.default_rng(MASTER_SEED+k)
        for b in range(8):
            idx = rng.choice(len(Zt), size=max(100,int(0.8*len(Zt))), replace=False)
            bk = KMeans(n_clusters=k, n_init=10, max_iter=400, random_state=MASTER_SEED+b)
            bk.fit(Zt[idx])
            preds.append(bk.predict(Zv))
        aris = [
            adjusted_rand_score(a,b)
            for a,b in itertools.combinations(preds,2)
        ]
        stability = float(np.mean(aris)) if aris else np.nan
        sel.append({
            "SCOPE":scope, "K":k, "OBS_VALIDATION_SILHOUETTE":obs,
            "GAUSSIAN_NULL_MEAN":float(np.nanmean(nulls)),
            "GAUSSIAN_NULL_95":float(np.nanquantile(nulls,0.95)),
            "EXCESS_SILHOUETTE":float(obs-np.nanmean(nulls)),
            "EMPIRICAL_NULL_P":pnull,
            "BOOTSTRAP_STABILITY_ARI":stability,
        })

    sel_df = pd.DataFrame(sel)
    best = sel_df.sort_values(
        ["EXCESS_SILHOUETTE","BOOTSTRAP_STABILITY_ARI","K"],
        ascending=[False,False,True]
    ).iloc[0]
    k = int(best["K"])
    km = KMeans(n_clusters=k, n_init=30, max_iter=600, random_state=MASTER_SEED)
    km.fit(Zt)

    model = {
        "scope":scope, "residualizer":R, "quantile":q, "pca":pca,
        "kmeans":km, "shuffle_clf":shuffle_clf, "gaussian_clf":gaussian_clf,
        "char_edges":ce, "turn_edges":te, "mu":mu, "cov":cov,
        "targets":list(Rt.columns),
        "selected_k":k,
    }

    pre = {
        "scope":scope, "train_ids":tr["BLIND_ID"].astype(str).tolist(),
        "val_ids":va["BLIND_ID"].astype(str).tolist(),
        "Xt":Xt, "Xv":Xv, "Zt":Zt, "Zv":Zv,
        "train_labels":km.predict(Zt), "val_labels":km.predict(Zv),
        "validation_silhouette":sampled_silhouette(Zv, km.predict(Zv), MASTER_SEED+300),
        "shuffle_auc_validation":shuffle_auc_val,
        "gaussian_auc_validation":gaussian_auc_val,
    }
    return model, pre, sel_df

def transform_scope(df, model):
    scope = model["scope"]
    if scope == "ALL":
        d = df.copy()
    elif scope == "NONCODE":
        d = df.loc[~df["IS_CODE"].astype(bool)].copy()
    else:
        d = df.loc[df["IS_CODE"].astype(bool)].copy()
    E = engineered(d)
    R = apply_residualizer(E, model["residualizer"])
    X = model["quantile"].transform(R)
    Z = model["pca"].transform(X)
    labels = model["kmeans"].predict(Z)
    return d, E, X, Z, labels

print("="*72)
print("F1-DEV R2 HARDENED METHODOLOGY CORRECTION")
print("="*72)
print("R1 negative/absence interpretations are NOT carried forward.")
print("Reserved 00086 external controls are not accessed.")

# Hash all split files before reading semantics.
input_hashes = []
for p,role in [(TRAIN_PATH,"TRAIN"),(VAL_PATH,"VALIDATION"),(HOLD_PATH,"SEALED_HOLDOUT")]:
    if not p.exists():
        raise RuntimeError(f"Missing required split: {p}")
    input_hashes.append({"ROLE":role,"PATH":str(p),"BYTES":p.stat().st_size,"SHA256":sha256_file(p)})
pd.DataFrame(input_hashes).to_csv(DIRS["manifest"]/ "INPUT_SPLIT_SHA256.csv", index=False)

# IMPORTANT: only TRAIN and VALIDATION are opened here.
train = pd.read_parquet(TRAIN_PATH, columns=PRE_COLS)
val = pd.read_parquet(VAL_PATH, columns=PRE_COLS)

# Pair/session leakage checks across train and validation.
if set(train["EVALUATION_SESSION_ID"].astype(str)) & set(val["EVALUATION_SESSION_ID"].astype(str)):
    raise RuntimeError("Train/validation session leakage detected.")

models, predata, selection_tables = {}, {}, []
for scope in ["ALL","NONCODE","CODE"]:
    print(f"Fitting pre-holdout scope: {scope}")
    m, p, s = fit_scope(train, val, scope)
    models[scope] = m
    predata[scope] = p
    selection_tables.append(s)
    joblib.dump(m, DIRS["models"]/f"{scope}_FROZEN_MODEL.joblib", compress=3)

pd.concat(selection_tables, ignore_index=True).to_csv(
    DIRS["cluster"]/ "PREHOLDOUT_CLUSTER_NULL_SELECTION.csv", index=False
)

pre_rows = []
for scope,p in predata.items():
    pre_rows.append({
        "SCOPE":scope,
        "N_TRAIN":len(p["train_ids"]),
        "N_VALIDATION":len(p["val_ids"]),
        "SELECTED_K":models[scope]["selected_k"],
        "VALIDATION_SILHOUETTE":p["validation_silhouette"],
        "AUC_STRATIFIED_SHUFFLE_VALIDATION":p["shuffle_auc_validation"],
        "AUC_COVARIANCE_MATCHED_GAUSSIAN_VALIDATION":p["gaussian_auc_validation"],
    })
pd.DataFrame(pre_rows).to_csv(DIRS["cal"]/ "PREHOLDOUT_CALIBRATION.csv", index=False)

# Freeze exact executable artifacts BEFORE reading holdout.
model_hashes = []
for p in sorted(DIRS["models"].glob("*.joblib")):
    model_hashes.append({"FILE":p.name,"SHA256":sha256_file(p),"BYTES":p.stat().st_size})

freeze_spec = {
    "version":VERSION,
    "frozen_utc":utcnow(),
    "script_sha256":sha256_file(Path(__file__)),
    "input_split_hashes":input_hashes,
    "frozen_model_hashes":model_hashes,
    "scopes":["ALL","NONCODE","CODE"],
    "interpretation_rules":{
        "R1_LOW_AMI_DOES_NOT_ESTABLISH_INVARIANCE":True,
        "R1_IS_CODE_AMI_DOES_NOT_ESTABLISH_CONFOUND_CAUSALITY":True,
        "R1_SILHOUETTE_DOES_NOT_TEST_RSOS_ABSENCE":True,
        "R1_GRAPH_STABILITY_DOES_NOT_ESTABLISH_RSOS_TOPOLOGY":True,
        "R1_COLUMN_SHUFFLE_AUC_IS_GENERIC_DEPENDENCE_SANITY_ONLY":True,
    },
    "holdout_rule":"HOLDOUT parquet is not opened with pandas until this spec and frozen model files exist.",
    "external_control_rule":"train-*-of-00086.parquet remains reserved and unread.",
    "claim_scope":"DEVELOPMENT METHODOLOGY ONLY; RSOS-SPECIFIC F1 TARGET REMAINS UNTESTED."
}
freeze_path = DIRS["manifest"]/ "R2_FROZEN_SPEC_BEFORE_HOLDOUT.json"
freeze_path.write_text(json.dumps(freeze_spec, indent=2), encoding="utf-8")
freeze_hash = sha256_file(freeze_path)
(DIRS["manifest"]/ "R2_FROZEN_SPEC_SHA256.txt").write_text(freeze_hash+"\n", encoding="ascii")

print("FROZEN BEFORE HOLDOUT:", freeze_hash)

# HOLDOUT IS OPENED ONLY NOW.
hold = pd.read_parquet(HOLD_PATH, columns=PRE_COLS)
if set(train["EVALUATION_SESSION_ID"].astype(str)) & set(hold["EVALUATION_SESSION_ID"].astype(str)):
    raise RuntimeError("Train/holdout session leakage detected.")
if set(val["EVALUATION_SESSION_ID"].astype(str)) & set(hold["EVALUATION_SESSION_ID"].astype(str)):
    raise RuntimeError("Validation/holdout session leakage detected.")

hold_rows = []
assignment_frames = {}

for scope in ["ALL","NONCODE","CODE"]:
    model = joblib.load(DIRS["models"]/f"{scope}_FROZEN_MODEL.joblib")
    d,E,X,Z,labels = transform_scope(hold, model)

    strata = make_strata(E, model["char_edges"], model["turn_edges"], scope=="ALL")
    sh = stratified_column_shuffle(X, strata, MASTER_SEED+500)
    ga = gaussian_sample(model["mu"], model["cov"], len(X), MASTER_SEED+600)

    y = np.r_[np.ones(len(X),dtype=int),np.zeros(len(X),dtype=int)]
    eval_sh = np.vstack([X,sh])
    eval_ga = np.vstack([X,ga])
    auc_sh = float(roc_auc_score(y, model["shuffle_clf"].predict_proba(eval_sh)[:,1]))
    auc_ga = float(roc_auc_score(y, model["gaussian_clf"].predict_proba(eval_ga)[:,1]))

    sil = sampled_silhouette(Z, labels, MASTER_SEED+700)

    # Null benchmark for selected K on holdout.
    null_sils = []
    Ztr = predata[scope]["Zt"]
    k = model["selected_k"]
    for r in range(20):
        ngt = gaussian_sample(np.zeros(Ztr.shape[1]), np.cov(Ztr,rowvar=False), len(Ztr), MASTER_SEED+7000+r)
        ngh = gaussian_sample(np.zeros(Ztr.shape[1]), np.cov(Ztr,rowvar=False), len(Z), MASTER_SEED+8000+r)
        nk = KMeans(n_clusters=k, n_init=8, max_iter=400, random_state=MASTER_SEED+r)
        nk.fit(ngt)
        null_sils.append(sampled_silhouette(ngh, nk.predict(ngh), MASTER_SEED+r))
    null_sils = np.asarray(null_sils,dtype=float)
    pnull = float((1+np.sum(null_sils>=sil))/(len(null_sils)+1))

    hold_rows.append({
        "SCOPE":scope,
        "N_HOLDOUT":len(d),
        "SELECTED_K":k,
        "HOLDOUT_SILHOUETTE":sil,
        "GAUSSIAN_NULL_SILHOUETTE_MEAN":float(np.nanmean(null_sils)),
        "EXCESS_SILHOUETTE":float(sil-np.nanmean(null_sils)),
        "EMPIRICAL_NULL_P":pnull,
        "AUC_STRATIFIED_SHUFFLE_HOLDOUT":auc_sh,
        "AUC_COVARIANCE_MATCHED_GAUSSIAN_HOLDOUT":auc_ga,
    })

    assignment_frames[scope] = pd.DataFrame({
        "BLIND_ID":d["BLIND_ID"].astype(str).to_numpy(),
        "SCOPE":scope,
        "U_ID":[f"U{int(x)+1:03d}" for x in labels],
    })

hold_df = pd.DataFrame(hold_rows)
hold_df.to_csv(DIRS["hold"]/ "HARDENED_HOLDOUT_RESULTS.csv", index=False)

# Only after all target holdout scoring: open unblinded metadata.
train_meta = pd.read_parquet(TRAIN_PATH, columns=META_COLS)
hold_meta = pd.read_parquet(HOLD_PATH, columns=META_COLS)

audit_rows = []
for scope,assn in assignment_frames.items():
    hm = assn.merge(hold_meta, on="BLIND_ID", how="left", validate="one_to_one")
    for var in ["PROVIDER","MODEL","WINNER_RAW","PERSPECTIVE_LABEL","LANGUAGE","IS_CODE","ERA_2025_06_08"]:
        if hm[var].nunique(dropna=False) < 2:
            continue
        v,p,chi2,shape = cramer_v(hm["U_ID"], hm[var].fillna("MISSING").astype(str))
        nmi = normalized_mutual_info_score(hm["U_ID"].astype(str), hm[var].fillna("MISSING").astype(str))
        ami = adjusted_mutual_info_score(hm["U_ID"].astype(str), hm[var].fillna("MISSING").astype(str))
        audit_rows.append({
            "SCOPE":scope,"VARIABLE":var,"CRAMERS_V":v,"CHI2_P":p,"CHI2":chi2,
            "TABLE_ROWS":shape[0],"TABLE_COLS":shape[1],"NMI":float(nmi),"AMI":float(ami),
            "INTERPRETATION":"ASSOCIATION_MEASURE_ONLY_NOT_CAUSAL_CONFOUND"
        })
pd.DataFrame(audit_rows).to_csv(DIRS["audit"]/ "POSTFREEZE_CLUSTER_ASSOCIATION_AUDIT.csv", index=False)

# Direct continuous-feature provider predictability audit on ALL scope.
all_model = joblib.load(DIRS["models"]/ "ALL_FROZEN_MODEL.joblib")
tr_all,Etr,Xtr,Ztr,Ltr = transform_scope(train, all_model)
ho_all,Eho,Xho,Zho,Lho = transform_scope(hold, all_model)

tm = pd.DataFrame({"BLIND_ID":tr_all["BLIND_ID"].astype(str).to_numpy()}).merge(
    train_meta[["BLIND_ID","PROVIDER","IS_CODE"]].assign(BLIND_ID=lambda x:x.BLIND_ID.astype(str)),
    on="BLIND_ID", how="left", validate="one_to_one"
)
hm = pd.DataFrame({"BLIND_ID":ho_all["BLIND_ID"].astype(str).to_numpy()}).merge(
    hold_meta[["BLIND_ID","PROVIDER","IS_CODE"]].assign(BLIND_ID=lambda x:x.BLIND_ID.astype(str)),
    on="BLIND_ID", how="left", validate="one_to_one"
)

provider_clf = ExtraTreesClassifier(
    n_estimators=300, min_samples_leaf=5, max_features="sqrt",
    class_weight="balanced", random_state=MASTER_SEED, n_jobs=-1
)
provider_clf.fit(Xtr, tm["PROVIDER"].astype(str))
pred = provider_clf.predict(Xho)
prov_bal = float(balanced_accuracy_score(hm["PROVIDER"].astype(str), pred))
prov_f1 = float(f1_score(hm["PROVIDER"].astype(str), pred, average="macro"))
prov_classes = int(tm["PROVIDER"].nunique())
prov_chance = 1.0/prov_classes if prov_classes else np.nan

code_clf = ExtraTreesClassifier(
    n_estimators=250, min_samples_leaf=5, max_features="sqrt",
    class_weight="balanced", random_state=MASTER_SEED+1, n_jobs=-1
)
code_clf.fit(Xtr, tm["IS_CODE"].astype(int))
code_pred = code_clf.predict(Xho)
code_bal = float(balanced_accuracy_score(hm["IS_CODE"].astype(int), code_pred))

pd.DataFrame([{
    "PROVIDER_CLASSES_TRAIN":prov_classes,
    "PROVIDER_BALANCED_ACCURACY_HOLDOUT":prov_bal,
    "PROVIDER_MACRO_F1_HOLDOUT":prov_f1,
    "PROVIDER_BALANCED_CHANCE":prov_chance,
    "IS_CODE_BALANCED_ACCURACY_HOLDOUT":code_bal,
    "NOTE":"Direct continuous-feature audit; unlike R1 AMI, this tests predictability before U-cluster information loss."
}]).to_csv(DIRS["audit"]/ "DIRECT_CONTINUOUS_FEATURE_PREDICTABILITY.csv", index=False)

# No-silent-omission ledger.
ledger = pd.DataFrame([
    ["R1_COLUMN_SHUFFLE_GENERIC_DEPENDENCE","RERUN_HARDENED","Residualized features + nuisance-stratified shuffle"],
    ["COVARIANCE_MATCHED_NULL","RUN","Gaussian/covariance-matched null added"],
    ["CODE_STRATIFIED_ANALYSIS","RUN","ALL + CODE + NONCODE all retained"],
    ["PHYSICAL_HOLDOUT_SEAL","RUN","Holdout opened only after frozen models/spec hash"],
    ["EXECUTABLE_FREEZE","RUN","Residualizers/QT/PCA/KMeans/classifiers serialized and hashed"],
    ["PROVIDER_CLUSTER_ASSOCIATION","RUN","Cramer's V + AMI/NMI"],
    ["PROVIDER_CONTINUOUS_PREDICTABILITY","RUN","Direct holdout multiclass prediction"],
    ["MODEL_CLUSTER_ASSOCIATION","RUN","Cramer's V + AMI/NMI"],
    ["MODEL_CONTINUOUS_PREDICTABILITY","NOT_RUN","High-cardinality optional audit; explicitly recorded"],
    ["RSOS_SPECIFIC_BLIND_MAPPING","UNTESTED","Requires separate target-construct stage; R2 does not infer absence"],
    ["PLANTED_RSOS_TOPOLOGY_POSITIVES","UNTESTED","Requires sequence/graph extractor rather than aggregate formatting features"],
    ["F2_EXTERNAL_00086_RARITY","RESERVED_NOT_READ","External controls remain sealed"],
    ["F3_PROSPECTIVE_CONFIRMATION","NOT_ELIGIBLE","Historical LM Arena development data"],
], columns=["MEASUREMENT","STATUS","REASON"])
ledger.to_csv(DIRS["ledger"]/ "NO_SILENT_OMISSION_LEDGER.csv", index=False)

summary = {
    "version":VERSION,
    "completed_utc":utcnow(),
    "frozen_spec_sha256":freeze_hash,
    "r1_interpretation_correction":{
        "provider_negligible":"RETRACT_AS_INFERENCE",
        "model_negligible":"RETRACT_AS_INFERENCE",
        "winner_negligible":"RETRACT_AS_INFERENCE",
        "language_negligible":"RETRACT_AS_INFERENCE",
        "era_negligible":"RETRACT_AS_INFERENCE",
        "is_code_material_confound":"RECLASSIFY_AS_UNRESOLVED_ASSOCIATION",
        "weak_moderate_cluster_separation":"DESCRIPTIVE_KMEANS_ONLY_NOT_RSOS_EVIDENCE",
        "graph_stability_extremely_stable":"DESCRIPTIVE_SAME_SOURCE_STABILITY_NOT_RSOS_TOPOLOGY",
    },
    "holdout_results":hold_rows,
    "provider_direct_audit":{
        "balanced_accuracy":prov_bal,
        "macro_f1":prov_f1,
        "balanced_chance":prov_chance,
    },
    "is_code_direct_audit":{
        "balanced_accuracy":code_bal,
    },
    "rsos_specific_status":"UNTESTED",
    "external_00086_read":False,
}
(DIRS["manifest"]/ "R2_SUMMARY.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

text = [
    "F1-DEV R2 HARDENED METHODOLOGY CORRECTION",
    "",
    "R1 negative/absence interpretations are retracted as inferential claims.",
    "R1 raw measurements remain preserved as historical diagnostics.",
    "",
    "HOLDOUT RESULTS",
    hold_df.to_string(index=False),
    "",
    "DIRECT PROVIDER / CODE PREDICTABILITY",
    f"Provider balanced accuracy: {prov_bal:.6f}",
    f"Provider macro F1: {prov_f1:.6f}",
    f"Provider balanced chance: {prov_chance:.6f}",
    f"IS_CODE balanced accuracy: {code_bal:.6f}",
    "",
    "RSOS-SPECIFIC STATUS: UNTESTED",
    "00086 EXTERNAL CONTROLS READ: NO",
    f"FROZEN SPEC SHA256: {freeze_hash}",
]
(DIRS["manifest"]/ "R2_SUMMARY.txt").write_text("\n".join(text)+"\n",encoding="utf-8")

COMPLETE.write_text(f"{VERSION}\n{utcnow()}\n{freeze_hash}\n", encoding="ascii")

print()
print("="*72)
print("F1-DEV R2 HARDENED COMPLETE")
print("="*72)
print(hold_df.to_string(index=False))
print()
print(f"Provider balanced accuracy: {prov_bal:.6f} (chance {prov_chance:.6f})")
print(f"Provider macro F1:          {prov_f1:.6f}")
print(f"IS_CODE balanced accuracy: {code_bal:.6f}")
print()
print("RSOS-specific status: UNTESTED")
print("Reserved 00086 controls read: NO")
print("Frozen spec SHA256:", freeze_hash)
print("Results:", OUT)
