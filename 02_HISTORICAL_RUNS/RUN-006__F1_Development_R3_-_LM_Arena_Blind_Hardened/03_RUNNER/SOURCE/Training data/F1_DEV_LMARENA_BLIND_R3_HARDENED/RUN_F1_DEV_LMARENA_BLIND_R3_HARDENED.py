from pathlib import Path
import json, hashlib, math, itertools
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
import networkx as nx

from scipy.stats import chi2_contingency
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, f1_score,
    silhouette_score, adjusted_mutual_info_score,
    normalized_mutual_info_score
)
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

BASE = Path(r"C:\RSOS\Training data\LMARENA_V12_V18_UPDATED")
OUT  = Path(r"C:\RSOS\Training data\F1_DEV_LMARENA_BLIND_R3_HARDENED")

TRAIN_PATH = BASE / r"06_DEVELOPMENT_SPLITS\DEVELOPMENT_TRAIN\train-00000-of-00007__DEVELOPMENT_TRAIN.parquet"
VAL_PATH   = BASE / r"06_DEVELOPMENT_SPLITS\DEVELOPMENT_VALIDATION\train-00000-of-00007__DEVELOPMENT_VALIDATION.parquet"
HOLD_PATH  = BASE / r"06_DEVELOPMENT_SPLITS\DEVELOPMENT_HOLDOUT\train-00000-of-00007__DEVELOPMENT_HOLDOUT.parquet"

MASTER_SEED = 120918
VERSION = "F1_DEV_LMARENA_BLIND_R3_HARDENED"

DIRS = {
    "manifest": OUT / "00_MANIFEST",
    "cal": OUT / "01_CALIBRATION",
    "nulls": OUT / "02_MATCHED_NULLS",
    "cluster": OUT / "03_CLUSTER_CURVES_NO_FORCED_K",
    "hold": OUT / "04_SEALED_HOLDOUT",
    "audit": OUT / "05_POSTFREEZE_NUISANCE_AUDIT",
    "plant": OUT / "06_PLANTED_V12_GRAPH_CALIBRATION",
    "ledger": OUT / "07_NO_SILENT_OMISSION",
    "models": OUT / "08_FROZEN_MODELS",
}
for p in DIRS.values():
    p.mkdir(parents=True, exist_ok=True)

COMPLETE = OUT / "F1_DEV_R3_COMPLETE.marker"
if COMPLETE.exists():
    raise SystemExit(f"Completed R3 already exists and will not be overwritten:\n{OUT}")

COLS = [
    "BLIND_ID","RECORD_ID","PAIR_ID","EVALUATION_SESSION_ID",
    "LANGUAGE","IS_CODE","ERA_2025_06_08","MODEL","PROVIDER",
    "WINNER_RAW","PERSPECTIVE_LABEL",
    "N_TURNS","TOTAL_CHARS","USER_CHARS","WORD_COUNT",
    "TYPE_TOKEN_RATIO","MEAN_TURN_CHARS","MAX_TURN_CHARS",
    "PARAGRAPH_COUNT","CODE_FENCE_COUNT","MARKDOWN_HEADER_COUNT",
    "LIST_ITEM_COUNT","COLON_COUNT","BRACKET_COUNT",
    "UPPERCASE_RATIO","DIGIT_RATIO","COMPRESSION_RATIO",
]

TARGET_FEATURES = [
    "TYPE_TOKEN_RATIO","UPPERCASE_RATIO","DIGIT_RATIO","COMPRESSION_RATIO",
    "USER_CHAR_SHARE","MEAN_MAX_TURN_RATIO","LOG_WORD_COUNT",
    "LOG_PARAGRAPH_COUNT","LOG_CODE_FENCE_COUNT","LOG_HEADER_COUNT",
    "LOG_LIST_ITEM_COUNT","LOG_COLON_COUNT","LOG_BRACKET_COUNT",
]

def utcnow():
    return datetime.now(timezone.utc).isoformat()

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def safe_ratio(a,b):
    a = pd.to_numeric(a, errors="coerce").astype(float)
    b = pd.to_numeric(b, errors="coerce").astype(float)
    return a / b.replace(0,np.nan)

def features(df):
    x = pd.DataFrame(index=df.index)
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
    return x.replace([np.inf,-np.inf],np.nan)

def nuisance_frame(df, top_languages):
    n = pd.DataFrame(index=df.index)
    n["IS_CODE"] = df["IS_CODE"].astype(bool).astype(int)
    lang = df["LANGUAGE"].fillna("MISSING").astype(str)
    n["LANG_GROUP"] = np.where(lang.isin(top_languages), lang, "OTHER")
    n["LOG_TOTAL_CHARS"] = np.log1p(pd.to_numeric(df["TOTAL_CHARS"], errors="coerce").clip(lower=0))
    n["LOG_N_TURNS"] = np.log1p(pd.to_numeric(df["N_TURNS"], errors="coerce").clip(lower=0))
    return n

def qedges(a, q):
    a = np.asarray(a,dtype=float)
    a = a[np.isfinite(a)]
    e = np.unique(np.quantile(a,np.linspace(0,1,q+1)))
    if len(e)<2:
        return np.array([-np.inf,np.inf])
    e[0],e[-1] = -np.inf,np.inf
    return e

def make_strata(n, char_edges, turn_edges):
    c = np.digitize(n["LOG_TOTAL_CHARS"].to_numpy(float), char_edges[1:-1])
    t = np.digitize(n["LOG_N_TURNS"].to_numpy(float), turn_edges[1:-1])
    return np.array([
        f"{code}|{lang}|{ci}|{ti}"
        for code,lang,ci,ti in zip(n["IS_CODE"],n["LANG_GROUP"],c,t)
    ],dtype=object)

def stratified_shuffle(X,strata,seed):
    rng = np.random.default_rng(seed)
    X = np.asarray(X,float)
    Y = X.copy()
    strata = np.asarray(strata)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata==s)
        if len(idx)<4:
            continue
        for j in range(X.shape[1]):
            Y[idx,j] = X[rng.permutation(idx),j]
    return Y

def gaussian_sample(mu,cov,n,seed):
    rng=np.random.default_rng(seed)
    cov=np.asarray(cov,float)
    cov=(cov+cov.T)/2 + np.eye(cov.shape[0])*1e-6
    return rng.multivariate_normal(mu,cov,size=n)

def fit_real_null_classifier(Xtr,Ntr,seed):
    y=np.r_[np.ones(len(Xtr),dtype=int),np.zeros(len(Ntr),dtype=int)]
    X=np.vstack([Xtr,Ntr])
    clf=ExtraTreesClassifier(
        n_estimators=350,min_samples_leaf=5,max_features="sqrt",
        class_weight="balanced",random_state=seed,n_jobs=-1
    )
    clf.fit(X,y)
    return clf

def eval_real_null(clf,X,N):
    y=np.r_[np.ones(len(X),dtype=int),np.zeros(len(N),dtype=int)]
    P=clf.predict_proba(np.vstack([X,N]))[:,1]
    return float(roc_auc_score(y,P))

def sampled_silhouette(Z,labels,seed):
    if len(np.unique(labels))<2:
        return np.nan
    return float(silhouette_score(
        Z,labels,sample_size=min(2500,len(Z)),random_state=seed
    ))

def cramer_v(x,y):
    tab=pd.crosstab(pd.Series(x,dtype="string"),pd.Series(y,dtype="string"))
    if tab.shape[0]<2 or tab.shape[1]<2:
        return np.nan,np.nan,np.nan
    chi2,p,dof,exp=chi2_contingency(tab.values)
    n=tab.values.sum()
    v=math.sqrt(chi2/(n*min(tab.shape[0]-1,tab.shape[1]-1)))
    return float(v),float(p),float(chi2)

def metadata_predict(trainX,holdX,trainy,holdy,seed):
    tr=pd.Series(trainy).fillna("MISSING").astype(str)
    ho=pd.Series(holdy).fillna("MISSING").astype(str)
    valid=set(tr.unique())
    mask=ho.isin(valid).to_numpy()
    if tr.nunique()<2 or mask.sum()==0:
        return np.nan,np.nan,int(tr.nunique()),np.nan
    clf=ExtraTreesClassifier(
        n_estimators=300,min_samples_leaf=5,max_features="sqrt",
        class_weight="balanced",random_state=seed,n_jobs=-1
    )
    clf.fit(trainX,tr)
    pred=clf.predict(holdX[mask])
    truth=ho.to_numpy()[mask]
    bal=float(balanced_accuracy_score(truth,pred))
    macro=float(f1_score(truth,pred,average="macro"))
    chance=1.0/tr.nunique()
    return bal,macro,int(tr.nunique()),float(chance)

print("="*78)
print("F1-DEV R3 HARDENED")
print("="*78)
print("NO NEGATIVE/ABSENCE INFERENCE IS PERMITTED FROM UNTESTED OR OMITTED TARGETS.")
print("Reserved 00086 external controls are not accessed.")

# ---------------- PRE-HOLDOUT ONLY ----------------
for p in [TRAIN_PATH,VAL_PATH,HOLD_PATH]:
    if not p.exists():
        raise RuntimeError(f"Missing split: {p}")

train=pd.read_parquet(TRAIN_PATH,columns=COLS)
val=pd.read_parquet(VAL_PATH,columns=COLS)

if set(train["EVALUATION_SESSION_ID"].astype(str)) & set(val["EVALUATION_SESSION_ID"].astype(str)):
    raise RuntimeError("Train/validation session leakage.")

top_languages=train["LANGUAGE"].fillna("MISSING").astype(str).value_counts().head(12).index.tolist()

Ft=features(train)
Fv=features(val)
med=Ft.median()
Ft=Ft.fillna(med)
Fv=Fv.fillna(med)

# No residualization. Null matching is done by nuisance strata.
qt=QuantileTransformer(
    n_quantiles=min(1000,len(Ft)),output_distribution="normal",
    subsample=None,random_state=MASTER_SEED
)
Xt=qt.fit_transform(Ft)
Xv=qt.transform(Fv)

Nt=nuisance_frame(train,top_languages)
Nv=nuisance_frame(val,top_languages)
char_edges=qedges(Nt["LOG_TOTAL_CHARS"],5)
turn_edges=qedges(Nt["LOG_N_TURNS"],4)
st=make_strata(Nt,char_edges,turn_edges)
sv=make_strata(Nv,char_edges,turn_edges)

sh_t=stratified_shuffle(Xt,st,MASTER_SEED+10)
sh_v=stratified_shuffle(Xv,sv,MASTER_SEED+11)

mu=Xt.mean(0)
cov=np.cov(Xt,rowvar=False)
ga_t=gaussian_sample(mu,cov,len(Xt),MASTER_SEED+20)
ga_v=gaussian_sample(mu,cov,len(Xv),MASTER_SEED+21)

clf_shuffle=fit_real_null_classifier(Xt,sh_t,MASTER_SEED+30)
clf_gauss=fit_real_null_classifier(Xt,ga_t,MASTER_SEED+31)

val_auc_shuffle=eval_real_null(clf_shuffle,Xv,sh_v)
val_auc_gauss=eval_real_null(clf_gauss,Xv,ga_v)

pca=PCA(n_components=0.90,svd_solver="full")
Zt=pca.fit_transform(Xt)
Zv=pca.transform(Xv)

# Cluster curve only; no K is declared "natural".
curve=[]
for k in range(2,16):
    km=KMeans(n_clusters=k,n_init=20,max_iter=500,random_state=MASTER_SEED)
    km.fit(Zt)
    obs=sampled_silhouette(Zv,km.predict(Zv),MASTER_SEED+k)

    nullvals=[]
    # Matched Gaussian null in PCA space; 49 draws, no significance claim.
    zmu=Zt.mean(0)
    zcov=np.cov(Zt,rowvar=False)
    for r in range(49):
        ngt=gaussian_sample(zmu,zcov,min(6000,len(Zt)),MASTER_SEED+10000+k*100+r)
        ngv=gaussian_sample(zmu,zcov,min(3000,len(Zv)),MASTER_SEED+20000+k*100+r)
        nk=KMeans(n_clusters=k,n_init=5,max_iter=300,random_state=MASTER_SEED+r)
        nk.fit(ngt)
        nullvals.append(sampled_silhouette(ngv,nk.predict(ngv),MASTER_SEED+r))
    nullvals=np.asarray(nullvals,float)
    curve.append({
        "K":k,
        "VALIDATION_SILHOUETTE":obs,
        "NULL_MEAN":float(np.nanmean(nullvals)),
        "NULL_95":float(np.nanquantile(nullvals,.95)),
        "EXCESS":float(obs-np.nanmean(nullvals)),
        "ABOVE_NULL_95":bool(obs>np.nanquantile(nullvals,.95)),
    })

curve_df=pd.DataFrame(curve)
curve_df.to_csv(DIRS["cluster"]/ "VALIDATION_CLUSTER_CURVE_NO_FORCED_K.csv",index=False)

# Freeze exact executable pipeline BEFORE semantic holdout access.
pipeline={
    "top_languages":top_languages,
    "median":med.to_dict(),
    "qt":qt,
    "pca":pca,
    "char_edges":char_edges,
    "turn_edges":turn_edges,
    "shuffle_classifier":clf_shuffle,
    "gaussian_classifier":clf_gauss,
    "mu":mu,
    "cov":cov,
    "feature_names":TARGET_FEATURES,
}
pipe_path=DIRS["models"]/ "FROZEN_PIPELINE.joblib"
joblib.dump(pipeline,pipe_path,compress=3)

freeze={
    "version":VERSION,
    "frozen_utc":utcnow(),
    "train_sha256":sha256_file(TRAIN_PATH),
    "validation_sha256":sha256_file(VAL_PATH),
    "holdout_filename_only":HOLD_PATH.name,
    "holdout_not_semantically_opened":True,
    "pipeline_sha256":sha256_file(pipe_path),
    "validation_auc_stratified_shuffle":val_auc_shuffle,
    "validation_auc_covariance_gaussian":val_auc_gauss,
    "cluster_policy":"NO_NATURAL_K_CLAIM. Entire K=2..15 curve is preserved. Boundary maxima are not selected.",
    "nuisance_matching":"IS_CODE + top-12 language group + total-char quintile + turn quartile",
    "negative_inference_rule":"UNTESTED, UNAVAILABLE, OMITTED, OR UNMEASURED TARGETS MAY NOT BE INTERPRETED AS ABSENCE OR FAILURE.",
    "external_control_rule":"train-*-of-00086.parquet is reserved and unread.",
}
freeze_path=DIRS["manifest"]/ "R3_FROZEN_SPEC_BEFORE_HOLDOUT.json"
freeze_path.write_text(json.dumps(freeze,indent=2),encoding="utf-8")
freeze_hash=sha256_file(freeze_path)
(DIRS["manifest"]/ "R3_FROZEN_SPEC_SHA256.txt").write_text(freeze_hash+"\n",encoding="ascii")

print("FROZEN BEFORE HOLDOUT:",freeze_hash)

# ---------------- HOLDOUT OPENED AFTER FREEZE ----------------
hold_hash=sha256_file(HOLD_PATH)
hold=pd.read_parquet(HOLD_PATH,columns=COLS)

for a,b,name in [
    (train,hold,"train/holdout"),
    (val,hold,"validation/holdout")
]:
    if set(a["EVALUATION_SESSION_ID"].astype(str)) & set(b["EVALUATION_SESSION_ID"].astype(str)):
        raise RuntimeError(f"Session leakage: {name}")

Fh=features(hold).fillna(med)
Xh=qt.transform(Fh)
Zh=pca.transform(Xh)

Nh=nuisance_frame(hold,top_languages)
sh_h=stratified_shuffle(Xh,make_strata(Nh,char_edges,turn_edges),MASTER_SEED+12)
ga_h=gaussian_sample(mu,cov,len(Xh),MASTER_SEED+22)

hold_auc_shuffle=eval_real_null(clf_shuffle,Xh,sh_h)
hold_auc_gauss=eval_real_null(clf_gauss,Xh,ga_h)

pd.DataFrame([
    {
        "NULL":"NUISANCE_STRATIFIED_COLUMN_SHUFFLE",
        "VALIDATION_AUC":val_auc_shuffle,
        "HOLDOUT_AUC":hold_auc_shuffle,
        "INTERPRETATION":"GENERIC_MULTIVARIATE_DEPENDENCE_ONLY_NOT_RSOS_SPECIFIC",
    },
    {
        "NULL":"COVARIANCE_MATCHED_GAUSSIAN",
        "VALIDATION_AUC":val_auc_gauss,
        "HOLDOUT_AUC":hold_auc_gauss,
        "INTERPRETATION":"HIGHER_ORDER_NON_GAUSSIAN_STRUCTURE_ONLY_NOT_RSOS_SPECIFIC",
    }
]).to_csv(DIRS["nulls"]/ "HARDENED_NULL_RESULTS.csv",index=False)

# Holdout cluster curve; still no forced K.
hcurve=[]
zmu=Zt.mean(0)
zcov=np.cov(Zt,rowvar=False)
for k in range(2,16):
    km=KMeans(n_clusters=k,n_init=20,max_iter=500,random_state=MASTER_SEED)
    km.fit(Zt)
    obs=sampled_silhouette(Zh,km.predict(Zh),MASTER_SEED+100+k)
    nullvals=[]
    for r in range(49):
        ngt=gaussian_sample(zmu,zcov,min(6000,len(Zt)),MASTER_SEED+30000+k*100+r)
        ngh=gaussian_sample(zmu,zcov,min(3000,len(Zh)),MASTER_SEED+40000+k*100+r)
        nk=KMeans(n_clusters=k,n_init=5,max_iter=300,random_state=MASTER_SEED+r)
        nk.fit(ngt)
        nullvals.append(sampled_silhouette(ngh,nk.predict(ngh),MASTER_SEED+r))
    nullvals=np.asarray(nullvals,float)
    hcurve.append({
        "K":k,"HOLDOUT_SILHOUETTE":obs,
        "NULL_MEAN":float(np.nanmean(nullvals)),
        "NULL_95":float(np.nanquantile(nullvals,.95)),
        "EXCESS":float(obs-np.nanmean(nullvals)),
        "ABOVE_NULL_95":bool(obs>np.nanquantile(nullvals,.95)),
    })
pd.DataFrame(hcurve).to_csv(DIRS["cluster"]/ "HOLDOUT_CLUSTER_CURVE_NO_FORCED_K.csv",index=False)

# ---------------- COMPLETE NUISANCE PREDICTABILITY AUDIT ----------------
audit=[]
for i,var in enumerate(["PROVIDER","MODEL","LANGUAGE","IS_CODE","WINNER_RAW","PERSPECTIVE_LABEL","ERA_2025_06_08"]):
    bal,macro,nclass,chance=metadata_predict(
        Xt,Xh,train[var],hold[var],MASTER_SEED+500+i
    )
    audit.append({
        "VARIABLE":var,
        "TRAIN_CLASSES":nclass,
        "BALANCED_CHANCE":chance,
        "HOLDOUT_BALANCED_ACCURACY":bal,
        "HOLDOUT_MACRO_F1":macro,
        "INTERPRETATION":"PREDICTABILITY_DIAGNOSTIC_ONLY; NOT AUTOMATICALLY A CAUSAL CONFOUND",
    })
audit_df=pd.DataFrame(audit)
audit_df.to_csv(DIRS["audit"]/ "COMPLETE_CONTINUOUS_FEATURE_NUISANCE_PREDICTABILITY.csv",index=False)

# Within-provider generic dependence: post-freeze diagnostic.
prov_rows=[]
providers=train["PROVIDER"].fillna("MISSING").astype(str).value_counts()
for prov,ntr in providers.items():
    trmask=train["PROVIDER"].fillna("MISSING").astype(str).eq(prov).to_numpy()
    hmask=hold["PROVIDER"].fillna("MISSING").astype(str).eq(prov).to_numpy()
    if trmask.sum()<300 or hmask.sum()<100:
        continue
    # Train provider-specific classifier from same frozen feature representation.
    # Diagnostic only; constructed after main freeze and cannot alter main result.
    pNt=nuisance_frame(train.loc[trmask],top_languages)
    pNh=nuisance_frame(hold.loc[hmask],top_languages)
    pst=make_strata(pNt,char_edges,turn_edges)
    psh=make_strata(pNh,char_edges,turn_edges)
    null_tr=stratified_shuffle(Xt[trmask],pst,MASTER_SEED+700)
    null_h=stratified_shuffle(Xh[hmask],psh,MASTER_SEED+701)
    pc=fit_real_null_classifier(Xt[trmask],null_tr,MASTER_SEED+702)
    auc=eval_real_null(pc,Xh[hmask],null_h)
    prov_rows.append({"PROVIDER":prov,"N_TRAIN":int(trmask.sum()),"N_HOLDOUT":int(hmask.sum()),"HOLDOUT_AUC":auc})
pd.DataFrame(prov_rows).to_csv(DIRS["audit"]/ "WITHIN_PROVIDER_GENERIC_DEPENDENCE.csv",index=False)

# ---------------- PLANTED V12 GRAPH CALIBRATION ----------------
AUTH_EDGES=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(3,1),(5,2),(6,3)]

def graph_features(edges):
    G=nx.DiGraph()
    G.add_nodes_from(range(8))
    G.add_edges_from(edges)
    A=nx.to_numpy_array(G,nodelist=range(8),dtype=float)
    eig=np.linalg.eigvals(A)
    scc=list(nx.strongly_connected_components(G))
    reach=[]
    paths=[]
    for a in G.nodes:
        for b in G.nodes:
            if a==b: continue
            if nx.has_path(G,a,b):
                reach.append(1)
                try: paths.append(nx.shortest_path_length(G,a,b))
                except: pass
            else: reach.append(0)
    two=sum(1 for a,b in G.edges if G.has_edge(b,a))//2
    three=0
    for cyc in nx.simple_cycles(G):
        if len(cyc)==3: three+=1
    three//=3 if three else 1
    return [
        G.number_of_edges(),
        float(np.std([d for _,d in G.in_degree()])),
        float(np.std([d for _,d in G.out_degree()])),
        len(scc),
        max(map(len,scc))/8.0,
        float(nx.transitivity(G.to_undirected())),
        float(np.max(np.abs(eig))),
        two,three,
        float(np.mean(reach)),
        float(np.mean(paths)) if paths else 0.0,
    ]

def relabel_edges(edges,perm):
    mp={i:int(perm[i]) for i in range(8)}
    return [(mp[a],mp[b]) for a,b in edges]

def degree_rewire(edges,rng,steps=8):
    e=set(edges)
    for _ in range(steps*20):
        if steps<=0: break
        a_b,c_d=rng.choice(len(e),size=2,replace=False)
        el=list(e)
        a,b=el[a_b]; c,d=el[c_d]
        if len({a,b,c,d})<4: continue
        cand1=(a,d); cand2=(c,b)
        if a==d or c==b or cand1 in e or cand2 in e: continue
        e.remove((a,b)); e.remove((c,d))
        e.add(cand1); e.add(cand2)
        steps-=1
    return list(e)

rng=np.random.default_rng(MASTER_SEED)
rows=[]
for rep in range(1500):
    perm=rng.permutation(8)
    pos=relabel_edges(AUTH_EDGES,perm)
    rows.append(["AUTH_ISOMORPHIC",1,*graph_features(pos)])

    rew=degree_rewire(AUTH_EDGES,rng,steps=5)
    rows.append(["DEGREE_REWIRE",0,*graph_features(rew)])

    i=int(rng.integers(0,10))
    dele=[e for j,e in enumerate(AUTH_EDGES) if j!=i]
    rows.append(["EDGE_DELETE",0,*graph_features(dele)])

    i=int(rng.integers(0,10))
    rev=list(AUTH_EDGES)
    a,b=rev[i]
    rev[i]=(b,a)
    rows.append(["EDGE_REVERSE",0,*graph_features(rev)])

gcols=["CONDITION","LABEL","EDGE_COUNT","IN_SD","OUT_SD","SCC_COUNT","LARGEST_SCC_FRAC",
       "TRANSITIVITY","SPECTRAL_RADIUS","TWO_CYCLES","THREE_CYCLES","REACHABILITY_FRAC","MEAN_REACHABLE_PATH"]
gdf=pd.DataFrame(rows,columns=gcols)
gx=gdf[gcols[2:]].to_numpy(float)
gy=gdf["LABEL"].to_numpy(int)
idx=np.arange(len(gdf))
tr_idx,te_idx=train_test_split(idx,test_size=.3,random_state=MASTER_SEED,stratify=gy)
gclf=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=3,max_features="sqrt",random_state=MASTER_SEED,n_jobs=-1)
gclf.fit(gx[tr_idx],gy[tr_idx])
gauc=float(roc_auc_score(gy[te_idx],gclf.predict_proba(gx[te_idx])[:,1]))
gdf.to_csv(DIRS["plant"]/ "PLANTED_V12_GRAPH_DATA.csv",index=False)
pd.DataFrame([{
    "TEST":"PLANTED_V12_GRAPH_AUTHENTIC_VS_LESION_REWIRE",
    "HOLDOUT_AUC":gauc,
    "INTERPRETATION":"MEASUREMENT-ENGINE CALIBRATION ONLY; DOES NOT SHOW V12 TOPOLOGY IN LM ARENA"
}]).to_csv(DIRS["plant"]/ "PLANTED_V12_GRAPH_CALIBRATION_RESULT.csv",index=False)

# ---------------- NO-SILENT-OMISSION LEDGER ----------------
ledger=pd.DataFrame([
    ["GENERIC_MULTIVARIATE_DEPENDENCE","RUN","NUISANCE_MATCHED","No RSOS-specific interpretation"],
    ["HIGHER_ORDER_NON_GAUSSIAN_STRUCTURE","RUN","COVARIANCE_MATCHED_NULL","No RSOS-specific interpretation"],
    ["CODE_NUISANCE","RUN","DIRECT_PREDICTABILITY_AND_MATCHED_NULL","Association only"],
    ["LANGUAGE_NUISANCE","RUN","DIRECT_PREDICTABILITY_AND_MATCHED_NULL","Association only"],
    ["PROVIDER_NUISANCE","RUN","DIRECT_PREDICTABILITY_PLUS_WITHIN_PROVIDER","Association only"],
    ["MODEL_NUISANCE","RUN","DIRECT_PREDICTABILITY","Association only"],
    ["WINNER_NUISANCE","RUN","DIRECT_PREDICTABILITY","Association only"],
    ["ERA_NUISANCE","RUN","DIRECT_PREDICTABILITY","Association only"],
    ["CLUSTER_K","RUN_NO_FORCED_SELECTION","K2_TO_K15_CURVES","No natural-K claim"],
    ["PLANTED_V12_GRAPH_CALIBRATION","RUN","SYNTHETIC_POSITIVE_NEGATIVE","Calibration only"],
    ["RSOS_SPECIFIC_BLIND_MAPPING","UNTESTED","REQUIRES_TARGET_CONSTRUCT_STAGE","NO NEGATIVE INFERENCE"],
    ["REAL_RSOS_CORPUS_COMPARISON","UNTESTED","REQUIRES_RSOS_CORPUS_FEATURE_EXTRACTION","NO NEGATIVE INFERENCE"],
    ["F2_EXTERNAL_00086_RARITY","RESERVED_NOT_READ","SEALED_FOR_F2","NO NEGATIVE INFERENCE"],
    ["F3_PROSPECTIVE_REPLICATION","NOT_ELIGIBLE","HISTORICAL_DATA","NO NEGATIVE INFERENCE"],
],columns=["MEASUREMENT","STATUS","METHOD_OR_REASON","INFERENCE_RULE"])
ledger.to_csv(DIRS["ledger"]/ "NO_SILENT_OMISSION_LEDGER.csv",index=False)

# Summary
curve_v=pd.read_csv(DIRS["cluster"]/ "VALIDATION_CLUSTER_CURVE_NO_FORCED_K.csv")
curve_h=pd.read_csv(DIRS["cluster"]/ "HOLDOUT_CLUSTER_CURVE_NO_FORCED_K.csv")
summary={
    "version":VERSION,
    "completed_utc":utcnow(),
    "frozen_spec_sha256":freeze_hash,
    "holdout_sha256_after_freeze":hold_hash,
    "generic_dependence":{
        "stratified_shuffle_validation_auc":val_auc_shuffle,
        "stratified_shuffle_holdout_auc":hold_auc_shuffle,
        "covariance_gaussian_validation_auc":val_auc_gauss,
        "covariance_gaussian_holdout_auc":hold_auc_gauss,
    },
    "cluster_policy":"NO_FORCED_K",
    "validation_K_above_null95":curve_v.loc[curve_v["ABOVE_NULL_95"],"K"].astype(int).tolist(),
    "holdout_K_above_null95":curve_h.loc[curve_h["ABOVE_NULL_95"],"K"].astype(int).tolist(),
    "planted_v12_graph_calibration_auc":gauc,
    "metadata_predictability":audit_df.to_dict(orient="records"),
    "rsos_specific_status":"UNTESTED",
    "negative_absence_inference_from_omission":"PROHIBITED_BY_METHOD",
    "external_00086_read":False,
}
(DIRS["manifest"]/ "R3_SUMMARY.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

lines=[
    "F1-DEV R3 HARDENED SUMMARY","",
    f"Frozen spec SHA256: {freeze_hash}",
    f"Holdout SHA256 (computed only after freeze): {hold_hash}","",
    f"Nuisance-stratified shuffle AUC: validation={val_auc_shuffle:.6f} holdout={hold_auc_shuffle:.6f}",
    f"Covariance-matched Gaussian AUC: validation={val_auc_gauss:.6f} holdout={hold_auc_gauss:.6f}",
    f"Planted V12 graph calibration AUC: {gauc:.6f}","",
    "CLUSTER POLICY: NO FORCED K",
    "Validation K above null95: "+str(summary["validation_K_above_null95"]),
    "Holdout K above null95: "+str(summary["holdout_K_above_null95"]),"",
    "RSOS-SPECIFIC STATUS: UNTESTED",
    "NEGATIVE/ABSENCE INFERENCE FROM OMITTED OR UNTESTED DATA: NOT PERMITTED",
    "00086 EXTERNAL CONTROLS READ: NO","",
    "DIRECT NUISANCE PREDICTABILITY:",
    audit_df.to_string(index=False),
]
(DIRS["manifest"]/ "R3_SUMMARY.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")

# Output provenance
files=[]
for q in sorted(x for x in OUT.rglob("*") if x.is_file()):
    files.append({
        "RELATIVE_PATH":str(q.relative_to(OUT)),
        "BYTES":q.stat().st_size,
        "SHA256":sha256_file(q),
    })
pd.DataFrame(files).to_csv(DIRS["manifest"]/ "OUTPUT_SHA256_MANIFEST.csv",index=False)

COMPLETE.write_text(f"{VERSION}\n{utcnow()}\n{freeze_hash}\n",encoding="ascii")

print()
print("="*78)
print("F1-DEV R3 HARDENED COMPLETE")
print("="*78)
print(f"Nuisance-stratified AUC: validation={val_auc_shuffle:.6f} holdout={hold_auc_shuffle:.6f}")
print(f"Covariance-Gaussian AUC: validation={val_auc_gauss:.6f} holdout={hold_auc_gauss:.6f}")
print(f"Planted V12 graph calibration AUC: {gauc:.6f}")
print("RSOS-specific status: UNTESTED")
print("Negative inference from omitted/unmeasured targets: NOT PERMITTED")
print("Reserved 00086 controls read: NO")
print("Frozen spec:",freeze_hash)
print("Results:",OUT)
