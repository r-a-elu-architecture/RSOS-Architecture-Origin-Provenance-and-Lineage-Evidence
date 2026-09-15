from pathlib import Path
import json, hashlib, math, os, platform, sys, zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score

ROOT = Path(r"C:\RSOS\Training data\F1_BR2_LMARENA_MATCHED_CONTROL")
INPUTS = ROOT / "00_INPUTS"
DIRS = {
    "manifest": ROOT / "00_MANIFEST",
    "br1": ROOT / "01_BR1_FROZEN",
    "schema": ROOT / "02_SCHEMA",
    "lm": ROOT / "03_LMARENA_PREP",
    "matching": ROOT / "04_MATCHING",
    "results": ROOT / "05_RESULTS",
    "robust": ROOT / "06_ROBUSTNESS",
    "ledger": ROOT / "07_LEDGER",
}
for p in DIRS.values():
    p.mkdir(parents=True, exist_ok=True)

MARKER = ROOT / "F1_BR2_COMPLETE.marker"
if MARKER.exists():
    raise SystemExit(f"BR2 already completed and will not be overwritten:\n{ROOT}")

SEED = 120918
rng = np.random.default_rng(SEED)
VERSION = "F1_BR2_LMARENA_MATCHED_CONTROL"

REQ = [
    "F1_BR1_BLIND_RSOS_CORPUS_REDISCOVERY_RESULTS_PACKAGE.zip",
    "train-00000-of-00007__V12_V18_PAIRS.parquet",
    "train-00000-of-00007__V12_V18_ARMS.parquet",
    "train-00000-of-00007__BLIND_STRUCTURAL_MATRIX.parquet",
    "LMARENA_SEARCH_INDEX.parquet",
    "LMARENA_SEARCH_INDEX.csv",
    "train-00000-of-00007__DEVELOPMENT_TRAIN.parquet",
    "train-00000-of-00007__DEVELOPMENT_VALIDATION.parquet",
    "train-00000-of-00007__DEVELOPMENT_HOLDOUT.parquet",
]

R1_FEATURES = [
    "TYPE_TOKEN_RATIO","UPPERCASE_RATIO","DIGIT_RATIO","COMPRESSION_RATIO",
    "USER_CHAR_SHARE","MEAN_MAX_TURN_RATIO","WORDS_PER_1K_CHAR","TURNS_PER_1K_CHAR",
    "PARAGRAPHS_PER_TURN","CODE_FENCES_PER_1K_CHAR","HEADERS_PER_1K_CHAR",
    "LIST_ITEMS_PER_1K_CHAR","COLONS_PER_1K_CHAR","BRACKETS_PER_1K_CHAR"
]

def now():
    return datetime.now(timezone.utc).isoformat()

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def write_json(p,obj):
    p.write_text(json.dumps(obj,indent=2,default=str),encoding="utf-8")

def find_col(df, names):
    cmap={str(c).upper():c for c in df.columns}
    for n in names:
        if n.upper() in cmap:
            return cmap[n.upper()]
    return None

def all_feature_cols(df, names):
    cmap={str(c).upper():c for c in df.columns}
    out=[]
    missing=[]
    for n in names:
        c=cmap.get(n.upper())
        if c is None:
            missing.append(n)
        else:
            out.append(c)
    return out,missing

def auc_ci(y,s,nboot=5000):
    y=np.asarray(y,int); s=np.asarray(s,float)
    auc=roc_auc_score(y,s)
    vals=[]
    for _ in range(nboot):
        idx=rng.integers(0,len(y),len(y))
        yy=y[idx]
        if len(np.unique(yy))<2:
            continue
        vals.append(roc_auc_score(yy,s[idx]))
    lo,hi=np.quantile(vals,[.025,.975])
    return float(auc),float(lo),float(hi)

def auc_p_mwu(pos,neg):
    # AUC = Mann-Whitney U/(n_pos*n_neg); one-sided test pos > neg.
    u,p=stats.mannwhitneyu(pos,neg,alternative="greater")
    return float(p)

def pair_signflip_p(diffs,nperm=20000):
    diffs=np.asarray(diffs,float)
    obs=float(np.mean(diffs))
    ge=0
    for _ in range(nperm):
        signs=rng.choice([-1.0,1.0],size=len(diffs))
        if float(np.mean(diffs*signs)) >= obs:
            ge+=1
    return (ge+1)/(nperm+1)

def smd(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    va=np.nanvar(a,ddof=1); vb=np.nanvar(b,ddof=1)
    den=math.sqrt(max((va+vb)/2,1e-12))
    return float((np.nanmean(a)-np.nanmean(b))/den)

# ------------------------------------------------------------------
# 0. Input freeze / reserved-data guard
# ------------------------------------------------------------------
for name in REQ:
    p=INPUTS/name
    if not p.exists():
        raise FileNotFoundError(f"Required BR2 input missing: {p}")

for p in ROOT.rglob("*"):
    if p.is_file():
        low=p.name.lower()
        if "00086" in low or "wildchat" in low:
            raise RuntimeError(f"RESERVED F2 DATA FOUND INSIDE BR2. Abort without reading: {p}")

input_rows=[]
for name in REQ:
    p=INPUTS/name
    input_rows.append({"FILE":name,"BYTES":p.stat().st_size,"SHA256":sha256_file(p)})
pd.DataFrame(input_rows).to_csv(DIRS["manifest"]/ "FROZEN_INPUT_SHA256.csv",index=False)

# Compare staging manifest if present.
staging_manifest=ROOT/"INPUT_SHA256_MANIFEST.csv"
if staging_manifest.exists():
    staged=pd.read_csv(staging_manifest)
    pd.DataFrame(input_rows).merge(staged,left_on="FILE",right_on="FILE",how="outer",suffixes=("_NOW","_STAGED")).to_csv(
        DIRS["manifest"]/ "INPUT_MANIFEST_CROSSCHECK.csv",index=False
    )

# ------------------------------------------------------------------
# 1. Extract frozen BR1 package
# ------------------------------------------------------------------
br1zip=INPUTS/"F1_BR1_BLIND_RSOS_CORPUS_REDISCOVERY_RESULTS_PACKAGE.zip"
with zipfile.ZipFile(br1zip) as z:
    z.extractall(DIRS["br1"])

candidates=list(DIRS["br1"].rglob("FROZEN_FEATURE_SETS.json"))
if len(candidates)!=1:
    raise RuntimeError(f"Expected exactly one FROZEN_FEATURE_SETS.json, found {len(candidates)}")
feature_sets_path=candidates[0]
br1base=feature_sets_path.parents[1]

models_dir=br1base/"06_MODELS"
features_dir=br1base/"05_FEATURES"

r1_model_path=models_dir/"R1_COMPATIBLE.joblib"
if not r1_model_path.exists():
    raise FileNotFoundError(r1_model_path)
r1_model=joblib.load(r1_model_path)

feature_sets=json.loads(feature_sets_path.read_text(encoding="utf-8"))
if feature_sets.get("R1_COMPATIBLE") != R1_FEATURES:
    raise RuntimeError("Frozen BR1 R1 feature list differs from BR2 expected list.")

br1_model_hash=sha256_file(r1_model_path)
(DIRS["manifest"]/ "BR1_R1_MODEL_SHA256.txt").write_text(br1_model_hash+"\n",encoding="ascii")

pos=pd.read_csv(features_dir/"LATER_HOLD_POS_FEATURES.csv")
same_user_neg=pd.read_csv(features_dir/"LATER_HOLD_NEG_FEATURES.csv")

for f in R1_FEATURES:
    if f not in pos.columns or f not in same_user_neg.columns:
        raise RuntimeError(f"BR1 frozen holdout feature missing: {f}")

pos["BR1_R1_SCORE"]=r1_model.predict_proba(pos[R1_FEATURES])[:,1]
same_user_neg["BR1_R1_SCORE"]=r1_model.predict_proba(same_user_neg[R1_FEATURES])[:,1]

# ------------------------------------------------------------------
# 2. Load LM Arena sources + schema audit
# ------------------------------------------------------------------
matrix=pd.read_parquet(INPUTS/"train-00000-of-00007__BLIND_STRUCTURAL_MATRIX.parquet")
arms=pd.read_parquet(INPUTS/"train-00000-of-00007__V12_V18_ARMS.parquet")
pairs=pd.read_parquet(INPUTS/"train-00000-of-00007__V12_V18_PAIRS.parquet")
search=pd.read_parquet(INPUTS/"LMARENA_SEARCH_INDEX.parquet")
dev_train=pd.read_parquet(INPUTS/"train-00000-of-00007__DEVELOPMENT_TRAIN.parquet")
dev_val=pd.read_parquet(INPUTS/"train-00000-of-00007__DEVELOPMENT_VALIDATION.parquet")
dev_hold=pd.read_parquet(INPUTS/"train-00000-of-00007__DEVELOPMENT_HOLDOUT.parquet")

schemas={}
for name,df in [
    ("BLIND_STRUCTURAL_MATRIX",matrix),("ARMS",arms),("PAIRS",pairs),("SEARCH_INDEX",search),
    ("DEV_TRAIN",dev_train),("DEV_VALIDATION",dev_val),("DEV_HOLDOUT",dev_hold)
]:
    schemas[name]={"rows":len(df),"columns":[str(c) for c in df.columns]}
write_json(DIRS["schema"]/ "PARQUET_SCHEMA_AUDIT.json",schemas)

mfeat,missing=all_feature_cols(matrix,R1_FEATURES)
if missing:
    raise RuntimeError(
        "BLIND_STRUCTURAL_MATRIX is missing frozen R1-compatible features: "+", ".join(missing)
    )
rename={c:f for c,f in zip(mfeat,R1_FEATURES)}
lm=matrix.rename(columns=rename).copy()

# ------------------------------------------------------------------
# 3. Merge independent metadata without ever using text as classifier input
# ------------------------------------------------------------------
# Prefer stable record/blind identifiers.
id_candidates=["BLIND_ID","RECORD_ID","ARM_ID","ID"]
lm_id=find_col(lm,id_candidates)
meta_sources=[("SEARCH_INDEX",search),("ARMS",arms)]
merge_log=[]

for src_name,src in meta_sources:
    if lm_id is None:
        break
    src_id=find_col(src,[lm_id]+id_candidates)
    if src_id is None:
        merge_log.append({"SOURCE":src_name,"STATUS":"NO_COMMON_ID"})
        continue
    # Avoid duplicate-column collisions; bring metadata only.
    keep=[src_id]
    for cand in [
        "PAIR_ID","PROVIDER","MODEL","MODEL_NAME","LANGUAGE","LANG","IS_CODE",
        "TOTAL_CHARS","RAW_CHARS","CHAR_COUNT","N_CHARS","N_TURNS","RAW_TURNS",
        "TURN_COUNT","COMPLEXITY","COMPLEXITY_SCORE","TOPIC","TOPIC_ID","SPLIT"
    ]:
        c=find_col(src,[cand])
        if c is not None and c not in keep:
            keep.append(c)
    d=src[keep].drop_duplicates(subset=[src_id]).copy()
    if src_id!=lm_id:
        d=d.rename(columns={src_id:lm_id})
    before=len(lm)
    lm=lm.merge(d,on=lm_id,how="left",suffixes=("","_"+src_name))
    merge_log.append({"SOURCE":src_name,"STATUS":"MERGED","ROWS_BEFORE":before,"ROWS_AFTER":len(lm),"ID":lm_id})

pd.DataFrame(merge_log).to_csv(DIRS["schema"]/ "METADATA_MERGE_LOG.csv",index=False)

# Score every LM record using the frozen BR1 model. No LM retraining.
lm["BR1_R1_SCORE"]=r1_model.predict_proba(lm[R1_FEATURES])[:,1]

# Pair ID, if available, is used to prevent two alternative arms from inflating N.
pair_col=find_col(lm,["PAIR_ID"])
if pair_col is not None:
    # Aggregate numerical fingerprint and nuisance metadata at prompt/pair level.
    agg={f:"mean" for f in R1_FEATURES}
    agg["BR1_R1_SCORE"]="mean"
    for c in lm.columns:
        cu=str(c).upper()
        if c in agg or c==pair_col:
            continue
        if cu in {"TOTAL_CHARS","RAW_CHARS","CHAR_COUNT","N_CHARS","N_TURNS","RAW_TURNS","TURN_COUNT","COMPLEXITY","COMPLEXITY_SCORE"}:
            agg[c]="mean"
        elif cu in {"PROVIDER","MODEL","MODEL_NAME","LANGUAGE","LANG","IS_CODE","TOPIC","TOPIC_ID","SPLIT"}:
            agg[c]="first"
    lm_unit=lm.groupby(pair_col,as_index=False).agg(agg)
    unit_kind="PAIR_AGGREGATED"
else:
    lm_unit=lm.copy()
    unit_kind="ARM_LEVEL_NO_PAIR_ID"

lm_unit.to_parquet(DIRS["lm"]/ "LMARENA_R1_SCORED.parquet",index=False,compression="zstd")
pd.DataFrame([{"LM_UNIT":unit_kind,"N_MATRIX_ROWS":len(matrix),"N_ANALYSIS_UNITS":len(lm_unit)}]).to_csv(
    DIRS["lm"]/ "LMARENA_UNIT_SUMMARY.csv",index=False
)

# ------------------------------------------------------------------
# 4. Identify matching covariates
# ------------------------------------------------------------------
def first_existing(df,names):
    return find_col(df,names)

lm_chars=first_existing(lm_unit,["TOTAL_CHARS","RAW_CHARS","CHAR_COUNT","N_CHARS"])
lm_turns=first_existing(lm_unit,["N_TURNS","RAW_TURNS","TURN_COUNT"])
lm_code=first_existing(lm_unit,["IS_CODE"])
lm_provider=first_existing(lm_unit,["PROVIDER"])
lm_lang=first_existing(lm_unit,["LANGUAGE","LANG"])
lm_model=first_existing(lm_unit,["MODEL","MODEL_NAME"])
lm_topic=first_existing(lm_unit,["TOPIC","TOPIC_ID"])
lm_complex=first_existing(lm_unit,["COMPLEXITY","COMPLEXITY_SCORE"])

availability=[
    ["LENGTH_MATCHING","MEASURED" if lm_chars is not None else "UNAVAILABLE",str(lm_chars)],
    ["TURN_MATCHING","MEASURED" if lm_turns is not None else "UNAVAILABLE",str(lm_turns)],
    ["CODE_MATCHING","MEASURED" if lm_code is not None else "UNAVAILABLE",str(lm_code)],
    ["PROVIDER_MATCHING","MEASURED" if lm_provider is not None else "UNAVAILABLE",str(lm_provider)],
    ["LANGUAGE_MATCHING","MEASURED" if lm_lang is not None else "UNAVAILABLE",str(lm_lang)],
    ["MODEL_MATCHING","MEASURED" if lm_model is not None else "UNAVAILABLE",str(lm_model)],
    ["TOPIC_MATCHING","MEASURED" if lm_topic is not None else "UNAVAILABLE",str(lm_topic)],
    ["COMPLEXITY_MATCHING","MEASURED" if lm_complex is not None else "UNAVAILABLE",str(lm_complex)],
]
pd.DataFrame(availability,columns=["CONTROL","STATUS","COLUMN"]).to_csv(
    DIRS["matching"]/ "MATCHING_COVARIATE_AVAILABILITY.csv",index=False
)

# ------------------------------------------------------------------
# 5. Global independent-population comparison
# ------------------------------------------------------------------
results=[]
def add_auc(test_name,pos_scores,neg_scores):
    y=np.r_[np.ones(len(pos_scores),int),np.zeros(len(neg_scores),int)]
    s=np.r_[np.asarray(pos_scores,float),np.asarray(neg_scores,float)]
    auc,lo,hi=auc_ci(y,s)
    p=auc_p_mwu(pos_scores,neg_scores)
    results.append({
        "TEST":test_name,"N_POS":len(pos_scores),"N_NEG":len(neg_scores),
        "AUC":auc,"CI95_LOW":lo,"CI95_HIGH":hi,"P_ONE_SIDED":p
    })

add_auc("LATER_LINEAGE_vs_LMARENA_ALL",pos["BR1_R1_SCORE"].values,lm_unit["BR1_R1_SCORE"].values)
add_auc("LATER_LINEAGE_vs_SAME_USER_NONLINEAGE",pos["BR1_R1_SCORE"].values,same_user_neg["BR1_R1_SCORE"].values)
add_auc("SAME_USER_NONLINEAGE_vs_LMARENA_ALL",same_user_neg["BR1_R1_SCORE"].values,lm_unit["BR1_R1_SCORE"].values)

# Provider-matched OpenAI subset if metadata permits.
openai_lm=None
if lm_provider is not None:
    mask=lm_unit[lm_provider].astype(str).str.lower().str.contains("openai",na=False)
    if mask.sum()>=max(50,len(pos)//2):
        openai_lm=lm_unit.loc[mask].copy()
        add_auc("LATER_LINEAGE_vs_LMARENA_OPENAI_PROVIDER",pos["BR1_R1_SCORE"].values,openai_lm["BR1_R1_SCORE"].values)

pd.DataFrame(results).to_csv(DIRS["results"]/ "BR2_PRIMARY_AUC_RESULTS.csv",index=False)

# ------------------------------------------------------------------
# 6. One-to-one nuisance matching (provider OpenAI preferred)
# ------------------------------------------------------------------
pool = openai_lm.copy() if openai_lm is not None else lm_unit.copy()
pool_name = "OPENAI_PROVIDER" if openai_lm is not None else "ALL_LMARENA"

# Construct comparable numeric covariates conservatively.
cov_pos=[]
cov_lm=[]
cov_names=[]

if lm_chars is not None:
    cov_pos.append(np.log1p(pos["raw_chars"].astype(float).values))
    cov_lm.append(np.log1p(pd.to_numeric(pool[lm_chars],errors="coerce").fillna(pool[lm_chars].median()).values))
    cov_names.append("LOG_CHARS")

if lm_turns is not None:
    cov_pos.append(np.log1p(pos["raw_turns"].astype(float).values))
    cov_lm.append(np.log1p(pd.to_numeric(pool[lm_turns],errors="coerce").fillna(pool[lm_turns].median()).values))
    cov_names.append("LOG_TURNS")

# Complexity proxy exists in both spaces even when no explicit metadata exists.
for f in ["CODE_FENCES_PER_1K_CHAR","HEADERS_PER_1K_CHAR","LIST_ITEMS_PER_1K_CHAR","MEAN_MAX_TURN_RATIO"]:
    cov_pos.append(pos[f].astype(float).values)
    cov_lm.append(pool[f].astype(float).values)
    cov_names.append("MATCH_"+f)

if not cov_names:
    raise RuntimeError("No matching covariates available.")

A=np.column_stack(cov_pos)
B=np.column_stack(cov_lm)
stack=np.vstack([A,B])
med=np.nanmedian(stack,axis=0)
mad=np.nanmedian(np.abs(stack-med),axis=0)*1.4826
sd=np.nanstd(stack,axis=0)
scale=np.where(mad>1e-9,mad,np.where(sd>1e-9,sd,1.0))
A=(A-med)/scale
B=(B-med)/scale

# Code mismatch penalty if explicit code flag exists.
# BR1 proxy uses presence of code fences.
cost=((A[:,None,:]-B[None,:,:])**2).sum(axis=2)
if lm_code is not None:
    pos_code=(pos["CODE_FENCES_PER_1K_CHAR"].values>0).astype(int)
    lm_code_vals=pool[lm_code]
    if lm_code_vals.dtype==bool:
        lmc=lm_code_vals.astype(int).values
    else:
        lmc=lm_code_vals.astype(str).str.lower().isin(["1","true","yes","code"]).astype(int).values
    cost += (pos_code[:,None]!=lmc[None,:])*4.0

# Language exact constraint only if BR1 language is explicitly available (it is normally not).
# Do NOT invent a language label from text.
row_ind,col_ind=linear_sum_assignment(cost)
matched=pool.iloc[col_ind].copy().reset_index(drop=True)
pp=pos.iloc[row_ind].copy().reset_index(drop=True)

pairs_out=pd.DataFrame({
    "PAIR_NO":np.arange(len(pp)),
    "RSOS_CONVERSATION_ID":pp["conversation_id"].astype(str),
    "RSOS_SCORE":pp["BR1_R1_SCORE"].astype(float),
    "LMARENA_SCORE":matched["BR1_R1_SCORE"].astype(float),
    "MATCH_COST":cost[row_ind,col_ind],
})
if pair_col is not None and pair_col in matched.columns:
    pairs_out["LMARENA_PAIR_ID"]=matched[pair_col].astype(str).values
elif lm_id is not None and lm_id in matched.columns:
    pairs_out["LMARENA_ID"]=matched[lm_id].astype(str).values

diff=pairs_out["RSOS_SCORE"].values-pairs_out["LMARENA_SCORE"].values
pair_win=float(np.mean(diff>0))
sign_p=pair_signflip_p(diff)
binom_p=float(stats.binomtest(int((diff>0).sum()),len(diff),.5,alternative="greater").pvalue)

pairs_out.to_csv(DIRS["matching"]/ "ONE_TO_ONE_MATCHED_PAIRS.csv",index=False)

# Covariate balance before/after.
balance=[]
for j,n in enumerate(cov_names):
    balance.append({
        "COVARIATE":n,
        "SMD_BEFORE":smd(A[:,j],B[:,j]),
        "SMD_AFTER":smd(A[row_ind,j],B[col_ind,j])
    })
pd.DataFrame(balance).to_csv(DIRS["matching"]/ "COVARIATE_BALANCE.csv",index=False)

matched_summary=pd.DataFrame([{
    "POOL":pool_name,
    "N_PAIRS":len(diff),
    "PAIR_WIN_RATE":pair_win,
    "MEAN_SCORE_DIFF":float(np.mean(diff)),
    "MEDIAN_SCORE_DIFF":float(np.median(diff)),
    "SIGNFLIP_P":sign_p,
    "BINOMIAL_P":binom_p,
}])
matched_summary.to_csv(DIRS["results"]/ "BR2_MATCHED_PAIR_RESULTS.csv",index=False)

# AUC on exactly matched LM controls.
add_auc("LATER_LINEAGE_vs_ONE_TO_ONE_MATCHED_LMARENA",pp["BR1_R1_SCORE"].values,matched["BR1_R1_SCORE"].values)
pd.DataFrame(results).to_csv(DIRS["results"]/ "BR2_PRIMARY_AUC_RESULTS.csv",index=False)

# ------------------------------------------------------------------
# 7. Development-split robustness
# ------------------------------------------------------------------
# If a split ID can be mapped, report. Otherwise keep UNAVAILABLE, never negative.
split_rows=[]
split_col=find_col(lm_unit,["SPLIT"])
if split_col is not None:
    for val,g in lm_unit.groupby(split_col):
        if len(g)>=20:
            y=np.r_[np.ones(len(pos),int),np.zeros(len(g),int)]
            s=np.r_[pos["BR1_R1_SCORE"].values,g["BR1_R1_SCORE"].values]
            a,lo,hi=auc_ci(y,s,nboot=2000)
            split_rows.append({"SPLIT":str(val),"N_LM":len(g),"AUC":a,"CI95_LOW":lo,"CI95_HIGH":hi})
if split_rows:
    pd.DataFrame(split_rows).to_csv(DIRS["robust"]/ "LMARENA_SPLIT_ROBUSTNESS.csv",index=False)

# ------------------------------------------------------------------
# 8. Joint model compatibility audit (no silent substitution)
# ------------------------------------------------------------------
joint_features=feature_sets.get("JOINT_MULTILAYER",[])
_,joint_missing=all_feature_cols(matrix,joint_features)
joint_status="MEASURED" if not joint_missing else "UNAVAILABLE"
write_json(DIRS["schema"]/ "JOINT_MODEL_COMPATIBILITY.json",{
    "status":joint_status,
    "missing_features":joint_missing,
    "note":"BR2 primary uses the frozen BR1 R1-compatible model. Joint model is only eligible if every frozen joint feature exists row-level in LM Arena."
})

# ------------------------------------------------------------------
# 9. No-silent-omission ledger
# ------------------------------------------------------------------
ledger=[
    ["BR1_FROZEN_R1_MODEL_APPLIED_WITHOUT_RETRAINING","MEASURED",br1_model_hash],
    ["LATER_LINEAGE_vs_LMARENA_ROW_LEVEL","MEASURED","Frozen BR1 R1-compatible score"],
    ["SAME_USER_NONLINEAGE_vs_LMARENA","MEASURED","Authorship/style diagnostic"],
    ["ONE_TO_ONE_LENGTH_TURN_COMPLEXITY_MATCHING","MEASURED","Uses available exact staged metadata plus shared structural nuisance proxies"],
    ["PROVIDER_MATCHING","MEASURED" if openai_lm is not None else "UNAVAILABLE","OpenAI LM subset preferred when present"],
    ["LANGUAGE_MATCHING","MEASURED" if lm_lang is not None else "UNAVAILABLE","No language label inferred if absent"],
    ["MODEL_MATCHING","MEASURED" if lm_model is not None else "UNAVAILABLE","Metadata availability only; exact-model match not forced if sparse"],
    ["TOPIC_MATCHING","MEASURED" if lm_topic is not None else "UNAVAILABLE","No topic label invented if absent"],
    ["JOINT_MULTILAYER_ROW_LEVEL_LMARENA",joint_status,"No feature substitution allowed"],
    ["F2_WILDCHAT_00086","RESERVED_NOT_READ","Guard aborts if such files appear in BR2 folder"],
]
pd.DataFrame(ledger,columns=["MEASUREMENT","STATUS","DETAIL"]).to_csv(
    DIRS["ledger"]/ "NO_SILENT_OMISSION_LEDGER.csv",index=False
)

# ------------------------------------------------------------------
# 10. Summary / provenance
# ------------------------------------------------------------------
primary=pd.DataFrame(results)
summary={
    "version":VERSION,
    "completed_utc":now(),
    "frozen_br1_model_sha256":br1_model_hash,
    "lm_unit_kind":unit_kind,
    "n_lm_units":int(len(lm_unit)),
    "n_later_lineage_holdout":int(len(pos)),
    "n_same_user_non_lineage":int(len(same_user_neg)),
    "matching_pool":pool_name,
    "matched_pairs":int(len(diff)),
    "matched_pair_win_rate":pair_win,
    "matched_signflip_p":sign_p,
    "joint_model_row_level_status":joint_status,
    "wildchat_00086_read":False,
    "primary_results":primary.to_dict("records"),
}
write_json(DIRS["manifest"]/ "BR2_SUMMARY.json",summary)

txt=[
    "F1-BR2 - LM ARENA ROW-LEVEL MATCHED CONTROL",
    "",
    f"Frozen BR1 R1 model SHA256: {br1_model_hash}",
    f"LM Arena analysis unit: {unit_kind}",
    f"LM Arena units: {len(lm_unit)}",
    f"Later-lineage BR1 holdout positives: {len(pos)}",
    f"Same-user non-lineage holdout controls: {len(same_user_neg)}",
    "",
]
for r in primary.to_dict("records"):
    txt.append(f"{r['TEST']}: AUC={r['AUC']:.6f} 95%CI=[{r['CI95_LOW']:.6f},{r['CI95_HIGH']:.6f}] p={r['P_ONE_SIDED']:.6g}")
txt += [
    "",
    f"Matched pool: {pool_name}",
    f"Matched pairs: {len(diff)}",
    f"Matched pair win rate: {pair_win:.6f}",
    f"Matched sign-flip p: {sign_p:.6g}",
    f"Joint multilayer LM row-level status: {joint_status}",
    "F2 00086/WildChat read: NO",
]
(DIRS["manifest"]/ "BR2_SUMMARY.txt").write_text("\n".join(txt)+"\n",encoding="utf-8")

env={
    "python":sys.version,
    "platform":platform.platform(),
    "numpy":np.__version__,
    "pandas":pd.__version__,
    "scipy":stats.__name__.split(".")[0],
    "joblib":joblib.__version__,
}
write_json(DIRS["manifest"]/ "ENVIRONMENT.json",env)

runner=Path(__file__).resolve()
(DIRS["manifest"]/ "PYTHON_RUNNER_SHA256.txt").write_text(sha256_file(runner)+"\n",encoding="ascii")

# Output hash manifest, before marker and final ZIP.
rows=[]
for p in sorted(ROOT.rglob("*")):
    if p.is_file() and p.name!="F1_BR2_LMARENA_MATCHED_CONTROL_COMPLETE.zip":
        rows.append({"RELATIVE_PATH":str(p.relative_to(ROOT)),"BYTES":p.stat().st_size,"SHA256":sha256_file(p)})
pd.DataFrame(rows).to_csv(DIRS["manifest"]/ "OUTPUT_SHA256_MANIFEST.csv",index=False)

MARKER.write_text(f"{VERSION}\n{now()}\n",encoding="ascii")

print()
print("="*80)
print("F1-BR2 COMPLETE")
print("="*80)
print((DIRS["manifest"]/ "BR2_SUMMARY.txt").read_text(encoding="utf-8"))
