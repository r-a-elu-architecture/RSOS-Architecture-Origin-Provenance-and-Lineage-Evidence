
from pathlib import Path
import json, re, zlib, math, hashlib, sys, platform, zipfile
from datetime import datetime, timezone
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score

ROOT = Path(r"C:\RSOS\Training data\F1_BR2_LMARENA_MATCHED_CONTROL")
INPUTS = ROOT / "00_INPUTS"
V = "F1_BR2_V2_EXACT_EXTRACTOR"

OUT = {
    "failed": ROOT / "10_V2_FAILED_RUN_LEDGER",
    "manifest": ROOT / "11_V2_MANIFEST",
    "schema": ROOT / "12_V2_SCHEMA",
    "features": ROOT / "13_V2_LMARENA_FEATURES",
    "matching": ROOT / "14_V2_MATCHING",
    "results": ROOT / "15_V2_RESULTS",
    "ledger": ROOT / "16_V2_NO_SILENT_OMISSION",
}
for p in OUT.values():
    p.mkdir(parents=True, exist_ok=True)

MARKER = ROOT / "F1_BR2_V2_COMPLETE.marker"
if MARKER.exists():
    raise SystemExit(f"V2 already completed; refusing overwrite: {MARKER}")

SEED = 120918
rng = np.random.default_rng(SEED)

REQ = [
    "F1_BR1_BLIND_RSOS_CORPUS_REDISCOVERY_RESULTS_PACKAGE.zip",
    "train-00000-of-00007__V12_V18_ARMS.parquet",
    "train-00000-of-00007__BLIND_STRUCTURAL_MATRIX.parquet",
    "LMARENA_SEARCH_INDEX.parquet",
    "train-00000-of-00007__DEVELOPMENT_HOLDOUT.parquet",
]
for name in REQ:
    if not (INPUTS/name).exists():
        raise FileNotFoundError(INPUTS/name)

for p in ROOT.rglob("*"):
    if p.is_file() and ("00086" in p.name.lower() or "wildchat" in p.name.lower()):
        raise RuntimeError(f"F2 RESERVED DATA PRESENT. ABORT WITHOUT READING: {p}")

def utcnow():
    return datetime.now(timezone.utc).isoformat()

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def write_json(p,x):
    p.write_text(json.dumps(x,indent=2,default=str),encoding="utf-8")

pd.DataFrame([{
    "RUN":"F1_BR2_V1",
    "STATUS":"FAILED_CALIBRATION",
    "REASON":"V1 incorrectly required ten derived BR1 feature names to exist directly in BLIND_STRUCTURAL_MATRIX. It terminated before LM Arena scoring.",
    "INFERENCE":"NONE",
    "CORRECTION":"V2 re-extracts deterministic frozen BR1 features directly from NORMALIZED_CONVERSATION_JSON in the staged ARM_LEVEL parquet."
}]).to_csv(OUT["failed"]/"V1_FAILURE_PRESERVED.csv", index=False)

DIRECT_RE=re.compile(r'''(?ix)\b(?:RSOS|RSSO|RSIA|RSX|CM1|LINX|MEMSTACK|CONTAIN|UGCL|CML|SRC[-\s]?X(?:\.5)?|R\.\s*A\.\s*Elu)\b|recursive\s+symbolic\s+(?:operating\s+system|system\s+operating|identity\s+architecture|(?:eXpression|expression)(?:\s+language)?)|\b(?:tier(?:s)?|glyph(?:s)?|archetype(?:s)?|override(?:s|d|ing)?|containment|resurrection|echo\s+protocol|activation\s+protocol)\b|\b(?:lion|dragon|eagle|whale|serpent)\b''')

def redact(s):
    return DIRECT_RE.sub(lambda m:' '*len(m.group(0)), s)

def safe(a,b):
    return float(a/b) if b else 0.0

def slope(vals):
    if len(vals)<2:
        return 0.0
    x=np.arange(len(vals),dtype=float)
    y=np.asarray(vals,dtype=float)
    if np.std(y)<1e-12:
        return 0.0
    return float(np.polyfit(x,y,1)[0]/max(1.0,float(np.mean(np.abs(y)))))

def entropy(vals):
    if not vals:
        return 0.0
    bins=[0,80,300,1000,3000,10000,10**9]
    c=np.histogram(vals,bins=bins)[0]
    p=c[c>0]/c.sum()
    return float(-(p*np.log2(p)).sum())

def deterministic_features(messages):
    turns=[]
    redacted_spans=0
    rawchars=0
    for m in messages:
        t=str(m.get("text","") or "")
        rawchars += len(t)
        bt=redact(t)
        redacted_spans += sum(1 for _ in DIRECT_RE.finditer(t))
        if bt.strip():
            turns.append({"role":str(m.get("role","") or ""),"text":bt})
    texts=[t["text"] for t in turns]
    roles=[t["role"] for t in turns]
    full="\n".join(texts)
    low=full.lower()
    total=len(full)
    toks=re.findall(r"[a-z0-9_]+",low)
    ctr=Counter(toks)
    lens=[len(x) for x in texts]
    n=len(turns)
    user="\n".join(t["text"] for t in turns if t["role"]=="user")
    ass="\n".join(t["text"] for t in turns if t["role"]=="assistant")
    ns=sum(1 for c in full if not c.isspace())
    up=sum(1 for c in full if c.isupper())
    dg=sum(1 for c in full if c.isdigit())
    roletrans=sum(1 for a,b in zip(roles,roles[1:]) if a!=b)
    dup=(len(texts)-len(set(texts))) if texts else 0
    para=sum(max(1,x.count("\n")+1) for x in texts)
    fences=full.count("```")//2
    heads=re.findall(r"(?m)^\s*(#{1,6})\s",full)
    listn=len(re.findall(r"(?m)^\s*(?:[-*+] |\d+[.)] )",full))
    br=sum(full.count(x) for x in "[]{}()")
    enc=full.encode("utf-8",errors="ignore")
    comp=len(zlib.compress(enc,6))/len(enc) if enc else 0.0
    mean=float(np.mean(lens)) if lens else 0.0
    mx=max(lens,default=0)
    f={
      "REDACTED_SPANS":redacted_spans,
      "N_TURNS":n,
      "TOTAL_CHARS":total,
      "TYPE_TOKEN_RATIO":safe(len(ctr),len(toks)),
      "UPPERCASE_RATIO":safe(up,ns),
      "DIGIT_RATIO":safe(dg,ns),
      "COMPRESSION_RATIO":comp,
      "USER_CHAR_SHARE":safe(len(user),total),
      "MEAN_MAX_TURN_RATIO":safe(mean,mx),
      "WORDS_PER_1K_CHAR":1000*safe(len(toks),total),
      "TURNS_PER_1K_CHAR":1000*safe(n,total),
      "PARAGRAPHS_PER_TURN":safe(para,n),
      "CODE_FENCES_PER_1K_CHAR":1000*safe(fences,total),
      "HEADERS_PER_1K_CHAR":1000*safe(len(heads),total),
      "LIST_ITEMS_PER_1K_CHAR":1000*safe(listn,total),
      "COLONS_PER_1K_CHAR":1000*safe(full.count(":"),total),
      "BRACKETS_PER_1K_CHAR":1000*safe(br,total),
      "ROLE_ALTERNATION_RATIO":safe(roletrans,max(1,n-1)) if n else 0.0,
      "DUPLICATE_TURN_RATIO":safe(dup,n),
      "USER_ASSISTANT_CHAR_RATIO":safe(len(user),len(ass)),
      "TURN_CHAR_CV":safe(float(np.std(lens)),mean) if lens else 0.0,
      "TURN_LENGTH_ENTROPY":entropy(lens),
      "QUESTION_PER_1K_CHAR":1000*safe(full.count("?"),total),
      "EXCLAM_PER_1K_CHAR":1000*safe(full.count("!"),total),
      "EQUAL_PER_1K_CHAR":1000*safe(full.count("="),total),
      "SEMICOLON_PER_1K_CHAR":1000*safe(full.count(";"),total),
      "PIPE_PER_1K_CHAR":1000*safe(full.count("|"),total),
      "HEADER_LEVEL_MEAN":float(np.mean([len(x) for x in heads])) if heads else 0.0,
      "HEADER_LEVEL_MAX":float(max([len(x) for x in heads],default=0)),
      "USER_TURN_MEAN":0.0,
      "ASSISTANT_TURN_MEAN":0.0,
      "RESPONSE_EXPANSION_RATIO":0.0,
    }
    ul=[len(t["text"]) for t in turns if t["role"]=="user"]
    al=[len(t["text"]) for t in turns if t["role"]=="assistant"]
    f["USER_TURN_MEAN"]=float(np.mean(ul)) if ul else 0.0
    f["ASSISTANT_TURN_MEAN"]=float(np.mean(al)) if al else 0.0
    f["RESPONSE_EXPANSION_RATIO"]=safe(
        float(np.mean(al)) if al else 0.0,
        float(np.mean(ul)) if ul else 0.0
    )
    return f, rawchars, len(messages)

portable_path=ROOT/"PORTABLE_FROZEN_BR1_MODELS.json"
if not portable_path.exists():
    raise FileNotFoundError(portable_path)
portable=json.loads(portable_path.read_text(encoding="utf-8"))

def portable_score(df,key):
    m=portable["models"][key]
    cols=m["features"]
    miss=[c for c in cols if c not in df.columns]
    if miss:
        raise RuntimeError(f"{key} missing features: {miss}")
    X=df[cols].to_numpy(float)
    mean=np.asarray(m["mean"],float)
    scale=np.asarray(m["scale"],float)
    coef=np.asarray(m["coef"],float)
    intercept=float(m["intercept"])
    z=((X-mean)/scale)@coef+intercept
    return 1/(1+np.exp(-np.clip(z,-709,709)))

br1dir=ROOT/"11_V2_BR1_FROZEN"
br1dir.mkdir(parents=True,exist_ok=True)
br1zip=INPUTS/"F1_BR1_BLIND_RSOS_CORPUS_REDISCOVERY_RESULTS_PACKAGE.zip"
with zipfile.ZipFile(br1zip) as z:
    z.extractall(br1dir)

fs=list(br1dir.rglob("FROZEN_FEATURE_SETS.json"))
if len(fs)!=1:
    raise RuntimeError(f"Expected one FROZEN_FEATURE_SETS.json; found {len(fs)}")
br1base=fs[0].parents[1]
featdir=br1base/"05_FEATURES"
modeldir=br1base/"06_MODELS"
resdir=br1base/"07_RESULTS"

hash_rows=[]
for key,fn in [("R1_COMPATIBLE","R1_COMPATIBLE.joblib"),("SURFACE_FREE","SURFACE_FREE.joblib")]:
    h=sha256_file(modeldir/fn)
    expected=portable["models"][key]["source_joblib_sha256"]
    if h!=expected:
        raise RuntimeError(f"Frozen model hash mismatch {key}: {h} != {expected}")
    hash_rows.append({
        "MODEL":key,
        "JOBLIB_SHA256":h,
        "PORTABLE_EQUIV_MAX_ABS_DIFF":portable["models"][key]["equivalence_max_abs_diff_vs_stored_BR1_scores"]
    })
pd.DataFrame(hash_rows).to_csv(OUT["manifest"]/"PORTABLE_MODEL_PROVENANCE.csv",index=False)

pos=pd.read_csv(featdir/"LATER_HOLD_POS_FEATURES.csv")
neg=pd.read_csv(featdir/"LATER_HOLD_NEG_FEATURES.csv")
stored=pd.read_csv(resdir/"HOLDOUT_SCORES_INTERNAL.csv")
combo=pd.concat([pos,neg],ignore_index=True)

equiv=[]
for key in ["R1_COMPATIBLE","SURFACE_FREE"]:
    p=portable_score(combo,key)
    smap=dict(zip(stored["conversation_id"].astype(str),stored[f"{key}_SCORE"].astype(float)))
    expected=np.array([smap[str(x)] for x in combo["conversation_id"]],float)
    md=float(np.max(np.abs(p-expected)))
    equiv.append({"MODEL":key,"MAX_ABS_DIFF":md,"PASS":md<1e-10})
    if md>=1e-10:
        raise RuntimeError(f"Portable scorer failed frozen score equivalence for {key}: {md}")
pd.DataFrame(equiv).to_csv(OUT["manifest"]/"PORTABLE_SCORE_EQUIVALENCE.csv",index=False)

for key in ["R1_COMPATIBLE","SURFACE_FREE"]:
    pos[f"{key}_SCORE"]=portable_score(pos,key)
    neg[f"{key}_SCORE"]=portable_score(neg,key)

arm_path=INPUTS/"train-00000-of-00007__V12_V18_ARMS.parquet"
cols=[
    "BLIND_ID","PAIR_ID","LANGUAGE","IS_CODE","MODEL","PROVIDER",
    "DEVELOPMENT_SPLIT","NORMALIZED_CONVERSATION_JSON","N_TURNS","TOTAL_CHARS"
]
arms=pd.read_parquet(arm_path,columns=cols)
write_json(OUT["schema"]/"ARM_SCHEMA_USED.json",{"rows":len(arms),"columns":cols})

rows=[]
errors=[]
for idx,r in arms.iterrows():
    try:
        turns=json.loads(r["NORMALIZED_CONVERSATION_JSON"])
        msgs=[
            {"role":str(t.get("role","") or ""),"text":str(t.get("text","") or "")}
            for t in turns if isinstance(t,dict)
        ]
        f,rawchars,rawturns=deterministic_features(msgs)
        if f["TOTAL_CHARS"]<800 or f["N_TURNS"]<3:
            continue
        rows.append({
            "BLIND_ID":r["BLIND_ID"],
            "PAIR_ID":r["PAIR_ID"],
            "LANGUAGE":r["LANGUAGE"],
            "IS_CODE":r["IS_CODE"],
            "MODEL":r["MODEL"],
            "PROVIDER":r["PROVIDER"],
            "DEVELOPMENT_SPLIT":r["DEVELOPMENT_SPLIT"],
            "raw_chars":rawchars,
            "raw_turns":rawturns,
            **f
        })
    except Exception as e:
        errors.append({
            "ROW_INDEX":int(idx),
            "BLIND_ID":r.get("BLIND_ID"),
            "PAIR_ID":r.get("PAIR_ID"),
            "ERROR":repr(e)
        })
    if (idx+1)%2500==0:
        print(f"LM exact extraction {idx+1}/{len(arms)}",flush=True)

lm=pd.DataFrame(rows)
if errors:
    pd.DataFrame(errors).to_csv(OUT["features"]/"EXTRACTION_ERRORS.csv",index=False)

for key in ["R1_COMPATIBLE","SURFACE_FREE"]:
    lm[f"{key}_SCORE"]=portable_score(lm,key)

lm.to_parquet(
    OUT["features"]/"LMARENA_EXACT_BR1_FEATURES.parquet",
    index=False,
    compression="zstd"
)

hold=lm[lm["DEVELOPMENT_SPLIT"].astype(str).str.upper().eq("DEVELOPMENT_HOLDOUT")].copy()
if len(hold)<100:
    hold=lm[lm["DEVELOPMENT_SPLIT"].astype(str).str.upper().str.contains("HOLDOUT",na=False)].copy()
if len(hold)<100:
    raise RuntimeError(f"LM Arena development holdout mapping failed: only {len(hold)} usable arms")

hold=hold.sort_values(["PAIR_ID","BLIND_ID"]).drop_duplicates("PAIR_ID",keep="first").reset_index(drop=True)
all_ind=lm.sort_values(["PAIR_ID","BLIND_ID"]).drop_duplicates("PAIR_ID",keep="first").reset_index(drop=True)

pd.DataFrame([{
    "LM_ARMS_TOTAL":len(arms),
    "LM_USABLE_ARMS":len(lm),
    "LM_INDEPENDENT_ALL_PAIRS":len(all_ind),
    "LM_INDEPENDENT_HOLDOUT_PAIRS":len(hold),
    "EXTRACTION_ERRORS":len(errors)
}]).to_csv(OUT["features"]/"EXTRACTION_COUNTS.csv",index=False)

def auc_ci(poss,negs,nboot=2000):
    poss=np.asarray(poss,float)
    negs=np.asarray(negs,float)
    y=np.r_[np.ones(len(poss),int),np.zeros(len(negs),int)]
    s=np.r_[poss,negs]
    auc=float(roc_auc_score(y,s))
    vals=[]
    for _ in range(nboot):
        pp=poss[rng.integers(0,len(poss),len(poss))]
        nn=negs[rng.integers(0,len(negs),len(negs))]
        yy=np.r_[np.ones(len(pp)),np.zeros(len(nn))]
        ss=np.r_[pp,nn]
        vals.append(roc_auc_score(yy,ss))
    lo,hi=np.quantile(vals,[.025,.975])
    p=float(stats.mannwhitneyu(poss,negs,alternative="greater").pvalue)
    return auc,float(lo),float(hi),p

results=[]
def add(test,key,a,b):
    auc,lo,hi,p=auc_ci(a,b)
    results.append({
        "TEST":test,
        "MODEL":key,
        "N_POS":len(a),
        "N_NEG":len(b),
        "AUC":auc,
        "CI95_LOW":lo,
        "CI95_HIGH":hi,
        "P_ONE_SIDED":p
    })

for key in ["R1_COMPATIBLE","SURFACE_FREE"]:
    add("LATER_LINEAGE_vs_LMARENA_DEVELOPMENT_HOLDOUT",key,pos[f"{key}_SCORE"],hold[f"{key}_SCORE"])
    add("LATER_LINEAGE_vs_SAMEUSER_NONLINEAGE",key,pos[f"{key}_SCORE"],neg[f"{key}_SCORE"])
    add("SAMEUSER_NONLINEAGE_vs_LMARENA_DEVELOPMENT_HOLDOUT",key,neg[f"{key}_SCORE"],hold[f"{key}_SCORE"])
    add("LATER_LINEAGE_vs_LMARENA_ALL_INDEPENDENT_PAIRS",key,pos[f"{key}_SCORE"],all_ind[f"{key}_SCORE"])

provider_rows=[]
for prov,g in hold.groupby("PROVIDER"):
    if len(g)<50:
        continue
    for key in ["R1_COMPATIBLE","SURFACE_FREE"]:
        a,lo,hi,p=auc_ci(pos[f"{key}_SCORE"],g[f"{key}_SCORE"],nboot=1000)
        provider_rows.append({
            "PROVIDER":str(prov),
            "MODEL":key,
            "N_LM":len(g),
            "AUC":a,
            "CI95_LOW":lo,
            "CI95_HIGH":hi,
            "P_ONE_SIDED":p
        })
pd.DataFrame(provider_rows).to_csv(OUT["results"]/"PROVIDER_STRATIFIED_AUC.csv",index=False)

openai=hold[hold["PROVIDER"].astype(str).str.lower().str.contains("openai",na=False)].copy()
pool=openai if len(openai)>=len(pos) else hold.copy()
pool_name="OPENAI_HOLDOUT" if len(openai)>=len(pos) else "ALL_PROVIDER_HOLDOUT"

def code_flag_df(df):
    return (pd.to_numeric(df["CODE_FENCES_PER_1K_CHAR"],errors="coerce").fillna(0)>0).astype(int).to_numpy()

A=np.column_stack([
    np.log1p(pos["raw_chars"].astype(float)),
    np.log1p(pos["raw_turns"].astype(float)),
    pos["PARAGRAPHS_PER_TURN"].astype(float),
    pos["CODE_FENCES_PER_1K_CHAR"].astype(float),
    pos["HEADERS_PER_1K_CHAR"].astype(float),
    pos["LIST_ITEMS_PER_1K_CHAR"].astype(float),
    pos["MEAN_MAX_TURN_RATIO"].astype(float),
])
B=np.column_stack([
    np.log1p(pool["raw_chars"].astype(float)),
    np.log1p(pool["raw_turns"].astype(float)),
    pool["PARAGRAPHS_PER_TURN"].astype(float),
    pool["CODE_FENCES_PER_1K_CHAR"].astype(float),
    pool["HEADERS_PER_1K_CHAR"].astype(float),
    pool["LIST_ITEMS_PER_1K_CHAR"].astype(float),
    pool["MEAN_MAX_TURN_RATIO"].astype(float),
])
names=[
    "LOG_CHARS","LOG_TURNS","PARAGRAPHS_PER_TURN",
    "CODE_FENCES_PER_1K_CHAR","HEADERS_PER_1K_CHAR",
    "LIST_ITEMS_PER_1K_CHAR","MEAN_MAX_TURN_RATIO"
]

stack=np.vstack([A,B])
med=np.nanmedian(stack,axis=0)
mad=np.nanmedian(np.abs(stack-med),axis=0)*1.4826
sd=np.nanstd(stack,axis=0)
scale=np.where(mad>1e-9,mad,np.where(sd>1e-9,sd,1.0))
Az=(A-med)/scale
Bz=(B-med)/scale
cost=((Az[:,None,:]-Bz[None,:,:])**2).sum(axis=2)
pc=code_flag_df(pos)
bc=code_flag_df(pool)
cost += (pc[:,None]!=bc[None,:])*9.0

ri,ci=linear_sum_assignment(cost)
mp=pos.iloc[ri].reset_index(drop=True)
mn=pool.iloc[ci].reset_index(drop=True)

def smd(x,y):
    x=np.asarray(x,float)
    y=np.asarray(y,float)
    den=math.sqrt(max((np.var(x,ddof=1)+np.var(y,ddof=1))/2,1e-12))
    return float((np.mean(x)-np.mean(y))/den)

balance=[]
for j,n in enumerate(names):
    balance.append({
        "COVARIATE":n,
        "SMD_BEFORE":smd(A[:,j],B[:,j]),
        "SMD_AFTER":smd(A[ri,j],B[ci,j])
    })
balance_df=pd.DataFrame(balance)
balance_df.to_csv(OUT["matching"]/"MATCH_BALANCE.csv",index=False)
max_abs_smd=float(balance_df["SMD_AFTER"].abs().max())

pair_rows=[]
for key in ["R1_COMPATIBLE","SURFACE_FREE"]:
    d=mp[f"{key}_SCORE"].to_numpy(float)-mn[f"{key}_SCORE"].to_numpy(float)
    wins=int((d>0).sum())
    n=len(d)
    binom=float(stats.binomtest(wins,n,.5,alternative="greater").pvalue)
    obs=float(np.mean(d))
    ge=0
    for _ in range(20000):
        signs=rng.choice([-1.0,1.0],size=n)
        if float(np.mean(d*signs))>=obs:
            ge+=1
    sp=(ge+1)/20001
    auc,lo,hi,p=auc_ci(mp[f"{key}_SCORE"],mn[f"{key}_SCORE"])
    pair_rows.append({
        "MODEL":key,
        "POOL":pool_name,
        "N_PAIRS":n,
        "MATCHED_AUC":auc,
        "CI95_LOW":lo,
        "CI95_HIGH":hi,
        "PAIR_WIN_RATE":wins/n,
        "MEAN_SCORE_DIFF":float(np.mean(d)),
        "BINOMIAL_P":binom,
        "SIGNFLIP_P":sp,
        "MAX_ABS_SMD_AFTER":max_abs_smd
    })
pd.DataFrame(pair_rows).to_csv(OUT["results"]/"MATCHED_CONTROL_RESULTS.csv",index=False)

pairs_out=pd.DataFrame({
    "PAIR_NO":np.arange(len(mp)),
    "RSOS_CONVERSATION_ID":mp["conversation_id"].astype(str),
    "LM_BLIND_ID":mn["BLIND_ID"].astype(str),
    "LM_PAIR_ID":mn["PAIR_ID"].astype(str),
    "LM_PROVIDER":mn["PROVIDER"].astype(str),
    "LM_MODEL":mn["MODEL"].astype(str),
    "MATCH_COST":cost[ri,ci],
})
for key in ["R1_COMPATIBLE","SURFACE_FREE"]:
    pairs_out[f"RSOS_{key}_SCORE"]=mp[f"{key}_SCORE"].to_numpy(float)
    pairs_out[f"LM_{key}_SCORE"]=mn[f"{key}_SCORE"].to_numpy(float)
pairs_out.to_csv(OUT["matching"]/"ONE_TO_ONE_MATCHED_PAIRS.csv",index=False)

support=pd.DataFrame([
    {
        "SET":"LATER_LINEAGE","N":len(pos),
        "RAW_CHARS_MEDIAN":float(pos.raw_chars.median()),
        "RAW_TURNS_MEDIAN":float(pos.raw_turns.median()),
        "RAW_CHARS_MIN":int(pos.raw_chars.min()),
        "RAW_CHARS_MAX":int(pos.raw_chars.max()),
        "RAW_TURNS_MIN":int(pos.raw_turns.min()),
        "RAW_TURNS_MAX":int(pos.raw_turns.max())
    },
    {
        "SET":"LM_HOLDOUT","N":len(hold),
        "RAW_CHARS_MEDIAN":float(hold.raw_chars.median()),
        "RAW_TURNS_MEDIAN":float(hold.raw_turns.median()),
        "RAW_CHARS_MIN":int(hold.raw_chars.min()),
        "RAW_CHARS_MAX":int(hold.raw_chars.max()),
        "RAW_TURNS_MIN":int(hold.raw_turns.min()),
        "RAW_TURNS_MAX":int(hold.raw_turns.max())
    },
    {
        "SET":"MATCHED_LM","N":len(mn),
        "RAW_CHARS_MEDIAN":float(mn.raw_chars.median()),
        "RAW_TURNS_MEDIAN":float(mn.raw_turns.median()),
        "RAW_CHARS_MIN":int(mn.raw_chars.min()),
        "RAW_CHARS_MAX":int(mn.raw_chars.max()),
        "RAW_TURNS_MIN":int(mn.raw_turns.min()),
        "RAW_TURNS_MAX":int(mn.raw_turns.max())
    },
])
support.to_csv(OUT["matching"]/"SUPPORT_DIAGNOSTICS.csv",index=False)

pd.DataFrame(results).to_csv(OUT["results"]/"BR2_V2_PRIMARY_RESULTS.csv",index=False)

gpt5_available=hold["MODEL"].astype(str).str.lower().str.contains("gpt-5",na=False).any()

ledger=[
    ["V1_RUN","FAILED_CALIBRATION","Stopped before LM scoring; preserved, not interpreted."],
    ["R1_EXACT_EXTRACTOR_LM_CONTROL","MEASURED","Same deterministic BR1 hard-blind extractor coordinates; portable frozen model."],
    ["SURFACE_FREE_EXACT_EXTRACTOR_LM_CONTROL","MEASURED","Same deterministic BR1 hard-blind extractor coordinates; portable frozen model."],
    ["OPERATOR_RELATIONAL_CROSS_CORPUS","UNAVAILABLE","Historical extractor used list(set)[:12] for some first-position coordinates, so exact cross-process reconstruction is not guaranteed until the representative subsets are recovered. Not false."],
    ["JOINT_MULTILAYER_CROSS_CORPUS","UNAVAILABLE","Depends on the same historical first-position coordinates. No approximation used."],
    ["LMARENA_DEVELOPMENT_HOLDOUT_PRIMARY","MEASURED",f"{len(hold)} independent pair controls after exact extraction."],
    ["PROVIDER_MATCHING","MEASURED" if pool_name=="OPENAI_HOLDOUT" else "UNAVAILABLE",pool_name],
    ["EXACT_MODEL_GPT5_MATCHING","MEASURED" if gpt5_available else "UNAVAILABLE","No substitution if GPT-5 is absent."],
    ["LANGUAGE_MATCHING","UNAVAILABLE","BR1 frozen feature package lacks a validated later-lineage language label."],
    ["TOPIC_MATCHING_LMARENA","UNAVAILABLE","No common frozen topic representation supplied in staged BR2 inputs; BR1 internal same-user topic control remains separate evidence."],
    ["LENGTH_TURN_COMPLEXITY_MATCHING","MEASURED",f"One-to-one assignment; max absolute SMD after matching={max_abs_smd:.4f}."],
    ["MATCH_QUALITY_GATE","MEASURED" if max_abs_smd<=0.25 else "FAILED_CALIBRATION",f"Gate max |SMD| <= 0.25; observed {max_abs_smd:.4f}."],
    ["F2_00086_WILDCHAT","RESERVED_NOT_READ","Guard aborts if reserved files appear under BR2 root."],
]
pd.DataFrame(ledger,columns=["MEASUREMENT","STATUS","DETAIL"]).to_csv(
    OUT["ledger"]/"NO_SILENT_OMISSION_LEDGER.csv",
    index=False
)

summary={
    "version":V,
    "completed_utc":utcnow(),
    "n_later_lineage":len(pos),
    "n_sameuser_non_lineage":len(neg),
    "n_lm_holdout_independent_pairs":len(hold),
    "n_matched_pairs":len(mp),
    "matching_pool":pool_name,
    "max_abs_smd_after":max_abs_smd,
    "primary_results":results,
    "matched_results":pair_rows,
    "operator_joint_status":"UNAVAILABLE_FOR_EXACT_CROSS_PROCESS_REEXTRACTION_DUE_HISTORICAL_SET_ORDER_NONDETERMINISM",
    "wildchat_00086_read":False,
}
write_json(OUT["manifest"]/"BR2_V2_SUMMARY.json",summary)

lines=[
    "F1-BR2 V2 — EXACT-EXTRACTOR LM ARENA MATCHED CONTROL",
    "",
    f"Later lineage holdout: {len(pos)}",
    f"Same-user non-lineage holdout: {len(neg)}",
    f"Independent LM Arena development-holdout pairs: {len(hold)}",
    f"One-to-one matched pairs: {len(mp)}",
    f"Matching pool: {pool_name}",
    f"Max |SMD| after matching: {max_abs_smd:.4f}",
    "",
]
for r in results:
    lines.append(
        f"{r['TEST']} | {r['MODEL']} | "
        f"AUC={r['AUC']:.6f} | 95% CI [{r['CI95_LOW']:.6f},{r['CI95_HIGH']:.6f}] | "
        f"p={r['P_ONE_SIDED']:.6g}"
    )
lines.append("")
for r in pair_rows:
    lines.append(
        f"MATCHED | {r['MODEL']} | AUC={r['MATCHED_AUC']:.6f} | "
        f"pair-win={r['PAIR_WIN_RATE']:.6f} | binom p={r['BINOMIAL_P']:.6g} | "
        f"signflip p={r['SIGNFLIP_P']:.6g}"
    )
lines += [
    "",
    "R1 + SURFACE_FREE: MEASURED using exact deterministic extractor.",
    "OPERATOR + JOINT cross-corpus: UNAVAILABLE for exact reconstruction in V2 because historical first-position extraction used unordered set truncation. No approximation used.",
    "00086/WildChat read: NO.",
]
(OUT["manifest"]/"BR2_V2_SUMMARY.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")

write_json(OUT["manifest"]/"ENVIRONMENT.json",{
    "python":sys.version,
    "platform":platform.platform(),
    "numpy":np.__version__,
    "pandas":pd.__version__,
    "scoring":"portable frozen scaler + logistic coefficients; sklearn joblib unpickling not used"
})

inrows=[]
for p in sorted(INPUTS.glob("*")):
    if p.is_file():
        inrows.append({
            "FILE":p.name,
            "BYTES":p.stat().st_size,
            "SHA256":sha256_file(p)
        })
pd.DataFrame(inrows).to_csv(OUT["manifest"]/"V2_INPUT_SHA256.csv",index=False)

MARKER.write_text(f"{V}\n{utcnow()}\n",encoding="ascii")

outrows=[]
for p in sorted(ROOT.rglob("*")):
    if p.is_file() and "COMPLETE_V2.zip" not in p.name:
        outrows.append({
            "RELATIVE_PATH":str(p.relative_to(ROOT)),
            "BYTES":p.stat().st_size,
            "SHA256":sha256_file(p)
        })
pd.DataFrame(outrows).to_csv(OUT["manifest"]/"V2_OUTPUT_SHA256_MANIFEST.csv",index=False)

print()
print("="*86)
print("F1-BR2 V2 COMPLETE")
print("="*86)
print((OUT["manifest"]/"BR2_V2_SUMMARY.txt").read_text(encoding="utf-8"))
