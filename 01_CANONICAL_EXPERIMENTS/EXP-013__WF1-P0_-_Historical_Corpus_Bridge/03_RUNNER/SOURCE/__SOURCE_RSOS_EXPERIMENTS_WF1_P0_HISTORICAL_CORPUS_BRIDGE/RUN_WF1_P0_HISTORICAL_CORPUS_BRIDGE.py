#!/usr/bin/env python3
"""
WF1-P0 — Historical Corpus Structural-Dynamic Bridge
DISCOVERY/AUDIT ONLY. Not a confirmatory test.

Purpose
-------
Bridge the frozen claim-discovery bundle to the canonical historical ChatGPT
corpus without using any public external corpus. Extract static structure,
temporal/trajectory features, generic structural-family activations,
user→assistant coupling, feedback/cycle motifs, and claim-ranked effect sizes.

Important
---------
- Public parquet data MUST NOT be supplied to this runner.
- This runner does not alter V12/F1/F2 frozen results.
- Outputs are hypothesis-generation inputs for later frozen H1/WF1 tests.
"""
from __future__ import annotations
import argparse, collections, csv, datetime as dt, hashlib, io, json, math, os, re, sys, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

VERSION = "WF1-P0-v1.0"
TOKEN_RE = re.compile(r"\b[\w'’-]+\b", re.UNICODE)

FAMILIES = {
    "recursion": [r"\brecurs\w*", r"\bfeedback\b", r"\bloop\w*\b", r"\bself[- ]?refer\w*"],
    "control": [r"\bcontrol\w*", r"\boverride\w*", r"\bregulat\w*", r"\bcybernet\w*", r"\bgovern\w*"],
    "hierarchy": [r"\bhierarch\w*", r"\btier\w*", r"\blevel\w*", r"\blayer\w*", r"\bnested\b"],
    "topology": [r"\btopolog\w*", r"\bgraph\w*", r"\bedge\w*", r"\bnode\w*", r"\bnetwork\w*", r"\brelational\b"],
    "trajectory": [r"\btraject\w*", r"\bstate[- ]?space\b", r"\btransition\w*", r"\bcurvature\b", r"\bconverg\w*", r"\bdwell\b", r"\bphase\b"],
    "reconstruction": [r"\breconstruct\w*", r"\brecover\w*", r"\brestore\w*", r"\brebuild\w*", r"\bregenerat\w*"],
    "persistence": [r"\bpersist\w*", r"\bsurviv\w*", r"\bcontinu\w*", r"\blongitud\w*", r"\bretention\b"],
    "compression": [r"\bcompress\w*", r"\bminimal[- ]?description\b", r"\bgenome\b", r"\bencode\w*", r"\bdecode\w*"],
    "meta": [r"\bmeta[- ]?\w*", r"\bobserver\b", r"\bcontroller\b", r"\bself[- ]?model\b", r"\bmodel[- ]?of[- ]?model\b"],
    "coupling": [r"\bcoupl\w*", r"\binteraction\w*", r"\bco[- ]?adapt\w*", r"\bbidirectional\b", r"\blagged\b"],
    "genealogy": [r"\bgenealog\w*", r"\bancestor\w*", r"\bdescendant\w*", r"\blineage\b", r"\bherit\w*"],
    "generativity": [r"\bgenerativ\w*", r"\bgenerate\w*", r"\bdescendant\w*", r"\boffspring\b", r"\binherit\w*"],
    "constraints": [r"\bconstraint\w*", r"\bdependency\b", r"\binvariant\w*", r"\bgate\w*", r"\brule\w*"],
    "composition": [r"\bcompos\w*", r"\bdecompos\w*", r"\bmodul\w*", r"\bfusion\b", r"\bsynthesis\b"],
    "symbolic": [r"\bsymbol\w*", r"\bglyph\w*", r"\bsigil\b", r"\barchetyp\w*", r"\bsemantic\b"],
    "cross_model": [r"\bcross[- ]?model\b", r"\bprovider\b", r"\bmodel famil\w*", r"\bmodel[- ]?independent\b"],
    "cross_domain": [r"\bcross[- ]?domain\b", r"\btransfer\b", r"\bgeneraliz\w*", r"\bdomain\w*"],
    "information": [r"\binformation\b", r"\bentropy\b", r"\bdensity\b", r"\bmutual information\b", r"\bbit\w*\b"],
    "robustness": [r"\brobust\w*", r"\bdamage\b", r"\bperturb\w*", r"\bablat\w*", r"\bdropout\b", r"\bresilien\w*"],
}
FAM_NAMES = list(FAMILIES)
FAM_RX = {k:[re.compile(p,re.I) for p in pats] for k,pats in FAMILIES.items()}

CAUSAL = re.compile(r"\b(because|therefore|thus|hence|causes?|causal|leads? to|results? in|so that|due to)\b", re.I)
COND = re.compile(r"\b(if|then|when|unless|otherwise|provided that|given that)\b", re.I)
SELF_REF = re.compile(r"\b(I|me|my|mine|myself|we|our|ours)\b", re.I)
META_REF = re.compile(r"\b(model|system|architecture|prompt|instruction|context|memory|state|process|observer|controller)\b", re.I)

def sha256(path: Path, chunk=1024*1024):
    h=hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def safe_div(a,b):
    return float(a)/float(b) if b else 0.0

def cosine(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    return float(np.dot(a,b)/(na*nb)) if na and nb else 0.0

def entropy_from_counts(vals):
    vals=np.asarray(vals,float)
    s=vals.sum()
    if s<=0: return 0.0
    p=vals[vals>0]/s
    return float(-(p*np.log2(p)).sum())

def text_features(text: str):
    text = text or ""
    toks = TOKEN_RE.findall(text.lower())
    n=max(len(toks),1)
    uniq=len(set(toks))
    lines=text.splitlines()
    sent=max(1,len(re.findall(r"[.!?]+(?:\s|$)", text)))
    out={
        "word_count":len(toks),
        "log_words":math.log1p(len(toks)),
        "type_token_ratio":safe_div(uniq,len(toks)),
        "avg_word_len":safe_div(sum(len(t) for t in toks),len(toks)),
        "sentence_count":sent,
        "question_rate":safe_div(text.count("?"),sent),
        "exclaim_rate":safe_div(text.count("!"),sent),
        "colon_per_1k":1000*safe_div(text.count(":"),n),
        "semicolon_per_1k":1000*safe_div(text.count(";"),n),
        "paren_per_1k":1000*safe_div(text.count("(")+text.count(")"),n),
        "bracket_per_1k":1000*safe_div(text.count("[")+text.count("]"),n),
        "arrow_per_1k":1000*safe_div(len(re.findall(r"(?:->|→|=>|⇒)",text)),n),
        "heading_per_1k":1000*safe_div(sum(bool(re.match(r"^\s*#{1,6}\s",x)) for x in lines),n),
        "bullet_per_1k":1000*safe_div(sum(bool(re.match(r"^\s*[-*+]\s+",x)) for x in lines),n),
        "numbered_per_1k":1000*safe_div(sum(bool(re.match(r"^\s*\d+[.)]\s+",x)) for x in lines),n),
        "code_fence_per_1k":1000*safe_div(text.count("```")/2,n),
        "causal_per_1k":1000*safe_div(len(CAUSAL.findall(text)),n),
        "conditional_per_1k":1000*safe_div(len(COND.findall(text)),n),
        "self_ref_per_1k":1000*safe_div(len(SELF_REF.findall(text)),n),
        "meta_ref_per_1k":1000*safe_div(len(META_REF.findall(text)),n),
    }
    for fam,rxs in FAM_RX.items():
        c=sum(len(rx.findall(text)) for rx in rxs)
        out["fam_"+fam]=1000*safe_div(c,n)
    return out

def extract_text_from_message(msg):
    if not isinstance(msg,dict): return ""
    content=msg.get("content")
    if isinstance(content,dict):
        parts=content.get("parts")
        if isinstance(parts,list):
            xs=[]
            for p in parts:
                if isinstance(p,str): xs.append(p)
                elif isinstance(p,dict): xs.append(json.dumps(p,ensure_ascii=False))
            return "\n".join(xs)
        if isinstance(content.get("text"),str): return content["text"]
    return ""

def conversation_objects(obj):
    if isinstance(obj,list):
        for x in obj:
            if isinstance(x,dict) and ("mapping" in x or "messages" in x):
                yield x
    elif isinstance(obj,dict):
        if "mapping" in obj or "messages" in obj:
            yield obj
        else:
            for v in obj.values():
                if isinstance(v,(list,dict)):
                    yield from conversation_objects(v)

def load_corpus_zip(path: Path):
    rows=[]
    seen=set()
    with zipfile.ZipFile(path) as z:
        json_names=[n for n in z.namelist() if n.lower().endswith(".json")]
        for name in json_names:
            try:
                obj=json.loads(z.read(name).decode("utf-8",errors="replace"))
            except Exception:
                continue
            for conv in conversation_objects(obj):
                cid=str(conv.get("id") or conv.get("conversation_id") or "")
                title=str(conv.get("title") or "")
                if not cid:
                    cid=hashlib.sha1((name+"|"+title+"|"+str(len(rows))).encode()).hexdigest()
                mapping=conv.get("mapping")
                if isinstance(mapping,dict):
                    local=[]
                    for node_id,node in mapping.items():
                        if not isinstance(node,dict): continue
                        msg=node.get("message")
                        if not isinstance(msg,dict): continue
                        author=msg.get("author") or {}
                        role=author.get("role") if isinstance(author,dict) else ""
                        if role not in ("user","assistant"): continue
                        text=extract_text_from_message(msg)
                        if not text.strip(): continue
                        ct=msg.get("create_time")
                        try: ct=float(ct) if ct is not None else np.nan
                        except Exception: ct=np.nan
                        mid=str(msg.get("id") or node_id)
                        key=(cid,mid)
                        if key in seen: continue
                        seen.add(key)
                        local.append((ct,mid,role,text))
                    local.sort(key=lambda x: (float("inf") if pd.isna(x[0]) else x[0], x[1]))
                    for order,(ct,mid,role,text) in enumerate(local):
                        rows.append({"conversation_id":cid,"conversation_title":title,"message_id":mid,
                                     "create_time":ct,"turn_order":order,"role":role,"text":text})
                elif isinstance(conv.get("messages"),list):
                    for order,msg in enumerate(conv["messages"]):
                        role=msg.get("role") or (msg.get("author") or {}).get("role")
                        if role not in ("user","assistant"): continue
                        text=msg.get("text") or extract_text_from_message(msg)
                        if not str(text).strip(): continue
                        mid=str(msg.get("id") or order)
                        key=(cid,mid)
                        if key in seen: continue
                        seen.add(key)
                        ct=msg.get("create_time")
                        try: ct=float(ct) if ct is not None else np.nan
                        except Exception: ct=np.nan
                        rows.append({"conversation_id":cid,"conversation_title":title,"message_id":mid,
                                     "create_time":ct,"turn_order":order,"role":role,"text":str(text)})
    if not rows:
        raise RuntimeError("No user/assistant messages could be extracted from the corpus ZIP.")
    return pd.DataFrame(rows)

def read_claim_bundle(path: Path):
    with zipfile.ZipFile(path) as z:
        def rd(name, **kw):
            with z.open(name) as f:
                return pd.read_csv(f, **kw)
        fam=rd("20_FAMILY_COUNTS.csv")
        rank=rd("21_CONVERSATION_RANKING.csv")
        fmap=rd("17_CLAIM_TO_MEASURABLE_FEATURE_MAP.csv")
        cols=["conversation_id","conversation_title","create_time","role","families","score"]
        uh=rd("14_ULTRA_HIGH_SIGNAL.csv",usecols=cols)
    return fam,rank,fmap,uh

def dominant_family(row):
    vals=[row.get("fam_"+f,0.0) for f in FAM_NAMES]
    if max(vals,default=0)<=0: return "none"
    return FAM_NAMES[int(np.argmax(vals))]

def build_message_features(msgs):
    records=[]
    for _,r in msgs.iterrows():
        f=text_features(str(r["text"]))
        rec={k:r[k] for k in ["conversation_id","conversation_title","message_id","create_time","turn_order","role"]}
        rec.update(f)
        records.append(rec)
    df=pd.DataFrame(records)
    df["dominant_family"]=df.apply(dominant_family,axis=1)
    return df

def conversation_features(mf):
    structural_cols=[c for c in mf.columns if c not in {
        "conversation_id","conversation_title","message_id","create_time","turn_order","role","dominant_family"
    }]
    fam_cols=["fam_"+x for x in FAM_NAMES]
    out=[]
    io_edges=collections.Counter()
    dom_edges=collections.Counter()
    for cid,g in mf.groupby("conversation_id",sort=False):
        g=g.sort_values(["turn_order","create_time"],na_position="last")
        X=g[structural_cols].fillna(0).to_numpy(float)
        Xg=np.log1p(np.maximum(X,0))
        deltas=np.diff(Xg,axis=0) if len(Xg)>1 else np.empty((0,Xg.shape[1]))
        dnorm=np.linalg.norm(deltas,axis=1) if len(deltas) else np.array([])
        transcos=[cosine(Xg[i],Xg[i+1]) for i in range(len(Xg)-1)]
        curv=[1.0-cosine(deltas[i],deltas[i+1]) for i in range(len(deltas)-1)]
        recur=[]
        for i in range(2,len(Xg)):
            recur.append(max(cosine(Xg[i],Xg[j]) for j in range(i-1)))
        dom=list(g["dominant_family"])
        dwell=safe_div(sum(dom[i]==dom[i-1] and dom[i]!="none" for i in range(1,len(dom))),max(1,len(dom)-1))
        aba=safe_div(sum(dom[i]==dom[i-2] and dom[i]!="none" and dom[i]!=dom[i-1] for i in range(2,len(dom))),max(1,len(dom)-2))
        for a,b in zip(dom[:-1],dom[1:]):
            dom_edges[(a,b)]+=1
        roles=list(g["role"])
        io_cos=[]; io_gain=[]
        for i in range(len(g)-1):
            if roles[i]=="user" and roles[i+1]=="assistant":
                a=Xg[i]; b=Xg[i+1]
                io_cos.append(cosine(a,b))
                io_gain.append(safe_div(np.linalg.norm(b),np.linalg.norm(a)))
                io_edges[(dom[i],dom[i+1])]+=1
        fam_sum=g[fam_cols].sum().to_numpy(float)
        rec={
            "conversation_id":cid,
            "conversation_title":str(g["conversation_title"].iloc[0]),
            "n_messages":len(g),
            "n_user":int((g["role"]=="user").sum()),
            "n_assistant":int((g["role"]=="assistant").sum()),
            "start_time":pd.to_numeric(g["create_time"],errors="coerce").min(),
            "end_time":pd.to_numeric(g["create_time"],errors="coerce").max(),
            "mean_delta_norm":float(np.mean(dnorm)) if len(dnorm) else 0.0,
            "sd_delta_norm":float(np.std(dnorm)) if len(dnorm) else 0.0,
            "mean_transition_cosine":float(np.mean(transcos)) if transcos else 0.0,
            "mean_curvature":float(np.mean(curv)) if curv else 0.0,
            "mean_recurrence":float(np.mean(recur)) if recur else 0.0,
            "dominant_dwell_rate":dwell,
            "ABA_feedback_rate":aba,
            "dominant_family_entropy":entropy_from_counts(fam_sum),
            "lagged_user_assistant_cosine":float(np.mean(io_cos)) if io_cos else 0.0,
            "lagged_user_assistant_gain":float(np.mean(io_gain)) if io_gain else 0.0,
            "n_user_assistant_pairs":len(io_cos),
            "unique_dominant_states":len(set(x for x in dom if x!="none")),
        }
        for c in structural_cols:
            rec["mean_"+c]=float(g[c].mean())
            rec["max_"+c]=float(g[c].max())
        out.append(rec)
    return pd.DataFrame(out), io_edges, dom_edges

def cohen_d(a,b):
    a=pd.to_numeric(pd.Series(a),errors="coerce").dropna().to_numpy(float)
    b=pd.to_numeric(pd.Series(b),errors="coerce").dropna().to_numpy(float)
    if len(a)<2 or len(b)<2: return np.nan
    va=np.var(a,ddof=1); vb=np.var(b,ddof=1)
    pooled=((len(a)-1)*va+(len(b)-1)*vb)/(len(a)+len(b)-2)
    if pooled<=0: return 0.0
    return float((np.mean(a)-np.mean(b))/math.sqrt(pooled))

def claim_measure_summary(fmap):
    cnt=collections.Counter(); conv=collections.defaultdict(set)
    for _,r in fmap.iterrows():
        for x in [x.strip() for x in str(r.get("candidate_measures","")).split("|") if x.strip()]:
            cnt[x]+=1; conv[x].add(str(r["conversation_id"]))
    return pd.DataFrame([{"candidate_measure":k,"rows":v,"unique_conversations":len(conv[k])}
                         for k,v in cnt.most_common()])

def claim_pair_lift(fmap):
    import itertools
    N=len(fmap); single=collections.Counter(); pairs=collections.Counter()
    for s in fmap["families"].fillna(""):
        fs=sorted(set(x.strip() for x in str(s).split("|") if x.strip()))
        single.update(fs)
        for a,b in itertools.combinations(fs,2): pairs[(a,b)]+=1
    rows=[]
    for (a,b),c in pairs.items():
        exp=single[a]*single[b]/N if N else 0
        if c>=10 and exp>0:
            rows.append({"family_a":a,"family_b":b,"cooccurrence_rows":c,
                         "expected_independence":exp,"lift":c/exp})
    return pd.DataFrame(rows).sort_values(["lift","cooccurrence_rows"],ascending=False)

def edge_df(counter, name_a="from_state", name_b="to_state"):
    total=sum(counter.values())
    from_tot=collections.Counter()
    for (a,b),c in counter.items(): from_tot[a]+=c
    rows=[]
    for (a,b),c in counter.items():
        rows.append({name_a:a,name_b:b,"count":c,
                     "p_to_given_from":safe_div(c,from_tot[a]),
                     "global_fraction":safe_div(c,total)})
    return pd.DataFrame(rows).sort_values("count",ascending=False)

def monthly_trajectory(cf):
    x=cf.copy()
    x["start_dt"]=pd.to_datetime(x["start_time"],unit="s",utc=True,errors="coerce")
    x=x[x["start_dt"].notna()]
    x["month"]=x["start_dt"].dt.strftime("%Y-%m")
    numeric=[c for c in x.columns if c not in {
        "conversation_id","conversation_title","start_dt","month"
    } and pd.api.types.is_numeric_dtype(x[c])]
    if not len(x): return pd.DataFrame()
    return x.groupby("month")[numeric].agg(["mean","median","count"]).reset_index()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--corpus",required=True)
    ap.add_argument("--claims",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    corpus=Path(args.corpus); claims=Path(args.claims); out=Path(args.out)
    if not corpus.exists(): raise FileNotFoundError(corpus)
    if not claims.exists(): raise FileNotFoundError(claims)
    out.mkdir(parents=True,exist_ok=True)
    protocol = f"""WF1-P0 HISTORICAL CORPUS STRUCTURAL-DYNAMIC BRIDGE
Version: {VERSION}
STATUS: DISCOVERY/AUDIT ONLY
PUBLIC EXTERNAL DATA: PROHIBITED IN THIS PHASE

Goals:
1. Map recovered historical claim families to measurable corpus features.
2. Measure static structural vectors, trajectories, recurrence, feedback motifs,
   user→assistant coupling, and role-conditioned transformations.
3. Identify candidate mechanisms not already exhausted by F1/F2/V12.
4. Generate a feature registry to freeze BEFORE any public parquet confirmation.

This phase MUST NOT:
- alter frozen V12/F1/F2 conclusions,
- use public parquet to select features,
- claim external rarity,
- claim population uniqueness,
- claim internal model mechanisms.
"""
    (out/"00_PROTOCOL.txt").write_text(protocol,encoding="utf-8")
    pd.DataFrame([
        {"input":"corpus","path":str(corpus.resolve()),"sha256":sha256(corpus),"bytes":corpus.stat().st_size},
        {"input":"claim_bundle","path":str(claims.resolve()),"sha256":sha256(claims),"bytes":claims.stat().st_size},
    ]).to_csv(out/"00_INPUT_SHA256.csv",index=False)
    fam,rank,fmap,uh=read_claim_bundle(claims)
    fam.to_csv(out/"01_CLAIM_FAMILY_COUNTS.csv",index=False)
    rank.to_csv(out/"02_CLAIM_CONVERSATION_RANKING.csv",index=False)
    cms=claim_measure_summary(fmap)
    cms.to_csv(out/"03_CANDIDATE_MEASURE_COUNTS.csv",index=False)
    claim_pair_lift(fmap).to_csv(out/"04_CLAIM_FAMILY_PAIR_LIFT.csv",index=False)

    print("Loading canonical corpus...")
    msgs=load_corpus_zip(corpus)
    print("Messages:",len(msgs),"Conversations:",msgs["conversation_id"].nunique())
    mf=build_message_features(msgs)
    mf.to_csv(out/"05_MESSAGE_STRUCTURAL_FEATURES.csv",index=False)
    cf,io_edges,dom_edges=conversation_features(mf)
    cf.to_csv(out/"06_CONVERSATION_STRUCTURAL_DYNAMIC_FEATURES.csv",index=False)
    edge_df(io_edges,"user_state","assistant_state").to_csv(out/"07_USER_TO_ASSISTANT_TRANSITIONS.csv",index=False)
    edge_df(dom_edges).to_csv(out/"08_DOMINANT_STATE_TRANSITION_GRAPH.csv",index=False)
    monthly_trajectory(cf).to_csv(out/"09_MONTHLY_TRAJECTORIES.csv",index=False)

    high_ids=set(str(x) for x in uh["conversation_id"].dropna().unique())
    ranked_ids=set(str(x) for x in rank["conversation_id"].dropna().unique())
    cf["ultra_high_signal_claim_conversation"]=cf["conversation_id"].astype(str).isin(high_ids)
    cf["claim_ranked_conversation"]=cf["conversation_id"].astype(str).isin(ranked_ids)
    numeric=[c for c in cf.columns if pd.api.types.is_numeric_dtype(cf[c])]
    effects=[]
    for label in ["ultra_high_signal_claim_conversation","claim_ranked_conversation"]:
        A=cf[cf[label]]; B=cf[~cf[label]]
        for c in numeric:
            effects.append({"comparison":label,"feature":c,
                            "n_positive":len(A),"n_control":len(B),
                            "mean_positive":pd.to_numeric(A[c],errors="coerce").mean(),
                            "mean_control":pd.to_numeric(B[c],errors="coerce").mean(),
                            "cohen_d":cohen_d(A[c],B[c])})
    eff=pd.DataFrame(effects)
    eff["abs_d"]=eff["cohen_d"].abs()
    eff.sort_values(["comparison","abs_d"],ascending=[True,False]).to_csv(out/"10_DISCOVERY_FEATURE_EFFECTS.csv",index=False)

    wanted = {
        "transition":"transition law, curvature, convergence, dwell",
        "trajectory":"trajectory/state-space geometry",
        "topology":"graph motifs, cycles, SCC, feedback edges",
        "meta":"nested meta-system/observer-controller depth",
        "coupling":"lagged user-assistant structural coupling",
        "reconstruction":"minimal-description reconstruction fidelity",
        "symbolic_reconstruction":"symbolic compression/reconstruction fidelity",
        "genealogy":"ancestor-descendant conservation",
        "generation":"descendant/module generation rate",
        "recovery":"perturbation/damage recovery",
    }
    sets=[]
    for _,r in fmap.iterrows():
        s=set(x.strip() for x in str(r.get("candidate_measures","")).split("|") if x.strip())
        sets.append((str(r["conversation_id"]),s))
    combos=[
        ("transition+trajectory",["transition","trajectory"]),
        ("transition+topology",["transition","topology"]),
        ("transition+recovery",["transition","recovery"]),
        ("transition+meta",["transition","meta"]),
        ("transition+coupling",["transition","coupling"]),
        ("transition+trajectory+topology",["transition","trajectory","topology"]),
        ("transition+trajectory+topology+meta",["transition","trajectory","topology","meta"]),
        ("transition+trajectory+topology+meta+coupling",["transition","trajectory","topology","meta","coupling"]),
        ("reconstruction+symbolic_reconstruction",["reconstruction","symbolic_reconstruction"]),
        ("genealogy+generation",["genealogy","generation"]),
    ]
    cr=[]
    for name,keys in combos:
        vals=[wanted[k] for k in keys]
        hits=[(cid,s) for cid,s in sets if all(v in s for v in vals)]
        cr.append({"combination":name,"rows":len(hits),"unique_conversations":len(set(cid for cid,_ in hits))})
    pd.DataFrame(cr).to_csv(out/"11_CANDIDATE_MEASURE_CONJUNCTIONS.csv",index=False)

    top=eff[eff["comparison"]=="ultra_high_signal_claim_conversation"].sort_values("abs_d",ascending=False).head(50)
    top.to_csv(out/"12_TOP50_DISCOVERY_FEATURES.csv",index=False)

    files=sorted([p for p in out.iterdir() if p.is_file() and p.name!="99_OUTPUT_SHA256.csv"])
    hashes=pd.DataFrame([{"file":p.name,"sha256":sha256(p),"bytes":p.stat().st_size} for p in files])
    hashes.to_csv(out/"99_OUTPUT_SHA256.csv",index=False)
    freeze={
        "experiment":"WF1-P0 Historical Corpus Structural-Dynamic Bridge",
        "version":VERSION,
        "status":"DISCOVERY_ONLY_COMPLETE",
        "utc":dt.datetime.now(dt.timezone.utc).isoformat(),
        "corpus_sha256":sha256(corpus),
        "claims_sha256":sha256(claims),
        "messages":int(len(msgs)),
        "conversations":int(msgs["conversation_id"].nunique()),
        "public_external_data_used":False,
        "next_required_step":"Adjudicate discovery outputs, define/freeze H1 feature registry, then and only then run untouched public external confirmation.",
    }
    (out/"99_DISCOVERY_FREEZE.json").write_text(json.dumps(freeze,indent=2),encoding="utf-8")
    print("\nWF1-P0 INTERNAL DISCOVERY COMPLETE")
    print("Output:",out)
    print("Public parquet used: NO")
    print("Next: adjudicate -> feature freeze -> untouched external confirmation")

if __name__=="__main__":
    main()
