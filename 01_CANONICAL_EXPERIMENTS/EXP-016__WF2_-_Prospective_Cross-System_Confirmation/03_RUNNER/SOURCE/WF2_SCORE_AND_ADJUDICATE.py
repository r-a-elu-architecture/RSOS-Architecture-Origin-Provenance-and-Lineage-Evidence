from pathlib import Path
from collections import defaultdict
import argparse, csv, hashlib, importlib.util, json, re
import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT=Path(r"C:\RSOS\RSOS_EXPERIMENTS_WF2_PROSPECTIVE_CROSS_SYSTEM_CONFIRMATION")
FOUND=ROOT/"02_INPUTS"/"00_FOUNDATION_FROZEN"
PLANROOT=ROOT/"02_INPUTS"/"01_EXECUTION_PLAN"
DET=FOUND/"DETECTOR_PACKAGE"
RAW=ROOT/"04_RAW"/"CELLS"
OUT=ROOT/"05_OUTPUTS"
DIAG=ROOT/"06_DIAGNOSTIC"
ADJ=ROOT/"07_ADJUDICATION"

DIRECT_RE=re.compile(r"\b(?:RSOS|RSSO|RSIA|RSX)\b",re.I)

def load_jsonl(p):
    out=[]
    with p.open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out

def load_adapter():
    p=DET/"RUN_WF1_B_FRESH14_ADAPTER_R3.py"
    s=importlib.util.spec_from_file_location("wf2adapter",p)
    m=importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

def auc(pos,neg):
    pos=np.asarray(pos,float); neg=np.asarray(neg,float)
    return float(sum(np.sum(p>neg)+.5*np.sum(p==neg) for p in pos)/(len(pos)*len(neg)))

def seed(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8],16)

def auc_ci(pos,neg,key,reps=2000):
    r=np.random.default_rng(seed(key)); vals=[]
    pos=np.asarray(pos,float); neg=np.asarray(neg,float)
    for _ in range(reps):
        vals.append(auc(
            r.choice(pos,len(pos),replace=True),
            r.choice(neg,len(neg),replace=True)
        ))
    return float(np.quantile(vals,.025)),float(np.quantile(vals,.975))

def holm(ps):
    n=len(ps); order=sorted(range(n),key=lambda i:ps[i])
    out=[1.0]*n; running=0.
    for rank,i in enumerate(order):
        running=max(running,(n-rank)*ps[i])
        out[i]=min(1.,running)
    return out

def transform(messages,name):
    out=[]
    for m in messages:
        t=m["content"]
        if name=="SURFACE_TERM_REDACT":
            t=DIRECT_RE.sub(lambda x:" "*len(x.group(0)),t)
        elif name=="SURFACE_MARKDOWN_FLATTEN":
            t=re.sub(r"(?m)^\s*#{1,6}\s*","",t)
            t=re.sub(r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+)","",t)
            t=t.replace("```","")
            t=re.sub(r"->|=>|→|⇒"," ",t)
        elif name=="SURFACE_CASE_PUNCT_NORMALIZED":
            t=re.sub(r"\s+"," ",re.sub(r"[^a-z0-9\s]"," ",t.lower())).strip()
        out.append({"role":m["role"],"content":t})
    return out

def score(adapter,models,wmodels,messages):
    msgs=adapter.normalize_messages(messages)
    feat,chars,turns,hits=adapter.main_features(msgs)
    df=pd.DataFrame([feat])
    row={"RAW_CHARS":chars,"RAW_TURNS":turns,"SYSTEM_NAME_HITS":hits}

    for name,model in models.items():
        cols=adapter.model_cols(name,model)
        row[name+"_MARGIN"]=float(model.decision_function(df[cols])[0])

    for w,model in wmodels.items():
        q=adapter.aggregate_window(msgs,w)
        if q is None:
            row[f"WINDOW{w}_MARGIN"]=float("nan")
        else:
            x=pd.DataFrame([q])
            cols=list(model.feature_names_in_)
            row[f"WINDOW{w}_MARGIN"]=float(model.decision_function(x[cols])[0])

    return row

def paired(rows,metric,key):
    p=defaultdict(dict)
    for r in rows:
        p[int(r["pair_index"])][r["label"]]=float(r[metric])

    z=[
        (i,v["LINEAGE_POS"],v["MATCHED_NEG"])
        for i,v in p.items()
        if set(v)>={"LINEAGE_POS","MATCHED_NEG"}
    ]

    z.sort()
    pos=np.array([x[1] for x in z])
    neg=np.array([x[2] for x in z])
    d=pos-neg

    if not len(z):
        return None

    pg=float(wilcoxon(d,alternative="greater",method="auto").pvalue)
    pl=float(wilcoxon(d,alternative="less",method="auto").pvalue)
    a=auc(pos,neg); lo,hi=auc_ci(pos,neg,key+"|"+metric)

    return {
        "metric":metric,
        "complete_pairs":len(z),
        "auc":a,
        "auc_ci_low":lo,
        "auc_ci_high":hi,
        "mean_paired_delta":float(np.mean(d)),
        "median_paired_delta":float(np.median(d)),
        "p_greater":pg,
        "p_less":pl
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--parity-only",action="store_true")
    args=ap.parse_args()

    adapter=load_adapter()
    checked=adapter.parity_gate()
    print("Frozen extractor parity: PASS",checked)

    if args.parity_only:
        print("PARITY-ONLY — PROSPECTIVE DATA READ: NO")
        return

    OUT.mkdir(parents=True,exist_ok=True)
    DIAG.mkdir(parents=True,exist_ok=True)
    ADJ.mkdir(parents=True,exist_ok=True)

    labels={x["blind_context_id"]:x for x in load_jsonl(PLANROOT/"WF2_LABEL_KEY.jsonl")}
    plan=load_jsonl(PLANROOT/"WF2_EXECUTION_PLAN.jsonl")

    models={
        n:joblib.load(DET/"frozen_models"/f"{n}.joblib")
        for n in [
            "JOINT_MULTILAYER_V2",
            "OPERATOR_RELATIONAL_V2",
            "SURFACE_FREE",
            "R1_COMPATIBLE"
        ]
    }

    wmodels={
        w:joblib.load(DET/"frozen_models"/f"WINDOW{w}_DETERMINISTIC_V2.joblib")
        for w in [8,12]
    }

    scored=[]; failed=[]

    for cell in plan:
        p=RAW/f"{cell['cell_id']}.json"

        if not p.exists():
            failed.append({"cell_id":cell["cell_id"],"reason":"MISSING"})
            continue

        raw=json.loads(p.read_text(encoding="utf-8"))

        if raw.get("status")!="COMPLETE":
            failed.append({"cell_id":cell["cell_id"],"reason":raw.get("status")})
            continue

        k=labels[raw["blind_context_id"]]

        conditions=[("RAW",raw["condition"])]

        if raw["provider"]=="OPENAI" and raw["condition"]=="BASELINE":
            conditions += [
                ("SURFACE_TERM_REDACT","SURFACE_TERM_REDACT"),
                ("SURFACE_MARKDOWN_FLATTEN","SURFACE_MARKDOWN_FLATTEN"),
                ("SURFACE_CASE_PUNCT_NORMALIZED","SURFACE_CASE_PUNCT_NORMALIZED")
            ]

        for trans,outcond in conditions:
            msgs=raw["scoring_messages"] if trans=="RAW" else transform(raw["scoring_messages"],trans)

            r={
                "provider":raw["provider"],
                "condition":outcond,
                "cell_id":raw["cell_id"],
                "pair_index":k["pair_index"],
                "label":k["label"]
            }
            r.update(score(adapter,models,wmodels,msgs))
            scored.append(r)

    pd.DataFrame(scored).to_csv(OUT/"WF2_CELL_SCORES.csv",index=False)

    if failed:
        pd.DataFrame(failed).to_csv(DIAG/"WF2_FAILED_OR_MISSING_CELLS.csv",index=False)

    groups=defaultdict(list)
    for r in scored:
        groups[(r["provider"],r["condition"])].append(r)

    metrics=[
        "JOINT_MULTILAYER_V2_MARGIN",
        "OPERATOR_RELATIONAL_V2_MARGIN",
        "SURFACE_FREE_MARGIN",
        "R1_COMPATIBLE_MARGIN",
        "WINDOW8_MARGIN",
        "WINDOW12_MARGIN"
    ]

    stats=[]

    for (provider,condition),rs in groups.items():
        for metric in metrics:
            s=paired(rs,metric,provider+"|"+condition)
            if s:
                s["provider"]=provider
                s["condition"]=condition
                stats.append(s)

    pd.DataFrame(stats).to_csv(OUT/"WF2_CONDITION_STATISTICS.csv",index=False)

    def get(provider,condition,metric="JOINT_MULTILAYER_V2_MARGIN"):
        for s in stats:
            if s["provider"]==provider and s["condition"]==condition and s["metric"]==metric:
                return s
        return None

    def pos(s,p=None):
        pv=s["p_greater"] if p is None else p
        return (
            s["complete_pairs"]==40 and
            s["auc"]>.5 and
            s["mean_paired_delta"]>0 and
            pv<.05
        )

    def neg(s,p=None):
        pv=s["p_less"] if p is None else p
        return (
            s["complete_pairs"]==40 and
            s["auc"]<.5 and
            s["mean_paired_delta"]<0 and
            pv<.05
        )

    # A
    a=get("OPENAI","BASELINE")

    if not a or a["complete_pairs"]!=40:
        Af="NULL_INCONCLUSIVE"; Ae="INCOMPLETE"; Av="UNRESOLVED"
    else:
        confirms=0
        for m in [
            "OPERATOR_RELATIONAL_V2_MARGIN",
            "SURFACE_FREE_MARGIN",
            "R1_COMPATIBLE_MARGIN"
        ]:
            s=get("OPENAI","BASELINE",m)
            confirms += int(bool(s and s["mean_paired_delta"]>0))

        if pos(a) and confirms>=2: Af="POSITIVE"
        elif neg(a): Af="NEGATIVE"
        elif pos(a): Af="MIXED"
        else: Af="NULL_INCONCLUSIVE"

        Ae="COMPLETED"; Av="VALID"

    A={
        "EXECUTION":Ae,
        "SCIENTIFIC_VALIDITY":Av,
        "FINDING":Af,
        "AUTHORITY":"FINAL_AUTHORITY" if Ae=="COMPLETED" else "UNRESOLVED_AUTHORITY",
        "PUBLICATION":"ELIGIBLE_FINAL" if Ae=="COMPLETED" else "DO_NOT_CITE_AS_RESULT",
        "DETAILS":{"primary":a}
    }

    # B with actual Holm correction
    bs=[get("ANTHROPIC","BASELINE"),get("GOOGLE","BASELINE")]

    if any(s is None or s["complete_pairs"]!=40 for s in bs):
        B={
            "EXECUTION":"INCOMPLETE",
            "SCIENTIFIC_VALIDITY":"UNRESOLVED",
            "FINDING":"NULL_INCONCLUSIVE",
            "AUTHORITY":"UNRESOLVED_AUTHORITY",
            "PUBLICATION":"DO_NOT_CITE_AS_RESULT",
            "DETAILS":bs
        }
    else:
        pg=holm([s["p_greater"] for s in bs])
        pl=holm([s["p_less"] for s in bs])

        for s,x,y in zip(bs,pg,pl):
            s["holm_p_greater"]=x
            s["holm_p_less"]=y

        np_=sum(pos(s,s["holm_p_greater"]) for s in bs)
        nn_=sum(neg(s,s["holm_p_less"]) for s in bs)

        if np_==2: bf="POSITIVE"
        elif nn_==2: bf="NEGATIVE"
        elif np_ or nn_: bf="MIXED"
        else: bf="NULL_INCONCLUSIVE"

        B={
            "EXECUTION":"COMPLETED",
            "SCIENTIFIC_VALIDITY":"VALID",
            "FINDING":bf,
            "AUTHORITY":"FINAL_AUTHORITY",
            "PUBLICATION":"ELIGIBLE_FINAL",
            "DETAILS":bs
        }

    # C with actual Holm correction
    dconds=["DOMAIN_TECHNICAL","DOMAIN_ORGANIZATIONAL","DOMAIN_EDUCATIONAL"]
    sconds=["SURFACE_TERM_REDACT","SURFACE_MARKDOWN_FLATTEN","SURFACE_CASE_PUNCT_NORMALIZED"]
    cs=[get("OPENAI",c) for c in dconds+sconds]

    if any(s is None or s["complete_pairs"]!=40 for s in cs):
        C={
            "EXECUTION":"INCOMPLETE",
            "SCIENTIFIC_VALIDITY":"UNRESOLVED",
            "FINDING":"NULL_INCONCLUSIVE",
            "AUTHORITY":"UNRESOLVED_AUTHORITY",
            "PUBLICATION":"DO_NOT_CITE_AS_RESULT",
            "DETAILS":cs
        }
    else:
        pg=holm([s["p_greater"] for s in cs])
        pl=holm([s["p_less"] for s in cs])

        for s,x,y in zip(cs,pg,pl):
            s["holm_p_greater"]=x
            s["holm_p_less"]=y

        dp=sum(
            pos(s,s["holm_p_greater"])
            for s in cs[:3]
        )

        sp=sum(
            pos(s,s["holm_p_greater"])
            for s in cs[3:]
        )

        rev=sum(
            neg(s,s["holm_p_less"])
            for s in cs
        )

        medauc=float(np.median([s["auc"] for s in cs]))
        meddel=float(np.median([s["mean_paired_delta"] for s in cs]))

        if dp>=2 and sp>=2 and rev==0: cf="POSITIVE"
        elif rev>=2 or (medauc<.5 and meddel<0): cf="NEGATIVE"
        elif dp or sp: cf="MIXED"
        else: cf="NULL_INCONCLUSIVE"

        C={
            "EXECUTION":"COMPLETED",
            "SCIENTIFIC_VALIDITY":"VALID",
            "FINDING":cf,
            "AUTHORITY":"FINAL_AUTHORITY",
            "PUBLICATION":"ELIGIBLE_FINAL",
            "DETAILS":cs
        }

    for name,obj in [
        ("WF2_A_V13_ADJUDICATION.json",A),
        ("WF2_B_V14_ADJUDICATION.json",B),
        ("WF2_C_V17_ADJUDICATION.json",C)
    ]:
        (ADJ/name).write_text(
            json.dumps(obj,indent=2,sort_keys=True),
            encoding="utf-8"
        )

    mods=[A,B,C]

    complete=all(x["EXECUTION"]=="COMPLETED" for x in mods)
    valid=all(x["SCIENTIFIC_VALIDITY"]=="VALID" for x in mods)
    fs=[x["FINDING"] for x in mods]

    if not complete or not valid:
        overall="NULL_INCONCLUSIVE"
        auth="UNRESOLVED_AUTHORITY"
        pub="DO_NOT_CITE_AS_RESULT"
    elif all(x=="POSITIVE" for x in fs):
        overall="POSITIVE"; auth="FINAL_AUTHORITY"; pub="ELIGIBLE_FINAL"
    elif all(x=="NEGATIVE" for x in fs):
        overall="NEGATIVE"; auth="FINAL_AUTHORITY"; pub="ELIGIBLE_FINAL"
    else:
        overall="MIXED"; auth="FINAL_AUTHORITY"; pub="ELIGIBLE_FINAL"

    O={
        "EXECUTION":"COMPLETED" if complete else "INCOMPLETE",
        "SCIENTIFIC_VALIDITY":"VALID" if valid else "UNRESOLVED",
        "FINDING":overall,
        "AUTHORITY":auth,
        "PUBLICATION":pub,
        "MODULE_FINDINGS":{
            "WF2_A_V13":A["FINDING"],
            "WF2_B_V14":B["FINDING"],
            "WF2_C_V17":C["FINDING"]
        }
    }

    (ADJ/"WF2_OVERALL_ADJUDICATION.json").write_text(
        json.dumps(O,indent=2,sort_keys=True),
        encoding="utf-8"
    )

    print(json.dumps(O,indent=2))

if __name__=="__main__":
    main()
