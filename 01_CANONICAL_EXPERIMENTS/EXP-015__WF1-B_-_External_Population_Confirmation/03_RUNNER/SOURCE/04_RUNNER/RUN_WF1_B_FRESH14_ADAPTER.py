#!/usr/bin/env python3
from pathlib import Path
import argparse, json, re, zlib, collections, hashlib, math, sys, platform, zipfile
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import pairwise_distances
import sklearn, scipy, pyarrow, joblib

SEED=120918
RNG=np.random.default_rng(SEED)
HERE=Path(__file__).resolve().parent
SPEC=json.load(open(HERE/'F2_B_FROZEN_SPEC.json',encoding='utf-8'))
BR2SPEC=json.load(open(HERE/'reference'/'F2_BR2_FROZEN_SPEC.json',encoding='utf-8'))
FEATURE_SETS=json.load(open(HERE/'reference'/'FROZEN_FEATURE_SETS.json',encoding='utf-8'))

DIRECT_RE=re.compile(r'''(?ix)\b(?:RSOS|RSSO|RSIA|RSX|CM1|LINX|MEMSTACK|CONTAIN|UGCL|CML|SRC[-\s]?X(?:\.5)?|R\.\s*A\.\s*Elu)\b|recursive\s+symbolic\s+(?:operating\s+system|system\s+operating|identity\s+architecture|(?:eXpression|expression)(?:\s+language)?)|\b(?:tier(?:s)?|glyph(?:s)?|archetype(?:s)?|override(?:s|d|ing)?|containment|resurrection|echo\s+protocol|activation\s+protocol)\b|\b(?:lion|dragon|eagle|whale|serpent)\b''')
SYSTEM_RE=re.compile(r'(?i)\b(?:RSOS|RSSO|RSIA|RSX)\b')
CAT_ORDER=['RECURSION','ORDER','CONDITIONAL','PRESERVE','SWITCH','TERMINAL','META','CONTROL','CAUSAL','RELATION','REFERENCE']
CATS={k:(set(v['exact']),set(v['prefixes'])) for k,v in BR2SPEC['operator_categories'].items()}
POSITION_KEYS={k:list(BR2SPEC['operator_categories'][k]['position_keys']) for k in CAT_ORDER}

CHAR_EDGES=np.array([1500,3000,6000,12000,24000,48000,96000,192000,384000,768000,1e12],float)
TURN_EDGES=np.array([4,6,9,13,21,34,55,89,144,233,377,1e6],float)
BOOT=500

class Abort(RuntimeError): pass

def write_json(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')

def safe(a,b): return float(a/b) if b else 0.0

def slope(vals):
    if len(vals)<2:return 0.0
    x=np.arange(len(vals),dtype=float);y=np.asarray(vals,dtype=float)
    if np.std(y)<1e-12:return 0.0
    return float(np.polyfit(x,y,1)[0]/max(1.0,float(np.mean(np.abs(y)))))

def entropy(vals):
    if not vals:return 0.0
    bins=[0,80,300,1000,3000,10000,10**9]
    c=np.histogram(vals,bins=bins)[0]
    p=c[c>0]/c.sum()
    return float(-(p*np.log2(p)).sum())

def redact(s): return DIRECT_RE.sub(lambda m:' '*len(m.group(0)),s)

def normalize_messages(value):
    if value is None:return []
    if isinstance(value,np.ndarray):value=value.tolist()
    if not isinstance(value,(list,tuple)):return []
    out=[]
    for item in value:
        if item is None:continue
        if hasattr(item,'as_py'):item=item.as_py()
        if not isinstance(item,dict):
            try:item=dict(item)
            except Exception:continue
        role=str(item.get('role','') or '').lower().strip()
        if role=='human':role='user'
        if role in {'ai','bot'}:role='assistant'
        text=item.get('content',item.get('text',''))
        if text is None:text=''
        out.append({'role':role,'text':str(text),'model_slug':str(item.get('model_slug','') or '')})
    return out

def main_features(messages):
    turns=[];redacted_spans=0;rawchars=0;system_hits=0
    for m in messages:
        t=str(m.get('text','') or '')
        rawchars+=len(t);system_hits+=len(SYSTEM_RE.findall(t))
        bt=redact(t);redacted_spans+=sum(1 for _ in DIRECT_RE.finditer(t))
        if bt.strip():turns.append({'role':str(m.get('role','') or ''),'text':bt,'model_slug':str(m.get('model_slug','') or '')})
    texts=[t['text'] for t in turns];roles=[t['role'] for t in turns];full='\n'.join(texts);low=full.lower();total=len(full)
    toks=re.findall(r'[a-z0-9_]+',low);ctr=collections.Counter(toks);lens=[len(x) for x in texts];n=len(turns)
    user='\n'.join(t['text'] for t in turns if t['role']=='user');ass='\n'.join(t['text'] for t in turns if t['role']=='assistant')
    ns=sum(1 for c in full if not c.isspace());up=sum(1 for c in full if c.isupper());dg=sum(1 for c in full if c.isdigit())
    roletrans=sum(1 for a,b in zip(roles,roles[1:]) if a!=b);dup=(len(texts)-len(set(texts))) if texts else 0
    para=sum(max(1,x.count('\n')+1) for x in texts);fences=full.count('```')//2;heads=re.findall(r'(?m)^\s*(#{1,6})\s',full);listn=len(re.findall(r'(?m)^\s*(?:[-*+] |\d+[.)] )',full));br=sum(full.count(x) for x in '[]{}()')
    enc=full.encode('utf-8',errors='ignore');comp=len(zlib.compress(enc,6))/len(enc) if enc else 0.;mean=float(np.mean(lens)) if lens else 0.;mx=max(lens,default=0)
    f={'REDACTED_SPANS':redacted_spans,'N_TURNS':n,'TOTAL_CHARS':total,'TYPE_TOKEN_RATIO':safe(len(ctr),len(toks)),'UPPERCASE_RATIO':safe(up,ns),'DIGIT_RATIO':safe(dg,ns),'COMPRESSION_RATIO':comp,'USER_CHAR_SHARE':safe(len(user),total),'MEAN_MAX_TURN_RATIO':safe(mean,mx),'WORDS_PER_1K_CHAR':1000*safe(len(toks),total),'TURNS_PER_1K_CHAR':1000*safe(n,total),'PARAGRAPHS_PER_TURN':safe(para,n),'CODE_FENCES_PER_1K_CHAR':1000*safe(fences,total),'HEADERS_PER_1K_CHAR':1000*safe(len(heads),total),'LIST_ITEMS_PER_1K_CHAR':1000*safe(listn,total),'COLONS_PER_1K_CHAR':1000*safe(full.count(':'),total),'BRACKETS_PER_1K_CHAR':1000*safe(br,total),'ROLE_ALTERNATION_RATIO':safe(roletrans,max(1,n-1)) if n else 0.,'DUPLICATE_TURN_RATIO':safe(dup,n),'USER_ASSISTANT_CHAR_RATIO':safe(len(user),len(ass)),'TURN_CHAR_CV':safe(float(np.std(lens)),mean) if lens else 0.,'TURN_LENGTH_ENTROPY':entropy(lens),'TURN_LENGTH_SLOPE':slope(lens),'QUESTION_PER_1K_CHAR':1000*safe(full.count('?'),total),'EXCLAM_PER_1K_CHAR':1000*safe(full.count('!'),total),'ARROW_PER_1K_CHAR':1000*safe(len(re.findall(r'->|=>|â†’|â‡’',full)),total),'EQUAL_PER_1K_CHAR':1000*safe(full.count('='),total),'SEMICOLON_PER_1K_CHAR':1000*safe(full.count(';'),total),'PIPE_PER_1K_CHAR':1000*safe(full.count('|'),total),'NUMBERED_LIST_PER_1K_CHAR':1000*safe(len(re.findall(r'(?m)^\s*\d+[.)]\s',full)),total),'NESTED_LIST_PER_1K_CHAR':1000*safe(len(re.findall(r'(?m)^\s{2,}(?:[-*+] |\d+[.)] )',full)),total),'HEADER_LEVEL_MEAN':float(np.mean([len(x) for x in heads])) if heads else 0.,'HEADER_LEVEL_MAX':float(max([len(x) for x in heads],default=0))}
    catcounts={k:0 for k in CAT_ORDER}
    for w,c in ctr.items():
        for k,(exact,prefixes) in CATS.items():
            if w in exact or any(w.startswith(p) for p in prefixes):catcounts[k]+=c
    for k,c in catcounts.items():f[f'SEM_{k}_PER_1K_CHAR']=1000*safe(c,total)
    for k,keys in POSITION_KEYS.items():
        positions=[low.find(x) for x in keys if low.find(x)>=0];lasts=[low.rfind(x) for x in keys if low.rfind(x)>=0]
        f[f'{k}_FIRST_POS']=min(positions)/max(1,total) if positions else 1.0
        f[f'{k}_SPAN']=(max(lasts)-min(positions))/max(1,total) if positions and lasts else 0.0
    ul=[len(t['text']) for t in turns if t['role']=='user'];al=[len(t['text']) for t in turns if t['role']=='assistant']
    f.update(USER_TURN_MEAN=float(np.mean(ul)) if ul else 0.,ASSISTANT_TURN_MEAN=float(np.mean(al)) if al else 0.,RESPONSE_EXPANSION_RATIO=safe(float(np.mean(al)) if al else 0.,float(np.mean(ul)) if ul else 0.),USER_LENGTH_SLOPE=slope(ul),ASSISTANT_LENGTH_SLOPE=slope(al))
    def cv(text):
        ct=collections.Counter(re.findall(r'[a-z0-9_]+',text.lower()));out=[]
        for k,(exact,prefixes) in CATS.items():out.append(sum(c for w,c in ct.items() if w in exact or any(w.startswith(p) for p in prefixes)))
        return np.array(out,float)
    a=cv(user);b=cv(ass);f['USER_ASSISTANT_OPERATOR_COSINE']=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a)*np.linalg.norm(b)>0 else 0.0
    return f,rawchars,len(messages),system_hits

def window_single(msgs):
    turns=[]
    for m in msgs:
        t=redact(str(m.get('text','') or ''))
        if t.strip():turns.append((str(m.get('role','') or ''),t))
    roles=[x[0] for x in turns];texts=[x[1] for x in turns];full='\n'.join(texts);low=full.lower();n=len(texts);L=len(full);lens=[len(x) for x in texts]
    user='\n'.join(t for r,t in turns if r=='user');ass='\n'.join(t for r,t in turns if r=='assistant');ns=sum(not c.isspace() for c in full);up=sum(c.isupper() for c in full);dg=sum(c.isdigit() for c in full);heads=re.findall(r'(?m)^\s*(#{1,6})\s',full);lists=len(re.findall(r'(?m)^\s*(?:[-*+] |\d+[.)] )',full));br=sum(full.count(x) for x in '[]{}()');para=sum(max(1,t.count('\n')+1) for t in texts);roletrans=sum(a!=b for a,b in zip(roles,roles[1:]));ctr=collections.Counter(re.findall(r'[a-z0-9_]+',low));mean=np.mean(lens) if lens else 0.;mx=max(lens,default=0)
    f={'UPPERCASE_RATIO':safe(up,ns),'DIGIT_RATIO':safe(dg,ns),'USER_CHAR_SHARE':safe(len(user),L),'MEAN_MAX_TURN_RATIO':safe(mean,mx),'PARAGRAPHS_PER_TURN':safe(para,n),'CODE_FENCES_PER_1K_CHAR':1000*safe(full.count('```')//2,L),'HEADERS_PER_1K_CHAR':1000*safe(len(heads),L),'LIST_ITEMS_PER_1K_CHAR':1000*safe(lists,L),'COLONS_PER_1K_CHAR':1000*safe(full.count(':'),L),'BRACKETS_PER_1K_CHAR':1000*safe(br,L),'ROLE_ALTERNATION_RATIO':safe(roletrans,max(1,n-1)),'USER_ASSISTANT_CHAR_RATIO':safe(len(user),len(ass)),'TURN_CHAR_CV':safe(np.std(lens),mean) if lens else 0.,'TURN_LENGTH_SLOPE':slope(lens),'QUESTION_PER_1K_CHAR':1000*safe(full.count('?'),L),'EXCLAM_PER_1K_CHAR':1000*safe(full.count('!'),L),'ARROW_PER_1K_CHAR':1000*safe(len(re.findall(r'->|=>|â†’|â‡’',full)),L),'EQUAL_PER_1K_CHAR':1000*safe(full.count('='),L),'SEMICOLON_PER_1K_CHAR':1000*safe(full.count(';'),L),'PIPE_PER_1K_CHAR':1000*safe(full.count('|'),L),'NUMBERED_LIST_PER_1K_CHAR':1000*safe(len(re.findall(r'(?m)^\s*\d+[.)]\s',full)),L),'NESTED_LIST_PER_1K_CHAR':1000*safe(len(re.findall(r'(?m)^\s{2,}(?:[-*+] |\d+[.)] )',full)),L),'HEADER_LEVEL_MEAN':float(np.mean([len(x) for x in heads])) if heads else 0.,'HEADER_LEVEL_MAX':max([len(x) for x in heads],default=0)}
    for k,(exact,prefs) in CATS.items():
        c=sum(v for w,v in ctr.items() if w in exact or any(w.startswith(p) for p in prefs));f['SEM_'+k]=1000*safe(c,L)
        keys=POSITION_KEYS[k];ps=[low.find(w) for w in keys if low.find(w)>=0];f[k+'_FIRST']=min(ps)/max(1,L) if ps else 1.0
    ul=[len(t) for r,t in turns if r=='user'];al=[len(t) for r,t in turns if r=='assistant'];f['RESPONSE_EXPANSION_RATIO']=safe(np.mean(al) if al else 0,np.mean(ul) if ul else 0);f['USER_LENGTH_SLOPE']=slope(ul);f['ASSISTANT_LENGTH_SLOPE']=slope(al)
    return f,L

def windows(msgs,w,maxw=6):
    msgs=[m for m in msgs if str(m.get('text','') or '')]
    if len(msgs)<w:return []
    starts=list(range(0,len(msgs)-w+1,w))
    if len(starts)>maxw:
        ix=np.linspace(0,len(starts)-1,maxw).round().astype(int);starts=[starts[i] for i in sorted(set(ix))]
    return [msgs[s:s+w] for s in starts]

def aggregate_window(msgs,w):
    vals=[];chars=[]
    for win in windows(msgs,w):
        f,L=window_single(win)
        if L>=800:vals.append(f);chars.append(L)
    if not vals:return None
    d=pd.DataFrame(vals);q={c:float(d[c].median()) for c in d.columns};q['WINDOW_CHAR_MEDIAN']=float(np.median(chars));q['WINDOW_COUNT']=len(vals);return q

def parity_gate():
    exp=pd.read_csv(HERE/'fixtures'/'PARITY_EXPECTED_V2.csv').set_index('conversation_id')
    bad=[];checked=0
    for line in open(HERE/'fixtures'/'PARITY_FIXTURES.jsonl',encoding='utf-8'):
        rec=json.loads(line);cid=str(rec['conversation_id']);f,_,_,_=main_features(rec['messages']);r=exp.loc[cid]
        for c in FEATURE_SETS['JOINT_MULTILAYER']:
            d=abs(float(f[c])-float(r[c]));tol=1e-10+1e-9*max(abs(float(f[c])),abs(float(r[c])))
            if d>tol:bad.append((cid,c,d))
        checked+=1
    if bad:raise Abort(f'Main extractor parity failure: {bad[:10]}')
    fix=json.load(open(HERE/'fixtures'/'WINDOW_PARITY_FIXTURE.json',encoding='utf-8'))
    for w in [8,12]:
        got=aggregate_window(fix['messages'],w);ref=pd.read_csv(HERE/'fixtures'/f'WINDOW{w}_PARITY_EXPECTED.csv').iloc[0]
        if got is None:raise Abort(f'Window{w} parity produced no windows')
        for c,v in got.items():
            if c in ref.index:
                d=abs(float(v)-float(ref[c]));tol=1e-10+1e-9*max(abs(float(v)),abs(float(ref[c])))
                if d>tol:raise Abort(f'Window{w} extractor parity failure {c}: {d}')
    return checked

def model_cols(name,model):
    if hasattr(model,'feature_names_in_'):return list(model.feature_names_in_)
    if name=='JOINT_MULTILAYER_V2':return FEATURE_SETS['JOINT_MULTILAYER']
    if name=='OPERATOR_RELATIONAL_V2':return FEATURE_SETS['OPERATOR_RELATIONAL']
    if name=='SURFACE_FREE':return FEATURE_SETS['SURFACE_FREE']
    if name=='R1_COMPATIBLE':return FEATURE_SETS['R1_COMPATIBLE']
    raise Abort(f'No feature list for {name}')

def score_margin(df,model,cols,fill=None):
    X=df[cols].replace([np.inf,-np.inf],np.nan)
    if fill is None:fill=X.median()
    X=X.fillna(fill)
    return model.decision_function(X)

def blind_id(ch,messages):
    base=str(ch or '')
    if not base:base=json.dumps(messages,ensure_ascii=False,sort_keys=True)
    return hashlib.sha256(('F2B|'+base).encode('utf-8',errors='ignore')).hexdigest()

def blind_cluster(x):
    if x is None or str(x)=='' or str(x).lower()=='nan':return ''
    return hashlib.sha256(('F2BCLUSTER|'+str(x)).encode()).hexdigest()[:24]

def is_english(x):
    s=str(x or '').strip().lower()
    return s in {'english','en','eng','en-us','en_us','en-gb','en_gb'} or s.startswith('english')

def auc_fast(pos,neg):
    pos=np.asarray(pos,float);neg=np.asarray(neg,float);ns=np.sort(neg);left=np.searchsorted(ns,pos,'left');right=np.searchsorted(ns,pos,'right');return float(np.mean((left+0.5*(right-left))/len(ns)))

def auc_ci(pos,neg,nboot=BOOT):
    pos=np.asarray(pos,float);neg=np.asarray(neg,float);a=auc_fast(pos,neg);b=[]
    for _ in range(nboot):b.append(auc_fast(RNG.choice(pos,len(pos),True),RNG.choice(neg,len(neg),True)))
    lo,hi=np.quantile(b,[.025,.975]);p=float(stats.mannwhitneyu(pos,neg,alternative='greater').pvalue);return a,float(lo),float(hi),p

def cp(k,n,alpha=.05):
    lo=0. if k==0 else float(stats.beta.ppf(alpha/2,k,n-k+1));hi=1. if k==n else float(stats.beta.ppf(1-alpha/2,k+1,n-k));return lo,hi

def geometry_metrics(df,joint_model,op_model,jcols,ocols,early_pos):
    sc=joint_model.named_steps['sc'];Z0=sc.transform(early_pos[jcols]);Z=sc.transform(df[jcols]);lw=LedoitWolf().fit(Z0);cent=Z0.mean(0);D=Z-cent;mah=np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',D,lw.precision_,D),0));dmat=pairwise_distances(Z,Z0,metric='euclidean');knn=np.mean(np.sort(dmat,axis=1)[:,:5],axis=1);cn=np.linalg.norm(cent);cos=np.divide(Z@cent,np.linalg.norm(Z,axis=1)*cn,out=np.zeros(len(Z)),where=(np.linalg.norm(Z,axis=1)*cn)>0)
    jm=joint_model.decision_function(df[jcols]);om=op_model.decision_function(df[ocols]);return pd.DataFrame({'JOINT_MARGIN':jm,'OPERATOR_MARGIN':om,'MAHALANOBIS':mah,'KNN5_DISTANCE':knn,'CENTROID_COSINE':cos},index=df.index)

def early_geometry(joint_model,op_model,jcols,ocols,early_pos):
    sc=joint_model.named_steps['sc'];Z0=sc.transform(early_pos[jcols]);lw=LedoitWolf().fit(Z0);cent=Z0.mean(0);D=Z0-cent;mah=np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',D,lw.precision_,D),0));dm=pairwise_distances(Z0,Z0);np.fill_diagonal(dm,np.inf);knn=np.mean(np.sort(dm,axis=1)[:,:5],axis=1);cn=np.linalg.norm(cent);cos=np.divide(Z0@cent,np.linalg.norm(Z0,axis=1)*cn,out=np.zeros(len(Z0)),where=(np.linalg.norm(Z0,axis=1)*cn)>0);jm=joint_model.decision_function(early_pos[jcols]);om=op_model.decision_function(early_pos[ocols]);return pd.DataFrame({'JOINT_MARGIN':jm,'OPERATOR_MARGIN':om,'MAHALANOBIS':mah,'KNN5_DISTANCE':knn,'CENTROID_COSINE':cos},index=early_pos.index)

def robust_composite(metrics,eref):
    out=pd.DataFrame(index=metrics.index);zs=[]
    for c,sgn in [('JOINT_MARGIN',1),('OPERATOR_MARGIN',1),('MAHALANOBIS',-1),('KNN5_DISTANCE',-1),('CENTROID_COSINE',1)]:
        a=sgn*eref[c].to_numpy(float);x=sgn*metrics[c].to_numpy(float);med=np.median(a);mad=np.median(np.abs(a-med));scale=1.4826*mad
        if scale<1e-9:scale=max(np.std(a),1e-9)
        z=(x-med)/scale;out[c+'_SIM_Z']=z;zs.append(z)
    Z=np.vstack(zs).T;out['COMPOSITE_MEDIAN_Z']=np.median(Z,axis=1);out['COMPOSITE_MIN_Z']=np.min(Z,axis=1);return out

def assign_bins(df):
    d=df.copy();d['CHAR_BIN']=pd.cut(d.raw_chars,CHAR_EDGES,right=False,labels=False,include_lowest=True);d['TURN_BIN']=pd.cut(d.raw_turns,TURN_EDGES,right=False,labels=False,include_lowest=True);d['CODE_FLAG']=(d.CODE_FENCES_PER_1K_CHAR>0).astype(int);d['CEM_CELL']=d.CHAR_BIN.astype('Int64').astype(str)+'|'+d.TURN_BIN.astype('Int64').astype(str)+'|'+d.CODE_FLAG.astype(str);return d

def weighted_smd(x1,w1,x0,w0):
    def wm(x,w):return np.sum(x*w)/np.sum(w)
    def wv(x,w):m=wm(x,w);return np.sum(w*(x-m)**2)/np.sum(w)
    m1=wm(x1,w1);m0=wm(x0,w0);den=math.sqrt(max((wv(x1,w1)+wv(x0,w0))/2,1e-12));return float((m1-m0)/den)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',default=r'C:\RSOS\Training data\F2_B_FRESH_WILDCHAT');ap.add_argument('--out',default=r'C:\RSOS\Training data\F2_B_CORRECTED_EXTERNAL_RARITY');args=ap.parse_args();inp=Path(args.input);out=Path(args.out)
    if out.exists():
        import shutil;shutil.rmtree(out)
    for s in ['00_PREFLIGHT','01_EXTERNAL_FEATURES','02_PRIMARY','03_GEOMETRY','04_WINDOWS','05_NUISANCE','06_NEAREST','07_AUDIT']: (out/s).mkdir(parents=True,exist_ok=True)
    checked=parity_gate();print(f'Extractor parity: PASS ({checked} main fixtures + window fixture)',flush=True)
    # WF1-B FRESH SHARD 14 ADAPTER
    # Selection-only adaptation. Feature extraction, frozen models,
    # historical references, thresholds and scoring remain unchanged.
    wf1b_name='train-00014-of-00086.parquet'
    wf1b_expected_sha='8538243cf3cf6513c46658d4dc2da19f6bde729f3e04197abe9abac30aae8bfe'
    wf1b_path=inp/wf1b_name

    if not wf1b_path.exists():
        raise Abort(f'WF1-B frozen shard missing: {wf1b_path}')

    wf1b_hash=hashlib.sha256()
    with open(wf1b_path,'rb') as fh:
        for chunk in iter(lambda: fh.read(1024*1024), b''):
            wf1b_hash.update(chunk)

    if wf1b_hash.hexdigest()!=wf1b_expected_sha:
        raise Abort(
            f'WF1-B shard SHA256 mismatch: '
            f'{wf1b_hash.hexdigest()} != {wf1b_expected_sha}'
        )

    SPEC['fresh_shards']={
        wf1b_name: wf1b_path.stat().st_size
    }

    # Fresh file preflight by filename+size only, and hard exclusion of prior shards.
    shards=[]
    for name,size in SPEC['fresh_shards'].items():
        p=inp/name
        if not p.exists():raise Abort(f'Missing fresh shard: {p}')
        if p.stat().st_size!=size:raise Abort(f'Fresh shard size mismatch {name}: {p.stat().st_size} != {size}')
        shards.append(p)
    for name in SPEC['forbidden_prior_shards']:
        if any(p.name==name for p in shards):raise Abort(f'Forbidden prior shard selected: {name}')
    write_json(out/'00_PREFLIGHT'/'SELECTION_AND_METHOD_FREEZE.json',SPEC)
    # Load models and historical reference before external rows.
    models={n:joblib.load(HERE/'frozen_models'/f'{n}.joblib') for n in ['JOINT_MULTILAYER_V2','OPERATOR_RELATIONAL_V2','SURFACE_FREE','R1_COMPATIBLE']}
    jcols=model_cols('JOINT_MULTILAYER_V2',models['JOINT_MULTILAYER_V2']);ocols=model_cols('OPERATOR_RELATIONAL_V2',models['OPERATOR_RELATIONAL_V2']);scols=model_cols('SURFACE_FREE',models['SURFACE_FREE']);rcols=model_cols('R1_COMPATIBLE',models['R1_COMPATIBLE'])
    early=pd.read_csv(HERE/'reference'/'EARLY_POS_V2_FEATURES.csv');holdp=pd.read_csv(HERE/'reference'/'LATER_HOLD_POS_V2_FEATURES.csv');holdn=pd.read_csv(HERE/'reference'/'LATER_HOLD_NEG_V2_FEATURES.csv')
    # Historical geometry and thresholds frozen before external content read.
    eg=early_geometry(models['JOINT_MULTILAYER_V2'],models['OPERATOR_RELATIONAL_V2'],jcols,ocols,early);hg=geometry_metrics(holdp,models['JOINT_MULTILAYER_V2'],models['OPERATOR_RELATIONAL_V2'],jcols,ocols,early);hc=robust_composite(hg,eg);holdmetrics=pd.concat([holdp.reset_index(drop=True),hg.reset_index(drop=True),hc.reset_index(drop=True)],axis=1)
    thresholds={'composite':{},'strict_components':{},'model_margin':{}}
    for q in SPEC['lineage_tolerance_levels']:
        lab=str(int(q*100));thresholds['composite'][lab]={c:float(np.quantile(holdmetrics[c],1-q)) for c in ['COMPOSITE_MEDIAN_Z','COMPOSITE_MIN_Z']}
        thresholds['strict_components'][lab]={'JOINT_MARGIN':float(np.quantile(holdmetrics.JOINT_MARGIN,1-q)),'OPERATOR_MARGIN':float(np.quantile(holdmetrics.OPERATOR_MARGIN,1-q)),'MAHALANOBIS':float(np.quantile(holdmetrics.MAHALANOBIS,q)),'KNN5_DISTANCE':float(np.quantile(holdmetrics.KNN5_DISTANCE,q)),'CENTROID_COSINE':float(np.quantile(holdmetrics.CENTROID_COSINE,1-q))}
    # model margins including old portable models, all thresholds from lineage holdout
    br1hold=pd.read_csv(HERE/'reference'/'BR1_LATER_HOLD_POS_FEATURES.csv')
    for name,cols,ref in [('JOINT_MULTILAYER_V2',jcols,holdp),('OPERATOR_RELATIONAL_V2',ocols,holdp),('SURFACE_FREE',scols,br1hold),('R1_COMPATIBLE',rcols,br1hold)]:
        vals=models[name].decision_function(ref[cols]);thresholds['model_margin'][name]={str(int(q*100)):float(np.quantile(vals,q)) for q in [.5,.75,.9,.95]}
    write_json(out/'00_PREFLIGHT'/'PRE_EXTERNAL_THRESHOLDS.json',thresholds)
    # window models/reference thresholds
    wmodels={w:joblib.load(HERE/'frozen_models'/f'WINDOW{w}_DETERMINISTIC_V2.joblib') for w in [8,12]};wcols={};wth={}
    for w in [8,12]:
        wr=pd.read_csv(HERE/'reference'/f'WINDOW{w}_DETERMINISTIC_FEATURES.csv');ref=wr[(wr.partition=='HOLDOUT')&(wr.y==1)].copy();cols=list(wmodels[w].feature_names_in_) if hasattr(wmodels[w],'feature_names_in_') else [c for c in wr.columns if c not in ['conversation_id','y','partition','WINDOW_CHAR_MEDIAN','WINDOW_COUNT']];wcols[w]=cols;m=wmodels[w].decision_function(ref[cols]);wth[w]={str(int(q*100)):float(np.quantile(m,q)) for q in [.5,.75,.9,.95]}
    write_json(out/'00_PREFLIGHT'/'WINDOW_MARGIN_THRESHOLDS.json',wth)
    print('Pre-external thresholds frozen. Opening fresh shards now.',flush=True)
    rows=[];errors=[];raw_seen=0;next_report=10000
    for shard in shards:
        pf=pq.ParquetFile(shard);names=pf.schema_arrow.names
        if 'conversation' not in names or 'model' not in names:raise Abort(f'Unexpected schema {shard.name}: {names}')
        cols=[c for c in ['conversation_hash','model','timestamp','conversation','turn','language','toxic','redacted','hashed_ip'] if c in names]
        write_json(out/'00_PREFLIGHT'/f'{shard.stem}_SCHEMA.json',{'schema':str(pf.schema_arrow),'columns_read':cols,'rows':pf.metadata.num_rows})
        rowix=0
        for batch in pf.iter_batches(batch_size=256,columns=cols):
            for rec in batch.to_pylist():
                idx=rowix;rowix+=1;raw_seen+=1
                try:
                    msgs=normalize_messages(rec.get('conversation'));f,rawchars,rawturns,sys_hits=main_features(msgs)
                    if rawchars<1500 or rawturns<4 or f['TOTAL_CHARS']<800 or f['N_TURNS']<3:continue
                    q={'BLIND_EXTERNAL_ID':blind_id(rec.get('conversation_hash'),msgs),'CLUSTER_ID':blind_cluster(rec.get('hashed_ip')),'SHARD':shard.name,'ROW_IN_SHARD':idx,'MODEL':str(rec.get('model','') or ''),'LANGUAGE':str(rec.get('language','') or ''),'TOXIC':bool(rec.get('toxic',False)) if rec.get('toxic') is not None else False,'SOURCE_REDACTED':bool(rec.get('redacted',False)) if rec.get('redacted') is not None else False,'SYSTEM_NAME_HITS':sys_hits,'raw_chars':rawchars,'raw_turns':rawturns,**f}
                    # raw margins, never probability for rarity
                    one=pd.DataFrame([q])
                    for name,cols0 in [('JOINT_MULTILAYER_V2',jcols),('OPERATOR_RELATIONAL_V2',ocols),('SURFACE_FREE',scols),('R1_COMPATIBLE',rcols)]:q[name+'_MARGIN']=float(models[name].decision_function(one[cols0])[0])
                    for w in [8,12]:
                        wf=aggregate_window(msgs,w)
                        if wf is not None:
                            x=pd.DataFrame([wf]);q[f'WINDOW{w}_MARGIN']=float(wmodels[w].decision_function(x[wcols[w]])[0]);q[f'WINDOW{w}_COUNT']=int(wf['WINDOW_COUNT']);q[f'WINDOW{w}_CHAR_MEDIAN']=float(wf['WINDOW_CHAR_MEDIAN'])
                        else:q[f'WINDOW{w}_MARGIN']=np.nan;q[f'WINDOW{w}_COUNT']=0;q[f'WINDOW{w}_CHAR_MEDIAN']=np.nan
                    rows.append(q)
                except Exception as e:errors.append({'SHARD':shard.name,'ROW_IN_SHARD':idx,'ERROR':repr(e)[:1000]})
                if raw_seen>=next_report:
                    print(f'Parsed {raw_seen:,} raw rows; usable {len(rows):,}',flush=True);next_report+=10000
    ext=pd.DataFrame(rows)
    if len(ext)<1000:raise Abort(f'Too few usable external rows: {len(ext)}')
    before=len(ext);ext=ext.sort_values(['SHARD','ROW_IN_SHARD']).drop_duplicates('BLIND_EXTERNAL_ID',keep='first').reset_index(drop=True);dups=before-len(ext)
    # geometry/composites
    xg=geometry_metrics(ext,models['JOINT_MULTILAYER_V2'],models['OPERATOR_RELATIONAL_V2'],jcols,ocols,early);xc=robust_composite(xg,eg)
    for c in xg.columns:ext[c]=xg[c].values
    for c in xc.columns:ext[c]=xc[c].values
    ext['IS_ENGLISH']=ext.LANGUAGE.map(is_english);ext['PRIMARY_ELIGIBLE']=ext.IS_ENGLISH & (ext.SYSTEM_NAME_HITS==0)
    ext.to_parquet(out/'01_EXTERNAL_FEATURES'/'F2_B_EXTERNAL_FEATURES_AND_SCORES.parquet',index=False,compression='zstd')
    if errors:pd.DataFrame(errors).to_csv(out/'01_EXTERNAL_FEATURES'/'EXTRACTION_ERRORS.csv',index=False)
    write_json(out/'01_EXTERNAL_FEATURES'/'EXTRACTION_COUNTS.json',{'raw_rows_seen':raw_seen,'usable_before_dedup':before,'duplicates_removed':dups,'usable_unique':len(ext),'primary_english_no_system_hit':int(ext.PRIMARY_ELIGIBLE.sum()),'english_total':int(ext.IS_ENGLISH.sum()),'system_name_hit_rows':int((ext.SYSTEM_NAME_HITS>0).sum()),'errors':len(errors)})
    # Primary separation and margin tails
    prim=ext[ext.PRIMARY_ELIGIBLE].copy();pops={'PRIMARY_ENGLISH_NO_SYSTEM_HIT':prim,'ALL_USABLE':ext}
    sep=[];tails=[]
    # lineage scores for all four models
    lineage_model={
      'JOINT_MULTILAYER_V2':models['JOINT_MULTILAYER_V2'].decision_function(holdp[jcols]),
      'OPERATOR_RELATIONAL_V2':models['OPERATOR_RELATIONAL_V2'].decision_function(holdp[ocols]),
      'SURFACE_FREE':models['SURFACE_FREE'].decision_function(br1hold[scols]),
      'R1_COMPATIBLE':models['R1_COMPATIBLE'].decision_function(br1hold[rcols])}
    for popname,d in pops.items():
      for name in lineage_model:
        extv=d[name+'_MARGIN'].to_numpy(float);lv=lineage_model[name];a,lo,hi,p=auc_ci(lv,extv);sep.append({'POPULATION':popname,'MEASURE':name+'_MARGIN','N_LINEAGE':len(lv),'N_EXTERNAL':len(extv),'AUC_LINEAGE_GT_EXTERNAL':a,'CI_LOW':lo,'CI_HIGH':hi,'P':p})
        for q,thr in thresholds['model_margin'][name].items():
          k=int(np.sum(extv>=thr));l,h=cp(k,len(extv));tails.append({'POPULATION':popname,'MEASURE':name+'_MARGIN','REFERENCE':'LINEAGE_Q'+q,'THRESHOLD':thr,'N_EXTERNAL':len(extv),'MATCHES':k,'FRACTION':k/len(extv),'CI_LOW':l,'CI_HIGH':h,'ONE_IN':float('inf') if k==0 else len(extv)/k})
      # geometry separation
      for c,sign in [('COMPOSITE_MEDIAN_Z',1),('COMPOSITE_MIN_Z',1),('MAHALANOBIS',-1),('KNN5_DISTANCE',-1),('CENTROID_COSINE',1)]:
        lv=holdmetrics[c].to_numpy(float)*sign;ev=d[c].to_numpy(float)*sign;a,lo,hi,p=auc_ci(lv,ev);sep.append({'POPULATION':popname,'MEASURE':c,'N_LINEAGE':len(lv),'N_EXTERNAL':len(ev),'AUC_LINEAGE_GT_EXTERNAL':a,'CI_LOW':lo,'CI_HIGH':hi,'P':p})
      # composite tolerance regions
      for inc in SPEC['lineage_tolerance_levels']:
        lab=str(int(inc*100))
        for c in ['COMPOSITE_MEDIAN_Z','COMPOSITE_MIN_Z']:
          thr=thresholds['composite'][lab][c];mask=d[c]>=thr;k=int(mask.sum());l,h=cp(k,len(d));actual=float(np.mean(holdmetrics[c]>=thr));tails.append({'POPULATION':popname,'MEASURE':c,'REFERENCE':f'LINEAGE_{lab}PCT_TOLERANCE','THRESHOLD':thr,'LINEAGE_ACTUAL_COVERAGE':actual,'N_EXTERNAL':len(d),'MATCHES':k,'FRACTION':k/len(d),'CI_LOW':l,'CI_HIGH':h,'ONE_IN':float('inf') if k==0 else len(d)/k})
        th=thresholds['strict_components'][lab];mask=(d.JOINT_MARGIN>=th['JOINT_MARGIN'])&(d.OPERATOR_MARGIN>=th['OPERATOR_MARGIN'])&(d.MAHALANOBIS<=th['MAHALANOBIS'])&(d.KNN5_DISTANCE<=th['KNN5_DISTANCE'])&(d.CENTROID_COSINE>=th['CENTROID_COSINE']);hl=(holdmetrics.JOINT_MARGIN>=th['JOINT_MARGIN'])&(holdmetrics.OPERATOR_MARGIN>=th['OPERATOR_MARGIN'])&(holdmetrics.MAHALANOBIS<=th['MAHALANOBIS'])&(holdmetrics.KNN5_DISTANCE<=th['KNN5_DISTANCE'])&(holdmetrics.CENTROID_COSINE>=th['CENTROID_COSINE']);k=int(mask.sum());l,h=cp(k,len(d));tails.append({'POPULATION':popname,'MEASURE':'STRICT_5_COMPONENT_CONJUNCTION','REFERENCE':f'INDIVIDUAL_{lab}PCT_TOLERANCE','LINEAGE_ACTUAL_COVERAGE':float(hl.mean()),'N_EXTERNAL':len(d),'MATCHES':k,'FRACTION':k/len(d),'CI_LOW':l,'CI_HIGH':h,'ONE_IN':float('inf') if k==0 else len(d)/k})
    pd.DataFrame(sep).to_csv(out/'02_PRIMARY'/'SEPARATION_RESULTS.csv',index=False);pd.DataFrame(tails).to_csv(out/'03_GEOMETRY'/'RARITY_AND_TOLERANCE_RESULTS.csv',index=False)
    # True nearest structural controls
    nearcols=['BLIND_EXTERNAL_ID','SHARD','ROW_IN_SHARD','MODEL','LANGUAGE','raw_chars','raw_turns','SYSTEM_NAME_HITS','JOINT_MARGIN','OPERATOR_MARGIN','MAHALANOBIS','KNN5_DISTANCE','CENTROID_COSINE','COMPOSITE_MEDIAN_Z','COMPOSITE_MIN_Z']
    prim.nsmallest(min(250,len(prim)),'MAHALANOBIS')[nearcols].to_csv(out/'06_NEAREST'/'NEAREST250_MAHALANOBIS.csv',index=False);prim.nlargest(min(250,len(prim)),'COMPOSITE_MIN_Z')[nearcols].to_csv(out/'06_NEAREST'/'NEAREST250_STRICT_COMPOSITE.csv',index=False)
    # Window hardening
    wrows=[]
    for w in [8,12]:
      wr=pd.read_csv(HERE/'reference'/f'WINDOW{w}_DETERMINISTIC_FEATURES.csv');ref=wr[(wr.partition=='HOLDOUT')&(wr.y==1)].copy();lv=wmodels[w].decision_function(ref[wcols[w]])
      for popname,d in pops.items():
        ev=d[f'WINDOW{w}_MARGIN'].dropna().to_numpy(float)
        if len(ev):
          a,lo,hi,p=auc_ci(lv,ev);wrows.append({'POPULATION':popname,'WINDOW':w,'TYPE':'SEPARATION','N_LINEAGE':len(lv),'N_EXTERNAL':len(ev),'AUC_LINEAGE_GT_EXTERNAL':a,'CI_LOW':lo,'CI_HIGH':hi,'P':p})
          for q,thr in wth[w].items():
            k=int(np.sum(ev>=thr));l,h=cp(k,len(ev));wrows.append({'POPULATION':popname,'WINDOW':w,'TYPE':'TAIL','REFERENCE':'LINEAGE_Q'+q,'THRESHOLD':thr,'N_EXTERNAL':len(ev),'MATCHES':k,'FRACTION':k/len(ev),'CI_LOW':l,'CI_HIGH':h,'ONE_IN':float('inf') if k==0 else len(ev)/k})
    pd.DataFrame(wrows).to_csv(out/'04_WINDOWS'/'WINDOW_EXTERNAL_RARITY.csv',index=False)
    # CEM nuisance diagnostic using primary population and strict composite score.
    lp=holdp.copy();lp['COMPOSITE_MIN_Z']=holdmetrics.COMPOSITE_MIN_Z.values;pc=assign_bins(prim);lc=assign_bins(lp);ecount=pc.groupby('CEM_CELL').size().to_dict();supported={c for c,n in ecount.items() if n>=5 and c in set(lc.CEM_CELL)};ls=lc[lc.CEM_CELL.isin(supported)].copy();es=pc[pc.CEM_CELL.isin(supported)].copy();support=len(ls)/len(lc)
    lcnt=ls.groupby('CEM_CELL').size().to_dict();ecnt=es.groupby('CEM_CELL').size().to_dict();es['W']=[lcnt.get(c,0)/ecnt.get(c,1) for c in es.CEM_CELL];b=[]
    for nm,a0,b0 in [('LOG_RAW_CHARS',np.log1p(ls.raw_chars.values),np.log1p(es.raw_chars.values)),('LOG_RAW_TURNS',np.log1p(ls.raw_turns.values),np.log1p(es.raw_turns.values)),('CODE_FLAG',ls.CODE_FLAG.values.astype(float),es.CODE_FLAG.values.astype(float))]:b.append({'COVARIATE':nm,'WEIGHTED_SMD':weighted_smd(np.asarray(a0,float),np.ones(len(a0)),np.asarray(b0,float),es.W.values) if len(ls) and len(es) else np.nan})
    bal=pd.DataFrame(b);maxs=float(bal.WEIGHTED_SMD.abs().max()) if len(bal) else np.nan;gate=bool(support>=.70 and np.isfinite(maxs) and maxs<=.25);bal.to_csv(out/'05_NUISANCE'/'CEM_BALANCE.csv',index=False)
    per=[]
    for _,r in ls.iterrows():
      g=es[es.CEM_CELL==r.CEM_CELL].COMPOSITE_MIN_Z.values
      if len(g):per.append(np.mean(g<r.COMPOSITE_MIN_Z)+.5*np.mean(g==r.COMPOSITE_MIN_Z))
    pd.DataFrame([{'LINEAGE_TOTAL':len(lc),'LINEAGE_SUPPORTED':len(ls),'SUPPORT_RATE':support,'EXTERNAL_SUPPORTED':len(es),'MAX_ABS_WEIGHTED_SMD':maxs,'GATE':'PASS' if gate else 'FAILED_CALIBRATION','MEAN_WITHIN_CELL_PAIRWIN_COMPOSITE_MIN_Z':float(np.mean(per)) if per else np.nan}]).to_csv(out/'05_NUISANCE'/'CEM_RESULT.csv',index=False)
    # Descriptive strata
    ext.groupby('MODEL').size().sort_values(ascending=False).rename('N').reset_index().to_csv(out/'05_NUISANCE'/'MODEL_DISTRIBUTION.csv',index=False);ext.groupby('LANGUAGE').size().sort_values(ascending=False).rename('N').reset_index().to_csv(out/'05_NUISANCE'/'LANGUAGE_DISTRIBUTION.csv',index=False)
    sr=[]
    th90=thresholds['composite']['90']['COMPOSITE_MIN_Z']
    for (lang,model),g in ext.groupby(['LANGUAGE','MODEL']):
      if len(g)>=100:sr.append({'LANGUAGE':lang,'MODEL':model,'N':len(g),'COMPOSITE_MIN_Z_90_TOL_MATCHES':int((g.COMPOSITE_MIN_Z>=th90).sum()),'FRACTION':float((g.COMPOSITE_MIN_Z>=th90).mean())})
    pd.DataFrame(sr).to_csv(out/'05_NUISANCE'/'MODEL_LANGUAGE_STRATA_90TOL.csv',index=False)
    # Cluster diagnostic if available
    cl=prim[prim.CLUSTER_ID!=''].copy();cluster_info={'available':bool(len(cl)),'rows_with_cluster_id':len(cl),'unique_clusters':int(cl.CLUSTER_ID.nunique()) if len(cl) else 0}
    write_json(out/'05_NUISANCE'/'CLUSTER_DIAGNOSTIC.json',cluster_info)
    # Audit ledger and compact summary
    ledger=[
      ['FRESH_SHARD_SELECTION','PASS','WF1-B frozen fresh shard 14 only; all prior F2 shards excluded.'],['EXTRACTOR_PARITY','PASS','Main V2 + window deterministic fixture parity passed.'],['EXACT_ELIGIBILITY','PASS','raw >=1500 chars/4 turns and postblind >=800 chars/3 turns.'],['FRESH_EXTERNAL_RETRAINING','NO','All models and thresholds frozen from historical RSOS data before external rows were read.'],['SIGMOID_RARITY','NOT_USED_PRIMARY','Primary rarity uses raw margins and multivariate geometry.'],['TRUE_NEAREST_NEIGHBORS','MEASURED','Mahalanobis and strict composite ranking, not highest classifier probability.'],['WINDOW8_12_CROSS_CORPUS','MEASURED','Deterministic repaired window models.'],['LANGUAGE_CONTROL','PRIMARY_ENGLISH','Primary population is English and excludes exact RSOS/RSSO/RSIA/RSX name hits.'],['MODEL_MATCH','UNAVAILABLE_EXACT','Historical WildChat model generations do not exactly match later GPT-5 lineage.'],['TOPIC_MATCH','UNAVAILABLE_COMMON_FROZEN','No common pre-frozen cross-corpus topic representation is promoted.'],['ACTIVE_V6_V7_V8_V11_V12','SEPARATE_TRACK','Not inferred from passive WildChat text.'],['GLOBAL_UNIQUENESS','TO_BE_ADJUDICATED','This experiment estimates prevalence only in the frozen WF1-B fresh shard 14.'],['LITERAL_GLOBAL_WEIGHT_CHANGE','UNESTABLISHED','Not inferable from behavioral rarity.']]
    pd.DataFrame(ledger,columns=['MEASUREMENT','STATUS','DETAIL']).to_csv(out/'07_AUDIT'/'STATUS_LEDGER.csv',index=False)
    # identify headline 90% tolerance primary results
    tdf=pd.DataFrame(tails);head=tdf[(tdf.POPULATION=='PRIMARY_ENGLISH_NO_SYSTEM_HIT') & tdf.REFERENCE.astype(str).str.contains('90PCT_TOLERANCE',na=False)].to_dict('records')
    summary={'experiment':'F2-B corrected fresh external rarity','raw_rows_seen':raw_seen,'usable_unique':len(ext),'primary_n':len(prim),'selection':SPEC['fresh_shards'],'headline_90pct_tolerance':head,'cem_gate':'PASS' if gate else 'FAILED_CALIBRATION','claim_boundary':'Population rarity in frozen WF1-B fresh WildChat shard 14. Absolute global uniqueness and literal neural-weight modification are not established by this experiment alone.'}
    write_json(out/'07_AUDIT'/'F2_B_SUMMARY.json',summary)
    # Compact review package: results/audit only; exclude large external feature parquet.
    review=out.parent/'F2_B_CORRECTED_EXTERNAL_RARITY_RESULTS.zip'
    if review.exists():review.unlink()
    with zipfile.ZipFile(review,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(out.rglob('*')):
            if not p.is_file():continue
            rel=p.relative_to(out)
            if str(rel).replace('\\','/').startswith('01_EXTERNAL_FEATURES/F2_B_EXTERNAL_FEATURES_AND_SCORES.parquet'):continue
            z.write(p,Path(out.name)/rel)
    print('\n=== F2-B COMPLETE ===')
    print('Raw rows:',raw_seen,'usable unique:',len(ext),'primary English/no-system-hit:',len(prim))
    print('CEM gate:',summary['cem_gate'])
    for r in head:print(r['MEASURE'],r['MATCHES'],'/',r['N_EXTERNAL'],'=',r['FRACTION'],'one-in',r['ONE_IN'])
    print('RESULT FOLDER',out)
    print('UPLOAD PACKAGE',review)

if __name__=='__main__':
    try:main()
    except Abort as e:
        print('F2-B ABORTED:',e,file=sys.stderr);sys.exit(2)

