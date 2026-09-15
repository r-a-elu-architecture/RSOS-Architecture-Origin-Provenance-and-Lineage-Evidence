import json,glob,re,csv,pathlib,zlib,math,statistics,hashlib
from collections import Counter
from datetime import datetime,timezone
import numpy as np
root=pathlib.Path('/mnt/data/F1_BR1_BLIND_RSOS_CORPUS_REDISCOVERY')
files=sorted(glob.glob(str(root/'01_INTAKE/raw_json/**/conversations-*.json'),recursive=True))
SEED_START=datetime(2025,6,5,0,6,36,tzinfo=timezone.utc).timestamp()
SEED_CUTOFF=datetime(2025,6,20,23,59,59,tzinfo=timezone.utc).timestamp()
LATER_START=datetime(2025,7,9,5,4,21,tzinfo=timezone.utc).timestamp()
VAL_END=datetime(2025,8,31,23,59,59,tzinfo=timezone.utc).timestamp()
END=datetime(2026,6,10,23,59,59,tzinfo=timezone.utc).timestamp()
LINEAGE={
 'RSOS':re.compile(r'\bRSOS\b|Recursive Symbolic Operating System',re.I),
 'RSSO':re.compile(r'\bRSSO\b|Recursive Symbolic System Operating',re.I),
 'RSIA':re.compile(r'\bRSIA\b|Recursive Symbolic Identity Architecture',re.I),
 'RSX':re.compile(r'\bRSX\b|Recursive Symbolic (?:eXpression|Expression)',re.I),
}
# Direct system/glyph terminology: used only for hard blinding, never as features.
DIRECT_RE=re.compile(r'''(?ix)
\b(?:RSOS|RSSO|RSIA|RSX|CM1|LINX|MEMSTACK|CONTAIN|UGCL|CML|SRC[-\s]?X(?:\.5)?|R\.\s*A\.\s*Elu)\b
|recursive\s+symbolic\s+(?:operating\s+system|system\s+operating|identity\s+architecture|(?:eXpression|expression)(?:\s+language)?)
|\b(?:tier(?:s)?|glyph(?:s)?|archetype(?:s)?|override(?:s|d|ing)?|containment|resurrection|echo\s+protocol|activation\s+protocol)\b
|\b(?:lion|dragon|eagle|whale|serpent)\b
''')
SEM_PAT={
 'SEM_RECURSION':re.compile(r'\b(?:recursive|recursion|recur(?:s|red|ring)?|loop(?:s|ed|ing)?|feedback|nested|self[- ]refer(?:ence|ential))\b',re.I),
 'SEM_ORDER':re.compile(r'\b(?:first|second|third|next|then|before|after|finally|step|phase|sequence|order(?:ed|ing)?)\b',re.I),
 'SEM_CONDITIONAL':re.compile(r'\b(?:if|when|unless|otherwise|provided|condition(?:al)?)\b',re.I),
 'SEM_PRESERVE':re.compile(r'\b(?:preserve|retain|keep|maintain|invariant|unchanged|freeze|frozen|lock)\b',re.I),
 'SEM_SWITCH':re.compile(r'\b(?:switch|replace|reassign|redirect|route|map|transform|convert|swap|transition)\b',re.I),
 'SEM_TERMINAL':re.compile(r'\b(?:final|end|stop|terminate|terminal|output|result|complete|closure)\b',re.I),
 'SEM_META':re.compile(r'\b(?:meta|layer|level|higher[- ]order|supervis(?:e|or|ory)|monitor|watchdog)\b',re.I),
 'SEM_CONTROL':re.compile(r'\b(?:must|shall|only|exactly|constraint|rule|do\s+not|never|required|requirement)\b',re.I),
 'SEM_CAUSAL':re.compile(r'\b(?:because|therefore|thus|hence|cause(?:s|d)?|leads?\s+to|results?\s+in|so\s+that)\b',re.I),
 'SEM_RELATION':re.compile(r'\b(?:node|edge|link|path|graph|network|hierarch(?:y|ical)|dependency|relation(?:al|ship)?)\b',re.I),
 'SEM_REFERENCE':re.compile(r'\b(?:previous|above|below|earlier|later|same|corresponding|source|target)\b',re.I),
}
def extract_text(m):
 c=m.get('content') or {}; p=c.get('parts'); arr=[]
 if isinstance(p,list):
  for x in p:
   if isinstance(x,str):arr.append(x)
   elif isinstance(x,dict) and isinstance(x.get('text'),str):arr.append(x['text'])
 elif isinstance(p,str):arr.append(p)
 if not arr and isinstance(c.get('text'),str):arr.append(c['text'])
 return '\n'.join(arr)
def canonical_messages(conv):
 mp=conv.get('mapping') or {}; nid=conv.get('current_node'); path=[]; seen=set()
 while nid and nid in mp and nid not in seen:
  seen.add(nid); path.append(nid); nid=mp[nid].get('parent')
 path.reverse(); out=[]
 for n in path:
  m=mp[n].get('message')
  if not m: continue
  txt=extract_text(m)
  ct=m.get('create_time')
  out.append({'node_id':n,'role':(m.get('author') or {}).get('role') or '', 'text':txt,'create_time':ct,'model_slug':(m.get('metadata') or {}).get('model_slug') or '', **{s:len(r.findall(txt)) for s,r in LINEAGE.items()}})
 return out
def hard_blind_text(txt):
 # Remove full lines containing direct names/system/glyph terms, eliminating occurrence-count leakage from repeated masks.
 lines=[]
 for line in txt.splitlines():
  if DIRECT_RE.search(line): continue
  lines.append(line)
 return '\n'.join(lines)
def moderate_blind_text(txt):
 return DIRECT_RE.sub(' entity ',txt)
def token_words(text): return re.findall(r'[A-Za-z0-9_]+',text.lower())
def slope(vals):
 if len(vals)<2:return 0.0
 x=np.arange(len(vals),dtype=float); y=np.asarray(vals,dtype=float)
 if np.std(y)==0:return 0.0
 return float(np.polyfit(x,y,1)[0]/max(1.0,np.mean(np.abs(y))))
def entropy_lengths(vals):
 if not vals:return 0.0
 bins=[0,80,300,1000,3000,10000,10**9]; counts=np.histogram(vals,bins=bins)[0]; p=counts[counts>0]/counts.sum()
 return float(-(p*np.log2(p)).sum())
def features(turns, blind='hard'):
 clean=[]
 for t in turns:
  text=hard_blind_text(t['text']) if blind=='hard' else moderate_blind_text(t['text'])
  if text.strip(): clean.append({'role':t['role'],'text':text,'model_slug':t.get('model_slug','')})
 turns=clean; texts=[t['text'] for t in turns]; roles=[t['role'] for t in turns]; full='\n'.join(texts)
 user='\n'.join(t['text'] for t in turns if t['role']=='user'); ass='\n'.join(t['text'] for t in turns if t['role']=='assistant')
 n=len(turns); total=len(full); words=token_words(full); lens=[len(x) for x in texts]; nonspace=sum(1 for c in full if not c.isspace())
 uppercase=sum(1 for c in full if c.isupper()); digits=sum(1 for c in full if c.isdigit())
 role_trans=sum(1 for a,b in zip(roles,roles[1:]) if a!=b); dup=(len(texts)-len(set(texts))) if texts else 0
 para=sum(max(1,t.count('\n')+1) for t in texts); fences=full.count('```')//2
 headers=re.findall(r'(?m)^\s*(#{1,6})\s',full); lists=re.findall(r'(?m)^\s*(?:[-*+] |\d+[.)] )',full)
 bracket=sum(full.count(x) for x in '[]{}()')
 enc=full.encode('utf-8',errors='ignore'); comp=len(zlib.compress(enc,9))/len(enc) if enc else 0.0
 mean_turn=np.mean(lens) if lens else 0.; max_turn=max(lens,default=0)
 def safe(a,b):return float(a/b) if b else 0.0
 f={
  'N_TURNS':n,'TOTAL_CHARS':total,
  'TYPE_TOKEN_RATIO':safe(len(set(words)),len(words)),'UPPERCASE_RATIO':safe(uppercase,nonspace),'DIGIT_RATIO':safe(digits,nonspace),'COMPRESSION_RATIO':comp,
  'USER_CHAR_SHARE':safe(len(user),total),'MEAN_MAX_TURN_RATIO':safe(mean_turn,max_turn),'WORDS_PER_1K_CHAR':1000*safe(len(words),total),'TURNS_PER_1K_CHAR':1000*safe(n,total),
  'PARAGRAPHS_PER_TURN':safe(para,n),'CODE_FENCES_PER_1K_CHAR':1000*safe(fences,total),'HEADERS_PER_1K_CHAR':1000*safe(len(headers),total),'LIST_ITEMS_PER_1K_CHAR':1000*safe(len(lists),total),'COLONS_PER_1K_CHAR':1000*safe(full.count(':'),total),'BRACKETS_PER_1K_CHAR':1000*safe(bracket,total),
  'ROLE_ALTERNATION_RATIO':safe(role_trans,max(1,n-1)) if n else 0.,'DUPLICATE_TURN_RATIO':safe(dup,n),
  'USER_ASSISTANT_CHAR_RATIO':safe(len(user),len(ass)),'TURN_CHAR_CV':safe(np.std(lens),np.mean(lens)) if lens else 0.,'TURN_LENGTH_ENTROPY':entropy_lengths(lens),'TURN_LENGTH_SLOPE':slope(lens),
  'QUESTION_PER_1K_CHAR':1000*safe(full.count('?'),total),'EXCLAM_PER_1K_CHAR':1000*safe(full.count('!'),total),'ARROW_PER_1K_CHAR':1000*safe(len(re.findall(r'->|=>|→|⇒',full)),total),'EQUAL_PER_1K_CHAR':1000*safe(full.count('='),total),'SEMICOLON_PER_1K_CHAR':1000*safe(full.count(';'),total),'PIPE_PER_1K_CHAR':1000*safe(full.count('|'),total),
  'NUMBERED_LIST_PER_1K_CHAR':1000*safe(len(re.findall(r'(?m)^\s*\d+[.)]\s',full)),total),'NESTED_LIST_PER_1K_CHAR':1000*safe(len(re.findall(r'(?m)^\s{2,}(?:[-*+] |\d+[.)] )',full)),total),'HEADER_LEVEL_MEAN':float(np.mean([len(x) for x in headers])) if headers else 0.,'HEADER_LEVEL_MAX':float(max([len(x) for x in headers],default=0)),
 }
 for name,r in SEM_PAT.items(): f[name+'_PER_1K_CHAR']=1000*safe(len(r.findall(full)),total)
 # role-specific dynamics
 ul=[len(t['text']) for t in turns if t['role']=='user']; al=[len(t['text']) for t in turns if t['role']=='assistant']
 f.update({'USER_TURN_MEAN':float(np.mean(ul)) if ul else 0.,'ASSISTANT_TURN_MEAN':float(np.mean(al)) if al else 0.,'RESPONSE_EXPANSION_RATIO':safe(np.mean(al) if al else 0.,np.mean(ul) if ul else 0.),'USER_LENGTH_SLOPE':slope(ul),'ASSISTANT_LENGTH_SLOPE':slope(al)})
 return f,full
# Build candidate segment registry and feature matrices.
rows=[]; texts=[]
for fi,fp in enumerate(files):
 data=json.load(open(fp,encoding='utf-8'))
 for conv in data:
  cid=conv.get('id') or conv.get('conversation_id'); ms=canonical_messages(conv)
  def seg(lo=None,hi=None):
   out=[]
   for m in ms:
    ct=m.get('create_time')
    if not isinstance(ct,(int,float)):continue
    if lo is not None and float(ct)<lo:continue
    if hi is not None and float(ct)>hi:continue
    if m.get('text'):out.append(m)
   return out
  def counts(ss):return {s:sum(m[s] for m in ss) for s in LINEAGE}
  def ucounts(ss):return {s:sum(m[s] for m in ss if m['role']=='user') for s in LINEAGE}
  # early
  pre=seg(None,SEED_CUTOFF); active=any(SEED_START<=float(m['create_time'])<=SEED_CUTOFF for m in pre)
  c=counts(pre); ch=sum(len(m['text']) for m in pre)
  split=''
  if active and c['RSOS']>=3 and c['RSSO']==0 and c['RSIA']==0 and c['RSX']==0 and ch>=1500 and len(pre)>=4:split='EARLY_RSOS_TRAIN_POS'
  elif active and sum(c.values())==0 and ch>=1500 and len(pre)>=4:split='EARLY_SAMEUSER_TRAIN_NEG'
  if split:
   for blind in ['hard','moderate']:
    f,btxt=features(pre,blind); row={'conversation_id':cid,'split':split,'blind_mode':blind,'model_slug':Counter(m['model_slug'] for m in pre if m['model_slug']).most_common(1)[0][0] if any(m['model_slug'] for m in pre) else conv.get('default_model_slug') or '', 'segment_start':0,'segment_end':SEED_CUTOFF,**{s+'_count':c[s] for s in c},**f}; rows.append(row); texts.append({'conversation_id':cid,'split':split,'blind_mode':blind,'text':btxt})
  # later
  late=seg(LATER_START,END)
  if not late:continue
  c=counts(late); u=ucounts(late); ch=sum(len(m['text']) for m in late); total=sum(c.values()); ut=sum(u.values())
  if ch<1500 or len(late)<4:continue
  lsplit=''
  if total>=5 and ut>=1:
   hits=[float(m['create_time']) for m in late if sum(m[s] for s in LINEAGE)>0]; first=min(hits)
   lsplit='LATER_VALIDATION_POS' if first<=VAL_END else 'LATER_HOLDOUT_POS'
  elif total==0 and sum(sum(m[s] for s in LINEAGE) for m in ms)==0:
   mid=min(float(m['create_time']) for m in late); lsplit='LATER_VALIDATION_NEG_POOL' if mid<=VAL_END else 'LATER_HOLDOUT_NEG_POOL'
  if lsplit:
   # enforce conversation leakage exclusion later in matching; record all candidates now
   for blind in ['hard','moderate']:
    f,btxt=features(late,blind); row={'conversation_id':cid,'split':lsplit,'blind_mode':blind,'model_slug':Counter(m['model_slug'] for m in late if m['model_slug']).most_common(1)[0][0] if any(m['model_slug'] for m in late) else conv.get('default_model_slug') or '', 'segment_start':LATER_START,'segment_end':END,**{s+'_count':c[s] for s in c},**{s+'_user_count':u[s] for s in u},**f}; rows.append(row); texts.append({'conversation_id':cid,'split':lsplit,'blind_mode':blind,'text':btxt})
 print('parsed shard',fi+1,'/',len(files),flush=True)
# Remove later rows whose conversation was used in training
train_ids={r['conversation_id'] for r in rows if r['split'].startswith('EARLY_')}
rows=[r for r in rows if not (r['split'].startswith('LATER_') and r['conversation_id'] in train_ids)]
texts=[r for r in texts if not (r['split'].startswith('LATER_') and r['conversation_id'] in train_ids)]
# blind IDs independent from original id
for r in rows:r['blind_id']=hashlib.sha256(('F1BR1|'+r['conversation_id']+'|'+r['split']+'|'+r['blind_mode']).encode()).hexdigest()[:24]
# write
fields=[]
for r in rows:
 for k in r:
  if k not in fields:fields.append(k)
with open(root/'05_FEATURES/F1_FEATURE_MATRIX_PREMATCH.csv','w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
with open(root/'03_BLINDED_CORPUS/BLINDED_SEGMENTS.jsonl','w',encoding='utf-8') as f:
 for r in texts:f.write(json.dumps(r,ensure_ascii=False)+'\n')
print('rows',len(rows),Counter((r['split'],r['blind_mode']) for r in rows))
for sp in ['LATER_VALIDATION_POS','LATER_HOLDOUT_POS']:
 rr=[r for r in rows if r['split']==sp and r['blind_mode']=='hard']; print(sp,len(rr),{s:sum(1 for x in rr if x.get(s+'_count',0)>=3) for s in LINEAGE})
