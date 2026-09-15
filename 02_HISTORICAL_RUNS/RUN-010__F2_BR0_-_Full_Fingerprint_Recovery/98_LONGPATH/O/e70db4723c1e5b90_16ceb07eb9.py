import json,re,zlib,math,csv,pathlib,collections
import numpy as np
root=pathlib.Path('/mnt/data/F1_BR1_BLIND_RSOS_CORPUS_REDISCOVERY')
DIRECT_RE=re.compile(r'''(?ix)
\b(?:RSOS|RSSO|RSIA|RSX|CM1|LINX|MEMSTACK|CONTAIN|UGCL|CML|SRC[-\s]?X(?:\.5)?|R\.\s*A\.\s*Elu)\b
|recursive\s+symbolic\s+(?:operating\s+system|system\s+operating|identity\s+architecture|(?:eXpression|expression)(?:\s+language)?)
|\b(?:tier(?:s)?|glyph(?:s)?|archetype(?:s)?|override(?:s|d|ing)?|containment|resurrection|echo\s+protocol|activation\s+protocol)\b
|\b(?:lion|dragon|eagle|whale|serpent)\b''')
SEM={
'RECURSION':re.compile(r'\b(?:recursive|recursion|recur\w*|loop\w*|feedback|nested|self[- ]refer\w*)\b',re.I),
'ORDER':re.compile(r'\b(?:first|second|third|next|then|before|after|finally|step|phase|sequence|order\w*)\b',re.I),
'CONDITIONAL':re.compile(r'\b(?:if|when|unless|otherwise|provided|condition\w*)\b',re.I),
'PRESERVE':re.compile(r'\b(?:preserve|retain|keep|maintain|invariant|unchanged|freeze|frozen|lock)\b',re.I),
'SWITCH':re.compile(r'\b(?:switch|replace|reassign|redirect|route|map|transform|convert|swap|transition)\w*\b',re.I),
'TERMINAL':re.compile(r'\b(?:final|end|stop|terminate|terminal|output|result|complete|closure)\w*\b',re.I),
'META':re.compile(r'\b(?:meta|layer|level|higher[- ]order|supervis\w*|monitor\w*|watchdog)\b',re.I),
'CONTROL':re.compile(r'\b(?:must|shall|only|exactly|constraint\w*|rule\w*|never|required|requirement\w*)\b|\bdo\s+not\b',re.I),
'CAUSAL':re.compile(r'\b(?:because|therefore|thus|hence|cause\w*|leads?\s+to|results?\s+in)\b',re.I),
'RELATION':re.compile(r'\b(?:node|edge|link|path|graph|network|hierarch\w*|dependen\w*|relation\w*)\b',re.I),
'REFERENCE':re.compile(r'\b(?:previous|above|below|earlier|later|same|corresponding|source|target)\b',re.I),
}
R1=['TYPE_TOKEN_RATIO','UPPERCASE_RATIO','DIGIT_RATIO','COMPRESSION_RATIO','USER_CHAR_SHARE','MEAN_MAX_TURN_RATIO','WORDS_PER_1K_CHAR','TURNS_PER_1K_CHAR','PARAGRAPHS_PER_TURN','CODE_FENCES_PER_1K_CHAR','HEADERS_PER_1K_CHAR','LIST_ITEMS_PER_1K_CHAR','COLONS_PER_1K_CHAR','BRACKETS_PER_1K_CHAR']
def hard_blind(txt):
 return '\n'.join(line for line in txt.splitlines() if not DIRECT_RE.search(line))
def words(txt):return re.findall(r'[A-Za-z0-9_]+',txt.lower())
def safe(a,b):return float(a/b) if b else 0.0
def slope(vals):
 if len(vals)<2:return 0.0
 x=np.arange(len(vals),dtype=float); y=np.asarray(vals,dtype=float)
 if np.std(y)<1e-12:return 0.0
 return float(np.polyfit(x,y,1)[0]/max(1.0,float(np.mean(np.abs(y)))))
def entropy(vals):
 if not vals:return 0.0
 bins=[0,80,300,1000,3000,10000,10**9]; c=np.histogram(vals,bins=bins)[0]; p=c[c>0]/c.sum(); return float(-(p*np.log2(p)).sum())
def seq_features(full):
 # paragraph-level operator sequence; categories are generic and system-name independent after hard blinding.
 units=[x for x in re.split(r'\n+',full) if x.strip()]
 seq=[]; positions={k:[] for k in SEM}
 for i,u in enumerate(units):
  hits=[k for k,r in SEM.items() if r.search(u)]
  for k in hits:positions[k].append(i)
  if hits: seq.append(hits[0])
 n=max(1,len(units)); out={}
 for k,p in positions.items():
  out[f'{k}_FIRST_POS']=p[0]/n if p else 1.0
  out[f'{k}_SPAN']=((p[-1]-p[0])/n) if len(p)>1 else 0.0
 uniq=len(set(seq)); out['OPERATOR_UNIQUE_CATEGORIES']=uniq
 if seq:
  cc=collections.Counter(seq); probs=np.array(list(cc.values()),dtype=float)/len(seq); out['OPERATOR_CATEGORY_ENTROPY']=float(-(probs*np.log2(probs)).sum())
  transitions=list(zip(seq,seq[1:])); out['OPERATOR_SELF_TRANSITION_RATIO']=safe(sum(a==b for a,b in transitions),len(transitions)); out['OPERATOR_TRANSITION_DIVERSITY']=safe(len(set(transitions)),len(transitions))
  out['OPERATOR_ABA_CYCLE_RATE']=safe(sum(1 for a,b,c in zip(seq,seq[1:],seq[2:]) if a==c and a!=b),max(1,len(seq)-2))
 else:
  out.update(OPERATOR_CATEGORY_ENTROPY=0.0,OPERATOR_SELF_TRANSITION_RATIO=0.0,OPERATOR_TRANSITION_DIVERSITY=0.0,OPERATOR_ABA_CYCLE_RATE=0.0)
 return out
def feats(messages):
 orig_chars=sum(len(m.get('text','')) for m in messages)
 turns=[]
 for m in messages:
  bt=hard_blind(m.get('text',''))
  if bt.strip():turns.append({'role':m.get('role',''),'text':bt,'model_slug':m.get('model_slug','')})
 texts=[t['text'] for t in turns]; roles=[t['role'] for t in turns]; full='\n'.join(texts); total=len(full); w=words(full); lens=[len(x) for x in texts]; n=len(turns)
 user='\n'.join(t['text'] for t in turns if t['role']=='user'); ass='\n'.join(t['text'] for t in turns if t['role']=='assistant')
 ns=sum(1 for c in full if not c.isspace()); up=sum(1 for c in full if c.isupper()); dg=sum(1 for c in full if c.isdigit())
 roletrans=sum(1 for a,b in zip(roles,roles[1:]) if a!=b); dup=(len(texts)-len(set(texts))) if texts else 0
 para=sum(max(1,x.count('\n')+1) for x in texts); fences=full.count('```')//2; heads=re.findall(r'(?m)^\s*(#{1,6})\s',full); lists=re.findall(r'(?m)^\s*(?:[-*+] |\d+[.)] )',full); br=sum(full.count(x) for x in '[]{}()')
 enc=full.encode('utf-8',errors='ignore'); comp=len(zlib.compress(enc,9))/len(enc) if enc else 0.; mean=float(np.mean(lens)) if lens else 0.; mx=max(lens,default=0)
 f={'BLIND_RETENTION_RATIO':safe(total,orig_chars),'N_TURNS':n,'TOTAL_CHARS':total,
 'TYPE_TOKEN_RATIO':safe(len(set(w)),len(w)),'UPPERCASE_RATIO':safe(up,ns),'DIGIT_RATIO':safe(dg,ns),'COMPRESSION_RATIO':comp,'USER_CHAR_SHARE':safe(len(user),total),'MEAN_MAX_TURN_RATIO':safe(mean,mx),'WORDS_PER_1K_CHAR':1000*safe(len(w),total),'TURNS_PER_1K_CHAR':1000*safe(n,total),'PARAGRAPHS_PER_TURN':safe(para,n),'CODE_FENCES_PER_1K_CHAR':1000*safe(fences,total),'HEADERS_PER_1K_CHAR':1000*safe(len(heads),total),'LIST_ITEMS_PER_1K_CHAR':1000*safe(len(lists),total),'COLONS_PER_1K_CHAR':1000*safe(full.count(':'),total),'BRACKETS_PER_1K_CHAR':1000*safe(br,total),
 'ROLE_ALTERNATION_RATIO':safe(roletrans,max(1,n-1)) if n else 0.,'DUPLICATE_TURN_RATIO':safe(dup,n),'USER_ASSISTANT_CHAR_RATIO':safe(len(user),len(ass)),'TURN_CHAR_CV':safe(float(np.std(lens)),mean) if lens else 0.,'TURN_LENGTH_ENTROPY':entropy(lens),'TURN_LENGTH_SLOPE':slope(lens),'QUESTION_PER_1K_CHAR':1000*safe(full.count('?'),total),'EXCLAM_PER_1K_CHAR':1000*safe(full.count('!'),total),'ARROW_PER_1K_CHAR':1000*safe(len(re.findall(r'->|=>|→|⇒',full)),total),'EQUAL_PER_1K_CHAR':1000*safe(full.count('='),total),'SEMICOLON_PER_1K_CHAR':1000*safe(full.count(';'),total),'PIPE_PER_1K_CHAR':1000*safe(full.count('|'),total),'NUMBERED_LIST_PER_1K_CHAR':1000*safe(len(re.findall(r'(?m)^\s*\d+[.)]\s',full)),total),'NESTED_LIST_PER_1K_CHAR':1000*safe(len(re.findall(r'(?m)^\s{2,}(?:[-*+] |\d+[.)] )',full)),total),'HEADER_LEVEL_MEAN':float(np.mean([len(x) for x in heads])) if heads else 0.,'HEADER_LEVEL_MAX':float(max([len(x) for x in heads],default=0))}
 for k,r in SEM.items():f[f'SEM_{k}_PER_1K_CHAR']=1000*safe(len(r.findall(full)),total)
 ul=[len(t['text']) for t in turns if t['role']=='user']; al=[len(t['text']) for t in turns if t['role']=='assistant']; f.update(USER_TURN_MEAN=float(np.mean(ul)) if ul else 0.,ASSISTANT_TURN_MEAN=float(np.mean(al)) if al else 0.,RESPONSE_EXPANSION_RATIO=safe(float(np.mean(al)) if al else 0.,float(np.mean(ul)) if ul else 0.),USER_LENGTH_SLOPE=slope(ul),ASSISTANT_LENGTH_SLOPE=slope(al)); f.update(seq_features(full))
 # semantic role echo
 def semvec(txt):return np.array([len(r.findall(txt)) for r in SEM.values()],dtype=float)
 a=semvec(user); b=semvec(ass); f['USER_ASSISTANT_OPERATOR_COSINE']=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a)*np.linalg.norm(b)>0 else 0.
 return f,full
sets=[('EARLY_POS','EARLY_POS_SEGMENTS.jsonl',1,'TRAIN'),('EARLY_NEG','EARLY_NEG_SEGMENTS.jsonl',0,'TRAIN'),('LATER_VAL_POS','LATER_VAL_POS_SEGMENTS.jsonl',1,'VALIDATION'),('LATER_VAL_NEG','LATER_VAL_NEG_SEGMENTS.jsonl',0,'VALIDATION'),('LATER_HOLD_POS','LATER_HOLD_POS_SEGMENTS.jsonl',1,'HOLDOUT'),('LATER_HOLD_NEG','LATER_HOLD_NEG_SEGMENTS.jsonl',0,'HOLDOUT')]
rows=[]; blindtexts=[]
for label,fn,y,partition in sets:
 p=root/'04_SPLITS'/fn
 for rec in map(json.loads,open(p,encoding='utf-8')):
  # filter actual segment size before/after blinding
  rawchars=sum(len(m.get('text','')) for m in rec['messages']); rawturns=len(rec['messages'])
  if rawchars<1500 or rawturns<4:continue
  f,btxt=feats(rec['messages'])
  if f['TOTAL_CHARS']<800 or f['N_TURNS']<3:continue
  models=[m.get('model_slug','') for m in rec['messages'] if m.get('model_slug')]; model=collections.Counter(models).most_common(1)[0][0] if models else rec.get('default_model_slug','')
  row={'conversation_id':rec['conversation_id'],'set_label':label,'partition':partition,'y':y,'model_slug':model,'raw_chars':rawchars,'raw_turns':rawturns,**f}; rows.append(row); blindtexts.append({'conversation_id':rec['conversation_id'],'set_label':label,'partition':partition,'y':y,'text':btxt})
 print(label,'kept',sum(r['set_label']==label for r in rows),flush=True)
fields=[]
for r in rows:
 for k in r:
  if k not in fields:fields.append(k)
with open(root/'05_FEATURES/F1_HARD_BLIND_FEATURES.csv','w',newline='',encoding='utf-8') as g:
 w=csv.DictWriter(g,fieldnames=fields);w.writeheader();w.writerows(rows)
with open(root/'03_BLINDED_CORPUS/F1_HARD_BLINDED_TEXT.jsonl','w',encoding='utf-8') as g:
 for r in blindtexts:g.write(json.dumps(r,ensure_ascii=False)+'\n')
print('total',len(rows));print(collections.Counter(r['set_label'] for r in rows));print('retention medians',{k:float(np.median([r['BLIND_RETENTION_RATIO'] for r in rows if r['set_label']==k])) for k in sorted(set(r['set_label'] for r in rows))})
