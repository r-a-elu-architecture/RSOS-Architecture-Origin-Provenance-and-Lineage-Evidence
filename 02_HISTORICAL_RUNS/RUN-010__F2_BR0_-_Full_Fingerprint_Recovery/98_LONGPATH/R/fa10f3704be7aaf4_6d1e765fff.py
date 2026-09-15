import pandas as pd,numpy as np,json,math,joblib,pathlib,os,collections,re
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu,wilcoxon,spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
root=pathlib.Path('/mnt/data/F1_BR1_BLIND_RSOS_CORPUS_REDISCOVERY')
sets=['EARLY_POS','EARLY_NEG','LATER_VAL_POS','LATER_VAL_NEG','LATER_HOLD_POS','LATER_HOLD_NEG']
df=pd.concat([pd.read_csv(root/'05_FEATURES'/f'{s}_FEATURES.csv') for s in sets],ignore_index=True)
train=df[df.partition=='TRAIN'].copy();val=df[df.partition=='VALIDATION'].copy();hold=df[df.partition=='HOLDOUT'].copy()
features=json.load(open(root/'06_MODELS/FROZEN_FEATURE_SETS.json'));cols=features['JOINT_MULTILAYER'];model=joblib.load(root/'06_MODELS/JOINT_MULTILAYER.joblib');scaler=model.named_steps['sc']
score=pd.read_csv(root/'07_RESULTS/HOLDOUT_SCORES_INTERNAL.csv').set_index('ROW_INDEX')['JOINT_MULTILAYER_SCORE'];score.index=score.index.astype(int);score=score.reindex(hold.index)
rng=np.random.default_rng(120918)
def auc_ci(g,scorecol='SCORE',nboot=400):
 y=g.y.values; s=g[scorecol].values; pos=s[y==1];neg=s[y==0];a=roc_auc_score(y,s);vals=[]
 for _ in range(nboot):
  pp=rng.choice(pos,len(pos),replace=True);nn=rng.choice(neg,len(neg),replace=True);vals.append(roc_auc_score(np.r_[np.ones(len(pp)),np.zeros(len(nn))],np.r_[pp,nn]))
 return a,*np.quantile(vals,[.025,.975]),mannwhitneyu(pos,neg,alternative='greater').pvalue
def family(s):
 s=str(s).lower()
 for x in ['gpt-5','gpt-4o','gpt-4-1','gpt-4-5','o3','o4','o1']:
  if x in s:return x
 return s.split('-')[0] if s else 'unknown'
# system-specific
reg=pd.read_csv(root/'04_SPLITS/LATER_POS_REGISTRY.csv'); hp=hold[hold.y==1][['conversation_id','model_slug']].copy();hp['SCORE']=score.loc[hp.index].values;hp=hp.merge(reg,on='conversation_id',how='left'); neg=hold[hold.y==0].copy();neg['SCORE']=score.loc[neg.index].values
sysrows=[]
for s in ['RSOS','RSSO','RSIA','RSX']:
 sub=hp[hp[s+'_count']>=3]
 if len(sub)>=3:
  g=pd.concat([pd.DataFrame({'y':np.ones(len(sub)),'SCORE':sub.SCORE.values}),pd.DataFrame({'y':np.zeros(len(neg)),'SCORE':neg.SCORE.values})]);a,lo,hi,p=auc_ci(g);sysrows.append({'SYSTEM':s,'N_POS':len(sub),'AUC':a,'CI_LOW':lo,'CI_HIGH':hi,'P':p,'MEDIAN_SCORE':sub.SCORE.median()})
pd.DataFrame(sysrows).to_csv(root/'07_RESULTS/SYSTEM_SPECIFIC_HOLDOUT.csv',index=False)
# model/code strata
h=hold.copy();h['SCORE']=score.loc[h.index].values;h['MODEL_FAMILY']=h.model_slug.map(family);h['IS_CODE']=(h.CODE_FENCES_PER_1K_CHAR>0).astype(int)
mr=[]
for fam,g in h.groupby('MODEL_FAMILY'):
 if g.y.sum()>=8 and (1-g.y).sum()>=8:
  a,lo,hi,p=auc_ci(g);mr.append({'STRATUM':'MODEL','VALUE':fam,'N':len(g),'N_POS':int(g.y.sum()),'N_NEG':int((1-g.y).sum()),'AUC':a,'CI_LOW':lo,'CI_HIGH':hi,'P':p})
for val0,g in h.groupby('IS_CODE'):
 if g.y.sum()>=8 and (1-g.y).sum()>=8:
  a,lo,hi,p=auc_ci(g);mr.append({'STRATUM':'IS_CODE','VALUE':int(val0),'N':len(g),'N_POS':int(g.y.sum()),'N_NEG':int((1-g.y).sum()),'AUC':a,'CI_LOW':lo,'CI_HIGH':hi,'P':p})
pd.DataFrame(mr).to_csv(root/'07_RESULTS/STRATIFIED_HOLDOUT.csv',index=False)
# representation ablations
med=train[cols].median();pos=hold[hold.y==1];base=score.loc[pos.index].values
groups={'ALL_OPERATOR':[c for c in cols if c.startswith('SEM_') or c.endswith('_FIRST_POS') or c.endswith('_SPAN') or c=='USER_ASSISTANT_OPERATOR_COSINE'],'DIRECTION_ORDER':[c for c in cols if 'ORDER' in c or c in ['TURN_LENGTH_SLOPE','USER_LENGTH_SLOPE','ASSISTANT_LENGTH_SLOPE']],'RECURSION':[c for c in cols if 'RECURSION' in c],'PRESERVATION':[c for c in cols if 'PRESERVE' in c],'SWITCH_META':[c for c in cols if 'SWITCH' in c or 'META' in c],'TERMINAL':[c for c in cols if 'TERMINAL' in c]}
ar=[]
for name,cc in groups.items():
 x=pos[cols].copy()
 for c in cc:x[c]=med[c]
 s=model.predict_proba(x)[:,1];d=base-s;p=wilcoxon(d,alternative='greater').pvalue if np.any(abs(d)>1e-12) else 1.0
 ar.append({'ABLATION':name,'N_FEATURES':len(cc),'BASE_MEAN':base.mean(),'ABLATE_MEAN':s.mean(),'MEAN_DROP':d.mean(),'MEDIAN_DROP':np.median(d),'POSITIVE_DROP_FRAC':(d>0).mean(),'WILCOXON_P':p})
pd.DataFrame(ar).to_csv(root/'08_ABLATIONS/REPRESENTATION_ABLATIONS.csv',index=False)
# genealogy prototype random-null
Zt=scaler.transform(train[cols]);Zh=scaler.transform(hold[cols]);yt=train.y.values;yh=hold.y.values;ep=Zt[yt==1];en=Zt[yt==0];cent=ep.mean(0);actual=roc_auc_score(yh,-np.linalg.norm(Zh-cent,axis=1));null=[]
for _ in range(1200):
 c=en[rng.choice(len(en),len(ep),replace=True)].mean(0);null.append(roc_auc_score(yh,-np.linalg.norm(Zh-c,axis=1)))
pnull=(1+sum(x>=actual for x in null))/(len(null)+1);later=Zh[yh==1];reverse=roc_auc_score(yt,-np.linalg.norm(Zt-later.mean(0),axis=1))
pd.DataFrame([{'FORWARD_EARLY_RSOS_PROTOTYPE_AUC':actual,'RANDOM_ANCESTOR_NULL_MEAN':np.mean(null),'RANDOM_NULL_95':np.quantile(null,.95),'RANDOM_P':pnull,'REVERSED_DESCENDANT_TO_EARLY_AUC':reverse}]).to_csv(root/'09_GENEALOGY/GENEALOGY_DIRECTIONALITY.csv',index=False)
# actual LM Arena top-40 feature dependency graph
b=root/'01_INTAKE/lmarena_r1_extra';fn=next(x for x in os.listdir(b) if x.endswith('F1_BLIND_FEATURE_RELATION_GRAPH_PREHOLDOUT.csv'));lm=pd.read_csv(b/fn)
def ev(g):
 v=[]
 for _,r in lm.iterrows():
  z=spearmanr(g[r.FEATURE_1],g[r.FEATURE_2],nan_policy='omit').statistic;v.append(0 if np.isnan(z) else z)
 return np.array(v)
e0=ev(train[train.y==1]);e1=ev(hold[hold.y==1]);eneg=ev(hold[hold.y==0]);elm=lm.VALIDATION_SPEARMAN.values
def comp(name,a,b):return {'COMPARISON':name,'EDGE_VECTOR_PEARSON':np.corrcoef(a,b)[0,1],'SIGN_AGREEMENT':(np.sign(a)==np.sign(b)).mean(),'MEAN_ABS_DIFF':np.mean(np.abs(a-b))}
lmr=[comp('EARLY_RSOS_vs_LATER_LINEAGE',e0,e1),comp('EARLY_RSOS_vs_SAMEUSER_NONLINEAGE',e0,eneg),comp('EARLY_RSOS_vs_LMARENA',e0,elm),comp('LATER_LINEAGE_vs_LMARENA',e1,elm)]
pd.DataFrame(lmr).to_csv(root/'07_RESULTS/LMARENA_RELATIONAL_GRAPH_CONTROL.csv',index=False)
# topic clusters on holdout only, unsupervised, hard direct-term redaction; cap text per conversation to 80k evenly sampled chars
DIRECT_RE=re.compile(r'''(?ix)\b(?:RSOS|RSSO|RSIA|RSX|CM1|LINX|MEMSTACK|CONTAIN|UGCL|CML|SRC[-\s]?X(?:\.5)?|R\.\s*A\.\s*Elu)\b|recursive\s+symbolic\s+(?:operating\s+system|system\s+operating|identity\s+architecture|(?:eXpression|expression)(?:\s+language)?)|\b(?:tier(?:s)?|glyph(?:s)?|archetype(?:s)?|override(?:s|d|ing)?|containment|resurrection|echo\s+protocol|activation\s+protocol)\b|\b(?:lion|dragon|eagle|whale|serpent)\b''')
def redact(s):return DIRECT_RE.sub(' ',s)
def cap(s,n=80000):
 if len(s)<=n:return s
 # three evenly spaced windows
 q=n//3;return s[:q]+'\n'+s[len(s)//2-q//2:len(s)//2+q//2]+'\n'+s[-q:]
texts={}
for lab in ['LATER_HOLD_POS','LATER_HOLD_NEG']:
 for rec in map(json.loads,open(root/'04_SPLITS'/f'{lab}_SEGMENTS.jsonl',encoding='utf-8')):
  texts[rec['conversation_id']]=cap(redact('\n'.join(m.get('text','') for m in rec['messages'])))
docs=[texts.get(cid,'') for cid in h.conversation_id];tf=TfidfVectorizer(max_features=2500,min_df=3,max_df=.9,stop_words='english',ngram_range=(1,2),sublinear_tf=True);X=tf.fit_transform(docs);k=8;cl=KMeans(n_clusters=k,random_state=120918,n_init=20).fit_predict(X);h2=h.copy();h2['TOPIC_CLUSTER']=cl
trows=[]
for c,g in h2.groupby('TOPIC_CLUSTER'):
 if g.y.sum()>=5 and (1-g.y).sum()>=5:
  a,lo,hi,p=auc_ci(g,nboot=250);trows.append({'TOPIC_CLUSTER':int(c),'N':len(g),'N_POS':int(g.y.sum()),'N_NEG':int((1-g.y).sum()),'AUC':a,'CI_LOW':lo,'CI_HIGH':hi,'P':p})
pd.DataFrame(trows).to_csv(root/'07_RESULTS/WITHIN_TOPIC_HOLDOUT.csv',index=False)
# exact topic-matched pairs, greedy with huge penalty for different topic and model family
used=set();pairs=[]
for i,p in h2[h2.y==1].sort_values('raw_chars',ascending=False).iterrows():
 cand=h2[(h2.y==0)&(~h2.index.isin(used))].copy();
 if cand.empty:break
 cost=(np.log1p(cand.raw_chars)-math.log1p(p.raw_chars))**2+1.5*(np.log1p(cand.raw_turns)-math.log1p(p.raw_turns))**2+4*(cand.TOPIC_CLUSTER!=p.TOPIC_CLUSTER).astype(float)+2*(cand.MODEL_FAMILY!=p.MODEL_FAMILY).astype(float)+1*((cand.IS_CODE)!=(p.IS_CODE)).astype(float)
 j=cost.idxmin();used.add(j);pairs.append((i,j,float(cost.loc[j]),int(cand.loc[j,'TOPIC_CLUSTER']==p.TOPIC_CLUSTER)))
d=np.array([score.loc[i]-score.loc[j] for i,j,_,_ in pairs]);wins=(d>0).sum(); same=sum(x[3] for x in pairs);pbin=__import__('scipy').stats.binomtest(int(wins),len(d),.5,alternative='greater').pvalue
pd.DataFrame([{'N_PAIRS':len(d),'SAME_TOPIC_PAIRS':same,'PAIR_ACCURACY':wins/len(d),'MEAN_SCORE_DIFF':d.mean(),'BINOM_P':pbin}]).to_csv(root/'07_RESULTS/TOPIC_MODEL_COMPLEXITY_MATCHED_HOLDOUT.csv',index=False)
print('SYSTEM\n',pd.DataFrame(sysrows).to_string(index=False));print('\nSTRATA\n',pd.DataFrame(mr).to_string(index=False));print('\nABLATIONS\n',pd.DataFrame(ar).to_string(index=False));print('\nGENEALOGY',actual,np.mean(null),np.quantile(null,.95),pnull,reverse);print('\nLM\n',pd.DataFrame(lmr).to_string(index=False));print('\nTOPICS\n',pd.DataFrame(trows).to_string(index=False));print('topic matched',len(d),same,wins/len(d),d.mean(),pbin)
