import pandas as pd,numpy as np,json,os,csv,math,joblib,pathlib,collections
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,accuracy_score,confusion_matrix
from scipy.stats import spearmanr,wilcoxon,binomtest,pearsonr
root=pathlib.Path('/mnt/data/F1_BR1_BLIND_RSOS_CORPUS_REDISCOVERY')
sets=['EARLY_POS','EARLY_NEG','LATER_VAL_POS','LATER_VAL_NEG','LATER_HOLD_POS','LATER_HOLD_NEG']
df=pd.concat([pd.read_csv(root/'05_FEATURES'/f'{s}_FEATURES.csv') for s in sets],ignore_index=True)
# identifiers excluded from any classifier
meta={'conversation_id','set_label','partition','y','model_slug','raw_chars','raw_turns','REDACTED_SPANS','N_TURNS','TOTAL_CHARS'}
R1=['TYPE_TOKEN_RATIO','UPPERCASE_RATIO','DIGIT_RATIO','COMPRESSION_RATIO','USER_CHAR_SHARE','MEAN_MAX_TURN_RATIO','WORDS_PER_1K_CHAR','TURNS_PER_1K_CHAR','PARAGRAPHS_PER_TURN','CODE_FENCES_PER_1K_CHAR','HEADERS_PER_1K_CHAR','LIST_ITEMS_PER_1K_CHAR','COLONS_PER_1K_CHAR','BRACKETS_PER_1K_CHAR']
operator=[c for c in df.columns if c.startswith('SEM_') or c.endswith('_FIRST_POS') or c.endswith('_SPAN') or c in ['ARROW_PER_1K_CHAR','NESTED_LIST_PER_1K_CHAR','NUMBERED_LIST_PER_1K_CHAR','TURN_LENGTH_SLOPE','USER_LENGTH_SLOPE','ASSISTANT_LENGTH_SLOPE','USER_ASSISTANT_OPERATOR_COSINE']]
# hardest surface-free set excludes lexical diversity/compression/word-count and raw length/redaction counts
surface=[c for c in df.columns if c not in meta and c not in operator and c not in ['TYPE_TOKEN_RATIO','COMPRESSION_RATIO','WORDS_PER_1K_CHAR'] and pd.api.types.is_numeric_dtype(df[c])]
joint=list(dict.fromkeys(surface+operator))
feature_sets={'SURFACE_FREE':surface,'OPERATOR_RELATIONAL':operator,'JOINT_MULTILAYER':joint,'R1_COMPATIBLE':R1}
train=df[df.partition=='TRAIN'].copy(); val=df[df.partition=='VALIDATION'].copy(); hold=df[df.partition=='HOLDOUT'].copy()
# matching helper
def family(s):
 s=str(s).lower()
 for x in ['gpt-5','gpt-4o','gpt-4-1','gpt-4-5','o3','o4','o1']:
  if x in s:return x
 return s.split('-')[0] if s else 'unknown'
def match_sets(pos,neg):
 used=set(); pairs=[]
 # prioritize largest / hardest first
 for _,p in pos.sort_values('raw_chars',ascending=False).iterrows():
  cand=neg[~neg.index.isin(used)].copy()
  if cand.empty:break
  cand['cost']=(np.log1p(cand.raw_chars)-math.log1p(p.raw_chars))**2+1.5*(np.log1p(cand.raw_turns)-math.log1p(p.raw_turns))**2
  cand['cost'] += 2.0*(cand.model_slug.map(family)!=family(p.model_slug)).astype(float)
  cand['cost'] += 1.0*((cand.CODE_FENCES_PER_1K_CHAR>0)!=(p.CODE_FENCES_PER_1K_CHAR>0)).astype(float)
  j=cand.cost.idxmin(); used.add(j); pairs.append((p.name,j,float(cand.loc[j,'cost'])))
 return pairs
# For validation match limited by 16 neg: match each negative to closest distinct positive by reversing roles
val_pairs_tmp=match_sets(val[val.y==0],val[val.y==1]); val_pairs=[(b,a,c) for a,b,c in val_pairs_tmp]
hold_pairs=match_sets(hold[hold.y==1],hold[hold.y==0])
pd.DataFrame(val_pairs,columns=['POS_INDEX','NEG_INDEX','COST']).to_csv(root/'04_SPLITS/VALIDATION_MATCHED_PAIRS.csv',index=False)
pd.DataFrame(hold_pairs,columns=['POS_INDEX','NEG_INDEX','COST']).to_csv(root/'04_SPLITS/HOLDOUT_MATCHED_PAIRS.csv',index=False)
# stats helpers
rng=np.random.default_rng(120918)
def auc_ci(y,s,nboot=2000):
 auc=roc_auc_score(y,s); vals=[]; n=len(y); y=np.asarray(y);s=np.asarray(s)
 for _ in range(nboot):
  ix=rng.integers(0,n,n)
  if len(np.unique(y[ix]))<2:continue
  vals.append(roc_auc_score(y[ix],s[ix]))
 return auc,float(np.quantile(vals,.025)),float(np.quantile(vals,.975))
def perm_p(y,s,n=5000):
 obs=roc_auc_score(y,s); ge=1
 for _ in range(n): ge += roc_auc_score(rng.permutation(y),s)>=obs
 return ge/(n+1)
def pair_metric(pairs,score_series):
 dif=np.array([score_series.loc[i]-score_series.loc[j] for i,j,_ in pairs]); wins=int((dif>0).sum()); ties=int((dif==0).sum()); n=len(dif)-ties
 p=binomtest(wins,n,0.5,alternative='greater').pvalue if n else 1.0
 return {'N_PAIRS':len(dif),'POS_GT_NEG':wins,'TIES':ties,'PAIR_ACCURACY':wins/max(1,len(dif)),'MEAN_SCORE_DIFF':float(dif.mean()) if len(dif) else np.nan,'BINOM_P':float(p)}
results=[]; models={}; preds={}; chosen={}
for fs,cols in feature_sets.items():
 X=train[cols].replace([np.inf,-np.inf],np.nan).fillna(train[cols].median()); y=train.y.astype(int).values
 best=None
 for C in [0.01,0.03,0.1,0.3,1.0,3.0]:
  pipe=Pipeline([('sc',StandardScaler()),('lr',LogisticRegression(C=C,max_iter=5000,class_weight='balanced',solver='liblinear',random_state=120918))])
  cv=StratifiedKFold(5,shuffle=True,random_state=120918); au=cross_val_score(pipe,X,y,cv=cv,scoring='roc_auc')
  cand=(float(au.mean()),float(au.std()),C,pipe)
  if best is None or cand[0]>best[0]:best=cand
 mean,std,C,pipe=best; pipe.fit(X,y); models[fs]=pipe; chosen[fs]=cols
 for part,d in [('VALIDATION',val),('HOLDOUT',hold)]:
  xx=d[cols].replace([np.inf,-np.inf],np.nan).fillna(train[cols].median()); score=pipe.predict_proba(xx)[:,1]; ser=pd.Series(score,index=d.index); preds[(fs,part)]=ser
  auc,lo,hi=auc_ci(d.y.values,score); p=perm_p(d.y.values,score)
  results.append({'FEATURE_SET':fs,'PARTITION':part,'N':len(d),'N_POS':int(d.y.sum()),'N_NEG':int((1-d.y).sum()),'TRAIN_CV_AUC':mean,'TRAIN_CV_SD':std,'C':C,'AUC':auc,'AUC_CI_LOW':lo,'AUC_CI_HIGH':hi,'PERM_P':p,'MATCHED':False})
  pairs=val_pairs if part=='VALIDATION' else hold_pairs
  idx=[i for p0 in pairs for i in p0[:2]]; dd=d.loc[idx]; ss=np.array([ser.loc[i] for i in idx]); yy=dd.y.values
  a2,l2,h2=auc_ci(yy,ss); pp=perm_p(yy,ss)
  pr=pair_metric(pairs,ser); results.append({'FEATURE_SET':fs,'PARTITION':part,'N':len(dd),'N_POS':int(dd.y.sum()),'N_NEG':int((1-dd.y).sum()),'TRAIN_CV_AUC':mean,'TRAIN_CV_SD':std,'C':C,'AUC':a2,'AUC_CI_LOW':l2,'AUC_CI_HIGH':h2,'PERM_P':pp,'MATCHED':True,**pr})
pd.DataFrame(results).to_csv(root/'07_RESULTS/PRIMARY_CLASSIFICATION_RESULTS.csv',index=False)
# save models and feature registries
for fs,m in models.items():joblib.dump(m,root/'06_MODELS'/f'{fs}_RIDGE_LOGISTIC.joblib')
json.dump(chosen,open(root/'06_MODELS/FROZEN_FEATURE_SETS.json','w'),indent=2)
# validation threshold from matched validation for primary joint; apply untouched holdout
primary='JOINT_MULTILAYER'; vs=preds[(primary,'VALIDATION')]; vpairs=val_pairs
vidx=[i for p in vpairs for i in p[:2]]; yy=val.loc[vidx].y.values; ss=np.array([vs.loc[i] for i in vidx]); thresholds=np.unique(ss); best=(-9,None)
for t in thresholds:
 pred=(ss>=t).astype(int); tn,fp,fn,tp=confusion_matrix(yy,pred,labels=[0,1]).ravel(); bal=.5*(tp/max(1,tp+fn)+tn/max(1,tn+fp))
 if bal>best[0]:best=(bal,float(t))
th=best[1]; hs=preds[(primary,'HOLDOUT')]; hp=(hs>=th).astype(int); bal=balanced_accuracy_score(hold.y,hp)
pd.DataFrame([{'VALIDATION_MATCHED_BAL_ACC':best[0],'FROZEN_THRESHOLD':th,'HOLDOUT_BAL_ACC':bal,'HOLDOUT_SENSITIVITY':float(((hp==1)&(hold.y.values==1)).sum()/max(1,(hold.y.values==1).sum())),'HOLDOUT_SPECIFICITY':float(((hp==0)&(hold.y.values==0)).sum()/max(1,(hold.y.values==0).sum()))}]).to_csv(root/'07_RESULTS/FROZEN_THRESHOLD_HOLDOUT.csv',index=False)
# System-specific holdout scores joined to label registry
reg=pd.read_csv(root/'04_SPLITS/LATER_POS_REGISTRY.csv'); hpos=hold[hold.y==1][['conversation_id','model_slug']].copy(); hpos['score']=hs.loc[hpos.index].values; hpos=hpos.merge(reg,on='conversation_id',how='left'); sysrows=[]
neg_scores=hs.loc[hold[hold.y==0].index].values
for sysn in ['RSOS','RSSO','RSIA','RSX']:
 sub=hpos[hpos[sysn+'_count']>=3];
 if len(sub)>=3:
  y=np.r_[np.ones(len(sub)),np.zeros(len(neg_scores))]; s=np.r_[sub.score.values,neg_scores]; a,l,h=auc_ci(y,s);sysrows.append({'SYSTEM':sysn,'N_POS':len(sub),'AUC_VS_SAMEUSER_NEG':a,'CI_LOW':l,'CI_HIGH':h,'MEDIAN_SCORE':float(sub.score.median())})
pd.DataFrame(sysrows).to_csv(root/'07_RESULTS/SYSTEM_SPECIFIC_HOLDOUT.csv',index=False)
# Within-model holdout invariance
mrows=[]
for fam,g in hold.assign(MODEL_FAMILY=hold.model_slug.map(family),SCORE=hs).groupby('MODEL_FAMILY'):
 if g.y.sum()>=8 and (1-g.y).sum()>=8:
  a,l,h=auc_ci(g.y.values,g.SCORE.values);mrows.append({'MODEL_FAMILY':fam,'N':len(g),'N_POS':int(g.y.sum()),'N_NEG':int((1-g.y).sum()),'AUC':a,'CI_LOW':l,'CI_HIGH':h})
pd.DataFrame(mrows).to_csv(root/'07_RESULTS/WITHIN_MODEL_HOLDOUT.csv',index=False)
# code/noncode strata
crows=[]
for code,g in hold.assign(IS_CODE=(hold.CODE_FENCES_PER_1K_CHAR>0).astype(int),SCORE=hs).groupby('IS_CODE'):
 if g.y.sum()>=5 and (1-g.y).sum()>=5:
  a,l,h=auc_ci(g.y.values,g.SCORE.values);crows.append({'IS_CODE':int(code),'N':len(g),'N_POS':int(g.y.sum()),'N_NEG':int((1-g.y).sum()),'AUC':a,'CI_LOW':l,'CI_HIGH':h})
pd.DataFrame(crows).to_csv(root/'07_RESULTS/CODE_STRATIFIED_HOLDOUT.csv',index=False)
# Representation-level operator ablations: neutralize targeted coordinates to training medians, positives only; paired score drop.
med=train[chosen[primary]].median(); pos=hold[hold.y==1]; base=hs.loc[pos.index]
ab_groups={
'DIRECTION_ORDER':[c for c in chosen[primary] if ('ORDER' in c or c in ['TURN_LENGTH_SLOPE','USER_LENGTH_SLOPE','ASSISTANT_LENGTH_SLOPE'])],
'RECURSION':[c for c in chosen[primary] if 'RECURSION' in c],
'PRESERVATION':[c for c in chosen[primary] if 'PRESERVE' in c],
'SWITCH_META':[c for c in chosen[primary] if ('SWITCH' in c or 'META' in c)],
'TERMINAL':[c for c in chosen[primary] if 'TERMINAL' in c],}
abrows=[]
for name,cc in ab_groups.items():
 xx=pos[chosen[primary]].copy();
 for c in cc:xx[c]=med[c]
 sc=models[primary].predict_proba(xx)[:,1]; diff=base.values-sc
 stat=wilcoxon(diff,alternative='greater',zero_method='wilcox').pvalue if np.any(np.abs(diff)>1e-15) else 1.0
 abrows.append({'ABLATION':name,'N_FEATURES':len(cc),'N_POS':len(pos),'BASE_MEAN_SCORE':float(base.mean()),'ABLATE_MEAN_SCORE':float(sc.mean()),'MEAN_SCORE_DROP':float(diff.mean()),'MEDIAN_SCORE_DROP':float(np.median(diff)),'POSITIVE_DROP_FRAC':float((diff>0).mean()),'WILCOXON_P_GREATER':float(stat)})
pd.DataFrame(abrows).to_csv(root/'08_ABLATIONS/REPRESENTATION_ABLATIONS.csv',index=False)
# Genealogy: early-RSOS prototype vs random same-user ancestor prototypes in primary standardized feature space.
cols=chosen[primary]; scaler=models[primary].named_steps['sc']; Ztrain=scaler.transform(train[cols]); Zh=scaler.transform(hold[cols]); early=Ztrain[train.y.values==1]; neganc=Ztrain[train.y.values==0]; target=hold.y.values
cent=early.mean(0); scores=-np.linalg.norm(Zh-cent,axis=1); actual=roc_auc_score(target,scores)
null=[]
for _ in range(3000):
 ix=rng.choice(len(neganc),size=len(early),replace=True); c=neganc[ix].mean(0); null.append(roc_auc_score(target,-np.linalg.norm(Zh-c,axis=1)))
pnull=(1+sum(x>=actual for x in null))/(len(null)+1)
# reversed descendant prototype diagnostic
laterZ=Zh[target==1]; score_early=-np.linalg.norm(Ztrain-laterZ.mean(0),axis=1); rev=roc_auc_score(train.y.values,score_early)
pd.DataFrame([{'FORWARD_EARLY_RSOS_PROTOTYPE_AUC':actual,'RANDOM_ANCESTOR_NULL_MEAN_AUC':float(np.mean(null)),'RANDOM_NULL_95':float(np.quantile(null,.95)),'RANDOM_ANCESTOR_P':pnull,'REVERSED_DESCENDANT_TO_EARLY_DIAGNOSTIC_AUC':rev}]).to_csv(root/'09_GENEALOGY/GENEALOGY_DIRECTIONALITY.csv',index=False)
# LM Arena actual 40-edge structural dependency graph comparison
# locate extracted relation graph
base=root/'01_INTAKE/lmarena_r1_extra'; fn=next(x for x in os.listdir(base) if x.endswith('F1_BLIND_FEATURE_RELATION_GRAPH_PREHOLDOUT.csv')); lm=pd.read_csv(base/fn)
def edgevec(g):
 vals=[]
 for _,r in lm.iterrows():
  a,b=r.FEATURE_1,r.FEATURE_2
  vals.append(spearmanr(g[a],g[b],nan_policy='omit').statistic if len(g)>=4 else np.nan)
 return np.nan_to_num(np.array(vals,float),nan=0.0)
ev_early=edgevec(train[train.y==1]); ev_later=edgevec(hold[hold.y==1]); ev_neg=edgevec(hold[hold.y==0]); ev_lm=lm.VALIDATION_SPEARMAN.to_numpy(float)
def sim(a,b):return float(np.corrcoef(a,b)[0,1])
lmrows=[{'COMPARISON':'EARLY_RSOS_vs_LATER_LINEAGE','PEARSON_EDGE_VECTOR':sim(ev_early,ev_later),'SIGN_AGREEMENT':float((np.sign(ev_early)==np.sign(ev_later)).mean())},{'COMPARISON':'EARLY_RSOS_vs_SAMEUSER_NONLINEAGE','PEARSON_EDGE_VECTOR':sim(ev_early,ev_neg),'SIGN_AGREEMENT':float((np.sign(ev_early)==np.sign(ev_neg)).mean())},{'COMPARISON':'EARLY_RSOS_vs_LMARENA_VALIDATION','PEARSON_EDGE_VECTOR':sim(ev_early,ev_lm),'SIGN_AGREEMENT':float((np.sign(ev_early)==np.sign(ev_lm)).mean())},{'COMPARISON':'LATER_LINEAGE_vs_LMARENA_VALIDATION','PEARSON_EDGE_VECTOR':sim(ev_later,ev_lm),'SIGN_AGREEMENT':float((np.sign(ev_later)==np.sign(ev_lm)).mean())}]
pd.DataFrame(lmrows).to_csv(root/'07_RESULTS/LMARENA_RELATIONAL_GRAPH_CONTROL.csv',index=False)
# Save scores blinded ID registry
out=hold[['conversation_id','y','model_slug']].copy();out['JOINT_SCORE']=hs.values;out['BLIND_ID']=out.conversation_id.map(lambda x:__import__('hashlib').sha256(('F1BR1|'+x).encode()).hexdigest()[:24]);out[['BLIND_ID','y','model_slug','JOINT_SCORE']].to_csv(root/'07_RESULTS/HOLDOUT_BLIND_SCORES.csv',index=False)
# print concise
print('\nPRIMARY RESULTS')
print(pd.DataFrame(results).query("FEATURE_SET=='JOINT_MULTILAYER'").to_string(index=False))
print('\nSYSTEM')
print(pd.DataFrame(sysrows).to_string(index=False))
print('\nMODEL')
print(pd.DataFrame(mrows).to_string(index=False))
print('\nABLATIONS')
print(pd.DataFrame(abrows).to_string(index=False))
print('\nGENEALOGY actual',actual,'nullmean',np.mean(null),'p',pnull,'reverse',rev)
print('\nLM CONTROL')
print(pd.DataFrame(lmrows).to_string(index=False))
