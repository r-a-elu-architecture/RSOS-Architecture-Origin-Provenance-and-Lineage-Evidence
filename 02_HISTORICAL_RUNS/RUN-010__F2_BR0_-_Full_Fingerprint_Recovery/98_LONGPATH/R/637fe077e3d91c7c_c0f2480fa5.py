import pandas as pd,numpy as np,json,math,joblib,pathlib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,confusion_matrix
from scipy.stats import mannwhitneyu,binomtest
root=pathlib.Path('/mnt/data/F1_BR1_BLIND_RSOS_CORPUS_REDISCOVERY')
sets=['EARLY_POS','EARLY_NEG','LATER_VAL_POS','LATER_VAL_NEG','LATER_HOLD_POS','LATER_HOLD_NEG']
df=pd.concat([pd.read_csv(root/'05_FEATURES'/f'{s}_FEATURES.csv') for s in sets],ignore_index=True)
meta={'conversation_id','set_label','partition','y','model_slug','raw_chars','raw_turns','REDACTED_SPANS','N_TURNS','TOTAL_CHARS'}
R1=['TYPE_TOKEN_RATIO','UPPERCASE_RATIO','DIGIT_RATIO','COMPRESSION_RATIO','USER_CHAR_SHARE','MEAN_MAX_TURN_RATIO','WORDS_PER_1K_CHAR','TURNS_PER_1K_CHAR','PARAGRAPHS_PER_TURN','CODE_FENCES_PER_1K_CHAR','HEADERS_PER_1K_CHAR','LIST_ITEMS_PER_1K_CHAR','COLONS_PER_1K_CHAR','BRACKETS_PER_1K_CHAR']
operator=[c for c in df.columns if c.startswith('SEM_') or c.endswith('_FIRST_POS') or c.endswith('_SPAN') or c in ['ARROW_PER_1K_CHAR','NESTED_LIST_PER_1K_CHAR','NUMBERED_LIST_PER_1K_CHAR','TURN_LENGTH_SLOPE','USER_LENGTH_SLOPE','ASSISTANT_LENGTH_SLOPE','USER_ASSISTANT_OPERATOR_COSINE']]
surface=[c for c in df.columns if c not in meta and c not in operator and c not in ['TYPE_TOKEN_RATIO','COMPRESSION_RATIO','WORDS_PER_1K_CHAR'] and pd.api.types.is_numeric_dtype(df[c])]
joint=list(dict.fromkeys(surface+operator))
feature_sets={'SURFACE_FREE':surface,'OPERATOR_RELATIONAL':operator,'JOINT_MULTILAYER':joint,'R1_COMPATIBLE':R1}
train=df[df.partition=='TRAIN'].copy(); val=df[df.partition=='VALIDATION'].copy(); hold=df[df.partition=='HOLDOUT'].copy()
# matching
def family(s):
 s=str(s).lower()
 for x in ['gpt-5','gpt-4o','gpt-4-1','gpt-4-5','o3','o4','o1']:
  if x in s:return x
 return s.split('-')[0] if s else 'unknown'
def match(pos,neg):
 used=set(); pairs=[]
 for _,p in pos.sort_values('raw_chars',ascending=False).iterrows():
  cand=neg[~neg.index.isin(used)].copy()
  if cand.empty:break
  cost=(np.log1p(cand.raw_chars)-math.log1p(p.raw_chars))**2+1.5*(np.log1p(cand.raw_turns)-math.log1p(p.raw_turns))**2
  cost+=2.0*(cand.model_slug.map(family)!=family(p.model_slug)).astype(float)
  cost+=1.0*((cand.CODE_FENCES_PER_1K_CHAR>0)!=(p.CODE_FENCES_PER_1K_CHAR>0)).astype(float)
  j=cost.idxmin();used.add(j);pairs.append((p.name,j,float(cost.loc[j])))
 return pairs
vp0=match(val[val.y==0],val[val.y==1]);vp=[(b,a,c) for a,b,c in vp0];hp=match(hold[hold.y==1],hold[hold.y==0])
pd.DataFrame(vp,columns=['POS_INDEX','NEG_INDEX','COST']).to_csv(root/'04_SPLITS/VALIDATION_MATCHED_PAIRS.csv',index=False)
pd.DataFrame(hp,columns=['POS_INDEX','NEG_INDEX','COST']).to_csv(root/'04_SPLITS/HOLDOUT_MATCHED_PAIRS.csv',index=False)
rng=np.random.default_rng(120918)
def auc_stats(y,s,boot=False):
 y=np.asarray(y,int);s=np.asarray(s,float);pos=s[y==1];neg=s[y==0];auc=roc_auc_score(y,s);p=float(mannwhitneyu(pos,neg,alternative='greater').pvalue)
 lo=hi=np.nan
 if boot:
  vals=[]
  for _ in range(700):
   pp=rng.choice(pos,size=len(pos),replace=True);nn=rng.choice(neg,size=len(neg),replace=True);vals.append(roc_auc_score(np.r_[np.ones(len(pp)),np.zeros(len(nn))],np.r_[pp,nn]))
  lo,hi=np.quantile(vals,[.025,.975])
 return float(auc),float(lo),float(hi),p
rows=[];models={};preds={};chosen={}
for name,cols in feature_sets.items():
 X=train[cols].replace([np.inf,-np.inf],np.nan).fillna(train[cols].median());y=train.y.astype(int).values
 best=None
 for C in [0.01,0.03,0.1,0.3,1,3]:
  pipe=Pipeline([('sc',StandardScaler()),('lr',LogisticRegression(C=C,max_iter=3000,class_weight='balanced',solver='liblinear',random_state=120918))])
  sc=cross_val_score(pipe,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=120918),scoring='roc_auc')
  if best is None or sc.mean()>best[0]:best=(sc.mean(),sc.std(),C,pipe)
 mean,std,C,pipe=best;pipe.fit(X,y);models[name]=pipe;chosen[name]=cols;joblib.dump(pipe,root/'06_MODELS'/f'{name}.joblib')
 for part,d,pairs in [('VALIDATION',val,vp),('HOLDOUT',hold,hp)]:
  xx=d[cols].replace([np.inf,-np.inf],np.nan).fillna(train[cols].median());s=pipe.predict_proba(xx)[:,1];ser=pd.Series(s,index=d.index);preds[(name,part)]=ser
  a,lo,hi,p=auc_stats(d.y.values,s,boot=(name=='JOINT_MULTILAYER'))
  rows.append({'FEATURE_SET':name,'PARTITION':part,'MATCHED':False,'N':len(d),'N_POS':int(d.y.sum()),'N_NEG':int((1-d.y).sum()),'TRAIN_CV_AUC':float(mean),'TRAIN_CV_SD':float(std),'C':C,'AUC':a,'CI_LOW':lo,'CI_HIGH':hi,'MW_P':p})
  dif=np.array([ser.loc[i]-ser.loc[j] for i,j,_ in pairs]);wins=int((dif>0).sum());ties=int((dif==0).sum());n=len(dif)-ties;bp=float(binomtest(wins,n,.5,alternative='greater').pvalue) if n else 1.
  # matched AUC
  ids=[q for pair in pairs for q in pair[:2]];md=d.loc[ids];ms=np.array([ser.loc[q] for q in ids]);ma,_,_,mp=auc_stats(md.y.values,ms,boot=False)
  rows.append({'FEATURE_SET':name,'PARTITION':part,'MATCHED':True,'N':len(ids),'N_POS':int(md.y.sum()),'N_NEG':int((1-md.y).sum()),'TRAIN_CV_AUC':float(mean),'TRAIN_CV_SD':float(std),'C':C,'AUC':ma,'CI_LOW':np.nan,'CI_HIGH':np.nan,'MW_P':mp,'PAIR_ACCURACY':wins/max(1,len(dif)),'PAIR_MEAN_DIFF':float(dif.mean()),'PAIR_BINOM_P':bp})
pd.DataFrame(rows).to_csv(root/'07_RESULTS/PRIMARY_CLASSIFICATION_RESULTS.csv',index=False)
json.dump(chosen,open(root/'06_MODELS/FROZEN_FEATURE_SETS.json','w'),indent=2)
# threshold from matched validation joint
name='JOINT_MULTILAYER'; ser=preds[(name,'VALIDATION')]; ids=[q for pair in vp for q in pair[:2]];yy=val.loc[ids].y.values;ss=np.array([ser.loc[q] for q in ids]);best=(-1,None)
for t in np.unique(ss):
 pred=(ss>=t).astype(int);tn,fp,fn,tp=confusion_matrix(yy,pred,labels=[0,1]).ravel();bal=.5*(tp/max(1,tp+fn)+tn/max(1,tn+fp));
 if bal>best[0]:best=(bal,float(t))
hs=preds[(name,'HOLDOUT')];pr=(hs.values>=best[1]).astype(int);bal=balanced_accuracy_score(hold.y.values,pr)
pd.DataFrame([{'VALIDATION_MATCHED_BAL_ACC':best[0],'FROZEN_THRESHOLD':best[1],'HOLDOUT_BAL_ACC':bal,'SENSITIVITY':float(((pr==1)&(hold.y.values==1)).sum()/hold.y.sum()),'SPECIFICITY':float(((pr==0)&(hold.y.values==0)).sum()/(1-hold.y).sum())}]).to_csv(root/'07_RESULTS/FROZEN_THRESHOLD_HOLDOUT.csv',index=False)
# save scores and indices for secondary script
for part,d in [('VALIDATION',val),('HOLDOUT',hold)]:
 out=d[['conversation_id','y','model_slug','raw_chars','raw_turns']].copy()
 for name in feature_sets:out[name+'_SCORE']=preds[(name,part)].loc[d.index].values
 out.to_csv(root/'07_RESULTS'/f'{part}_SCORES_INTERNAL.csv',index=True,index_label='ROW_INDEX')
print(pd.DataFrame(rows).to_string(index=False))
print('threshold',best,'holdbal',bal)
