"""Fixed adjustments to the original broadband participant classifier only."""
import argparse,json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'eeg_original_v3'))
import numpy as np,pandas as pd,joblib
from scipy.stats import trim_mean
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score,confusion_matrix,f1_score
from threadpoolctl import threadpool_limits
from eeg_pipeline import TRIAL_KEY
from optimize_eeg_v2 import participant_features,pipeline,prepare
from improve_original import spatial_features,digest,selection_scores

def trimmed_features(frame):
 cols=[c for c in frame if c not in TRIAL_KEY]
 keys=['subject_id','group','condition'];rows=[];indices=[]
 for key,f in frame.groupby(keys,observed=True,sort=True):
  indices.append(key);rows.append(trim_mean(f[cols].to_numpy(),.1,axis=0))
 grouped=pd.DataFrame(rows,index=pd.MultiIndex.from_tuples(indices,names=keys),columns=cols)
 wide=grouped.unstack('condition');wide.columns=[f'{condition}__{feature}' for feature,condition in wide.columns];return wide.reset_index()

def build(spatial):
 tables={}
 for kind,prefix in [('cov','logcov_'),('corr','logcorr_')]:
  f=spatial[TRIAL_KEY+[c for c in spatial if c.startswith(prefix)]]
  for agg in ['mean','median','trimmed']:
   t=trimmed_features(f) if agg=='trimmed' else participant_features(f,agg);tables[kind+'_'+agg]=t.sort_values('subject_id').reset_index(drop=True)
 specs=[]
 def add(kind,agg,n,c):
  name=f'{kind}_{agg}_pca{n}_C{c}';version=kind+'_'+agg
  if not any(s[0]==name for s in specs):specs.append((name,version,pipeline(LogisticRegression(C=c,max_iter=4000,class_weight='balanced',random_state=183),pca=n)))
 # Original three candidates first: preserves the original tie-breaking order.
 add('cov','mean',6,.1);add('cov','median',6,.1);add('corr','mean',6,.1)
 original={s[0] for s in specs}
 for kind in ['cov','corr']:
  for n in [3,6,9]:
   for c in [.01,.1]:add(kind,'mean',n,c)
 for agg in ['median','trimmed']:
  for c in [.01,.1]:add('cov',agg,6,c)
 for agg in ['median','trimmed']:add('corr',agg,6,.1)
 assert len(specs)==18
 families={'original_spatial':original,'covariance_only':{s[0] for s in specs if s[0].startswith('cov_')},'correlation_only':{s[0] for s in specs if s[0].startswith('corr_')},'adjusted_selection':{s[0] for s in specs}}
 return tables,specs,families

def run(spatial,out):
 tables,specs,families=build(spatial);data=prepare('subject',tables);meta=tables['cov_mean'][['subject_id','group']];assert len(meta)==20 and spatial.subject_id.nunique()==20 and len(spatial)==600;logs=[];rows=[];individual=[];chosen=[];splits=[];start=time.perf_counter()
 (out/'candidate_plan.json').write_text(json.dumps({'candidates':[{'name':n,'feature_set':v} for n,v,e in specs],'families':{k:sorted(v) for k,v in families.items()},'seeds':[183,184,185],'PCA':[3,6,9],'C':[.01,.1],'trimmed_fraction_each_tail':.1},indent=2))
 for seed in [183,184,185]:
  for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(meta,meta.group),1):
   ta=set(meta.iloc[tr].subject_id);vb=set(meta.iloc[va].subject_id);splits.append({'seed':seed,'fold':fold,'training_subjects':sorted(ta),'validation_subjects':sorted(vb)});scores=selection_scores('subject',data,specs,meta,ta,seed,fold,logs);cache={}
   for name,version,est in specs:
    f,x,_=data[version];train=f.subject_id.isin(ta);val=f.subject_id.isin(vb);m=clone(est).fit(x[train],f.loc[train,'group']);p=f.loc[val,['subject_id','group']].copy();p['prediction']=m.predict(x[val]);p['seed']=seed;p['fold']=fold;p['candidate']=name;individual.append(p);cache[name]=p
   for family,names in families.items():
    name=max([s[0] for s in specs if s[0] in names],key=lambda n:scores[n]);p=cache[name].drop(columns='candidate').copy();p['suite']=family;rows.append(p);chosen.append({'seed':seed,'fold':fold,'suite':family,'candidate':name,'inner_score':scores[name]})
   pd.concat(rows,ignore_index=True).to_csv(out/'predictions.csv',index=False);print('finished',seed,fold,'seconds',round(time.perf_counter()-start,1),flush=True)
 scores=selection_scores('subject',data,specs,meta,set(meta.subject_id),183,'final',logs)
 for family in ['original_spatial','adjusted_selection']:
  name,version,est=max([s for s in specs if s[0] in families[family]],key=lambda s:scores[s[0]]);f,x,cols=data[version];m=clone(est).fit(x,f.group);joblib.dump({'model':m,'columns':cols,'feature_set':version,'candidate':name,'training_subject_ids':meta.subject_id.tolist(),'channels':json.loads((out/'channels.json').read_text()),'note':'All-data exploratory refit, not independently tested; probabilities uncalibrated'},out/f'{family}_model.joblib')
 pd.DataFrame(logs).to_csv(out/'inner_scores.csv',index=False);pd.DataFrame(chosen).to_csv(out/'selected_candidates.csv',index=False);pd.concat(individual,ignore_index=True).to_csv(out/'individual_predictions_exploratory.csv',index=False);(out/'splits.json').write_text(json.dumps(splits,indent=2));pred=pd.concat(rows,ignore_index=True);metrics=[]
 for (suite,seed),p in pred.groupby(['suite','seed']):metrics.append({'suite':suite,'seed':int(seed),'balanced_accuracy':balanced_accuracy_score(p.group,p.prediction),'macro_f1':f1_score(p.group,p.prediction,average='macro'),'labels':['alcoholic','control'],'confusion_matrix':confusion_matrix(p.group,p.prediction,labels=['alcoholic','control']).tolist()})
 (out/'metrics.json').write_text(json.dumps({'results':metrics,'seconds_including_final_refits':time.perf_counter()-start,'participants':20,'trials':600,'condition_model':'unchanged; not evaluated here'},indent=2));print(pd.DataFrame(metrics).groupby('suite').balanced_accuracy.agg(['mean','min','max']),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--reference',default='eeg_original_v3/results');p.add_argument('--out',default='eeg_spatial_tuning/results');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);ref=Path(a.reference);identity=json.loads((ref/'inputs.json').read_text());assert digest(a.csv)==identity['source_sha256'],'Use the same original dataset source'
 with threadpool_limits(2):
  spatial=spatial_features(a.csv,ref);(out/'channels.json').write_text((ref/'spatial_channels.json').read_text());(out/'inputs.json').write_text(json.dumps({'source_sha256':digest(a.csv),'spatial_sha256':digest(ref/'spatial_trials.csv')},indent=2));run(spatial,out)
if __name__=='__main__':main()
