"""Original-data spatial/temporal ablations; nested, repeated participant-separated CV."""
from __future__ import annotations
import argparse,hashlib,json,time,warnings
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.signal import butter,sosfiltfilt
from sklearn.base import clone
from sklearn.covariance import oas
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from eeg_pipeline import read_csv,clean_data,KEY,TRIAL_KEY
from optimize_eeg_v2 import build_tables,participant_features,pipeline,candidates,prepare,score_predictions,extended_features
SEEDS=[183,184,185]

def digest(path):
 with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def log_vector(cov):
 eig,q=np.linalg.eigh(cov);floor=max(np.trace(cov)/len(cov)*1e-6,1e-10)
 log=(q*np.log(np.maximum(eig,floor)))@q.T
 i,j=np.triu_indices(len(cov));v=log[i,j];return v*np.where(i==j,1.,np.sqrt(2.))

def spatial_epoch(values):
 cov,shrink=oas(values.T)
 sd=np.sqrt(np.maximum(np.diag(cov),1e-10));corr=cov/sd[:,None]/sd[None,:]
 return np.r_[log_vector(cov),log_vector(corr)],shrink

def spatial_features(csv,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);identity={'source_sha256':digest(csv),'extraction_version':1,'filter':'order4 1-40Hz 256Hz','exclude':['X','Y','ND']}
 dest=out/'spatial_trials.csv';manifest=out/'spatial_identity.json'
 if dest.exists() and manifest.exists() and json.loads(manifest.read_text())==identity:return pd.read_csv(dest)
 d=clean_data(read_csv(csv));channels=sorted(c for c in d.channel.astype(str).unique() if c.upper() not in ['X','Y','ND'])
 sos=butter(4,[1,40],fs=256,btype='bandpass',output='sos');rows=[];quality=[]
 for idx,(key,t) in enumerate(d.groupby(TRIAL_KEY,observed=True,sort=True)):
  matrix=t.pivot(index='channel',columns='sample',values='voltage_uv').reindex(index=channels,columns=range(256))
  if matrix.isna().any().any():raise ValueError('Spatial features require all selected electrodes and 256 complete samples')
  v=sosfiltfilt(sos,matrix.to_numpy(),axis=-1);features,shrink=spatial_epoch(v)
  n=len(features)//2;names=[f'logcov_{i}' for i in range(n)]+[f'logcorr_{i}' for i in range(n)]
  rows.append(dict(zip(TRIAL_KEY,key),**dict(zip(names,features))))
  quality.append(dict(zip(TRIAL_KEY,key),shrinkage=float(shrink),flat_channels=int((v.std(axis=1)<1e-8).sum())))
  if idx%100==0:print('spatial extraction',idx,flush=True)
 frame=pd.DataFrame(rows);frame.to_csv(dest,index=False);pd.DataFrame(quality).to_csv(out/'spatial_quality.csv',index=False)
 manifest.write_text(json.dumps(identity,indent=2));(out/'spatial_channels.json').write_text(json.dumps(channels,indent=2));return frame

def temporal_features(extended):
 """Relative-to-first-bin contrasts, not stimulus-aligned ERP measurements."""
 d=extended[KEY].copy()
 for b in range(1,8):d[f'rel_bin{b}_mean_uv']=extended[f'bin{b}_mean_uv']-extended['bin0_mean_uv']
 d['late_minus_early_uv']=extended[[f'bin{b}_mean_uv' for b in range(4,8)]].mean(axis=1)-extended[[f'bin{b}_mean_uv' for b in range(4)]].mean(axis=1)
 from eeg_pipeline import trial_table
 return trial_table(d)

def join(a,b):
 keys=['subject_id','group'] if 'trial' not in a else TRIAL_KEY
 if not a[keys].equals(b[keys]):raise ValueError('Feature identity/order mismatch')
 duplicate=(set(a)-set(keys))&(set(b)-set(keys))
 if duplicate:raise ValueError('Feature name collision')
 return pd.concat([a,b.drop(columns=keys)],axis=1)

def tables_and_specs(base,extended,spatial):
 tables=build_tables(base,extended);temp=temporal_features(extended).sort_values(['subject_id','trial']).reset_index(drop=True);spatial=spatial.sort_values(['subject_id','trial']).reset_index(drop=True)
 cov=spatial[TRIAL_KEY+[c for c in spatial if c.startswith('logcov_')]].copy();corr=spatial[TRIAL_KEY+[c for c in spatial if c.startswith('logcorr_')]].copy()
 for name,frame in [('temporal',temp),('cov',cov),('corr',corr)]:
  tables['condition'][name]=frame
  for agg in ['mean','median']:tables['subject'][name+'_'+agg]=participant_features(frame,agg).sort_values('subject_id').reset_index(drop=True)
 for agg in ['mean','median']:
  tables['subject']['hybrid_'+agg]=join(tables['subject']['ext_'+agg],tables['subject']['cov_'+agg])
 tables['condition']['hybrid']=join(tables['condition']['ext'],cov)
 specs={}
 def lr(c):return LogisticRegression(C=c,max_iter=4000,class_weight='balanced',random_state=183)
 specs['subject']=candidates('subject')+[
  ('relative_mean_pca6_lr','temporal_mean',pipeline(lr(.1),pca=6)),
  ('relative_median_select32_lr','temporal_median',pipeline(lr(.01),k=32)),
  ('cov_mean_pca6_lr','cov_mean',pipeline(lr(.1),pca=6)),
  ('cov_median_pca6_lr','cov_median',pipeline(lr(.1),pca=6)),
  ('corr_mean_pca6_lr','corr_mean',pipeline(lr(.1),pca=6)),
  ('hybrid_mean_pca6_lr','hybrid_mean',pipeline(lr(.1),pca=6)),
  ('hybrid_median_select64_lr','hybrid_median',pipeline(lr(.1),k=64))]
 specs['condition']=candidates('condition')+[
  ('relative_select128_lr','temporal',pipeline(lr(.01),k=128)),
  ('relative_select128_rbf','temporal',pipeline(SVC(C=1,class_weight='balanced'),k=128)),
  ('cov_select128_lr','cov',pipeline(lr(.01),k=128)),
  ('corr_select128_lr','corr',pipeline(lr(.01),k=128)),
  ('cov_pca12_lr','cov',pipeline(lr(.1),pca=12)),
  ('hybrid_select256_lr','hybrid',pipeline(lr(.1),k=256)),
  ('hybrid_select128_rbf','hybrid',pipeline(SVC(C=1,class_weight='balanced'),k=128))]
 return tables,specs

def suites(task,specs):
 baseline=set(n for n,_,_ in candidates(task));temporal={n for n,_,_ in specs if n.startswith('relative_')};spatial={n for n,_,_ in specs if n.startswith(('cov_','corr_'))}
 return {'v2_baseline':baseline,'temporal_contrasts':temporal,'spatial_only':spatial,'combined_selection':set(n for n,_,_ in specs)}

def selection_scores(task,data,specs,meta,ids,seed,outer,log):
 m=meta[meta.subject_id.isin(ids)].reset_index(drop=True);splits=list(StratifiedKFold(3,shuffle=True,random_state=284).split(m,m.group));scores={};target='group' if task=='subject' else 'condition'
 for name,version,est in specs:
  frame,x,_=data[version];values=[]
  for inner,(tr,va) in enumerate(splits,1):
   a=set(m.iloc[tr].subject_id);b=set(m.iloc[va].subject_id);assert not a&b;train=frame.subject_id.isin(a);val=frame.subject_id.isin(b)
   model=clone(est).fit(x[train],frame.loc[train,target]);pred=model.predict(x[val])
   if task=='condition':
    v=pd.DataFrame({'subject_id':frame.loc[val,'subject_id'],'truth':frame.loc[val,target],'prediction':pred});score=np.mean([balanced_accuracy_score(d.truth,d.prediction) for _,d in v.groupby('subject_id')])
   else:score=balanced_accuracy_score(frame.loc[val,target],pred)
   values.append(score);log.append({'task':task,'seed':seed,'outer_fold':outer,'candidate':name,'feature_set':version,'inner_fold':inner,'balanced_accuracy':float(score),'training_subjects':','.join(sorted(a)),'validation_subjects':','.join(sorted(b))})
  scores[name]=float(np.mean(values))
 return scores

def experiment(base,extended,spatial,out,seeds=SEEDS):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);tables,specs=tables_and_specs(base,extended,spatial);meta=tables['subject']['base_mean'][['subject_id','group']].copy();logs=[];rows=[];splitlog=[];selected=[];start=time.perf_counter()
 for seed in seeds:
  splits=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(meta,meta.group))
  for task in ['subject','condition']:
   data=prepare(task,tables[task]);groups=suites(task,specs[task]);target='group' if task=='subject' else 'condition'
   for fold,(tr,va) in enumerate(splits,1):
    trainids=set(meta.iloc[tr].subject_id);validids=set(meta.iloc[va].subject_id);assert not trainids&validids
    splitlog.append({'seed':seed,'task':task,'fold':fold,'training_subjects':sorted(trainids),'validation_subjects':sorted(validids)})
    scores=selection_scores(task,data,specs[task],meta,trainids,seed,fold,logs);cache={}
    for suite,names in groups.items():
     best=max([s for s in specs[task] if s[0] in names],key=lambda s:scores[s[0]])
     name,version,est=best;frame,x,_=data[version];train=frame.subject_id.isin(trainids);val=frame.subject_id.isin(validids)
     if name not in cache:cache[name]=clone(est).fit(x[train],frame.loc[train,target])
     model=cache[name];pred=frame.loc[val,['subject_id','group'] if task=='subject' else TRIAL_KEY].copy();pred['prediction']=model.predict(x[val]);pred['seed']=seed;pred['fold']=fold;pred['task']=task;pred['suite']=suite
     if task=='subject':
      if hasattr(model,'predict_proba'):pred['alcoholic_score']=model.predict_proba(x[val])[:,list(model.classes_).index('alcoholic')];pred['score_type']='uncalibrated_probability'
      else:pred['alcoholic_score']=model.decision_function(x[val])*(-1 if model.classes_[1]!='alcoholic' else 1);pred['score_type']='decision_margin'
     rows.append(pred);selected.append({'seed':seed,'task':task,'fold':fold,'suite':suite,'candidate':name,'feature_set':version,'inner_score':scores[name]})
    print('finished seed/task/fold',seed,task,fold,'elapsed',round(time.perf_counter()-start,1),flush=True)
   # Checkpoint after each task, to preserve long experiments.
   pd.concat(rows,ignore_index=True).to_csv(out/'all_oof_predictions.csv',index=False);pd.DataFrame(logs).to_csv(out/'inner_selection_scores.csv',index=False)
 predictions=pd.concat(rows,ignore_index=True);pd.DataFrame(selected).to_csv(out/'selected_candidates.csv',index=False);(out/'participant_splits.json').write_text(json.dumps(splitlog,indent=2))
 summary=[]
 for (task,suite,seed),d in predictions.groupby(['task','suite','seed']):
  target='group' if task=='subject' else 'condition';r=score_predictions(d,target,sorted(d[target].unique()));r.update(task=task,suite=suite,seed=int(seed))
  if task=='subject':r['mean_fold_auc']=float(np.mean([roc_auc_score(v.group.eq('alcoholic'),v.alcoholic_score) for _,v in d.groupby('fold')]))
  else:r['participant_mean_balanced_accuracy']=float(np.mean([balanced_accuracy_score(v.condition,v.prediction) for _,v in d.groupby('subject_id')]))
  summary.append(r)
 report={'subjects':len(meta),'trials':len(tables['condition']['base']),'seeds':seeds,'nested_validation':'5 outer participant folds; 3 inner participant folds; suite selection uses inner folds only','seconds':time.perf_counter()-start,'timing':'Recording-relative bins only; stimulus onset is unverified','caveat':'Exploratory development on repeatedly examined 20 participants; repeated seeds are not independent cohorts','results':summary};(out/'metrics.json').write_text(json.dumps(report,indent=2))
 pd.DataFrame([{k:v for k,v in r.items() if k not in ['labels','confusion_matrix']} for r in summary]).to_csv(out/'accuracy_by_seed.csv',index=False)
 # One final deployable selection per task, using all-subject inner CV only.
 for task in ['subject','condition']:
  data=prepare(task,tables[task]);target='group' if task=='subject' else 'condition';scores=selection_scores(task,data,specs[task],meta,set(meta.subject_id),183,'final',logs);name,version,est=max(specs[task],key=lambda s:scores[s[0]]);frame,x,cols=data[version];model=clone(est).fit(x,frame[target]);joblib.dump({'model':model,'columns':cols,'feature_set':version,'candidate':name,'task':task,'training_subject_ids':meta.subject_id.tolist(),'note':'All-subject refit; no independent test score; probabilities uncalibrated'},out/f'{task}_model_v3.joblib')
  report[task+'_deployment']={'candidate':name,'feature_set':version,'inner_score':scores[name]}
  if task=='subject':
   spatial_names=suites(task,specs[task])['spatial_only'];sn,sv,se=max([s for s in specs[task] if s[0] in spatial_names],key=lambda s:scores[s[0]]);sf,sx,sc=data[sv];sm=clone(se).fit(sx,sf[target]);joblib.dump({'model':sm,'columns':sc,'feature_set':sv,'candidate':sn,'task':task,'training_subject_ids':meta.subject_id.tolist(),'note':'Exploratory spatial-only all-subject refit; no independent test score; probabilities uncalibrated'},out/'subject_spatial_experimental.joblib');report['subject_spatial_experimental']={'candidate':sn,'feature_set':sv,'inner_score':scores[sn]}
 (out/'metrics.json').write_text(json.dumps(report,indent=2));pd.DataFrame(logs).to_csv(out/'inner_selection_scores.csv',index=False);return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--baseline-features',required=True);p.add_argument('--extended-features');p.add_argument('--out',default='results');p.add_argument('--seeds',default='183,184,185');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 base=pd.read_csv(a.baseline_features)
 if a.extended_features:
  cache_identity=Path(a.extended_features).parent/'feature_cache_identity.json'
  if not cache_identity.exists():raise ValueError('Cached extended features require their source-identity manifest; otherwise omit --extended-features to regenerate.')
  identity=json.loads(cache_identity.read_text())
  if identity['raw_sha256']!=digest(a.csv) or identity['baseline_features_sha256']!=digest(a.baseline_features):raise ValueError('Cached v2 features do not match this source/baseline.')
  extended=pd.read_csv(a.extended_features)
 else:extended=extended_features(a.csv,out)
 # Cached baseline/extended files are required to come from this exact source.
 (out/'inputs.json').write_text(json.dumps({'source_sha256':digest(a.csv),'baseline_sha256':digest(a.baseline_features),'extended_sha256':digest(a.extended_features) if a.extended_features else None},indent=2))
 with threadpool_limits(limits=2):spatial=spatial_features(a.csv,out);experiment(base,extended,spatial,out,[int(s) for s in a.seeds.split(',')])
if __name__=='__main__':main()
