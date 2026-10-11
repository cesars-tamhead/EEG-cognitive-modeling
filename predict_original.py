"""Prediction with trusted v3 model artifacts; not an independent accuracy estimate."""
import argparse,warnings
from pathlib import Path
import joblib,pandas as pd
from threadpoolctl import threadpool_limits
from eeg_pipeline import read_csv,clean_data,extract_features,TRIAL_KEY
from optimize_eeg_v2 import extended_features,prepare
from improve_original import spatial_features,tables_and_specs
p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--models',required=True);p.add_argument('--out',default='predictions');p.add_argument('--subject-model',default='subject_model_v3.joblib',choices=['subject_model_v3.joblib','subject_spatial_experimental.joblib']);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
with threadpool_limits(limits=2):
 base,_=extract_features(clean_data(read_csv(a.csv)),out/'features');ext=extended_features(a.csv,out/'features');sp=spatial_features(a.csv,out/'features');tables,_=tables_and_specs(base,ext,sp)
 for task in ['subject','condition']:
  b=joblib.load(Path(a.models)/(a.subject_model if task=='subject' else 'condition_model_v3.joblib'));frame=tables[task][b['feature_set']];model=b['model']
  if set(frame.subject_id)&set(b['training_subject_ids']):warnings.warn('Participants overlap training: predictions are not an independent evaluation.')
  x=frame.reindex(columns=b['columns'])
  if x.isna().mean(axis=1).max()>.1:raise ValueError('Missing electrode or condition coverage exceeds 10%; refusing incomplete prediction')
  ids=['subject_id','group'] if task=='subject' else TRIAL_KEY;d=frame[ids].copy();d['prediction']=model.predict(x.to_numpy())
  if hasattr(model,'predict_proba'):
   prob=model.predict_proba(x.to_numpy())
   for i,label in enumerate(model.classes_):d[f'uncalibrated_probability_{label}']=prob[:,i]
  d.to_csv(out/f'{task}_predictions.csv',index=False)
