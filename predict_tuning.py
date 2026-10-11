"""Participant prediction using trusted saved models; not independent evaluation."""
import argparse,json,warnings,joblib
from pathlib import Path
from threadpoolctl import threadpool_limits
from run_tuning import build,spatial_features

def main():
 p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--model',required=True);p.add_argument('--features-dir',default='spatial_prediction_features');p.add_argument('--out',default='participant_predictions.csv');a=p.parse_args()
 with threadpool_limits(2):
  spatial=spatial_features(a.csv,a.features_dir);b=joblib.load(a.model)
  if json.loads((Path(a.features_dir)/'spatial_channels.json').read_text())!=b['channels']:raise ValueError('Electrodes differ from training schema')
  if set(spatial.subject_id)&set(b['training_subject_ids']):warnings.warn('Training participant overlap: these predictions are not independent accuracy')
  tables,_,_=build(spatial);f=tables[b['feature_set']];x=f.reindex(columns=b['columns'])
  if x.isna().any().any():raise ValueError('Incomplete condition/electrode coverage')
  result=f[['subject_id','group']].copy();result['prediction']=b['model'].predict(x.to_numpy());prob=b['model'].predict_proba(x.to_numpy())
  for i,label in enumerate(b['model'].classes_):result['uncalibrated_probability_'+label]=prob[:,i]
  result.to_csv(a.out,index=False)
if __name__=='__main__':main()
