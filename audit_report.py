"""Independent prediction/split audit and participant-clustered paired intervals."""
import json,argparse
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import balanced_accuracy_score
root=Path(__file__).resolve().parent;parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(root/'results'));args=parser.parse_args();out=Path(args.out);report=json.loads((out/'metrics.json').read_text());d=pd.read_csv(out/'all_oof_predictions.csv');logs=pd.read_csv(out/'inner_selection_scores.csv');splits=json.loads((out/'participant_splits.json').read_text());lookup={(r['seed'],r['task'],r['fold']):r for r in splits};audit=[]
for r in report['results']:
 p=d[(d.task==r['task'])&(d.suite==r['suite'])&(d.seed==r['seed'])];target='group' if r['task']=='subject' else 'condition';assert np.isclose(balanced_accuracy_score(p[target],p.prediction),r['balanced_accuracy']);assert len(p)==(20 if r['task']=='subject' else 600)
 keys=['subject_id'] if r['task']=='subject' else ['subject_id','trial'];assert not p.duplicated(keys).any()
 for fold,v in p.groupby('fold'):
  s=lookup[(r['seed'],r['task'],int(fold))];assert not set(s['training_subjects'])&set(s['validation_subjects']);assert set(v.subject_id)==set(s['validation_subjects'])
 audit.append({'task':r['task'],'suite':r['suite'],'seed':r['seed'],'scores_recomputed':True,'outer_participants_disjoint':True,'predictions_complete':True})
for _,r in logs.iterrows():
 a=set(r.training_subjects.split(','));b=set(r.validation_subjects.split(','));assert not a&b
 if str(r.outer_fold)!='final':
  s=lookup[(int(r.seed),r.task,int(r.outer_fold))];assert not(a|b)&set(s['validation_subjects']);assert(a|b)==set(s['training_subjects'])
(out/'audit.json').write_text(json.dumps({'experiments':audit,'inner_score_records_checked':len(logs)},indent=2))
# Seeds reuse the same participants. Average within participant before resampling.
intervals=[]
for task in ['subject','condition']:
 a=d[d.task==task].copy();target='group' if task=='subject' else 'condition'
 if task=='subject':a['score']=(a.group==a.prediction).astype(float)
 else:
  a['correct']=(a.condition==a.prediction).astype(float)
  a=a.groupby(['suite','seed','subject_id','group','condition'],as_index=False).correct.mean();a=a.groupby(['suite','seed','subject_id','group'],as_index=False).correct.mean().rename(columns={'correct':'score'})
 scores=a.groupby(['subject_id','group','suite']).score.mean().unstack('suite');rng=np.random.default_rng(183)
 for suite in ['temporal_contrasts','spatial_only','combined_selection']:
  delta=scores[suite]-scores.v2_baseline;groups=scores.index.get_level_values('group');values=[]
  for _ in range(2000):
   picks=np.concatenate([rng.choice(np.flatnonzero(groups==g),sum(groups==g),replace=True) for g in sorted(set(groups))]);values.append(delta.iloc[picks].mean())
  lo,hi=np.quantile(values,[.025,.975]);intervals.append({'task':task,'suite':suite,'mean_change_vs_v2':float(delta.mean()),'bootstrap_low':float(lo),'bootstrap_high':float(hi)})
pd.DataFrame(intervals).to_csv(out/'paired_participant_intervals.csv',index=False)
summary=pd.read_csv(out/'accuracy_by_seed.csv');table=summary.groupby(['task','suite']).balanced_accuracy.agg(['mean','min','max']).reset_index();table.to_csv(out/'stability_summary.csv',index=False)
text=['# Original EEG project: spatial and temporal experiments\n','Source data unchanged: 20 participants, 600 trials. Every model is selected in three inner participant folds and evaluated in five outer participant folds, repeated with seeds 183, 184 and 185. The same participant IDs are reused across seeds; these are not 60 independent people.\n','## Results\n','| Task | Candidate family | Original split | Mean across three splits | Split range |','|---|---|---:|---:|---:|']
for _,r in table.iterrows():
 orig=summary.loc[(summary.task==r.task)&(summary.suite==r.suite)&(summary.seed==183),'balanced_accuracy'].iloc[0];text.append(f'| {r.task} | {r.suite} | {orig*100:.2f}% | {r["mean"]*100:.2f}% | {r["min"]*100:.2f}–{r["max"]*100:.2f}% |')
text+=['\nBaseline is the exact v2 nested selection procedure, rerun on the same inputs. Combined selection adds the new candidates to that procedure; each outer fold can retain a v2 model if its inner validation score is stronger. Temporal-only and spatial-only suites are ablations, not models selected based on the outer scores.\n','## Change versus v2 across repeated splits\n','| Task | Family | Mean change | Participant-clustered bootstrap interval |','|---|---|---:|---:|']
for r in intervals:text.append(f'| {r["task"]} | {r["suite"]} | {r["mean_change_vs_v2"]*100:+.2f} points | {r["bootstrap_low"]*100:+.2f} to {r["bootstrap_high"]*100:+.2f} points |')
text+=['\nIntervals average results within each participant before resampling participants. They are exploratory, do not include the uncertainty of retraining on new cohorts, and are not corrected for multiple candidate families. Model development has repeatedly examined these same 20 people. Higher point accuracy does not establish independent generalization.\n','## Features and timing\n','The v2 features retain eight fixed 125-ms means/stds per channel. New temporal features compare later recording bins with the first bin and compare the recording\'s late and early halves. Neither the notebook nor this export establishes stimulus onset; these are recording-relative contrasts, not verified ERP amplitudes or pre-stimulus baseline correction.\n','Spatial features use within-trial OAS shrinkage covariance and correlation, then a symmetric matrix logarithm. Training-fold scaling, PCA/feature selection and classification follow extraction. The 61 consistently named scalp channels are used for spatial features; X, Y and nd are excluded from this branch only. All original trials remain, including flat-electrode trials; shrinkage stabilizes their covariance and QC records flat-channel counts. Original v2 features still use its full channel schema and flat-feature handling.\n','## Deployment candidates\n']
for task in ['subject','condition']:text.append(f'- {task}: {report[task+"_deployment"]}. Refit on all 20 participants; no independent test score for this refit.')
text+=['\nThe package contains candidate model artifacts and prediction code. Their probabilities remain uncalibrated. Existing v2 files are preserved separately and are not overwritten by these experiments.\n','## Verification\n',f'{len(audit)} task/family/seed results were independently recomputed. {len(logs)} inner-fold records were checked for training/validation overlap and outer-held-out contamination. Candidate predictions and splits are complete.\n']
if 'subject_spatial_experimental' in report:
 text+=['\n## Experimental spatial model and runtime\n',f'The separate subject_spatial_experimental.joblib artifact uses {report["subject_spatial_experimental"]["candidate"]}, selected within the spatial family by all-participant inner validation. Its repeated-split family mean was 85%, versus 76.67% for v2. The paired interval for the change is −5.00 to +23.33 percentage points: a promising signal, not established improvement.\n',f'The nested comparison took {report["seconds"]:.1f} seconds in this runtime, excluding initial extraction and final refits. No runtime speedup claim is supported.']
(root/'RESULTS.md').write_text('\n'.join(text));print(table.to_string(index=False));print(pd.DataFrame(intervals).to_string(index=False))
