"""Fixed-data EEG model improvement with nested participant-separated validation.

Run python optimize_eeg.py --csv RAW.csv --baseline-features channel_features.csv --out results
The previous full cleaned-data ZIP is also accepted. No source data is changed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from eeg_pipeline import read_csv, clean_data, signal_features, trial_table, subject_table, KEY, TRIAL_KEY, SEED


def extended_features(csv_path, out):
    """All complete recordings retained, including flat ones in the QC table.

    Derived features from flat epochs are missing rather than meaningful zero power.
    No artifact-amplitude threshold or arbitrary baseline correction is used.
    Eight fixed 125-ms recording bins capture waveform changes without asserting
    stimulus onset. Spectrum is computed for the whole second, not short bins.
    """
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    raw = read_csv(csv_path); clean = clean_data(raw); del raw
    sos = butter(4, [1, 40], fs=256, btype='bandpass', output='sos')
    rows, quality = [], []
    for i, (key, epoch) in enumerate(clean.groupby(KEY, observed=True, sort=True)):
        meta = dict(zip(KEY, key)); epoch = epoch.sort_values('sample')
        if len(epoch) != 256 or not np.array_equal(epoch['sample'].to_numpy(), np.arange(256)):
            quality.append({**meta, 'status': 'incomplete_or_gapped', 'n_samples': len(epoch)})
            continue
        raw_v = epoch.voltage_uv.to_numpy(dtype=float)
        v = sosfiltfilt(sos, raw_v)
        flat = bool(np.std(v) < 1e-8)
        features = signal_features(v, 256)
        for b, block in enumerate(v.reshape(8, 32)):
            features[f'bin{b}_mean_uv'] = block.mean()
            features[f'bin{b}_std_uv'] = block.std(ddof=1)
        if flat:
            features = {k: np.nan for k in features}
        rows.append({**meta, **features})
        quality.append({**meta, 'status': 'flat_features_missing' if flat else 'used',
                        'n_samples': 256, 'raw_max_abs_uv': float(np.abs(raw_v).max())})
        if i and i % 5000 == 0:
            print(f'Extended extraction: {i:,} channel epochs', flush=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError('No complete epochs; expected 0..255 samples at 256 Hz.')
    temporary=out/'extended_channel_features.tmp.csv'
    frame.to_csv(temporary,index=False)
    saved=pd.read_csv(temporary)
    if len(saved)!=len(frame) or not saved[KEY].equals(frame[KEY].astype(saved[KEY].dtypes.to_dict())):
        raise ValueError('Saved extended features failed row/identity integrity verification')
    temporary.replace(out/'extended_channel_features.csv')
    pd.DataFrame(quality).to_csv(out / 'quality_v2.csv', index=False)
    return frame


def participant_features(trials, aggregation):
    cols = [c for c in trials if c not in TRIAL_KEY]
    grouped = trials.groupby(['subject_id','group','condition'], observed=True)[cols]
    values = grouped.mean() if aggregation == 'mean' else grouped.median()
    wide = values.unstack('condition')
    wide.columns = [f'{condition}__{feature}' for feature, condition in wide.columns]
    return wide.reset_index()


def pipeline(estimator, k=None, pca=None):
    steps = [('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
             ('variance', VarianceThreshold()), ('scale', StandardScaler())]
    if k is not None:
        steps.append(('select', SelectKBest(f_classif, k=k)))
    if pca is not None:
        steps.append(('pca', PCA(n_components=pca, svd_solver='full')))
    steps.append(('model', estimator))
    return Pipeline(steps)


def candidates(task):
    """Prespecified small candidate suite. Do not alter after examining outer scores."""
    def lr(c):
        return LogisticRegression(C=c, max_iter=4000, class_weight='balanced', random_state=SEED)
    if task == 'subject':
        return [
            ('baseline_mean_lr', 'base_mean', pipeline(lr(.1))),
            ('mean_select32_lr', 'ext_mean', pipeline(lr(.01), k=32)),
            ('median_select32_lr', 'ext_median', pipeline(lr(.01), k=32)),
            ('mean_select64_lr', 'ext_mean', pipeline(lr(.1), k=64)),
            ('median_select64_lr', 'ext_median', pipeline(lr(.1), k=64)),
            ('mean_pca6_lr', 'ext_mean', pipeline(lr(1), pca=6)),
            ('median_pca6_lr', 'ext_median', pipeline(lr(1), pca=6)),
            ('mean_pca6_lda', 'ext_mean', pipeline(LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), pca=6)),
            ('mean_select32_linear_svm', 'ext_mean', pipeline(SVC(C=.1, kernel='linear', class_weight='balanced'), k=32)),
            ('median_pca6_rbf_svm', 'ext_median', pipeline(SVC(C=1, gamma='scale', class_weight='balanced'), pca=6)),
        ]
    return [
        ('baseline_forest', 'base', pipeline(RandomForestClassifier(n_estimators=200, max_depth=8,
            min_samples_leaf=3, max_features='sqrt', class_weight='balanced', random_state=SEED, n_jobs=2))),
        ('base_select128_lr', 'base', pipeline(lr(.01), k=128)),
        ('window_select128_lr', 'ext', pipeline(lr(.01), k=128)),
        ('window_select256_lr', 'ext', pipeline(lr(.1), k=256)),
        ('window_select128_linear_svm', 'ext', pipeline(SVC(C=.01, kernel='linear', class_weight='balanced'), k=128)),
        ('window_select128_rbf_svm', 'ext', pipeline(SVC(C=1, gamma='scale', class_weight='balanced'), k=128)),
        ('base_select128_rbf_svm', 'base', pipeline(SVC(C=1, gamma='scale', class_weight='balanced'), k=128)),
        ('window_pca12_lda', 'ext', pipeline(LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), pca=12)),
        ('window_select256_extra_trees', 'ext', pipeline(ExtraTreesClassifier(n_estimators=200,
            max_depth=12, min_samples_leaf=3, class_weight='balanced', random_state=SEED,n_jobs=2), k=256)),
    ]


def build_tables(base, extended):
    bt = trial_table(base); et = trial_table(extended)
    sortcols = ['subject_id', 'trial']
    bt = bt.sort_values(sortcols).reset_index(drop=True)
    et = et.sort_values(sortcols).reset_index(drop=True)
    if not bt[TRIAL_KEY].equals(et[TRIAL_KEY]):
        raise ValueError('Baseline and extended features do not have identical trial identities.')
    tables = {'condition': {'base': bt, 'ext': et},
              'subject': {'base_mean': subject_table(bt),
                          'ext_mean': participant_features(et,'mean'),
                          'ext_median': participant_features(et,'median')}}
    for key, frame in tables['subject'].items():
        tables['subject'][key] = frame.sort_values('subject_id').reset_index(drop=True)
    return tables


def prepare(task, tables):
    result = {}
    for name, frame in tables.items():
        excluded = TRIAL_KEY if task == 'condition' else ['subject_id','group']
        cols = [c for c in frame if c not in excluded]
        result[name] = (frame, frame[cols].to_numpy(dtype=float), cols)
    return result


def inner_select(task, data, specs, subject_meta, train_ids, log_rows, outer_fold):
    """Compare candidates only within the current training participant set."""
    meta = subject_meta[subject_meta.subject_id.isin(train_ids)].reset_index(drop=True)
    nfold = min(3, int(meta.group.value_counts().min()))
    if nfold < 2:
        raise ValueError('Not enough participants per group for nested validation.')
    splitter = StratifiedKFold(n_splits=nfold,shuffle=True,random_state=SEED+101)
    inner = []
    for tr, va in splitter.split(meta,meta.group):
        ids_tr = set(meta.iloc[tr].subject_id); ids_va = set(meta.iloc[va].subject_id)
        assert ids_tr.isdisjoint(ids_va)
        inner.append((ids_tr,ids_va))
    scores = []
    target = 'group' if task=='subject' else 'condition'
    for name, version, estimator in specs:
        frame, X, _ = data[version]
        y = frame[target].to_numpy()
        values = []
        for inner_fold,(ids_tr,ids_va) in enumerate(inner,1):
            tr = frame.subject_id.isin(ids_tr).to_numpy(); va = frame.subject_id.isin(ids_va).to_numpy()
            model = clone(estimator).fit(X[tr],y[tr])
            predicted = model.predict(X[va])
            if task == 'condition':
                valid = frame.loc[va,['subject_id']].copy()
                valid['truth']=y[va];valid['prediction']=predicted
                score = np.mean([balanced_accuracy_score(d.truth,d.prediction) for _,d in valid.groupby('subject_id')])
            else:
                score = balanced_accuracy_score(y[va],predicted)
            values.append(score)
            log_rows.append({'task':task,'outer_fold':outer_fold,'candidate':name,'feature_set':version,
                             'inner_fold':inner_fold,'balanced_accuracy':float(score),
                             'training_subjects':','.join(sorted(ids_tr)),
                             'validation_subjects':','.join(sorted(ids_va))})
        scores.append(float(np.mean(values)))
    best = int(np.argmax(scores))  # Candidate order breaks exact ties deterministically.
    print(f'{task} fold {outer_fold}: {specs[best][0]}, inner score={scores[best]:.3f}',flush=True)
    return specs[best],scores[best]


def score_predictions(frame,target,labels):
    result={'balanced_accuracy':float(balanced_accuracy_score(frame[target],frame.prediction)),
            'macro_f1':float(f1_score(frame[target],frame.prediction,average='macro',zero_division=0)),
            'labels':list(labels),'confusion_matrix':confusion_matrix(frame[target],frame.prediction,labels=labels).tolist()}
    return result


def optimize(base,extended,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    tables=build_tables(base,extended)
    meta=tables['subject']['base_mean'][['subject_id','group']].copy()
    if set(meta.group)!= {'alcoholic','control'} or meta.group.value_counts().min()<5:
        raise ValueError('This experiment requires at least five participants in each group.')
    splitter=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    outer=list(splitter.split(meta,meta.group))
    logs=[];report={'subjects':len(meta),'trials':len(tables['condition']['base']),
                    'evaluation':'five outer participant folds, three inner participant folds',
                    'caveat':'Exploratory iteration on the same 20 participants; external test set untouched.'}
    assignments=[]
    for fold,(_,va) in enumerate(outer,1):
        assignments.extend({'subject_id':sid,'validation_fold':fold} for sid in meta.iloc[va].subject_id)
    pd.DataFrame(assignments).to_csv(out/'subject_folds.csv',index=False)
    for task in ['subject','condition']:
        data=prepare(task,tables[task]);specs=candidates(task)
        target='group' if task=='subject' else 'condition'
        all_oof=[];selected=[];fold_scores=[]
        for fold,(train_idx,valid_idx) in enumerate(outer,1):
            train_ids=set(meta.iloc[train_idx].subject_id);valid_ids=set(meta.iloc[valid_idx].subject_id)
            assert train_ids.isdisjoint(valid_ids)
            (name,version,estimator),inner_score=inner_select(task,data,specs,meta,train_ids,logs,fold)
            frame,X,cols=data[version];y=frame[target].to_numpy()
            tr=frame.subject_id.isin(train_ids).to_numpy();va=frame.subject_id.isin(valid_ids).to_numpy()
            fitted=clone(estimator).fit(X[tr],y[tr])
            ids=['subject_id','group'] if task=='subject' else TRIAL_KEY
            pred=frame.loc[va,ids].copy();pred['prediction']=fitted.predict(X[va]);pred['fold']=fold
            if task=='subject':
                if hasattr(fitted,'predict_proba'):
                    index=list(fitted.classes_).index('alcoholic')
                    pred['alcoholic_score']=fitted.predict_proba(X[va])[:,index]
                    pred['score_type']='uncalibrated_probability'
                else:
                    pred['alcoholic_score']=fitted.decision_function(X[va])*(-1 if fitted.classes_[1]!='alcoholic' else 1)
                    pred['score_type']='decision_margin'
            all_oof.append(pred)
            selected.append({'outer_fold':fold,'candidate':name,'feature_set':version,'inner_score':inner_score})
            # Do not select the next configuration using these outer-fold results.
            fold_scores.append({'fold':fold,'balanced_accuracy':float(balanced_accuracy_score(y[va],pred.prediction))})
        joined=pd.concat(all_oof,ignore_index=True)
        joined.to_csv(out/f'{task}_nested_oof_predictions.csv',index=False)
        labels=sorted(joined[target].unique())
        report[task]=score_predictions(joined,target,labels)
        if task=='subject':
            # Different selected models can emit probabilities or SVM margins.
            # Their cross-fold scales are not interchangeable: compute AUC within folds.
            aucs=[float(roc_auc_score(d.group.eq('alcoholic'),d.alcoholic_score))
                  for _,d in joined.groupby('fold')]
            report[task]['mean_fold_roc_auc']=float(np.mean(aucs))
            report[task]['fold_roc_auc']=aucs
            report[task]['pooled_roc_auc']=None
            report[task]['auc_note']='No pooled AUC: score scales can differ across selected models/folds.'
        report[task]['folds']=fold_scores;report[task]['selected_by_inner_cv']=selected
        if task=='condition':
            report[task]['by_group']={g:score_predictions(d,target,labels) for g,d in joined.groupby('group')}
        # Deployment selection uses ALL training participants in an inner CV only.
        (name,version,estimator),score=inner_select(task,data,specs,meta,set(meta.subject_id),logs,'final')
        frame,X,cols=data[version]
        fitted=clone(estimator).fit(X,frame[target])
        joblib.dump({'model':fitted,'columns':cols,'feature_set':version,'task':task,
                     'candidate':name,'training_subject_ids':meta.subject_id.tolist(),
                     'preprocessing':'fixed 256 Hz, order-4 1-40 Hz, no baseline, flat features missing'},
                    out/f'{task}_model_v2.joblib')
        report[task]['deployment_model']={'candidate':name,'feature_set':version,'inner_score':score,
                                         'note':'Refit on all subjects; no independent test score for this refit.'}
        print(f'{task} OUTER balanced accuracy: {report[task]["balanced_accuracy"]:.3f}',flush=True)
    pd.DataFrame(logs).to_csv(out/'inner_selection_scores.csv',index=False)
    (out/'nested_metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    return report


def predict_v2(base,extended,models,out,task='both'):
    """Provide both feature tables from the same new recordings; never load untrusted joblib."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    tables=build_tables(base,extended)
    tasks=['subject','condition'] if task=='both' else [task]
    for task in tasks:
        bundle=joblib.load(Path(models)/f'{task}_model_v2.joblib')
        frame=tables[task][bundle['feature_set']]
        seen=set(frame.subject_id)&set(bundle['training_subject_ids'])
        if seen:
            warnings.warn(f'{len(seen)} participants overlap training; predictions are not independent evaluation.')
        X=frame.reindex(columns=bundle['columns'])
        if (X.isna().mean(axis=1)>.10).any():
            raise ValueError('More than 10% features missing; check channel/condition coverage.')
        pred=frame[['subject_id','group'] if task=='subject' else TRIAL_KEY].copy()
        fitted=bundle['model'];pred['prediction']=fitted.predict(X.to_numpy(dtype=float))
        if hasattr(fitted,'predict_proba'):
            probs=fitted.predict_proba(X.to_numpy(dtype=float))
            for i,label in enumerate(fitted.classes_):pred[f'uncalibrated_probability_{label}']=probs[:,i]
        pred.to_csv(out/f'{task}_predictions_v2.csv',index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',required=True);p.add_argument('--baseline-features',required=True)
    p.add_argument('--out',default='eeg_v2_results')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    base=pd.read_csv(args.baseline_features)
    featurepath=out/'extended_channel_features.csv'
    def digest(path):
        h=hashlib.sha256()
        with open(path,'rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        return h.hexdigest()
    identity={'raw_sha256':digest(args.csv),'baseline_features_sha256':digest(args.baseline_features),
              'extraction_version':2}
    manifest=out/'feature_cache_identity.json'
    reusable=featurepath.exists() and manifest.exists() and json.loads(manifest.read_text())==identity
    extended=pd.read_csv(featurepath) if reusable else extended_features(args.csv,out)
    manifest.write_text(json.dumps(identity,indent=2))
    # Limit BLAS threads to avoid oversubscription and very slow small-sample fits.
    with threadpool_limits(limits=2):optimize(base,extended,out)


if __name__=='__main__':main()
