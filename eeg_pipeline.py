"""Week 3 EEG preprocessing + two exploratory, subject-separated models.

Run: python eeg_pipeline.py train --csv large_train_eeg_messy.csv --out results
Predict: python eeg_pipeline.py predict --csv new.csv --models results --out predictions
Python >=3.10; numpy, pandas, scipy, scikit-learn >=1.3, joblib.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import warnings
import zipfile

import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

KEY = ['subject_id', 'group', 'condition', 'trial', 'channel']
TRIAL_KEY = KEY[:4]
LABELS = ['control', 'alcoholic']
BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
         'beta': (13, 30), 'gamma_30_40': (30, 40)}
SEED = 183


def read_csv(path):
    """Read plain CSV, or the cleaned CSV in our previous export ZIP.

    Category strings reduce RAM. Identifiers/source filenames never become features.
    Loading the full CSV requires several GB of RAM; Colab's 12 GB runtime is suitable.
    """
    types = {k: 'category' for k in ['subject_id', 'group', 'condition', 'channel', 'source_file']}
    path = Path(path)
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if Path(n).name == 'large_train_eeg_clean.csv']
            if len(names) != 1:
                raise ValueError('ZIP must contain exactly one large_train_eeg_clean.csv.')
            with archive.open(names[0]) as stream:
                return pd.read_csv(stream, dtype=types)
    return pd.read_csv(path, dtype=types)


def clean_data(raw):
    """Reuse the notebook's parse-error, missing-voltage and first-duplicate rules.

    The extra validation checks prevent silently corrupt epochs. For unknown participants,
    group may be absent; supply the placeholder 'unknown' (never a model input).
    """
    raw = raw.copy()
    if 'group' not in raw:
        raw['group'] = 'unknown'
    if 'condition' not in raw:
        raw['condition'] = 'unknown'
    required = KEY + ['sample', 'voltage_uv']
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    valid_parse = raw['parse_error'].isna() if 'parse_error' in raw else pd.Series(True, index=raw.index)
    clean = raw.loc[valid_parse & raw['voltage_uv'].notna()].copy()
    clean = clean.drop_duplicates(subset=KEY + ['sample']).reset_index(drop=True)
    if clean[KEY + ['sample']].isna().any().any():
        raise ValueError('Missing subject, group, condition, trial, channel, or sample identifiers.')
    for name in ['sample', 'trial', 'voltage_uv']:
        clean[name] = pd.to_numeric(clean[name], errors='raise')
        if not np.isfinite(clean[name]).all():
            raise ValueError(f'Nonfinite {name} values.')
    for name in ['sample', 'trial']:
        if not (clean[name] == np.floor(clean[name])).all():
            raise ValueError(f'{name} must contain integer indices.')
    # group/condition are part of the original duplicate key. Check ambiguous recordings too.
    if (clean.groupby('subject_id', observed=True)['group'].nunique() > 1).any():
        raise ValueError('One subject has multiple group labels.')
    if (clean.groupby(['subject_id', 'trial'], observed=True)['condition'].nunique() > 1).any():
        raise ValueError('One subject/trial has multiple conditions; verify trial identifiers.')
    print(f'Cleaning: {len(raw):,} -> {len(clean):,} rows', flush=True)
    return clean


def signal_features(values, fs):
    """Time-domain, Hjorth and relative spectral patterns from one filtered epoch.

    One-second epochs provide coarse (~1 Hz) spectral estimates. Full-epoch Welch
    with detrend=False avoids a second implicit demeaning; no baseline correction.
    Bands are half-open except 30-40 Hz, so bins are not counted twice.
    """
    v = np.asarray(values, dtype=float)
    d1, d2 = np.diff(v), np.diff(v, n=2)
    variance = np.var(v)
    eps = np.finfo(float).eps
    mobility = np.sqrt(np.var(d1) / max(variance, eps))
    complexity = np.sqrt(np.var(d2) / max(np.var(d1), eps)) / max(mobility, eps)
    features = {'mean_uv': v.mean(), 'std_uv': v.std(ddof=1),
                'peak_to_peak_uv': np.ptp(v), 'rms_uv': np.sqrt(np.mean(v*v)),
                'line_length_uv': np.mean(np.abs(d1)), 'hjorth_mobility': mobility,
                'hjorth_complexity': complexity}
    freq, psd = welch(v, fs=fs, nperseg=len(v), noverlap=0, detrend=False)
    df = freq[1] - freq[0]
    total = np.sum(psd[(freq >= 1) & (freq <= 40)]) * df
    features['log_total_power'] = np.log10(max(total, eps))
    for name, (low, high) in BANDS.items():
        mask = (freq >= low) & ((freq <= high) if high == 40 else (freq < high))
        power = np.sum(psd[mask]) * df
        features[f'relative_{name}'] = power / max(total, eps)
    return features


def extract_features(clean, out=None, fs=256, expected_samples=256,
                     sample_start=0, exclude_channels=()):
    """Filter ALL condition/channel epochs independently using the Week 3 settings.

    Incomplete/noncontiguous epochs are logged and skipped rather than treating gaps
    as uniformly sampled EEG. No voltage-threshold rejection, ICA, rereferencing,
    resampling, or unverified baseline correction is added.
    """
    if fs <= 80:
        raise ValueError('Sampling rate must exceed 80 Hz for the 40 Hz cutoff.')
    sos = butter(4, [1, 40], btype='bandpass', fs=fs, output='sos')
    records, quality = [], []
    for i, (key, epoch) in enumerate(clean.groupby(KEY, observed=True, sort=True)):
        meta = dict(zip(KEY, key))
        if str(meta['channel']) in exclude_channels:
            continue
        epoch = epoch.sort_values('sample')
        if len(epoch) != expected_samples or not np.array_equal(
            epoch['sample'].to_numpy(), np.arange(sample_start, sample_start+expected_samples)
        ):
            quality.append({**meta, 'n_samples': len(epoch), 'status': 'skipped_incomplete_or_gapped'})
            continue
        values = epoch['voltage_uv'].to_numpy(dtype=float)
        filtered = sosfiltfilt(sos, values)
        records.append({**meta, **signal_features(filtered, fs)})
        quality.append({**meta, 'n_samples': len(epoch), 'status': 'used',
                        'raw_max_abs_uv': float(np.max(np.abs(values)))})
        if i and i % 5000 == 0:
            print(f'Processed {i:,} channel epochs', flush=True)
    if not records:
        raise ValueError('No complete epochs. Check sample_start/expected_samples and excluded channels.')
    features = pd.DataFrame(records)
    qc = pd.DataFrame(quality)
    if out is not None:
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        features.to_csv(out / 'channel_features.csv', index=False)
        qc.to_csv(out / 'epoch_quality.csv', index=False)
    print(f'Features: {len(features):,} epochs; {len(qc)-len(features):,} skipped', flush=True)
    return features, qc


def trial_table(features):
    """One trial row with electrode__feature columns; labels stay separate."""
    numeric = [c for c in features if c not in KEY]
    wide = features.pivot(index=TRIAL_KEY, columns='channel', values=numeric)
    wide.columns = [f'{channel}__{feature}' for feature, channel in wide.columns]
    return wide.reset_index()


def subject_table(trials):
    """One participant row, averaging trials equally within each condition.

    Condition labels organize the aggregation for the WHO model but group labels,
    IDs, filenames, trial numbers and condition names are never numeric inputs.
    """
    cols = [c for c in trials if c not in TRIAL_KEY]
    means = trials.groupby(['subject_id', 'group', 'condition'], observed=True)[cols].mean()
    wide = means.unstack('condition')
    wide.columns = [f'{condition}__{feature}' for feature, condition in wide.columns]
    return wide.reset_index()


def make_model(task):
    if task == 'subject':
        estimator = LogisticRegression(C=0.1, max_iter=5000, class_weight='balanced', random_state=SEED)
    else:
        estimator = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=3,
                                           max_features='sqrt', class_weight='balanced',
                                           random_state=SEED, n_jobs=-1)
    return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                         StandardScaler(), estimator)


def metrics(y, prediction, labels):
    return {'balanced_accuracy': float(balanced_accuracy_score(y, prediction)),
            'macro_f1': float(f1_score(y, prediction, labels=labels, average='macro', zero_division=0)),
            'confusion_labels': list(labels),
            'confusion_matrix': confusion_matrix(y, prediction, labels=labels).tolist()}


def train(features, out, folds=5):
    """Evaluate fixed baselines on unseen participants, then fit deployment copies.

    All imputation/scaling is fitted INSIDE each fold. No tuning or model selection
    uses these evaluation folds. Condition group comparisons use held-out predictions.
    """
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    trials = trial_table(features)
    subjects = subject_table(trials)
    if set(subjects['group']) != set(LABELS):
        raise ValueError("Training group labels must be exactly 'control' and 'alcoholic'.")
    counts = subjects['group'].value_counts()
    if folds < 2 or folds > counts.min():
        raise ValueError(f'folds must be between 2 and {counts.min()} for these participants.')
    condition_labels = sorted(trials['condition'].unique())
    if len(condition_labels) < 2:
        raise ValueError('Condition model needs at least two stimulus conditions.')
    subject_cols = [c for c in subjects if c not in ['subject_id', 'group']]
    condition_cols = [c for c in trials if c not in TRIAL_KEY]
    Xs = subjects[subject_cols]; ys = subjects['group']
    Xt = trials[condition_cols]; yt = trials['condition']
    subject_oof, condition_oof, fold_scores, split_rows = [], [], [], []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(Xs, ys), 1):
        train_ids = set(subjects.iloc[train_idx]['subject_id'])
        valid_ids = set(subjects.iloc[valid_idx]['subject_id'])
        assert train_ids.isdisjoint(valid_ids)
        tr = trials['subject_id'].isin(train_ids); va = trials['subject_id'].isin(valid_ids)
        if set(yt[tr]) != set(condition_labels) or set(yt[va]) != set(condition_labels):
            raise ValueError('A fold is missing conditions; inspect coverage before modeling.')
        who = make_model('subject').fit(Xs.iloc[train_idx], ys.iloc[train_idx])
        how = make_model('condition').fit(Xt.loc[tr], yt.loc[tr])
        sp = subjects.iloc[valid_idx][['subject_id', 'group']].copy()
        sp['prediction'] = who.predict(Xs.iloc[valid_idx])
        pos = list(who.classes_).index('alcoholic')
        sp['alcoholic_probability'] = who.predict_proba(Xs.iloc[valid_idx])[:, pos]
        sp['fold'] = fold; subject_oof.append(sp)
        cp = trials.loc[va, TRIAL_KEY].copy()
        cp['prediction'] = how.predict(Xt.loc[va]); cp['fold'] = fold
        prob = how.predict_proba(Xt.loc[va])
        for j, label in enumerate(how.classes_):
            cp[f'probability_{label}'] = prob[:, j]
        condition_oof.append(cp)
        fold_scores.append({'fold': fold, 'subject': metrics(sp['group'], sp['prediction'], LABELS),
                            'condition': metrics(cp['condition'], cp['prediction'], condition_labels)})
        for sid in valid_ids:
            split_rows.append({'subject_id': sid, 'validation_fold': fold})
        print(f'Fold {fold}/{folds} complete (subjects separated)', flush=True)
    sp = pd.concat(subject_oof, ignore_index=True)
    cp = pd.concat(condition_oof, ignore_index=True)
    report = {'n_subjects': len(subjects), 'n_trials': len(trials), 'folds': fold_scores,
              'subject_oof': metrics(sp['group'], sp['prediction'], LABELS),
              'condition_oof': metrics(cp['condition'], cp['prediction'], condition_labels)}
    report['subject_oof']['roc_auc'] = float(roc_auc_score(sp['group'].eq('alcoholic'), sp['alcoholic_probability']))
    report['condition_oof_by_group'] = {
        str(g): metrics(d['condition'], d['prediction'], condition_labels)
        for g, d in cp.groupby('group')}
    # Subject-weighted accuracy prevents participants with more trials dominating comparisons.
    cp['correct'] = cp['condition'].eq(cp['prediction']).astype(int)
    participant_accuracy = cp.groupby(['subject_id', 'group'], observed=True)['correct'].mean().reset_index()
    participant_accuracy.to_csv(out / 'condition_accuracy_by_subject.csv', index=False)
    report['condition_mean_subject_accuracy_by_group'] = participant_accuracy.groupby('group')['correct'].mean().to_dict()
    sp.to_csv(out / 'subject_oof_predictions.csv', index=False)
    cp.to_csv(out / 'condition_oof_predictions.csv', index=False)
    pd.DataFrame(split_rows).to_csv(out / 'subject_folds.csv', index=False)
    # Final models fit all training subjects: their training score is NOT evaluation.
    for task, X, y, cols in [('subject', Xs, ys, subject_cols), ('condition', Xt, yt, condition_cols)]:
        fitted = make_model(task).fit(X, y)
        joblib.dump({'model': fitted, 'columns': cols, 'training_subject_ids': subjects['subject_id'].tolist()},
                    out / f'{task}_model.joblib')
    (out / 'metrics.json').write_text(json.dumps(report, indent=2))
    pattern_comparison(features, out)
    print(json.dumps({k: report[k] for k in ['n_subjects', 'n_trials', 'subject_oof', 'condition_oof']}, indent=2))
    return report


def pattern_comparison(features, out):
    """Descriptive electrode patterns at participant level; no causal/significance claims.

    Each subject contributes one mean per channel/condition. Export recognition
    contrasts when both match/nonmatch exist. Contrasts compare the two conditions
    within each subject before comparing groups; missing pairs are omitted.
    """
    cols = [c for c in features if c not in KEY]
    means = features.groupby(['subject_id', 'group', 'condition', 'channel'], observed=True)[cols].mean().reset_index()
    means.to_csv(Path(out) / 'subject_condition_channel_patterns.csv', index=False)
    summary = means.groupby(['group', 'condition', 'channel'], observed=True)[cols].agg(['mean', 'std', 'count'])
    summary.columns = ['__'.join(c) for c in summary.columns]
    summary.to_csv(Path(out) / 'group_condition_channel_patterns.csv')
    conditions = set(means['condition'])
    if {'S2 match', 'S2 nomatch'} <= conditions:
        a = means[means['condition'].eq('S2 match')].set_index(['subject_id', 'group', 'channel'])[cols]
        b = means[means['condition'].eq('S2 nomatch')].set_index(['subject_id', 'group', 'channel'])[cols]
        delta = (a - b).dropna(how='all')
        delta.to_csv(Path(out) / 'recognition_match_minus_nonmatch_by_subject.csv')
        grouped = delta.groupby(['group', 'channel']).mean()
        difference = grouped.xs('alcoholic', level='group') - grouped.xs('control', level='group')
        difference.to_csv(Path(out) / 'recognition_group_difference.csv')


def predict(features, models, out, task='both'):
    """Generate trial-condition and subject-group predictions for new recordings.

    WHO requires the known experimental condition labels to build condition-specific
    summaries. HOW excludes those labels from inputs. Do not infer WHO aggregation
    from predicted conditions without a separately evaluated pipeline.
    """
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    trials = trial_table(features)
    jobs = []
    if task in ['subject', 'both']:
        jobs.append(('subject', subject_table(trials), ['subject_id']))
    if task in ['condition', 'both']:
        jobs.append(('condition', trials, TRIAL_KEY))
    for task, table, ids in jobs:
        bundle = joblib.load(Path(models) / f'{task}_model.joblib')
        seen = set(table['subject_id']) & set(bundle['training_subject_ids'])
        if seen:
            warnings.warn(f'{task}: {len(seen)} participants were used in training; predictions are not independent evaluation.')
        X = table.reindex(columns=bundle['columns'])
        if (X.isna().mean(axis=1) > 0.10).any():
            raise ValueError(f'{task}: more than 10% expected features missing. Check channels and conditions.')
        prediction = table[ids].copy()
        fitted = bundle['model']
        prediction['prediction'] = fitted.predict(X)
        probabilities = fitted.predict_proba(X)
        for j, label in enumerate(fitted.classes_):
            prediction[f'probability_{label}'] = probabilities[:, j]
        prediction.to_csv(out / f'{task}_predictions.csv', index=False)
    print(f'Predictions saved in {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'predict', 'features'])
    parser.add_argument('--csv', required=True, help='Plain CSV or previous cleaned-data ZIP')
    parser.add_argument('--out', default='eeg_results')
    parser.add_argument('--models', default='eeg_results')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--task', choices=['both', 'subject', 'condition'], default='both', help='Prediction task')
    parser.add_argument('--sample-start', type=int, default=0)
    parser.add_argument('--exclude-channels', nargs='*', default=[], help='Only exclude verified non-EEG channels')
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    config = {'fs': 256, 'samples': 256, 'sample_start': args.sample_start,
              'exclude_channels': args.exclude_channels, 'filter': 'Butterworth order 4, 1-40 Hz, sosfiltfilt',
              'baseline_correction': False, 'automatic_artifact_rejection': False}
    if args.command == 'predict':
        config = json.loads((Path(args.models) / 'preprocessing.json').read_text())
    raw = read_csv(args.csv)
    clean = clean_data(raw)
    del raw
    features, _ = extract_features(clean, out, fs=config['fs'], expected_samples=config['samples'],
                                   sample_start=config['sample_start'], exclude_channels=config['exclude_channels'])
    del clean
    (out / 'preprocessing.json').write_text(json.dumps(config, indent=2))
    if args.command == 'train':
        train(features, out, args.folds)
    elif args.command == 'predict':
        predict(features, args.models, out, args.task)


if __name__ == '__main__':
    main()
