# Original EEG spatial and temporal project

This project keeps the original data unchanged and compares the existing v2 procedure with new recording-relative temporal contrasts, electrode covariance/correlation features, and combined candidate selection. See RESULTS.md for measured results and uncertainty.

## Run

Python is preferred for the existing NumPy/SciPy/scikit-learn pipeline. Use Python 3.11 or later.

```bash
pip install -r requirements.txt
python prepare_and_run.py --csv /path/large_train_eeg_week3_results.zip --out results
```

The ZIP must contain the original cleaned CSV. A compatible original CSV also works. The source schema and cleaning behavior are inherited from eeg_pipeline.py. The helper regenerates original and extended features, validates cache identity and runs all comparisons. Allow several GB of RAM; intermediate feature files are intentionally omitted from this download. The notebook provides the same workflow in Colab. It uses a local upload and does not require Drive write access.

To reuse previously generated v2 features:

```bash
python improve_original.py --csv /path/source.zip --baseline-features /path/channel_features.csv --extended-features /path/extended_channel_features.csv --out results
```

The extended feature directory must contain feature_cache_identity.json matching the exact source and baseline hashes. Omit --extended-features to regenerate them.

## Prediction

```bash
python predict_original.py --csv /path/new_data.csv --models results --out predictions
python predict_original.py --csv /path/new_data.csv --models results --out spatial_predictions --subject-model subject_spatial_experimental.joblib
```

The default models use final inner validation over all participants. The spatial-only participant model is a separate exploratory option: its family had a better repeated-split mean, but that observation is not independent confirmation. No original v2 file is overwritten. Model artifacts are all-data refits, not independently tested models. Only load trusted joblib files. Participant predictions require coverage of the original three conditions and electrode schema; arbitrary EEG formats require an adapter. Predictions on training people are not an accuracy evaluation.

uncalibrated_probability is a classifier probability output whose reliability has not been calibrated. A score of 0.8 does not establish an 80% real-world probability. An SVM decision_margin is a ranking score, not a probability.

## Evaluation and checks

Every candidate's imputation, scaling, feature selection/PCA and classifier are fitted inside training participants. Model choice uses three inner folds; five outer folds provide predictions. Seeds 183/184/185 reuse the same 20 people. They measure split sensitivity, not independent replication. Participant classification is alcoholic/control; trial classification is S1 obj/S2 match/S2 nomatch.

The spatial branch uses 61 scalp channels, 256 samples, assumed 256 Hz inherited from the existing pipeline, a 1–40 Hz filter, OAS shrinkage and log covariance/correlation. X, Y and nd are excluded from that branch. All 600 trials remain. Flat-channel counts are recorded. Stimulus onset is unverified; temporal contrasts refer to recording bins, not verified ERP windows. Confirm acquisition metadata before changing this assumption.

```bash
python -m unittest test_original.py
python audit_report.py
```

The cached-feature identity test runs when the original local feature files are available. The report audit uses the included small prediction and split files. RESULTS.md and results/*.csv include confusion matrices (metrics.json), classification metrics, repeated-split summaries, inner selection logs, and paired participant bootstrap intervals. Runtime in metrics.json covers model comparison and excludes initial feature extraction/final refits. run.log records progress; no speed improvement over v2 is established.

OAS reference: https://arxiv.org/abs/0907.4698
Scikit-learn implementation: https://scikit-learn.org/stable/modules/generated/sklearn.covariance.oas.html
