# Original EEG project: spatial and temporal experiments

Source data unchanged: 20 participants, 600 trials. Every model is selected in three inner participant folds and evaluated in five outer participant folds, repeated with seeds 183, 184 and 185. The same participant IDs are reused across seeds; these are not 60 independent people.

## Results

| Task | Candidate family | Original split | Mean across three splits | Split range |
|---|---|---:|---:|---:|
| condition | combined_selection | 59.17% | 57.78% | 56.83–59.17% |
| condition | spatial_only | 40.17% | 40.89% | 40.17–41.67% |
| condition | temporal_contrasts | 51.17% | 52.56% | 51.17–54.17% |
| condition | v2_baseline | 59.17% | 57.56% | 56.33–59.17% |
| subject | combined_selection | 80.00% | 73.33% | 70.00–80.00% |
| subject | spatial_only | 80.00% | 85.00% | 80.00–90.00% |
| subject | temporal_contrasts | 50.00% | 56.67% | 50.00–70.00% |
| subject | v2_baseline | 80.00% | 76.67% | 70.00–80.00% |

Baseline is the exact v2 nested selection procedure, rerun on the same inputs. Combined selection adds the new candidates to that procedure; each outer fold can retain a v2 model if its inner validation score is stronger. Temporal-only and spatial-only suites are ablations, not models selected based on the outer scores.

## Change versus v2 across repeated splits

| Task | Family | Mean change | Participant-clustered bootstrap interval |
|---|---|---:|---:|
| subject | temporal_contrasts | -20.00 points | -38.33 to -1.67 points |
| subject | spatial_only | +8.33 points | -5.00 to +23.33 points |
| subject | combined_selection | -3.33 points | -10.00 to +3.33 points |
| condition | temporal_contrasts | -5.00 points | -9.22 to -0.56 points |
| condition | spatial_only | -16.67 points | -21.56 to -11.61 points |
| condition | combined_selection | +0.22 points | -0.50 to +0.94 points |

Intervals average results within each participant before resampling participants. They are exploratory, do not include the uncertainty of retraining on new cohorts, and are not corrected for multiple candidate families. Model development has repeatedly examined these same 20 people. Higher point accuracy does not establish independent generalization.

## Features and timing

The v2 features retain eight fixed 125-ms means/stds per channel. New temporal features compare later recording bins with the first bin and compare the recording's late and early halves. Neither the notebook nor this export establishes stimulus onset; these are recording-relative contrasts, not verified ERP amplitudes or pre-stimulus baseline correction.

Spatial features use within-trial OAS shrinkage covariance and correlation, then a symmetric matrix logarithm. Training-fold scaling, PCA/feature selection and classification follow extraction. The 61 consistently named scalp channels are used for spatial features; X, Y and nd are excluded from this branch only. All original trials remain, including flat-electrode trials; shrinkage stabilizes their covariance and QC records flat-channel counts. Original v2 features still use its full channel schema and flat-feature handling.

## Deployment candidates

- subject: {'candidate': 'median_select64_lr', 'feature_set': 'ext_median', 'inner_score': 0.7916666666666666}. Refit on all 20 participants; no independent test score for this refit.
- condition: {'candidate': 'window_select256_lr', 'feature_set': 'ext', 'inner_score': 0.6034391534391533}. Refit on all 20 participants; no independent test score for this refit.

The package contains candidate model artifacts and prediction code. Their probabilities remain uncalibrated. Existing v2 files are preserved separately and are not overwritten by these experiments.

## Verification

24 task/family/seed results were independently recomputed. 1584 inner-fold records were checked for training/validation overlap and outer-held-out contamination. Candidate predictions and splits are complete.


## Experimental spatial model and runtime

The separate subject_spatial_experimental.joblib artifact uses cov_mean_pca6_lr, selected within the spatial family by all-participant inner validation. Its repeated-split family mean was 85%, versus 76.67% for v2. The paired interval for the change is −5.00 to +23.33 percentage points: a promising signal, not established improvement.

The nested comparison took 307.2 seconds in this runtime, excluding initial extraction and final refits. No runtime speedup claim is supported.
## Fresh rerun

Regenerated original and temporal features retained all 38,400 electrode recordings and all 600 trials. All 7,440 task/family/seed predictions and outer folds match the earlier run exactly. The initial temporal CSV was incomplete and training stopped at its identity guard; rebuilding resolved it. The extractor now verifies the saved row count and recording identities before replacing the feature file. Three refitted models passed prediction checks. Comparison runtime was 307.2 seconds, excluding extraction and final refits.
