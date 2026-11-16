# Validation threshold manifest v1.0

Run from the repository root with the environment used for the shared evaluation utilities.

```sh
python -m scripts.freeze_gmm_thresholds
python -m scripts.freeze_gmm_thresholds --check
```

The first command writes results/phase2/gmm/threshold_manifest.json. The second recomputes calibration and checks the saved manifest for exact equality. Both use the committed validation score and label files. The external raw dataset and model deserialization are unnecessary.

The freeze checks byte hashes of the GMM, its Phase 1 scaler, split manifest, validation scores and labels against the #30 provenance. It also verifies exact validation membership against data/splits/val_indices.csv. The score loader validates model identity, split, finite values, unique stable IDs and the explicit higher-is-more-suspicious convention. Labels join by stable identity. Score direction is supported by the baseline's negative-log-likelihood implementation and artifact provenance.

The manifest records these shared fields for each model.

| Field | Meaning |
|---|---|
| schema_version | 1.0 |
| model | gmm or the AE export's model identity |
| calibration_split | validation |
| score_direction | higher_is_more_suspicious |
| score_transform | Formula used to produce the score |
| selection_rule | Maximum recall within the validation FPR budget, choosing the highest threshold when recall ties |
| validation_rows / validation_fraud_rows | Calibration sample counts |
| thresholds | Policies at budgets 0.001, 0.01 and 0.05 |
| provenance | Paths and SHA-256 identities of inputs, model, preprocessing and calibration code, plus feature order |

Each threshold record contains fpr_budget, threshold, validation_recall, validation_fpr, true_positives, false_positives, status, decision and reason. A successful policy has status ok, decision score_gte_threshold and a null reason. Apply score >= threshold without test-driven adjustment.

A no_alerts_within_budget policy has decision no_alerts, a null threshold, zero recall/FPR and its status as reason. An undefined class distribution has decision undefined, null threshold and null undefined metrics, with reason undefined_no_positive or undefined_no_negative. Consumers must branch on status/decision before comparing scores. JSON contains no NaN or Infinity. Every budget has an explicit record.

The reusable src.threshold_calibration.calibrate_validation function accepts any shared-contract model export and returns the same calibration fields. AE calibration should add its checkpoint and preprocessing identities under provenance and save a separate manifest. The #46 reference in issue #37 resolves to an earlier merged split PR. This schema supplies a concrete format for the separately owned AE task without altering the AE model-selection manifest.

## Frozen GMM result

Calibration uses 42,558 validation rows, including 55 frauds.

| FPR budget | Threshold | Validation recall | Actual validation FPR | TP | FP |
|---|---:|---:|---:|---:|---:|
| 0.1% | 143.65081901156682 | 0.763636 | 0.000211750 | 42 | 9 |
| 1% | 67.95949186438423 | 0.800000 | 0.002376303 | 44 | 101 |
| 5% | 39.18202235082998 | 0.872727 | 0.032468296 | 48 | 1380 |

The actual FPR may be below its budget because the existing selection rule retains the highest threshold achieving maximum feasible recall. No test score files, labels or metrics enter calibration. Prior Phase 1 holdout exposure remains documented in the split contract. The saved provenance includes the baseline report's hash for traceability, while only its artifact identities are consumed.
