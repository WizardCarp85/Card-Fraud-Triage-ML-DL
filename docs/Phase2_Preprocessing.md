# Phase 2 Shared 29-Feature Preprocessing (Issue #28)

Implementation: `data/preprocessing.py`. Run from the repo root:

```
python -m data.preprocessing
```

Inputs: `data/creditcard.csv` (SHA-256 verified by the split manifest, issue #25) and `data/split_manifest.json`.

## Feature contract
- 29 ordered features: `V1`..`V28`, then `Amount_log = log1p(Amount)`.
- `Class` is never a feature. `Time` is not added.
- Row ID = 0-based raw row index of `creditcard.csv`, as in the manifest. Labels and row IDs are returned beside `X`, not inside it.

## Scaling
- `RobustScaler` fitted on the manifest's `train_legit` rows only (198,242 rows, no fraud).
- The frozen scaler is applied with `transform` only to train, validation and test.
- The rebuilt scaler matches the Phase 1 artifact `results/scaler_robust.joblib` exactly (max abs difference of `center_` and `scale_` is 0.0, tolerance 1e-6 rel / 1e-9 abs).

## Saved artifacts (`data/artifacts/`)
| File | Purpose |
|---|---|
| `scaler_robust_phase2.joblib` | Fitted scaler |
| `feature_metadata.json` | Feature names, scaler parameters and hash, tolerances, library versions |
| `preprocessing_verification_report.json` | Per-check pass/fail, per-split shapes, fraud counts, array hashes |
| `features_phase2.npz` | Scaled matrices, labels, row IDs (git-ignored, rebuildable) |

## Verification performed
Shapes, float64 dtype, finite values, row-ID membership equal to the manifest, disjoint splits, fraud counts equal to the manifest, no fraud in `train_legit`, scaler parameters finite and positive.

## Reproducibility and tolerances
- Same machine and library versions: output is bit-identical. Two consecutive runs produced identical verification reports (including SHA-256 of every matrix).
- Across machines or library patch versions: `rtol=1e-9`, `atol=1e-12` for features and scaler parameters.
- Versions used: see `environment` in `feature_metadata.json` (Python 3.9.6, numpy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, joblib 1.5.3).

## Known limitations
- Depends on the unmerged split manifest PR (#25 / PR #46).
- The Phase 1 scaler was compared with a different serialisation environment, so comparison uses statistics, not file bytes.
- The loader (#31) and denoising (#33) are separate issues.
