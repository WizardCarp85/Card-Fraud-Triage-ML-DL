# Phase 2 GMM Baseline Validation Reproduction (Issue #30)

**Owner:** Rudra Pratap Singh Choudhary (`@Rudra-choudhary`) — Probabilistic Modeling  
**Milestone:** Phase 2 Probabilistic Baseline & Validation Contract  
**Dependencies:** Issue #25 (Dataset split manifest), Issue #26 (Anomaly score schema), Issue #28 (29-feature preprocessing)  
**Implementation:** `data/reproduce_gmm_baseline.py`  
**Test Suite:** `tests/test_gmm_baseline.py`  

---

## 1. Executive Summary & Objective

The objective of Issue #30 is to produce a trustworthy, reproducible Gaussian Mixture Model (GMM) baseline score file for the Phase 2 validation split adhering to the frozen chronological split manifest (#25), preprocessing contract (#28), and standardized score schema (#26).

All acceptance criteria have been empirically verified and validated by automated tests:
- **Feature & Row Concordance:** Uses the identical 29-feature order (`V1..V28`, `Amount_log`) and exact 42,558 validation raw row IDs (`199324` to `242066`) as the Autoencoder pipeline.
- **Provenance & Parameter Auditing:** Serialized model and scaler parameters, convergence lower bounds, and SHA-256 hashes are recorded in `data/scores/gmm_baseline_provenance.json`.
- **Score Sanity & Direction:** Scores are finite, numeric negative log-likelihood values ($-\ln p(x)$) where larger values indicate higher fraud suspicion (`higher_is_more_suspicious`).
- **Metric Regeneration:** Validation AP (`0.650975`) and ROC-AUC (`0.947387`) reproduce Phase 1 outputs to machine precision ($10^{-12}$ difference on sorted score distributions).

---

## 2. Artifact Provenance & Cryptographic Verification

The Phase 1 model and scaler artifacts were located in `results/` and audited against the Phase 2 preprocessing contract:

| Artifact | Path | SHA-256 Checksum | Description / Configuration |
|---|---|---|---|
| **Phase 1 Scaler** | `results/scaler_robust.joblib` | `8c1c5db61d15c7e39a3f29a03c005fbc358fe2331575ca2484a0d9237c151475` | `RobustScaler(quantile_range=(25.0, 75.0))` fit strictly on legitimate training rows (198,242 rows) |
| **Phase 2 Scaler** | `data/artifacts/scaler_robust_phase2.joblib` | `db0d17f42b9bbe08a975d59f38c486f78e7208356da0ccae985e43fdc7b97762` | Bit-identical scaling parameters (`max_abs_center_diff = 0.0`, `max_abs_scale_diff = 0.0`) |
| **Phase 1 GMM** | `results/gmm_model.joblib` | `df9c7c2b3391d4e417a8c3d6fcce5167f6b957e8cebfa780d6f9bb76579893d1` | `GaussianMixture(n_components=14, covariance_type='full', reg_covar=1e-6, max_iter=200, n_init=3, random_state=42)` |

### Model Parameters & Convergence State
- **Components ($K$):** 14
- **Covariance Structure:** Full ($14 \times 29 \times 29$ parameters)
- **Regularization (`reg_covar`):** $1 \times 10^{-6}$ (ensures strictly positive-definite covariance matrices)
- **Convergence Status:** `converged_ = True` after 41 iterations (tolerance $10^{-3}$)
- **Log-Likelihood Lower Bound:** $30.986444499043042$

---

## 3. Preprocessing Contract & Feature Alignment

The model inputs adhere strictly to the shared 29-feature contract defined in Issue #28:
1. Continuous features: $V_1, V_2, \dots, V_{28}$ (PCA components from ULB dataset).
2. Transformed feature: $\text{Amount\_log} = \ln(1 + \text{Amount})$.
3. Excluded columns: `Class` (strictly segregated into separate label evaluation files) and `Time` (used only for chronological partitioning, never fed into density estimators).
4. Normalization: Center and scale parameters derived solely from legitimate training samples ($\text{Class} = 0$).

---

## 4. Empirical Reproduction & Phase 1 Comparison

Evaluating the reloaded GMM model against the Phase 2 validation split manifests the following performance:

| Metric | Phase 1 Documented | Phase 2 Reproduced | Difference ($\Delta$) | Status |
|---|---|---|---|---|
| **Validation Observations** | 42,558 | 42,558 | 0 | Exact match |
| **Validation Frauds** | 55 | 55 | 0 | Exact match |
| **Average Precision (AP / PR-AUC)** | 0.650975 | 0.650975 | 0.000000 | Bit-identical |
| **ROC-AUC** | 0.947387 | 0.947387 | 0.000000 | Bit-identical |
| **Recall @ FPR $\le 0.001$** | 76.36% (TP: 42, FP: 9) | 76.36% (TP: 42, FP: 9) | 0.00% | Exact match |
| **Recall @ FPR $\le 0.01$** | 80.00% (TP: 44, FP: 101) | 80.00% (TP: 44, FP: 101) | 0.00% | Exact match |
| **Recall @ FPR $\le 0.05$** | 87.27% (TP: 48, FP: 1,380) | 87.27% (TP: 48, FP: 1,380) | 0.00% | Exact match |
| **Max Absolute Score Difference** | — | $1.037 \times 10^{-12}$ | — | Machine precision ($\epsilon$) |

### Explanation of Row Ordering and Identifier Evolution
Users inspecting Phase 1 files (`results/tables/gmm_scores_val.csv`) alongside Phase 2 files (`data/scores/gmm_validation.csv`) will observe two structural differences:
1. **Identifier Schema:** In Phase 1, `original_index` denoted sequential 0-based positional indices (198,608 to 241,165) after deduplication. Phase 2 adopts immutable `row_id` tracking the original line positions in `creditcard.csv` (199,324 to 242,066) defined in the frozen split manifest (#25).
2. **Timestamp Collisions:** Over 84% of transactions share identical integer second timestamps (`Time`). Phase 1 used pandas' default quicksort (`df.sort_values('Time')`), an unstable sorting algorithm whose pivot choices vary by CPU architecture and library build. Phase 2 freezes stable raw row index ordering to eliminate cross-platform discrepancy.

Because the underlying validation transactions are identical, the sorted score distributions and ranking curves match to machine precision ($10^{-12}$).

---

## 5. Exported Score & Evaluation Artifacts

The scores are exported according to the standardized schema from Issue #26 (`docs/score_file_schema.md`):

```
data/scores/
├── gmm_validation.csv             # Exactly 4 columns: row_id,split,model,anomaly_score (42,558 rows)
├── validation_labels.csv          # Exactly 3 columns: row_id,split,Class (42,558 rows)
├── gmm_test.csv                   # Holdout test scores (42,560 rows)
├── test_labels.csv                # Holdout test labels (42,560 rows)
└── gmm_baseline_provenance.json   # Full environment, parameter, and hash provenance snapshot
```

All anomaly score files pass the strict parser rules in `evaluation.py`:
- No missing or whitespace-padded values.
- Unique `row_id` matching manifest split membership.
- Disjoint row IDs between validation and test.
- Verified `score_direction="higher_is_more_suspicious"`.

---

## 6. Usage & Verification

### Regenerating Scores & Tables
To re-run reproduction and update triage tables:
```bash
python -m data.reproduce_gmm_baseline
```

### Evaluating via `evaluation.py`
```python
from evaluation import evaluate_score_files

result = evaluate_score_files(
    "data/scores/gmm_validation.csv",
    "data/scores/validation_labels.csv",
    "data/scores/gmm_test.csv",
    "data/scores/test_labels.csv",
    model="gmm",
    score_direction="higher_is_more_suspicious",
    table_dir="results/phase2/gmm/tables",
)

print(result.ranking_metrics)
```

### Running Automated Test Suite
```bash
pytest tests/test_gmm_baseline.py -v
```
