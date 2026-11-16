# Phase 2 Workflow Orchestration Guide
**Project:** Card Fraud Triage (AML + DL)  
**Owner:** Yashkumar Nimje (`@WizardCarp85`) — Data Architecture and Workflow Integration  
**Dependencies:** Issue #25 (Dataset split), Issue #26 (Preprocessing)  
**Coordinated Interfaces:** Issue #28 (GMM Baseline), Issue #30 (Autoencoder Architecture), Issue #32 (Evaluation)  

---

## 1. Overview & Architecture

Phase 2 introduces deep learning (PyTorch Autoencoders) alongside the probabilistic GMM baseline to learn non-linear transaction reconstruction manifolds. 

To ensure exact reproducibility, cross-platform consistency, and strict isolation between Phase 1 and Phase 2, this orchestration framework provides:
- **Unified Configuration**: Single documented configuration hierarchy in [`src/config.py`](file:///Users/Wizard/Documents/Projects/Working/Card-Fraud-Triage-ML-DL/src/config.py).
- **Standardized Model Contract**: Models adhere to [`BaseAnomalyModel`](file:///Users/Wizard/Documents/Projects/Working/Card-Fraud-Triage-ML-DL/src/models/base.py) with `.fit(X_train)` and `.score(X) -> 1D anomaly scores`. No model-specific code is duplicated in the runner.
- **Fail-Fast Safety Checks**: Immediate, actionable errors when source data or frozen split manifests are missing.
- **Collision & Overwrite Prevention**: Reruns fail clearly if an existing artifact directory exists unless `--overwrite` is explicitly specified.
- **Complete Run Provenance**: Every execution captures git commit, system platform, Python environment, package versions, input SHA-256 hashes, config snapshots, and an artifact inventory.
- **Phase 1 Isolation**: All Phase 2 outputs are saved under `artifacts/phase2/<run_id>/`, never modifying or overwriting Phase 1 files in `results/`.

```
src/
├── config.py                 # Single source of truth configuration (dataclasses & JSON)
├── data.py                   # Validates and loads splits from frozen split_manifest.json
├── preprocessing.py          # Leakage-safe log1p & RobustScaler fitted on legit-only train
├── provenance.py             # Captures git commit, versions, input hashes, and hardware
├── evaluation_runner.py      # Invokes evaluation.py and exports triage and threshold metrics
├── run.py                    # Core pipeline orchestrator & CLI
└── models/
    ├── base.py               # Abstract BaseAnomalyModel interface
    ├── gmm.py                # Gaussian Mixture Model wrapper (negative log-likelihood)
    └── autoencoder.py        # PyTorch Deep Autoencoder (reconstruction MSE scoring)
```

---

## 2. Quickstart & Exact Commands

### Prerequisites
A teammate checking out a clean clone needs only:
1. Place `creditcard.csv` in `data/creditcard.csv` (or use `--data-path`).
2. Verify that `data/split_manifest.json` exists (if not, run `python data/freeze_split.py`).

### 1. Run Complete Pipeline (End-to-End)
Runs ingestion, leakage-free scaling, model training, scoring, and evaluation:

```bash
# Run with GMM baseline (default)
python run_phase2.py --model gmm

# Run with Deep Autoencoder (PyTorch)
python run_phase2.py --model autoencoder

# Run both models in one unified run
python run_phase2.py --model all
```

### 2. Common CLI Options & Overrides
```bash
# Specify custom run identifier
python run_phase2.py --model autoencoder --run-id ae_experiment_v1

# Override training epochs for Autoencoder
python run_phase2.py --model autoencoder --epochs 20

# Override global reproducibility seed
python run_phase2.py --model gmm --seed 123

# Allow overwriting an existing run directory
python run_phase2.py --model gmm --run-id my_run --overwrite

# Run with custom JSON configuration file
python run_phase2.py --config configs/my_experiment.json
```

---

## 3. Configuration Contract

The master configuration in [`src/config.py`](file:///Users/Wizard/Documents/Projects/Working/Card-Fraud-Triage-ML-DL/src/config.py) specifies all parameters:

| Config Section | Key Parameters | Default Value | Description |
|---|---|---|---|
| `data` | `raw_data_path`<br>`manifest_path` | `data/creditcard.csv`<br>`data/split_manifest.json` | Paths to dataset and frozen split manifest |
| `preprocessing` | `scaler_type`<br>`log_transform_amount`<br>`fit_on_legit_only` | `"robust"`<br>`True`<br>`True` | Outlier-resistant `RobustScaler` fit strictly on legit train |
| `gmm` | `n_components`<br>`covariance_type`<br>`reg_covar` | `14`<br>`"full"`<br>`1e-6` | Optimal GMM configuration established in Phase 1 |
| `autoencoder` | `latent_dim`<br>`hidden_dims`<br>`learning_rate`<br>`epochs` | `8`<br>`(20, 14)`<br>`1e-3`<br>`15` | PyTorch symmetric encoder-decoder topology |
| `evaluation` | `fpr_budgets`<br>`review_capacities` | `(0.001, 0.01, 0.05)`<br>`(100, 500, 1000)` | Operational review budgets and capacity thresholds |
| `artifacts` | `base_output_dir`<br>`run_id`<br>`overwrite` | `"artifacts/phase2"`<br>`None` (auto-generated)<br>`False` | Destination directory and collision safety |
| `seed` | `seed` | `42` | Global random seed enforced across NumPy, PyTorch, Python |

---

## 4. Expected Output Artifacts

Each execution generates a self-contained directory under `artifacts/phase2/<run_id>/`:

```
artifacts/phase2/run_YYYYMMDD_HHMMSS_xxxxxx/
├── config.json                     # Complete JSON snapshot of configuration used
├── provenance.json                 # Git commit, OS, Python version, library versions, input SHA-256
├── run_manifest.json               # Final run summary, metric results, and artifact checksums
├── logs/
│   └── pipeline.log                # Complete execution timestamped log
├── preprocessed/
│   └── scaler_robust.joblib        # Fitted feature scaler (fit only on legitimate training data)
├── models/
│   ├── gmm/
│   │   ├── gmm_model.joblib        # Serialized GaussianMixture estimator
│   │   └── gmm_meta.json           # Convergence status, lower bound, iteration count
│   └── autoencoder/
│       ├── autoencoder.pt          # PyTorch model state_dict weights
│       └── autoencoder_meta.json   # Architecture config, loss curves, training history
├── scores/
│   ├── scores_val.csv              # original_index, anomaly_score, Class (Validation split)
│   └── scores_test.csv             # original_index, anomaly_score, Class (Test split)
└── tables/
    ├── final_ranking_metrics.csv   # PR-AUC (Average Precision) and ROC-AUC per split
    ├── validation_fpr_thresholds.csv # Calibrated score thresholds at FPR 0.1%, 1.0%, 5.0%
    ├── test_threshold_metrics.csv  # Caught frauds and false alarms under validation thresholds
    ├── test_triage_table.csv       # Hit-rates and precisions across review capacities (Top-K)
    └── summary_metrics.json        # Machine-readable metric summary
```

---

## 5. Provenance & Safety Mechanisms

### 1. Fail-Fast on Missing Inputs
If a teammate runs the pipeline without the source data or frozen manifest, execution halts immediately with instructions:
```
[PIPELINE ERROR] Execution failed: Required dataset not found at 'data/creditcard.csv'.
Action required: Please provide 'creditcard.csv' at the configured path or download it via kagglehub ('mlg-ulb/creditcardfraud').
```

### 2. Silent Overwrite Prevention
If a target run directory already exists:
```
[PIPELINE ERROR] Execution failed: Run directory 'artifacts/phase2/my_run' already exists. Refusing to overwrite prior runs silently. Please specify a new run ID or pass '--overwrite'.
```

### 3. Isolation from Phase 1
Phase 1 results reside in `results/`. The Phase 2 runner writes exclusively to `artifacts/phase2/`, ensuring historical benchmarks are never overwritten or contaminated.

### 4. Unsupervised Purity Enforcement
During preprocessing, `PreprocessingPipeline` programmatically asserts that `(df_train_legit['Class'] == 0).all()`. If any fraudulent record is present in the training corpus, preprocessing immediately raises `ValueError`.

---

## 6. Verification & Automated Tests

Run the complete test suite:
```bash
python -m unittest discover tests
```

Tests include:
- `tests/test_data_split.py`: Validates Issue #25 dataset split reproduction, checksums, and counts.
- `tests/test_evaluation.py`: Validates Issue #32 leakage-safe metric definitions and triage policies.
- `tests/test_orchestration.py`: Validates Issue orchestration, configuration roundtrips, failure modes, model interfaces, and end-to-end execution.
