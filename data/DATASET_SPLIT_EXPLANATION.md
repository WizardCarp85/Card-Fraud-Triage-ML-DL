# Phase 2 Data Provenance & Chronological Split Contract
**Project:** Card Fraud Triage (AML + DL)  
**Author / Owner:** Yashkumar Nimje (`@WizardCarp85`) — Team Lead & Data Architecture  
**Milestone:** Phase 2 Data Milestone (Issue #25)  
**Status:** Frozen & Verified  

---

## 1. Executive Summary & Objective

This document defines the **authoritative data-provenance contract** for Phase 2 of the Card Fraud Triage project. Every Phase 2 model (e.g., Deep Autoencoders, hybrid GMM-Autoencoder score fusion, semi-supervised calibration) and downstream evaluation run must strictly adhere to the dataset split and preprocessing boundaries established herein.

### Acceptance Criteria Verification Summary
| Criterion | Status | Empirical Evidence |
|---|---|---|
| **Reproducibility** | **PASSED** | Fresh runs reproduce identical row memberships, row identifiers, and array SHA-256 hashes. |
| **Mutual Exclusivity** | **PASSED** | Zero overlap: $\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset$, $\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset$, $\mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$. |
| **Completeness** | **PASSED** | Exact partition: $|\mathcal{S}_{\text{train}}| + |\mathcal{S}_{\text{val}}| + |\mathcal{S}_{\text{test}}| = 283,726 = |\mathcal{D}_{\text{dedup}}|$. |
| **Unsupervised Purity** | **PASSED** | Training subset for model fitting contains strictly legitimate transactions ($\text{Class} = 0$, $N = 198,242$, Fraud = $0$). |
| **Historical Concordance** | **PASSED** | 100% agreement with Phase 1 documented counts (Train: 198,608; Val: 42,558; Test: 42,560). |
| **Cryptographic Integrity** | **PASSED** | Source file checksums and split membership hashes cryptographically frozen in `data/split_manifest.json`. |

---

## 2. Source Data Provenance & Cryptographic Checksums

The project utilizes the benchmark **Credit Card Fraud Detection Dataset (ULB / Kaggle)**, representing European cardholder transactions over a 48-hour observation window (September 2013).

```
File Identifier:       creditcard.csv
File Size:             150,828,752 bytes (143.84 MB)
Total Observations:    284,807 transactions
Feature Dimensionality: 31 columns (Time, V1–V28, Amount, Class)
Missing / NaN Values:  0 (zero missing cells)
SHA-256 Checksum:      76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89
MD5 Checksum:          e90efcb83d69faf99fcab8b0255024de
```

---

## 3. Data Hygiene & Deduplication Strategy

In exploratory data analysis (Phase 1, Section 2), duplicate row inspection identified **1,081 duplicate transactions** out of 284,807 raw rows.

### Deduplication Policy
- Strategy: `df.drop_duplicates(keep='first')`
- Rationale: Financial fraud datasets often log re-transmitted API retries or identical webhook pings. Retaining duplicate rows in unsupervised density estimation introduces artificial high-density modes, distorting mixture components and reconstruction manifolds.
- Raw Observations: `284,807`
- Duplicate Observations Dropped: `1,081`
- Unique Deduplicated Observations: `283,726`

---

## 4. Chronological Splitting Methodology

Real-world financial transaction streams arrive sequentially. Random cross-validation or random train-test splitting leaks future cardholder spending habits into the past (lookahead bias).

To replicate production deployment, the deduplicated dataset is partitioned strictly chronologically based on elapsed transaction time (`Time`):

```
Chronological Timeline: 0.0s ──────────────────────────── 132,906.0s ──────────────────────── 151,320.0s ──────────────── 172,792.0s
Partitions:            [       Train: 70% (~36.9 hrs)       ] [   Validation: 15% (~5.1 hrs)   ] [    Test: 15% (~6.0 hrs)    ]
Row Counts:            [          198,608 rows              ] [          42,558 rows           ] [         42,560 rows        ]
Frauds:                [            366 frauds              ] [            55 frauds           ] [           52 frauds        ]
Base Rate:             [             0.1843%                ] [             0.1292%            ] [            0.1222%         ]
```

### Split Boundaries
1. **Train Partition ($n = 198,608$, $70.00\%$):**
   - Time span: `0.0s` $\to$ `132,906.0s`
   - Boundary condition: `Time < 132906.0s` plus boundary tie-breaker row `199323`.
2. **Validation Partition ($n = 42,558$, $15.00\%$):**
   - Time span: `132,906.0s` $\to$ `151,320.0s`
   - Boundary condition: Boundary tie-breaker row `199324`, all rows strictly between `132,906.0s < Time < 151,320.0s`, plus boundary tie-breaker row `242066`.
3. **Holdout Test Partition ($n = 42,560$, $15.00\%$):**
   - Time span: `151,320.0s` $\to$ `172,792.0s`
   - Boundary condition: Boundary tie-breaker rows `[242063, 242064, 242065]`, plus all rows with `Time > 151,320.0s`.

---

## 5. Timestamp Collision & Tie-Breaking Analysis

### Root Cause of Timestamp Collisions
The dataset's `Time` attribute records elapsed seconds as floating-point integers (`0.0, 1.0, 2.0, ... 172792.0`). Across the 48-hour window (172,800 seconds) and 283,726 transactions:
- **79,413 unique seconds** contain multiple simultaneous transactions.
- **238,547 transactions** (84.08% of the dataset) share an elapsed second with at least one other transaction.

### Boundary Collisions & Non-Stable Sort Platform Variance
When `df.sort_values('Time')` was called in Phase 1, pandas used its default quicksort routine without a secondary tie-breaker. Because quicksort is **unstable**, the relative ordering of rows sharing identical timestamps depends on pivot choices, which vary across Python/NumPy versions, CPU architectures (x86_64 vs Apple Silicon arm64), and operating systems.

Two timestamp collisions occurred precisely at the split boundaries:
1. **Train / Validation Boundary (`Time == 132906.0s`):**
   - Two transactions exist: raw index `199323` and `199324`.
   - Both transactions are legitimate (`Class == 0`).
   - Phase 1 placement: row `199323` was placed at index `198,607` (Train); row `199324` at index `198,608` (Validation).
2. **Validation / Test Boundary (`Time == 151320.0s`):**
   - Four transactions exist: raw indices `242063`, `242064`, `242065`, and `242066`.
   - All four transactions are legitimate (`Class == 0`).
   - Phase 1 placement: row `242066` was placed at index `241,165` (Validation); rows `242065`, `242063`, `242064` were placed at indices `241,166`, `241,167`, `241,168` (Test).

### Verification Against Phase 1 Serialized Score Artifacts
By inspecting the serialized Phase 1 score files [`results/tables/gmm_scores_val.csv`](file:///Users/Wizard/Documents/Projects/Working/Card-Fraud-Triage-ML-DL/results/tables/gmm_scores_val.csv) (42,558 rows) and [`results/tables/gmm_scores_test.csv`](file:///Users/Wizard/Documents/Projects/Working/Card-Fraud-Triage-ML-DL/results/tables/gmm_scores_test.csv) (42,560 rows), we verified the exact anomaly scores and matched each row to its underlying raw index.

Crucially:
- **Zero boundary transactions are fraudulent.**
- All 366 training frauds occurred strictly at `Time < 132906.0s`.
- All 55 validation frauds occurred strictly at `132906.0s < Time < 151320.0s`.
- All 52 test frauds occurred strictly at `Time > 151320.0s`.
- As a mathematical consequence, **the fraud counts per split (366 / 55 / 52) are 100% invariant to how timestamp ties are broken.**

### Frozen Tie-Breaking Contract
To ensure that all future runs produce identical row memberships across all platforms, the boundary row memberships are explicitly frozen in the manifest:
- **Train:** All rows with `Time < 132906.0s` + row `199323`.
- **Validation:** Row `199324` + all rows with `132906.0s < Time < 151320.0s` + row `242066`.
- **Test:** Rows `[242063, 242064, 242065]` + all rows with `Time > 151320.0s`.

---

## 6. Split Partition Verification & Class Distribution

| Partition | Total Transactions | Legitimate (`Class == 0`) | Fraud (`Class == 1`) | Fraud Base Rate | Legit Base Rate | Time Horizon (Seconds) |
|---|---|---|---|---|---|---|
| **Train (70%)** | 198,608 | 198,242 | 366 | 0.1843% | 99.8157% | `0.0s` $\to$ `132,906.0s` |
| **Validation (15%)** | 42,558 | 42,503 | 55 | 0.1292% | 99.8708% | `132,906.0s` $\to$ `151,320.0s` |
| **Test (15%)** | 42,560 | 42,508 | 52 | 0.1222% | 99.8778% | `151,320.0s` $\to$ `172,792.0s` |
| **Total Deduplicated** | **283,726** | **283,253** | **473** | **0.1667%** | **99.8333%** | `0.0s` $\to$ `172,792.0s` |

---

## 7. Legitimate-Only Training Protocol (Unsupervised Purity)

In the card fraud triage paradigm, anomaly detection models model the probability distribution of legitimate purchasing behavior:
$$\mathcal{M}: \mathbb{R}^D \to \mathbb{R}^+ \quad \text{where} \quad \mathcal{M}(x) \approx p(x \mid \text{legitimate})$$

### Operational Protocol
1. **Masking:** During feature scaling (`RobustScaler`) and model fitting (Deep Autoencoder / GMM), all 366 fraudulent transactions present in the training partition are **strictly excluded**.
2. **Fit Corpus:** Models are trained exclusively on $\mathcal{X}_{\text{train, legit}}$ ($N = 198,242$).
3. **Verification:**
   - Total rows in `train_legit`: `198,242`
   - Fraudulent transactions in `train_legit`: `0` ($0.0000\%$)
   - Legitimate transactions in `train_legit`: `198,242` ($100.0000\%$)

---

## 8. Prior Phase 1 Exposure of the Holdout Test Set

### Disclosure of Exposure
During Phase 1, the holdout test set ($N = 42,560$, 52 frauds) was evaluated to benchmark the production GMM model ($K=14$, `full` covariance). The resulting metrics were documented in:
- `results/tables/gmm_scores_test.csv` (PR-AUC: 0.6773, ROC-AUC: 0.9067)
- `reports/report.pdf` (Formal technical IEEE report)
- `presentation/Phase1_Slides.pdf` (Viva slide deck)
- `docs/Card_Fraud_Triage_Project_Documentation.md`

### Scientific Safeguards for Phase 2
Because the Phase 1 test metrics are publicly documented, the holdout test set is **no longer a strictly blind distribution**. To maintain experimental rigor and prevent test-set snooping / benchmark contamination during Phase 2:

1. **Validation-Only Model Selection:**
   All Phase 2 architectural decisions (e.g., Autoencoder bottleneck latent dimension, encoder/decoder layer topologies, activation functions, learning rates, loss formulations) must be selected strictly by evaluating on the **Validation set** ($N = 42,558$, 55 frauds).
2. **Validation-Only Triage Calibration:**
   All operational decision thresholds (e.g., FPR budgets at 0.1%, 1.0%, 5.0%, or Top-$K$ review capacities) must be calibrated **strictly on validation anomaly scores**.
3. **Single Test Evaluation:**
   The holdout test set must only be scored once per candidate architecture after all hyperparameters and triage thresholds are frozen. Iterative hyperparameter tuning against test scores is prohibited.
4. **Leakage-Safe Code Enforcement:**
   All Phase 2 evaluations must utilize [`evaluation.py`](file:///Users/Wizard/Documents/Projects/Working/Card-Fraud-Triage-ML-DL/evaluation.py), which enforces threshold selection exclusively on validation inputs and rejects test-driven tuning.

---

## 9. Artifact Inventory & Programmatic Usage

### Manifest Artifacts
- **Machine-Readable Manifest:** [`data/split_manifest.json`](file:///Users/Wizard/Documents/Projects/Working/Card-Fraud-Triage-ML-DL/data/split_manifest.json)
- **Index Exports:**
  - [`data/splits/train_indices.csv`](file:///Users/Wizard/Documents/Projects/Working/Card-Fraud-Triage-ML-DL/data/splits/train_indices.csv) (198,608 rows)
  - [`data/splits/train_legit_indices.csv`](file:///Users/Wizard/Documents/Projects/Working/Card-Fraud-Triage-ML-DL/data/splits/train_legit_indices.csv) (198,242 rows)
  - [`data/splits/val_indices.csv`](file:///Users/Wizard/Documents/Projects/Working/Card-Fraud-Triage-ML-DL/data/splits/val_indices.csv) (42,558 rows)
  - [`data/splits/test_indices.csv`](file:///Users/Wizard/Documents/Projects/Working/Card-Fraud-Triage-ML-DL/data/splits/test_indices.csv) (42,560 rows)
- **Automated Pipeline Script:** [`data/freeze_split.py`](file:///Users/Wizard/Documents/Projects/Working/Card-Fraud-Triage-ML-DL/data/freeze_split.py)
- **Unit Test Suite:** [`tests/test_data_split.py`](file:///Users/Wizard/Documents/Projects/Working/Card-Fraud-Triage-ML-DL/tests/test_data_split.py)

### Programmatic Usage in Phase 2
Any Phase 2 Python script or PyTorch training pipeline can load the frozen splits directly:

```python
from data.freeze_split import load_split_datasets

# Load all splits with guaranteed data provenance
splits = load_split_datasets(
    data_filepath="data/creditcard.csv",
    manifest_filepath="data/split_manifest.json"
)

df_train = splits["train"]            # Full training partition (198,608 rows)
df_train_legit = splits["train_legit"] # Legitimate-only training subset for model fitting (198,242 rows)
df_val = splits["val"]                # Validation partition (42,558 rows)
df_test = splits["test"]              # Holdout test partition (42,560 rows)
```
