# Phase 2 Autoencoder Experiment Plan and Model-Selection Rule

**Document Status:** Pre-Registered & Frozen (Pending Team Review)  
**Author:** Abhigya Sachdeva  
**Milestone:** Phase 2 Anomaly Detection Exploration (Issue #27)  
**Target Repository:** `WizardCarp85/Card-Fraud-Triage-ML-DL`  

---

## 1. PURPOSE

This document formally pre-registers the candidate model architectures, hyperparameter factors, training configurations, execution design, run-log schema, and model-selection decision rules for Phase 2 Autoencoder (AE) anomaly detection exploration on the Credit Card Fraud Detection benchmark (ULB / Kaggle).

### Pre-Registration & Governance Statement
- **Frozen Protocol:** This experimental plan is fixed prior to conducting any training, hyperparameter tuning, or ablation runs.
- **Change Control:** Once merged into `main`, this plan is strictly frozen. Any subsequent alteration to candidate architectures, hyperparameter spaces, stopping criteria, or selection rules requires opening a formal new GitHub issue, documenting technical justification, and obtaining team sign-off. Post-hoc adjustments following empirical observations are strictly prohibited.
- **Strict Holdout Test Exclusion:** All holdout test partition rows ($N = 42,560$, 52 frauds) and test labels (`Class`) are strictly excluded from candidate training, checkpoint evaluation, threshold tuning, and model selection. The holdout test partition is reserved exclusively for the one-time final paired evaluation against the frozen Phase-1 GMM baseline in Issue #38. No test-set metric shall be computed, printed, or saved anywhere within the execution of this experiment matrix.

---

## 2. FIXED TRAINING SETTINGS (Identical for All Candidates)

All candidate models share an identical training backbone, optimization configuration, stopping policy, and feature representation. Only the declared experimental factors defined in Section 3 may vary.

| Setting | Pre-Registered Specification | Operational Rule / Rationale |
|---|---|---|
| **Architecture Topology** | Fully-connected symmetric undercomplete Autoencoder:<br>$29 \to 32 \to B \to 32 \to 29$<br>($B$ = declared bottleneck factor) | Encoder: $\text{Linear}(29, 32) \to \text{ReLU} \to \text{Linear}(32, B)$.<br>Decoder: $\text{Linear}(B, 32) \to \text{ReLU} \to \text{Linear}(32, 29)$. |
| **Hidden Activations** | Rectified Linear Unit (`ReLU`) | Applied after hidden layers ($32$ and $B$). |
| **Output Activation** | Linear (identity mapping) | No non-linear activation on output reconstruction layer to allow unbounded continuous feature reconstruction. |
| **Loss Function** | Mean-Squared Error ($\text{MSE}$) across all 29 features:<br>$\mathcal{L}_{\text{MSE}}(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{29} \sum_{j=1}^{29} (x_j - \hat{x}_j)^2$ | Reconstruction target is **ALWAYS** the clean input $\mathbf{x}_{\text{clean}}$, even when training input denoising is applied. |
| **Optimizer** | Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$) | Standard adaptive gradient optimizer. |
| **Learning Rate** | $\eta = 1 \times 10^{-3}$ ($0.001$), no scheduler | Fixed constant learning rate across all epochs.<br>*(Repository Status Note: As of Issue #27, the repository does not yet contain a committed Issue #32 reference AE configuration file. Consequently, the fixed baseline values specified here—Adam, $\text{LR} = 1\times 10^{-3}$, batch size 256, max epochs 100, no scheduler—serve as the authoritative baseline settings for Issue #32 and subsequent runs).* |
| **Batch Size** | 256 transactions per mini-batch | Mini-batch gradient descent. |
| **Max Epochs** | 100 epochs | Upper bound on optimization iterations. |
| **Early Stopping** | Patience = 10 epochs monitored on legitimate-validation reconstruction loss | Checkpoint with the lowest legitimate-validation reconstruction loss ($\text{Class} = 0$ rows in validation split) is preserved.<br>**Explicit Monitoring Clarification:** Validation reconstruction loss is used exclusively for training convergence monitoring and checkpoint preservation; it is **NOT** the model-selection metric. |
| **Training Partition** | Legitimate training rows only ($N = 198,242$, 0 frauds) | Loaded strictly via frozen data loaders (#28/#31) referencing `data/splits/train_legit_indices.csv` from the Issue #25 manifest. All 366 training frauds are strictly excluded. |
| **Input Features & Order** | Frozen 29-feature matrix:<br>$(V_1, V_2, \dots, V_{28}, \log_{1p}(\text{Amount}))$ | Exactly 29 continuous features scaled via `RobustScaler` fitted strictly on legitimate training data (Issue #28). Elapsed `Time` and label `Class` are excluded from model inputs. |

---

## 3. DECLARED FACTORS (Only These May Differ Between Runs)

Only two controlled factors are varied across experimental candidates:

### Factor A: Bottleneck Latent Width ($B$)
- **Levels:** $B \in \{4, 8, 16\}$
- **Specifications:**
  - $B = 8$: Reference baseline architecture (Issue #32), providing an intermediate compression ratio ($29 \to 8$, $\approx 3.63:1$).
  - $B = 4$: Highly compressed bottleneck ($29 \to 4$, $\approx 7.25:1$), enforcing severe latent compression to isolate primary legitimate variance axes.
  - $B = 16$: Moderately compressed bottleneck ($29 \to 16$, $\approx 1.81:1$), providing higher expressive capacity for complex manifold geometries while remaining strictly undercomplete.

### Factor B: Input Denoising (Clean vs. Gaussian Noise)
- **Levels:** `clean` vs. `denoised`
- **Specifications:**
  - `clean`: Training inputs $\mathbf{x}$ are fed directly to the encoder without corruption.
  - `denoised`: Zero-mean additive Gaussian noise $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$ with standard deviation $\sigma = 0.5$ in the already-scaled feature units (Issue #33) is injected into training inputs: $\tilde{\mathbf{x}} = \mathbf{x} + \boldsymbol{\epsilon}$.
  - **Clean Target Invariant:** The reconstruction target remains the original clean input vector $\mathbf{x}_{\text{clean}}$ ($\text{target} = \mathbf{x}_{\text{clean}}$).
  - **Deterministic Evaluation Invariant:** Validation and holdout test partition inputs remain 100% clean and deterministic; noise is never added during validation or downstream holdout test evaluation.
  - **Seeding:** Input noise generation is deterministically seeded per run using the designated run seed.
  - **Evaluation Scope:** Denoising is not crossed across all widths; it is evaluated exclusively at the single winning bottleneck width identified in Stage 1 screening.

---

## 4. DESIGN

The experimental matrix comprises a two-stage sequential ablation protocol designed to bound computational cost while guaranteeing statistical reproducibility.

```
Stage 1: Screening (Issue #35)
├── Step 1A: Width Exploration (Clean Inputs, Seed 42)
│   ├── Run 1: Width B = 4  (Clean)
│   ├── Run 2: Width B = 8  (Clean, Issue #32 Reference Baseline)
│   └── Run 3: Width B = 16 (Clean)
│       └── [Apply Selection Rule -> Identify Best Width B*]
└── Step 1B: Denoising Exploration (Seed 42)
    └── Run 4: Width B = B* (Denoised, sigma = 0.5)
        └── [Apply Selection Rule -> Identify Stage-1 Winner C*]

Stage 2: Confirmation & Stability (Issue #36)
├── Multi-Seed Evaluation of Winner C* vs. Reference Baseline (B=8 Clean)
│   ├── Candidate C*                  x Seeds {0, 7, 42} -> 3 runs (seed 42 reused/re-logged)
│   └── Reference Baseline (B=8 Clean) x Seeds {0, 7, 42} -> 3 runs (seed 42 reused/re-logged)
└── [Compute Mean Validation AP across 3 seeds -> Final Model Selection]
```

### Stage 1: Screening Protocol (Issue #35)
- **Objective:** Identify the optimal latent bottleneck width $B^*$, and evaluate whether input denoising improves anomaly triage fidelity at that optimal width.
- **Step 1A (Bottleneck Widths):** 3 candidates ($B \in \{4, 8, 16\}$) $\times$ `clean` inputs $\times$ seed 42 $\to$ 3 runs.
- **Selection of $B^*$:** The primary selection rule (Section 5) is applied to identify the top-performing width $B^*$.
- **Step 1B (Denoising Ablation):** 1 candidate ($B = B^*$) $\times$ `denoised` ($\sigma = 0.5$) $\times$ seed 42 $\to$ 1 run. Compared directly against the clean $B^*$ run from Step 1A.
- **Total Stage 1 Executions:** Exactly 4 runs.

### Stage 2: Confirmation & Multi-Seed Stability Protocol (Issue #36)
- **Objective:** Verify multi-seed stability and guard against seed-dependent optimization artifacts.
- **Candidates:**
  1. The rule-based winning configuration from Stage 1 ($C^*$).
  2. The width-8 clean reference baseline ($C_{\text{ref}} = B8\text{-clean}$).
  *(Note: If $C^*$ is $B8\text{-clean}$, the runner-up configuration from Stage 1 is evaluated as the comparative baseline, ensuring exactly two distinct architectures undergo confirmation).*
- **Seeds:** Evaluated across the three pre-registered project seeds: $\{0, 7, 42\}$.
- **Total Stage 2 Executions:** At most 6 runs (2 candidate configurations $\times$ 3 seeds).

### Run Budget & Failure Logging Policy
- **Grand Total Runs:** $\le 10$ runs across the entire Phase 2 Autoencoder exploration matrix.
- **Zero Run-Dropping Policy:** Every executed run must be committed to the run-log CSV. If a run experiences numerical instability, NaN divergence, or out-of-memory errors, the run is logged with `epochs_run`, failure status, error trace, and NaN metric entries. Failed runs are never silently discarded or overwritten.

---

## 5. SELECTION RULE (Fixed in Advance)

Candidate selection is hierarchical, quantitative, and deterministic. All selection criteria are evaluated strictly on the validation partition ($N = 42,558$, 55 frauds).

```
Candidate Ranking Hierarchy:
1. Primary Metric: Validation Average Precision (PR-AUC) [Higher is better]
   │
   ├── Difference >= 0.005?
   │   └── YES -> Select candidate with higher validation AP
   │
   └── NO (Difference < 0.005, Indifference Band)
       │
       2. Secondary Metric: Validation Precision@100 [Higher is better]
          │
          ├── Differing Precision@100?
          │   └── YES -> Select candidate with higher Precision@100
          │
          └── NO (Identical Precision@100)
              │
              3. Deterministic Tie-Breaker
                 ├── Prefer smaller model (B = 4 > B = 8 > B = 16)
                 └── Prefer simpler regularization (clean > denoised)
```

### Detailed Decision Criteria
1. **Primary Ranking Metric:** Validation Average Precision (`val_AP` / PR-AUC) computed across the full validation split ($N = 42,558$, 55 frauds) using mean-squared reconstruction error as the anomaly score ($s(\mathbf{x}) = \frac{1}{29} \sum_{j=1}^{29}(x_j - \hat{x}_j)^2$). Higher value is superior.
2. **Secondary Ranking Metric:** Validation Precision@100 (`val_Precision@100`), representing the fraud precision within the top-100 highest anomaly scores.
   - **Threshold Condition:** Invoked if and only if the primary validation AP values of two competing models differ by less than $0.005$ ($|\text{val\_AP}_1 - \text{val\_AP}_2| < 0.005$). In this indifference band, the model with higher Precision@100 is selected.
3. **Deterministic Tie-Break Policy:** If two candidates remain tied within the $0.005$ AP indifference band and share identical Precision@100:
   - **Parsimony Rule:** Prefer the smaller model architecture (smaller bottleneck width: $B=4$ preferred over $B=8$, and $B=8$ preferred over $B=16$).
   - **Simplicity Rule:** Prefer uncorrupted training over denoising (`clean` preferred over `denoised`).
4. **Multi-Seed Aggregation Policy (Stage 2):**
   - The final production candidate architecture is selected based on the **MEAN validation Average Precision** across all three Stage-2 confirmation seeds:
     $$\overline{\text{val\_AP}} = \frac{1}{3} \sum_{s \in \{0, 7, 42\}} \text{val\_AP}^{(s)}$$
   - **Strict Prohibition Against Cherry-Picking:** Selection by the single best seed is strictly prohibited.
   - **Stability Requirement:** Any seed exhibiting instability (e.g., loss divergence or severe performance collapse $|\text{val\_AP}^{(s)} - \overline{\text{val\_AP}}| > 0.10$) must be explicitly flagged and reported, not concealed or selectively re-seeded.
5. **Strict Holdout Exclusions:** Zero holdout test rows, zero test labels, and zero test-derived performance statistics participate in model selection, checkpoint picking, hyperparameter pruning, or threshold determination.

---

## 6. RUN-LOG SCHEMA

Every executed run in the experimental matrix is appended as an individual row to the RFC 4180-compliant CSV run-log located at `results/phase2/ae_run_log.csv`.

### CSV Header
```csv
run_id,config_json,seed,git_commit,dataset_manifest_hash,start_time,end_time,runtime_s,epochs_run,early_stopped,final_train_loss,val_recon_loss,val_AP,val_Precision@100,val_recall_FPR1pct,val_recall_FPR5pct,checkpoint_path,checkpoint_sha256
```

### Column Specifications
| # | Column Name | Data Type | Schema Format & Description |
|---|---|---|---|
| 1 | `run_id` | String | Unique identifier encoding stage, width, noise mode, seed (e.g., `RUN-S1-W08-CLN-S42`). |
| 2 | `config_json` | String (JSON) | Serialized JSON string specifying model hyperparameter values. |
| 3 | `seed` | Integer | Random seed value ($0, 7, \text{or } 42$). |
| 4 | `git_commit` | String | Full or abbreviated Git commit SHA under which the run was executed. |
| 5 | `dataset_manifest_hash` | String | SHA-256 checksum of `data/split_manifest.json` verifying split integrity. |
| 6 | `start_time` | String (ISO 8601) | Timestamp of run initialization (`YYYY-MM-DDTHH:MM:SSZ`). |
| 7 | `end_time` | String (ISO 8601) | Timestamp of run completion or termination (`YYYY-MM-DDTHH:MM:SSZ`). |
| 8 | `runtime_s` | Float | Wall-clock execution duration in seconds. |
| 9 | `epochs_run` | Integer | Total epochs executed before early stopping or maximum limit. |
| 10 | `early_stopped` | Boolean | `True` if early stopping triggered; `False` if maximum epochs reached. |
| 11 | `final_train_loss` | Float | Final training reconstruction MSE loss on legitimate training rows. |
| 12 | `val_recon_loss` | Float | Validation reconstruction MSE loss on legitimate validation rows at best checkpoint. |
| 13 | `val_AP` | Float | Validation Average Precision (PR-AUC) across the full validation split. |
| 14 | `val_Precision@100` | Float | Validation precision at review capacity of 100 highest-scoring transactions. |
| 15 | `val_recall_FPR1pct` | Float | Validation fraud recall at 1.0% False Positive Rate budget threshold. |
| 16 | `val_recall_FPR5pct` | Float | Validation fraud recall at 5.0% False Positive Rate budget threshold. |
| 17 | `checkpoint_path` | String | Relative filesystem path to the serialized PyTorch model checkpoint (`.pt`). |
| 18 | `checkpoint_sha256` | String | Cryptographic SHA-256 checksum of the serialized model checkpoint file. |

---

## 7. GUARDRAILS

To preserve scientific validity, prevent benchmark contamination, and ensure end-to-end auditability, the following guardrails are strictly enforced:

1. **Legitimate-Only Training Enforced by Loader Tests:**
   - Model fitting operates strictly on legitimate transactions ($\text{Class} = 0$, $N = 198,242$).
   - All 366 training frauds are strictly excised.
   - Enforced programmatically by automated data loader tests (`tests/test_data_split.py`) and split manifest assertions. The `Class` column is physically stripped from training input tensors and loss routines.
2. **Phase-2 Validation Split Prior Exposure Disclosure:**
   - **Methodological Statement:** The Phase-2 validation split has prior Phase-1 holdout exposure (having been utilized during Phase 1 for threshold calibration and exploratory triage analysis, alongside the Phase-1 holdout test set evaluation documented in `reports/report.pdf` and `results/tables/gmm_scores_test.csv`).
   - **Phase-2 Safeguard:** Because the historical performance envelope is known, the holdout test partition is strictly quarantined to prevent iterative test-set snooping. All Phase-2 architectural pruning, candidate selection, and threshold tuning are restricted entirely to the validation split ($N = 42,558$).
   - **One-Time Test Evaluation:** The holdout test split is consumed exactly once during the final paired evaluation in Issue #38.
3. **Committed Artifact Integrity & Traceability:**
   - **No Fabricated Numbers:** Every metric, runtime, and loss value reported in documentation, presentations, or tables must originate from an executed run recorded in `results/phase2/ae_run_log.csv`.
   - **Auditable Provenance:** Every log row cryptographically links the git commit hash, dataset manifest hash, and serialized model checkpoint checksum.

---

## 8. SIGN-OFF AND PRE-REGISTRATION RECORD

| Role | Name | Status | Timestamp / Date | Signature / Note |
|---|---|---|---|---|
| **Author / ML Research Engineer** | Abhigya Sachdeva | Pre-Registered | 2026-11-16 | Initial pre-registration document authored and committed prior to ablation runs. |
| **Peer Reviewer / Team** | Team (`@WizardCarp85`) | Pending | — | Review pending formal pull request approval. |
