# Model Card & Limitations Record: Frozen Phase 2 Undercomplete Autoencoder

**Model Name:** Phase 2 Frozen Undercomplete Autoencoder (`Phase2_Frozen_Undercomplete_Autoencoder`)  
**Card Version:** 1.0  
**Date:** 2026-11-17  
**Author:** Abhigya Sachdeva (`@ABHIGYA0205`)  
**Governing Issue:** Issue #43 (Model Card & Limitations Record)  
**Parent Milestone:** Phase 2 Anomaly Detection Exploration  
**Status:** FROZEN  

---

## 1. MODEL IDENTITY

Every architectural parameter, configuration setting, and cryptographic hash in this section is transcribed directly from the authoritative model manifest.

| Attribute | Specification |
|---|---|
| **Model Family** | Autoencoder (Unsupervised Anomaly Detection) |
| **Model Name** | `Phase2_Frozen_Undercomplete_Autoencoder` |
| **Architecture Topology** | Fully connected symmetric undercomplete autoencoder: `29 -> 32 -> 16 -> 32 -> 29` |
| **Input Dimension** | 29 |
| **Bottleneck Dimension ($B$)** | 16 |
| **Hidden Layer Dimension** | 32 (`hidden_dims = (32,)`) |
| **Realized Backbone** | `BatchNorm1d` + `LeakyReLU(negative_slope=0.1)` per Addendum A1 |
| **Output Layer Activation** | Linear (identity mapping, continuous reconstruction) |
| **Loss Function** | Mean Squared Error (MSE) over 29 features: $\mathcal{L}_{\text{MSE}}(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{29} \sum_{j=1}^{29} (x_j - \hat{x}_j)^2$ |
| **Optimizer** | `Adam(lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)` |
| **Learning Rate** | $0.001$ ($1\times 10^{-3}$, constant, no scheduler) |
| **Weight Decay** | $0.0$ |
| **Batch Size** | 256 transactions per mini-batch |
| **Max Epochs** | 100 epochs |
| **Early Stopping** | Patience = 10 epochs monitored on legitimate validation reconstruction MSE loss |
| **Denoising Regularization** | Training input noise $\sigma = 0.5$ (Gaussian, scaled feature units, clean targets) |
| **Selected Checkpoint Run ID** | `RUN-S2-W16-DNZ-S7` |
| **Selected Seed** | 7 |
| **Checkpoint Path** | `results/phase2/checkpoints/RUN-S2-W16-DNZ-S7.pt` |
| **Checkpoint SHA-256** | `0d247ab9757cd3b15caeaba0049dd7825c8049998dd57a649307c87c64b4a1fb` |
| **Dataset Manifest Hash** | `89f87ff166fd4438be88b58e77e133d3444124acb8f37f7eca1a6a123af7a7e3` |
| **Executing Git Commit** | `b3b686da6727d64f3566ca0bcc19bdfc84a9e122` |

Source: [results/ae_frozen_model_manifest.json](../results/ae_frozen_model_manifest.json#L1-L120)

---

## 2. INPUTS & PREPROCESSING

The model consumes a standardized, fixed-dimension tabular vector for each transaction:

- **Feature Count:** Exactly 29 continuous features.
- **Feature Composition:** Principal components $V_1, V_2, \dots, V_{28}$ plus log-transformed transaction amount $\text{Amount\_log} = \log_{1p}(\text{Amount})$.
- **Scaler:** `sklearn.preprocessing.RobustScaler` fitted strictly on legitimate training rows only (`Class == 0`, Issue #28), guaranteeing zero leakage of fraud statistics into feature center/scale parameters.
- **Frozen Feature Order:**
  ```
  V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15,
  V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount_log
  ```
- **Excluded Columns:** Elapsed `Time` is omitted to prevent time-decay artifacts; label `Class` is strictly quarantined and physically stripped from model inputs and loss routines.

Source: [results/ae_frozen_model_manifest.json](../results/ae_frozen_model_manifest.json#L75-L114)  
Source: [docs/Phase2_Preprocessing.md](../docs/Phase2_Preprocessing.md)  
Source: [docs/ae_experiment_plan.md](../docs/ae_experiment_plan.md#L36-L38)

---

## 3. TRAINING DATA POLICY

The model operates under strict semi-supervised/unsupervised anomaly detection principles:

- **Legitimate Training Rows Only:** The model was trained strictly on legitimate transactions ($N = 198,242$, zero frauds). All 366 training fraud transactions were excised to ensure the autoencoder learns exclusively the legitimate transaction manifold.
- **Frozen Chronological Partition:** Training, validation, and test sets are derived from the pre-registered 70/15/15 chronological split manifest produced in Issue #25 (`data/split_manifest.json`, SHA-256 `89f87ff166fd4438be88b58e77e133d3444124acb8f37f7eca1a6a123af7a7e3`).
- **Denoising Augmentation Policy:** Denoising (additive zero-mean Gaussian noise with $\sigma = 0.5$ in scaled feature units) was applied strictly as a training input augmentation with uncorrupted clean targets ($\mathbf{target} = \mathbf{x}_{\text{clean}}$) to force contractive projections toward the legitimate manifold, and is never applied during validation or inference evaluation.
- **Holdout Test Quarantine:** The test partition ($N = 42,560$, 52 frauds) is quarantined and was never touched during training, hyperparameter exploration, stability verification, or model selection.

Source: [docs/ae_experiment_plan.md](../docs/ae_experiment_plan.md#L36-L38)  
Source: [docs/ae_experiment_plan.md](../docs/ae_experiment_plan.md#L52-L61)  
Source: [results/ae_seed_stability.md](../results/ae_seed_stability.md#L24)

---

## 4. SCORE & INFERENCE

### Score Definition & Contract
The anomaly score is the per-row Mean Squared Error (MSE) reconstruction loss across all 29 features:
$$\text{score}(\mathbf{x}) = \frac{1}{29} \sum_{j=1}^{29} (x_j - \hat{x}_j)^2$$

- **Score Direction:** `higher_is_more_suspicious`. Transactions lying far from the learned legitimate manifold incur high reconstruction error and are assigned higher anomaly scores.
- **Determinism:** Model inference is strictly deterministic under evaluation mode (`eval()`, `torch.no_grad()`). Re-scoring the validation partition produces identical scores across runs with a verified maximum absolute difference $\max |\Delta| = 0.0$.

### Exact Loading & Scoring Interface
Inference conforms to the merged codebase modules:
- **Model Network:** `src.models.autoencoder.PyTorchAutoencoderNet(input_dim=29, latent_dim=16, hidden_dims=(32,))`
- **Model Wrapper:** `src.models.autoencoder.AutoencoderAnomalyModel`
- **Checkpoint Loading:**
  ```python
  import torch
  from src.models.autoencoder import PyTorchAutoencoderNet

  net = PyTorchAutoencoderNet(input_dim=29, latent_dim=16, hidden_dims=(32,))
  state_dict = torch.load("results/phase2/checkpoints/RUN-S2-W16-DNZ-S7.pt", map_location=torch.device("cpu"))
  net.load_state_dict(state_dict)
  net.eval()
  ```
- **Batch Scoring:**
  ```python
  with torch.no_grad():
      recon = net(batch_tensor)
      scores = torch.mean((batch_tensor - recon) ** 2, dim=1).cpu().numpy()
  ```
- **Evaluation Export:** `src.evaluation_runner.run_evaluation` records scores to CSV files conforming to `docs/score_file_schema.md` (`row_id,split,model,anomaly_score`) with model column set to `"autoencoder"`.

Source: [results/ae_frozen_model_manifest.json](../results/ae_frozen_model_manifest.json#L115-L119)  
Source: [src/models/autoencoder.py](../src/models/autoencoder.py#L17-L53)  
Source: [src/models/autoencoder.py](../src/models/autoencoder.py#L120-L138)  
Source: [docs/autoencoder_methods_note.md](../docs/autoencoder_methods_note.md#L70-L93)

---

## 5. WHY THIS MODEL (SELECTION RATIONALE)

The frozen model was chosen through a two-stage pre-registered decision matrix documented in [docs/ae_experiment_plan.md](../docs/ae_experiment_plan.md) (Section 5). All decisions were made exclusively on the validation split ($N = 42,558$, 55 frauds).

### Stage 1: Bottleneck & Denoising Screening (Issue #35)
1. **Bottleneck Width Search ($B \in \{4, 8, 16\}$, clean inputs, seed 42):**
   - Width $B=16$ achieved validation AP of $0.198333$, outperforming the $B=8$ reference baseline ($0.131272$) by $\Delta \text{AP} = +0.067061 \ge 0.005$ and width $B=4$ ($0.075492$) by $\Delta \text{AP} = +0.122841$.
   - Selected: $B^* = 16$.
2. **Denoising Regularization Ablation at $B^* = 16$ (seed 42):**
   - Injecting Gaussian input noise ($\sigma = 0.5$) surged validation AP from $0.198333$ (clean) to $0.340914$ (denoised) ($\Delta \text{AP} = +0.142581 \ge 0.005$, $+71.9\%$ relative gain).
   - Validation Precision@100 simultaneously improved from $0.3200 \to 0.3900$ (+7 additional frauds detected in top-100 reviews).
   - Stage 1 Champion: `RUN-S1-W16-DNZ-S42`.

Source: [results/ae_ablations_stage1.md](../results/ae_ablations_stage1.md#L23-L87)  
Source: [results/phase2/ae_run_log.csv](../results/phase2/ae_run_log.csv#L2-L5)

### Stage 2: Confirmation & Checkpoint Freeze (Issue #36)
1. **Multi-Seed Stability Confirmation (Seeds $\{0, 7, 42\}$):**
   - `W16-DNZ` achieved a 3-seed mean validation AP of $0.325692 \pm 0.014674$ ($\text{CV} = 4.5\%$).
   - The reference baseline `W08-CLN` achieved a 3-seed mean validation AP of $0.171576 \pm 0.037512$.
   - $\Delta \overline{\text{AP}} = 0.325692 - 0.171576 = +0.154116 \ge 0.005$ ($+89.8\%$ relative improvement).
   - Selection by single best seed was prohibited; `W16-DNZ` won decisively on mean validation AP.
2. **Closest-to-Mean Checkpoint Freeze:**
   - Seed 0: $|0.311635 - 0.325692| = 0.014057$
   - Seed 7: $|0.324527 - 0.325692| = 0.001165$ (**Closest to Mean**)
   - Seed 42: $|0.340914 - 0.325692| = 0.015222$
   - Checkpoint `RUN-S2-W16-DNZ-S7` (seed 7) was frozen per plan Section 5.4.

Source: [results/ae_seed_stability.md](../results/ae_seed_stability.md#L43-L90)  
Source: [results/ae_frozen_model_manifest.json](../results/ae_frozen_model_manifest.json#L29-L74)

---

## 6. VALIDATION ROBUSTNESS SUMMARY

Multi-seed evaluation across seeds $\{0, 7, 42\}$ demonstrated exceptional consistency:

| Metric | Multi-Seed Value (Mean $\pm$ Std) | Individual Seed Runs |
|---|---|---|
| **Validation AP (PR-AUC)** | $0.3257 \pm 0.0147$ ($0.325692 \pm 0.014674$) | Seed 0: $0.311635$<br>Seed 7: $0.324527$<br>Seed 42: $0.340914$ |
| **Validation Precision@100** | $0.3900 \pm 0.0000$ ($0.390000 \pm 0.000000$) | Seed 0: $0.3900$ (39/100)<br>Seed 7: $0.3900$ (39/100)<br>Seed 42: $0.3900$ (39/100) |
| **Validation Recall @ 1% FPR** | $0.7879 \pm 0.0105$ ($0.787879 \pm 0.010497$) | Seed 0: $0.781818$<br>Seed 7: $0.800000$<br>Seed 42: $0.781818$ |
| **Validation Recall @ 5% FPR** | $0.8303 \pm 0.0105$ ($0.830303 \pm 0.010497$) | Seed 0: $0.836364$<br>Seed 7: $0.836364$<br>Seed 42: $0.818182$ |
| **Coefficient of Variation (CV)** | $4.5\%$ | Dispersion: $\frac{\sigma}{\mu} = \frac{0.014674}{0.325692} \approx 4.5\%$ |
| **Stability Audit Status** | **STABLE** | All individual seeds within $\pm 0.10$ stability window |

Source: [results/ae_seed_stability.md](../results/ae_seed_stability.md#L30-L58)  
Source: [results/ae_frozen_model_manifest.json](../results/ae_frozen_model_manifest.json#L41-L53)  
Source: [results/phase2/ae_run_log.csv](../results/phase2/ae_run_log.csv#L5-L7)

---

## 7. OPERATIONAL THRESHOLDS

Operating thresholds for this autoencoder model are calibrated strictly on validation scores and recorded in a dedicated threshold manifest conforming to [docs/threshold_manifest_schema.md](../docs/threshold_manifest_schema.md) as part of the threshold calibration tasks.

Threshold manifest: [](../results/phase2/autoencoder/threshold_manifest.json) (calibrated and frozen in Issue #62).

| Budget (FPR) | Threshold | Validation Recall | Validation FPR | Validation Status |
|:---:|:---:|:---:|:---:|:---:|
| **0.1%** |  | 47.27% (26/55) | 0.0823% (35/42,503) |  |
| **1.0%** |  | 80.00% (44/55) | 0.8423% (358/42,503) |  |
| **5.0%** |  | 83.64% (46/55) | 2.7292% (1,160/42,503) |  |

*(Note: In accordance with project governance, operational threshold values originate exclusively from executed, committed calibration manifests).*

Source: [docs/threshold_manifest_schema.md](../docs/threshold_manifest_schema.md#L1-L44)

---

## 8. INTENDED USE

- **Primary Application:** Fraud-review prioritization aid (alert triage). The model produces continuous anomaly scores used to rank incoming transactions into a prioritized review queue (top-$K$ review triage queue) for human fraud investigation teams.
- **Decision Workflow:** High-scoring transactions are routed to human analysts who examine context, customer history, and auxiliary transaction records.
- **Human-in-the-Loop Requirement:** Human analyst review is strictly required before taking any adverse operational action against a transaction or customer account.

Source: [docs/ae_experiment_plan.md](../docs/ae_experiment_plan.md#L108-L146)  
Source: [docs/autoencoder_methods_note.md](../docs/autoencoder_methods_note.md#L8-L26)

---

## 9. PROHIBITED INTERPRETATIONS

To prevent operational harm and regulatory non-compliance, the following interpretations and deployments are strictly prohibited:

1. **Not an Autonomous Decision System:** The model is not an automated transaction approval/rejection engine.
2. **Score Is Not Proof of Fraud:** Reconstruction error quantifies out-of-distribution geometric deviation from the learned legitimate manifold; a high score is not definitive proof or legal evidence of fraudulent intent.
3. **No Unilateral Adverse Action:** The anomaly score must not be used alone to automatically decline card authorizations, initiate chargebacks, cancel cards, or freeze/close user accounts without human analyst verification.
4. **No Direct Probability Interpretation:** Raw MSE reconstruction error is an uncalibrated geometric distance metric, not a posterior probability of fraud $P(\text{Class}=1 \mid \mathbf{x})$.

Source: [docs/ae_experiment_plan.md](../docs/ae_experiment_plan.md#L182-L197)  
Source: [docs/autoencoder_methods_note.md](../docs/autoencoder_methods_note.md#L95-L100)

---

## 10. KNOWN LIMITATIONS

The model possesses six documented limitations that must be accounted for in system integration and evaluation:

1. **Reconstruction False Negatives on Subtle Frauds:** Some fraudulent transactions have feature signatures closely aligned with legitimate spending patterns and reconstruct with low MSE error. The autoencoder measures distance from learned legitimate geometry, not fraudulent intent.  
   *Source:* [docs/autoencoder_methods_note.md](../docs/autoencoder_methods_note.md#L95-L100), Section 5 (Limitations & Failure Modes)
2. **Validation Split Prior Phase-1 Exposure:** The validation split was previously exposed during Phase 1 for exploratory threshold tuning and triage analysis. This prior exposure is disclosed in the pre-registered plan; the final holdout test evaluation in Issue #38 serves as the untainted final arbiter.  
   *Source:* [docs/ae_experiment_plan.md](../docs/ae_experiment_plan.md#L190-L193)
3. **Seed Variability:** Across random training initializations, validation AP exhibits a measured standard deviation of $\pm 0.0147$ ($\text{CV} = 4.5\%$, range $0.3116$ to $0.3409$). This variability is transparently reported and managed by freezing a confirmed closest-to-mean checkpoint.  
   *Source:* [results/ae_seed_stability.md](../results/ae_seed_stability.md#L47-L58)
4. **Dataset & Issuer Scope:** The model was trained and evaluated on transactions from September 2013 by a single European card issuer (ULB dataset). The frozen dataset's Time range spans 0 to 172,792 seconds (~2 days), per the split manifest. Generalization to other card products, merchant categories, geographical regions, or newer payment flows is untested.  
   *Source:* [data/split_manifest.json](../data/split_manifest.json) (time_range_seconds)  
   *Source:* [docs/Phase2_Data_Provenance_Split.md](../docs/Phase2_Data_Provenance_Split.md#L1-L15)  
   *Source:* [docs/ae_experiment_plan.md](../docs/ae_experiment_plan.md#L10-L18)
5. **Validation Ranking Trails Baseline GMM (Resolved in Paired Evaluation):** On the shared validation split, the frozen autoencoder's validation AP ($0.3257$) trailed the reproduced Phase 1 GMM baseline validation AP ($0.6510$). In the final paired holdout test evaluation (Issue #38), GMM demonstrated superior full-ranking AP ($0.6773$ vs. $0.4867$), whereas the autoencoder achieved higher precision and caught 2 additional frauds in the top-100 review queue ($40$ vs. $38$ caught frauds, $60$ vs. $62$ false alarms). At 1,000 reviews, both systems caught 41 frauds. Model selection in practice therefore depends on operational review capacity and workload budgets.  
   *Source:* [results/phase2/paired/ranking.csv](../results/phase2/paired/ranking.csv)  
   *Source:* [results/phase2/paired/triage.csv](../results/phase2/paired/triage.csv)  
   *Source:* [results/ae_seed_stability.md](../results/ae_seed_stability.md#L48)
6. **Realized Hidden Activation Stack (Addendum A1):** The realized network architecture uses `BatchNorm1d` + `LeakyReLU(negative_slope=0.1)` rather than the plain `ReLU` specified in the initial pre-registration document (Section 2), reflecting the team-merged implementation (#32). All experimental candidates shared this identical realized backbone.  
   *Source:* [docs/ae_experiment_plan.md](../docs/ae_experiment_plan.md#L210-L214)

---

## 11. EVIDENCE INDEX

All artifacts supporting this model card are committed in the repository:

### Model Manifest & Run Log
| Artifact Type | Repository Path | Description |
|---|---|---|
| **Authoritative Manifest** | [`results/ae_frozen_model_manifest.json`](../results/ae_frozen_model_manifest.json) | Complete JSON identity, hyperparameters, metrics, and checksums |
| **Committed Run Log** | [`results/phase2/ae_run_log.csv`](../results/phase2/ae_run_log.csv) | 18-column CSV recording all 8 Stage 1 and Stage 2 runs |

### Model Checkpoints (8 Committed Artifacts)
| Run ID | Configuration | Checkpoint Path | SHA-256 Checksum |
|---|---|---|---|
| `RUN-S1-W04-CLN-S42` | $B=4$, Clean, Seed 42 | [`results/phase2/checkpoints/RUN-S1-W04-CLN-S42.pt`](../results/phase2/checkpoints/RUN-S1-W04-CLN-S42.pt) | `42dfef99c824352a2b1b4874649abadc2f3d1fbdd64060befee71f8657d192e1` |
| `RUN-S1-W08-CLN-S42` | $B=8$, Clean, Seed 42 | [`results/phase2/checkpoints/RUN-S1-W08-CLN-S42.pt`](../results/phase2/checkpoints/RUN-S1-W08-CLN-S42.pt) | `3ec564bc495016fd21b188e7a1aa083d036b092307cf63f6d00ec405456a7554` |
| `RUN-S1-W16-CLN-S42` | $B=16$, Clean, Seed 42 | [`results/phase2/checkpoints/RUN-S1-W16-CLN-S42.pt`](../results/phase2/checkpoints/RUN-S1-W16-CLN-S42.pt) | `f07945797e17a6772499034d6d8236b9b684041e35ae6f19819a582dc186ac09` |
| `RUN-S1-W16-DNZ-S42` | $B=16$, Denoise, Seed 42 | [`results/phase2/checkpoints/RUN-S1-W16-DNZ-S42.pt`](../results/phase2/checkpoints/RUN-S1-W16-DNZ-S42.pt) | `9ea1c6766bf0678f39dbb573cf9889a3ba956e6204dda66365a169519ee26ade` |
| `RUN-S2-W16-DNZ-S0` | $B=16$, Denoise, Seed 0 | [`results/phase2/checkpoints/RUN-S2-W16-DNZ-S0.pt`](../results/phase2/checkpoints/RUN-S2-W16-DNZ-S0.pt) | `1cd885cec1ed00f669d514b85d55b20c48e014384f52800378f4e04d0c2ee769` |
| **`RUN-S2-W16-DNZ-S7`** | **$B=16$, Denoise, Seed 7 (FROZEN)** | [`results/phase2/checkpoints/RUN-S2-W16-DNZ-S7.pt`](../results/phase2/checkpoints/RUN-S2-W16-DNZ-S7.pt) | **`0d247ab9757cd3b15caeaba0049dd7825c8049998dd57a649307c87c64b4a1fb`** |
| `RUN-S2-W08-CLN-S0` | $B=8$, Clean, Seed 0 | [`results/phase2/checkpoints/RUN-S2-W08-CLN-S0.pt`](../results/phase2/checkpoints/RUN-S2-W08-CLN-S0.pt) | `374cc5af40fe697ac4f85c775e46e4a63636a08efd461d4fda079de33cf9c492` |
| `RUN-S2-W08-CLN-S7` | $B=8$, Clean, Seed 7 | [`results/phase2/checkpoints/RUN-S2-W08-CLN-S7.pt`](../results/phase2/checkpoints/RUN-S2-W08-CLN-S7.pt) | `4d4fcbdaf724b7c52882840b780c9a70806716d310e1c4a9ea363fafeff9b11e` |

### Validation Score Files (8 Committed Artifacts)
| Run ID | Score File Path | Line Count | Schema |
|---|---|:---:|---|
| `RUN-S1-W04-CLN-S42` | [`results/phase2/scores/RUN-S1-W04-CLN-S42_scores_val.csv`](../results/phase2/scores/RUN-S1-W04-CLN-S42_scores_val.csv) | 42,559 | `row_id,split,model,anomaly_score` |
| `RUN-S1-W08-CLN-S42` | [`results/phase2/scores/RUN-S1-W08-CLN-S42_scores_val.csv`](../results/phase2/scores/RUN-S1-W08-CLN-S42_scores_val.csv) | 42,559 | `row_id,split,model,anomaly_score` |
| `RUN-S1-W16-CLN-S42` | [`results/phase2/scores/RUN-S1-W16-CLN-S42_scores_val.csv`](../results/phase2/scores/RUN-S1-W16-CLN-S42_scores_val.csv) | 42,559 | `row_id,split,model,anomaly_score` |
| `RUN-S1-W16-DNZ-S42` | [`results/phase2/scores/RUN-S1-W16-DNZ-S42_scores_val.csv`](../results/phase2/scores/RUN-S1-W16-DNZ-S42_scores_val.csv) | 42,559 | `row_id,split,model,anomaly_score` |
| `RUN-S2-W16-DNZ-S0` | [`results/phase2/scores/RUN-S2-W16-DNZ-S0_scores_val.csv`](../results/phase2/scores/RUN-S2-W16-DNZ-S0_scores_val.csv) | 42,559 | `row_id,split,model,anomaly_score` |
| **`RUN-S2-W16-DNZ-S7`** | [`results/phase2/scores/RUN-S2-W16-DNZ-S7_scores_val.csv`](../results/phase2/scores/RUN-S2-W16-DNZ-S7_scores_val.csv) | **42,559** | `row_id,split,model,anomaly_score` |
| `RUN-S2-W08-CLN-S0` | [`results/phase2/scores/RUN-S2-W08-CLN-S0_scores_val.csv`](../results/phase2/scores/RUN-S2-W08-CLN-S0_scores_val.csv) | 42,559 | `row_id,split,model,anomaly_score` |
| `RUN-S2-W08-CLN-S7` | [`results/phase2/scores/RUN-S2-W08-CLN-S7_scores_val.csv`](../results/phase2/scores/RUN-S2-W08-CLN-S7_scores_val.csv) | 42,559 | `row_id,split,model,anomaly_score` |

### Documentation & Visual Artifacts
| Artifact | Repository Path | Description |
|---|---|---|
| **Frozen Experiment Plan** | [`docs/ae_experiment_plan.md`](../docs/ae_experiment_plan.md) | Pre-registered experiment protocol and selection rules (includes Addendum A1) |
| **Stage 1 Ablation Doc** | [`results/ae_ablations_stage1.md`](../results/ae_ablations_stage1.md) | Empirical analysis and rule application for Stage 1 width/denoising search |
| **Stage 1 Ablation Figure** | [`results/figures/ae_stage1_val_ap.png`](../results/figures/ae_stage1_val_ap.png) | Stage-1 ablation plot: validation AP across candidate architectures |
| **Stage 2 Stability Doc** | [`results/ae_seed_stability.md`](../results/ae_seed_stability.md) | Multi-seed stability analysis, audit, and model freeze decision |

---

## 12. VERSION & SIGN-OFF

| Attribute | Specification |
|---|---|
| **Model Card Version** | 1.0 |
| **Release Date** | 2026-11-17 |
| **Author** | Abhigya Sachdeva (`@ABHIGYA0205`) |
| **Role** | Anomaly Detection Modeling Lead |
| **Downstream Consumer** | Issue #62 — AE validation threshold calibration (@WizardCarp85) |
| **Final Evaluation** | Issue #38 — Paired GMM vs Autoencoder holdout test evaluation (@somraj112) |
