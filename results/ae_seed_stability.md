# Stage 2 Autoencoder Seed Stability Analysis & Final Checkpoint Freeze

**Document Status:** Complete & Frozen  
**Author:** Abhigya Sachdeva (`@Abhi`)  
**Milestone:** Phase 2 Anomaly Detection Exploration (Issue #36)  
**Governing Plan:** [docs/ae_experiment_plan.md](file:///Users/Abhi/Downloads/projects/Card-Fraud-Triage-ML-DL/docs/ae_experiment_plan.md) (Section 4 & Section 5.4)  
**Frozen Manifest:** [results/ae_frozen_model_manifest.json](file:///Users/Abhi/Downloads/projects/Card-Fraud-Triage-ML-DL/results/ae_frozen_model_manifest.json)  
**Committed Run Log:** [results/phase2/ae_run_log.csv](file:///Users/Abhi/Downloads/projects/Card-Fraud-Triage-ML-DL/results/phase2/ae_run_log.csv)  

---

## 1. Executive Summary & Objective

In accordance with Stage 2 of the frozen Phase 2 experimental plan ([docs/ae_experiment_plan.md](file:///Users/Abhi/Downloads/projects/Card-Fraud-Triage-ML-DL/docs/ae_experiment_plan.md)), we evaluated multi-seed stability across seeds $\{0, 7, 42\}$ for:
1. The Stage-1 rule-based champion: **`W16-DNZ`** (bottleneck $B=16$, training input denoising $\sigma = 0.5$, clean targets).
2. The Stage-1 reference baseline: **`W08-CLN`** (bottleneck $B=8$, clean training inputs).

Per plan Section 4 ("seed 42 reused/re-logged"), the two seed-42 runs from Stage 1 were reused without re-training. Four new runs were executed under identical settings for seeds $\{0, 7\}$ to form a balanced 6-run confirmation matrix.

### Key Outcomes:
- **Seed Stability:** Neither configuration exhibited instability. All individual seed runs fell well within the pre-registered stability tolerance ($|\text{val\_AP}(\text{seed}) - \overline{\text{val\_AP}}| \le 0.10$).
- **Configuration Winner:** `W16-DNZ` outperformed the reference baseline `W08-CLN` by **$+0.1541$ mean validation AP** ($0.3257$ vs. $0.1716$, $+89.8\%$ relative improvement), winning definitively under Section 5.4.
- **Frozen Production Checkpoint:** Under the pre-registered closest-to-mean policy, **`RUN-S2-W16-DNZ-S7`** (seed 7, $\text{val\_AP} = 0.3245$, distance to mean = $0.001165$) is frozen as the authoritative Phase 2 Autoencoder model.
- **Strict Protocol Compliance:** Training used legitimate rows only ($N = 198,242$, 0 frauds). Model selection and stability measurement used **validation data only** ($N = 42,558$, 55 frauds). Zero holdout test transactions or labels were accessed, computed, or saved.

---

## 2. Full 6-Run Stability Evaluation Table

All 6 runs are recorded in [results/phase2/ae_run_log.csv](file:///Users/Abhi/Downloads/projects/Card-Fraud-Triage-ML-DL/results/phase2/ae_run_log.csv). The table below marks the two reused Stage-1 seed-42 runs alongside the four Stage-2 runs:

| Run ID | Configuration | Seed | Status | Epochs (Best/Total) | Final Train Loss | Best Val Recon Loss | Validation AP (PR-AUC) | Validation P@100 | Val Recall @ 1% FPR | Val Recall @ 5% FPR | Runtime (s) | Checkpoint SHA-256 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **`RUN-S2-W16-DNZ-S0`** | $B=16$, Denoise $\sigma=0.5$ | 0 | Stage 2 Run | 53 / 63 | 0.240899 | 0.174538 | **0.311635** | 0.3900 | 0.781818 | 0.836364 | 48.02 | `1cd885cec1ed00f6...` |
| **`RUN-S2-W16-DNZ-S7`** | $B=16$, Denoise $\sigma=0.5$ | 7 | **FROZEN WINNER** | 90 / 100 | 0.218740 | 0.134965 | **0.324527** | 0.3900 | 0.800000 | 0.836364 | 73.83 | `0d247ab9757cd3b1...` |
| **`RUN-S1-W16-DNZ-S42`** | $B=16$, Denoise $\sigma=0.5$ | 42 | *Reused Stage 1* | 99 / 100 | 0.206413 | 0.126785 | **0.340914** | 0.3900 | 0.781818 | 0.818182 | 76.32 | `9ea1c6766bf0678f...` |
| **`RUN-S2-W08-CLN-S0`** | $B=8$, Clean Ref | 0 | Stage 2 Run | 97 / 100 | 0.290377 | 0.316788 | **0.205460** | 0.2700 | 0.763636 | 0.818182 | 70.91 | `374cc5af40fe697a...` |
| **`RUN-S2-W08-CLN-S7`** | $B=8$, Clean Ref | 7 | Stage 2 Run | 90 / 100 | 0.279171 | 0.314058 | **0.177996** | 0.2600 | 0.800000 | 0.818182 | 72.99 | `4d4fcbdaf724b7c5...` |
| **`RUN-S1-W08-CLN-S42`** | $B=8$, Clean Ref | 42 | *Reused Stage 1* | 99 / 100 | 0.279307 | 0.301306 | **0.131272** | 0.2100 | 0.763636 | 0.836364 | 75.40 | `3ec564bc495016fd...` |

---

## 3. Stability Statistics & Instability Audit

Across the three evaluation seeds $\{0, 7, 42\}$, summary statistics are computed using sample standard deviation ($N-1$ degrees of freedom):

### Configuration 1: `W16-DNZ` ($B=16$, Denoise $\sigma=0.5$)
- **Mean Validation AP:** $\mathbf{0.325692 \pm 0.014674}$ ($\text{CV} = 4.5\%$)
- **Mean Validation Precision@100:** $\mathbf{0.390000 \pm 0.000000}$ (Perfect invariance: exactly 39 frauds in top-100 on every seed)
- **Mean Validation Recall @ 1% FPR:** $0.787879 \pm 0.010497$
- **Mean Validation Recall @ 5% FPR:** $0.830303 \pm 0.010497$
- **Mean Runtime:** $66.06\text{ s}$
- **Instability Audit:**
  - Seed 0 deviation: $|0.311635 - 0.325692| = 0.014057 \le 0.10$
  - Seed 7 deviation: $|0.324527 - 0.325692| = 0.001165 \le 0.10$
  - Seed 42 deviation: $|0.340914 - 0.325692| = 0.015222 \le 0.10$
  - **Verdict:** **STABLE** (Zero unstable seeds).

### Configuration 2: `W08-CLN` ($B=8$, Clean Reference Baseline)
- **Mean Validation AP:** $\mathbf{0.171576 \pm 0.037512}$ ($\text{CV} = 21.9\%$)
- **Mean Validation Precision@100:** $\mathbf{0.246667 \pm 0.032146}$
- **Mean Validation Recall @ 1% FPR:** $0.775758 \pm 0.021001$
- **Mean Validation Recall @ 5% FPR:** $0.824242 \pm 0.010497$
- **Mean Runtime:** $73.10\text{ s}$
- **Instability Audit:**
  - Seed 0 deviation: $|0.205460 - 0.171576| = 0.033884 \le 0.10$
  - Seed 7 deviation: $|0.177996 - 0.171576| = 0.006420 \le 0.10$
  - Seed 42 deviation: $|0.131272 - 0.171576| = 0.040304 \le 0.10$
  - **Verdict:** **STABLE** (Zero unstable seeds).

---

## 4. Selection Rule Application & Checkpoint Freeze

### Configuration Selection (Section 5.4)
The plan requires selecting the final configuration by **mean validation AP** across the 3 confirmation seeds:
$$\Delta \overline{\text{AP}} = \overline{\text{AP}}(\text{W16-DNZ}) - \overline{\text{AP}}(\text{W08-CLN}) = 0.325692 - 0.171576 = \mathbf{+0.154116}$$

Since $\Delta \overline{\text{AP}} = +0.1541 \ge 0.005$, **`W16-DNZ`** wins decisively on the primary mean AP criterion:
- Selection by the single best seed was strictly prohibited.
- Tie-break logic was not invoked because the difference vastly exceeds the $0.005$ indifference threshold.

### Checkpoint Freeze (Closest-to-Mean Policy)
To select the single production checkpoint that best reflects expected operational behavior:
1. Seed 0 ($0.311635$): $|\text{AP} - \text{mean}| = 0.014057$
2. Seed 7 ($0.324527$): $|\text{AP} - \text{mean}| = \mathbf{0.001165}$ (**Closest to Mean**)
3. Seed 42 ($0.340914$): $|\text{AP} - \text{mean}| = 0.015222$

Seed 7 is closest to the 3-seed mean ($0.001165$ difference). Therefore, **`RUN-S2-W16-DNZ-S7`** is frozen as the authoritative Phase 2 Autoencoder checkpoint.

---

## 5. Continuity with Addendum A1

This stage strictly continues the architectural specification established in **Addendum A1** of [docs/ae_experiment_plan.md](file:///Users/Abhi/Downloads/projects/Card-Fraud-Triage-ML-DL/docs/ae_experiment_plan.md):
- Realized backbone topology: $29 \to 32 \to B \to 32 \to 29$ ($B=16$ for champion, $B=8$ for baseline).
- Activation stack: `BatchNorm1d` + `LeakyReLU(negative_slope=0.1)`.
- Optimizer: Adam ($\text{LR} = 1\times 10^{-3}$, weight decay $= 0$, no learning rate scheduler).
- Loss & Scoring: Per-sample MSE reconstruction error across all 29 scaled features.
- Early Stopping: Monitored on legitimate validation MSE loss with patience 10.

All 6 confirmation runs utilized this identical realized backbone.

---

## 6. Handover to Downstream Evaluator (Somraj — Issue #37)

> **To: Somraj (`@Somraj`), Downstream Evaluation & Triage Calibration Lead**  
> **From: Abhigya Sachdeva (`@Abhi`), Anomaly Detection Modeling Lead**  
> **Date: 2026-11-17**  
> **Subject: Handover of Frozen Phase 2 Autoencoder Checkpoint for Threshold Calibration (Issue #37)**  
>
> The Phase 2 Autoencoder model search and multi-seed confirmation are formally complete. Model selection was governed strictly by validation Average Precision across seeds $\{0, 7, 42\}$, and the winning configuration (`W16-DNZ`, bottleneck dimension 16, input denoising $\sigma = 0.5$) achieved a mean validation AP of **$0.3257 \pm 0.0147$** with zero seed instability.
>
> In accordance with the pre-registered closest-to-mean policy, the following checkpoint is formally frozen and ready for your operational threshold calibration in Issue #37:
> - **Model Identifier:** `Phase2_Frozen_Undercomplete_Autoencoder`
> - **Run ID:** `RUN-S2-W16-DNZ-S7`
> - **Checkpoint Path:** [`results/phase2/checkpoints/RUN-S2-W16-DNZ-S7.pt`](file:///Users/Abhi/Downloads/projects/Card-Fraud-Triage-ML-DL/results/phase2/checkpoints/RUN-S2-W16-DNZ-S7.pt)
> - **Checkpoint SHA-256:** `0d247ab9757cd3b15caeaba0049dd7825c8049998dd57a649307c87c64b4a1fb`
> - **Validation Score File:** [`results/phase2/scores/RUN-S2-W16-DNZ-S7_scores_val.csv`](file:///Users/Abhi/Downloads/projects/Card-Fraud-Triage-ML-DL/results/phase2/scores/RUN-S2-W16-DNZ-S7_scores_val.csv) (42,559 lines, schema: `row_id,split,model,anomaly_score`)
> - **Score Contract:** Per-row MSE reconstruction error over the 29 features (`score_direction="higher_is_more_suspicious"`).
> - **Authoritative Manifest:** Full hyperparameter, architecture, and cryptographic provenance specifications are recorded in [`results/ae_frozen_model_manifest.json`](file:///Users/Abhi/Downloads/projects/Card-Fraud-Triage-ML-DL/results/ae_frozen_model_manifest.json).
>
> Please use this frozen model and its validation scores for calibrating FPR budgets ($0.1\%$, $1.0\%$, $5.0\%$) and Top-$K$ capacities ($100, 500, 1000$). Per Golden Rule 3, the holdout test set remains strictly quarantined until final paired evaluation in Issue #38.
