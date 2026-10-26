# Project 26: Card Fraud Triage
## Unsupervised Anomaly Detection & Density Scoring via Gaussian Mixture Models (Phase 1)

### Technical Project Documentation

---

## 1. Executive Summary & Problem Formulation

In modern financial payment rails, transactional volume reaches millions of events per day, while fraudulent transactions constitute an extreme minority (e.g., 0.172% in benchmark data). 

Under extreme class imbalance:
- Standard supervised classification models optimizing cross-entropy or accuracy collapse toward trivial majority-class predictions (e.g., predicting all transactions as legitimate yields 99.83% raw accuracy while detecting zero frauds).
- Labeled fraud instances are scarce, delayed in arrival, and non-exhaustive with respect to adversarial fraud patterns.

### The Card Fraud Triage Paradigm
Rather than modeling the volatile patterns of fraudulent transactions, this project applies **unsupervised anomaly detection** focused exclusively on learning the probability density distribution of verified legitimate card behavior:
$$\mathcal{M}: \mathbb{R}^D \to \mathbb{R}^+ \quad \text{where} \quad \mathcal{M}(x) \approx p(x \mid \text{legitimate})$$

Transactions with low estimated probability density correspond to anomalous behavior. By ranking transactions in descending order of an anomaly score:
$$\text{Score}(x) = -\ln p(x \mid \hat{\Theta})$$
the model produces a **capacity-constrained triage queue** for fraud investigation teams. This enables security analysts to inspect high-risk transactions under fixed operational review budgets.

---

## 2. Team Members & Responsibilities

| Member | Enrollment No. | Institutional Email | Project Role & Primary Ownership |
|---|---|---|---|
| **Yashkumar Nimje** | `2401020113` | `yashkumar.nimje2024@nst.rishihood.edu.in` | **Team Lead & Data Architecture**<br>Repository structure, dataset ingestion, exploratory data analysis (EDA), chronological split design |
| **Saksham Narotra** | `2401010402` | `saksham.narotra2024@nst.rishihood.edu.in` | **Feature Engineering & Preprocessing**<br>Data hygiene, logarithmic skew correction, robust feature scaling, leakage prevention |
| **Rudra Pratap Singh Choudhary** | `2401010394` | `rudra.choudhary2024@nst.rishihood.edu.in` | **Probabilistic Modeling (GMM Core)**<br>GMM mathematical formulation, Expectation-Maximization (EM) implementation, density scoring engine |
| **Abhigya Sachdeva** | `2401020003` | `abhigya.sachdeva2024@nst.rishihood.edu.in` | **Model Selection & Robustness**<br>AIC/BIC model selection, covariance structure ablations, dimensionality effects, seed stability checks |
| **Somraj Nandi** | `2401020070` | `somraj.nandi2024@nst.rishihood.edu.in` | **Evaluation Framework & Decision Systems**<br>Precision-Recall AUC, Recall@FPR budgets, Precision@K review triage tables, technical report assembly |

### Module Ownership & Deliverables

#### 1. Yashkumar Nimje — Team Lead & Data Architecture
- **Version Control & Repository Architecture:** Maintain GitHub repository structure, documentation integrity, and clean commit standards.
- **Dataset Ingestion & Validation:** Verify checksums, data types, and null-value absence on the ULB dataset.
- **Exploratory Data Analysis (EDA):** Quantify class imbalance (0.172% base rate), distribution properties of `Time` and `Amount`, PCA feature correlations, and discriminatory separation between normal and anomalous observations.
- **Chronological Data Partitioning:** Implement strict temporal ordering based on transaction timestamp to mimic production deployment without lookahead bias.

#### 2. Saksham Narotra — Feature Engineering & Preprocessing
- **Feature Hygiene:** Duplicate detection strategy and extreme outlier assessment.
- **Domain Transformations:** Apply $\log(1 + x)$ transformation on `Amount` to mitigate heavy right-skewness; evaluate temporal hour-of-day features.
- **Scaler Standardization:** Formulate and validate `StandardScaler` versus `RobustScaler` behavior on PCA and transformed features.
- **Leakage Prevention:** Enforce strict containment guarantees (scalers fit exclusively on legitimate training samples).

#### 3. Rudra Pratap Singh Choudhary — Probabilistic Modeling (GMM Core)
- **Theoretical Formulation:** Formulate the Gaussian Mixture Model density function and the Expectation-Maximization (EM) algorithm (Dempster et al., 1977).
- **Model Training Engine:** Fit GMM parameters ($\pi_k, \mu_k, \Sigma_k$) exclusively on the legitimate training distribution.
- **Scoring Pipeline:** Vectorized extraction of negative log-likelihood scores:
  $$s(x) = -\ln \left( \sum_{k=1}^K \pi_k \mathcal{N}(x; \mu_k, \Sigma_k) \right)$$
- **Numerical Conditioning:** Prevent singular covariance matrices via regularized covariance diagonal offsets (`reg_covar`).

#### 4. Abhigya Sachdeva — Model Selection & Robustness
- **Hyperparameter Optimization:** Model selection across component counts ($k \in [2, 15]$) using Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC).
- **Covariance Matrix Topology:** Comparative empirical analysis across `full`, `diag`, `tied`, and `spherical` covariance structures.
- **Ablation Studies:** Measure sensitivity across feature sets, PCA dimensionality reductions, and initialization seeds.
- **Convergence Validation:** Audit EM log-likelihood convergence tolerances and iteration ceilings.

#### 5. Somraj Nandi — Evaluation Framework & Decision Systems
- **Evaluation Pipeline:** Standardized calculation of PR-AUC, ROC-AUC, Recall@FPR budgets (0.1%, 1.0%, 5.0%), and Precision@$K$ ($K \in \{100, 500, 1000\}$).
- **Triage Decision Engineering:** Threshold calibration performed strictly on validation sets and evaluated on unseen test data.
- **Operational Trade-Off Modeling:** Quantify human analyst workload constraints vs. caught fraud revenue preservation.
- **Technical Reporting:** Consolidate experimental artifacts, comparative tables, and visualizations into publication-grade documentation.

---

## 3. Dataset Specification

The project utilizes the benchmark **Credit Card Fraud Detection Dataset (ULB / Kaggle)**, representing anonymized European cardholder transactions over a two-day observation window (September 2013).

### Key Dataset Statistics
- **Total Transactions:** 284,807
- **Legitimate Transactions:** 284,315 (99.828%)
- **Fraudulent Transactions:** 492 (0.172%, Base Rate: $\approx 1 : 578$)
- **Dimensionality:** 31 numeric fields (`Time`, `V1`–`V28`, `Amount`, `Class`)

### Feature Architecture

| Column Identifier | Type | Representation | Preprocessing Requirement |
|---|---|---|---|
| `Time` | Float | Seconds elapsed relative to dataset start | Chronological sort index; cyclical hour extraction |
| `V1` – `V28` | Float | Orthogonal principal components from PCA | Maintained as continuous coordinates; robust scaling |
| `Amount` | Float | Transaction monetary value | Logarithmic transform: $\log_{1p}(x)$, followed by scaling |
| `Class` | Integer | Binary Ground Truth (`0`: Legitimate, `1`: Fraud) | Supervised evaluation label only; masked during model fitting |

---

## 4. Methodological Principles & Leakage Prevention

Unsupervised anomaly detection requires strict experimental isolation to prevent data snooping and information leakage.

```
Full Chronological Data (284,807 rows)
  │
  ├── [Earliest ~70%] Training Split (Legitimate only: Class == 0)
  │      └── Fit Scaler & Train GMM
  │
  ├── [Middle ~15%] Validation Split (Legitimate + Fraud)
  │      └── Apply Scaler, Calibrate Thresholds, Tune k & Covariance
  │
  └── [Latest ~15%] Test Split (Legitimate + Fraud — Untouched)
         └── Apply Scaler, Compute Final Metrics Once
```

### Core Experimental Constraints
1. **Unsupervised Purity:** The GMM is trained exclusively on normal (legitimate) transactions ($\text{Class} = 0$). Fraud records never enter the training phase.
2. **Scaler Isolation:** Feature scalers are fitted strictly on the legitimate training subset and subsequently applied to transform validation and test subsets.
3. **Chronological Splitting:** Transactions are strictly ordered by `Time` into Train (first 70%), Validation (subsequent 15%), and Test (final 15%) partitions to replicate production temporal dynamics.
4. **Validation-Only Calibration:** All operational triage thresholds and hyperparameter selections (optimal $k$, covariance type) are finalized on the validation split.
5. **Untouched Test Benchmark:** The test set is evaluated exactly once using frozen preprocessing pipelines and thresholds.
6. **Imbalance-Aware Metrics:** Traditional classification accuracy is disallowed as a primary evaluation criterion; performance is reported through Precision-Recall AUC (PR-AUC), operational Recall at fixed False Positive Rates, and Precision at fixed review capacities.

---

## 5. Machine Learning Architecture: Gaussian Mixture Model (GMM)

### 5.1 Probabilistic Density Estimation
A Gaussian Mixture Model models the probability density distribution of legitimate transactions as a weighted sum of $K$ multivariate Gaussian components:
$$p(x \mid \Theta) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$$
subject to $\sum_{k=1}^K \pi_k = 1$ and $\pi_k \ge 0$, where:
- $\pi_k$: Mixture weight of component $k$
- $\mu_k \in \mathbb{R}^D$: Mean vector of component $k$
- $\Sigma_k \in \mathbb{R}^{D \times D}$: Covariance matrix of component $k$

### 5.2 The Expectation-Maximization (EM) Optimization
Parameters $\Theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K$ are estimated via maximum likelihood using the Expectation-Maximization algorithm (Dempster, Laird, & Rubin, 1977):

1. **Expectation Step (E-step):** Compute posterior responsibilities $\gamma_{ik}$ that component $k$ generated observation $x_i$:
   $$\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$$

2. **Maximization Step (M-step):** Update mixture parameters using sample-weighted expectations:
   $$N_k = \sum_{i=1}^N \gamma_{ik}$$
   $$\pi_k^{\text{new}} = \frac{N_k}{N}, \quad \mu_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} x_i$$
   $$\Sigma_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} (x_i - \mu_k^{\text{new}})(x_i - \mu_k^{\text{new}})^T + \epsilon I$$
   where $\epsilon I$ (`reg_covar`) guarantees positive definiteness and avoids covariance singularity.

### 5.3 Anomaly Scoring Function
The anomaly score for any observation $x$ is defined as the negative log-likelihood under the fitted density:
$$\text{Score}(x) = -\ln \left( \sum_{k=1}^K \hat{\pi}_k \mathcal{N}(x \mid \hat{\mu}_k, \hat{\Sigma}_k) \right)$$
Observations falling outside the high-density regions of legitimate behavior yield small likelihoods, resulting in high anomaly scores.

### 5.4 Model Selection (AIC & BIC)
Optimal component complexity is determined by minimizing information criteria balancing goodness-of-fit against model parameterization:
$$\text{AIC} = 2P - 2\ln \hat{L}$$
$$\text{BIC} = P \ln N - 2\ln \hat{L}$$
where $P$ denotes the number of freely estimated parameters and $\hat{L}$ is the maximized likelihood. BIC applies a stronger penalty on parameter growth, preventing overfitting in multi-component high-dimensional regimes.

---

## 6. Evaluation Framework & Operational Triage Metrics

Due to the extreme rarity of fraud, evaluation centers on precision-recall dynamics and operational triage constraints.

| Metric | Formulation / Definition | Operational Purpose |
|---|---|---|
| **PR-AUC** | $\int_0^1 P(R) \, dR$ (Area Under Precision-Recall Curve) | **Primary Metric:** Evaluates detection precision across all recall thresholds without inflation from true negatives. |
| **ROC-AUC** | $\int_0^1 \text{TPR}(\text{FPR}) \, d\text{FPR}$ | Secondary baseline; susceptible to over-optimism under low base rates. |
| **Recall @ FPR $\alpha$** | $\text{TPR}$ when $\text{FPR} \le \alpha \quad (\alpha \in \{0.1\%, 1.0\%, 5.0\%\})$ | Quantifies fraud capture rate when false alarm volume is capped by a strict operational budget. |
| **Precision @ Top-$K$** | $\frac{\text{True Frauds in Top-}K}{K} \quad (K \in \{100, 500, 1000\})$ | Measures hit-rate efficiency for analyst queues under fixed daily investigation capacities. |

### Operational Triage Decision Table Structure
Thresholds are set on the validation set at fixed review capacity constraints and evaluated on the holdout test set:

$$\text{Review Capacity } (K) \implies \text{Fraud Caught (Recall)} \quad \vert \quad \text{Precision} \quad \vert \quad \text{False Positive Alarms}$$

---

## 7. Technical Challenges & Engineering Mitigations

| Technical Challenge | Root Cause | Engineering Solution |
|---|---|---|
| **Covariance Singularity** | Zero-variance sub-spaces or collinearities cause $|\Sigma_k| \to 0$, causing log-likelihood divergence. | Apply covariance regularization (`reg_covar = 1e-06` to `1e-04`) along the diagonal; evaluate diagonal covariance structures (`covariance_type='diag'`). |
| **Curse of Dimensionality** | In 30 dimensions, distances concentrate, causing densities in the outer tails to become vanishingly small. | Comparative ablation of full 30-D feature space versus PCA-reduced latent spaces (10–20 components); robust feature scaling. |
| **Local Optima in EM** | Non-convex log-likelihood surface with sensitivity to parameter initialization. | Run multiple initializations with fixed global seeds (`SEED = 42`); verify stability across independent seed draws. |
| **Data Leakage Risk** | Threshold selection or scaling parameters informed by fraud or test distributions. | Strictly modularized sklearn pipelines; fit transforms solely on $\mathcal{X}_{\text{train, legit}}$; validate threshold selection on $\mathcal{X}_{\text{val}}$. |

---

## 8. Repository Structure

```
card-fraud-triage-project26/
├── README.md                                # Project overview, quickstart, and configuration
├── notebooks/
│   ├── Card_Fraud_Triage_Phase1.ipynb       # Master end-to-end reproducible notebook
│   └── experiments/                         # Exploratory and ablation experiment notebooks
├── reports/
│   ├── Phase1_Report.pdf                    # Formal technical report (LaTeX/IEEE format)
│   ├── Dempster_1977_EM_Summary.md          # Theoretical derivation & literature notes
│   └── Member3_Report_Sections.md           # Model methodology and mathematical write-ups
├── presentation/
│   └── Phase1_Slides.pdf                    # Project review and viva presentation slide deck
├── results/
│   ├── figures/                             # Exported PR curves, AIC/BIC plots, EDA distributions
│   └── tables/                              # Serialized metrics and ablation result CSVs
└── docs/
    └── Card_Fraud_Triage_Project_Documentation.md  # Comprehensive technical system documentation
```

---

## 9. Future Work & Roadmap (Phase 2)

Phase 1 establishes a classical probabilistic machine learning baseline via Gaussian Mixture Density Scoring. Planned Phase 2 extensions include:

1. **Deep Generative Modeling:** Implement a PyTorch-based Deep Autoencoder to learn non-linear reconstruction manifolds of legitimate transaction behavior without parametric Gaussian assumptions.
2. **Hybrid Score Fusion:** Combine GMM probabilistic density scores with Autoencoder reconstruction error ($L_2$ norm) into an ensemble anomaly index:
   $$S_{\text{hybrid}}(x) = \alpha \cdot \tilde{S}_{\text{GMM}}(x) + (1 - \alpha) \cdot \tilde{S}_{\text{AE}}(x)$$
3. **Semi-Supervised Fine-Tuning:** Explore extreme-value calibration (EVT) and contrastive loss embeddings to refine the triage boundary using verified historical fraud alerts.
