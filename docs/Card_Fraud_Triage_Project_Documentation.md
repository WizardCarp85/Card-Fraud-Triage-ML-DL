# Project 26: Card Fraud Triage
## Phase 1 Only — Gaussian Mixture Density Scoring (Machine Learning)

### Complete Project Documentation & Team Work Division — 5 Members

**Course:** AML + DL Project — **Phase 1 (Machine Learning Evaluation) only**  
**Platform:** Google Colab + GitHub  
**Team Size:** Exactly 5 members  
**Dataset:** Credit Card Fraud Detection — ULB / Kaggle  
**Current Scope:** GMM-based anomaly detection on legitimate transactions  
**NOT in current scope:** Autoencoder and Hybrid approach — those are Phase 2 and will be added later. Mention them only as *Future Work* in the report/conclusion.

> **Phase 1 will be evaluated on:** Literature Review · Dataset Quality & EDA · Feature Engineering · Theoretical Rigor · Model Application · GitHub Repository & Code Quality · Project Presentation · Viva Voce · Project Report (LaTeX/IEEE format)

---

# 1. What Is This Project? (In Simple Words)

Banks see lakhs of card transactions every day. Almost all are genuine, and only a tiny fraction are fraud.

In our dataset:
- Total transactions: **2,84,807**
- Fraud transactions: **492 only (0.172%)**
- Legitimate transactions: **99.828%**

So this is **NOT a normal classification problem**. If a model predicts "legitimate" for every transaction, it will still get **99.83% accuracy** — but it will catch **0 frauds**. That is why the project instructions say: **do not rely on accuracy.**

### Our Approach: Anomaly Detection

Instead of teaching the model "what fraud looks like" (we have too few fraud examples), we teach it **only what a normal / legitimate transaction looks like**. Then:

> Anything that does NOT look normal = suspicious = send for manual review (triage).

This is exactly how real bank fraud teams work — the model prepares a **ranked review queue**, and human analysts check the most suspicious transactions first. That is why the project is called **Card Fraud *Triage***.

### The Phase 1 Model: Gaussian Mixture Model (GMM)

| Aspect | Details |
|---|---|
| **Model** | Gaussian Mixture Model (`sklearn.mixture.GaussianMixture`) |
| **Type** | Classical Machine Learning — probabilistic density estimation |
| **Core Idea** | Legitimate transactions form a few Gaussian "clusters/blobs" in feature space. The GMM learns the *probability density* of normal behaviour. |
| **Trained On** | **Only legitimate transactions** from the training split. Fraud rows never enter training. |
| **Anomaly Score** | `score = -log-likelihood` of a transaction under the fitted GMM. Low likelihood → high score → more suspicious. |
| **Model Selection** | Sweep number of components k, choose using **AIC and BIC** curves + stability |
| **Key Theory** | EM Algorithm — Dempster et al. (1977), mixture densities, covariance structures, AIC/BIC penalised likelihood |
| **Known Limitation (discuss in report)** | Assumes the legitimate manifold is approximately a Gaussian mixture; can become unstable in high dimensions (covariance singularity / curse of dimensionality). This limitation is exactly what the Phase 2 Autoencoder will later address. |

### What the Final Story of Phase 1 Should Be

1. Legitimate behaviour can be modelled as a probability density.
2. Fraud transactions fall in **low-density regions** of that model.
3. By ranking transactions by `-log-likelihood`, we can build a review queue that catches a useful fraction of fraud while keeping false alarms within a budget analysts can handle.
4. Every claim is backed by PR curves, Recall@FPR budgets, and Precision@review-capacity numbers — not accuracy.

---

# 2. Dataset Details

**Credit Card Fraud Detection (ULB / Kaggle)** — real European cardholder transactions from September 2013 (2 days).

| Column | Meaning |
|---|---|
| `Time` | Seconds elapsed since the first transaction in the dataset |
| `V1` – `V28` | PCA-transformed features (original features hidden for privacy) |
| `Amount` | Transaction amount — highly skewed, needs log transform / robust scaling |
| `Class` | **1 = Fraud, 0 = Legitimate** ← the label, hidden during unsupervised training |

**Download:** Kaggle → search "Credit Card Fraud Detection" → `creditcard.csv` (~150 MB). Requires a free Kaggle account. Upload it once to the team's shared Google Drive folder — do not re-upload every session.

**Important properties to discuss in EDA:**
- Extreme class imbalance (0.172% fraud) — report the **base rate** in every split and every results table.
- `V1–V28` are already PCA-scaled; `Time` and `Amount` are NOT — they need transformation.
- No missing values expected (verify in EDA, don't assume).
- Duplicates exist in this dataset — check, decide handling, and document the decision.

---

# 3. The Golden Rules of Phase 1 (Examiners Will Check These)

1. **Train ONLY on legitimate transactions.** Fraud rows must never enter GMM training.
2. **Fit the scaler ONLY on legitimate training rows.** Never fit on full data — that is data leakage.
3. **Chronological split** (sort by `Time`): Train (earliest) → Validation → Test (latest). This mimics real deployment.
4. **Select thresholds ONLY on validation data.** Never tune a threshold on the test set.
5. **Test set stays untouched** until the final evaluation — run it once, report honestly.
6. **No accuracy as a headline metric.** Report:
   - **PR-AUC** (Precision–Recall AUC — main metric for imbalanced data)
   - **ROC-AUC** (secondary, for completeness only)
   - **Recall @ fixed False Positive Rate** — FPR = 0.1%, 1%, 5% ("false positive budgets")
   - **Precision @ Review Capacity** — Precision@100, @500, @1000 ("if analysts can check only the top-K transactions, how many are real fraud?")
7. **Keep the fraud label hidden during unsupervised training** — use it only for splitting, validation thresholding, and final evaluation.
8. **Report base rates everywhere** — overall and per split.

---

# 4. Splitting & Preprocessing Design (Freeze This Before Modelling)

### Chronological Split

Sort the full data by `Time`, then split:

| Split | Portion | Content Used For |
|---|---|---|
| **Train** | Earliest ~70% | Take **only legitimate rows** → fit scaler + train GMM |
| **Validation** | Next ~15% | Mixed (legit + fraud) → choose k sanity-check, choose thresholds, compare variants |
| **Test** | Latest ~15% | Mixed, **untouched** → final metrics, reported once |

For each split, report: total rows, fraud count, fraud % (base rate). Note: with a chronological split, the fraud rate may differ slightly across splits — that is realistic and should be commented on, not hidden.

### Preprocessing Pipeline (same for train/val/test)

1. `Amount_log = log1p(Amount)` — Amount is heavily right-skewed.
2. Optional engineered feature: hour of day from `Time` — `(Time // 3600) % 24`. If used, justify it; if dropped, say why.
3. Scale all features with the chosen scaler (`StandardScaler` vs `RobustScaler` — compare in an experiment, pick one, justify).
4. Scaler is fit **only on legitimate training rows**, then applied to validation and test.
5. Save the fitted scaler to Drive (`joblib.dump`) so the pipeline is reproducible.

---

# 5. Google Colab Workflow — Step by Step

Everyone works in **one shared master notebook** (plus individual scratch notebooks for experiments). Suggested name: `Card_Fraud_Triage_Phase1.ipynb`

### Setup in Colab
```python
# 1. Mount Google Drive (dataset + outputs live here; Colab RAM is temporary)
from google.colab import drive
drive.mount('/content/drive')

# 2. Stack: pandas, numpy, scikit-learn, matplotlib, seaborn (all pre-installed in Colab)
import pandas, numpy, sklearn

# 3. Load dataset from Drive
df = pandas.read_csv('/content/drive/MyDrive/CardFraud/creditcard.csv')

# 4. Fix the seed everywhere for reproducibility
SEED = 42
numpy.random.seed(SEED)
```

No GPU is needed for Phase 1 — GMM runs on CPU. If full-covariance GMM gets slow on 30 features, use the experiments in Section 7 (`diag` covariance, PCA reduction, subsampling for the sweep) and document the choice.

### Final Notebook Structure (Phase 1 Deliverable)

| Section | Content | Owner |
|---|---|---|
| 0. Setup | Imports, seed, Drive mount, config (paths, seed, capacities K = 100/500/1000) | M1 |
| 1. Data Loading | Load CSV, shape, dtypes, head, column description | M1 |
| 2. EDA | Class imbalance & base rate, missing/duplicates, Amount & Time distributions, fraud vs legit comparison, correlation heatmap, key insights written in markdown | M1 |
| 3. Splitting | Chronological split by `Time`, per-split counts & base rates table | M1 + M2 |
| 4. Feature Engineering & Preprocessing | log1p(Amount), optional hour feature, scaler comparison, scaler fit on legit-train only, before/after plots, leakage checklist | M2 |
| 5. GMM Theory (markdown) | Mixture density, EM algorithm E/M steps, AIC/BIC formulas, covariance types, why −log-likelihood is an anomaly score | M3 |
| 6. GMM Model Selection | Sweep k = 2…15, AIC/BIC curves, covariance-type comparison, final k justification | M3 + M4 |
| 7. GMM Training & Scoring | Fit final GMM on legit-train, score validation & test (−log-likelihood), score distribution plots (legit vs fraud) | M3 |
| 8. Evaluation | PR curve, PR-AUC, ROC-AUC, Recall@FPR (0.1/1/5%), Precision@K (100/500/1000) | M5 |
| 9. Threshold & Triage Analysis | Thresholds chosen on validation, applied to test, review-capacity tables, FP/FN trade-off plots, business interpretation | M5 |
| 10. Experiments / Ablations | Covariance type, scaler choice, k sensitivity, (optional) PCA-dimension effect — table + takeaways | M4 |
| 11. Limitations & Conclusion | GMM assumptions & high-dimension instability, honest result discussion, Future Work = Phase 2 Autoencoder + Hybrid (1 short paragraph only) | M3 + M5 |
| 12. Reproducibility | Library versions, seed, how to re-run end-to-end | M1 |

---

# 6. Work Division — 5 Members (Phase 1 Only)

> **Viva is INDIVIDUAL.** Every member must be able to explain the *whole* Phase 1 project, but each member **owns** one part deeply and must defend its theory, code, decisions, and results alone.

## Member Roles at a Glance

| Member | Role Title | Owns (Main Responsibility) |
|---|---|---|
| **Member 1** | **Team Lead + Dataset & EDA Owner** | Coordination, GitHub, dataset, EDA, chronological split |
| **Member 2** | **Feature Engineering & Preprocessing Owner** | Cleaning, transforms, scaling, leakage prevention |
| **Member 3** | **GMM Model Owner (Core ML)** | GMM theory (EM, AIC/BIC), model training, density scoring |
| **Member 4** | **Model Experiments & Robustness Owner** | k-sweep support, covariance comparison, ablation studies, stability checks |
| **Member 5** | **Evaluation, Threshold Analysis & Reporting Owner** | Metrics, PR curves, threshold/triage analysis, LaTeX report & slides assembly |

---

## Member 1 — Team Lead + Dataset & EDA Owner

**Responsibilities:**
- Create & manage the GitHub repo, folder structure, README; ensure all 5 members commit regularly **under their own names** (individual contribution is assessed).
- Coordinate team meetings, track the Phase 1 deadline, maintain the task checklist in this document.
- Download the dataset from Kaggle, verify integrity, place it in the shared Drive folder.
- **EDA (Notebook Sections 1–2):** shape, dtypes, missing values, duplicates, class imbalance (base rate 0.172%), `Amount` and `Time` distributions, fraud-vs-legit feature comparisons, correlation heatmap — every plot followed by a written insight, not just a figure.
- Implement the **chronological split** with Member 2 and produce the per-split base-rate table.
- Write report sections: *Introduction, Problem Statement, Dataset Description, EDA.*

**Must be able to answer in viva:**
- Why is accuracy meaningless here? What is the base rate?
- Why a chronological split instead of a random split? (Simulates real deployment; avoids future-information leakage)
- What did EDA reveal about fraud — Amount patterns, time patterns, which V-features separate fraud from legit?

**Deliverables:** GitHub repo + README, EDA notebook section, dataset documentation, split summary table.

---

## Member 2 — Feature Engineering & Preprocessing Owner

**Responsibilities:**
- Data cleaning: duplicates decision, outlier inspection for `Amount`.
- **Feature engineering:** `Amount_log = log1p(Amount)`, optional hour-of-day feature from `Time`; keep V1–V28 as-is (already PCA components) and explain why.
- **Scaling:** Compare `StandardScaler` vs `RobustScaler` experimentally; fit the final scaler **only on legitimate training rows**; save it to Drive.
- Build the reusable preprocessing pipeline applied identically to train/validation/test.
- Write and verify the **data-leakage checklist** in the notebook (scaler fit, GMM fit, threshold selection — all checked).
- Write report sections: *Feature Engineering, Preprocessing, Data Leakage Prevention.*

**Must be able to answer in viva:**
- Why fit the scaler only on legit training data? What leakage happens otherwise?
- Why log-transform `Amount`? When is RobustScaler better than StandardScaler?
- Why must preprocessing be identical across splits for the results to be valid?

**Deliverables:** Preprocessing pipeline code, before/after distribution plots, saved scaler file, leakage checklist.

---

## Member 3 — GMM Model Owner (Phase 1 Core)

**Responsibilities:**
- Own the theory write-up (**Notebook Section 5**, in markdown): mixture density formula, **EM algorithm** (E-step responsibilities, M-step parameter updates), AIC/BIC formulas, covariance types, why low log-likelihood = anomaly.
- Implement GMM (`GaussianMixture`) trained **only on legitimate training data**.
- Lead **model selection with Member 4:** sweep components k = 2…15, plot **AIC and BIC curves**, justify the final k (minimum BIC / elbow + stability, not just "lowest number wins").
- Compute anomaly score = **−log-likelihood** for validation and test; plot score distributions for legit vs fraud.
- Handle numerical issues: `reg_covar`, convergence warnings, document any fix.
- Summarise **Dempster et al. (1977)** in 1 page and teach it to the team.
- Write report sections: *Literature Review (GMM/EM), Methodology — GMM, GMM Results & Discussion.*

**Must be able to answer in viva:**
- Explain the EM algorithm step-by-step. Why does a GMM give a probability density?
- AIC vs BIC — formulas, how the penalty differs, why BIC penalises complexity more.
- Why can a GMM become unstable in high dimensions? (Covariance singularity, curse of dimensionality)
- Why is −log-likelihood a valid anomaly score? What does `reg_covar` do?

**Deliverables:** Theory section, AIC/BIC analysis, trained GMM model, GMM score files (validation + test), score distribution plots.

---

## Member 4 — Model Experiments & Robustness Owner

**Responsibilities:**
- Run the **covariance-type comparison** (`full`, `diag`, `tied`, `spherical`) — metrics + training time + stability (warnings/singularity) — and recommend the final configuration with reasoning.
- Support the k-sweep: check sensitivity of PR-AUC to k around the chosen value (is the result fragile to k?).
- Run **ablation experiments (Notebook Section 10):**
  - Scaler choice: StandardScaler vs RobustScaler (with Member 2)
  - Covariance type (above)
  - Number of components sensitivity
  - Optional: PCA-reduced features (e.g., 30 → 10 dims) — does dimension reduction stabilise GMM?
  - Optional: `reg_covar` sensitivity if convergence issues appear
- Check **stability/reproducibility:** refit with different seeds, confirm metrics don't swing wildly; report any variance.
- Maintain the experiments table that goes into the report.
- Write report sections: *Experimental Setup, Ablation Studies, Robustness & Stability.*

**Must be able to answer in viva:**
- `full` vs `diag` covariance — parameter count, expressiveness vs stability trade-off in 30 dimensions.
- How did you know the chosen k was not a fluke? What does the sensitivity analysis show?
- What happens to GMM results if the scaler changes, and why?

**Deliverables:** Covariance comparison table, ablation table, stability-check results, experiment plots.

---

## Member 5 — Evaluation, Threshold Analysis & Reporting Owner

**Responsibilities:**
- Build the **evaluation code** used for every experiment (one shared set of functions so all comparisons are fair):
  - `pr_auc`, `roc_auc`, `recall_at_fpr(0.1%, 1%, 5%)`, `precision_at_k(100, 500, 1000)`
- Plot the **Precision–Recall curve** for the final GMM, plus threshold-vs-precision/recall plots.
- **Threshold & triage analysis (Notebook Section 9):** choose thresholds on *validation only* (e.g., threshold at a fixed FPR budget, and threshold at review capacity K), apply to test, and build the trade-off table:
  > *"If analysts review the top 500 flagged transactions, we catch X% of all fraud, Y% of the reviewed ones are real fraud, and Z false alarms are generated."*
- Interpret results in business terms: analyst workload vs fraud caught — this is the "triage" story of the project.
- Assemble the final **LaTeX/IEEE report** and **presentation slides**, integrating every member's sections consistently.
- Write report sections: *Evaluation Methodology, Threshold & Business Analysis, Limitations, Conclusion & Future Work.*

**Must be able to answer in viva:**
- Why PR-AUC over ROC-AUC for imbalanced data?
- Recall@FPR vs Precision@K — what real-world constraint does each represent?
- How exactly was the threshold chosen without touching the test set?
- If the bank can review only 200 transactions a day, what threshold/capacity do you recommend and why?

**Deliverables:** Evaluation functions, PR-curve figure, final metrics table, threshold trade-off tables, assembled report + slides.

---

## Shared Responsibilities (All 5 Members)

- Every member reads the **Dempster et al. (1977)** summary (prepared by M3) and can explain EM in their own words.
- Every member must run the **full final notebook end-to-end** at least once before submission.
- Code review cycle: M1→M2, M2→M3, M3→M4, M4→M5, M5→M1 — each member reviews one other member's section.
- On Day 2 (or end of Day 1): one 30–45 minute mock viva — each member explains *another* member's part.
- Final report proofreading and presentation rehearsal — together.

---

# 7. Phase 1 Evaluation Criteria → Where It Is Covered

| Evaluation Criterion | Covered In | Main Owner |
|---|---|---|
| Literature Review | Report: EM/GMM (Dempster 1977), density-based anomaly detection background | M3 (all contribute) |
| Dataset Quality & EDA | Notebook §1–2, Report: Dataset + EDA | M1 |
| Feature Engineering | Notebook §4, Report: Feature Engineering | M2 |
| Theoretical Rigor | Notebook §5 (EM, AIC/BIC, covariance theory), viva prep | M3 |
| Model Application | Notebook §6–7, GMM training & scoring | M3 + M4 |
| GitHub Repository & Code Quality | Repo structure, commits by all 5, README, clean notebook | M1 (all commit) |
| Project Presentation | Slides | M5 (all present) |
| Viva Voce | Mock viva practice, per-member viva questions in §6 | All |
| Project Report (LaTeX/IEEE) | Final report | M5 assembles, all write their sections |

---

# 8. 1–2 Day Execution Plan (No Long Timeline)

> This project has to be completed in **1–2 days**, so all 5 members work **in parallel**, not one after another. Do not wait for one member to finish before starting your part.

## Day 1 — Build Everything

| Time Block | Work | Owner(s) | Output |
|---|---|---|---|
| First 1 hour | Create GitHub repo, download Kaggle dataset, upload to Drive, create Colab notebook, fix seed & folder paths | M1 (all join) | Repo + dataset + blank notebook ready |
| Next 2–3 hours (parallel) | **M1:** EDA plots & insights · **M2:** preprocessing pipeline · **M3:** GMM theory notes + baseline GMM · **M4:** k-sweep / covariance experiments · **M5:** evaluation functions (PR-AUC, Recall@FPR, Precision@K) | All 5 in parallel | All core sections drafted |
| Next 2 hours | Merge everything into the master notebook in order (§0–§10), run GMM end-to-end, get first test-set metrics | M3 + M5 lead merge, all help fix errors | First full result table |
| End of Day 1 | Threshold & triage analysis, save all figures/tables, each member writes their report section draft | M5 + respective owners | Notebook runs fully, report drafts ready |

## Day 2 — Finish, Polish, Prepare (if you have a 2nd day)

| Time Block | Work | Owner(s) | Output |
|---|---|---|---|
| First 2 hours | Fix errors, finish ablations, re-run **complete notebook top-to-bottom on a fresh Colab runtime** | All | Clean, error-free notebook |
| Next 2 hours | Assemble LaTeX/IEEE report + presentation slides from Day-1 drafts, add figures/tables | M5 assembles, all contribute | Report PDF + slides |
| Next 1–2 hours | README, GitHub cleanup, final commits by all 5 members, mock viva (each member explains another's part) | All | Submission-ready Phase 1 |
| Final 30 mins | Final check: notebook runs, report opens, slides open, dataset NOT committed to GitHub | M1 | Ready to submit |

## If You Have Only 1 Day

Compress Day 2 into the evening: skip optional experiments (PCA ablation, `reg_covar` sweep), keep only — EDA → preprocessing → GMM + AIC/BIC → evaluation → threshold analysis → short report + slides. **Never skip:** leakage checklist, validation-only threshold selection, and one full clean notebook run.

**Hard rule:** the submitted notebook must run top-to-bottom in Colab without errors on a fresh runtime. Test this before submitting, not after.

---

# 9. GitHub Repository Structure (Phase 1)

```
card-fraud-triage-project26/
├── README.md                  ← Project overview, Phase 1 scope, setup, how to run, team table
├── notebooks/
│   ├── Card_Fraud_Triage_Phase1.ipynb   ← FINAL master notebook (submission)
│   └── experiments/           ← individual scratch notebooks (e.g., m4_covariance_test.ipynb)
├── reports/
│   └── Phase1_Report.pdf      ← LaTeX/IEEE format
├── presentation/
│   └── Phase1_Slides.pdf
├── results/
│   ├── figures/               ← PR curve, AIC/BIC, EDA plots
│   └── tables/                ← metrics & threshold CSVs
└── docs/
    └── Project_Documentation.md   ← this file
```

**Git rules:** commit small & often, meaningful messages (`feat(gmm): add BIC sweep k=2-15`), never commit the raw dataset CSV, every member must have visible commits (individual contribution is assessed).

---

# 10. Final Results Format (What Success Looks Like)

Final GMM results on the **untouched test set** (fill with your real numbers):

| Model | PR-AUC | ROC-AUC | Recall @ FPR 0.1% | Recall @ FPR 1% | Recall @ FPR 5% | Precision @ 100 | Precision @ 500 | Precision @ 1000 |
|---|---|---|---|---|---|---|---|---|---|
| GMM (final config) | — | — | — | — | — | — | — | — |

Plus a **triage trade-off table** (thresholds from validation, applied to test):

| Review Capacity (Top-K) | Fraud Caught (Recall) | Precision | False Alarms |
|---|---|---|---|
| 100 | — | — | — |
| 500 | — | — | — |
| 1000 | — | — | — |

**Expected story (hypothesis to test, not guaranteed):** fraud transactions concentrate in the lowest-likelihood region, so the top-K review queue should have precision *far above* the 0.172% base rate, and Recall@FPR 1% should be meaningfully high. Whatever the actual result — **report honestly and explain WHY** using the GMM's assumptions. A well-explained weak result scores better than an unexplained strong one.

---

# 11. Risks & How We Handle Them (Phase 1)

| Risk | Solution |
|---|---|
| GMM too slow with `full` covariance in 30-D | Use `diag` covariance, subsample legit data for the k-sweep, or PCA-reduce first — document the choice (M4 experiment) |
| Covariance singularity / convergence warnings | Increase `reg_covar`, switch covariance type, check for duplicate/constant features |
| Results change every run | Fixed seed (42) everywhere, stability check by M4, report variance if any |
| Colab session resets, work lost | Everything saved to Drive + notebook pushed to GitHub after each session |
| Threshold accidentally tuned on test | Threshold code takes validation scores only — code-review checkpoint by M5 |
| One member can't explain another's part in viva | Cross-teaching during Day 1 merge + mock viva on Day 2 (mandatory) |

---

# 12. Viva Quick-Prep Checklist — Phase 1 (Everyone)

- [ ] Why anomaly detection instead of classification?
- [ ] Base rate of the dataset & why accuracy is useless
- [ ] Chronological split & every leakage-prevention step
- [ ] EM algorithm — E-step and M-step, in plain words
- [ ] AIC vs BIC — formulas and penalty difference
- [ ] Covariance types and high-dimensional instability of GMM
- [ ] Why −log-likelihood is the anomaly score
- [ ] PR-AUC, Recall@FPR, Precision@K — definitions & business meaning
- [ ] How the threshold was chosen (validation only) & the triage recommendation
- [ ] Limitations of GMM & what Phase 2 (Autoencoder + Hybrid) will address — one short paragraph

---

# 13. Out of Scope for Now (Phase 2 — Later)

- ❌ Autoencoder (PyTorch) — Phase 2
- ❌ Hybrid GMM + Autoencoder score fusion — Phase 2 (bonus marks)
- ❌ Any DL code in the Phase 1 notebook

In Phase 1, these appear **only** as a short *Future Work* paragraph in the report/conclusion.

---

*Updated for Phase 1 only, to be completed in 1–2 days. Fill in member names against Member 1–5 in Section 6, create the GitHub repo, and start with the Day 1 tasks in Section 8.*
