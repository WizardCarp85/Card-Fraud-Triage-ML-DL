# Project Results

This folder contains all the experimental outputs, visual figures, and serialized evaluation tables for **Project 26: Card Fraud Triage (Phase 1)**.

---

## Figures

Visualizations generated during exploratory data analysis, model selection, density scoring, and ablation studies:

| File | Description | Purpose |
|---|---|---|
| `aic_bic_curve.png` | Model selection curves for component counts $k = 2 \dots 15$ | Justification of the optimal component count ($k=14$) via BIC minimization |
| `gmm_score_distributions.png` | Anomaly score ($-\ln p(x)$) distributions for legitimate vs fraud transactions | Evaluates score separation across validation and holdout test splits |
| `amount_log_transform.png` | Distribution of transaction amount before and after $\log(1 + x)$ transform | Visualizes right-skew compression for outlier handling |
| `scaler_comparison.png` | Distribution comparison of StandardScaler vs RobustScaler | Justifies the adoption of RobustScaler on heavy-tailed features |
| `covariance_comparison.png` | PR-AUC and runtime comparisons across covariance topologies | Evaluates `full`, `diag`, `tied`, and `spherical` covariance structures |
| `k_sensitivity.png` | Validation PR-AUC across various component counts | Checks performance stability across component choices |
| `pca_ablation.png` | Impact of PCA dimensionality reduction on anomaly detection | Demonstrates information retention across 29, 20, 10, and 5 dimensions |
| `eda_amount_time_distribution.png` | Histograms and KDE plots for Time and Amount variables | Exploratory analysis of temporal dynamics and transaction values |
| `eda_fraud_vs_legit_comparison.png` | Boxplots and kernel density comparisons of legitimate vs fraud cases | Analyzes discriminatory power between classes |
| `eda_correlation_heatmap.png` | Feature correlation matrix across all numeric columns | Validates orthogonality of PCA features and localized correlations |

---

## Tables

Serialized data tables containing quantitative results, evaluation metrics, and ablation experiment logs:

| File | Description | Key Contents |
|---|---|---|
| `production_model_evaluation.csv` | Final evaluation metrics of the production GMM | Test PR-AUC, ROC-AUC, Recall@FPR budgets (0.1%, 1.0%, 5.0%), and Precision@$K$ |
| `triage_capacity_tradeoffs.csv` | Capacity-constrained operational triage review queue | Fraud caught (Recall), Precision hit rate, and false alarm trade-offs at Top-100, 250, 500, 1000, 2000 |
| `ablation_summary.csv` | Combined summary of all 23 ablation experiments | Comparison of covariance types, component counts, scalers, dimensions, regularization, and random seeds |
| `ablation_covariance.csv` | Performance comparison across covariance structures | Validates that full covariance is required to capture localized correlations |
| `ablation_k_sensitivity.csv` | Sensitivity evaluation across component counts $k$ | Compares PR-AUC, ROC-AUC, and parameters for $k \in \{2, 3, 4, 5, 6, 8, 14\}$ |
| `ablation_scaler.csv` | Scaler ablation results | Compares RobustScaler vs StandardScaler convergence and PR-AUC |
| `ablation_pca.csv` | Feature dimensionality ablation results | Measures impact of reducing features from 29 to 20, 10, and 5 principal components |
| `ablation_reg_covar.csv` | Covariance regularization sensitivity results | Evaluates diagonal covariance regularization ($\epsilon = 10^{-6}, 10^{-4}, 10^{-3}$) |
| `ablation_seed_stability.csv` | Reproducibility results across multiple random seeds | Tests model stability and variance across seeds 42, 0, and 7 |
| `gmm_bic_aic_sweep.csv` | Component count sweep logs for $k = 2 \dots 15$ | Contains empirical AIC, BIC, total log-likelihood, iterations, and convergence status |
| `chronological_split_summary.csv` | Chronological split statistics | Train (70%), Validation (15%), and Test (15%) row counts, legitimate counts, and base rates |
| `scaler_comparison_stats.csv` | Feature scaling summary statistics | Empirical means and standard deviations before and after scaling transformations |
| `gmm_score_summary_val_test.csv` | Anomaly score distribution statistics | Class-wise median, mean, standard deviation, and percentiles across splits |
