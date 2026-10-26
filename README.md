# Card Fraud Triage (Project 26)

> **Unsupervised Anomaly Detection & Density Scoring via Gaussian Mixture Models (Phase 1)**  
> **Course:** Advanced Machine Learning & Deep Learning (AML + DL)

An unsupervised machine learning pipeline for real-time card fraud triage, modeling the probability density of legitimate transactions and prioritizing suspicious events for analyst review via negative log-likelihood scoring.


---

## Team Members

| Name | Enrollment No. | Email | Role |
|---|---|---|---|
| **Yashkumar Nimje** | `2401020113` | `yashkumar.nimje2024@nst.rishihood.edu.in` | Team Lead & Data Architecture |
| **Saksham Narotra** | `2401010402` | `saksham.narotra2024@nst.rishihood.edu.in` | Feature Engineering & Preprocessing |
| **Rudra Pratap Singh Choudhary** | `2401010394` | `rudra.choudhary2024@nst.rishihood.edu.in` | Probabilistic Modeling (GMM Core) |
| **Abhigya Sachdeva** | `2401020003` | `abhigya.sachdeva2024@nst.rishihood.edu.in` | Model Selection & Robustness |
| **Somraj Nandi** | `2401020070` | `somraj.nandi2024@nst.rishihood.edu.in` | Evaluation Framework & Decision Systems |

---

## Key Highlights

- **Unsupervised Modeling:** GMM is fit strictly on legitimate transactions ($\mathcal{X}_{\text{legit}}$); fraud labels are excluded during model training.
- **Leakage-Free Temporal Split:** Chronological 70% Train / 15% Validation / 15% Test partitioning.
- **Evaluation on Business Constraints:** Focus on PR-AUC, Recall@FPR budgets (0.1%, 1%, 5%), and Precision@$K$ review queues rather than misleading raw accuracy.
- **Detailed Documentation:** See [Card_Fraud_Triage_Project_Documentation.md](file:///Users/Wizard/Documents/Projects/Working/card-fraud-triage-project26/docs/Card_Fraud_Triage_Project_Documentation.md) for full architectural derivations and methodology.

---

## Repository Structure

```
card-fraud-triage-project26/
├── README.md                                # Project overview and team members
├── notebooks/
│   ├── Card_Fraud_Triage_Phase1.ipynb       # Master end-to-end reproducible notebook
│   └── experiments/                         # Scratch and ablation notebooks
├── reports/
│   ├── Phase1_Report.pdf                    # Formal technical report (LaTeX/IEEE format)
│   ├── Dempster_1977_EM_Summary.md          # Theoretical derivation & literature notes
│   └── Member3_Report_Sections.md           # Model methodology and mathematical write-ups
├── presentation/
│   └── Phase1_Slides.pdf                    # Presentation slides
├── results/
│   ├── figures/                             # Exported PR curves, AIC/BIC plots, EDA figures
│   └── tables/                              # Metrics and ablation result CSVs
└── docs/
    └── Card_Fraud_Triage_Project_Documentation.md  # Detailed technical project documentation
```
