# Phase 2 anomaly score files

Export one model and split per UTF-8 CSV with exactly four columns.

| Column | Meaning |
|---|---|
| row_id | Stable nonempty text identifier. Use the frozen split raw_row_id, renamed to row_id. |
| split | train, validation or test. Map the manifest's val to validation. |
| model | Model identifier such as gmm or ae. |
| anomaly_score | Finite numeric score, larger means more suspicious. |

IDs are unique within a file and disjoint between splits. Values with surrounding whitespace are rejected. Preserve source IDs through preprocessing and inference. The example CSV is synthetic.

GMM producers export negative log likelihood. AE producers export mean squared reconstruction error. Explicitly pass score_direction="higher_is_more_suspicious". The loader checks this declaration. Numeric values alone cannot establish whether a producer negated log likelihood, so check that in its scoring implementation.

Score files contain only these four columns. Store labels separately in an evaluation CSV with exactly row_id,split,Class and binary Class values 0 or 1. Evaluation requires exactly matching identity sets and split names, rejecting extra label rows too. Evaluation label loading accepts validation and test only.

load_anomaly_scores returns label-free records sorted by lexical row_id, preserving leading zeros. Existing stable ranking functions therefore break score ties by lexical ID consistently across models and shuffled files. Thresholds flag all members of a tied score group together.

```python
from evaluation import evaluate_score_files

result = evaluate_score_files(
    "gmm_validation.csv", "validation_labels.csv",
    "gmm_test.csv", "test_labels.csv",
    model="gmm",
    score_direction="higher_is_more_suspicious",
    table_dir="results/phase2/gmm/tables",
)
```

Use the same interface for model="ae". The adapter delegates to existing AP, ROC-AUC, FPR threshold and top-K utilities. Supply smaller review_capacities for tiny fixtures. The interface prevents labels and features from entering score schemas. Training code must separately enforce legitimate-only fitting.

Phase 1 CSVs have Class and an original_index representing a reset sorted position. Reconstruct verified raw IDs from split membership before exporting this schema and move Class to the separate evaluation file. Renaming original_index alone would lose source identity.
