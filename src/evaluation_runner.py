"""Evaluation runner integrating with evaluation.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation import evaluate_final_model
from src.config import EvaluationConfig


def run_evaluation(
    y_val: np.ndarray,
    scores_val: np.ndarray,
    val_indices: np.ndarray,
    y_test: np.ndarray,
    scores_test: np.ndarray,
    test_indices: np.ndarray,
    config: EvaluationConfig,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Execute evaluation using evaluation.py and save isolated tables and scores.

    Thresholds are calibrated strictly on validation scores and evaluated on unseen test scores.
    """
    output_dir = Path(run_dir)
    tables_dir = output_dir / "tables"
    scores_dir = output_dir / "scores"
    tables_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run evaluation framework
    eval_result = evaluate_final_model(
        y_validation=y_val,
        score_validation=scores_val,
        y_test=y_test,
        score_test=scores_test,
        fpr_budgets=config.fpr_budgets,
        review_capacities=config.review_capacities,
        table_dir=tables_dir,
    )

    # 2. Save score tables with original indices
    df_val_scores = pd.DataFrame(
        {
            "original_index": val_indices,
            "anomaly_score": scores_val,
            "Class": y_val,
        }
    )
    val_scores_path = scores_dir / "scores_val.csv"
    df_val_scores.to_csv(val_scores_path, index=False)

    df_test_scores = pd.DataFrame(
        {
            "original_index": test_indices,
            "anomaly_score": scores_test,
            "Class": y_test,
        }
    )
    test_scores_path = scores_dir / "scores_test.csv"
    df_test_scores.to_csv(test_scores_path, index=False)

    # 3. Export summary JSON
    summary: dict[str, Any] = {
        "validation": {
            "ap": float(eval_result.ranking_metrics.loc[eval_result.ranking_metrics["split"] == "validation", "average_precision_ap"].iloc[0]),
            "roc_auc": float(eval_result.ranking_metrics.loc[eval_result.ranking_metrics["split"] == "validation", "roc_auc"].iloc[0]),
        },
        "test": {
            "ap": float(eval_result.ranking_metrics.loc[eval_result.ranking_metrics["split"] == "test", "average_precision_ap"].iloc[0]),
            "roc_auc": float(eval_result.ranking_metrics.loc[eval_result.ranking_metrics["split"] == "test", "roc_auc"].iloc[0]),
        },
        "threshold_selections": [
            {
                "fpr_budget": s.fpr_budget,
                "threshold": float(s.threshold),
                "validation_recall": float(s.validation_recall),
                "validation_fpr": float(s.validation_fpr),
            }
            for s in eval_result.selections
        ],
    }

    summary_file = tables_dir / "summary_metrics.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "ranking_metrics": eval_result.ranking_metrics,
        "validation_thresholds": eval_result.validation_thresholds,
        "test_threshold_metrics": eval_result.test_threshold_metrics,
        "test_triage": eval_result.test_triage,
        "summary": summary,
        "artifacts": {
            "scores_val": str(val_scores_path),
            "scores_test": str(test_scores_path),
            "summary_json": str(summary_file),
            "ranking_metrics_csv": str(tables_dir / "final_ranking_metrics.csv"),
            "test_triage_csv": str(tables_dir / "test_triage_table.csv"),
            "validation_thresholds_csv": str(tables_dir / "validation_fpr_thresholds.csv"),
            "test_threshold_metrics_csv": str(tables_dir / "test_threshold_metrics.csv"),
        },
    }
