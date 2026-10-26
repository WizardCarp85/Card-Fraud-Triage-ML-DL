"""Synthetic unit tests for leakage-safe fraud-triage evaluation utilities."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation import (
    average_precision,
    evaluate_final_model,
    precision_at_k,
    recall_at_fpr,
    roc_auc,
    select_threshold_at_fpr,
    triage_table,
)


class EvaluationTests(unittest.TestCase):
    """Test metric definitions, threshold policies, validation, and output files."""

    def test_perfect_ranking_metrics(self) -> None:
        labels = np.array([0, 1, 0, 1])
        scores = np.array([0.1, 0.9, 0.2, 0.8])

        self.assertAlmostEqual(average_precision(labels, scores), 1.0)
        self.assertAlmostEqual(roc_auc(labels, scores), 1.0)
        self.assertAlmostEqual(recall_at_fpr(labels, scores, 0.0), 1.0)
        self.assertAlmostEqual(precision_at_k(labels, scores, 2), 1.0)

    def test_threshold_selection_maximizes_validation_recall_within_budget(self) -> None:
        validation_labels = np.array([1, 0, 1, 0, 0])
        validation_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

        selection = select_threshold_at_fpr(validation_labels, validation_scores, 1 / 3)

        self.assertAlmostEqual(selection.threshold, 0.7)
        self.assertAlmostEqual(selection.validation_recall, 1.0)
        self.assertAlmostEqual(selection.validation_fpr, 1 / 3)

    def test_triage_table_reports_business_counts(self) -> None:
        labels = np.array([1, 0, 1, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.1])

        row = triage_table(labels, scores, capacities=(3,)).iloc[0]

        self.assertEqual(row["frauds_caught"], 2)
        self.assertAlmostEqual(row["recall"], 1.0)
        self.assertAlmostEqual(row["precision"], 2 / 3)
        self.assertEqual(row["false_alarms"], 1)

    def test_invalid_arrays_raise_value_error(self) -> None:
        invalid_cases = [([], []), ([0, 1], [0.1]), ([0, 2], [0.1, 0.2]), ([0, 1], [0.1, np.nan])]
        for labels, scores in invalid_cases:
            with self.subTest(labels=labels, scores=scores):
                with self.assertRaises(ValueError):
                    average_precision(labels, scores)

    def test_invalid_k_and_fpr_budget_raise_value_error(self) -> None:
        labels = np.array([0, 1])
        scores = np.array([0.1, 0.9])

        with self.assertRaises(ValueError):
            precision_at_k(labels, scores, 0)
        with self.assertRaises(ValueError):
            precision_at_k(labels, scores, 3)
        with self.assertRaises(ValueError):
            recall_at_fpr(labels, scores, 1.1)

    def test_no_positive_examples_are_explicitly_undefined(self) -> None:
        labels = np.array([0, 0, 0])
        scores = np.array([0.9, 0.5, 0.1])

        self.assertTrue(math.isnan(average_precision(labels, scores)))
        self.assertTrue(math.isnan(roc_auc(labels, scores)))
        self.assertTrue(math.isnan(recall_at_fpr(labels, scores, 0.01)))
        self.assertEqual(select_threshold_at_fpr(labels, scores, 0.01).status, "undefined_no_positive")

    def test_final_evaluation_saves_outputs_and_keeps_validation_threshold(self) -> None:
        y_validation = np.array([1, 0, 1, 0, 0])
        score_validation = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        y_test = np.array([0, 1, 0, 1, 0])
        score_test = np.array([0.95, 0.85, 0.75, 0.65, 0.55])

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            result = evaluate_final_model(
                y_validation,
                score_validation,
                y_test,
                score_test,
                fpr_budgets=(1 / 3,),
                review_capacities=(2,),
                table_dir=output_root / "tables",
            )

            self.assertAlmostEqual(result.selections[0].threshold, 0.7)
            self.assertAlmostEqual(result.test_threshold_metrics.loc[0, "threshold"], 0.7)
            self.assertTrue((result.test_triage["review_capacity"] == 2).all())
            self.assertTrue((output_root / "tables" / "final_ranking_metrics.csv").exists())
            self.assertTrue((output_root / "tables" / "test_triage_table.csv").exists())


if __name__ == "__main__":
    unittest.main()
