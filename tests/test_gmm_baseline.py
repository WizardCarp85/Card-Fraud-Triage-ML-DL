"""Automated regression and acceptance tests for the Phase 2 GMM baseline (Issue #30)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from data.reproduce_gmm_baseline import (
    PHASE1_GMM_PATH,
    PHASE1_SCALER_PATH,
    load_gmm_artifacts,
)
from evaluation import (
    evaluate_score_files,
    load_anomaly_scores,
    load_evaluation_labels,
)


class GMMBaselineValidationTests(unittest.TestCase):
    """Verify all acceptance criteria for Issue #30."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.scores_dir = Path("data/scores")
        cls.val_scores_path = cls.scores_dir / "gmm_validation.csv"
        cls.val_labels_path = cls.scores_dir / "validation_labels.csv"
        cls.test_scores_path = cls.scores_dir / "gmm_test.csv"
        cls.test_labels_path = cls.scores_dir / "test_labels.csv"
        cls.prov_path = cls.scores_dir / "gmm_baseline_provenance.json"
        cls.manifest_path = Path("data/split_manifest.json")

        with open(cls.manifest_path, "r", encoding="utf-8") as f:
            cls.manifest = json.load(f)

    def test_score_and_label_files_exist(self) -> None:
        """Verify that committed score and label files exist."""
        self.assertTrue(self.val_scores_path.is_file(), "gmm_validation.csv missing")
        self.assertTrue(self.val_labels_path.is_file(), "validation_labels.csv missing")
        self.assertTrue(self.test_scores_path.is_file(), "gmm_test.csv missing")
        self.assertTrue(self.test_labels_path.is_file(), "test_labels.csv missing")
        self.assertTrue(self.prov_path.is_file(), "gmm_baseline_provenance.json missing")

    def test_score_file_schema_conformance(self) -> None:
        """Verify strict adherence to Issue #26 score schema."""
        df_scores = pd.read_csv(self.val_scores_path, dtype=str)
        self.assertListEqual(
            list(df_scores.columns),
            ["row_id", "split", "model", "anomaly_score"],
            "Score file must have exactly 4 columns: row_id,split,model,anomaly_score",
        )
        self.assertTrue((df_scores["split"] == "validation").all())
        self.assertTrue((df_scores["model"] == "gmm").all())

        df_labels = pd.read_csv(self.val_labels_path, dtype=str)
        self.assertListEqual(
            list(df_labels.columns),
            ["row_id", "split", "Class"],
            "Label file must have exactly 3 columns: row_id,split,Class",
        )
        self.assertTrue((df_labels["split"] == "validation").all())
        self.assertTrue(df_labels["Class"].isin(["0", "1"]).all())

    def test_validation_row_ids_and_counts_match_manifest(self) -> None:
        """Verify row IDs and fraud counts match the frozen split manifest exactly."""
        df_scores = pd.read_csv(self.val_scores_path)
        df_labels = pd.read_csv(self.val_labels_path)

        manifest_val_ids = [str(x) for x in self.manifest["split_indices"]["val"]]
        self.assertEqual(len(df_scores), len(manifest_val_ids))
        self.assertEqual(len(df_labels), len(manifest_val_ids))

        # Check exact row IDs set and count
        self.assertEqual(set(df_scores["row_id"].astype(str)), set(manifest_val_ids))
        self.assertEqual(set(df_labels["row_id"].astype(str)), set(manifest_val_ids))

        # Check fraud counts match manifest
        expected_fraud = self.manifest["split_statistics"]["validation"]["fraud_rows"]
        self.assertEqual(int(df_labels["Class"].astype(int).sum()), expected_fraud)
        self.assertEqual(expected_fraud, 55)

    def test_score_contract_loaders_and_lexical_sorting(self) -> None:
        """Verify load_anomaly_scores and load_evaluation_labels load without error."""
        scores = load_anomaly_scores(
            self.val_scores_path,
            expected_split="validation",
            expected_model="gmm",
            score_direction="higher_is_more_suspicious",
        )
        labels = load_evaluation_labels(
            self.val_labels_path,
            expected_split="validation",
        )
        self.assertEqual(len(scores), 42558)
        self.assertEqual(len(labels), 42558)

        # Confirm scores are sorted lexically by row_id per the contract
        self.assertTrue(scores["row_id"].is_monotonic_increasing)

        # Invalid direction rejected
        with self.assertRaises(ValueError):
            load_anomaly_scores(
                self.val_scores_path,
                expected_split="validation",
                expected_model="gmm",
                score_direction="lower_is_more_suspicious",
            )

    def test_score_values_are_finite_and_higher_is_suspicious(self) -> None:
        """Verify scores are finite and fraud transactions have higher scores than legitimate ones."""
        df_scores = pd.read_csv(self.val_scores_path)
        scores = df_scores["anomaly_score"].to_numpy(dtype=float)

        self.assertTrue(np.isfinite(scores).all(), "Scores must be strictly finite.")
        self.assertFalse(np.isnan(scores).any(), "Scores must contain no NaN.")

        # Join with labels to verify score direction
        df_labels = pd.read_csv(self.val_labels_path)
        merged = df_scores.merge(df_labels, on=["row_id", "split"])

        legit_scores = merged.loc[merged["Class"] == 0, "anomaly_score"]
        fraud_scores = merged.loc[merged["Class"] == 1, "anomaly_score"]

        self.assertGreater(
            fraud_scores.median(),
            legit_scores.median(),
            "Fraud scores must be significantly higher than legitimate transactions.",
        )
        self.assertGreater(
            fraud_scores.mean(),
            legit_scores.mean(),
            "Mean fraud anomaly score must exceed mean legitimate score.",
        )

    def test_validation_ap_and_roc_regeneration(self) -> None:
        """Verify that validation AP and ROC-AUC match Phase 1 outputs to high precision."""
        df_scores = pd.read_csv(self.val_scores_path)
        df_labels = pd.read_csv(self.val_labels_path)
        merged = df_scores.merge(df_labels, on=["row_id", "split"])

        y_true = merged["Class"].to_numpy(dtype=int)
        y_score = merged["anomaly_score"].to_numpy(dtype=float)

        ap = average_precision_score(y_true, y_score)
        roc = roc_auc_score(y_true, y_score)

        # Expected Phase 1 values: AP = 0.650975, ROC-AUC = 0.947387
        self.assertAlmostEqual(ap, 0.650975, places=5)
        self.assertAlmostEqual(roc, 0.947387, places=5)

    def test_provenance_and_model_parameters_recorded(self) -> None:
        """Verify provenance captures model parameters, scaler configuration, and checksums."""
        with open(self.prov_path, "r", encoding="utf-8") as f:
            prov = json.load(f)

        self.assertIn("input_artifacts", prov)
        self.assertIn("phase1_gmm_model", prov["input_artifacts"])
        self.assertIn("phase1_scaler", prov["input_artifacts"])

        gmm_meta = prov["input_artifacts"]["phase1_gmm_model"]
        self.assertEqual(gmm_meta["parameters"]["n_components"], 14)
        self.assertEqual(gmm_meta["parameters"]["covariance_type"], "full")
        self.assertEqual(gmm_meta["parameters"]["reg_covar"], 1e-6)
        self.assertEqual(gmm_meta["parameters"]["random_state"], 42)
        self.assertTrue(gmm_meta["convergence"]["converged"])

        scaler_meta = prov["input_artifacts"]["phase1_scaler"]
        self.assertEqual(scaler_meta["n_features_in"], 29)

    def test_evaluate_score_files_end_to_end(self) -> None:
        """Verify evaluate_score_files integration runs cleanly and produces triage tables."""
        res = evaluate_score_files(
            self.val_scores_path,
            self.val_labels_path,
            self.test_scores_path,
            self.test_labels_path,
            model="gmm",
            score_direction="higher_is_more_suspicious",
        )
        val_row = res.ranking_metrics.loc[res.ranking_metrics["split"] == "validation"].iloc[0]
        self.assertAlmostEqual(float(val_row["average_precision_ap"]), 0.650975, places=5)
        self.assertAlmostEqual(float(val_row["roc_auc"]), 0.947387, places=5)

        test_row = res.ranking_metrics.loc[res.ranking_metrics["split"] == "test"].iloc[0]
        self.assertAlmostEqual(float(test_row["average_precision_ap"]), 0.677302, places=5)
        self.assertAlmostEqual(float(test_row["roc_auc"]), 0.906694, places=5)

        self.assertEqual(len(res.selections), 3)


if __name__ == "__main__":
    unittest.main()

