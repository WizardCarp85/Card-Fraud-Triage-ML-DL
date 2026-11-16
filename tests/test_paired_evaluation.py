"""Paired evaluation regressions using synthetic score exports."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.paired_evaluation import compare_score_files, load_frozen_policy


class PairedTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.ids = [str(i) for i in range(1001)]
        labels = pd.DataFrame({"row_id": self.ids, "split": "test", "Class": [1, 1] + [0] * 999})
        self.labels = self.root / "labels.csv"
        labels.to_csv(self.labels, index=False)
        self.paths = {}
        self.policies = {}
        for model in ("gmm", "autoencoder"):
            scores = pd.DataFrame({
                "row_id": self.ids, "split": "test", "model": model,
                "anomaly_score": [3, 2] + [0] * 999,
            }).iloc[::-1]
            self.paths[model] = self.root / f"{model}.csv"
            scores.to_csv(self.paths[model], index=False)
            self.policies[model] = {
                "score_direction": "higher_is_more_suspicious",
                "thresholds": [{
                    "fpr_budget": budget, "threshold": 2.5, "status": "ok",
                    "decision": "score_gte_threshold", "validation_fpr": 0,
                } for budget in (0.001, 0.01, 0.05)],
            }

    def evaluate(self):
        return compare_score_files(self.paths, self.labels, self.policies, self.ids)[0]

    def test_metrics_apply_frozen_threshold_without_selection(self):
        with patch("evaluation.select_threshold_at_fpr", side_effect=AssertionError("Recalibration")):
            tables = self.evaluate()
        self.assertEqual(tables["ranking"].average_precision_ap.tolist(), [1, 1])
        self.assertEqual(tables["thresholds"].frauds_caught.tolist(), [1] * 6)
        self.assertEqual(tables["thresholds"].missed_frauds.tolist(), [1] * 6)
        self.assertEqual(tables["overlap"].both_caught.tolist(), [2] * 3)
        self.assertEqual(tables["overlap"].a_only.tolist(), [0] * 3)

    def test_no_alert_policy_keeps_status(self):
        for policy in self.policies.values():
            for row in policy["thresholds"]:
                row.update(status="no_alerts_within_budget", decision="no_alerts", threshold=None)
        result = self.evaluate()["thresholds"]
        self.assertEqual(result.flagged_transactions.tolist(), [0] * 6)
        self.assertTrue(result.threshold.isna().all())

    def test_mismatched_membership_rejected(self):
        scores = pd.read_csv(self.paths["autoencoder"])
        scores.loc[0, "row_id"] = 99999
        scores.to_csv(self.paths["autoencoder"], index=False)
        with self.assertRaisesRegex(ValueError, "exactly the frozen"):
            self.evaluate()

    def test_unique_catches_and_tie_order(self):
        scores = pd.read_csv(self.paths["autoencoder"], dtype={"row_id": str})
        scores["anomaly_score"] = 0
        scores.loc[scores.row_id == "0", "anomaly_score"] = 3
        scores.to_csv(self.paths["autoencoder"], index=False)
        first = self.evaluate()["triage"]
        scores.iloc[::-1].to_csv(self.paths["autoencoder"], index=False)
        self.assertTrue(first.equals(self.evaluate()["triage"]))

    def test_missing_freeze_fails_before_output(self):
        with self.assertRaises(FileNotFoundError):
            load_frozen_policy(self.root / "missing.json", "autoencoder")

    def test_tampered_provenance_rejected(self):
        identity = {"path": str(self.labels), "sha256": "wrong"}
        policy = {
            "schema_version": "1.0", "model": "gmm", "calibration_split": "validation",
            "score_direction": "higher_is_more_suspicious",
            "provenance": {name: identity for name in (
                "model_artifact", "preprocessing_artifact", "split_manifest",
                "validation_scores", "validation_labels")},
        }
        path = self.root / "policy.json"
        path.write_text(json.dumps(policy))
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            load_frozen_policy(path, "gmm")


if __name__ == "__main__":
    unittest.main()
