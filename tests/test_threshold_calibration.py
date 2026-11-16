"""Validation-only GMM freeze and edge-case coverage."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.freeze_gmm_thresholds import build_manifest
from src.threshold_calibration import calibrate_validation


class CalibrationTests(unittest.TestCase):
    def calibrate(self, labels, scores):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            score_path, label_path = root / "scores.csv", root / "labels.csv"
            score_path.write_text("row_id,split,model,anomaly_score\n" + "".join(
                f"{i},validation,gmm,{score}\n" for i, score in enumerate(scores)))
            label_path.write_text("row_id,split,Class\n" + "".join(
                f"{i},validation,{label}\n" for i, label in enumerate(labels)))
            return calibrate_validation(score_path, label_path, model="gmm", score_transform="negative_log_likelihood")

    def test_ties_with_no_feasible_alerts_are_explicit(self):
        manifest = self.calibrate([0, 1], [1, 1])
        for row in manifest["thresholds"]:
            self.assertEqual(row["status"], "no_alerts_within_budget")
            self.assertEqual(row["decision"], "no_alerts")
            self.assertIsNone(row["threshold"])
            self.assertEqual(row["validation_recall"], 0)
        json.dumps(manifest, allow_nan=False)

    def test_undefined_classes_keep_reason(self):
        for labels, reason in (([0, 0], "undefined_no_positive"),
                               ([1, 1], "undefined_no_negative")):
            manifest = self.calibrate(labels, [1, 2])
            for row in manifest["thresholds"]:
                self.assertEqual(row["reason"], reason)
                self.assertEqual(row["decision"], "undefined")
                self.assertIsNone(row["threshold"])
                self.assertIsNone(row["validation_fpr"])
            json.dumps(manifest, allow_nan=False)

    def test_saved_manifest_reproduces_without_opening_test_artifacts(self):
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            if "test" in path.name:
                raise AssertionError(f"Test artifact read during calibration: {path}")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open):
            actual = build_manifest()
        saved = json.loads(Path("results/phase2/gmm/threshold_manifest.json").read_text())
        self.assertEqual(actual, saved)
        self.assertEqual(actual["validation_rows"], 42558)
        self.assertEqual(actual["validation_fraud_rows"], 55)
        self.assertEqual([r["true_positives"] for r in actual["thresholds"]], [42, 44, 48])
        for row in actual["thresholds"]:
            self.assertLessEqual(row["validation_fpr"], row["fpr_budget"])

    def test_modified_model_hash_is_rejected(self):
        with patch("scripts.freeze_gmm_thresholds.artifact_identity",
                   return_value={"path": "model", "sha256": "wrong"}):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                build_manifest()


if __name__ == "__main__":
    unittest.main()
