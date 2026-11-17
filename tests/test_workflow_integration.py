"""End-to-end integration and regression tests for Phase 2 data-to-score workflow."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

import evaluation
from src.boundary_checks import (
    BoundaryValidationError,
    execute_frozen_test_evaluation,
    validate_feature_contract,
    validate_loader_streams,
    validate_score_contract,
    validate_split_isolation,
)
from src.config import (
    ArtifactConfig,
    AutoencoderConfig,
    DataConfig,
    EvaluationConfig,
    GMMConfig,
    Phase2Config,
    PreprocessingConfig,
)
from src.loaders import make_autoencoder_loaders
from src.run import execute_pipeline


class WorkflowIntegrationTests(unittest.TestCase):
    """End-to-end regression tests across all stage boundaries of Phase 2 pipeline."""

    def setUp(self) -> None:
        """Create a deterministic synthetic dataset fixture (100 rows, 31 cols)."""
        np.random.seed(42)
        self.n_rows = 100
        cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]

        data = np.random.randn(self.n_rows, 31)
        data[:, 0] = np.arange(self.n_rows)  # Time
        data[:, 29] = np.random.exponential(25.0, self.n_rows)  # Amount
        data[:, 30] = 0  # Legit by default

        # Inject fraud strictly into validation and test partitions
        data[65:70, 30] = 1  # 5 frauds in val
        data[85:90, 30] = 1  # 5 frauds in test

        self.df = pd.DataFrame(data, columns=cols)
        self.df["Class"] = self.df["Class"].astype(int)

        self.manifest = {
            "manifest_version": "1.0.0",
            "split_indices": {
                "train": list(range(60)),
                "train_legit": list(range(60)),  # Strictly 0 fraud
                "val": list(range(60, 80)),
                "test": list(range(80, 100)),
            },
        }

    def test_end_to_end_pipeline_with_boundary_checks(self) -> None:
        """Execute full data-to-score pipeline on synthetic fixture exercising all stage contracts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_path = tmp_path / "creditcard.csv"
            manifest_path = tmp_path / "split_manifest.json"

            self.df.to_csv(csv_path, index=False)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(self.manifest, f)

            config = Phase2Config(
                seed=42,
                data=DataConfig(
                    raw_data_path=str(csv_path),
                    manifest_path=str(manifest_path),
                ),
                preprocessing=PreprocessingConfig(scaler_type="robust", fit_on_legit_only=True),
                gmm=GMMConfig(n_components=2, max_iter=10, n_init=1, random_state=42),
                autoencoder=AutoencoderConfig(
                    epochs=2,
                    batch_size=16,
                    latent_dim=4,
                    hidden_dims=(8,),
                    random_state=42,
                ),
                evaluation=EvaluationConfig(review_capacities=(5, 10)),
                artifacts=ArtifactConfig(
                    base_output_dir=str(tmp_path / "artifacts"),
                    run_id="syn_integration_run",
                    overwrite=True,
                ),
            )

            # Run full integrated pipeline for both models
            run_dir = execute_pipeline(config, models_to_run=["gmm", "autoencoder"])
            self.assertTrue(run_dir.is_dir())

            # Verify artifact tree layout
            self.assertTrue((run_dir / "preprocessed" / "scaler_robust.joblib").is_file())
            self.assertTrue((run_dir / "preprocessed" / "validation_labels.csv").is_file())
            self.assertTrue((run_dir / "preprocessed" / "test_labels.csv").is_file())
            self.assertTrue((run_dir / "run_manifest.json").is_file())
            self.assertTrue((run_dir / "provenance.json").is_file())

            for model_name in ("gmm", "autoencoder"):
                model_dir = run_dir / model_name
                self.assertTrue((model_dir / "scores" / "scores_val.csv").is_file())
                self.assertTrue((model_dir / "scores" / "scores_test.csv").is_file())
                self.assertTrue((model_dir / "tables" / "final_ranking_metrics.csv").is_file())
                self.assertTrue((model_dir / "tables" / "test_threshold_metrics.csv").is_file())

            # Check run manifest content
            with open(run_dir / "run_manifest.json", "r", encoding="utf-8") as f:
                manifest_json = json.load(f)
            self.assertEqual(manifest_json["status"], "COMPLETED")
            self.assertEqual(set(manifest_json["models_evaluated"]), {"gmm", "autoencoder"})

    def test_boundary_check_split_isolation_rejects_fraud_in_train_legit(self) -> None:
        """Verify split isolation fails fast when fraud is present in train_legit."""
        contaminated_splits = {
            "train": self.df.iloc[:60].copy(),
            "train_legit": self.df.iloc[:60].copy(),
            "val": self.df.iloc[60:80].copy(),
            "test": self.df.iloc[80:100].copy(),
        }
        # Inject 1 fraud into train_legit
        contaminated_splits["train_legit"].iloc[0, self.df.columns.get_loc("Class")] = 1

        with self.assertRaises(BoundaryValidationError) as ctx:
            validate_split_isolation(contaminated_splits, self.manifest)
        self.assertIn("train_legit contains 1 fraud rows", str(ctx.exception))

    def test_boundary_check_split_isolation_rejects_overlapping_splits(self) -> None:
        """Verify split isolation detects row ID overlaps across splits."""
        overlapping_splits = {
            "train": self.df.iloc[:60].copy(),
            "train_legit": self.df.iloc[:60].copy(),
            "val": self.df.iloc[55:75].copy(),  # Overlaps rows 55..59 with train
            "test": self.df.iloc[80:100].copy(),
        }
        with self.assertRaises(BoundaryValidationError) as ctx:
            validate_split_isolation(overlapping_splits)
        self.assertIn("Split contamination", str(ctx.exception))

    def test_boundary_check_feature_contract_rejects_forbidden_columns(self) -> None:
        """Verify feature contract strictly rejects Class or Time columns in feature space."""
        prep_data = {
            "feature_names": [f"V{i}" for i in range(1, 29)] + ["Class"],  # Forbidden Class
            "X_train_legit": np.zeros((60, 29)),
            "X_val": np.zeros((20, 29)),
            "X_test": np.zeros((20, 29)),
        }
        with self.assertRaises(BoundaryValidationError) as ctx:
            validate_feature_contract(prep_data)
        self.assertIn("Feature contract violation", str(ctx.exception))

    def test_boundary_check_feature_contract_rejects_dimension_mismatch(self) -> None:
        """Verify feature contract rejects non-29-dimensional matrices."""
        prep_data = {
            "feature_names": [f"V{i}" for i in range(1, 29)] + ["Amount_log"],
            "X_train_legit": np.zeros((60, 28)),  # Wrong shape (28 instead of 29)
            "X_val": np.zeros((20, 28)),
            "X_test": np.zeros((20, 28)),
        }
        with self.assertRaises(BoundaryValidationError) as ctx:
            validate_feature_contract(prep_data)
        self.assertIn("violates (N, 29) contract", str(ctx.exception))

    def test_boundary_check_loader_streams_and_label_isolation(self) -> None:
        """Verify loader bundle enforces 4 streams, (x, row_id) batch contracts, and zero labels."""
        splits = {
            "train_legit": {
                "X": np.random.randn(60, 29),
                "y": np.zeros(60, dtype=int),
                "row_ids": np.arange(60),
            },
            "val": {
                "X": np.random.randn(20, 29),
                "y": np.zeros(20, dtype=int),
                "row_ids": np.arange(60, 80),
            },
            "test": {
                "X": np.random.randn(20, 29),
                "y": np.zeros(20, dtype=int),
                "row_ids": np.arange(80, 100),
            },
        }
        bundle = make_autoencoder_loaders(
            splits,
            self.manifest,
            batch_size=16,
            seed=42,
            feature_names=[f"V{i}" for i in range(1, 29)] + ["Amount_log"],
        )
        validate_loader_streams(bundle, self.manifest)

    def test_boundary_check_score_contract_enforces_schema_and_finiteness(self) -> None:
        """Verify score contract validates required columns, finite scores, and expected IDs."""
        score_df = pd.DataFrame({
            "row_id": np.arange(60, 80),
            "split": ["validation"] * 20,
            "model": ["gmm"] * 20,
            "anomaly_score": np.random.exponential(1.0, 20),
        })
        validate_score_contract(score_df, "gmm", "validation", np.arange(60, 80))

        # Test failure on NaN score
        nan_df = score_df.copy()
        nan_df.iloc[0, nan_df.columns.get_loc("anomaly_score")] = np.nan
        with self.assertRaises(BoundaryValidationError) as ctx:
            validate_score_contract(nan_df, "gmm", "validation", np.arange(60, 80))
        self.assertIn("non-finite values", str(ctx.exception))

        # Test failure on duplicated row ID
        dup_df = score_df.copy()
        dup_df.iloc[1, dup_df.columns.get_loc("row_id")] = dup_df.iloc[0]["row_id"]
        with self.assertRaises(BoundaryValidationError) as ctx:
            validate_score_contract(dup_df, "gmm", "validation", np.arange(60, 80))
        self.assertIn("Duplicate row IDs", str(ctx.exception))

    def test_frozen_evaluation_path_cannot_trigger_training_or_calibration(self) -> None:
        """Verify the frozen evaluation path never triggers training or threshold calibration."""
        # Patch fit and select_threshold_at_fpr to ensure they are NEVER invoked
        with (
            patch.object(
                GaussianMixture,
                "fit",
                side_effect=AssertionError("Training invoked during frozen evaluation!"),
            ),
            patch.object(
                evaluation,
                "select_threshold_at_fpr",
                side_effect=AssertionError("Threshold calibration invoked during frozen evaluation!"),
            ),
            tempfile.TemporaryDirectory() as tmp_dir,
        ):
            out = execute_frozen_test_evaluation(output_dir=tmp_dir)
            self.assertIn("threshold_metrics", out)
            self.assertIn("triage_table", out)
            self.assertIn("manifest", out)

            # Check that thresholds match the frozen manifests for GMM and AE
            thresh_df = out["threshold_metrics"]
            self.assertEqual(set(thresh_df["model"].unique()), {"gmm", "autoencoder"})
            self.assertEqual(len(thresh_df), 6)  # 2 models * 3 budgets

            # Verify manifest records zero training
            self.assertFalse(out["manifest"]["training_performed"])
            self.assertIn("frozen validation thresholds", out["manifest"]["threshold_selection"])


if __name__ == "__main__":
    unittest.main()
