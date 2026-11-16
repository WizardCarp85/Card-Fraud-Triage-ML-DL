"""Unit and integration tests for Phase 2 workflow orchestration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ArtifactConfig,
    AutoencoderConfig,
    DataConfig,
    GMMConfig,
    Phase2Config,
    PreprocessingConfig,
)
from src.data import DatasetNotFoundError, ManifestNotFoundError, validate_data_inputs
from src.models import get_model
from src.models.autoencoder import AutoencoderAnomalyModel
from src.models.gmm import GMMAnomalyModel
from src.preprocessing import PreprocessingPipeline
from src.provenance import generate_provenance
from src.run import execute_pipeline, setup_run_directory


class OrchestrationTests(unittest.TestCase):
    """Test suite for configuration, failure modes, models, and orchestration pipeline."""

    def test_config_defaults_and_roundtrip(self) -> None:
        """Verify default configuration values and JSON serialization roundtrip."""
        config = Phase2Config()
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.data.target_column, "Class")
        self.assertEqual(config.preprocessing.scaler_type, "robust")
        self.assertEqual(config.gmm.n_components, 14)
        self.assertEqual(config.autoencoder.latent_dim, 8)

        json_str = config.to_json()
        restored = Phase2Config.from_dict(json.loads(json_str))
        self.assertEqual(config.to_dict(), restored.to_dict())

    def test_missing_dataset_raises_dataset_not_found(self) -> None:
        """Verify that a missing dataset triggers a clear, actionable exception."""
        cfg = DataConfig(raw_data_path="nonexistent/creditcard.csv")
        with self.assertRaises(DatasetNotFoundError) as ctx:
            validate_data_inputs(cfg)
        self.assertIn("Action required", str(ctx.exception))
        self.assertIn("creditcard.csv", str(ctx.exception))

    def test_missing_manifest_raises_manifest_not_found(self) -> None:
        """Verify that a missing manifest triggers a clear, actionable exception."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_csv = Path(tmp_dir) / "creditcard.csv"
            fake_csv.write_text("Time,Amount,Class\n0,10,0\n", encoding="utf-8")
            cfg = DataConfig(
                raw_data_path=str(fake_csv),
                manifest_path=str(Path(tmp_dir) / "missing_manifest.json"),
            )
            with self.assertRaises(ManifestNotFoundError) as ctx:
                validate_data_inputs(cfg)
            self.assertIn("freeze_split.py", str(ctx.exception))

    def test_run_directory_collision_prevention(self) -> None:
        """Verify that prior runs cannot be silently overwritten without explicit flag."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            art_cfg = ArtifactConfig(
                base_output_dir=tmp_dir,
                run_id="test_run_unique",
                overwrite=False,
            )
            config = Phase2Config(artifacts=art_cfg)

            # First setup succeeds
            run_dir = setup_run_directory(config)
            self.assertTrue(run_dir.is_dir())

            # Second setup without overwrite must fail clearly
            with self.assertRaises(FileExistsError) as ctx:
                setup_run_directory(config)
            self.assertIn("already exists", str(ctx.exception))
            self.assertIn("--overwrite", str(ctx.exception))

            # With overwrite=True, setup succeeds
            overwrite_cfg = ArtifactConfig(
                base_output_dir=tmp_dir,
                run_id="test_run_unique",
                overwrite=True,
            )
            config_overwrite = Phase2Config(artifacts=overwrite_cfg)
            run_dir_ov = setup_run_directory(config_overwrite)
            self.assertEqual(run_dir, run_dir_ov)

    def test_gmm_model_interface(self) -> None:
        """Verify GMM model conforms to BaseAnomalyModel lifecycle."""
        cfg = GMMConfig(n_components=2, max_iter=20, n_init=1, random_state=42)
        model = GMMAnomalyModel(cfg)
        self.assertEqual(model.name, "gmm")

        # Synthetic normal data
        rng = np.random.default_rng(42)
        X_train = rng.normal(0, 1, size=(200, 29))
        X_test = rng.normal(0, 1, size=(50, 29))

        model.fit(X_train)
        scores = model.score(X_test)

        self.assertEqual(scores.shape, (50,))
        self.assertTrue(np.all(np.isfinite(scores)))

        # Save and reload
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save(tmp_dir)
            reloaded = GMMAnomalyModel(cfg).load(tmp_dir)
            reloaded_scores = reloaded.score(X_test)
            np.testing.assert_allclose(scores, reloaded_scores)

    def test_autoencoder_model_interface(self) -> None:
        """Verify Autoencoder conforms to BaseAnomalyModel lifecycle."""
        cfg = AutoencoderConfig(
            input_dim=29,
            latent_dim=4,
            hidden_dims=(16, 8),
            epochs=2,
            batch_size=32,
            random_state=42,
        )
        model = AutoencoderAnomalyModel(cfg)
        self.assertEqual(model.name, "autoencoder")

        rng = np.random.default_rng(42)
        X_train = rng.normal(0, 1, size=(100, 29))
        X_test = rng.normal(0, 1, size=(30, 29))

        model.fit(X_train)
        scores = model.score(X_test)

        self.assertEqual(scores.shape, (30,))
        self.assertTrue(np.all(np.isfinite(scores)))
        # Non-negative MSE scores
        self.assertTrue(np.all(scores >= 0))

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save(tmp_dir)
            reloaded = AutoencoderAnomalyModel(cfg).load(tmp_dir)
            reloaded_scores = reloaded.score(X_test)
            np.testing.assert_allclose(scores, reloaded_scores, rtol=1e-5)

    def test_model_factory(self) -> None:
        """Verify get_model factory resolves valid models and rejects invalid ones."""
        config = Phase2Config()
        gmm = get_model("gmm", config)
        self.assertIsInstance(gmm, GMMAnomalyModel)

        ae = get_model("autoencoder", config)
        self.assertIsInstance(ae, AutoencoderAnomalyModel)

        with self.assertRaises(ValueError):
            get_model("unsupported_model", config)

    def test_preprocessing_pipeline_purity(self) -> None:
        """Verify feature engineering and strict unsupervised scaler isolation."""
        cfg = PreprocessingConfig(scaler_type="robust", fit_on_legit_only=True)
        pipeline = PreprocessingPipeline(cfg)

        # Create synthetic splits with 29 features: V1..V28 + Amount + Class
        v_data = {f"V{i}": np.random.randn(50) for i in range(1, 29)}
        v_data["Amount"] = np.random.exponential(50, size=50)
        v_data["Class"] = np.zeros(50, dtype=int)

        train_legit = pd.DataFrame(v_data)
        train_all = train_legit.copy()
        # Add a fraud row to train_all
        fraud_row = pd.DataFrame(
            {**{f"V{i}": [1.0] for i in range(1, 29)}, "Amount": [100.0], "Class": [1]}
        )
        train_all = pd.concat([train_all, fraud_row], ignore_index=True)

        val_df = pd.concat([train_legit.iloc[:20].copy(), fraud_row], ignore_index=True)
        test_df = pd.concat([train_legit.iloc[20:40].copy(), fraud_row], ignore_index=True)

        splits = {
            "train": train_all,
            "train_legit": train_legit,
            "val": val_df,
            "test": test_df,
        }

        prep_data = pipeline.fit_transform_splits(splits)
        self.assertEqual(prep_data["X_train_legit"].shape[1], 29)
        self.assertEqual(prep_data["X_val"].shape[1], 29)
        self.assertEqual(prep_data["X_test"].shape[1], 29)
        self.assertIn("Amount_log", prep_data["feature_names"])

    def test_provenance_audit(self) -> None:
        """Verify provenance records environment, versions, and hashes."""
        config = Phase2Config()
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_data = Path(tmp_dir) / "data.csv"
            dummy_data.write_text("test", encoding="utf-8")
            dummy_manifest = Path(tmp_dir) / "manifest.json"
            dummy_manifest.write_text("{}", encoding="utf-8")

            prov = generate_provenance(config.to_dict(), dummy_data, dummy_manifest)
            self.assertIn("timestamp_utc", prov)
            self.assertIn("git", prov)
            self.assertIn("packages", prov)
            self.assertIn("inputs", prov)
            self.assertIsNotNone(prov["inputs"]["source_data_sha256"])


    def test_end_to_end_pipeline_execution(self) -> None:
        """Verify execute_pipeline runs end-to-end and creates isolated artifacts."""
        from src.config import EvaluationConfig

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create synthetic creditcard.csv (100 rows, 31 columns)
            cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
            data_arr = np.random.randn(100, 31)
            data_arr[:, 0] = np.arange(100)  # Time
            data_arr[:, 29] = np.random.exponential(20, 100)  # Amount
            data_arr[:, 30] = 0  # Class 0
            data_arr[70:75, 30] = 1  # 5 fraud rows in val
            data_arr[90:95, 30] = 1  # 5 fraud rows in test

            df_syn = pd.DataFrame(data_arr, columns=cols)
            syn_csv = tmp_path / "creditcard.csv"
            df_syn.to_csv(syn_csv, index=False)

            # Create synthetic manifest
            manifest_data = {
                "manifest_version": "1.0.0",
                "split_indices": {
                    "train": list(range(60)),
                    "train_legit": list(range(60)),  # all legit
                    "val": list(range(60, 80)),
                    "test": list(range(80, 100)),
                },
            }
            syn_manifest = tmp_path / "manifest.json"
            with open(syn_manifest, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f)

            cfg = Phase2Config(
                data=DataConfig(
                    raw_data_path=str(syn_csv),
                    manifest_path=str(syn_manifest),
                ),
                gmm=GMMConfig(n_components=2, max_iter=10, n_init=1, random_state=42),
                evaluation=EvaluationConfig(
                    fpr_budgets=(0.1, 0.2),
                    review_capacities=(5, 10),
                ),
                artifacts=ArtifactConfig(
                    base_output_dir=str(tmp_path / "artifacts"),
                    run_id="syn_run",
                ),
                seed=42,
            )

            run_dir = execute_pipeline(cfg, models_to_run=["gmm"])

            self.assertTrue(run_dir.is_dir())
            self.assertTrue((run_dir / "config.json").is_file())
            self.assertTrue((run_dir / "provenance.json").is_file())
            self.assertTrue((run_dir / "run_manifest.json").is_file())
            self.assertTrue((run_dir / "preprocessed" / "scaler_robust.joblib").is_file())
            self.assertTrue((run_dir / "models" / "gmm" / "gmm_model.joblib").is_file())
            self.assertTrue((run_dir / "scores" / "scores_val.csv").is_file())
            self.assertTrue((run_dir / "scores" / "scores_test.csv").is_file())
            self.assertTrue((run_dir / "tables" / "final_ranking_metrics.csv").is_file())
            self.assertTrue((run_dir / "tables" / "test_triage_table.csv").is_file())


if __name__ == "__main__":
    unittest.main()
