"""Data-layer tests for the shared 29-feature preprocessing pipeline (issues #28 and #40).

Covered invariants:
    * feature order and definition (V1..V28, Amount_log); Class and Time are never features
    * scaling provenance: the scaler is fitted on legitimate training rows only and is frozen
    * row-ID preservation and exact split membership against the manifest
    * label exclusion from model inputs
    * agreement with committed artifacts, the Phase 1 scaler, and the orchestration
      pipeline in ``src/preprocessing.py``, within documented tolerance
    * repeated runs are bit-identical
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from data.preprocessing import (
    FEATURE_ATOL,
    FEATURE_COLS,
    FEATURE_RTOL,
    PHASE1_SCALER_ATOL,
    PHASE1_SCALER_RTOL,
    SCALER_PARAM_ATOL,
    SCALER_PARAM_RTOL,
    PreprocessingError,
    build_feature_frame,
    build_matrices,
    fit_scaler,
    verify_matrices,
)

DATA_PATH = Path("data/creditcard.csv")
MANIFEST_PATH = Path("data/split_manifest.json")
ARTIFACT_DIR = Path("data/artifacts")


def _raw_frame(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {"Time": np.arange(n, dtype=float)}
    cols.update({f"V{i}": rng.normal(size=n) for i in range(1, 29)})
    cols["Amount"] = rng.exponential(50, size=n)
    cols["Class"] = np.zeros(n, dtype=int)
    # Deliberately shuffled column order: the pipeline must impose its own.
    df = pd.DataFrame(cols)
    return df[list(reversed(df.columns))]


class FeatureContractTests(unittest.TestCase):
    def test_feature_order_and_count(self) -> None:
        self.assertEqual(FEATURE_COLS, [f"V{i}" for i in range(1, 29)] + ["Amount_log"])
        feats = build_feature_frame(_raw_frame())
        self.assertEqual(list(feats.columns), FEATURE_COLS)
        self.assertEqual(feats.shape[1], 29)

    def test_class_and_time_excluded(self) -> None:
        feats = build_feature_frame(_raw_frame())
        self.assertNotIn("Class", feats.columns)
        self.assertNotIn("Time", feats.columns)
        self.assertNotIn("Amount", feats.columns)

    def test_amount_log_is_log1p(self) -> None:
        df = _raw_frame()
        df.loc[0, "Amount"] = 0.0
        feats = build_feature_frame(df)
        np.testing.assert_array_equal(feats["Amount_log"].to_numpy(), np.log1p(df["Amount"].to_numpy()))
        self.assertEqual(feats["Amount_log"].iloc[0], 0.0)

    def test_v_columns_unchanged_and_index_preserved(self) -> None:
        df = _raw_frame()
        df.index = np.arange(1000, 1000 + len(df))
        feats = build_feature_frame(df)
        np.testing.assert_array_equal(feats.index.to_numpy(), df.index.to_numpy())
        for c in FEATURE_COLS[:-1]:
            np.testing.assert_array_equal(feats[c].to_numpy(), df[c].to_numpy())

    def test_label_change_does_not_change_features(self) -> None:
        df = _raw_frame()
        flipped = df.copy()
        flipped["Class"] = 1
        pd.testing.assert_frame_equal(build_feature_frame(df), build_feature_frame(flipped))

    def test_invalid_inputs_rejected(self) -> None:
        cases = {
            "missing column": lambda d: d.drop(columns=["V7"]),
            "negative amount": lambda d: d.assign(Amount=-d["Amount"]),
            "nan": lambda d: d.assign(V3=np.nan),
            "inf": lambda d: d.assign(V5=np.inf),
        }
        for name, mutate in cases.items():
            with self.subTest(name), self.assertRaises(PreprocessingError):
                build_feature_frame(mutate(_raw_frame()))

    def test_fit_scaler_rejects_wrong_order_or_label(self) -> None:
        feats = build_feature_frame(_raw_frame())
        with self.assertRaises(PreprocessingError):
            fit_scaler(feats[list(reversed(FEATURE_COLS))])
        with self.assertRaises(PreprocessingError):
            fit_scaler(feats.assign(Class=0))

    def test_fit_scaler_matches_reference_robust_scaler(self) -> None:
        feats = build_feature_frame(_raw_frame())
        ours = fit_scaler(feats)
        ref = RobustScaler().fit(feats.to_numpy())
        np.testing.assert_array_equal(ours.center_, ref.center_)
        np.testing.assert_array_equal(ours.scale_, ref.scale_)


@unittest.skipUnless(DATA_PATH.is_file(), "data/creditcard.csv not present")
class RealDataPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text())
        cls.idx = cls.manifest["split_indices"]
        cls.built = build_matrices(DATA_PATH, MANIFEST_PATH)
        cls.raw = pd.read_csv(DATA_PATH)

    def test_verification_report_passes(self) -> None:
        report = verify_matrices(self.built)
        failed = [k for k, v in report["checks"].items() if not v]
        self.assertEqual(report["status"], "PASSED", failed)

    def test_row_ids_preserved_and_membership_exact(self) -> None:
        for name in ("train", "val", "test"):
            np.testing.assert_array_equal(self.built["splits"][name]["row_ids"], self.idx[name])
        np.testing.assert_array_equal(self.built["train_legit"]["row_ids"], self.idx["train_legit"])
        ids = [set(self.idx[n]) for n in ("train", "val", "test")]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])

    def test_rows_map_back_to_source_by_row_id(self) -> None:
        s = self.built["splits"]["val"]
        scaler = self.built["scaler"]
        pick = np.array([0, 1234, len(s["row_ids"]) - 1])
        raw_rows = build_feature_frame(self.raw.loc[s["row_ids"][pick]])
        np.testing.assert_allclose(scaler.transform(raw_rows.to_numpy()), s["X"][pick],
                                   rtol=FEATURE_RTOL, atol=FEATURE_ATOL)
        np.testing.assert_array_equal(self.raw.loc[s["row_ids"][pick], "Class"].to_numpy(), s["y"][pick])

    def test_scaler_fitted_on_legit_train_only(self) -> None:
        feats = build_feature_frame(self.raw)
        legit = RobustScaler().fit(feats.loc[self.idx["train_legit"]].to_numpy())
        with_fraud = RobustScaler().fit(feats.loc[self.idx["train"]].to_numpy())
        scaler = self.built["scaler"]
        np.testing.assert_array_equal(scaler.center_, legit.center_)
        np.testing.assert_array_equal(scaler.scale_, legit.scale_)
        # Fitting on all training rows (including 366 frauds) gives a detectably different scaler.
        self.assertFalse(np.array_equal(scaler.center_, with_fraud.center_))
        self.assertEqual(int(self.raw.loc[self.idx["train_legit"], "Class"].sum()), 0)

    def test_scaler_is_frozen_for_val_and_test(self) -> None:
        feats = build_feature_frame(self.raw)
        val_refit = RobustScaler().fit(feats.loc[self.idx["val"]].to_numpy())
        self.assertFalse(np.array_equal(self.built["scaler"].center_, val_refit.center_))
        expected = self.built["scaler"].transform(feats.loc[self.idx["test"]].to_numpy())
        np.testing.assert_array_equal(self.built["splits"]["test"]["X"], expected)

    def test_labels_not_in_model_inputs(self) -> None:
        for name in ("train", "val", "test"):
            s = self.built["splits"][name]
            self.assertEqual(s["X"].shape[1], 29)
            # The label is not recoverable as a column of X.
            for j in range(29):
                self.assertFalse(np.array_equal(s["X"][:, j], s["y"].astype(float)))
        self.assertEqual(self.built["feature_names"], FEATURE_COLS)

    def test_matches_committed_scaler_artifacts(self) -> None:
        meta = json.loads((ARTIFACT_DIR / "feature_metadata.json").read_text())
        self.assertEqual(meta["feature_names"], FEATURE_COLS)
        scaler = self.built["scaler"]
        np.testing.assert_allclose(scaler.center_, meta["scaler"]["center_"], rtol=SCALER_PARAM_RTOL, atol=SCALER_PARAM_ATOL)
        np.testing.assert_allclose(scaler.scale_, meta["scaler"]["scale_"], rtol=SCALER_PARAM_RTOL, atol=SCALER_PARAM_ATOL)
        saved = joblib.load(ARTIFACT_DIR / "scaler_robust_phase2.joblib")
        np.testing.assert_allclose(scaler.center_, saved.center_, rtol=SCALER_PARAM_RTOL, atol=SCALER_PARAM_ATOL)
        np.testing.assert_allclose(scaler.scale_, saved.scale_, rtol=SCALER_PARAM_RTOL, atol=SCALER_PARAM_ATOL)

    def test_matches_phase1_scaler(self) -> None:
        p1 = joblib.load("results/scaler_robust.joblib")
        np.testing.assert_allclose(self.built["scaler"].center_, p1.center_, rtol=PHASE1_SCALER_RTOL, atol=PHASE1_SCALER_ATOL)
        np.testing.assert_allclose(self.built["scaler"].scale_, p1.scale_, rtol=PHASE1_SCALER_RTOL, atol=PHASE1_SCALER_ATOL)

    def test_repeated_runs_bit_identical(self) -> None:
        again = build_matrices(DATA_PATH, MANIFEST_PATH)
        for name in ("train", "val", "test"):
            for key in ("X", "y", "row_ids"):
                np.testing.assert_array_equal(self.built["splits"][name][key], again["splits"][name][key])

    def test_orchestration_pipeline_produces_same_inputs(self) -> None:
        """src/preprocessing.py (used by run_phase2.py) must feed models the same matrices."""
        from src.config import DataConfig, PreprocessingConfig
        from src.data import load_split_partitions
        from src.preprocessing import PreprocessingPipeline

        parts = load_split_partitions(DataConfig(raw_data_path=str(DATA_PATH), manifest_path=str(MANIFEST_PATH)))
        prep = PreprocessingPipeline(PreprocessingConfig()).fit_transform_splits(parts)
        self.assertEqual(prep["feature_names"], FEATURE_COLS)
        pairs = {
            "X_train_legit": self.built["train_legit"]["X"],
            "X_train": self.built["splits"]["train"]["X"],
            "X_val": self.built["splits"]["val"]["X"],
            "X_test": self.built["splits"]["test"]["X"],
        }
        for key, ours in pairs.items():
            np.testing.assert_allclose(prep[key], ours, rtol=FEATURE_RTOL, atol=FEATURE_ATOL, err_msg=key)
        np.testing.assert_array_equal(prep["val_indices"], self.built["splits"]["val"]["row_ids"])
        np.testing.assert_array_equal(prep["test_indices"], self.built["splits"]["test"]["row_ids"])
        np.testing.assert_array_equal(prep["y_val"], self.built["splits"]["val"]["y"])


if __name__ == "__main__":
    unittest.main()
