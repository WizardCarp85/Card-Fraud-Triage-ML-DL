"""Unit tests for Phase 2 dataset split reproduction, integrity, and manifest contract."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from data.freeze_split import (
    EXPECTED_DEDUP_ROWS,
    EXPECTED_DUPLICATE_ROWS,
    EXPECTED_SOURCE_MD5,
    EXPECTED_SOURCE_ROWS,
    EXPECTED_SOURCE_SHA256,
    EXPECTED_TEST_FRAUD,
    EXPECTED_TEST_LEGIT,
    EXPECTED_TEST_ROWS,
    EXPECTED_TOTAL_FRAUD,
    EXPECTED_TOTAL_LEGIT,
    EXPECTED_TRAIN_FRAUD,
    EXPECTED_TRAIN_LEGIT,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_VAL_FRAUD,
    EXPECTED_VAL_LEGIT,
    EXPECTED_VAL_ROWS,
    compute_array_hash,
    compute_file_checksums,
    deduplicate_dataset,
    load_raw_dataset,
    load_split_datasets,
    partition_splits,
)


class DatasetSplitTests(unittest.TestCase):
    """Test suite verifying data provenance, chronological split contract, and manifest."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.data_path = Path("data/creditcard.csv")
        cls.manifest_path = Path("data/split_manifest.json")
        cls.splits_dir = Path("data/splits")
        cls.df_raw = load_raw_dataset(cls.data_path)
        cls.df_dedup, cls.dup_indices = deduplicate_dataset(cls.df_raw)
        cls.splits = partition_splits(cls.df_dedup)

    def test_source_file_checksums(self) -> None:
        """Verify the raw source file matches the canonical Kaggle/ULB SHA-256 and MD5."""
        checksums = compute_file_checksums(self.data_path)
        self.assertEqual(checksums["sha256"], EXPECTED_SOURCE_SHA256)
        self.assertEqual(checksums["md5"], EXPECTED_SOURCE_MD5)

    def test_source_dimensions_and_columns(self) -> None:
        """Verify raw dataset shape and expected feature columns."""
        self.assertEqual(len(self.df_raw), EXPECTED_SOURCE_ROWS)
        self.assertEqual(self.df_raw.shape[1], 31)
        expected_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
        self.assertListEqual(list(self.df_raw.columns), expected_cols)
        self.assertEqual(self.df_raw.isna().sum().sum(), 0)

    def test_deduplication_hygiene(self) -> None:
        """Verify exact count of duplicate transactions removed."""
        self.assertEqual(len(self.dup_indices), EXPECTED_DUPLICATE_ROWS)
        self.assertEqual(len(self.df_dedup), EXPECTED_DEDUP_ROWS)
        self.assertEqual(self.df_dedup.duplicated().sum(), 0)

    def test_split_counts_match_phase1_figures(self) -> None:
        """Verify partition row counts match Phase 1 documented figures."""
        self.assertEqual(len(self.splits["train"]), EXPECTED_TRAIN_ROWS)
        self.assertEqual(len(self.splits["val"]), EXPECTED_VAL_ROWS)
        self.assertEqual(len(self.splits["test"]), EXPECTED_TEST_ROWS)
        self.assertEqual(len(self.splits["train_legit"]), EXPECTED_TRAIN_LEGIT)

    def test_mutual_exclusivity_no_overlap(self) -> None:
        """Verify that no transaction appears in more than one partition."""
        s_train = set(self.splits["train"])
        s_val = set(self.splits["val"])
        s_test = set(self.splits["test"])

        self.assertEqual(len(s_train & s_val), 0, "Train and Val share overlapping rows!")
        self.assertEqual(len(s_train & s_test), 0, "Train and Test share overlapping rows!")
        self.assertEqual(len(s_val & s_test), 0, "Val and Test share overlapping rows!")

    def test_partition_completeness(self) -> None:
        """Verify that the union of splits exactly reconstructs the deduplicated dataset."""
        union_splits = (
            set(self.splits["train"])
            | set(self.splits["val"])
            | set(self.splits["test"])
        )
        self.assertEqual(union_splits, set(self.df_dedup.index))
        self.assertEqual(len(union_splits), EXPECTED_DEDUP_ROWS)

    def test_fraud_and_legitimate_counts(self) -> None:
        """Verify fraud and legitimate counts per split match historical records."""
        df_train = self.df_raw.loc[self.splits["train"]]
        df_val = self.df_raw.loc[self.splits["val"]]
        df_test = self.df_raw.loc[self.splits["test"]]

        train_fraud = int(df_train["Class"].sum())
        val_fraud = int(df_val["Class"].sum())
        test_fraud = int(df_test["Class"].sum())

        self.assertEqual(train_fraud, EXPECTED_TRAIN_FRAUD)
        self.assertEqual(val_fraud, EXPECTED_VAL_FRAUD)
        self.assertEqual(test_fraud, EXPECTED_TEST_FRAUD)
        self.assertEqual(train_fraud + val_fraud + test_fraud, EXPECTED_TOTAL_FRAUD)

        self.assertEqual(len(df_train) - train_fraud, EXPECTED_TRAIN_LEGIT)
        self.assertEqual(len(df_val) - val_fraud, EXPECTED_VAL_LEGIT)
        self.assertEqual(len(df_test) - test_fraud, EXPECTED_TEST_LEGIT)
        self.assertEqual(
            (len(df_train) - train_fraud)
            + (len(df_val) - val_fraud)
            + (len(df_test) - test_fraud),
            EXPECTED_TOTAL_LEGIT,
        )

    def test_training_unsupervised_purity(self) -> None:
        """Verify that the legitimate training subset contains zero fraud records."""
        df_train_legit = self.df_raw.loc[self.splits["train_legit"]]
        self.assertEqual(int(df_train_legit["Class"].sum()), 0)
        self.assertEqual(len(df_train_legit), EXPECTED_TRAIN_LEGIT)
        # Verify it is a strict subset of train
        self.assertTrue(set(self.splits["train_legit"]).issubset(set(self.splits["train"])))

    def test_chronological_ordering_boundaries(self) -> None:
        """Verify temporal consistency and boundary timestamp constraints."""
        df_train = self.df_raw.loc[self.splits["train"]]
        df_val = self.df_raw.loc[self.splits["val"]]
        df_test = self.df_raw.loc[self.splits["test"]]

        self.assertLessEqual(df_train["Time"].max(), df_val["Time"].min())
        self.assertLessEqual(df_val["Time"].max(), df_test["Time"].min())
        self.assertEqual(df_train["Time"].max(), 132906.0)
        self.assertEqual(df_val["Time"].min(), 132906.0)
        self.assertEqual(df_val["Time"].max(), 151320.0)
        self.assertEqual(df_test["Time"].min(), 151320.0)

    def test_fresh_run_reproducibility(self) -> None:
        """Verify an independent fresh run generates identical split membership."""
        fresh_splits = partition_splits(self.df_dedup)
        for name in ("train", "val", "test", "train_legit"):
            self.assertListEqual(self.splits[name], fresh_splits[name])
            self.assertEqual(
                compute_array_hash(self.splits[name]),
                compute_array_hash(fresh_splits[name]),
            )

    def test_manifest_file_validity(self) -> None:
        """Verify split_manifest.json exists, is valid JSON, and matches split indices."""
        self.assertTrue(self.manifest_path.exists())
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["manifest_version"], "1.0.0")
        self.assertEqual(manifest["phase"], 2)
        self.assertEqual(
            manifest["source_dataset"]["checksums"]["sha256"],
            EXPECTED_SOURCE_SHA256,
        )
        self.assertEqual(
            manifest["split_statistics"]["train"]["total_rows"],
            EXPECTED_TRAIN_ROWS,
        )
        self.assertEqual(
            manifest["split_statistics"]["train_legitimate_only"]["fraud_rows"],
            0,
        )
        self.assertEqual(
            manifest["split_statistics"]["validation"]["fraud_rows"],
            EXPECTED_VAL_FRAUD,
        )
        self.assertEqual(
            manifest["split_statistics"]["test"]["fraud_rows"],
            EXPECTED_TEST_FRAUD,
        )

        # Check prior exposure documentation is present
        self.assertEqual(
            manifest["prior_phase1_holdout_exposure"]["status"],
            "EXPLICITLY_DOCUMENTED",
        )

        # Check indices match
        for name in ("train", "val", "test", "train_legit"):
            self.assertListEqual(manifest["split_indices"][name], self.splits[name])

    def test_load_split_datasets_helper(self) -> None:
        """Verify the load_split_datasets helper function loads identical frames."""
        loaded = load_split_datasets(self.data_path, self.manifest_path)
        self.assertEqual(len(loaded["train"]), EXPECTED_TRAIN_ROWS)
        self.assertEqual(len(loaded["train_legit"]), EXPECTED_TRAIN_LEGIT)
        self.assertEqual(len(loaded["val"]), EXPECTED_VAL_ROWS)
        self.assertEqual(len(loaded["test"]), EXPECTED_TEST_ROWS)
        self.assertEqual(int(loaded["train_legit"]["Class"].sum()), 0)


if __name__ == "__main__":
    unittest.main()
