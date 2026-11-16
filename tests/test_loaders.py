"""Tests for the legitimate-only autoencoder data loaders (issue #31)."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from src.loaders import (
    LoaderValidationError,
    build_autoencoder_loaders,
    make_autoencoder_loaders,
)


def _synthetic(seed: int = 0) -> tuple[dict, dict]:
    """Small fake splits: train_legit 50 legit, val 30 (3 fraud), test 20 (2 fraud)."""
    rng = np.random.default_rng(seed)

    def split(ids: list[int], fraud_ids: set[int]) -> dict:
        ids_arr = np.asarray(ids, dtype=np.int64)
        return {
            "X": rng.normal(size=(len(ids), 29)),
            "y": np.asarray([1 if i in fraud_ids else 0 for i in ids], dtype=np.int64),
            "row_ids": ids_arr,
        }

    train_ids = list(range(0, 50))
    val_ids = list(range(100, 130))
    test_ids = list(range(200, 220))
    splits = {
        "train_legit": split(train_ids, set()),
        "val": split(val_ids, {105, 110, 120}),
        "test": split(test_ids, {201, 219}),
    }
    manifest = {"split_indices": {"train_legit": train_ids, "val": val_ids, "test": test_ids}}
    return splits, manifest


def _collect(loader) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate (model input, row_id) over all batches. Train batches are (x_in, x_tgt, row_id)."""
    xs, ids = zip(*[(b[0], b[-1]) for b in loader])
    return torch.cat(xs), torch.cat(ids)


class LoaderTests(unittest.TestCase):
    def test_train_stream_is_legitimate_only(self) -> None:
        splits, manifest = _synthetic()
        b = make_autoencoder_loaders(splits, manifest, batch_size=8)
        _, ids = _collect(b.loaders["train"])
        self.assertEqual(sorted(ids.tolist()), manifest["split_indices"]["train_legit"])
        self.assertEqual(int(b.labels["train"].sum()), 0)

    def test_fraud_in_train_legit_is_rejected(self) -> None:
        splits, manifest = _synthetic()
        splits["train_legit"]["y"][3] = 1
        with self.assertRaises(LoaderValidationError):
            make_autoencoder_loaders(splits, manifest)

    def test_val_and_test_membership_match_manifest(self) -> None:
        splits, manifest = _synthetic()
        b = make_autoencoder_loaders(splits, manifest, batch_size=7)
        _, val_ids = _collect(b.loaders["val_ranking"])
        _, test_ids = _collect(b.loaders["test"])
        self.assertEqual(val_ids.tolist(), manifest["split_indices"]["val"])
        self.assertEqual(test_ids.tolist(), manifest["split_indices"]["test"])
        _, vl_ids = _collect(b.loaders["val_legit"])
        self.assertEqual(vl_ids.tolist(), [i for i in manifest["split_indices"]["val"] if i not in {105, 110, 120}])

    def test_membership_mismatch_is_rejected(self) -> None:
        splits, manifest = _synthetic()
        manifest["split_indices"]["test"] = manifest["split_indices"]["test"][:-1] + [999]
        with self.assertRaises(LoaderValidationError):
            make_autoencoder_loaders(splits, manifest)

    def test_batches_contain_no_labels(self) -> None:
        splits, manifest = _synthetic()
        b = make_autoencoder_loaders(splits, manifest, batch_size=8)
        for name, loader in b.loaders.items():
            batch = next(iter(loader))
            # train: (x_input, x_target, row_id); evaluation streams: (x, row_id). No labels.
            self.assertEqual(len(batch), 3 if name == "train" else 2, name)
            x, rid = batch[0], batch[-1]
            self.assertEqual(x.dtype, torch.float32)
            self.assertEqual(x.shape[1], 29)
            self.assertEqual(rid.dtype, torch.int64)

    def test_row_ids_align_features(self) -> None:
        splits, manifest = _synthetic()
        b = make_autoencoder_loaders(splits, manifest, batch_size=8)
        x, ids = _collect(b.loaders["train"])
        lookup = {int(r): i for i, r in enumerate(splits["train_legit"]["row_ids"])}
        expected = splits["train_legit"]["X"][[lookup[int(r)] for r in ids]].astype(np.float32)
        np.testing.assert_array_equal(x.numpy(), expected)

    def test_same_seed_reproduces_batch_order(self) -> None:
        splits, manifest = _synthetic()
        order = lambda seed: _collect(make_autoencoder_loaders(splits, manifest, batch_size=8, seed=seed).loaders["train"])[1].tolist()
        self.assertEqual(order(7), order(7))
        self.assertNotEqual(order(7), order(8))

    def test_batch_size_is_configurable(self) -> None:
        splits, manifest = _synthetic()
        b = make_autoencoder_loaders(splits, manifest, batch_size=16)
        self.assertEqual([len(batch[0]) for batch in b.loaders["train"]], [16, 16, 16, 2])

    def test_invalid_inputs_are_rejected(self) -> None:
        for mutate in (
            lambda s: s["val"].__setitem__("X", s["val"]["X"][:, :28]),
            lambda s: s["val"]["X"].__setitem__((0, 0), np.nan),
            lambda s: s["test"].__setitem__("X", s["test"]["X"].astype(np.int64)),
            lambda s: s["test"].__setitem__("row_ids", s["test"]["row_ids"].astype(np.float64)),
        ):
            splits, manifest = _synthetic()
            mutate(splits)
            with self.assertRaises(LoaderValidationError):
                make_autoencoder_loaders(splits, manifest)


@unittest.skipUnless(Path("data/creditcard.csv").is_file(), "data/creditcard.csv not present")
class RealDataLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import json

        cls.manifest = json.loads(Path("data/split_manifest.json").read_text())
        cls.bundle = build_autoencoder_loaders(batch_size=1024, seed=42)

    def test_counts_and_membership(self) -> None:
        idx = self.manifest["split_indices"]
        b = self.bundle
        self.assertEqual(b.row_ids["train"].tolist(), idx["train_legit"])
        self.assertEqual(b.row_ids["val_ranking"].tolist(), idx["val"])
        self.assertEqual(b.row_ids["test"].tolist(), idx["test"])
        self.assertEqual(int(b.labels["train"].sum()), 0)
        self.assertEqual(int(b.labels["val_legit"].sum()), 0)
        self.assertEqual(len(b.row_ids["val_legit"]), 42503)
        self.assertEqual(int(b.labels["val_ranking"].sum()), 55)
        self.assertEqual(int(b.labels["test"].sum()), 52)


if __name__ == "__main__":
    unittest.main()
