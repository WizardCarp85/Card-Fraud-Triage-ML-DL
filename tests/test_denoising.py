"""Tests for train-time denoising augmentation in the autoencoder loaders (issue #33)."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from src.loaders import LoaderValidationError, build_autoencoder_loaders, make_autoencoder_loaders
from tests.test_loaders import _synthetic

EVAL_STREAMS = ("val_legit", "val_ranking", "test")


def _epoch(loader) -> list[tuple[torch.Tensor, ...]]:
    return [tuple(t.clone() for t in batch) for batch in loader]


def _cat(batches, i: int) -> torch.Tensor:
    return torch.cat([b[i] for b in batches])


class DenoisingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.splits, self.manifest = _synthetic()
        self.clean_X = {k: v["X"].copy() for k, v in self.splits.items()}

    def _bundle(self, noise_std: float, seed: int = 42, noise_seed: int | None = None):
        return make_autoencoder_loaders(self.splits, self.manifest, batch_size=8, seed=seed,
                                        noise_std=noise_std, noise_seed=noise_seed)

    def test_noise_applied_to_train_inputs_only(self) -> None:
        batches = _epoch(self._bundle(0.5).loaders["train"])
        x_in, x_tgt, ids = _cat(batches, 0), _cat(batches, 1), _cat(batches, 2)
        self.assertFalse(torch.equal(x_in, x_tgt))
        # Targets are the exact clean scaled rows for those row IDs.
        lookup = {int(r): i for i, r in enumerate(self.splits["train_legit"]["row_ids"])}
        expected = self.clean_X["train_legit"][[lookup[int(r)] for r in ids]].astype(np.float32)
        np.testing.assert_array_equal(x_tgt.numpy(), expected)

    def test_eval_streams_are_clean_and_deterministic(self) -> None:
        noisy, clean = self._bundle(0.5), self._bundle(0.0)
        for name in EVAL_STREAMS:
            a, b = _epoch(noisy.loaders[name]), _epoch(noisy.loaders[name])
            c = _epoch(clean.loaders[name])
            for x in (a, b):
                self.assertTrue(torch.equal(_cat(x, 0), _cat(c, 0)), name)
                self.assertTrue(torch.equal(_cat(x, 1), _cat(c, 1)), name)
        val = _cat(_epoch(noisy.loaders["val_ranking"]), 0).numpy()
        np.testing.assert_array_equal(val, self.clean_X["val"].astype(np.float32))

    def test_zero_noise_reproduces_clean_path(self) -> None:
        batches = _epoch(self._bundle(0.0).loaders["train"])
        for x_in, x_tgt, _ in batches:
            self.assertTrue(torch.equal(x_in, x_tgt))
        # Same rows in the same order as the clean (no-noise) default.
        default = make_autoencoder_loaders(self.splits, self.manifest, batch_size=8, seed=42)
        d = _epoch(default.loaders["train"])
        self.assertTrue(torch.equal(_cat(batches, 0), _cat(d, 0)))
        self.assertTrue(torch.equal(_cat(batches, 2), _cat(d, 2)))

    def test_seeded_noise_is_reproducible(self) -> None:
        a, b = self._bundle(0.3), self._bundle(0.3)
        for _ in range(2):  # two epochs: fresh noise each epoch, identical across rebuilds
            ea, eb = _epoch(a.loaders["train"]), _epoch(b.loaders["train"])
            for i in range(3):
                self.assertTrue(torch.equal(_cat(ea, i), _cat(eb, i)))
        b2 = self._bundle(0.3)
        e1, e2 = _epoch(b2.loaders["train"]), _epoch(b2.loaders["train"])
        self.assertFalse(torch.equal(_cat(e1, 0), _cat(e2, 0)), "noise should be redrawn each epoch")

    def test_noise_seed_independent_of_shuffle(self) -> None:
        a = _epoch(self._bundle(0.3, noise_seed=1).loaders["train"])
        b = _epoch(self._bundle(0.3, noise_seed=2).loaders["train"])
        self.assertTrue(torch.equal(_cat(a, 2), _cat(b, 2)))  # same batch order
        self.assertTrue(torch.equal(_cat(a, 1), _cat(b, 1)))  # same clean targets
        self.assertFalse(torch.equal(_cat(a, 0), _cat(b, 0)))  # different noise

    def test_noise_scale_matches_config(self) -> None:
        splits, manifest = _synthetic()
        rng = np.random.default_rng(1)
        n = 4000
        ids = list(range(n))
        splits["train_legit"] = {"X": rng.normal(size=(n, 29)), "y": np.zeros(n, dtype=np.int64),
                                 "row_ids": np.arange(n, dtype=np.int64)}
        manifest["split_indices"]["train_legit"] = ids
        b = make_autoencoder_loaders(splits, manifest, batch_size=512, noise_std=0.2)
        e = _epoch(b.loaders["train"])
        diff = _cat(e, 0) - _cat(e, 1)
        self.assertAlmostEqual(float(diff.std()), 0.2, delta=0.005)
        self.assertAlmostEqual(float(diff.mean()), 0.0, delta=0.005)

    def test_labels_and_row_ids_unaffected(self) -> None:
        noisy, clean = self._bundle(0.5), self._bundle(0.0)
        for name in ("train",) + EVAL_STREAMS:
            np.testing.assert_array_equal(noisy.row_ids[name], clean.row_ids[name])
            np.testing.assert_array_equal(noisy.labels[name], clean.labels[name])
        self.assertTrue(torch.equal(_cat(_epoch(noisy.loaders["train"]), 2),
                                    _cat(_epoch(clean.loaders["train"]), 2)))
        for k, X in self.clean_X.items():  # source arrays are not modified in place
            np.testing.assert_array_equal(self.splits[k]["X"], X)

    def test_invalid_noise_is_rejected(self) -> None:
        for bad in (-0.1, float("nan"), float("inf"), "abc"):
            with self.assertRaises(LoaderValidationError):
                self._bundle(bad)  # type: ignore[arg-type]

    def test_bundle_records_noise_config(self) -> None:
        b = self._bundle(0.25, seed=3)
        self.assertEqual((b.noise_std, b.noise_seed, b.seed), (0.25, 3, 3))
        self.assertEqual(self._bundle(0.25, seed=3, noise_seed=9).noise_seed, 9)


@unittest.skipUnless(Path("data/creditcard.csv").is_file(), "data/creditcard.csv not present")
class RealDataDenoisingTests(unittest.TestCase):
    def test_real_data_noise_train_only(self) -> None:
        b = build_autoencoder_loaders(batch_size=4096, seed=42, noise_std=0.1)
        x_in, x_tgt, _ = next(iter(b.loaders["train"]))
        self.assertAlmostEqual(float((x_in - x_tgt).std()), 0.1, delta=0.002)
        clean = build_autoencoder_loaders(batch_size=4096, seed=42)
        for name in EVAL_STREAMS:
            self.assertTrue(torch.equal(next(iter(b.loaders[name]))[0], next(iter(clean.loaders[name]))[0]))


if __name__ == "__main__":
    unittest.main()
