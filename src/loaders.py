"""Legitimate-only PyTorch data loaders for the Phase 2 autoencoder (issue #31).

Streams
-------
``train``         Legitimate training rows only (manifest ``train_legit``). Shuffled
                  deterministically from ``seed``.
``val_legit``     Legitimate validation rows only, in manifest order. Use it to monitor
                  reconstruction loss (early stopping, model selection).
``val_ranking``   All validation rows (legitimate and fraud), in manifest order. Yields no
                  labels. Use it to produce anomaly scores for ranking evaluation.
``test``          All test rows, in manifest order. Yields no labels.

Every batch is ``(x, row_id)``:
    x       float32 tensor, shape (batch, 29), the scaled feature vector
    row_id  int64 tensor, shape (batch,), 0-based raw row index of creditcard.csv

Labels are never part of a batch. They are kept on ``LoaderBundle.labels`` (keyed by
split, aligned with ``LoaderBundle.row_ids``) so evaluation can join scores to labels by
row ID after scoring.

Usage
-----
    from src.loaders import build_autoencoder_loaders

    bundle = build_autoencoder_loaders(batch_size=256, seed=42)
    for x, row_id in bundle.loaders["train"]:
        ...  # train on x; target is x
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

N_FEATURES = 29
STREAMS = ("train", "val_legit", "val_ranking", "test")


class LoaderValidationError(ValueError):
    """Raised when loader inputs violate a shape, dtype, finiteness or membership rule."""


class FeatureDataset(Dataset):
    """Holds a feature matrix and its row IDs. Returns ``(x, row_id)``; no labels."""

    def __init__(self, X: np.ndarray, row_ids: np.ndarray, name: str = "") -> None:
        _check_matrix(X, row_ids, name)
        self.name = name
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.row_ids = torch.from_numpy(np.ascontiguousarray(row_ids, dtype=np.int64))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.row_ids[i]


@dataclass
class LoaderBundle:
    """Loaders plus the metadata needed to export and evaluate scores."""

    loaders: dict[str, DataLoader]
    row_ids: dict[str, np.ndarray]
    labels: dict[str, np.ndarray]
    split_of_stream: dict[str, str] = field(
        default_factory=lambda: {
            "train": "train_legit",
            "val_legit": "val",
            "val_ranking": "val",
            "test": "test",
        }
    )
    feature_names: list[str] = field(default_factory=list)
    batch_size: int = 256
    seed: int = 42


def _check_matrix(X: np.ndarray, row_ids: np.ndarray, name: str) -> None:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[1] != N_FEATURES:
        raise LoaderValidationError(f"{name}: X must have shape (n, {N_FEATURES}), got {getattr(X, 'shape', None)}.")
    if not np.issubdtype(X.dtype, np.floating):
        raise LoaderValidationError(f"{name}: X must be floating point, got {X.dtype}.")
    if not np.isfinite(X).all():
        raise LoaderValidationError(f"{name}: X contains NaN or infinite values.")
    row_ids = np.asarray(row_ids)
    if row_ids.shape != (X.shape[0],):
        raise LoaderValidationError(f"{name}: row_ids shape {row_ids.shape} does not match X rows {X.shape[0]}.")
    if not np.issubdtype(row_ids.dtype, np.integer):
        raise LoaderValidationError(f"{name}: row_ids must be integers, got {row_ids.dtype}.")
    if len(np.unique(row_ids)) != len(row_ids):
        raise LoaderValidationError(f"{name}: row_ids contain duplicates.")


def _check_membership(name: str, row_ids: np.ndarray, expected: list[int] | np.ndarray) -> None:
    if not np.array_equal(np.asarray(row_ids, dtype=np.int64), np.asarray(expected, dtype=np.int64)):
        raise LoaderValidationError(f"{name}: row IDs do not match the split manifest exactly.")


def make_autoencoder_loaders(
    splits: dict[str, dict[str, np.ndarray]],
    manifest: dict[str, Any],
    batch_size: int = 256,
    seed: int = 42,
    feature_names: list[str] | None = None,
) -> LoaderBundle:
    """Build the four loader streams from scaled matrices and the split manifest.

    Args:
        splits: ``{"train_legit"|"val"|"test": {"X": (n,29) float, "y": (n,) int, "row_ids": (n,) int}}``.
            This is the output format of ``data.preprocessing.build_matrices`` (issue #28).
        manifest: the parsed ``data/split_manifest.json``.
        batch_size: batch size for every stream.
        seed: seed for the training shuffle. Same seed gives the same batch order.
    """
    if batch_size < 1:
        raise LoaderValidationError("batch_size must be >= 1.")
    idx = manifest["split_indices"]

    for name in ("train_legit", "val", "test"):
        if name not in splits:
            raise LoaderValidationError(f"Missing split '{name}'.")
        s = splits[name]
        _check_matrix(s["X"], s["row_ids"], name)
        if np.asarray(s["y"]).shape != (len(s["row_ids"]),):
            raise LoaderValidationError(f"{name}: labels are not aligned with row_ids.")
        _check_membership(name, s["row_ids"], idx[name])

    tl = splits["train_legit"]
    if np.asarray(tl["y"]).any():
        raise LoaderValidationError("train_legit contains fraud rows; training must be legitimate-only.")

    val = splits["val"]
    val_legit_mask = np.asarray(val["y"]) == 0

    datasets = {
        "train": FeatureDataset(tl["X"], tl["row_ids"], "train"),
        "val_legit": FeatureDataset(val["X"][val_legit_mask], val["row_ids"][val_legit_mask], "val_legit"),
        "val_ranking": FeatureDataset(val["X"], val["row_ids"], "val_ranking"),
        "test": FeatureDataset(splits["test"]["X"], splits["test"]["row_ids"], "test"),
    }

    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                            generator=generator, drop_last=False, num_workers=0),
    }
    for name in ("val_legit", "val_ranking", "test"):
        loaders[name] = DataLoader(datasets[name], batch_size=batch_size, shuffle=False,
                                   drop_last=False, num_workers=0)

    row_ids = {name: datasets[name].row_ids.numpy().copy() for name in STREAMS}
    labels = {
        "train": np.asarray(tl["y"], dtype=np.int64).copy(),
        "val_legit": np.asarray(val["y"], dtype=np.int64)[val_legit_mask].copy(),
        "val_ranking": np.asarray(val["y"], dtype=np.int64).copy(),
        "test": np.asarray(splits["test"]["y"], dtype=np.int64).copy(),
    }
    return LoaderBundle(
        loaders=loaders,
        row_ids=row_ids,
        labels=labels,
        feature_names=list(feature_names or []),
        batch_size=batch_size,
        seed=seed,
    )


def build_autoencoder_loaders(
    data_path: str | Path = "data/creditcard.csv",
    manifest_path: str | Path = "data/split_manifest.json",
    batch_size: int = 256,
    seed: int = 42,
) -> LoaderBundle:
    """Run the #28 preprocessing pipeline and return the autoencoder loaders."""
    from data.preprocessing import build_matrices

    built = build_matrices(data_path, manifest_path)
    splits = {
        "train_legit": built["train_legit"],
        "val": built["splits"]["val"],
        "test": built["splits"]["test"],
    }
    return make_autoencoder_loaders(
        splits, built["manifest"], batch_size=batch_size, seed=seed,
        feature_names=built["feature_names"],
    )


if __name__ == "__main__":
    bundle = build_autoencoder_loaders()
    for name in STREAMS:
        loader = bundle.loaders[name]
        x, rid = next(iter(loader))
        print(f"{name:12s} rows={len(loader.dataset):>7,}  batches={len(loader):>4}  "
              f"fraud_rows={int(bundle.labels[name].sum()):>3}  first batch x={tuple(x.shape)} {x.dtype}, "
              f"row_id={tuple(rid.shape)} {rid.dtype}")
