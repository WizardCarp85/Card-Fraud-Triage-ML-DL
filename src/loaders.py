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

Batch format
------------
``train`` yields ``(x_input, x_target, row_id)``:
    x_input   float32 (batch, 29). The scaled features, plus Gaussian noise when
              ``noise_std > 0`` (denoising autoencoder, issue #33).
    x_target  float32 (batch, 29). The clean scaled features. Always the reconstruction target.
    row_id    int64 (batch,). The 0-based raw row index of creditcard.csv.
With ``noise_std == 0``, ``x_input`` equals ``x_target`` exactly (the clean-input path).

``val_legit``, ``val_ranking`` and ``test`` yield ``(x, row_id)``. They are always clean.

Denoising (issue #33)
---------------------
Noise is ``N(0, noise_std^2)``, drawn per element. It is added after the frozen #28
RobustScaler transform, so ``noise_std`` is in scaled feature units. For RobustScaler,
1.0 is one inter-quartile range of the legitimate training rows. New noise is drawn for
every training batch from a dedicated ``torch.Generator`` seeded with ``noise_seed``, so
rebuilding the loaders with the same seeds reproduces every epoch's noisy inputs exactly.

Labels are never part of a batch. They are kept on ``LoaderBundle.labels`` (keyed by
split, aligned with ``LoaderBundle.row_ids``) so evaluation can join scores to labels by
row ID after scoring.

Usage
-----
    from src.loaders import build_autoencoder_loaders

    bundle = build_autoencoder_loaders(batch_size=256, seed=42, noise_std=0.1)
    for x_input, x_target, row_id in bundle.loaders["train"]:
        loss = criterion(model(x_input), x_target)
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


class DenoisingCollate:
    """Collate for the training stream: returns ``(x_input, x_target, row_id)``.

    ``x_target`` is the clean batch. ``x_input`` is ``x_target`` plus ``N(0, noise_std^2)``
    noise drawn from a dedicated, seeded generator. When ``noise_std == 0`` no random
    numbers are drawn, and ``x_input`` is an exact copy of ``x_target``.
    """

    def __init__(self, noise_std: float = 0.0, noise_seed: int = 42) -> None:
        self.noise_std = _validate_noise_std(noise_std)
        self.noise_seed = int(noise_seed)
        self.generator = torch.Generator().manual_seed(self.noise_seed)

    def __call__(
        self, batch: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_target = torch.stack([b[0] for b in batch])  # new tensor; dataset is never modified
        row_id = torch.stack([b[1] for b in batch])
        if self.noise_std > 0.0:
            noise = torch.randn(x_target.shape, generator=self.generator, dtype=x_target.dtype)
            x_input = x_target + self.noise_std * noise
        else:
            x_input = x_target.clone()
        return x_input, x_target, row_id


def _validate_noise_std(noise_std: float) -> float:
    try:
        value = float(noise_std)
    except (TypeError, ValueError) as exc:
        raise LoaderValidationError(f"noise_std must be a number, got {noise_std!r}.") from exc
    if not np.isfinite(value) or value < 0.0:
        raise LoaderValidationError(f"noise_std must be finite and >= 0, got {noise_std!r}.")
    return value


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
    noise_std: float = 0.0
    noise_seed: int = 42


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
    noise_std: float = 0.0,
    noise_seed: int | None = None,
) -> LoaderBundle:
    """Build the four loader streams from scaled matrices and the split manifest.

    Args:
        splits: ``{"train_legit"|"val"|"test": {"X": (n,29) float, "y": (n,) int, "row_ids": (n,) int}}``.
            This is the output format of ``data.preprocessing.build_matrices`` (issue #28).
            ``X`` must already be transformed by the frozen #28 scaler.
        manifest: the parsed ``data/split_manifest.json``.
        batch_size: batch size for every stream.
        seed: seed for the training shuffle. Same seed gives the same batch order.
        noise_std: std of the Gaussian noise added to training inputs only, in scaled
            feature units. 0 (the default) disables denoising.
        noise_seed: seed for the noise generator. Defaults to ``seed``. It is independent of
            the shuffle generator, so changing the noise never changes the batch order.
    """
    if batch_size < 1:
        raise LoaderValidationError("batch_size must be >= 1.")
    noise_std = _validate_noise_std(noise_std)
    noise_seed = int(seed if noise_seed is None else noise_seed)
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
                            generator=generator, drop_last=False, num_workers=0,
                            collate_fn=DenoisingCollate(noise_std, noise_seed)),
    }
    # Evaluation streams use the default collate: clean, deterministic, never noised.
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
        noise_std=noise_std,
        noise_seed=noise_seed,
    )


def build_autoencoder_loaders(
    data_path: str | Path = "data/creditcard.csv",
    manifest_path: str | Path = "data/split_manifest.json",
    batch_size: int = 256,
    seed: int = 42,
    noise_std: float = 0.0,
    noise_seed: int | None = None,
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
        feature_names=built["feature_names"], noise_std=noise_std, noise_seed=noise_seed,
    )


if __name__ == "__main__":
    bundle = build_autoencoder_loaders(noise_std=0.1)
    x_in, x_tgt, _ = next(iter(bundle.loaders["train"]))
    print(f"denoising    noise_std={bundle.noise_std}  observed std(x_input - x_target)="
          f"{float((x_in - x_tgt).std()):.4f}")
    for name in STREAMS:
        loader = bundle.loaders[name]
        batch = next(iter(loader))
        x, rid = batch[0], batch[-1]
        print(f"{name:12s} rows={len(loader.dataset):>7,}  batches={len(loader):>4}  "
              f"fraud_rows={int(bundle.labels[name].sum()):>3}  first batch x={tuple(x.shape)} {x.dtype}, "
              f"row_id={tuple(rid.shape)} {rid.dtype}")
