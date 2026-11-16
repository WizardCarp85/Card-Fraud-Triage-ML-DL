"""Dataset loading, validation, and split retrieval for Phase 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DataConfig


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the raw dataset CSV file cannot be located."""


class ManifestNotFoundError(FileNotFoundError):
    """Raised when the Phase 2 frozen split manifest cannot be located."""


def validate_data_inputs(data_config: DataConfig) -> tuple[Path, Path]:
    """Validate that the source dataset and split manifest exist before running.

    Raises:
        DatasetNotFoundError: If creditcard.csv is missing with actionable instructions.
        ManifestNotFoundError: If split_manifest.json is missing with instructions.
    """
    data_path = Path(data_config.raw_data_path)
    if not data_path.is_file():
        raise DatasetNotFoundError(
            f"Required dataset not found at '{data_path}'.\n"
            "Action required: Please provide 'creditcard.csv' at the configured path or "
            "download it via kagglehub ('mlg-ulb/creditcardfraud')."
        )

    manifest_path = Path(data_config.manifest_path)
    if not manifest_path.is_file():
        raise ManifestNotFoundError(
            f"Frozen split manifest not found at '{manifest_path}'.\n"
            "Action required: Run 'python data/freeze_split.py' to generate the verified "
            "Phase 2 split manifest."
        )

    return data_path, manifest_path


def load_split_partitions(data_config: DataConfig) -> dict[str, pd.DataFrame]:
    """Load the frozen dataset partitions (train, train_legit, val, test) from disk.

    Returns:
        dict containing 'train', 'train_legit', 'val', and 'test' DataFrames.
    """
    data_path, manifest_path = validate_data_inputs(data_config)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest: dict[str, Any] = json.load(f)

    indices = manifest.get("split_indices", {})
    required_splits = ("train", "train_legit", "val", "test")
    for s in required_splits:
        if s not in indices:
            raise ValueError(f"Manifest at '{manifest_path}' is missing split index list for '{s}'.")

    df_raw = pd.read_csv(data_path)

    return {
        "train": df_raw.loc[indices["train"]].copy(),
        "train_legit": df_raw.loc[indices["train_legit"]].copy(),
        "val": df_raw.loc[indices["val"]].copy(),
        "test": df_raw.loc[indices["test"]].copy(),
    }
