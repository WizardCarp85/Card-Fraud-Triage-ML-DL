"""Leakage-safe preprocessing and feature scaling for Phase 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.config import PreprocessingConfig


class PreprocessingPipeline:
    """Manages feature transformations and scaling with strict leakage-free guarantees."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        self.scaler: TransformerMixin | None = None
        self.feature_names: list[str] = [f"V{i}" for i in range(1, config.v_columns_count + 1)]
        if config.log_transform_amount:
            self.feature_names.append(config.amount_log_col)

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature transformations such as log1p on Amount."""
        out = df.copy()
        if self.config.log_transform_amount:
            if "Amount" not in out.columns:
                raise ValueError("DataFrame missing required 'Amount' column for log transform.")
            out[self.config.amount_log_col] = np.log1p(out["Amount"])
        return out

    def fit_transform_splits(
        self,
        splits: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        """Fit scaler exclusively on legitimate training samples and transform all splits.

        Args:
            splits: dict containing 'train', 'train_legit', 'val', 'test' DataFrames.

        Returns:
            dict containing X_train_legit, X_train, X_val, X_test, y_train, y_val, y_test,
            feature_names, and the fitted scaler.
        """
        # Prepare DataFrames with engineered features
        df_train = self._prepare_features(splits["train"])
        df_train_legit = self._prepare_features(splits["train_legit"])
        df_val = self._prepare_features(splits["val"])
        df_test = self._prepare_features(splits["test"])

        # Enforce unsupervised purity check
        if self.config.fit_on_legit_only:
            if (df_train_legit["Class"] != 0).any():
                raise ValueError(
                    "Violation of unsupervised purity: fraud rows detected in 'train_legit'!"
                )
            fit_df = df_train_legit
        else:
            fit_df = df_train

        # Initialize scaler
        if self.config.scaler_type.lower() == "robust":
            self.scaler = RobustScaler()
        elif self.config.scaler_type.lower() == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported scaler_type '{self.config.scaler_type}'. Choose 'robust' or 'standard'.")

        # Fit strictly on designated training corpus
        self.scaler.fit(fit_df[self.feature_names])

        # Transform partitions
        X_train_legit = self.scaler.transform(df_train_legit[self.feature_names])
        X_train = self.scaler.transform(df_train[self.feature_names])
        X_val = self.scaler.transform(df_val[self.feature_names])
        X_test = self.scaler.transform(df_test[self.feature_names])

        y_train = df_train["Class"].to_numpy(dtype=np.int8)
        y_val = df_val["Class"].to_numpy(dtype=np.int8)
        y_test = df_test["Class"].to_numpy(dtype=np.int8)

        return {
            "X_train_legit": X_train_legit,
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "val_indices": df_val.index.to_numpy(),
            "test_indices": df_test.index.to_numpy(),
            "feature_names": self.feature_names,
            "scaler": self.scaler,
        }

    def save_scaler(self, output_path: str | Path) -> None:
        """Serialize fitted scaler to disk."""
        if self.scaler is None:
            raise RuntimeError("Cannot save unfitted scaler.")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path)

    def load_scaler(self, input_path: str | Path) -> None:
        """Load previously fitted scaler from disk."""
        self.scaler = joblib.load(input_path)
