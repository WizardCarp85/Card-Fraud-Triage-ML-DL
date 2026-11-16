"""Abstract base interface for Phase 2 anomaly detection models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseAnomalyModel(ABC):
    """Abstract interface for unsupervised anomaly scoring models.

    All implementations must return 1-dimensional numpy arrays of anomaly scores
    where higher values indicate greater suspicion of fraud.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the human-readable identifier of the model."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> BaseAnomalyModel:
        """Fit model parameters on feature matrix X (legitimate training samples)."""

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for feature matrix X.

        Returns:
            1D array of float scores where higher scores denote higher anomaly/fraud risk.
        """

    @abstractmethod
    def save(self, output_dir: str | Path) -> dict[str, str]:
        """Save model weights and artifacts to the target directory.

        Returns:
            dict mapping artifact names to their saved file paths.
        """

    @abstractmethod
    def load(self, model_dir: str | Path) -> BaseAnomalyModel:
        """Load trained model weights from the target directory."""
