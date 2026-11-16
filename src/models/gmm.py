"""Gaussian Mixture Model (GMM) probabilistic anomaly scoring implementation."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.mixture import GaussianMixture

from src.config import GMMConfig
from src.models.base import BaseAnomalyModel


class GMMAnomalyModel(BaseAnomalyModel):
    """Unsupervised density estimation via Gaussian Mixture Models.

    Models the probability density of legitimate transactions p(x) and scores
    anomalies using the negative log-likelihood: Score(x) = -ln p(x).
    """

    def __init__(self, config: GMMConfig) -> None:
        self.config = config
        self.model: GaussianMixture | None = None

    @property
    def name(self) -> str:
        return "gmm"

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> GMMAnomalyModel:
        """Fit GMM on legitimate transactions using EM algorithm."""
        self.model = GaussianMixture(
            n_components=self.config.n_components,
            covariance_type=self.config.covariance_type,
            reg_covar=self.config.reg_covar,
            max_iter=self.config.max_iter,
            n_init=self.config.n_init,
            random_state=self.config.random_state,
        )
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute negative log-likelihood anomaly scores (-ln p(x))."""
        if self.model is None:
            raise RuntimeError("Model must be fitted before computing anomaly scores.")
        # score_samples computes log-likelihood per sample; negative log-likelihood is anomaly score
        log_likelihood = self.model.score_samples(X)
        return -log_likelihood

    def save(self, output_dir: str | Path) -> dict[str, str]:
        """Save fitted GMM model and configuration."""
        if self.model is None:
            raise RuntimeError("Cannot save unfitted GMM model.")
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        model_file = out_path / "gmm_model.joblib"
        joblib.dump(self.model, model_file)

        meta_file = out_path / "gmm_meta.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": self.name,
                    "converged": bool(self.model.converged_),
                    "n_iter": int(self.model.n_iter_),
                    "lower_bound": float(self.model.lower_bound_),
                    "config": {
                        "n_components": self.config.n_components,
                        "covariance_type": self.config.covariance_type,
                        "reg_covar": self.config.reg_covar,
                        "max_iter": self.config.max_iter,
                        "n_init": self.config.n_init,
                        "random_state": self.config.random_state,
                    },
                },
                f,
                indent=2,
            )

        return {
            "model_path": str(model_file),
            "meta_path": str(meta_file),
        }

    def load(self, model_path_or_dir: str | Path) -> GMMAnomalyModel:
        """Load fitted GMM model from file or directory."""
        path = Path(model_path_or_dir)
        if path.is_file():
            model_file = path
        else:
            model_file = path / "gmm_model.joblib"
        if not model_file.is_file():
            raise FileNotFoundError(f"GMM model file not found at '{model_file}'.")
        self.model = joblib.load(model_file)
        return self
