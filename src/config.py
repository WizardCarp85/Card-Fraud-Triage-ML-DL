"""Unified configuration for Phase 2 data, model training, scoring, and evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    """Paths and specifications for raw dataset and frozen splits."""

    raw_data_path: str = "data/creditcard.csv"
    manifest_path: str = "data/split_manifest.json"
    splits_dir: str = "data/splits"
    target_column: str = "Class"
    time_column: str = "Time"
    amount_column: str = "Amount"


@dataclass(frozen=True)
class PreprocessingConfig:
    """Feature engineering, transformations, and scaling settings."""

    scaler_type: str = "robust"  # "robust" or "standard"
    log_transform_amount: bool = True
    amount_log_col: str = "Amount_log"
    v_columns_count: int = 28
    fit_on_legit_only: bool = True  # Strict unsupervised purity requirement


@dataclass(frozen=True)
class GMMConfig:
    """Hyperparameters for Gaussian Mixture Model anomaly detection."""

    n_components: int = 14
    covariance_type: str = "full"
    reg_covar: float = 1e-6
    max_iter: int = 200
    n_init: int = 3
    random_state: int = 42


@dataclass(frozen=True)
class AutoencoderConfig:
    """Hyperparameters for Deep PyTorch Autoencoder anomaly detection."""

    input_dim: int = 29
    latent_dim: int = 8
    hidden_dims: tuple[int, ...] = (20, 14)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    epochs: int = 15
    random_state: int = 42


@dataclass(frozen=True)
class EvaluationConfig:
    """Operational constraints and review capacity budgets."""

    fpr_budgets: tuple[float, ...] = (0.001, 0.01, 0.05)
    review_capacities: tuple[int, ...] = (100, 500, 1000)


@dataclass(frozen=True)
class ArtifactConfig:
    """Directory paths for outputs, isolating runs from Phase 1 results."""

    base_output_dir: str = "artifacts/phase2"
    run_id: str | None = None
    overwrite: bool = False


@dataclass
class Phase2Config:
    """Master configuration encompassing all aspects of a Phase 2 execution run."""

    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    gmm: GMMConfig = field(default_factory=GMMConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a nested dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, filepath: str | Path) -> None:
        """Save configuration to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Phase2Config:
        """Instantiate configuration from a dictionary."""
        return cls(
            data=DataConfig(**data.get("data", {})),
            preprocessing=PreprocessingConfig(**data.get("preprocessing", {})),
            gmm=GMMConfig(**data.get("gmm", {})),
            autoencoder=AutoencoderConfig(
                **{
                    **data.get("autoencoder", {}),
                    "hidden_dims": tuple(data.get("autoencoder", {}).get("hidden_dims", (20, 14))),
                }
            ),
            evaluation=EvaluationConfig(
                **{
                    **data.get("evaluation", {}),
                    "fpr_budgets": tuple(data.get("evaluation", {}).get("fpr_budgets", (0.001, 0.01, 0.05))),
                    "review_capacities": tuple(data.get("evaluation", {}).get("review_capacities", (100, 500, 1000))),
                }
            ),
            artifacts=ArtifactConfig(**data.get("artifacts", {})),
            seed=int(data.get("seed", 42)),
        )

    @classmethod
    def from_json(cls, filepath: str | Path) -> Phase2Config:
        """Load configuration from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
