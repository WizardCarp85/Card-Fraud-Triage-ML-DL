"""Phase 2 anomaly detection models registry and factory."""

from __future__ import annotations

from src.config import Phase2Config
from src.models.autoencoder import AutoencoderAnomalyModel
from src.models.base import BaseAnomalyModel
from src.models.gmm import GMMAnomalyModel

SUPPORTED_MODELS = ("gmm", "autoencoder")


def get_model(name: str, config: Phase2Config) -> BaseAnomalyModel:
    """Instantiate an anomaly model by name using configuration settings.

    Args:
        name: Name of model ('gmm' or 'autoencoder').
        config: Master Phase 2 configuration object.

    Returns:
        Instance conforming to BaseAnomalyModel.
    """
    normalized = name.strip().lower()
    if normalized == "gmm":
        return GMMAnomalyModel(config.gmm)
    elif normalized in ("autoencoder", "ae"):
        return AutoencoderAnomalyModel(config.autoencoder)
    else:
        raise ValueError(
            f"Unsupported model '{name}'. Supported models are: {list(SUPPORTED_MODELS)}"
        )


__all__ = [
    "SUPPORTED_MODELS",
    "AutoencoderAnomalyModel",
    "BaseAnomalyModel",
    "GMMAnomalyModel",
    "get_model",
]
