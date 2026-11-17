"""Deep PyTorch Autoencoder anomaly scoring implementation for Phase 2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import AutoencoderConfig
from src.models.base import BaseAnomalyModel


class PyTorchAutoencoderNet(nn.Module):
    """Symmetric feedforward autoencoder architecture."""

    def __init__(
        self,
        input_dim: int = 29,
        latent_dim: int = 8,
        hidden_dims: tuple[int, ...] = (20, 14),
    ) -> None:
        super().__init__()
        # Encoder
        encoder_layers: list[nn.Module] = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(current_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.LeakyReLU(negative_slope=0.1))
            current_dim = h_dim
        encoder_layers.append(nn.Linear(current_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirrored)
        decoder_layers: list[nn.Module] = []
        current_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(current_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.LeakyReLU(negative_slope=0.1))
            current_dim = h_dim
        decoder_layers.append(nn.Linear(current_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent space and reconstruct input coordinates."""
        latent = self.encoder(x)
        return self.decoder(latent)


class AutoencoderAnomalyModel(BaseAnomalyModel):
    """Deep Autoencoder reconstruction anomaly model.

    Trains an autoencoder strictly on legitimate card behavior to learn the
    non-linear manifold. Anomaly score is the per-sample reconstruction error
    (Mean Squared Error): Score(x) = ||x - x_hat||_2^2 / D.
    """

    def __init__(self, config: AutoencoderConfig) -> None:
        self.config = config
        self.net: PyTorchAutoencoderNet | None = None
        self.training_history: list[float] = []

    @property
    def name(self) -> str:
        return "autoencoder"

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> AutoencoderAnomalyModel:
        """Train autoencoder on legitimate training samples."""
        # Enforce reproducibility
        torch.manual_seed(self.config.random_state)
        np.random.seed(self.config.random_state)

        input_dim = X.shape[1]
        self.net = PyTorchAutoencoderNet(
            input_dim=input_dim,
            latent_dim=self.config.latent_dim,
            hidden_dims=self.config.hidden_dims,
        )

        dataset = TensorDataset(torch.from_numpy(X.astype(np.float32)))
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.MSELoss()

        self.net.train()
        self.training_history = []
        for epoch in range(1, self.config.epochs + 1):
            epoch_loss = 0.0
            total_batches = 0
            for (batch_x,) in loader:
                optimizer.zero_grad()
                reconstructed = self.net(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                total_batches += 1

            avg_loss = epoch_loss / max(1, total_batches)
            self.training_history.append(avg_loss)

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample mean squared reconstruction error."""
        if self.net is None:
            raise RuntimeError("Autoencoder must be fitted before computing anomaly scores.")

        self.net.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(X.astype(np.float32))
            # Process in evaluation mini-batches for memory efficiency
            scores: list[np.ndarray] = []
            batch_size = 2048
            for start in range(0, len(tensor_x), batch_size):
                batch = tensor_x[start : start + batch_size]
                recon = self.net(batch)
                # Mean squared error across feature dimension per transaction
                mse = torch.mean((batch - recon) ** 2, dim=1)
                scores.append(mse.cpu().numpy())

        return np.concatenate(scores).astype(np.float64)

    def save(self, output_dir: str | Path) -> dict[str, str]:
        """Save PyTorch weights and training metadata."""
        if self.net is None:
            raise RuntimeError("Cannot save unfitted Autoencoder model.")
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        model_file = out_path / "autoencoder.pt"
        torch.save(self.net.state_dict(), model_file)

        meta_file = out_path / "autoencoder_meta.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": self.name,
                    "final_loss": self.training_history[-1] if self.training_history else None,
                    "epochs_trained": len(self.training_history),
                    "training_history": self.training_history,
                    "config": {
                        "input_dim": self.config.input_dim,
                        "latent_dim": self.config.latent_dim,
                        "hidden_dims": list(self.config.hidden_dims),
                        "learning_rate": self.config.learning_rate,
                        "weight_decay": self.config.weight_decay,
                        "batch_size": self.config.batch_size,
                        "epochs": self.config.epochs,
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

    def load(self, model_dir: str | Path) -> AutoencoderAnomalyModel:
        """Load trained weights into network architecture."""
        path = Path(model_dir)
        model_file = path / "autoencoder.pt"
        meta_file = path / "autoencoder_meta.json"

        if not model_file.is_file():
            raise FileNotFoundError(f"Autoencoder weight file not found at '{model_file}'.")

        input_dim = self.config.input_dim
        latent_dim = self.config.latent_dim
        hidden_dims = self.config.hidden_dims

        if meta_file.is_file():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            cfg = meta.get("config", {})
            input_dim = cfg.get("input_dim", input_dim)
            latent_dim = cfg.get("latent_dim", latent_dim)
            hidden_dims = tuple(cfg.get("hidden_dims", hidden_dims))

        self.net = PyTorchAutoencoderNet(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
        )
        self.net.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))
        self.net.eval()
        return self
