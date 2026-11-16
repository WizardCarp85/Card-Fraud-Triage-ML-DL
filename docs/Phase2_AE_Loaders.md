# Phase 2 Autoencoder Data Loaders (Issues #31 and #33)

Implementation: `src/loaders.py`. Built on the #28 preprocessing pipeline (`data/preprocessing.py`) and the #25 split manifest.

## Streams
| Stream | Rows | Fraud rows | Order | Batch | Use |
|---|---|---|---|---|---|
| `train` | 198,242 (`train_legit`) | 0 | Shuffled, seeded | `(x_input, x_target, row_id)` | Autoencoder training |
| `val_legit` | 42,503 | 0 | Manifest order | `(x, row_id)` | Reconstruction-loss monitoring |
| `val_ranking` | 42,558 (all val) | 55 (labels not yielded) | Manifest order | `(x, row_id)` | Scores for ranking evaluation |
| `test` | 42,560 (all test) | 52 (labels not yielded) | Manifest order | `(x, row_id)` | Final scoring |

- `x`, `x_input` and `x_target` are `float32` with shape `(batch, 29)`.
- `row_id` is `int64`, the 0-based raw row index of `creditcard.csv`.
- `x_target` is always the clean scaled row. `x_input` is the same row plus optional noise (see below).
- Labels are never in a batch. They are stored on `bundle.labels[stream]`, aligned with `bundle.row_ids[stream]`, so evaluation can join scores to labels by row ID.

## Denoising augmentation (issue #33)
Set `noise_std > 0` to train a denoising autoencoder.

- **Where:** training inputs only. `x_target` stays clean, so the model learns to reconstruct the original vector. `val_legit`, `val_ranking` and `test` never get noise, whatever `noise_std` is.
- **What:** independent `N(0, noise_std²)` per element, added after the frozen #28 RobustScaler transform. `noise_std` is in scaled units: for RobustScaler, 1.0 is one inter-quartile range of the legitimate training rows, per feature.
- **When:** new noise is drawn for every batch, so each epoch sees different corruption.
- **Seeding:** the noise uses its own `torch.Generator` seeded with `noise_seed` (default: `seed`). Rebuilding the loaders with the same `seed` and `noise_seed` reproduces every epoch's inputs exactly. Changing `noise_seed` changes only the noise, never the batch order.
- **Zero noise:** with `noise_std=0` (the default), no random numbers are drawn and `x_input` is an exact copy of `x_target`. Batch order is the same as the clean path.
- **Validation:** `noise_std` must be a finite number ≥ 0. Otherwise `LoaderValidationError` is raised.
- `bundle.noise_std` and `bundle.noise_seed` record the settings used, for run provenance.

## Checks enforced at construction
- `X` has shape `(n, 29)`, is floating point, and is finite.
- Row IDs are integers, unique, and aligned with `X` and the labels.
- Row IDs for `train_legit`, `val` and `test` equal the manifest lists exactly, in order.
- `train_legit` contains no fraud rows.

Any violation raises `LoaderValidationError`.

## Reproducibility
The training shuffle uses `torch.Generator().manual_seed(seed)` with `num_workers=0`. The same seed gives the same batch order, and a different seed gives a different order. The other streams are not shuffled.

## Usage example
```python
from src.loaders import build_autoencoder_loaders

bundle = build_autoencoder_loaders(batch_size=256, seed=42, noise_std=0.1)  # 0.0 = plain autoencoder

for x_input, x_target, row_id in bundle.loaders["train"]:
    loss = criterion(model(x_input), x_target)  # target is always the clean row
    ...

# Reconstruction-loss monitoring on clean legitimate validation rows
for x, row_id in bundle.loaders["val_legit"]:
    val_loss = criterion(model(x), x)

# Ranking scores for evaluation, joined to labels by row ID afterwards
scores, ids = [], []
for x, row_id in bundle.loaders["val_ranking"]:
    scores.append(((model(x) - x) ** 2).mean(dim=1))
    ids.append(row_id)
```

Smoke run: `python -m src.loaders` prints the configured and observed noise std, and the rows, batches and fraud count for each stream.

Tests: `python -m pytest tests/test_loaders.py tests/test_denoising.py`. The synthetic tests always run. The real-data tests run only when `data/creditcard.csv` is present.

## Limitations
- `src/models/autoencoder.py` still builds its own internal `DataLoader` and does not use these streams yet, so `noise_std` has no effect on `run_phase2.py` until the training loop is switched over.
- `noise_std` is not in `src/config.py` yet. When the training loop uses these loaders, add it to `AutoencoderConfig` (default `0.0`) and pass it through.
- The same `noise_std` applies to all 29 features. Because RobustScaler normalises each feature by its inter-quartile range, that is a comparable relative corruption per feature.
- Tested with Python 3.9.6, torch 2.8.0, numpy 2.0.2.
