# Phase 2 Autoencoder Data Loaders (Issue #31)

Implementation: `src/loaders.py`. Built on the #28 preprocessing pipeline (`data/preprocessing.py`) and the #25 split manifest.

## Streams
| Stream | Rows | Fraud rows | Order | Use |
|---|---|---|---|---|
| `train` | 198,242 (`train_legit`) | 0 | Shuffled, seeded | Autoencoder training |
| `val_legit` | 42,503 | 0 | Manifest order | Reconstruction-loss monitoring |
| `val_ranking` | 42,558 (all val) | 55 (labels not yielded) | Manifest order | Scores for ranking evaluation |
| `test` | 42,560 (all test) | 52 (labels not yielded) | Manifest order | Final scoring |

Each batch is `(x, row_id)`: `x` is `float32` with shape `(batch, 29)`, and `row_id` is `int64`, the 0-based raw row index of `creditcard.csv`. Labels are never in a batch. They are stored on `bundle.labels[stream]`, aligned with `bundle.row_ids[stream]`, so evaluation can join scores to labels by row ID.

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

bundle = build_autoencoder_loaders(batch_size=256, seed=42)

for x, row_id in bundle.loaders["train"]:
    recon = model(x)
    loss = criterion(recon, x)  # target is the input itself
    ...

# Ranking scores for evaluation, joined to labels by row ID afterwards
scores, ids = [], []
for x, row_id in bundle.loaders["val_ranking"]:
    scores.append(((model(x) - x) ** 2).mean(dim=1))
    ids.append(row_id)
```

Smoke run: `python -m src.loaders` prints the rows, batches and fraud count for each stream.

Tests: `python -m pytest tests/test_loaders.py`. The synthetic tests always run. The real-data test runs only when `data/creditcard.csv` is present.

## Limitations
- `src/models/autoencoder.py` still builds its own internal `DataLoader`. Switching the training loop to these loaders belongs to the training interface work in #32.
- Tested with Python 3.9.6, torch 2.8.0, numpy 2.0.2.
