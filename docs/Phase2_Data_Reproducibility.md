# Phase 2 Data-Pipeline Tests and Reproducibility (Issue #40)

This covers the data layer only: the split (#25), preprocessing (#28), loaders (#31) and denoising (#33). Workflow-level integration is tracked separately.

## Environment
Pinned in `requirements-data.txt`:

| Package | Version |
|---|---|
| Python | 3.9.6 (macOS arm64) |
| numpy | 2.0.2 |
| pandas | 2.3.3 |
| scikit-learn | 1.6.1 |
| joblib | 1.5.3 |
| torch | 2.8.0 |
| pytest | 8.4.2 |

## Setup and rebuild from a clean clone
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-data.txt
# place creditcard.csv at data/creditcard.csv (SHA-256 76274b69…a89, checked against the manifest)

python -m data.verify_reproducibility                    # rebuild everything and compare with committed artifacts
python -m pytest tests/test_data_split.py tests/test_preprocessing.py tests/test_loaders.py tests/test_denoising.py
```
`data.verify_reproducibility` writes to a temporary directory and to `data/reproducibility/reproducibility_report.json`. It never overwrites `data/split_manifest.json`, `data/splits/` or `data/artifacts/`. It exits with 1 on any failure. Pass `--reference <report.json>` to also compare fingerprints with a report from another run or machine.

## Tests
| File | Invariants |
|---|---|
| `tests/test_data_split.py` (#25) | Source checksum, deduplication, split counts, disjoint and complete splits, legit-only training membership, chronological boundaries |
| `tests/test_preprocessing.py` (#28, new) | Feature order and definition; `Class` and `Time` excluded; `log1p(Amount)`; invalid input rejected; scaler fitted on `train_legit` only (differs from a fit that includes the frauds); scaler frozen for val/test; row IDs preserved and mapping back to source rows; exact manifest membership; labels not in X; matches committed artifacts and the Phase 1 scaler within tolerance; repeated runs bit-identical; **`src/preprocessing.py` (used by `run_phase2.py`) produces the same matrices** |
| `tests/test_loaders.py` (#31) | Legit-only training stream, manifest membership, no labels in batches, seeded batch order, dtypes and shapes |
| `tests/test_denoising.py` (#33) | Noise on training inputs only, clean targets and evaluation streams, zero-noise path, seeded noise, labels and row IDs unaffected |

The real-data tests are skipped when `data/creditcard.csv` is absent. The synthetic tests always run.

## Numerical tolerances
| Quantity | Tolerance |
|---|---|
| Same machine and library versions | Bit-identical (verified) |
| Feature matrices, scaler parameters, across machines or library patch versions | `rtol=1e-9`, `atol=1e-12` |
| Rebuilt scaler vs Phase 1 `results/scaler_robust.joblib` | `rtol=1e-6`, `atol=1e-9` (observed difference: 0.0) |
| Split membership and row IDs | Exact |

## Results (committed in `data/reproducibility/`)
| File | Content |
|---|---|
| `pytest_output.txt` | 50 data-layer tests, all passed (development venv) |
| `reproducibility_report.json` | `verify_reproducibility` in the development venv: PASSED |
| `clean_env_run.log` | Fresh clone and fresh venv from `requirements-data.txt` only: verification PASSED, reference comparison PASSED, 50 tests passed, documented rebuild steps |
| `clean_env_reproducibility_report.json` | Full report from the clean environment |

What the clean-environment run showed:
- Split membership, row-ID hashes and counts were identical to the committed manifest and the `data/splits/*.csv` files.
- `python -m data.preprocessing` reproduced `data/artifacts/` **byte-for-byte**: no git diff for the scaler, metadata or report.
- Loader batch order and seeded noise were identical to the development environment. The evaluation streams were unchanged by noise.
- Two consecutive runs gave bit-identical feature matrices.

## Known limitations
1. **`content_sha256` in `data/split_manifest.json` depends on the CSV float parser.** The committed values are reproduced only with `pd.read_csv(..., float_precision="round_trip")`. With the pandas 2.3.3 default parser, some values differ in the last bit (max absolute difference 9.9e-17, well inside tolerance), so the byte-level content hashes differ. Labels and row IDs are identical under both parsers. As a result, re-running `python data/freeze_split.py` on this pinned environment rewrites the 4 `content_sha256` lines (see `clean_env_run.log`). `verify_reproducibility` checks both parsers and passes when one reproduces every committed hash and the two parsers agree within tolerance. Suggested follow-up for #25: hash with a fixed parser, e.g. `float_precision="round_trip"`, or hash rounded values.
2. Verified on one machine (macOS arm64, Python 3.9.6). Other OS/BLAS builds may change feature bytes at the 1e-16 level. The reference comparison reports this, and the tolerance check is the acceptance criterion.
3. `tests/test_orchestration.py` and `tests/test_gmm_baseline.py` import `datetime.UTC`, which needs Python ≥ 3.11, so they do not run on the pinned Python 3.9. `tests/test_autoencoder_baseline.py` needs an `artifacts/phase2/run_*` directory that is not committed. These are outside the data layer and were not changed.
4. `requirements-data.txt` covers the data layer only. It is not a full project environment (no matplotlib, seaborn or Jupyter).
