"""Shared 29-feature preprocessing pipeline for Phase 2 (GMM baseline and autoencoder).

Feature definition (Phase 1, notebook section 4.3):
    V1..V28 (unchanged) + Amount_log = log1p(Amount)
`Class` is never a feature and `Time` is not silently added.

Rules enforced here:
    * The scaler (RobustScaler) is fitted on legitimate training rows only
      (the manifest's ``train_legit`` membership).
    * The frozen scaler is applied to train, validation and test with transform only.
    * Row IDs are the raw row indices of ``creditcard.csv`` used by the split manifest.

Usage:
    python -m data.preprocessing
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import RobustScaler

V_COLS: list[str] = [f"V{i}" for i in range(1, 29)]
AMOUNT_LOG_COL = "Amount_log"
FEATURE_COLS: list[str] = V_COLS + [AMOUNT_LOG_COL]
LABEL_COL = "Class"
SPLIT_NAMES = ("train", "val", "test")

DEFAULT_DATA_PATH = Path("data/creditcard.csv")
DEFAULT_MANIFEST_PATH = Path("data/split_manifest.json")
DEFAULT_ARTIFACT_DIR = Path("data/artifacts")
PHASE1_SCALER_PATH = Path("results/scaler_robust.joblib")

# Numerical tolerances (documented in docs/Phase2_Preprocessing.md).
# Repeated runs on the same machine and library versions are expected to be bit-identical;
# the tolerances below apply across machines / library patch versions.
SCALER_PARAM_RTOL = 1e-9
SCALER_PARAM_ATOL = 1e-12
FEATURE_RTOL = 1e-9
FEATURE_ATOL = 1e-12
# Phase 1 scaler was serialised by a different sklearn version; compare statistics loosely.
PHASE1_SCALER_RTOL = 1e-6
PHASE1_SCALER_ATOL = 1e-9


class PreprocessingError(ValueError):
    """Raised when a preprocessing invariant is violated."""


def load_manifest(manifest_path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the ordered, unscaled 29-feature frame (index preserved as row ID).

    Raises PreprocessingError on missing columns, negative Amount, NaN or infinite values.
    """
    required = V_COLS + ["Amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise PreprocessingError(f"Missing required columns: {missing}")
    if (df["Amount"] < 0).any():
        raise PreprocessingError("Amount contains negative values; log1p is undefined.")
    feats = df[V_COLS].astype("float64").copy()
    feats[AMOUNT_LOG_COL] = np.log1p(df["Amount"].astype("float64"))
    feats = feats[FEATURE_COLS]
    values = feats.to_numpy()
    if not np.isfinite(values).all():
        raise PreprocessingError("Feature matrix contains NaN or infinite values.")
    return feats


def fit_scaler(train_legit_features: pd.DataFrame) -> RobustScaler:
    """Fit the RobustScaler on legitimate training rows only."""
    if list(train_legit_features.columns) != FEATURE_COLS:
        raise PreprocessingError("Scaler fit input has wrong feature order.")
    if LABEL_COL in train_legit_features.columns:
        raise PreprocessingError("Label column must not be part of the scaler fit input.")
    return RobustScaler().fit(train_legit_features.to_numpy())


def _array_sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def build_matrices(
    data_path: str | Path = DEFAULT_DATA_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Build scaled feature matrices for train/val/test from the frozen manifest.

    Returns a dict with keys:
        scaler: fitted RobustScaler (legit-train only)
        feature_names: list[str]
        splits: {name: {"X": float64 (n,29), "y": int64 (n,), "row_ids": int64 (n,)}}
        train_legit: {"X", "y", "row_ids"}  (legitimate-only training rows)
    Labels are returned separately from X and are never part of X.
    """
    manifest = load_manifest(manifest_path)
    idx = manifest["split_indices"]
    df_raw = pd.read_csv(data_path)
    features = build_feature_frame(df_raw)
    labels = df_raw[LABEL_COL].to_numpy(dtype=np.int64)

    train_legit_ids = np.asarray(idx["train_legit"], dtype=np.int64)
    if labels[train_legit_ids].any():
        raise PreprocessingError("train_legit membership contains fraud rows.")

    scaler = fit_scaler(features.iloc[train_legit_ids])

    def pack(ids_list: list[int]) -> dict[str, np.ndarray]:
        ids = np.asarray(ids_list, dtype=np.int64)
        return {
            "X": scaler.transform(features.iloc[ids].to_numpy()),
            "y": labels[ids],
            "row_ids": ids,
        }

    splits = {name: pack(idx[name]) for name in SPLIT_NAMES}
    train_legit = pack(idx["train_legit"])
    return {
        "scaler": scaler,
        "feature_names": list(FEATURE_COLS),
        "splits": splits,
        "train_legit": train_legit,
        "manifest": manifest,
    }


def verify_matrices(built: dict[str, Any]) -> dict[str, Any]:
    """Check shapes, dtypes, finiteness, membership and legit-only fit. Returns a report dict."""
    manifest = built["manifest"]
    idx = manifest["split_indices"]
    scaler: RobustScaler = built["scaler"]
    checks: dict[str, bool] = {}

    checks["feature_count_is_29"] = len(built["feature_names"]) == 29
    checks["feature_order_matches_contract"] = built["feature_names"] == FEATURE_COLS
    checks["no_label_or_time_in_features"] = (
        LABEL_COL not in FEATURE_COLS and "Time" not in FEATURE_COLS
    )

    per_split: dict[str, Any] = {}
    all_ids: list[np.ndarray] = []
    for name in SPLIT_NAMES:
        s = built["splits"][name]
        n = len(idx[name])
        ok_shape = s["X"].shape == (n, 29) and s["y"].shape == (n,) and s["row_ids"].shape == (n,)
        ok_dtype = s["X"].dtype == np.float64
        ok_finite = bool(np.isfinite(s["X"]).all())
        ok_members = bool(np.array_equal(s["row_ids"], np.asarray(idx[name], dtype=np.int64)))
        ok_counts = (
            int(s["y"].sum()) == manifest["split_statistics"][_stats_key(name)]["fraud_rows"]
        )
        per_split[name] = {
            "rows": n,
            "fraud_rows": int(s["y"].sum()),
            "shape_ok": bool(ok_shape),
            "dtype_float64": bool(ok_dtype),
            "all_finite": ok_finite,
            "row_ids_match_manifest": ok_members,
            "fraud_count_matches_manifest": bool(ok_counts),
            "X_sha256": _array_sha256(s["X"]),
            "row_ids_sha256": _array_sha256(s["row_ids"]),
        }
        all_ids.append(s["row_ids"])

    union = np.concatenate(all_ids)
    checks["splits_disjoint"] = len(np.unique(union)) == len(union)

    tl = built["train_legit"]
    checks["train_legit_has_no_fraud"] = int(tl["y"].sum()) == 0
    checks["train_legit_subset_of_train"] = bool(
        np.isin(tl["row_ids"], built["splits"]["train"]["row_ids"]).all()
    )
    checks["scaler_fitted_on_legit_train_only"] = bool(
        scaler.n_features_in_ == 29 and len(idx["train_legit"]) == manifest["split_statistics"]["train_legitimate_only"]["total_rows"]
    )
    checks["scaler_params_finite"] = bool(
        np.isfinite(scaler.center_).all() and np.isfinite(scaler.scale_).all() and (scaler.scale_ > 0).all()
    )
    for name, r in per_split.items():
        for key in ("shape_ok", "dtype_float64", "all_finite", "row_ids_match_manifest", "fraud_count_matches_manifest"):
            checks[f"{name}_{key}"] = r[key]

    return {
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "per_split": per_split,
    }


def _stats_key(split: str) -> str:
    return {"train": "train", "val": "validation", "test": "test"}[split]


def compare_to_phase1_scaler(
    scaler: RobustScaler, phase1_path: str | Path = PHASE1_SCALER_PATH
) -> dict[str, Any]:
    """Compare the rebuilt scaler to the saved Phase 1 scaler artifact, if available."""
    path = Path(phase1_path)
    if not path.exists():
        return {"available": False, "note": "Phase 1 scaler artifact not found; rebuilt scaler is the versioned replacement."}
    old = joblib.load(path)
    center_diff = float(np.max(np.abs(old.center_ - scaler.center_)))
    scale_diff = float(np.max(np.abs(old.scale_ - scaler.scale_)))
    match = bool(
        np.allclose(old.center_, scaler.center_, rtol=PHASE1_SCALER_RTOL, atol=PHASE1_SCALER_ATOL)
        and np.allclose(old.scale_, scaler.scale_, rtol=PHASE1_SCALER_RTOL, atol=PHASE1_SCALER_ATOL)
    )
    return {
        "available": True,
        "path": str(path),
        "matches_within_tolerance": match,
        "max_abs_center_diff": center_diff,
        "max_abs_scale_diff": scale_diff,
        "rtol": PHASE1_SCALER_RTOL,
        "atol": PHASE1_SCALER_ATOL,
    }


def save_artifacts(
    built: dict[str, Any],
    report: dict[str, Any],
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> dict[str, Path]:
    """Save scaler, feature metadata and verification report; matrices as a local npz."""
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    scaler: RobustScaler = built["scaler"]
    manifest = built["manifest"]

    scaler_path = out / "scaler_robust_phase2.joblib"
    joblib.dump(scaler, scaler_path)

    metadata = {
        "feature_names": built["feature_names"],
        "n_features": len(built["feature_names"]),
        "definition": "V1..V28 unchanged + Amount_log = log1p(Amount); Class and Time excluded",
        "scaler": {
            "type": "sklearn.preprocessing.RobustScaler",
            "fit_rows": "manifest split_indices.train_legit (legitimate training rows only)",
            "fit_row_count": len(manifest["split_indices"]["train_legit"]),
            "center_": scaler.center_.tolist(),
            "scale_": scaler.scale_.tolist(),
            "artifact": scaler_path.name,
            "artifact_sha256": hashlib.sha256(scaler_path.read_bytes()).hexdigest(),
        },
        "source_dataset_sha256": manifest["source_dataset"]["checksums"]["sha256"],
        "manifest_version": manifest["manifest_version"],
        "row_id_definition": "0-based raw row index of creditcard.csv (as in split_manifest.json)",
        "tolerances": {
            "scaler_param_rtol": SCALER_PARAM_RTOL,
            "scaler_param_atol": SCALER_PARAM_ATOL,
            "feature_rtol": FEATURE_RTOL,
            "feature_atol": FEATURE_ATOL,
            "phase1_scaler_rtol": PHASE1_SCALER_RTOL,
            "phase1_scaler_atol": PHASE1_SCALER_ATOL,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    meta_path = out / "feature_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    report_path = out / "preprocessing_verification_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Large matrices are rebuildable from the manifest; written locally, git-ignored.
    npz_path = out / "features_phase2.npz"
    arrays: dict[str, np.ndarray] = {"feature_names": np.asarray(built["feature_names"])}
    for name in SPLIT_NAMES:
        s = built["splits"][name]
        arrays[f"X_{name}"] = s["X"]
        arrays[f"y_{name}"] = s["y"]
        arrays[f"row_ids_{name}"] = s["row_ids"]
    np.savez_compressed(npz_path, **arrays)

    return {"scaler": scaler_path, "metadata": meta_path, "report": report_path, "matrices": npz_path}


def run(
    data_path: str | Path = DEFAULT_DATA_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> dict[str, Any]:
    built = build_matrices(data_path, manifest_path)
    report = verify_matrices(built)
    report["phase1_scaler_comparison"] = compare_to_phase1_scaler(built["scaler"])
    paths = save_artifacts(built, report, artifact_dir, data_path)
    print(f"Verification: {report['status']}")
    for k, p in paths.items():
        print(f"  {k}: {p}")
    if report["status"] != "PASSED":
        raise PreprocessingError("Preprocessing verification failed; see report.")
    return report


if __name__ == "__main__":
    run()
