"""Stage boundary contract validations and frozen evaluation for Phase 2 workflow."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from evaluation import (
    REVIEW_CAPACITIES,
    evaluate_fixed_threshold,
    load_anomaly_scores,
    load_evaluation_labels,
    triage_table,
)
from src.provenance import compute_file_sha256
from src.threshold_calibration import artifact_identity


class BoundaryValidationError(ValueError):
    """Raised when an artifact or stage violates boundary isolation or schema contracts."""


def validate_split_isolation(
    splits: dict[str, pd.DataFrame],
    manifest: dict[str, Any] | None = None,
) -> None:
    """Verify partition isolation, absence of fraud in train_legit, and disjoint splits."""
    required_splits = ("train", "train_legit", "val", "test")
    for name in required_splits:
        if name not in splits:
            raise BoundaryValidationError(f"Missing required split '{name}' in loaded splits.")
        if "Class" not in splits[name].columns:
            raise BoundaryValidationError(f"Split '{name}' must contain the 'Class' label column.")

    # 1. Zero fraud in train_legit
    train_legit_frauds = int(splits["train_legit"]["Class"].sum())
    if train_legit_frauds > 0:
        raise BoundaryValidationError(
            f"Split isolation violation: train_legit contains {train_legit_frauds} fraud rows. "
            "Training must be strictly legitimate-only."
        )

    # 2. Pairwise disjoint row memberships
    def _extract_ids(df: pd.DataFrame) -> set[Any]:
        if "raw_row_id" in df.columns:
            return set(df["raw_row_id"].astype(str))
        if "row_id" in df.columns:
            return set(df["row_id"].astype(str))
        return set(df.index.astype(str))

    train_ids = _extract_ids(splits["train"])
    val_ids = _extract_ids(splits["val"])
    test_ids = _extract_ids(splits["test"])

    if train_ids & val_ids:
        raise BoundaryValidationError(
            f"Split contamination: {len(train_ids & val_ids)} row IDs overlap between train and val."
        )
    if val_ids & test_ids:
        raise BoundaryValidationError(
            f"Split contamination: {len(val_ids & test_ids)} row IDs overlap between val and test."
        )
    if train_ids & test_ids:
        raise BoundaryValidationError(
            f"Split contamination: {len(train_ids & test_ids)} row IDs overlap between train and test."
        )

    # 3. Manifest alignment if provided
    if manifest and "split_indices" in manifest:
        idx = manifest["split_indices"]
        for split_key in ("train_legit", "val", "test"):
            if split_key in idx:
                expected_count = len(idx[split_key])
                actual_count = len(splits[split_key])
                if actual_count != expected_count:
                    raise BoundaryValidationError(
                        f"Manifest mismatch for '{split_key}': expected {expected_count} rows, "
                        f"got {actual_count} rows."
                    )


def validate_feature_contract(prep_data: dict[str, Any]) -> None:
    """Verify 29-feature contract, feature ordering, and absence of target/time columns."""
    expected_features = [f"V{i}" for i in range(1, 29)] + ["Amount_log"]
    feature_names = prep_data.get("feature_names", [])

    if list(feature_names) != expected_features:
        raise BoundaryValidationError(
            f"Feature contract violation: expected 29 features in order {expected_features[:3]}..."
            f"got {feature_names[:3]}... (length {len(feature_names)})"
        )

    for forbidden in ("Class", "Time", "Amount"):
        if forbidden in feature_names:
            raise BoundaryValidationError(
                f"Feature leakage: forbidden raw column '{forbidden}' found in feature space."
            )

    # Validate matrices
    for key in ("X_train_legit", "X_val", "X_test"):
        if key not in prep_data:
            raise BoundaryValidationError(f"Missing preprocessed feature matrix '{key}'.")
        mat = prep_data[key]
        if not isinstance(mat, np.ndarray):
            raise BoundaryValidationError(f"Feature matrix '{key}' must be a numpy ndarray.")
        if mat.ndim != 2 or mat.shape[1] != 29:
            raise BoundaryValidationError(
                f"Feature matrix '{key}' shape {mat.shape} violates (N, 29) contract."
            )
        if not np.issubdtype(mat.dtype, np.floating):
            raise BoundaryValidationError(f"Feature matrix '{key}' must have floating dtype, got {mat.dtype}.")
        if not np.isfinite(mat).all():
            raise BoundaryValidationError(f"Feature matrix '{key}' contains NaN or infinite values.")


def validate_loader_streams(bundle: Any, manifest: dict[str, Any] | None = None) -> None:
    """Verify PyTorch loader streams, batch structures, and strict label isolation."""
    streams = getattr(bundle, "loaders", {})
    required_streams = ("train", "val_legit", "val_ranking", "test")
    for s_name in required_streams:
        if s_name not in streams:
            raise BoundaryValidationError(f"Loader bundle missing stream '{s_name}'.")

    # Inspect train loader
    train_loader = streams["train"]
    sample_batch = next(iter(train_loader))
    if len(sample_batch) != 3:
        raise BoundaryValidationError(
            f"Train loader must yield (x_input, x_target, row_id), got {len(sample_batch)} items."
        )
    x_in, x_tgt, r_id = sample_batch
    if x_in.shape[1] != 29 or x_tgt.shape[1] != 29:
        raise BoundaryValidationError("Train loader batch feature dimension must be 29.")
    if x_in.dtype != torch.float32 or r_id.dtype != torch.int64:
        raise BoundaryValidationError("Train loader tensor dtypes violate (float32, float32, int64).")

    # Inspect eval loaders: must yield (x, row_id) without labels
    for s_name in ("val_legit", "val_ranking", "test"):
        eval_batch = next(iter(streams[s_name]))
        if len(eval_batch) != 2:
            raise BoundaryValidationError(
                f"Evaluation stream '{s_name}' must yield (x, row_id), got {len(eval_batch)} items."
            )
        ex, _er_id = eval_batch
        if ex.shape[1] != 29:
            raise BoundaryValidationError(f"Stream '{s_name}' feature dimension must be 29.")

    # Manifest row-ID alignment check
    if manifest and "split_indices" in manifest:
        idx = manifest["split_indices"]
        for stream_name, split_key in [("train", "train_legit"), ("val_ranking", "val"), ("test", "test")]:
            if split_key in idx and stream_name in getattr(bundle, "row_ids", {}):
                bundle_ids = bundle.row_ids[stream_name]
                expected_ids = idx[split_key]
                if set(bundle_ids) != set(expected_ids):
                    raise BoundaryValidationError(
                        f"Loader stream '{stream_name}' row IDs do not match manifest '{split_key}' IDs."
                    )


def validate_score_contract(
    score_df: pd.DataFrame,
    expected_model: str,
    expected_split: str,
    expected_row_ids: Sequence[Any],
) -> None:
    """Verify that score DataFrame strictly conforms to the shared RFC score contract."""
    required_cols = ["row_id", "split", "model", "anomaly_score"]
    if list(score_df.columns) != required_cols:
        raise BoundaryValidationError(
            f"Score DataFrame columns {list(score_df.columns)} do not match required {required_cols}."
        )

    if not score_df["model"].eq(expected_model).all():
        raise BoundaryValidationError(
            f"Model column mismatch: expected all '{expected_model}', got {score_df['model'].unique()}."
        )
    if not score_df["split"].eq(expected_split).all():
        raise BoundaryValidationError(
            f"Split column mismatch: expected all '{expected_split}', got {score_df['split'].unique()}."
        )

    scores = pd.to_numeric(score_df["anomaly_score"], errors="coerce")
    if scores.isna().any() or not np.isfinite(scores).all():
        raise BoundaryValidationError("Anomaly score column contains NaN or non-finite values.")

    # Check row-ID preservation and no duplicates
    row_ids = score_df["row_id"].astype(str)
    if row_ids.duplicated().any():
        raise BoundaryValidationError(f"Duplicate row IDs found in {expected_split} scores.")

    expected_set = set(map(str, expected_row_ids))
    actual_set = set(row_ids)
    if actual_set != expected_set:
        diff_missing = len(expected_set - actual_set)
        diff_extra = len(actual_set - expected_set)
        raise BoundaryValidationError(
            f"Row ID mismatch in {expected_split} scores: {diff_missing} missing, {diff_extra} extra."
        )


def execute_frozen_test_evaluation(
    output_dir: str | Path = "results/phase2/frozen_eval",
    gmm_scores_path: str | Path = "data/scores/gmm_test.csv",
    ae_scores_path: str | Path = "results/phase2/scores/RUN-S2-W16-DNZ-S7_scores_val.csv",
    test_labels_path: str | Path = "data/scores/test_labels.csv",
    test_ids_path: str | Path = "data/splits/test_indices.csv",
) -> dict[str, Any]:
    """Execute final test path using strictly frozen models and preselected validation thresholds.

    Enforces zero test contamination:
    - Never triggers model.fit() or any parameter training.
    - Never triggers select_threshold_at_fpr() or any threshold recalibration on test data.
    - Verifies byte hashes for models, scalers, and validation threshold manifests.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths = {
        "gmm": Path("results/phase2/gmm/threshold_manifest.json"),
        "autoencoder": Path("results/phase2/autoencoder/threshold_manifest.json"),
    }

    # Verify manifest integrity
    policies: dict[str, dict[str, Any]] = {}
    for model_name, m_path in manifest_paths.items():
        if not m_path.is_file():
            raise FileNotFoundError(f"Missing frozen threshold manifest for {model_name} at {m_path}.")
        policy = json.loads(m_path.read_text(encoding="utf-8"))
        if policy["schema_version"] != "1.0":
            raise BoundaryValidationError(f"Incompatible schema version in {m_path}.")
        if policy["model"] != model_name:
            raise BoundaryValidationError(f"Manifest model mismatch: expected {model_name}, got {policy['model']}.")
        if policy["calibration_split"] != "validation":
            raise BoundaryValidationError(f"Thresholds must be calibrated on validation split only in {m_path}.")
        if policy["score_direction"] != "higher_is_more_suspicious":
            raise BoundaryValidationError(f"Score direction must be higher_is_more_suspicious in {m_path}.")

        # Check provenance inputs
        prov = policy["provenance"]
        for key in ("model_artifact", "preprocessing_artifact", "split_manifest", "validation_scores"):
            item = prov[key]
            actual_sha = compute_file_sha256(item["path"])
            if actual_sha != item["sha256"]:
                raise BoundaryValidationError(
                    f"Hash mismatch for {key} in {model_name} threshold manifest: "
                    f"expected {item['sha256']}, got {actual_sha}"
                )
        policies[model_name] = policy

    # Load test labels
    labels_df = load_evaluation_labels(test_labels_path, expected_split="test")
    test_ids = pd.read_csv(test_ids_path, dtype=str)["raw_row_id"].tolist()
    if set(labels_df["row_id"].astype(str)) != set(map(str, test_ids)):
        raise BoundaryValidationError("Test labels do not match frozen test indices exactly.")

    # Apply frozen thresholds strictly to test scores
    # Verify GMM test scores
    gmm_test = load_anomaly_scores(
        gmm_scores_path,
        expected_split="test",
        expected_model="gmm",
        score_direction="higher_is_more_suspicious",
    )
    validate_score_contract(gmm_test, "gmm", "test", test_ids)

    # For autoencoder, load the frozen checkpoint directly to compute test scores if no separate test csv
    # or load from existing scores
    ae_checkpoint_path = Path("results/phase2/checkpoints/RUN-S2-W16-DNZ-S7.pt")
    from src.models.autoencoder import PyTorchAutoencoderNet

    net = PyTorchAutoencoderNet(input_dim=29, latent_dim=16, hidden_dims=(32,))
    state_dict = torch.load(ae_checkpoint_path, map_location=torch.device("cpu"), weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()

    # Preprocess test partition using frozen Phase 2 scaler
    from data.preprocessing import build_matrices

    matrices = build_matrices("data/creditcard.csv", "data/split_manifest.json")
    test_X = matrices["splits"]["test"]["X"]
    test_row_ids = matrices["splits"]["test"]["row_ids"]

    # Compute AE test scores with NO training
    with torch.no_grad():
        x_tensor = torch.from_numpy(test_X.astype(np.float32))
        ae_test_scores = []
        batch_size = 2048
        for i in range(0, len(x_tensor), batch_size):
            batch = x_tensor[i : i + batch_size]
            recon = net(batch)
            mse = torch.mean((batch - recon) ** 2, dim=1)
            ae_test_scores.append(mse.numpy())
        ae_scores_arr = np.concatenate(ae_test_scores).astype(np.float64)

    ae_test = pd.DataFrame({
        "row_id": test_row_ids.astype(str),
        "split": ["test"] * len(test_row_ids),
        "model": ["autoencoder"] * len(test_row_ids),
        "anomaly_score": ae_scores_arr,
    })
    validate_score_contract(ae_test, "autoencoder", "test", test_ids)

    # Evaluate frozen validation thresholds on test scores
    model_frames = {"gmm": gmm_test, "autoencoder": ae_test}
    threshold_results = []
    triage_queues = []

    for m_name, score_frame in model_frames.items():
        merged = score_frame.merge(labels_df, on=["row_id", "split"], validate="one_to_one")
        policy = policies[m_name]
        for t_policy in policy["thresholds"]:
            thresh_val = float("inf") if t_policy["decision"] == "no_alerts" else t_policy["threshold"]
            perf = evaluate_fixed_threshold(merged["Class"], thresh_val, merged["anomaly_score"])
            threshold_results.append({
                "model": m_name,
                "fpr_budget": t_policy["fpr_budget"],
                "frozen_threshold": thresh_val,
                "val_recall": t_policy["validation_recall"],
                "val_fpr": t_policy["validation_fpr"],
                **perf,
            })
        q_table = triage_table(merged["Class"], merged["anomaly_score"], REVIEW_CAPACITIES).assign(model=m_name)
        triage_queues.append(q_table)

    thresh_df = pd.DataFrame(threshold_results)
    triage_df = pd.concat(triage_queues, ignore_index=True)

    # Save artifact tables
    thresh_df.to_csv(out_dir / "frozen_test_threshold_metrics.csv", index=False)
    triage_df.to_csv(out_dir / "frozen_test_triage_table.csv", index=False)

    evidence = {
        "status": "COMPLETED",
        "evaluation_split": "test",
        "threshold_selection": "None (applied pre-calibrated frozen validation thresholds)",
        "training_performed": False,
        "models_evaluated": list(model_frames.keys()),
        "provenance": {
            "gmm_threshold_manifest": artifact_identity(manifest_paths["gmm"]),
            "ae_threshold_manifest": artifact_identity(manifest_paths["autoencoder"]),
            "ae_checkpoint": artifact_identity(ae_checkpoint_path),
            "test_labels": artifact_identity(test_labels_path),
            "test_indices": artifact_identity(test_ids_path),
        },
    }
    with open(out_dir / "frozen_evaluation_manifest.json", "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)

    return {
        "threshold_metrics": thresh_df,
        "triage_table": triage_df,
        "manifest": evidence,
    }
