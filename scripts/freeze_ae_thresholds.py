"""Freeze Autoencoder policies from committed validation exports.

Run from the repository root with python -m scripts.freeze_ae_thresholds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluation import load_anomaly_scores
from src.threshold_calibration import artifact_identity, calibrate_validation


def build_manifest() -> dict:
    """Verify Autoencoder provenance before producing validation-only policies."""
    ae_manifest_path = Path("results/ae_frozen_model_manifest.json")
    ae_manifest = json.loads(ae_manifest_path.read_text(encoding="utf-8"))

    selected = ae_manifest["selected_checkpoint"]
    checkpoint_path = selected["checkpoint_path"]
    checkpoint_sha256 = selected["checkpoint_sha256"]

    # Preprocessing identity from Phase 2 shared 29-feature pipeline (Issue #28)
    feature_meta_path = Path("data/artifacts/feature_metadata.json")
    feature_meta = json.loads(feature_meta_path.read_text(encoding="utf-8"))
    scaler_rel_path = feature_meta["scaler"]["artifact"]
    scaler_path = f"data/artifacts/{scaler_rel_path}"
    scaler_sha256 = feature_meta["scaler"]["artifact_sha256"]

    split_manifest_path = "data/split_manifest.json"
    split_manifest_sha256 = ae_manifest["data_provenance"]["dataset_manifest_hash"]

    # Selected winning AE run ID validation score export
    val_scores_path = f"results/phase2/scores/{selected['run_id']}_scores_val.csv"
    val_labels_path = "data/scores/validation_labels.csv"

    identities = {
        "model_artifact": {"path": checkpoint_path, "sha256": checkpoint_sha256},
        "preprocessing_artifact": {"path": scaler_path, "sha256": scaler_sha256},
        "split_manifest": {"path": split_manifest_path, "sha256": split_manifest_sha256},
        "validation_scores": {
            "path": val_scores_path,
            "sha256": "4a065fbb48fddd422c58874c71c19dcf0a1ce0abf6d7b1d0009eed317958e827",
        },
        "validation_labels": {
            "path": val_labels_path,
            "sha256": "12e3c74b01c2ccdf3a9e94489f8b7767b34fb1ea9395d4a9e160cfe4b81fd307",
        },
    }

    verified = {}
    for name, record in identities.items():
        path = record["path"]
        identity = artifact_identity(path)
        if identity["sha256"] != record["sha256"]:
            raise ValueError(
                f"Autoencoder provenance hash mismatch for {name}: "
                f"expected {record['sha256']}, got {identity['sha256']}"
            )
        verified[name] = identity

    score_path = verified["validation_scores"]["path"]
    label_path = verified["validation_labels"]["path"]

    scores = load_anomaly_scores(
        score_path,
        expected_split="validation",
        expected_model="autoencoder",
        score_direction="higher_is_more_suspicious",
    )

    split_path = Path("data/splits/val_indices.csv")
    ids = pd.read_csv(split_path, dtype=str)["raw_row_id"]
    if ids.duplicated().any() or set(ids) != set(scores.row_id):
        raise ValueError("Validation scores must match frozen validation split IDs.")

    score_transform = ae_manifest.get("score_contract", {}).get(
        "formula", "(1 / 29) * sum((x_j - x_hat_j)^2)"
    )
    if "score(x) = " in score_transform:
        score_transform = score_transform.replace("score(x) = ", "")

    manifest = calibrate_validation(
        score_path,
        label_path,
        model="autoencoder",
        score_transform=score_transform,
    )

    manifest["provenance"] = {
        **verified,
        "validation_split_ids": artifact_identity(split_path),
        "model_selection_manifest": artifact_identity(ae_manifest_path),
        "feature_metadata": artifact_identity(feature_meta_path),
        "calibration_code": artifact_identity("src/threshold_calibration.py"),
        "freeze_code": artifact_identity("scripts/freeze_ae_thresholds.py"),
        "feature_order": feature_meta["feature_names"],
        "preprocessing": (
            "V1–V28, log1p(Amount), Phase 2 RobustScaler fitted strictly on legitimate "
            "training rows (Issue #28)"
        ),
    }
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase2/autoencoder/threshold_manifest.json"),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the existing manifest without changing it.",
    )
    args = parser.parse_args()
    manifest = build_manifest()
    if args.check:
        if not args.output.exists():
            raise FileNotFoundError(f"Manifest does not exist at {args.output}")
        if json.loads(args.output.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Saved threshold manifest differs from recalibration.")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(f"Verified Autoencoder validation thresholds for {manifest['validation_rows']} rows.")


if __name__ == "__main__":
    main()
