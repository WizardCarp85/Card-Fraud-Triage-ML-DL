"""Freeze GMM policies from committed validation exports.

Run from the repository root with python -m scripts.freeze_gmm_thresholds.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.threshold_calibration import artifact_identity, calibrate_validation
from evaluation import load_anomaly_scores


def build_manifest():
    """Verify baseline provenance before producing validation-only policies."""
    provenance_path = Path("data/scores/gmm_baseline_provenance.json")
    provenance = json.loads(provenance_path.read_text())
    inputs = provenance["input_artifacts"]
    outputs = provenance["output_artifacts"]
    identities = {
        "model_artifact": inputs["phase1_gmm_model"],
        "preprocessing_artifact": inputs["phase1_scaler"],
        "split_manifest": {"path": inputs["manifest_path"], "sha256": inputs["manifest_sha256"]},
        "validation_scores": outputs["validation_scores_csv"],
        "validation_labels": outputs["validation_labels_csv"],
    }
    verified = {}
    for name, record in identities.items():
        path = record.get("path", record.get("artifact_path"))
        identity = artifact_identity(path)
        if identity["sha256"] != record["sha256"]:
            raise ValueError(f"Baseline provenance hash mismatch for {name}.")
        verified[name] = identity
    score_path = verified["validation_scores"]["path"]
    label_path = verified["validation_labels"]["path"]
    scores = load_anomaly_scores(
        score_path, expected_split="validation", expected_model="gmm",
        score_direction="higher_is_more_suspicious",
    )
    split_path = Path("data/splits/val_indices.csv")
    ids = pd.read_csv(split_path, dtype=str)["raw_row_id"]
    if ids.duplicated().any() or set(ids) != set(scores.row_id):
        raise ValueError("Validation scores must match frozen validation split IDs.")
    manifest = calibrate_validation(
        score_path, label_path, model="gmm", score_transform="-GaussianMixture.score_samples(X)",
    )
    manifest["provenance"] = {
        **verified,
        "validation_split_ids": artifact_identity(split_path),
        "baseline_provenance": artifact_identity(provenance_path),
        "calibration_code": artifact_identity("src/threshold_calibration.py"),
        "freeze_code": artifact_identity("scripts/freeze_gmm_thresholds.py"),
        "feature_order": inputs["phase1_scaler"]["feature_names_in"],
        "preprocessing": "V1–V28, log1p(Amount), Phase 1 RobustScaler fitted on legitimate training rows",
    }
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/phase2/gmm/threshold_manifest.json"))
    parser.add_argument("--check", action="store_true", help="Verify the existing manifest without changing it.")
    args = parser.parse_args()
    manifest = build_manifest()
    if args.check:
        if json.loads(args.output.read_text()) != manifest:
            raise ValueError("Saved threshold manifest differs from recalibration.")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Verified GMM validation thresholds for {manifest['validation_rows']} rows.")


if __name__ == "__main__":
    main()
