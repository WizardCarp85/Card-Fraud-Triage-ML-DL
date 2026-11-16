"""Validation-only calibration using the shared score contract and metric utilities."""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path

from evaluation import FPR_BUDGETS, load_anomaly_scores, load_evaluation_labels, select_threshold_at_fpr
from src.provenance import compute_file_sha256


def calibrate_validation(scores_path, labels_path, *, model, score_transform):
    """Return JSON-safe policies. This interface accepts validation files only."""
    scores = load_anomaly_scores(
        scores_path, expected_split="validation", expected_model=model,
        score_direction="higher_is_more_suspicious",
    )
    labels = load_evaluation_labels(labels_path, expected_split="validation")
    if set(scores.row_id) != set(labels.row_id):
        raise ValueError("Validation score and label IDs must match exactly.")
    aligned = scores.merge(labels, on=["row_id", "split"], validate="one_to_one")
    policies = []
    for budget in FPR_BUDGETS:
        selection = select_threshold_at_fpr(aligned.Class, aligned.anomaly_score, budget)
        row = asdict(selection)
        # JSON null carries undefined/non-finite values without nonstandard NaN/Infinity.
        row = {key: (None if isinstance(value, float) and not math.isfinite(value) else value)
               for key, value in row.items()}
        row["decision"] = {
            "ok": "score_gte_threshold",
            "no_alerts_within_budget": "no_alerts",
        }.get(selection.status, "undefined")
        row["reason"] = None if selection.status == "ok" else selection.status
        policies.append(row)
    return {
        "schema_version": "1.0",
        "model": model,
        "calibration_split": "validation",
        "score_direction": "higher_is_more_suspicious",
        "score_transform": score_transform,
        "selection_rule": "maximum validation recall within FPR budget; highest threshold breaks recall ties",
        "validation_rows": len(aligned),
        "validation_fraud_rows": int(aligned.Class.sum()),
        "thresholds": policies,
    }


def artifact_identity(path: str | Path) -> dict:
    """Require an existing artifact and record its byte-level identity."""
    digest = compute_file_sha256(path)
    if digest is None:
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": digest}
