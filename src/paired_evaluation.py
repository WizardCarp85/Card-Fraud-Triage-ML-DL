"""Compare frozen score exports using preselected validation policies."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import (
    FPR_BUDGETS, REVIEW_CAPACITIES, average_precision, evaluate_fixed_threshold,
    load_anomaly_scores, load_evaluation_labels, roc_auc, triage_table,
)
from src.threshold_calibration import artifact_identity


def load_frozen_policy(path, model):
    """Validate policy identity before reading final test scores."""
    policy = json.loads(Path(path).read_text())
    if (policy["schema_version"] != "1.0" or policy["model"] != model
            or policy["calibration_split"] != "validation"
            or policy["score_direction"] != "higher_is_more_suspicious"):
        raise ValueError("Incompatible frozen threshold manifest.")
    provenance = policy["provenance"]
    for name in ("model_artifact", "preprocessing_artifact", "split_manifest",
                 "validation_scores", "validation_labels"):
        identity = provenance[name]
        if artifact_identity(identity["path"])["sha256"] != identity["sha256"]:
            raise ValueError(f"Frozen provenance mismatch for {name}.")
    if [r["fpr_budget"] for r in policy["thresholds"]] != list(FPR_BUDGETS):
        raise ValueError("Expected exactly the three frozen FPR budgets.")
    for row in policy["thresholds"]:
        if row["status"] == "ok":
            if (row["decision"] != "score_gte_threshold"
                    or row["threshold"] is None or not np.isfinite(row["threshold"])
                    or not 0 <= row["validation_fpr"] <= row["fpr_budget"]):
                raise ValueError("Invalid frozen threshold.")
        elif row["status"] == "no_alerts_within_budget":
            if row["decision"] != "no_alerts" or row["threshold"] is not None:
                raise ValueError("Invalid no-alert policy.")
        else:
            raise ValueError("Cannot perform final evaluation with undefined policies.")
    return policy


def compare_score_files(score_paths, labels_path, policies, expected_ids):
    """Return paired tables and plot data with no threshold selection or fitting."""
    labels = load_evaluation_labels(labels_path, expected_split="test")
    expected = list(map(str, expected_ids))
    if len(expected) != len(set(expected)) or set(labels.row_id) != set(expected):
        raise ValueError("Test labels must match the frozen test membership exactly.")
    frames, ranking, thresholds, queues = {}, [], [], []
    for model, path in score_paths.items():
        policy = policies[model]
        scores = load_anomaly_scores(
            path, expected_split="test", expected_model=model,
            score_direction=policy["score_direction"],
        )
        if set(scores.row_id) != set(expected):
            raise ValueError("Both models must use exactly the frozen test IDs.")
        frame = scores.merge(labels, on=["row_id", "split"], validate="one_to_one")
        frames[model] = frame
        ranking.append({
            "model": model, "n_transactions": len(frame), "n_fraud": int(frame.Class.sum()),
            "fraud_rate": float(frame.Class.mean()),
            "average_precision_ap": average_precision(frame.Class, frame.anomaly_score),
            "roc_auc": roc_auc(frame.Class, frame.anomaly_score),
        })
        for row in policy["thresholds"]:
            threshold = float("inf") if row["decision"] == "no_alerts" else row["threshold"]
            metrics = evaluate_fixed_threshold(frame.Class, threshold, frame.anomaly_score)
            thresholds.append({"model": model, **row, **metrics,
                               "missed_frauds": int(frame.Class.sum()) - metrics["frauds_caught"]})
        queues.append(triage_table(frame.Class, frame.anomaly_score).assign(model=model))
    if len(frames) != 2:
        raise ValueError("A paired evaluation requires exactly two models.")
    overlap = []
    for capacity in REVIEW_CAPACITIES:
        catches = []
        for frame in frames.values():
            top = frame.sort_values("anomaly_score", ascending=False, kind="mergesort").head(capacity)
            catches.append(set(top.loc[top.Class == 1, "row_id"]))
        first, second = catches
        overlap.append({
            "review_capacity": capacity, "model_a": list(frames)[0], "model_b": list(frames)[1],
            "both_caught": len(first & second), "a_only": len(first - second),
            "b_only": len(second - first), "union_caught": len(first | second),
            "a_only_ids": json.dumps(sorted(first - second)),
            "b_only_ids": json.dumps(sorted(second - first)),
        })
    return {
        "ranking": pd.DataFrame(ranking), "thresholds": pd.DataFrame(thresholds),
        "triage": pd.concat(queues, ignore_index=True), "overlap": pd.DataFrame(overlap),
    }, frames
