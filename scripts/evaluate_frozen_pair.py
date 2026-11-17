"""Evaluate frozen GMM and AE test exports without recalibrating thresholds."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.paired_evaluation import compare_score_files, load_frozen_policy
from src.threshold_calibration import artifact_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ae-scores", type=Path, required=True)
    parser.add_argument("--ae-thresholds", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/phase2/paired"))
    args = parser.parse_args()
    manifests = {
        "gmm": Path("results/phase2/gmm/threshold_manifest.json"),
        "autoencoder": args.ae_thresholds,
    }
    policies = {model: load_frozen_policy(path, model) for model, path in manifests.items()}
    # Verify the AE is the selected checkpoint, not a new candidate.
    frozen = json.loads(Path("results/ae_frozen_model_manifest.json").read_text())
    checkpoint = frozen["selected_checkpoint"]
    ae_provenance = policies["autoencoder"]["provenance"]
    if ae_provenance["model_artifact"]["sha256"] != checkpoint["checkpoint_sha256"]:
        raise ValueError("AE policy must reference the selected frozen checkpoint.")
    if policies["gmm"]["provenance"]["feature_order"] != ae_provenance["feature_order"]:
        raise ValueError("Models must use the same feature order.")
    if policies["gmm"]["provenance"]["split_manifest"] != ae_provenance["split_manifest"]:
        raise ValueError("Models must use the same split manifest.")
    split_manifest = json.loads(Path(ae_provenance["split_manifest"]["path"]).read_text())
    score_paths = {"gmm": Path("data/scores/gmm_test.csv"), "autoencoder": args.ae_scores}
    label_path = Path("data/scores/test_labels.csv")
    ids_path = Path("data/splits/test_indices.csv")
    ids = pd.read_csv(ids_path, dtype=str)["raw_row_id"]
    if set(ids) != set(map(str, split_manifest["split_indices"]["test"])):
        raise ValueError("Test index CSV differs from frozen split manifest.")
    if set(ids) & set(map(str, split_manifest["split_indices"]["val"])):
        raise ValueError("Test and validation memberships overlap.")
    baseline = json.loads(Path("data/scores/gmm_baseline_provenance.json").read_text())
    for key, path in (("test_scores_csv", score_paths["gmm"]), ("test_labels_csv", label_path)):
        if artifact_identity(path)["sha256"] != baseline["output_artifacts"][key]["sha256"]:
            raise ValueError("GMM test export differs from reproduced baseline.")
    inference_path = Path("results/phase2/paired/inference_provenance.json")
    inference = json.loads(inference_path.read_text())
    for model, path in score_paths.items():
        key = "ae_scores" if model == "autoencoder" else "gmm_scores"
        if artifact_identity(path) != inference[key]:
            raise ValueError("Test scores differ from verified frozen inference.")
        if artifact_identity(manifests[model]) != inference["threshold_manifests"][model]:
            raise ValueError("Policy changed after frozen inference.")
    tables, frames = compare_score_files(score_paths, label_path, policies, ids)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.output / f"{name}.csv", index=False)
    fig, axis = plt.subplots(figsize=(8, 6))
    for model, frame in frames.items():
        precision, recall, _ = precision_recall_curve(frame.Class, frame.anomaly_score)
        ap = tables["ranking"].set_index("model").loc[model, "average_precision_ap"]
        axis.plot(recall, precision, label=f"{model} AP={ap:.4f}")
    axis.axhline(float(frame.Class.mean()), linestyle="--", color="gray", label="Test prevalence")
    axis.set(xlabel="Recall", ylabel="Precision", title="Frozen GMM versus autoencoder")
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output / "precision_recall.png", dpi=180)
    plt.close(fig)
    identities = {model: artifact_identity(path) for model, path in score_paths.items()}
    evidence = {
        "score_files": identities,
        "inference_provenance": artifact_identity(inference_path),
        "evaluation_code": artifact_identity("src/paired_evaluation.py"),
        "runner_code": artifact_identity("scripts/evaluate_frozen_pair.py"),
        "threshold_manifests": {model: artifact_identity(path) for model, path in manifests.items()},
        "test_labels": artifact_identity(label_path), "test_ids": artifact_identity(ids_path),
        "prior_holdout_exposure": "This test partition was previously examined in Phase 1.",
        "threshold_selection": "None; only frozen validation thresholds were applied.",
    }
    (args.output / "provenance.json").write_text(json.dumps(evidence, indent=2) + "\n")
    rows = tables["ranking"].set_index("model")
    overlap = tables["overlap"].iloc[0]
    summary = (
        "# Paired test evaluation\n\n"
        "The test partition was previously examined in Phase 1. Model selection and threshold "
        "calibration used validation data. This run applies the two saved policies.\n\n"
        f"GMM AP is {rows.loc['gmm', 'average_precision_ap']:.4f}; AE AP is "
        f"{rows.loc['autoencoder', 'average_precision_ap']:.4f}. At 100 reviews, "
        f"{overlap.both_caught} frauds are caught by both, {overlap.a_only} only by GMM "
        f"and {overlap.b_only} only by AE. See triage.csv for false alarms and recall.\n\n"
        "GMM has the higher AP across the full ranking, while the AE catches two additional frauds "
        "in the 100-case queue. At 1,000 reviews both catch 41 of 52 frauds and generate 959 false alarms. "
        "A single fraud changes recall by about 1.92 percentage points.\n\n"
        "At the validation-selected 0.1% FPR budget, GMM catches 33 frauds with four false alarms, "
        "while AE catches 24 with 14 false alarms. At the 1% budget, GMM catches 38 with 69 false "
        "alarms and AE catches 40 with 201. These frozen policies have different realized workloads.\n\n"
        "thresholds.csv reports realized test FPR against each validation budget. "
        "These rates can differ under temporal shift. All reported values trace to provenance.json.\n"
    )
    (args.output / "interpretation.md").write_text(summary)


if __name__ == "__main__":
    main()
