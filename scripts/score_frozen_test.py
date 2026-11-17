"""Score the frozen test rows without fitting models or selecting thresholds."""
import argparse
import json
import platform
from pathlib import Path
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
import torch

from data.preprocessing import build_feature_frame
from evaluation import load_anomaly_scores
from src.models.autoencoder import AutoencoderAnomalyModel, PyTorchAutoencoderNet
from src.config import AutoencoderConfig
from src.paired_evaluation import load_frozen_policy
from src.threshold_calibration import artifact_identity


def score_test(data_path):
    policies = {
        model: load_frozen_policy(Path(f"results/phase2/{model}/threshold_manifest.json"), model)
        for model in ("gmm", "autoencoder")
    }
    frozen = json.loads(Path("results/ae_frozen_model_manifest.json").read_text())
    checkpoint = frozen["selected_checkpoint"]
    ae = policies["autoencoder"]["provenance"]
    if checkpoint["checkpoint_sha256"] != ae["model_artifact"]["sha256"]:
        raise ValueError("AE threshold policy must identify the selected checkpoint.")
    split = json.loads(Path(ae["split_manifest"]["path"]).read_text())
    if policies["gmm"]["provenance"]["split_manifest"] != ae["split_manifest"]:
        raise ValueError("Models must use the same frozen split.")
    source = artifact_identity(data_path)
    if source["sha256"] != split["source_dataset"]["checksums"]["sha256"]:
        raise ValueError("Raw dataset checksum mismatch.")
    ids = split["split_indices"]["test"]
    raw = pd.read_csv(data_path)
    # The model path receives features only; labels remain in evaluation exports.
    features = build_feature_frame(raw.loc[ids])
    if list(features.columns) != ae["feature_order"]:
        raise ValueError("Feature order mismatch.")
    matrices = {}
    for model, policy in policies.items():
        provenance = policy["provenance"]
        if provenance["feature_order"] != list(features.columns):
            raise ValueError("Model feature order mismatch.")
        scaler = joblib.load(provenance["preprocessing_artifact"]["path"])
        values = features if hasattr(scaler, "feature_names_in_") else features.to_numpy()
        matrices[model] = scaler.transform(values)
    np.testing.assert_allclose(matrices["gmm"], matrices["autoencoder"], rtol=1e-6, atol=1e-9)
    gmm = joblib.load(policies["gmm"]["provenance"]["model_artifact"]["path"])
    gmm_scores = -gmm.score_samples(matrices["gmm"])
    existing = load_anomaly_scores(
        "data/scores/gmm_test.csv", expected_split="test", expected_model="gmm",
        score_direction="higher_is_more_suspicious",
    ).set_index("row_id")
    reference = existing.loc[list(map(str, ids)), "anomaly_score"].to_numpy()
    # Verify the committed GMM export from #30 against fresh frozen inference.
    np.testing.assert_allclose(gmm_scores, reference, rtol=1e-9, atol=1e-9)
    model = AutoencoderAnomalyModel(AutoencoderConfig())
    model.net = PyTorchAutoencoderNet(
        input_dim=frozen["architecture"]["input_dimension"],
        latent_dim=frozen["architecture"]["bottleneck_dimension"],
        hidden_dims=(frozen["architecture"]["hidden_dimension"],),
    )
    model.net.load_state_dict(torch.load(ae["model_artifact"]["path"], map_location="cpu", weights_only=True))
    ae_scores = model.score(matrices["autoencoder"])
    path = Path("data/scores/autoencoder_test.csv")
    frame = pd.DataFrame({"row_id": ids, "split": "test", "model": "autoencoder", "anomaly_score": ae_scores})
    if path.exists():
        prior = load_anomaly_scores(
            path, expected_split="test", expected_model="autoencoder",
            score_direction="higher_is_more_suspicious",
        ).set_index("row_id")
        np.testing.assert_allclose(prior.loc[list(map(str, ids)), "anomaly_score"], ae_scores, rtol=1e-6, atol=1e-8)
    frame.to_csv(path, index=False)
    evidence = {
        "source_dataset": {"sha256": source["sha256"], "filename": Path(data_path).name},
        "models": {name: policy["provenance"] for name, policy in policies.items()},
        "threshold_manifests": {name: artifact_identity(f"results/phase2/{name}/threshold_manifest.json")
                                for name in policies},
        "scoring_code": artifact_identity("scripts/score_frozen_test.py"),
        "inference_code": artifact_identity("src/models/autoencoder.py"),
        "rows": len(ids),
        "gmm_regenerated_max_absolute_difference": float(np.max(np.abs(gmm_scores - reference))),
        "feature_matrix_max_absolute_difference": float(np.max(np.abs(matrices["gmm"] - matrices["autoencoder"]))),
        "ae_scores": artifact_identity(path),
        "gmm_scores": artifact_identity("data/scores/gmm_test.csv"),
        "inference_mode": "CPU, eval mode, no_grad, per-feature mean squared reconstruction error",
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("torch", "numpy", "pandas", "scikit-learn", "joblib")},
    }
    output = Path("results/phase2/paired/inference_provenance.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"Verified frozen inference on {len(ids)} test rows.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    score_test(args.data)


if __name__ == "__main__":
    main()
