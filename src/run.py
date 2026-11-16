"""Orchestration runner for Phase 2 data, GMM, and Autoencoder pipelines."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.config import Phase2Config
from src.data import (
    DatasetNotFoundError,
    ManifestNotFoundError,
    load_split_partitions,
    validate_data_inputs,
)
from src.evaluation_runner import run_evaluation
from src.models import SUPPORTED_MODELS, BaseAnomalyModel, get_model
from src.preprocessing import PreprocessingPipeline
from src.provenance import compute_file_sha256, generate_provenance


def setup_logger(log_file: Path | None = None) -> logging.Logger:
    """Configure structured console and file logging."""
    logger = logging.getLogger("phase2_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def setup_run_directory(config: Phase2Config) -> Path:
    """Create a unique run artifact directory, failing if an existing run would be overwritten."""
    base_dir = Path(config.artifacts.base_output_dir)

    if config.artifacts.run_id:
        run_dir = base_dir / config.artifacts.run_id
    else:
        timestamp_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        unique_token = uuid.uuid4().hex[:6]
        run_dir = base_dir / f"run_{timestamp_str}_{unique_token}"

    if run_dir.exists() and not config.artifacts.overwrite:
        raise FileExistsError(
            f"Run directory '{run_dir}' already exists. "
            "Refusing to overwrite prior runs silently. "
            "Please specify a new run ID or pass '--overwrite'."
        )

    # Create directory tree
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "preprocessed").mkdir(exist_ok=True)
    (run_dir / "models").mkdir(exist_ok=True)
    (run_dir / "scores").mkdir(exist_ok=True)
    (run_dir / "tables").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    return run_dir


def execute_pipeline(
    config: Phase2Config,
    models_to_run: list[str] | None = None,
) -> Path:
    """Execute end-to-end Phase 2 orchestration pipeline.

    Steps:
    1. Validate input dataset and split manifest exist (fail fast if missing).
    2. Set up isolated run directory (fail if collision without overwrite flag).
    3. Load data partitions from frozen manifest.
    4. Run leakage-safe preprocessing & save scaler.
    5. For each requested model:
       a. Fit strictly on legitimate training samples.
       b. Compute anomaly scores on val and test splits.
       c. Run evaluation framework using evaluation.py and export triage tables.
    6. Record provenance, config snapshot, and run manifest.
    """
    # 1. Input validation
    data_path, manifest_path = validate_data_inputs(config.data)

    # 2. Setup run directory & logger
    run_dir = setup_run_directory(config)
    logger = setup_logger(run_dir / "logs" / "pipeline.log")

    logger.info("=" * 70)
    logger.info("PHASE 2 WORKFLOW ORCHESTRATION RUNNER")
    logger.info("=" * 70)
    logger.info(f"Target Run Directory : {run_dir.resolve()}")
    logger.info(f"Source Dataset       : {data_path.resolve()}")
    logger.info(f"Split Manifest       : {manifest_path.resolve()}")

    # Determine models
    if not models_to_run:
        models_to_run = ["gmm"]
    elif "all" in [m.lower() for m in models_to_run]:
        models_to_run = list(SUPPORTED_MODELS)

    # Enforce global seed
    np.random.seed(config.seed)

    # 3. Load data splits
    logger.info("[Step 1/4] Ingesting frozen dataset partitions from manifest...")
    splits = load_split_partitions(config.data)
    logger.info(f"           Train rows      : {len(splits['train']):,} (Fraud: {splits['train']['Class'].sum()})")
    logger.info(f"           Train-legit rows: {len(splits['train_legit']):,} (Fraud: {splits['train_legit']['Class'].sum()})")
    logger.info(f"           Val rows        : {len(splits['val']):,} (Fraud: {splits['val']['Class'].sum()})")
    logger.info(f"           Test rows       : {len(splits['test']):,} (Fraud: {splits['test']['Class'].sum()})")

    # 4. Preprocessing
    logger.info("[Step 2/4] Running leakage-safe feature transformations & scaling...")
    preprocessor = PreprocessingPipeline(config.preprocessing)
    prep_data = preprocessor.fit_transform_splits(splits)

    scaler_file = run_dir / "preprocessed" / f"scaler_{config.preprocessing.scaler_type}.joblib"
    preprocessor.save_scaler(scaler_file)
    logger.info(f"           Scaler fitted strictly on legitimate training samples ({len(prep_data['X_train_legit']):,} rows).")
    logger.info(f"           Fitted scaler saved to: {scaler_file}")

    # Track output files for manifest
    output_files: dict[str, str] = {
        "scaler": str(scaler_file),
    }

    # 5. Model training & scoring
    model_summaries: dict[str, Any] = {}
    for model_name in models_to_run:
        logger.info(f"[Step 3/4] Training and scoring model: '{model_name}'...")
        model: BaseAnomalyModel = get_model(model_name, config)

        # Fit model
        logger.info(f"           Fitting {model.name} on X_train_legit...")
        model.fit(prep_data["X_train_legit"])

        # Save model
        model_save_dir = run_dir / "models" / model.name
        saved_artifacts = model.save(model_save_dir)
        output_files.update(saved_artifacts)
        logger.info(f"           Model artifacts saved to: {model_save_dir}")

        # Compute scores
        logger.info(f"           Scoring validation and test splits with {model.name}...")
        val_scores = model.score(prep_data["X_val"])
        test_scores = model.score(prep_data["X_test"])

        # 6. Evaluation
        logger.info(f"[Step 4/4] Evaluating {model.name} via evaluation framework...")
        eval_run_dir = run_dir if len(models_to_run) == 1 else (run_dir / model.name)
        eval_out = run_evaluation(
            y_val=prep_data["y_val"],
            scores_val=val_scores,
            val_indices=prep_data["val_indices"],
            y_test=prep_data["y_test"],
            scores_test=test_scores,
            test_indices=prep_data["test_indices"],
            config=config.evaluation,
            run_dir=eval_run_dir,
        )

        output_files.update(eval_out["artifacts"])
        model_summaries[model.name] = eval_out["summary"]

        val_ap = eval_out["summary"]["validation"]["ap"]
        test_ap = eval_out["summary"]["test"]["ap"]
        logger.info(f"           Validation AP (PR-AUC): {val_ap:.4f}")
        logger.info(f"           Holdout Test AP (PR-AUC): {test_ap:.4f}")

    # 7. Provenance and manifest generation
    logger.info("Finalizing run provenance, configuration snapshot, and output manifest...")
    provenance = generate_provenance(config.to_dict(), data_path, manifest_path)
    provenance_file = run_dir / "provenance.json"
    with open(provenance_file, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    config_file = run_dir / "config.json"
    config.save(config_file)

    # Compute checksums for all created artifacts
    artifact_records = {}
    for label, path_str in output_files.items():
        artifact_records[label] = {
            "path": path_str,
            "sha256": compute_file_sha256(path_str),
            "size_bytes": Path(path_str).stat().st_size if Path(path_str).exists() else None,
        }

    run_manifest = {
        "run_id": run_dir.name,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "status": "COMPLETED",
        "models_evaluated": list(model_summaries.keys()),
        "metrics_summary": model_summaries,
        "artifacts": artifact_records,
    }
    run_manifest_file = run_dir / "run_manifest.json"
    with open(run_manifest_file, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    logger.info("=" * 70)
    logger.info(f"RUN SUCCESSFULLY COMPLETED: {run_dir}")
    logger.info(f"Manifest saved to         : {run_manifest_file}")
    logger.info(f"Provenance saved to       : {provenance_file}")
    logger.info("=" * 70)

    return run_dir


def build_cli_parser() -> argparse.ArgumentParser:
    """Build command-line interface arguments parser."""
    parser = argparse.ArgumentParser(
        description="Phase 2 Card Fraud Triage Orchestration Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["all", "preprocess", "train", "eval"],
        help="Pipeline command to execute.",
    )
    parser.add_argument(
        "--model",
        default="gmm",
        choices=["gmm", "autoencoder", "all"],
        help="Model architecture(s) to train and evaluate.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to custom JSON configuration file.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Override path to raw creditcard.csv dataset.",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Override path to split_manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Base directory to store Phase 2 run artifacts.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Explicit identifier/subdirectory for this run.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs for neural models.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed for reproducibility.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing run directory if it exists.",
    )
    return parser


def main() -> None:
    """CLI entry point for executing Phase 2 workflows."""
    parser = build_cli_parser()
    args = parser.parse_args()

    # Load base config
    if args.config:
        config = Phase2Config.from_json(args.config)
    else:
        config = Phase2Config()

    # Apply CLI overrides
    if args.data_path:
        config.data = config.data.__class__(**{**config.data.__dict__, "raw_data_path": args.data_path})
    if args.manifest_path:
        config.data = config.data.__class__(**{**config.data.__dict__, "manifest_path": args.manifest_path})
    if args.output_dir:
        config.artifacts = config.artifacts.__class__(**{**config.artifacts.__dict__, "base_output_dir": args.output_dir})
    if args.run_id:
        config.artifacts = config.artifacts.__class__(**{**config.artifacts.__dict__, "run_id": args.run_id})
    if args.overwrite:
        config.artifacts = config.artifacts.__class__(**{**config.artifacts.__dict__, "overwrite": True})
    if args.seed is not None:
        config.seed = args.seed
    if args.epochs is not None:
        config.autoencoder = config.autoencoder.__class__(**{**config.autoencoder.__dict__, "epochs": args.epochs})

    models_to_run = [args.model] if args.model != "all" else list(SUPPORTED_MODELS)

    try:
        execute_pipeline(config, models_to_run)
    except (
        DatasetNotFoundError,
        ManifestNotFoundError,
        FileExistsError,
        ValueError,
        RuntimeError,
        OSError,
    ) as exc:
        print(f"\n[PIPELINE ERROR] Execution failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
