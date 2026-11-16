"""Stage 2 Autoencoder Seed Stability and Checkpoint Freezing Runner.

Follows the pre-registered plan in docs/ae_experiment_plan.md (Issue #27, #36).
Reuses the seed-42 runs from Stage 1 (RUN-S1-W16-DNZ-S42 and RUN-S1-W08-CLN-S42).
Runs the 4 missing seed runs (seeds 0, 7 for both configurations).
Measures stability across seeds {0, 7, 42}, selects final configuration by mean AP,
and freezes the final production checkpoint.

All training uses legitimate training rows only. Model selection uses validation only.
No test data or labels are evaluated.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch

from evaluation import average_precision, precision_at_k, recall_at_fpr
from scripts.run_stage1_ablations import (
    compute_file_sha256,
    get_git_commit,
    train_and_evaluate_run,
)
from src.loaders import build_autoencoder_loaders
from src.models.autoencoder import PyTorchAutoencoderNet


def run_stage2(
    data_path: Path,
    manifest_path: Path,
    run_log_path: Path,
    checkpoints_dir: Path,
    scores_dir: Path,
    device: torch.device,
    max_epochs: int = 100,
    patience: int = 10,
) -> dict[str, Any]:
    """Execute Stage 2 seed stability evaluation and return structured results."""
    manifest_hash = compute_file_sha256(manifest_path)
    git_commit = get_git_commit()

    if not run_log_path.is_file():
        sys.exit(f"Run log not found at {run_log_path}. Stage 1 must be run first.")

    df_log = pd.read_csv(run_log_path)
    stage1_runs = {row["run_id"]: row.to_dict() for _, row in df_log.iterrows()}

    # Check that required Stage-1 seed-42 runs exist
    ref_s42_id = "RUN-S1-W08-CLN-S42"
    wnr_s42_id = "RUN-S1-W16-DNZ-S42"

    if ref_s42_id not in stage1_runs:
        sys.exit(f"Required Stage-1 reference run {ref_s42_id} not found in {run_log_path}")
    if wnr_s42_id not in stage1_runs:
        sys.exit(f"Required Stage-1 champion run {wnr_s42_id} not found in {run_log_path}")

    print("=" * 75)
    print("PHASE 2 AUTOENCODER STAGE 2 SEED STABILITY SUITE (Issue #36)")
    print(f"Data Path:      {data_path.resolve()}")
    print(f"Manifest:       {manifest_path.resolve()} (SHA-256: {manifest_hash})")
    print(f"Git Commit:     {git_commit}")
    print(f"Compute Device: {device}")
    print("Reusing Stage 1 seed-42 runs:")
    print(f"  - {ref_s42_id}: val_AP = {stage1_runs[ref_s42_id]['val_AP']}")
    print(f"  - {wnr_s42_id}: val_AP = {stage1_runs[wnr_s42_id]['val_AP']}")
    print("=" * 75)

    # Missing runs for Stage 2
    runs_to_execute = [
        # Champion: B=16, Denoise sigma=0.5
        ("RUN-S2-W16-DNZ-S0", 16, True, 0.5, 0),
        ("RUN-S2-W16-DNZ-S7", 16, True, 0.5, 7),
        # Reference baseline: B=8, Clean
        ("RUN-S2-W08-CLN-S0", 8, False, 0.0, 0),
        ("RUN-S2-W08-CLN-S7", 8, False, 0.0, 7),
    ]

    new_results: list[dict[str, Any]] = []
    for run_id, bottleneck, denoise, noise_std, seed in runs_to_execute:
        res = train_and_evaluate_run(
            run_id=run_id,
            bottleneck=bottleneck,
            denoise=denoise,
            noise_std=noise_std,
            seed=seed,
            data_path=data_path,
            manifest_path=manifest_path,
            manifest_hash=manifest_hash,
            git_commit=git_commit,
            checkpoints_dir=checkpoints_dir,
            scores_dir=scores_dir,
            device=device,
            max_epochs=max_epochs,
            patience=patience,
        )
        new_results.append(res)

    # Append new runs to results/phase2/ae_run_log.csv without modifying Stage-1 rows
    df_new = pd.DataFrame(new_results)[list(df_log.columns)]
    df_updated = pd.concat([df_log, df_new], ignore_index=True)
    df_updated.to_csv(run_log_path, index=False)
    print(f"\nAppended {len(new_results)} Stage-2 runs to {run_log_path}. Total rows: {len(df_updated)}")

    # Stability analysis across 6 runs
    # Compile 6 runs into dictionary mapping config -> list of 3 runs
    all_runs_dict = {row["run_id"]: row.to_dict() for _, row in df_updated.iterrows()}

    w16_dnz_runs = [
        all_runs_dict["RUN-S2-W16-DNZ-S0"],
        all_runs_dict["RUN-S2-W16-DNZ-S7"],
        all_runs_dict[wnr_s42_id],
    ]

    w08_cln_runs = [
        all_runs_dict["RUN-S2-W08-CLN-S0"],
        all_runs_dict["RUN-S2-W08-CLN-S7"],
        all_runs_dict[ref_s42_id],
    ]

    def compute_stats(runs: list[dict[str, Any]], name: str) -> dict[str, Any]:
        aps = [float(r["val_AP"]) for r in runs]
        p100s = [float(r["val_Precision@100"]) for r in runs]
        rec1s = [float(r["val_recall_FPR1pct"]) for r in runs]
        rec5s = [float(r["val_recall_FPR5pct"]) for r in runs]
        runtimes = [float(r["runtime_s"]) for r in runs]
        epochs = [int(r["epochs_run"]) for r in runs]

        mean_ap = float(np.mean(aps))
        std_ap = float(np.std(aps, ddof=1))  # sample standard deviation
        mean_p100 = float(np.mean(p100s))
        std_p100 = float(np.std(p100s, ddof=1))

        # Check instability flag: |val_AP(seed) - mean(val_AP)| > 0.10
        unstable_seeds = []
        for r in runs:
            diff = abs(float(r["val_AP"]) - mean_ap)
            if diff > 0.10:
                unstable_seeds.append((r["run_id"], r["seed"], float(r["val_AP"]), diff))

        return {
            "name": name,
            "runs": runs,
            "aps": aps,
            "mean_ap": mean_ap,
            "std_ap": std_ap,
            "mean_p100": mean_p100,
            "std_p100": std_p100,
            "mean_rec1": float(np.mean(rec1s)),
            "mean_rec5": float(np.mean(rec5s)),
            "mean_runtime": float(np.mean(runtimes)),
            "mean_epochs": float(np.mean(epochs)),
            "unstable_seeds": unstable_seeds,
        }

    stats_w16 = compute_stats(w16_dnz_runs, "W16-DNZ (B=16, Denoise sigma=0.5)")
    stats_w08 = compute_stats(w08_cln_runs, "W08-CLN (B=8, Clean Reference)")

    print("\n" + "=" * 75)
    print("STAGE 2 STABILITY ANALYSIS (6 RUNS TOTAL)")
    print("=" * 75)
    print(f"\nConfiguration 1: {stats_w16['name']}")
    for r in stats_w16["runs"]:
        reused_note = " (Reused from Stage 1)" if r["run_id"] == wnr_s42_id else ""
        print(f"  {r['run_id']:18s} | Seed {r['seed']:2d} | val_AP: {float(r['val_AP']):.4f} | P@100: {float(r['val_Precision@100']):.4f} | Epochs: {r['epochs_run']:3d}{reused_note}")
    print(f"  --> MEAN val_AP: {stats_w16['mean_ap']:.4f} +/- {stats_w16['std_ap']:.4f}")
    print(f"  --> MEAN P@100:  {stats_w16['mean_p100']:.4f} +/- {stats_w16['std_p100']:.4f}")
    if stats_w16["unstable_seeds"]:
        for uid, s, ap, d in stats_w16["unstable_seeds"]:
            print(f"  [FLAG] Unstable seed detected: {uid} (seed {s}) |AP - mean| = {d:.4f} > 0.10")
    else:
        print("  [OK] No unstable seeds (|val_AP - mean| <= 0.10 for all seeds)")

    print(f"\nConfiguration 2: {stats_w08['name']}")
    for r in stats_w08["runs"]:
        reused_note = " (Reused from Stage 1)" if r["run_id"] == ref_s42_id else ""
        print(f"  {r['run_id']:18s} | Seed {r['seed']:2d} | val_AP: {float(r['val_AP']):.4f} | P@100: {float(r['val_Precision@100']):.4f} | Epochs: {r['epochs_run']:3d}{reused_note}")
    print(f"  --> MEAN val_AP: {stats_w08['mean_ap']:.4f} +/- {stats_w08['std_ap']:.4f}")
    print(f"  --> MEAN P@100:  {stats_w08['mean_p100']:.4f} +/- {stats_w08['std_p100']:.4f}")
    if stats_w08["unstable_seeds"]:
        for uid, s, ap, d in stats_w08["unstable_seeds"]:
            print(f"  [FLAG] Unstable seed detected: {uid} (seed {s}) |AP - mean| = {d:.4f} > 0.10")
    else:
        print("  [OK] No unstable seeds (|val_AP - mean| <= 0.10 for all seeds)")

    # 5. Selection per plan Section 5.4: choose final CONFIGURATION by MEAN validation AP
    print("\n" + "=" * 75)
    print("FINAL MODEL SELECTION (SECTION 5.4)")
    print("=" * 75)
    diff_mean_ap = stats_w16["mean_ap"] - stats_w08["mean_ap"]
    print(f"Mean val_AP comparison:")
    print(f"  W16-DNZ: {stats_w16['mean_ap']:.4f}")
    print(f"  W08-CLN: {stats_w08['mean_ap']:.4f}")
    print(f"  Difference: {diff_mean_ap:+.4f}")

    if abs(diff_mean_ap) >= 0.005:
        if diff_mean_ap > 0:
            winning_stats = stats_w16
            selection_rationale = (
                f"W16-DNZ wins on 3-seed MEAN validation AP ({stats_w16['mean_ap']:.4f} "
                f"vs {stats_w08['mean_ap']:.4f}, diff = {diff_mean_ap:+.4f} >= 0.005)"
            )
        else:
            winning_stats = stats_w08
            selection_rationale = (
                f"W08-CLN wins on 3-seed MEAN validation AP ({stats_w08['mean_ap']:.4f} "
                f"vs {stats_w16['mean_ap']:.4f}, diff = {-diff_mean_ap:+.4f} >= 0.005)"
            )
    else:
        # Within 0.005 band -> tie break: smaller model (B=8 over B=16; clean over denoised)
        winning_stats = stats_w08
        selection_rationale = (
            f"Within 0.005 mean AP band (|diff| = {abs(diff_mean_ap):.4f} < 0.005). "
            f"Tie-break selects W08-CLN by parsimony and simplicity (smaller bottleneck B=8, clean inputs)."
        )

    print(f"Decision: {selection_rationale}")

    # Checkpoint choice for winning configuration:
    # freeze the seed whose single-run val AP is CLOSEST to configuration's 3-seed mean AP.
    # Tie on closeness -> prefer seed 42.
    best_seed_run = None
    min_dist = float("inf")
    for r in winning_stats["runs"]:
        dist = abs(float(r["val_AP"]) - winning_stats["mean_ap"])
        print(f"  Candidate checkpoint {r['run_id']} (seed {r['seed']}): val_AP = {float(r['val_AP']):.4f}, dist to mean = {dist:.6f}")
        if dist < min_dist - 1e-7:
            min_dist = dist
            best_seed_run = r
        elif abs(dist - min_dist) <= 1e-7 and int(r["seed"]) == 42:
            best_seed_run = r

    assert best_seed_run is not None
    checkpoint_rationale = (
        f"Selected {best_seed_run['run_id']} (seed {best_seed_run['seed']}) because its validation AP "
        f"({float(best_seed_run['val_AP']):.4f}) is closest to the 3-seed mean AP "
        f"({winning_stats['mean_ap']:.4f}) with distance {min_dist:.6f}."
    )
    print(f"\nFrozen Checkpoint Choice: {best_seed_run['run_id']}")
    print(f"Rationale: {checkpoint_rationale}")
    print("=" * 75)

    return {
        "stats_w16": stats_w16,
        "stats_w08": stats_w08,
        "winning_stats": winning_stats,
        "winning_run": best_seed_run,
        "selection_rationale": selection_rationale,
        "checkpoint_rationale": checkpoint_rationale,
        "all_runs": [all_runs_dict[r] for r in [
            "RUN-S2-W16-DNZ-S0", "RUN-S2-W16-DNZ-S7", wnr_s42_id,
            "RUN-S2-W08-CLN-S0", "RUN-S2-W08-CLN-S7", ref_s42_id
        ]],
        "git_commit": git_commit,
        "dataset_manifest_hash": manifest_hash,
    }


def write_frozen_manifest(stage2_data: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    """Serialize the authoritative frozen autoencoder manifest JSON."""
    winning_run = stage2_data["winning_run"]
    winning_stats = stage2_data["winning_stats"]
    cfg = json.loads(winning_run["config_json"])

    # Checkpoint hashes for all 3 seeds of winning configuration
    seed_checkpoint_hashes = {
        str(r["seed"]): {
            "run_id": r["run_id"],
            "val_AP": float(r["val_AP"]),
            "checkpoint_path": str(r["checkpoint_path"]),
            "checkpoint_sha256": str(r["checkpoint_sha256"]),
        }
        for r in winning_stats["runs"]
    }

    manifest_dict = {
        "manifest_version": "1.0.0",
        "model_family": "Autoencoder",
        "model_name": "Phase2_Frozen_Undercomplete_Autoencoder",
        "status": "FROZEN",
        "author": "Abhigya Sachdeva (@Abhi)",
        "governing_issue": "Issue #36 (Stage 2 Confirmation & Checkpoint Freeze)",
        "selection_rule_applied": "docs/ae_experiment_plan.md Section 5.4 (Mean Validation AP across seeds {0, 7, 42})",
        "architecture": {
            "topology": f"29 -> 32 -> {cfg['bottleneck']} -> 32 -> 29",
            "input_dimension": 29,
            "bottleneck_dimension": cfg["bottleneck"],
            "hidden_dimension": 32,
            "realized_backbone": "BatchNorm1d + LeakyReLU(negative_slope=0.1) per Addendum A1",
            "output_layer": "Linear (identity mapping, continuous reconstruction)",
        },
        "hyperparameters": {
            "bottleneck": cfg["bottleneck"],
            "denoise": cfg["denoise"],
            "noise_std": cfg["noise_std"],
            "learning_rate": cfg["learning_rate"],
            "batch_size": cfg["batch_size"],
            "max_epochs": cfg["max_epochs"],
            "patience": cfg["patience"],
            "weight_decay": 0.0,
            "optimizer": "Adam(lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)",
            "loss_function": "Mean Squared Error (MSE) over 29 features",
        },
        "selected_checkpoint": {
            "run_id": winning_run["run_id"],
            "seed": int(winning_run["seed"]),
            "checkpoint_path": str(winning_run["checkpoint_path"]),
            "checkpoint_sha256": str(winning_run["checkpoint_sha256"]),
            "val_AP": float(winning_run["val_AP"]),
            "val_Precision@100": float(winning_run["val_Precision@100"]),
            "val_recall_FPR1pct": float(winning_run["val_recall_FPR1pct"]),
            "val_recall_FPR5pct": float(winning_run["val_recall_FPR5pct"]),
            "val_recon_loss": float(winning_run["val_recon_loss"]),
            "selection_rationale": stage2_data["checkpoint_rationale"],
        },
        "multi_seed_stability_metrics": {
            "seeds_evaluated": [0, 7, 42],
            "mean_val_AP": round(winning_stats["mean_ap"], 6),
            "std_val_AP": round(winning_stats["std_ap"], 6),
            "mean_val_Precision@100": round(winning_stats["mean_p100"], 6),
            "std_val_Precision@100": round(winning_stats["std_p100"], 6),
            "mean_val_recall_FPR1pct": round(winning_stats["mean_rec1"], 6),
            "mean_val_recall_FPR5pct": round(winning_stats["mean_rec5"], 6),
            "stability_flag": "STABLE" if not winning_stats["unstable_seeds"] else "UNSTABLE",
            "seed_checkpoint_hashes": seed_checkpoint_hashes,
        },
        "data_provenance": {
            "dataset_manifest_hash": stage2_data["dataset_manifest_hash"],
            "git_commit": stage2_data["git_commit"],
            "preprocessing": {
                "scaler": "RobustScaler (Issue #28, fitted strictly on legitimate training partition)",
                "feature_count": 29,
                "feature_names": [f"V{i}" for i in range(1, 29)] + ["Amount_log"],
                "target_column": "Class (strictly excluded from training and model inputs)",
            },
        },
        "score_contract": {
            "score_definition": "Per-transaction Mean Squared Error (MSE) reconstruction loss across 29 features",
            "score_direction": "higher_is_more_suspicious",
            "formula": "score(x) = (1 / 29) * sum((x_j - x_hat_j)^2)",
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2)

    print(f"\nSaved authoritative frozen model manifest to: {manifest_path}")
    return manifest_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2 AE seed stability and model freeze.")
    parser.add_argument("--data-path", type=str, default="data/creditcard.csv", help="Path to raw creditcard.csv")
    parser.add_argument("--manifest-path", type=str, default="data/split_manifest.json", help="Path to split manifest")
    parser.add_argument("--run-log", type=str, default="results/phase2/ae_run_log.csv", help="Run-log CSV")
    parser.add_argument("--checkpoints-dir", type=str, default="results/phase2/checkpoints", help="Checkpoints dir")
    parser.add_argument("--scores-dir", type=str, default="results/phase2/scores", help="Scores dir")
    parser.add_argument("--output-manifest", type=str, default="results/ae_frozen_model_manifest.json", help="Output frozen manifest")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch compute device ('cpu' or 'mps')")
    parser.add_argument("--max-epochs", type=int, default=100, help="Max epochs (plan: 100)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (plan: 10)")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    manifest_path = Path(args.manifest_path)
    run_log_path = Path(args.run_log)
    checkpoints_dir = Path(args.checkpoints_dir)
    scores_dir = Path(args.scores_dir)
    output_manifest_path = Path(args.output_manifest)
    device = torch.device(args.device)

    stage2_data = run_stage2(
        data_path=data_path,
        manifest_path=manifest_path,
        run_log_path=run_log_path,
        checkpoints_dir=checkpoints_dir,
        scores_dir=scores_dir,
        device=device,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )

    write_frozen_manifest(stage2_data, output_manifest_path)
    print("\nStage 2 runner execution complete.")


if __name__ == "__main__":
    main()
