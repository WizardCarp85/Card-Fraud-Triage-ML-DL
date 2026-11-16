"""Stage 1 Autoencoder Architecture and Denoising Ablation Runner.

Follows the pre-registered plan in docs/ae_experiment_plan.md (Issue #27, #35).
Runs 3 clean-input ablations (widths 4, 8, 16, seed 42), selects optimal width B*,
and runs the denoising ablation (sigma = 0.5, seed 42) at B*.

All training uses legitimate training rows only. Model selection uses validation only.
No test data or labels are evaluated.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from evaluation import (
    average_precision,
    precision_at_k,
    recall_at_fpr,
)
from src.loaders import build_autoencoder_loaders
from src.models.autoencoder import PyTorchAutoencoderNet


def get_git_commit() -> str:
    """Retrieve the current HEAD git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def compute_file_sha256(path: str | Path) -> str:
    """Compute cryptographic SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def train_and_evaluate_run(
    run_id: str,
    bottleneck: int,
    denoise: bool,
    noise_std: float,
    seed: int,
    data_path: str | Path,
    manifest_path: str | Path,
    manifest_hash: str,
    git_commit: str,
    checkpoints_dir: Path,
    scores_dir: Path,
    device: torch.device,
    max_epochs: int = 100,
    patience: int = 10,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Execute training and validation evaluation for a single candidate configuration."""
    print(f"\n{'=' * 75}")
    print(f"RUN START: {run_id}")
    print(f"Configuration: Bottleneck={bottleneck}, Denoise={denoise} (std={noise_std}), Seed={seed}")
    print(f"{'=' * 75}")

    start_dt = datetime.now(timezone.utc)
    start_time_iso = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.time()

    # Enforce reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Build loaders via merged #31/#33 module
    bundle = build_autoencoder_loaders(
        data_path=data_path,
        manifest_path=manifest_path,
        batch_size=batch_size,
        seed=seed,
        noise_std=noise_std if denoise else 0.0,
        noise_seed=seed,
    )

    train_loader = bundle.loaders["train"]
    val_legit_loader = bundle.loaders["val_legit"]
    val_ranking_loader = bundle.loaders["val_ranking"]

    # 2. Reusable model architecture from #32 (hidden_dims=(32,))
    net = PyTorchAutoencoderNet(
        input_dim=29,
        latent_dim=bottleneck,
        hidden_dims=(32,),
    ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_val_recon_loss = float("inf")
    best_epoch = 0
    best_state_dict: dict[str, Any] | None = None
    patience_counter = 0
    final_train_loss = float("nan")
    early_stopped = False

    # 3. Training loop with legitimate-validation early stopping
    print(f"{'Epoch':>5} | {'Train MSE Loss':>15} | {'Val Legit MSE':>15} | {'Patience':>8}")
    print("-" * 52)

    for epoch in range(1, max_epochs + 1):
        net.train()
        epoch_loss_sum = 0.0
        n_batches = 0

        for batch_x_in, batch_x_tgt, _ in train_loader:
            batch_x_in = batch_x_in.to(device)
            batch_x_tgt = batch_x_tgt.to(device)

            optimizer.zero_grad()
            reconstructed = net(batch_x_in)
            loss = criterion(reconstructed, batch_x_tgt)
            loss.backward()
            optimizer.step()

            epoch_loss_sum += float(loss.item())
            n_batches += 1

        avg_train_loss = epoch_loss_sum / max(1, n_batches)
        final_train_loss = avg_train_loss

        # Evaluate on legitimate validation rows only (clean, deterministic)
        net.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for val_x, _ in val_legit_loader:
                val_x = val_x.to(device)
                val_recon = net(val_x)
                v_loss = criterion(val_recon, val_x)
                val_loss_sum += float(v_loss.item())
                val_batches += 1

        avg_val_recon_loss = val_loss_sum / max(1, val_batches)

        # Early stopping tracking on legitimate validation reconstruction loss
        if avg_val_recon_loss < best_val_recon_loss - 1e-6:
            best_val_recon_loss = avg_val_recon_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(net.state_dict())
            patience_counter = 0
            improved_flag = "*"
        else:
            patience_counter += 1
            improved_flag = ""

        if epoch % 5 == 0 or epoch == 1 or improved_flag or patience_counter >= patience:
            print(
                f"{epoch:5d} | {avg_train_loss:15.5f} | {avg_val_recon_loss:15.5f} {improved_flag:1s} | {patience_counter:8d}"
            )

        if patience_counter >= patience:
            early_stopped = True
            print(f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}).")
            break

    # Restore best checkpoint
    if best_state_dict is not None:
        net.load_state_dict(best_state_dict)

    # 4. Save checkpoint
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / f"{run_id}.pt"
    torch.save(net.state_dict(), checkpoint_path)
    checkpoint_sha256 = compute_file_sha256(checkpoint_path)

    # 5. Score full validation split via #34 inference path (per-sample MSE)
    net.eval()
    val_scores_list: list[np.ndarray] = []
    val_row_ids_list: list[np.ndarray] = []

    with torch.no_grad():
        for x_val, row_id_val in val_ranking_loader:
            x_val = x_val.to(device)
            recon_val = net(x_val)
            # MSE per sample across 29 features
            sample_mse = torch.mean((x_val - recon_val) ** 2, dim=1)
            val_scores_list.append(sample_mse.cpu().numpy())
            val_row_ids_list.append(row_id_val.numpy())

    scores_val = np.concatenate(val_scores_list).astype(np.float64)
    row_ids_val = np.concatenate(val_row_ids_list).astype(np.int64)

    # Sanity check alignment
    expected_row_ids = bundle.row_ids["val_ranking"]
    assert np.array_equal(row_ids_val, expected_row_ids), "Validation row IDs do not match manifest!"

    # Save validation score file compliant with schema
    scores_dir.mkdir(parents=True, exist_ok=True)
    score_file_path = scores_dir / f"{run_id}_scores_val.csv"
    df_scores = pd.DataFrame(
        {
            "row_id": row_ids_val,
            "split": ["validation"] * len(row_ids_val),
            "model": ["autoencoder"] * len(row_ids_val),
            "anomaly_score": scores_val,
        }
    )
    df_scores.to_csv(score_file_path, index=False)

    # 6. Compute VALIDATION ONLY metrics using existing evaluation.py
    y_val = bundle.labels["val_ranking"]
    val_ap = average_precision(y_val, scores_val)
    val_p100 = precision_at_k(y_val, scores_val, k=100)
    val_rec_fpr1 = recall_at_fpr(y_val, scores_val, fpr_budget=0.01)
    val_rec_fpr5 = recall_at_fpr(y_val, scores_val, fpr_budget=0.05)

    end_dt = datetime.now(timezone.utc)
    end_time_iso = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    runtime_s = round(time.time() - t0, 2)

    config_dict = {
        "architecture": f"29-32-{bottleneck}-32-29",
        "bottleneck": bottleneck,
        "hidden_dims": [32],
        "denoise": denoise,
        "noise_std": float(noise_std) if denoise else 0.0,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "best_epoch": best_epoch,
    }

    result = {
        "run_id": run_id,
        "config_json": json.dumps(config_dict, separators=(",", ":")),
        "seed": seed,
        "git_commit": git_commit,
        "dataset_manifest_hash": manifest_hash,
        "start_time": start_time_iso,
        "end_time": end_time_iso,
        "runtime_s": runtime_s,
        "epochs_run": best_epoch if early_stopped else max_epochs,
        "early_stopped": early_stopped,
        "final_train_loss": round(float(final_train_loss), 6),
        "val_recon_loss": round(float(best_val_recon_loss), 6),
        "val_AP": round(float(val_ap), 6),
        "val_Precision@100": round(float(val_p100), 6),
        "val_recall_FPR1pct": round(float(val_rec_fpr1), 6),
        "val_recall_FPR5pct": round(float(val_rec_fpr5), 6),
        "checkpoint_path": str(checkpoint_path.relative_to(Path.cwd())),
        "checkpoint_sha256": checkpoint_sha256,
    }

    print(f"\nRUN SUMMARY: {run_id}")
    print(f"  Runtime (s):            {runtime_s}")
    print(f"  Epochs Run:             {result['epochs_run']} (Early Stopped: {early_stopped})")
    print(f"  Best Val Recon Loss:    {result['val_recon_loss']:.6f}")
    print(f"  Validation AP (PR-AUC): {result['val_AP']:.4f}")
    print(f"  Validation P@100:       {result['val_Precision@100']:.4f}")
    print(f"  Validation Recall@1%FPR:{result['val_recall_FPR1pct']:.4f}")
    print(f"  Validation Recall@5%FPR:{result['val_recall_FPR5pct']:.4f}")
    print(f"  Checkpoint SHA-256:     {checkpoint_sha256[:16]}...")
    print(f"{'=' * 75}\n")

    return result


def apply_selection_rule(
    candidate_a: dict[str, Any],
    candidate_b: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Apply the frozen Section-5 selection rule between two candidates.

    Rule:
    1. Primary: Validation AP (higher is better).
    2. Secondary: Precision@100 if |AP_A - AP_B| < 0.005.
    3. Tie-break: smaller bottleneck width, or clean over denoising.
    """
    cfg_a = json.loads(candidate_a["config_json"])
    cfg_b = json.loads(candidate_b["config_json"])

    ap_a = candidate_a["val_AP"]
    ap_b = candidate_b["val_AP"]
    p100_a = candidate_a["val_Precision@100"]
    p100_b = candidate_b["val_Precision@100"]

    diff_ap = ap_a - ap_b

    if abs(diff_ap) >= 0.005:
        if diff_ap > 0:
            reason = (
                f"{candidate_a['run_id']} wins on primary metric val_AP "
                f"({ap_a:.4f} vs {ap_b:.4f}, diff={diff_ap:+.4f} >= 0.005)"
            )
            return candidate_a, reason
        else:
            reason = (
                f"{candidate_b['run_id']} wins on primary metric val_AP "
                f"({ap_b:.4f} vs {ap_a:.4f}, diff={-diff_ap:+.4f} >= 0.005)"
            )
            return candidate_b, reason

    # Within 0.005 AP indifference band -> use Precision@100
    diff_p100 = p100_a - p100_b
    if abs(diff_p100) > 1e-6:
        if diff_p100 > 0:
            reason = (
                f"{candidate_a['run_id']} wins in 0.005 AP band (|diff|={abs(diff_ap):.4f} < 0.005) "
                f"on secondary metric Precision@100 ({p100_a:.4f} vs {p100_b:.4f})"
            )
            return candidate_a, reason
        else:
            reason = (
                f"{candidate_b['run_id']} wins in 0.005 AP band (|diff|={abs(diff_ap):.4f} < 0.005) "
                f"on secondary metric Precision@100 ({p100_b:.4f} vs {p100_a:.4f})"
            )
            return candidate_b, reason

    # Deterministic tie-break
    # Parsimony: smaller bottleneck
    w_a = cfg_a["bottleneck"]
    w_b = cfg_b["bottleneck"]
    if w_a != w_b:
        winner = candidate_a if w_a < w_b else candidate_b
        loser = candidate_b if w_a < w_b else candidate_a
        reason = (
            f"{winner['run_id']} wins tie-break by parsimony (smaller bottleneck width "
            f"{min(w_a, w_b)} < {max(w_a, w_b)})"
        )
        return winner, reason

    # Simplicity: clean over denoising
    if cfg_a["denoise"] != cfg_b["denoise"]:
        winner = candidate_a if not cfg_a["denoise"] else candidate_b
        reason = f"{winner['run_id']} wins tie-break by simplicity (clean over denoised)"
        return winner, reason

    # Exact tie
    return candidate_a, f"{candidate_a['run_id']} and {candidate_b['run_id']} are identical"


def plot_stage1_results(results: list[dict[str, Any]], figure_path: Path) -> None:
    """Generate the Stage 1 validation AP comparison figure."""
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    clean_results = [r for r in results if not json.loads(r["config_json"])["denoise"]]
    denoise_results = [r for r in results if json.loads(r["config_json"])["denoise"]]

    clean_results.sort(key=lambda r: json.loads(r["config_json"])["bottleneck"])

    widths = [json.loads(r["config_json"])["bottleneck"] for r in clean_results]
    clean_aps = [r["val_AP"] for r in clean_results]
    clean_p100s = [r["val_Precision@100"] for r in clean_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Validation AP by bottleneck width (Clean vs Denoised)
    ax1.plot(widths, clean_aps, "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="Clean Inputs (Seed 42)")

    if denoise_results:
        dnz = denoise_results[0]
        dnz_cfg = json.loads(dnz["config_json"])
        dnz_w = dnz_cfg["bottleneck"]
        dnz_ap = dnz["val_AP"]
        ax1.scatter(
            [dnz_w],
            [dnz_ap],
            color="#d62728",
            marker="s",
            s=120,
            zorder=5,
            label=f"Denoised (B={dnz_w}, $\\sigma=0.5$)",
        )
        ax1.annotate(
            f"Denoised B={dnz_w}\nAP={dnz_ap:.4f}",
            (dnz_w, dnz_ap),
            textcoords="offset points",
            xytext=(-20, 15),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
        )

    for w, ap in zip(widths, clean_aps):
        ax1.annotate(
            f"B={w}\nAP={ap:.4f}",
            (w, ap),
            textcoords="offset points",
            xytext=(0, -22),
            ha="center",
            fontsize=9,
        )

    ax1.set_xlabel("Bottleneck Latent Width ($B$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Validation Average Precision (PR-AUC)", fontsize=11, fontweight="bold")
    ax1.set_title("Stage 1: Validation PR-AUC across Latent Widths", fontsize=12, fontweight="bold")
    ax1.set_xticks(widths)
    ax1.set_xticklabels([f"B={w}" for w in widths])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="best", frameon=True)

    # Panel 2: Secondary metric Precision@100
    ax2.plot(widths, clean_p100s, "s--", color="#2ca02c", linewidth=2.0, markersize=7, label="Clean Inputs (P@100)")
    if denoise_results:
        dnz_p100 = denoise_results[0]["val_Precision@100"]
        ax2.scatter(
            [dnz_w],
            [dnz_p100],
            color="#d62728",
            marker="^",
            s=120,
            zorder=5,
            label=f"Denoised (P@100={dnz_p100:.4f})",
        )

    ax2.set_xlabel("Bottleneck Latent Width ($B$)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Validation Precision@100", fontsize=11, fontweight="bold")
    ax2.set_title("Stage 1: Secondary Metric (Precision@100)", fontsize=12, fontweight="bold")
    ax2.set_xticks(widths)
    ax2.set_xticklabels([f"B={w}" for w in widths])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="best", frameon=True)

    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()
    print(f"Saved Stage 1 ablation figure to: {figure_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 1 AE architecture and denoising ablations.")
    parser.add_argument("--data-path", type=str, default="data/creditcard.csv", help="Path to raw creditcard.csv")
    parser.add_argument("--manifest-path", type=str, default="data/split_manifest.json", help="Path to split manifest")
    parser.add_argument("--output-csv", type=str, default="results/phase2/ae_run_log.csv", help="Output run-log CSV")
    parser.add_argument("--checkpoints-dir", type=str, default="results/phase2/checkpoints", help="Checkpoints dir")
    parser.add_argument("--scores-dir", type=str, default="results/phase2/scores", help="Validation scores dir")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch compute device ('cpu' or 'mps')")
    parser.add_argument("--max-epochs", type=int, default=100, help="Max epochs per run (plan: 100)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (plan: 10)")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    manifest_path = Path(args.manifest_path)
    output_csv = Path(args.output_csv)
    checkpoints_dir = Path(args.checkpoints_dir)
    scores_dir = Path(args.scores_dir)
    device = torch.device(args.device)

    if not data_path.is_file():
        sys.exit(f"Data file not found at {data_path}")
    if not manifest_path.is_file():
        sys.exit(f"Manifest not found at {manifest_path}")

    manifest_hash = compute_file_sha256(manifest_path)
    git_commit = get_git_commit()

    print("=" * 75)
    print("PHASE 2 AUTOENCODER STAGE 1 ABLATION SUITE (Issue #35)")
    print(f"Data Path:      {data_path.resolve()}")
    print(f"Manifest:       {manifest_path.resolve()} (SHA-256: {manifest_hash})")
    print(f"Git Commit:     {git_commit}")
    print(f"Compute Device: {device}")
    print("=" * 75)

    all_run_results: list[dict[str, Any]] = []

    # -------------------------------------------------------------
    # Step 1A: 3 Clean-input ablations, seed 42 (Widths 4, 8, 16)
    # -------------------------------------------------------------
    clean_runs = [
        ("RUN-S1-W04-CLN-S42", 4),
        ("RUN-S1-W08-CLN-S42", 8),
        ("RUN-S1-W16-CLN-S42", 16),
    ]

    clean_results: list[dict[str, Any]] = []
    for run_id, width in clean_runs:
        res = train_and_evaluate_run(
            run_id=run_id,
            bottleneck=width,
            denoise=False,
            noise_std=0.0,
            seed=42,
            data_path=data_path,
            manifest_path=manifest_path,
            manifest_hash=manifest_hash,
            git_commit=git_commit,
            checkpoints_dir=checkpoints_dir,
            scores_dir=scores_dir,
            device=device,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        clean_results.append(res)
        all_run_results.append(res)

    # -------------------------------------------------------------
    # Step 1B: Apply Section-5 Selection Rule to pick B*
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("APPLYING SECTION-5 SELECTION RULE TO CHOOSE B*")
    print("=" * 75)

    # Sort candidates by val_AP descending
    sorted_clean = sorted(clean_results, key=lambda r: r["val_AP"], reverse=True)
    best_candidate = sorted_clean[0]

    for candidate in sorted_clean[1:]:
        best_candidate, reason = apply_selection_rule(best_candidate, candidate)
        print(f"Comparison: {reason}")

    b_star = json.loads(best_candidate["config_json"])["bottleneck"]
    print(f"\n---> OPTIMAL BOTTLENECK WIDTH B* = {b_star} ({best_candidate['run_id']})")
    print("=" * 75)

    # -------------------------------------------------------------
    # Step 1C: Denoising Ablation at B* (sigma = 0.5, seed 42)
    # -------------------------------------------------------------
    denoise_run_id = f"RUN-S1-W{b_star:02d}-DNZ-S42"
    denoise_res = train_and_evaluate_run(
        run_id=denoise_run_id,
        bottleneck=b_star,
        denoise=True,
        noise_std=0.5,
        seed=42,
        data_path=data_path,
        manifest_path=manifest_path,
        manifest_hash=manifest_hash,
        git_commit=git_commit,
        checkpoints_dir=checkpoints_dir,
        scores_dir=scores_dir,
        device=device,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    all_run_results.append(denoise_res)

    # -------------------------------------------------------------
    # Step 1D: Compare Clean vs Denoised at B*
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("APPLYING SECTION-5 SELECTION RULE: CLEAN VS DENOISED AT B*")
    print("=" * 75)
    stage1_winner, dnz_reason = apply_selection_rule(best_candidate, denoise_res)
    print(f"Comparison: {dnz_reason}")
    print(f"\n---> STAGE 1 RULE-BASED WINNER: {stage1_winner['run_id']}")
    print("=" * 75)

    # -------------------------------------------------------------
    # Step 1E: Append all 4 runs to results/phase2/ae_run_log.csv
    # -------------------------------------------------------------
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "run_id",
        "config_json",
        "seed",
        "git_commit",
        "dataset_manifest_hash",
        "start_time",
        "end_time",
        "runtime_s",
        "epochs_run",
        "early_stopped",
        "final_train_loss",
        "val_recon_loss",
        "val_AP",
        "val_Precision@100",
        "val_recall_FPR1pct",
        "val_recall_FPR5pct",
        "checkpoint_path",
        "checkpoint_sha256",
    ]

    df_new = pd.DataFrame(all_run_results)[columns]

    if output_csv.is_file():
        df_existing = pd.read_csv(output_csv)
        # Avoid duplicate run_ids
        df_combined = pd.concat([df_existing[~df_existing["run_id"].isin(df_new["run_id"])], df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(output_csv, index=False)
    print(f"\nSuccessfully appended {len(all_run_results)} runs to {output_csv}")

    # -------------------------------------------------------------
    # Step 1F: Generate Stage 1 Ablation Plot
    # -------------------------------------------------------------
    fig_path = Path("results/figures/ae_stage1_val_ap.png")
    plot_stage1_results(all_run_results, fig_path)

    print("\nStage 1 ablations execution finished successfully.")


if __name__ == "__main__":
    main()
