"""Rebuild every Phase 2 data artifact and compare it with the committed copies (issue #40).

Steps:
    1. Source file checksum equals the manifest.
    2. Split rebuild (#25): the regenerated manifest and split index CSVs equal the committed ones.
    3. Preprocessing rebuild (#28), run twice: the two runs are bit-identical, and the scaler
       equals the committed artifact within documented tolerance.
    4. Loaders (#31, #33), built twice: training batch order and seeded noise are identical,
       and evaluation streams are clean.
    5. Optional: fingerprints equal a reference report from another run or environment.

Nothing under data/ is overwritten. Rebuilt artifacts go to a temporary directory.

Usage:
    python -m data.verify_reproducibility
    python -m data.verify_reproducibility --reference data/reproducibility/reproducibility_report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch

from data import freeze_split as fs
from data import preprocessing as pp
from src.loaders import build_autoencoder_loaders

DATA_PATH = Path("data/creditcard.csv")
MANIFEST_PATH = Path("data/split_manifest.json")
SPLITS_DIR = Path("data/splits")
ARTIFACT_DIR = Path("data/artifacts")
DEFAULT_OUTPUT = Path("data/reproducibility/reproducibility_report.json")
SPLIT_NAMES = ("train", "val", "test", "train_legit")
LOADER_SEED = 42
LOADER_BATCH = 256
LOADER_NOISE_STD = 0.1
NOISE_BATCHES_HASHED = 5


def _sha(arr: np.ndarray | torch.Tensor) -> str:
    if isinstance(arr, torch.Tensor):
        arr = arr.numpy()
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def check_source() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    sums = fs.compute_file_checksums(DATA_PATH)
    expected = manifest["source_dataset"]["checksums"]
    return {"ok": sums == expected, "sha256": sums["sha256"], "expected_sha256": expected["sha256"]}


def check_splits() -> dict[str, Any]:
    committed = json.loads(MANIFEST_PATH.read_text())
    df_raw = fs.load_raw_dataset(DATA_PATH)
    df_dedup, _ = fs.deduplicate_dataset(df_raw)
    splits = fs.partition_splits(df_dedup)
    integrity = fs.verify_split_integrity(df_raw, df_dedup, splits)
    rebuilt = fs.generate_manifest_data(DATA_PATH, splits, df_raw, df_dedup)

    # content_sha256 hashes raw float bytes, which depend on the CSV float parser (last-bit
    # differences). Compare everything else strictly, and content hashes per parser below.
    def strip_content(m: dict[str, Any]) -> dict[str, Any]:
        m = json.loads(json.dumps(m))
        for stats in m.get("split_statistics", {}).values():
            stats.pop("content_sha256", None)
        return m

    c_strip, r_strip = strip_content(committed), strip_content(rebuilt)
    diff_keys = sorted(k for k in set(c_strip) | set(r_strip) if c_strip.get(k) != r_strip.get(k))
    content = check_content_hashes(committed, splits, df_raw)

    csv_ok = {}
    for name in SPLIT_NAMES:
        ids = pd.read_csv(SPLITS_DIR / f"{name}_indices.csv")["raw_row_id"].tolist()
        csv_ok[name] = ids == splits[name]
    return {
        "ok": (not diff_keys and all(csv_ok.values()) and integrity["status"] == "PASSED"
               and content["ok"]),
        "integrity": integrity["status"],
        "manifest_identical_excluding_content_hashes": not diff_keys,
        "manifest_differing_top_level_keys": diff_keys,
        "content_hashes": content,
        "split_csvs_identical": csv_ok,
        "row_id_sha256": {n: fs.compute_array_hash(splits[n]) for n in SPLIT_NAMES},
    }


CONTENT_KEYS = {"train": "train", "val": "validation", "test": "test", "train_legit": "train_legitimate_only"}


def check_content_hashes(committed: dict[str, Any], splits: dict[str, list[int]],
                         df_default: pd.DataFrame) -> dict[str, Any]:
    """Recompute split content hashes under the default and round_trip CSV float parsers.

    Passes if one parser reproduces every committed hash bit-for-bit, and the two parsers
    agree within the documented feature tolerance.
    """
    df_rt = pd.read_csv(DATA_PATH, float_precision="round_trip")
    per_parser: dict[str, dict[str, bool]] = {}
    for label, df in (("default", df_default), ("round_trip", df_rt)):
        per_parser[label] = {
            n: fs.compute_dataframe_content_hash(df.loc[splits[n]])
            == committed["split_statistics"][CONTENT_KEYS[n]]["content_sha256"]
            for n in SPLIT_NAMES
        }
    num = df_default.columns.drop("Class")
    a, b = df_default[num].to_numpy(), df_rt[num].to_numpy()
    max_abs = float(np.max(np.abs(a - b)))
    within = bool(np.allclose(a, b, rtol=pp.FEATURE_RTOL, atol=pp.FEATURE_ATOL))
    labels_equal = bool(np.array_equal(df_default["Class"].to_numpy(), df_rt["Class"].to_numpy()))
    reproducing = [p for p, r in per_parser.items() if all(r.values())]
    return {
        "ok": bool(reproducing) and within and labels_equal,
        "committed_hashes_reproduced_by_parser": reproducing,
        "per_parser_match": per_parser,
        "default_vs_round_trip_max_abs_diff": max_abs,
        "default_vs_round_trip_within_feature_tolerance": within,
        "labels_identical_across_parsers": labels_equal,
    }


def check_preprocessing(tmp: Path) -> dict[str, Any]:
    runs = []
    for i in range(2):
        built = pp.build_matrices(DATA_PATH, MANIFEST_PATH)
        report = pp.verify_matrices(built)
        pp.save_artifacts(built, report, tmp / f"run{i}", DATA_PATH)
        runs.append((built, report))
    (b0, r0), (b1, r1) = runs

    repeat_identical = all(
        np.array_equal(b0["splits"][n][k], b1["splits"][n][k])
        for n in ("train", "val", "test") for k in ("X", "y", "row_ids")
    ) and np.array_equal(b0["scaler"].center_, b1["scaler"].center_) and np.array_equal(b0["scaler"].scale_, b1["scaler"].scale_)

    meta = json.loads((ARTIFACT_DIR / "feature_metadata.json").read_text())
    committed_report = json.loads((ARTIFACT_DIR / "preprocessing_verification_report.json").read_text())
    committed_scaler = joblib.load(ARTIFACT_DIR / "scaler_robust_phase2.joblib")
    sc = b0["scaler"]
    tol = dict(rtol=pp.SCALER_PARAM_RTOL, atol=pp.SCALER_PARAM_ATOL)
    scaler_vs_metadata = bool(np.allclose(sc.center_, meta["scaler"]["center_"], **tol)
                              and np.allclose(sc.scale_, meta["scaler"]["scale_"], **tol))
    scaler_vs_joblib = bool(np.allclose(sc.center_, committed_scaler.center_, **tol)
                            and np.allclose(sc.scale_, committed_scaler.scale_, **tol))
    max_diff = float(max(np.max(np.abs(sc.center_ - committed_scaler.center_)),
                         np.max(np.abs(sc.scale_ - committed_scaler.scale_))))

    x_hash_match = {
        n: r0["per_split"][n]["X_sha256"] == committed_report["per_split"][n]["X_sha256"]
        for n in ("train", "val", "test")
    }
    ignore = {"environment", "scaler"}
    new_meta = json.loads((tmp / "run0" / "feature_metadata.json").read_text())
    meta_same = {k: new_meta[k] == meta[k] for k in meta if k not in ignore}

    return {
        "ok": (r0["status"] == "PASSED" and repeat_identical and scaler_vs_metadata
               and scaler_vs_joblib and all(meta_same.values())),
        "verification_status": r0["status"],
        "repeated_runs_bit_identical": bool(repeat_identical),
        "scaler_matches_committed_metadata_within_tolerance": scaler_vs_metadata,
        "scaler_matches_committed_joblib_within_tolerance": scaler_vs_joblib,
        "scaler_max_abs_diff_vs_committed": max_diff,
        "tolerance": {"rtol": pp.SCALER_PARAM_RTOL, "atol": pp.SCALER_PARAM_ATOL},
        "feature_metadata_fields_identical": meta_same,
        # Informational: exact bytes can legitimately differ across numpy/BLAS builds.
        "X_sha256_identical_to_committed_report": x_hash_match,
        "X_sha256": {n: r0["per_split"][n]["X_sha256"] for n in ("train", "val", "test")},
    }


def _loader_fingerprint(noise_std: float) -> dict[str, Any]:
    bundle = build_autoencoder_loaders(DATA_PATH, MANIFEST_PATH, batch_size=LOADER_BATCH,
                                       seed=LOADER_SEED, noise_std=noise_std)
    order, noisy, clean_eval = [], [], {}
    for i, (x_in, x_tgt, rid) in enumerate(bundle.loaders["train"]):
        order.append(rid)
        if i < NOISE_BATCHES_HASHED:
            noisy.append(x_in)
    for name in ("val_legit", "val_ranking", "test"):
        clean_eval[name] = _sha(torch.cat([x for x, _ in bundle.loaders[name]]))
    return {
        "train_batch_order_sha256": _sha(torch.cat(order)),
        "train_noisy_input_sha256_first_batches": _sha(torch.cat(noisy)),
        "eval_streams_sha256": clean_eval,
        "train_fraud_rows": int(bundle.labels["train"].sum()),
    }


def check_loaders() -> dict[str, Any]:
    a = _loader_fingerprint(LOADER_NOISE_STD)
    b = _loader_fingerprint(LOADER_NOISE_STD)
    clean = _loader_fingerprint(0.0)
    return {
        "ok": a == b and a["eval_streams_sha256"] == clean["eval_streams_sha256"]
              and a["train_batch_order_sha256"] == clean["train_batch_order_sha256"]
              and a["train_fraud_rows"] == 0,
        "repeated_builds_identical": a == b,
        "eval_streams_unaffected_by_noise": a["eval_streams_sha256"] == clean["eval_streams_sha256"],
        "batch_order_unaffected_by_noise": a["train_batch_order_sha256"] == clean["train_batch_order_sha256"],
        "settings": {"seed": LOADER_SEED, "batch_size": LOADER_BATCH, "noise_std": LOADER_NOISE_STD,
                     "noise_batches_hashed": NOISE_BATCHES_HASHED},
        "fingerprint": a,
    }


def compare_reference(report: dict[str, Any], ref_path: Path) -> dict[str, Any]:
    ref = json.loads(ref_path.read_text())
    fields = {
        "split_row_ids": (report["splits"]["row_id_sha256"], ref["splits"]["row_id_sha256"]),
        "loader_batch_order": (report["loaders"]["fingerprint"]["train_batch_order_sha256"],
                               ref["loaders"]["fingerprint"]["train_batch_order_sha256"]),
        "loader_noise": (report["loaders"]["fingerprint"]["train_noisy_input_sha256_first_batches"],
                         ref["loaders"]["fingerprint"]["train_noisy_input_sha256_first_batches"]),
        "feature_matrices_X": (report["preprocessing"]["X_sha256"], ref["preprocessing"]["X_sha256"]),
    }
    result = {k: a == b for k, (a, b) in fields.items()}
    # Split membership must match exactly. The other fingerprints are bitwise; if they differ
    # across environments, the in-tolerance scaler check in step 3 is the acceptance criterion.
    return {"ok": result["split_row_ids"], "identical": result, "reference": str(ref_path),
            "reference_environment": ref.get("environment")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--reference", type=Path, default=None)
    args = ap.parse_args(argv)

    report: dict[str, Any] = {
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__, "torch": torch.__version__,
        },
    }
    report["source"] = check_source()
    report["splits"] = check_splits()
    with tempfile.TemporaryDirectory() as tmp:
        report["preprocessing"] = check_preprocessing(Path(tmp))
    report["loaders"] = check_loaders()
    if args.reference is not None:
        report["reference_comparison"] = compare_reference(report, args.reference)

    sections = [k for k in ("source", "splits", "preprocessing", "loaders", "reference_comparison") if k in report]
    report["status"] = "PASSED" if all(report[k]["ok"] for k in sections) else "FAILED"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for k in sections:
        print(f"{k:22s} {'PASSED' if report[k]['ok'] else 'FAILED'}")
    print(f"overall                {report['status']}  -> {args.output}")
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    sys.exit(main())
