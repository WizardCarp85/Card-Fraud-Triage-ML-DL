"""Provenance and environment audit capturing runtime state and artifacts."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def get_git_info() -> dict[str, Any]:
    """Capture git commit hash, branch, and status if available."""
    info: dict[str, Any] = {
        "commit": "unknown",
        "branch": "unknown",
        "is_dirty": False,
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        info["commit"] = commit
    except (subprocess.SubprocessError, OSError):
        pass

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info["branch"] = branch
    except (subprocess.SubprocessError, OSError):
        pass

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        info["is_dirty"] = len(status) > 0
    except (subprocess.SubprocessError, OSError):
        pass

    return info


def get_package_versions() -> dict[str, str]:
    """Capture versions of core machine learning libraries."""
    versions: dict[str, str] = {}
    packages = ["numpy", "pandas", "sklearn", "torch", "joblib"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not_installed"
    return versions


def compute_file_sha256(filepath: str | Path) -> str | None:
    """Compute SHA-256 hash of a file if it exists."""
    path = Path(filepath)
    if not path.is_file():
        return None
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def generate_provenance(
    config_dict: dict[str, Any],
    data_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Generate comprehensive provenance record for an execution run."""
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git": get_git_info(),
        "system": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": get_package_versions(),
        "inputs": {
            "source_data_path": str(data_path),
            "source_data_sha256": compute_file_sha256(data_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": compute_file_sha256(manifest_path),
        },
        "config": config_dict,
    }
