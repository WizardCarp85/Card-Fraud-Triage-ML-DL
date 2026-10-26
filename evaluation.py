"""Reusable, leakage-safe evaluation utilities for fraud-triage anomaly scores.

All functions assume that larger scores indicate more suspicious transactions.
Thresholds are selected exclusively on validation data; test labels are used only
to report the performance of those already-selected thresholds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

if TYPE_CHECKING:
    from matplotlib.figure import Figure


FPR_BUDGETS: tuple[float, ...] = (0.001, 0.01, 0.05)
REVIEW_CAPACITIES: tuple[int, ...] = (100, 500, 1000)


@dataclass(frozen=True)
class ThresholdSelection:
    """A threshold selected on validation data for one false-positive-rate budget."""

    fpr_budget: float
    threshold: float
    validation_recall: float
    validation_fpr: float
    true_positives: int
    false_positives: int
    status: str = "ok"


@dataclass(frozen=True)
class ThresholdEvaluation:
    """Final test performance of a threshold selected on validation data."""

    fpr_budget: float
    threshold: float
    validation_recall: float
    validation_fpr: float
    test_recall: float
    test_fpr: float
    test_precision: float
    flagged_transactions: int
    frauds_caught: int
    false_alarms: int
    status: str = "ok"


@dataclass
class EvaluationResult:
    """Tables and threshold selections produced by :func:`evaluate_final_model`."""

    ranking_metrics: pd.DataFrame
    validation_thresholds: pd.DataFrame
    test_threshold_metrics: pd.DataFrame
    test_triage: pd.DataFrame
    selections: tuple[ThresholdSelection, ...]


def _validate_inputs(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated one-dimensional binary labels and finite scores.

    Raises:
        ValueError: If inputs are empty, lengths differ, labels are not binary,
            or scores contain NaN or infinite values.
    """

    labels = np.asarray(y_true)
    try:
        anomaly_scores = np.asarray(scores, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("scores must be numeric.") from error

    if labels.ndim != 1 or anomaly_scores.ndim != 1:
        raise ValueError("y_true and scores must be one-dimensional arrays.")
    if labels.size == 0 or anomaly_scores.size == 0:
        raise ValueError("y_true and scores must not be empty.")
    if labels.size != anomaly_scores.size:
        raise ValueError("y_true and scores must have the same length.")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("y_true must contain binary labels encoded as 0 and 1.")
    if not np.isfinite(anomaly_scores).all():
        raise ValueError("scores must contain only finite values.")

    return labels.astype(np.int8, copy=False), anomaly_scores


def _validate_fpr_budget(fpr_budget: float) -> float:
    """Validate and normalize a false-positive-rate budget."""

    budget = float(fpr_budget)
    if not 0.0 <= budget <= 1.0:
        raise ValueError("fpr_budget must be between 0 and 1, inclusive.")
    return budget


def _validate_k(k: int, n_samples: int) -> int:
    """Validate a top-K review capacity against the available records."""

    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer.")
    if not 1 <= int(k) <= n_samples:
        raise ValueError(f"k must be between 1 and {n_samples}, inclusive.")
    return int(k)


def average_precision(
    y_true: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray
) -> float:
    """Compute Average Precision (AP), the project's PR-AUC summary metric.

    AP is computed with :func:`sklearn.metrics.average_precision_score`, not a
    trapezoidal area approximation. It is undefined when no positive examples
    are present, so this function returns ``np.nan`` in that case.
    """

    labels, anomaly_scores = _validate_inputs(y_true, scores)
    if labels.sum() == 0:
        return float("nan")
    return float(average_precision_score(labels, anomaly_scores))


def roc_auc(
    y_true: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray
) -> float:
    """Compute ROC-AUC, or ``np.nan`` when either class is absent."""

    labels, anomaly_scores = _validate_inputs(y_true, scores)
    if labels.sum() == 0 or (labels == 0).sum() == 0:
        return float("nan")
    return float(roc_auc_score(labels, anomaly_scores))


def select_threshold_at_fpr(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    fpr_budget: float,
) -> ThresholdSelection:
    """Select a validation threshold that maximizes recall within an FPR budget.

    A transaction is flagged when ``score >= threshold``. The function considers
    every distinct score in descending order, so ties are handled as one group.
    If several feasible thresholds achieve the same maximum recall, the highest
    threshold is retained because it flags no additional transactions.

    The returned threshold must be selected using validation labels and scores
    only. No test inputs are accepted by this function.
    """

    labels, anomaly_scores = _validate_inputs(y_true, scores)
    budget = _validate_fpr_budget(fpr_budget)
    positives = int(labels.sum())
    negatives = int((labels == 0).sum())

    if positives == 0:
        return ThresholdSelection(
            budget, float("nan"), float("nan"), float("nan"), 0, 0, "undefined_no_positive"
        )
    if negatives == 0:
        return ThresholdSelection(
            budget, float("nan"), float("nan"), float("nan"), 0, 0, "undefined_no_negative"
        )

    order = np.argsort(-anomaly_scores, kind="mergesort")
    ordered_scores = anomaly_scores[order]
    ordered_labels = labels[order]

    best = ThresholdSelection(budget, float("inf"), 0.0, 0.0, 0, 0, "no_alerts_within_budget")
    tp = fp = 0
    start = 0
    while start < len(ordered_scores):
        end = start + 1
        while end < len(ordered_scores) and ordered_scores[end] == ordered_scores[start]:
            end += 1

        group = ordered_labels[start:end]
        tp += int(group.sum())
        fp += int((group == 0).sum())
        recall = tp / positives
        fpr = fp / negatives
        threshold = float(ordered_scores[start])

        if fpr <= budget and (
            recall > best.validation_recall
            or (np.isclose(recall, best.validation_recall) and threshold > best.threshold)
        ):
            best = ThresholdSelection(budget, threshold, recall, fpr, tp, fp)
        start = end

    return best


def recall_at_fpr(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    fpr_budget: float,
) -> float:
    """Return maximum recall achievable at or below a false-positive-rate budget."""

    return select_threshold_at_fpr(y_true, scores, fpr_budget).validation_recall


def precision_at_k(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    k: int,
) -> float:
    """Return precision among the K highest-scoring, most suspicious records."""

    labels, anomaly_scores = _validate_inputs(y_true, scores)
    review_capacity = _validate_k(k, len(labels))
    top_indices = np.argsort(-anomaly_scores, kind="mergesort")[:review_capacity]
    return float(labels[top_indices].mean())


def evaluate_fixed_threshold(
    y_true: Sequence[int] | np.ndarray, threshold: float, scores: Sequence[float] | np.ndarray
) -> dict[str, float | int]:
    """Evaluate one pre-selected threshold without tuning it.

    ``threshold`` may be positive infinity, representing a policy that flags no
    transactions. NaN thresholds are invalid because they have no policy meaning.
    """

    labels, anomaly_scores = _validate_inputs(y_true, scores)
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("threshold must be numeric.") from error
    if np.isnan(threshold_value):
        raise ValueError("threshold must not be NaN.")

    flagged = anomaly_scores >= threshold_value
    tp = int(np.sum((labels == 1) & flagged))
    fp = int(np.sum((labels == 0) & flagged))
    positives = int(labels.sum())
    negatives = int((labels == 0).sum())
    flagged_count = int(flagged.sum())

    return {
        "test_recall": float(tp / positives) if positives else float("nan"),
        "test_fpr": float(fp / negatives) if negatives else float("nan"),
        "test_precision": float(tp / flagged_count) if flagged_count else float("nan"),
        "flagged_transactions": flagged_count,
        "frauds_caught": tp,
        "false_alarms": fp,
    }


def apply_validation_thresholds_to_test(
    y_test: Sequence[int] | np.ndarray,
    score_test: Sequence[float] | np.ndarray,
    selections: Iterable[ThresholdSelection],
) -> pd.DataFrame:
    """Report test performance for thresholds already selected on validation data.

    This function never changes a threshold. Test labels are used only to compute
    the final recall, FPR, precision, and fraud/false-alarm counts.
    """

    _validate_inputs(y_test, score_test)
    rows: list[dict[str, float | int | str]] = []
    for selection in selections:
        row: dict[str, float | int | str] = {
            "fpr_budget": selection.fpr_budget,
            "threshold": selection.threshold,
            "validation_recall": selection.validation_recall,
            "validation_fpr": selection.validation_fpr,
            "status": selection.status,
        }
        if selection.status != "ok":
            row.update(
                {
                    "test_recall": float("nan"),
                    "test_fpr": float("nan"),
                    "test_precision": float("nan"),
                    "flagged_transactions": 0,
                    "frauds_caught": 0,
                    "false_alarms": 0,
                }
            )
        else:
            row.update(evaluate_fixed_threshold(y_test, selection.threshold, score_test))
        rows.append(row)
    return pd.DataFrame(rows)


def triage_table(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    capacities: Iterable[int] = REVIEW_CAPACITIES,
) -> pd.DataFrame:
    """Build a top-K review-queue table for the supplied evaluation split."""

    labels, anomaly_scores = _validate_inputs(y_true, scores)
    positives = int(labels.sum())
    rows: list[dict[str, float | int]] = []
    for capacity in capacities:
        k = _validate_k(capacity, len(labels))
        top_indices = np.argsort(-anomaly_scores, kind="mergesort")[:k]
        frauds_caught = int(labels[top_indices].sum())
        rows.append(
            {
                "review_capacity": k,
                "reviewed_transactions": k,
                "frauds_caught": frauds_caught,
                "recall": float(frauds_caught / positives) if positives else float("nan"),
                "precision": float(frauds_caught / k),
                "false_alarms": int(k - frauds_caught),
            }
        )
    return pd.DataFrame(rows)


def ranking_metrics_table(
    y_validation: Sequence[int] | np.ndarray,
    score_validation: Sequence[float] | np.ndarray,
    y_test: Sequence[int] | np.ndarray,
    score_test: Sequence[float] | np.ndarray,
) -> pd.DataFrame:
    """Return AP and ROC-AUC for validation and final test evaluation."""

    rows: list[dict[str, float | int | str]] = []
    for split, labels, scores in (
        ("validation", y_validation, score_validation),
        ("test", y_test, score_test),
    ):
        checked_labels, _ = _validate_inputs(labels, scores)
        rows.append(
            {
                "split": split,
                "n_transactions": int(len(checked_labels)),
                "n_fraud": int(checked_labels.sum()),
                "fraud_rate": float(checked_labels.mean()),
                "average_precision_ap": average_precision(labels, scores),
                "roc_auc": roc_auc(labels, scores),
            }
        )
    return pd.DataFrame(rows)


def plot_precision_recall_curve(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    output_path: str | Path | None = None,
    title: str = "Precision–Recall Curve — Final GMM Test Evaluation",
) -> Figure:
    """Plot a labelled PR curve and optional baseline for a final evaluation split."""

    import matplotlib.pyplot as plt

    labels, anomaly_scores = _validate_inputs(y_true, scores)
    figure, axis = plt.subplots(figsize=(8, 6))
    prevalence = float(labels.mean())

    if labels.sum() == 0:
        axis.text(0.5, 0.5, "PR curve undefined: no positive examples", ha="center", va="center")
    else:
        precision, recall, _ = precision_recall_curve(labels, anomaly_scores)
        ap = average_precision(labels, anomaly_scores)
        axis.plot(recall, precision, color="#1f77b4", linewidth=2, label=f"Average Precision = {ap:.3f}")
        axis.axhline(prevalence, color="#7f7f7f", linestyle="--", label=f"Fraud prevalence = {prevalence:.3%}")

    axis.set(title=title, xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.05))
    axis.grid(True, alpha=0.3)
    axis.legend(loc="lower left")
    figure.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180, bbox_inches="tight")
    return figure


def plot_validation_threshold_tradeoffs(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    selections: Iterable[ThresholdSelection],
    output_path: str | Path | None = None,
) -> Figure:
    """Plot validation precision, recall, and FPR across score thresholds."""

    import matplotlib.pyplot as plt

    labels, anomaly_scores = _validate_inputs(y_true, scores)
    positives = int(labels.sum())
    negatives = int((labels == 0).sum())
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    if positives == 0 or negatives == 0:
        for axis in axes:
            axis.text(0.5, 0.5, "Threshold trade-off undefined: both classes are required", ha="center", va="center")
            axis.set_axis_off()
    else:
        order = np.argsort(-anomaly_scores, kind="mergesort")
        ordered_scores, ordered_labels = anomaly_scores[order], labels[order]
        thresholds: list[float] = []
        precisions: list[float] = []
        recalls: list[float] = []
        fprs: list[float] = []
        tp = fp = 0
        start = 0
        while start < len(ordered_scores):
            end = start + 1
            while end < len(ordered_scores) and ordered_scores[end] == ordered_scores[start]:
                end += 1
            group = ordered_labels[start:end]
            tp += int(group.sum())
            fp += int((group == 0).sum())
            thresholds.append(float(ordered_scores[start]))
            precisions.append(tp / (tp + fp))
            recalls.append(tp / positives)
            fprs.append(fp / negatives)
            start = end

        axes[0].plot(thresholds, precisions, label="Validation precision", color="#2ca02c")
        axes[0].plot(thresholds, recalls, label="Validation recall", color="#1f77b4")
        axes[0].set(title="Validation Threshold Trade-off", xlabel="Anomaly-score threshold", ylabel="Metric")
        axes[1].plot(thresholds, fprs, label="Validation FPR", color="#d62728")
        axes[1].set(title="Validation False-Positive Rate", xlabel="Anomaly-score threshold", ylabel="FPR")
        for selection in selections:
            if selection.status == "ok":
                for axis in axes:
                    axis.axvline(selection.threshold, color="#444444", linestyle="--", alpha=0.5)
        for axis in axes:
            axis.grid(True, alpha=0.3)
            axis.legend()
    figure.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180, bbox_inches="tight")
    return figure


def evaluate_final_model(
    y_validation: Sequence[int] | np.ndarray,
    score_validation: Sequence[float] | np.ndarray,
    y_test: Sequence[int] | np.ndarray,
    score_test: Sequence[float] | np.ndarray,
    *,
    fpr_budgets: Iterable[float] = FPR_BUDGETS,
    review_capacities: Iterable[int] = REVIEW_CAPACITIES,
    table_dir: str | Path | None = None,
    figure_dir: str | Path | None = None,
) -> EvaluationResult:
    """Run leakage-safe final evaluation and optionally save CSVs and figures.

    Thresholds are selected before this function evaluates test thresholds: each
    call to :func:`select_threshold_at_fpr` receives only validation labels and
    validation scores. Test labels appear only in reporting functions.
    """

    _validate_inputs(y_validation, score_validation)
    _validate_inputs(y_test, score_test)
    selections = tuple(
        select_threshold_at_fpr(y_validation, score_validation, budget) for budget in fpr_budgets
    )
    result = EvaluationResult(
        ranking_metrics=ranking_metrics_table(y_validation, score_validation, y_test, score_test),
        validation_thresholds=pd.DataFrame([asdict(selection) for selection in selections]),
        test_threshold_metrics=apply_validation_thresholds_to_test(y_test, score_test, selections),
        test_triage=triage_table(y_test, score_test, review_capacities),
        selections=selections,
    )

    if table_dir is not None:
        tables = Path(table_dir)
        tables.mkdir(parents=True, exist_ok=True)
        result.ranking_metrics.to_csv(tables / "final_ranking_metrics.csv", index=False)
        result.validation_thresholds.to_csv(tables / "validation_fpr_thresholds.csv", index=False)
        result.test_threshold_metrics.to_csv(tables / "test_threshold_metrics.csv", index=False)
        result.test_triage.to_csv(tables / "test_triage_table.csv", index=False)
    if figure_dir is not None:
        figures = Path(figure_dir)
        plot_precision_recall_curve(y_test, score_test, figures / "final_gmm_pr_curve.png")
        plot_validation_threshold_tradeoffs(
            y_validation,
            score_validation,
            selections,
            figures / "validation_threshold_tradeoffs.png",
        )
    return result
