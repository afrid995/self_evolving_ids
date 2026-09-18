"""
evaluate.py -- Metrics and evaluation utilities for Self-Evolving IDS.

Usage (standalone):
    python src/evaluate.py

This module provides reusable evaluation functions used across all phases:
  - Classification reports (per-class precision, recall, F1)
  - Confusion matrix display
  - Macro / weighted F1
  - Continual learning metrics (forgetting, backward transfer) — Phase 5
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)


# ============================================================
# 1. Classification report
# ============================================================
def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    """
    Print a detailed classification report and return the dict version.

    Args:
        y_true:      Ground truth labels (integer-encoded)
        y_pred:      Predicted labels (integer-encoded)
        class_names: Human-readable class names

    Returns:
        The sklearn classification_report as a dict
    """
    report_str = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )
    print(report_str)

    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return report_dict


# ============================================================
# 2. Confusion matrix
# ============================================================
def print_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> np.ndarray:
    """
    Print a formatted confusion matrix and return the raw array.

    Rows = actual class, Columns = predicted class.
    """
    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    # Header
    max_name = max(len(n) for n in class_names)
    header = " " * (max_name + 2) + "  ".join(
        f"{n:>{max_name}}" for n in class_names
    )
    print("\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  {header}")

    # Rows
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>{max_name}}" for v in row)
        print(f"  {class_names[i]:>{max_name}}  {row_str}")

    print()
    return cm


# ============================================================
# 3. Summary metrics
# ============================================================
def compute_summary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute key summary metrics.

    Returns:
        Dict with accuracy, macro_f1, weighted_f1
    """
    metrics = {
        "accuracy":    accuracy_score(y_true, y_pred),
        "macro_f1":    f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    return metrics


# ============================================================
# 4. Continual learning metrics (Phase 5 — stubs for now)
# ============================================================
def forgetting_measure(
    old_accuracy: float,
    new_accuracy_on_old: float,
) -> float:
    """
    Measure how much the model forgot about old tasks.

    Forgetting = old_accuracy - new_accuracy_on_old
    Positive value means forgetting occurred.
    """
    return old_accuracy - new_accuracy_on_old


def backward_transfer(
    accuracies_before: list[float],
    accuracies_after: list[float],
) -> float:
    """
    Average backward transfer across tasks.

    BWT = mean(acc_after[i] - acc_before[i]) for all old tasks i.
    Negative BWT means forgetting.
    """
    if len(accuracies_before) != len(accuracies_after):
        raise ValueError("Lists must have the same length")
    if not accuracies_before:
        return 0.0
    diffs = [a - b for a, b in zip(accuracies_after, accuracies_before)]
    return sum(diffs) / len(diffs)


def average_accuracy(accuracies: list[float]) -> float:
    """Average accuracy across all tasks seen so far."""
    if not accuracies:
        return 0.0
    return sum(accuracies) / len(accuracies)


# ============================================================
# Main (quick demo)
# ============================================================
if __name__ == "__main__":
    print("evaluate.py -- Quick demo with dummy data")
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    y_pred = np.array([0, 0, 1, 2, 2, 2, 3, 1])
    names = ["DoS", "Normal", "Probe", "U2R"]

    print_classification_report(y_true, y_pred, names)
    print_confusion_matrix(y_true, y_pred, names)
    metrics = compute_summary_metrics(y_true, y_pred)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
