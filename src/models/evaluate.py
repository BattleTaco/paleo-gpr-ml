"""Model evaluation utilities."""

from __future__ import annotations

import numpy as np


def precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute precision, recall, and F1-score for binary predictions.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 or 1).
    y_pred:
        Predicted probabilities or binary labels.
    threshold:
        Decision threshold when *y_pred* contains probabilities.

    Returns
    -------
    dict
        Dictionary with keys ``precision``, ``recall``, and ``f1``.
    """
    y_pred_bin = (y_pred >= threshold).astype(int)
    tp = int(np.sum((y_pred_bin == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred_bin == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred_bin == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def iou(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Compute Intersection over Union for binary masks.

    Parameters
    ----------
    y_true:
        Ground-truth binary mask.
    y_pred:
        Predicted probabilities or binary mask.
    threshold:
        Decision threshold when *y_pred* contains probabilities.

    Returns
    -------
    float
        IoU score in ``[0, 1]``.
    """
    y_pred_bin = (y_pred >= threshold).astype(int)
    intersection = int(np.sum((y_pred_bin == 1) & (y_true == 1)))
    union = int(np.sum((y_pred_bin == 1) | (y_true == 1)))
    return intersection / union if union > 0 else 0.0
