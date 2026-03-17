"""Visualisation utilities for experiment results."""

from __future__ import annotations

import numpy as np


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Confusion Matrix",
    ax=None,
):
    """Plot a confusion matrix as a heatmap.

    Parameters
    ----------
    cm:
        Square confusion matrix array of shape ``(n_classes, n_classes)``.
    class_names:
        Optional list of class label strings.
    title:
        Plot title.
    ax:
        Optional existing ``matplotlib.axes.Axes`` to draw on.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt  # lazy import

    if ax is None:
        _, ax = plt.subplots()

    n = cm.shape[0]
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    ticks = np.arange(n)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if class_names:
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    return ax


def plot_metrics_history(
    history: dict[str, list[float]],
    title: str = "Training History",
    ax=None,
):
    """Plot scalar metric histories (e.g. loss, accuracy) over epochs.

    Parameters
    ----------
    history:
        Dictionary mapping metric name to a list of per-epoch values.
    title:
        Plot title.
    ax:
        Optional existing ``matplotlib.axes.Axes`` to draw on.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt  # lazy import

    if ax is None:
        _, ax = plt.subplots()

    for name, values in history.items():
        ax.plot(values, label=name)

    ax.set_xlabel("Epoch")
    ax.set_title(title)
    ax.legend()
    return ax
