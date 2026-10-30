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


# Reference categorical slots 1-3 (dataviz palette, light mode), one per training condition.
# Marker shape is the secondary encoding so identity never rests on color alone.
TRANSFER_STYLE = {
    "void": {"color": "#2a78d6", "marker": "o", "label": "trained on syn void"},
    "bone": {"color": "#eb6834", "marker": "s", "label": "trained on syn bone"},
    "real_void": {"color": "#1baf7a", "marker": "D", "label": "trained on real infra + syn void"},
}
EVAL_ORDER = ("void", "anti_bone", "bone")


def plot_ap_by_condition(df, ax=None, title: str = "", legend: bool = True):
    """AP50 with 95% CI per (training condition, eval domain), as dot-and-whisker.

    ``df`` needs columns train, eval, AP50, AP50_lo, AP50_hi (summary.csv / per_stratum.csv).
    Eval domains are ordered void -> anti_bone -> bone, so a polarity effect reads as a step
    between the last two columns at matched |contrast|.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.6))
    conds = [c for c in TRANSFER_STYLE if c in set(df["train"])]
    width = 0.22
    for i, c in enumerate(conds):
        st = TRANSFER_STYLE[c]
        sub = df[df["train"] == c].set_index("eval").reindex(list(EVAL_ORDER))
        x = np.arange(len(EVAL_ORDER)) + (i - (len(conds) - 1) / 2) * width
        ax.errorbar(
            x,
            sub["AP50"],
            yerr=[sub["AP50"] - sub["AP50_lo"], sub["AP50_hi"] - sub["AP50"]],
            fmt=st["marker"],
            color=st["color"],
            ms=7,
            lw=2,
            capsize=0,
            label=st["label"],
        )
    ax.set_xticks(np.arange(len(EVAL_ORDER)), ["void", "anti_bone\n(void polarity)", "bone"])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("AP@0.5 (95% CI)")
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(axis="y", color="#d9d8d3", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if legend:
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    return ax


def plot_transfer_figure(summary, per_stratum, path=None):
    """Pooled panel plus one small multiple per host x freq stratum. Saves to ``path`` if given."""
    import matplotlib.pyplot as plt

    strata = per_stratum.groupby(["host", "freq_mhz"])
    fig, axes = plt.subplots(
        1, 1 + strata.ngroups, figsize=(4.2 * (1 + strata.ngroups), 3.8), sharey=True
    )
    plot_ap_by_condition(summary, axes[0], "All 72 cells (pooled out-of-fold)", legend=False)
    for ax, ((host, freq), sub) in zip(axes[1:], strata):
        plot_ap_by_condition(sub, ax, f"{host.replace('_', ' ')}, {int(freq)} MHz", legend=False)
        ax.set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    if path is not None:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig
