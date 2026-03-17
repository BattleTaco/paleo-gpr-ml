"""Visualisation utilities for GPR data."""

from __future__ import annotations

import numpy as np


def plot_bscan(
    bscan: np.ndarray,
    dt: float = 1.0,
    dx: float = 1.0,
    cmap: str = "seismic",
    title: str = "GPR B-scan",
    ax=None,
):
    """Plot a GPR B-scan as a wiggle/colour image.

    Parameters
    ----------
    bscan:
        2-D array of shape ``(n_traces, n_samples)``.
    dt:
        Time step in nanoseconds (used for y-axis labelling).
    dx:
        Spatial step in metres (used for x-axis labelling).
    cmap:
        Matplotlib colour map name.
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

    n_traces, n_samples = bscan.shape
    extent = [0, n_traces * dx, n_samples * dt, 0]
    vmax = np.percentile(np.abs(bscan), 98)
    ax.imshow(
        bscan.T,
        aspect="auto",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        extent=extent,
    )
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Two-way travel time (ns)")
    ax.set_title(title)
    return ax
