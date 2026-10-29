"""Preprocessing utilities for raw GPR data."""

from __future__ import annotations

import numpy as np


def time_zero_correction(bscan: np.ndarray) -> np.ndarray:
    """Shift each A-scan so that the first significant arrival is at index 0.

    Parameters
    ----------
    bscan:
        2-D array of shape ``(n_traces, n_samples)``.

    Returns
    -------
    np.ndarray
        Time-zero corrected B-scan.
    """
    corrected = np.zeros_like(bscan)
    for i, trace in enumerate(bscan):
        onset = int(np.argmax(np.abs(trace) > 0.01 * np.max(np.abs(trace))))
        corrected[i] = np.roll(trace, -onset)
    return corrected


def dewow(bscan: np.ndarray, window: int = 50) -> np.ndarray:
    """Remove low-frequency wow from each A-scan using a running mean filter.

    Parameters
    ----------
    bscan:
        2-D array of shape ``(n_traces, n_samples)``.
    window:
        Half-window size in samples.

    Returns
    -------
    np.ndarray
        Dewowed B-scan.
    """
    from scipy.ndimage import uniform_filter1d  # lazy import

    trend = uniform_filter1d(bscan, size=2 * window + 1, axis=1, mode="nearest")
    return bscan - trend


def normalize(bscan: np.ndarray) -> np.ndarray:
    """Scale B-scan values to the range ``[-1, 1]``.

    Parameters
    ----------
    bscan:
        2-D array of shape ``(n_traces, n_samples)``.

    Returns
    -------
    np.ndarray
        Normalised B-scan.
    """
    max_val = np.max(np.abs(bscan))
    if max_val == 0:
        return bscan
    return bscan / max_val


def remove_background(bscan: np.ndarray, trace_axis: int = 0, method: str = "mean") -> np.ndarray:
    """Subtract the across-trace average from every time sample (standard GPR background removal).

    The direct air/ground wave and flat layer reflections are nearly identical on every trace, so
    removing the average trace isolates laterally varying features like target hyperbolas.

    Parameters
    ----------
    bscan:
        2-D B-scan. ``trace_axis`` says which axis indexes antenna position. This module's
        convention is ``(n_traces, n_samples)`` so the default is 0; gprMax output is
        ``(n_samples, n_traces)``, so call with ``trace_axis=1`` there.
    method:
        ``"mean"`` (classic) or ``"median"`` (less biased when a hyperbola spans many traces).
    """
    if bscan.ndim != 2:
        raise ValueError(f"Expected a 2-D B-scan, got shape {bscan.shape}.")
    if method == "mean":
        ref = bscan.mean(axis=trace_axis, keepdims=True)
    elif method == "median":
        ref = np.median(bscan, axis=trace_axis, keepdims=True)
    else:
        raise ValueError(f"Unknown background removal method: {method!r}")
    return bscan - ref
