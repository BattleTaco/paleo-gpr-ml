"""Feature engineering for GPR and fossil data."""

from __future__ import annotations

import numpy as np


def instantaneous_amplitude(bscan: np.ndarray) -> np.ndarray:
    """Compute the instantaneous amplitude (envelope) of a GPR B-scan.

    Uses the analytic signal via the Hilbert transform.

    Parameters
    ----------
    bscan:
        2-D array of shape ``(n_traces, n_samples)``.

    Returns
    -------
    np.ndarray
        Instantaneous amplitude with the same shape as *bscan*.
    """
    from scipy.signal import hilbert  # lazy import

    analytic = hilbert(bscan, axis=1)
    return np.abs(analytic)


def instantaneous_phase(bscan: np.ndarray) -> np.ndarray:
    """Compute the instantaneous phase of a GPR B-scan.

    Parameters
    ----------
    bscan:
        2-D array of shape ``(n_traces, n_samples)``.

    Returns
    -------
    np.ndarray
        Instantaneous phase in radians, same shape as *bscan*.
    """
    from scipy.signal import hilbert  # lazy import

    analytic = hilbert(bscan, axis=1)
    return np.unwrap(np.angle(analytic), axis=1)


def hyperbola_velocity_scan(
    bscan: np.ndarray,
    dt: float,
    dx: float,
    velocities: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform a velocity semblance scan to estimate subsurface propagation velocity.

    Parameters
    ----------
    bscan:
        2-D array of shape ``(n_traces, n_samples)``.
    dt:
        Time step in seconds.
    dx:
        Spatial step in metres.
    velocities:
        Array of velocities (m/s) to test.  Defaults to a log-spaced range
        from 0.05 c to 0.3 c (typical soil range).

    Returns
    -------
    semblance : np.ndarray
        Semblance panel of shape ``(len(velocities), n_samples)``.
    velocities : np.ndarray
        Velocity array used.
    """
    c = 3e8  # speed of light in m/s
    if velocities is None:
        velocities = np.linspace(0.05 * c, 0.30 * c, 50)

    n_traces, n_samples = bscan.shape
    # Build offsets centred on the middle trace so that xs[i] corresponds to
    # bscan[i] and every index stays within [0, n_traces).
    xs = (np.arange(n_traces) - n_traces // 2) * dx
    semblance = np.zeros((len(velocities), n_samples))

    for vi, v in enumerate(velocities):
        stack = np.zeros(n_samples)
        for t0_idx in range(n_samples):
            t0 = t0_idx * dt
            total = 0.0
            for trace_idx, x in enumerate(xs):
                t_hyp = np.sqrt(t0**2 + (x / v) ** 2)
                sample_idx = int(round(t_hyp / dt))
                if sample_idx < n_samples:
                    total += bscan[trace_idx, sample_idx]
            stack[t0_idx] = total
        semblance[vi] = stack

    return semblance, velocities
